from __future__ import annotations

import html
import io
import json
import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable
from urllib.parse import quote
from uuid import uuid4

import pandas as pd
import streamlit as st


# Streamlit Community Cloud launches the entrypoint from a generated source
# directory. Add the repository root explicitly so the package import works
# consistently in the cloud, locally, and in Streamlit's test runner.
PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from prototype_phase12.backend import (
    REVIEW_STATUSES,
    DataUploadStore,
    FraudScoringBackend,
    ModelUpdateStore,
    OperationalQueueStore,
    ReviewStore,
    TrainingRunService,
    build_audit_package,
    dataframe_to_csv_bytes,
    dataframe_to_parquet_bytes,
    prepare_scored_export,
    utc_timestamp,
)


ROOT = PACKAGE_ROOT
ARTIFACT_DIR = ROOT / "artifacts"
RUNTIME_DIR = ROOT / "runtime"
STYLE_PATH = ROOT / "styles.css"
DEMO_FLAGGED_TOTAL = 31_541
NAV_ITEMS = ("Overview", "Review queue", "Data uploads", "Monitoring", "Model updates")
OPTIONAL_IDENTITY_FIELDS = (
    "customer_name",
    "policyholder_name",
    "customer_id",
    "policyholder_id",
    "policy_id",
    "claim_reference",
)
TRAINING_TARGET = "respuesta_dicot_c"
LEAKAGE_SOURCE_FIELDS = ("situacionvts", "scoreada", "respuesta_dicot1")
INVESTIGATION_CONTEXT_FIELDS = (
    "tipocliente",
    "tomadornivel",
    "sexo",
    "tipodocumento",
    "tomadorexperian",
    "edad_conductor1",
    "tomadorcodigopostal",
    "tomadormunicipioid",
    "tomadorprovinciaid",
    "tomadornacionalidadid",
    "canalnombre",
    "garantia",
    "garantia_agrupada",
    "antiguedad_poliza",
    "dias_notificacion",
    "zona",
    "comarcaid",
    "formapago",
    "tipo",
    "categoria",
    "anyomatricula",
    "valorvehiculomercado",
    "valorvehiculofabrica",
    "potencia",
)
CONTEXT_LABEL_OVERRIDES = {
    "tipocliente": "Client document category",
    "tomadorcodigopostal": "Policyholder postal code",
    "tomadormunicipioid": "Policyholder municipality code",
    "tomadorprovinciaid": "Policyholder province code",
    "tomadornacionalidadid": "Policyholder nationality code",
}


st.set_page_config(
    page_title="Fraud Review",
    page_icon=":material/policy:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <!--
    THESIS: Fraud review is a precise operating tool, not a generic blue dashboard.
    OWN-WORLD: X-inspired black navigation, white working canvas, editorial rules, pill actions, and a single sky-blue state color.
    STORY: Add claims, review the ranked queue, investigate evidence, and govern manual model updates.
    FIRST VIEWPORT: A compact black rail frames one clear task on a bright, high-contrast canvas.
    FORM: Monochrome operations desk with progressive disclosure; seed key c5a48679.
    FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md
    -->
    """,
    unsafe_allow_html=True,
)
st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_backend() -> FraudScoringBackend:
    return FraudScoringBackend(ARTIFACT_DIR)


@st.cache_resource
def load_review_store() -> ReviewStore:
    return ReviewStore(RUNTIME_DIR / "reviews.sqlite3")


@st.cache_resource
def load_model_update_store() -> ModelUpdateStore:
    return ModelUpdateStore(RUNTIME_DIR / "reviews.sqlite3")


@st.cache_resource
def load_data_upload_store() -> DataUploadStore:
    return DataUploadStore(RUNTIME_DIR / "reviews.sqlite3")


@st.cache_resource
def load_operational_queue_store() -> OperationalQueueStore:
    return OperationalQueueStore(RUNTIME_DIR / "operational_queue")


@st.cache_resource
def load_training_service() -> TrainingRunService:
    return TrainingRunService(
        project_root=ROOT.parent,
        artifact_dir=ARTIFACT_DIR,
        runtime_dir=RUNTIME_DIR,
        historical_train_path=ROOT.parent / "data" / "train_model_dataset.parquet",
        historical_test_path=ROOT.parent / "data" / "test_model_dataset.parquet",
        database_path=RUNTIME_DIR / "reviews.sqlite3",
    )


@st.cache_data(show_spinner=False)
def load_demo_batch() -> tuple[pd.DataFrame, pd.DataFrame]:
    scoring_backend = load_backend()
    queue_path = ARTIFACT_DIR / scoring_backend.config["demo_flagged_queue_path"]
    queue = pd.read_parquet(queue_path)
    if len(queue) != DEMO_FLAGGED_TOTAL:
        raise ValueError(
            f"The demonstration queue contains {len(queue):,} rows; "
            f"{DEMO_FLAGGED_TOTAL:,} flagged claims were expected."
        )
    probabilities = queue.pop("fraud_probability").astype(float)
    source = queue.reset_index(drop=True)
    scores = pd.DataFrame(
        {
            "claim_id": source["claim_id"].astype(str),
            "row_index": source["row_index"].astype(int),
            "fraud_probability": probabilities,
            "threshold": scoring_backend.threshold,
            "flagged": True,
            "classification_label": "Flagged for review",
        }
    )
    return scores, source


@st.cache_data(show_spinner=False, max_entries=32)
def load_demo_claim(claim_id: str) -> pd.DataFrame:
    scoring_backend = load_backend()
    source_path = ARTIFACT_DIR / scoring_backend.config["demo_claim_source_path"]
    frame = pd.read_parquet(
        source_path,
        filters=[("claim_id", "==", str(claim_id))],
    )
    if frame.empty:
        raise KeyError(f"Claim not found in the demonstration source: {claim_id}")
    return frame.head(1).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_global_shap() -> list[dict[str, Any]]:
    return load_backend().global_shap_importance(top_n=10)


backend = load_backend()
review_store = load_review_store()
model_update_store = load_model_update_store()
data_upload_store = load_data_upload_store()
operational_queue_store = load_operational_queue_store()
training_service = load_training_service()


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def probability_text(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}%"


def compact_number(value: float) -> str:
    absolute = abs(float(value))
    if absolute >= 1:
        return f"{value:.2f}"
    if absolute >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4f}"


def display_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "fraud_review"


def read_upload(uploaded_file: Any) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    payload = uploaded_file.getvalue()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(payload))
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(payload))
    raise ValueError("Upload a CSV or Parquet file.")


def normalize_outcome_labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    text = (
        values.astype("string")
        .str.strip()
        .str.upper()
        .map({"NO FRAUDE": 0, "FRAUDE": 1, "0": 0, "1": 1})
    )
    return numeric.fillna(text)


def validate_training_upload(frame: pd.DataFrame) -> dict[str, Any]:
    required = set(backend.approved_features) | {TRAINING_TARGET}
    optional = {*LEAKAGE_SOURCE_FIELDS, "claim_id", "row_index"}
    missing = sorted(required - set(frame.columns))
    extra = sorted(set(frame.columns) - required - optional)
    if TRAINING_TARGET in frame:
        labels = normalize_outcome_labels(frame[TRAINING_TARGET])
    else:
        labels = pd.Series(index=frame.index, dtype="float64")
    invalid_labels = int((labels.isna() | ~labels.isin([0, 1])).sum())
    fraud_rows = int((labels == 1).sum())
    return {
        "valid": bool(len(frame)) and not missing and not extra and not invalid_labels,
        "row_count": int(len(frame)),
        "source_columns": int(len(frame.columns)),
        "missing": missing,
        "extra": extra,
        "invalid_labels": invalid_labels,
        "fraud_rows": fraud_rows,
        "leakage_fields_present": [
            field for field in LEAKAGE_SOURCE_FIELDS if field in frame.columns
        ],
        "schema": "Legacy source schema"
        if all(field in frame.columns for field in LEAKAGE_SOURCE_FIELDS)
        else "Model-ready labelled schema",
    }


def current_model_profile() -> dict[str, Any]:
    active_config = json.loads(
        (ARTIFACT_DIR / "model_config.json").read_text(encoding="utf-8")
    )
    profile = dict(active_config.get("model_profile", {}))
    profile.setdefault("model_version", active_config.get("model_track", "Active model"))
    profile.setdefault("threshold", active_config.get("q_star_threshold", backend.threshold))
    profile.setdefault("training_snapshot", "Not recorded")
    profile.setdefault("training_rows", 0)
    profile.setdefault("training_fraud", 0)
    profile.setdefault("evaluation_rows", 0)
    profile.setdefault("evaluation_fraud", 0)
    profile.setdefault("fraud_detected", 0)
    profile.setdefault("fraud_missed", 0)
    profile.setdefault("legitimate_flagged", 0)
    profile.setdefault("roc_auc", None)
    profile.setdefault("pr_auc", None)
    return profile


def navigate(page: str, claim_id: str | None = None) -> None:
    if page != "Case" and "case" in st.query_params:
        del st.query_params["case"]
    st.session_state["active_page"] = page
    st.session_state["animate_next_page"] = True
    if page in NAV_ITEMS:
        st.session_state["nav_choice"] = page
        st.session_state["mobile_nav_choice"] = page
    if claim_id is not None:
        st.session_state["selected_claim_id"] = claim_id


def sync_navigation() -> None:
    if "case" in st.query_params:
        del st.query_params["case"]
    st.session_state["active_page"] = st.session_state["nav_choice"]
    st.session_state["mobile_nav_choice"] = st.session_state["nav_choice"]
    st.session_state["animate_next_page"] = True


def sync_mobile_navigation() -> None:
    if "case" in st.query_params:
        del st.query_params["case"]
    st.session_state["active_page"] = st.session_state["mobile_nav_choice"]
    st.session_state["nav_choice"] = st.session_state["mobile_nav_choice"]
    st.session_state["animate_next_page"] = True


def clear_case_selection() -> None:
    st.session_state["selected_claim_id"] = None


def clear_upload_result() -> None:
    st.session_state.pop("upload_result", None)


def activate_upload(operation: str) -> tuple[bool, str]:
    result = st.session_state.get("upload_result")
    if not result:
        return False, "Score a file before updating the queue."

    upload_scores = result["scores"].copy()
    upload_source = result["source"].copy()
    upload_scores["claim_id"] = upload_scores["claim_id"].astype(str)
    upload_source["claim_id"] = upload_source["claim_id"].astype(str)
    previous_name, current_scores, current_source, previous_timestamp, previous_is_demo = (
        active_batch()
    )
    previous_upload_count = int(st.session_state.get("active_batch_upload_count", 1))
    starts_operational_queue = bool(previous_is_demo)
    previous_state = {
        "name": previous_name,
        "scores": current_scores.copy(),
        "source": current_source.copy(),
        "timestamp": previous_timestamp,
        "is_demo": previous_is_demo,
        "upload_count": previous_upload_count,
    }

    if operation == "append" and not starts_operational_queue:
        current_scores["claim_id"] = current_scores["claim_id"].astype(str)
        current_source["claim_id"] = current_source["claim_id"].astype(str)
        duplicate_ids = sorted(
            set(upload_source["claim_id"]) & set(current_source["claim_id"])
        )
        if duplicate_ids:
            preview = ", ".join(duplicate_ids[:5])
            return (
                False,
                f"Append is blocked because {len(duplicate_ids):,} claim IDs already exist: {preview}.",
            )
        st.session_state["active_batch_scores"] = pd.concat(
            [current_scores, upload_scores], ignore_index=True
        )
        st.session_state["active_batch_source"] = pd.concat(
            [current_source, upload_source], ignore_index=True, sort=False
        )
        batch_count = int(st.session_state.get("active_batch_upload_count", 1)) + 1
        st.session_state["active_batch_upload_count"] = batch_count
        st.session_state["active_batch_name"] = f"Active queue ({batch_count} batches)"
        message = (
            f"Added {len(upload_scores):,} claims from {result['batch_name']} to {previous_name}."
        )
    else:
        st.session_state["active_batch_name"] = result["batch_name"]
        st.session_state["active_batch_scores"] = upload_scores
        st.session_state["active_batch_source"] = upload_source
        st.session_state["active_batch_upload_count"] = 1
        if starts_operational_queue:
            message = (
                f"Started the current review queue with {result['batch_name']} "
                f"({len(upload_scores):,} claims). The demonstration queue is no longer active."
            )
        else:
            message = (
                f"Replaced the active review queue with {result['batch_name']} "
                f"({len(upload_scores):,} claims)."
            )

    st.session_state["active_batch_timestamp"] = result["scored_at_utc"]
    st.session_state["active_batch_validation"] = result["validation"]
    active_profile = current_model_profile()
    upload_id = uuid4().hex
    current_state = {
        "name": st.session_state["active_batch_name"],
        "scores": st.session_state["active_batch_scores"].copy(),
        "source": st.session_state["active_batch_source"].copy(),
        "timestamp": st.session_state["active_batch_timestamp"],
        "is_demo": False,
        "upload_count": int(st.session_state["active_batch_upload_count"]),
    }
    try:
        operational_queue_store.commit(
            upload_id,
            previous=previous_state,
            current=current_state,
        )
        data_upload_store.record_activation(
            upload_id=upload_id,
            batch_name=result["batch_name"],
            source_file=result["source_file"],
            operation=(
                "Start"
                if starts_operational_queue
                else "Add"
                if operation == "append"
                else "Replace"
            ),
            row_count=len(upload_scores),
            flagged_count=int(upload_scores["flagged"].sum()),
            labelled_rows=int(result.get("labelled_rows", 0)),
            fraud_labels=int(result.get("fraud_labels", 0)),
            label_status=str(result.get("label_status", "Labels needed")),
            scored_at_utc=result["scored_at_utc"],
            model_version=str(active_profile["model_version"]),
        )
    except Exception as exc:
        try:
            operational_queue_store.undo(upload_id)
        except Exception:
            pass
        _restore_queue_session(previous_state)
        return False, f"The queue update could not be saved. {exc}"
    st.session_state["selected_claim_id"] = None
    st.session_state["activation_notice"] = message
    st.session_state["pending_navigation"] = "Review queue"
    return True, message


def _restore_queue_session(state: dict[str, Any]) -> None:
    active_keys = (
        "active_batch_name",
        "active_batch_scores",
        "active_batch_source",
        "active_batch_timestamp",
        "active_batch_validation",
        "active_batch_upload_count",
    )
    if state.get("is_demo"):
        for key in active_keys:
            st.session_state.pop(key, None)
        return
    st.session_state["active_batch_name"] = state["name"]
    st.session_state["active_batch_scores"] = state["scores"].copy()
    st.session_state["active_batch_source"] = state["source"].copy()
    st.session_state["active_batch_timestamp"] = state["timestamp"]
    st.session_state["active_batch_upload_count"] = int(state.get("upload_count", 1))


def undo_latest_upload(upload_id: str) -> tuple[bool, str]:
    try:
        data_upload_store.undo_latest(upload_id)
        previous = operational_queue_store.undo(upload_id)
    except Exception as exc:
        try:
            data_upload_store.reactivate(upload_id)
        except Exception:
            pass
        return False, f"The queue update could not be undone. {exc}"
    _restore_queue_session(previous)
    st.session_state.pop("upload_result", None)
    st.session_state["selected_claim_id"] = None
    return True, "The latest upload was removed and the previous queue was restored."


for key, default in {
    "active_page": "Overview",
    "nav_choice": "Overview",
    "mobile_nav_choice": "Overview",
    "selected_claim_id": None,
    "review_capacity": 100,
    "upload_mode": "Append new cases",
    "animate_next_page": True,
}.items():
    st.session_state.setdefault(key, default)

if pending_page := st.session_state.pop("pending_navigation", None):
    st.session_state["active_page"] = pending_page
    st.session_state["nav_choice"] = pending_page
    st.session_state["mobile_nav_choice"] = pending_page
    st.session_state["animate_next_page"] = True

requested_case = st.query_params.get("case")
if requested_case:
    if st.session_state.get("active_page") != "Case":
        st.session_state["animate_next_page"] = True
    st.session_state["selected_claim_id"] = str(requested_case)
    st.session_state["active_page"] = "Case"
    st.session_state["nav_choice"] = "Review queue"
    st.session_state["mobile_nav_choice"] = "Review queue"


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="brand-name">Fraud Review</div>', unsafe_allow_html=True)
        st.radio(
            "Navigation",
            NAV_ITEMS,
            key="nav_choice",
            label_visibility="collapsed",
            on_change=sync_navigation,
        )


def render_mobile_navigation() -> None:
    with st.container(key="mobile_nav"):
        st.selectbox(
            "Section",
            NAV_ITEMS,
            key="mobile_nav_choice",
            label_visibility="collapsed",
            on_change=sync_mobile_navigation,
        )


def consume_page_transition_class() -> str:
    return " page-transition-active" if st.session_state.pop("animate_next_page", False) else ""


def render_page_heading(title: str, copy: str = "") -> None:
    copy_html = f"<p>{escape(copy)}</p>" if copy else ""
    transition_class = consume_page_transition_class()
    st.markdown(
        f'<div class="page-heading{transition_class}"><h1>{escape(title)}</h1>{copy_html}</div>',
        unsafe_allow_html=True,
    )


def render_stat_strip(items: Iterable[tuple[str, str]]) -> None:
    rows = list(items)
    cells = "".join(
        f'<div class="stat-item"><div class="stat-value">{escape(value)}</div>'
        f'<div class="stat-label">{escape(label)}</div></div>'
        for label, value in rows
    )
    st.markdown(
        f'<div class="stat-strip" style="--stat-count:{len(rows)}">{cells}</div>',
        unsafe_allow_html=True,
    )


def active_batch() -> tuple[str, pd.DataFrame, pd.DataFrame, str, bool]:
    if "active_batch_scores" in st.session_state:
        return (
            st.session_state["active_batch_name"],
            st.session_state["active_batch_scores"].copy(),
            st.session_state["active_batch_source"].copy(),
            st.session_state["active_batch_timestamp"],
            False,
        )
    stored_queue = operational_queue_store.load_active()
    if stored_queue is not None:
        return (
            str(stored_queue["name"]),
            stored_queue["scores"].copy(),
            stored_queue["source"].copy(),
            str(stored_queue["timestamp"]),
            False,
        )
    demo_scores, demo_source = load_demo_batch()
    return (
        "Claims_Aug_2026",
        demo_scores.copy(),
        demo_source.copy(),
        "Report-matched baseline",
        True,
    )


def with_review_state(scores: pd.DataFrame) -> pd.DataFrame:
    merged = scores.copy()
    reviews = review_store.list_all()
    if not reviews.empty:
        merged = merged.merge(reviews, on="claim_id", how="left")
    for column, default in {
        "status": "Pending",
        "notes": "",
        "reviewed_at_utc": "",
    }.items():
        if column not in merged:
            merged[column] = default
        merged[column] = merged[column].fillna(default)
    return merged.sort_values("fraud_probability", ascending=False).reset_index(drop=True)


def claim_frame(claim_id: str, source: pd.DataFrame) -> pd.DataFrame:
    matches = source[source["claim_id"].astype(str) == str(claim_id)]
    if matches.empty:
        raise KeyError(f"Claim not found in the active batch: {claim_id}")
    return matches.head(1).copy()


def claim_summary(claim_id: str, source: pd.DataFrame, is_demo: bool) -> dict[str, Any]:
    if is_demo:
        for item in backend.list_claims():
            if item["claim_id"] == claim_id:
                return item.get("summary", {})
    row = claim_frame(claim_id, source).iloc[0]
    priority = [
        "garantia",
        "garantia_agrupada",
        "antiguedad_poliza",
        "dias_notificacion",
        "comarcaid",
        "tomadornivel",
        "edad_conductor1",
        "formapago",
        "tipo",
    ]
    summary: dict[str, Any] = {}
    for feature in priority:
        if feature in row.index:
            summary[backend.label_map.get(feature, feature)] = row[feature]
    return summary


def present_value(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def first_row_value(row: pd.Series, fields: Iterable[str]) -> Any:
    for field in fields:
        if field in row.index and present_value(row[field]):
            return row[field]
    return None


def policyholder_context(row: pd.Series) -> str:
    parts: list[str] = []
    tier = first_row_value(row, ("tomadornivel", "customer_tier"))
    age = first_row_value(row, ("edad_conductor1", "driver_age"))
    area = first_row_value(row, ("zona", "tomadorcomarcaid", "comarcaid"))
    if present_value(tier):
        parts.append(str(tier))
    if present_value(age):
        parts.append(f"Age {float(age):.0f}")
    if present_value(area):
        parts.append(str(area))
    return " · ".join(parts) or "Profile not supplied"


def context_label(field: str) -> str:
    return CONTEXT_LABEL_OVERRIDES.get(
        field, backend.label_map.get(field, field.replace("_", " ").title())
    )


def context_display_value(label: str, value: Any) -> str:
    if not present_value(value):
        return "Not available"
    label_lower = label.lower()
    if label == "Policyholder sex":
        return {"f": "Female", "m": "Male"}.get(str(value).lower(), str(value))
    if "driver age" in label_lower:
        return f"{float(value):.1f} years"
    if "market value" in label_lower or "factory value" in label_lower:
        return f"€{float(value):,.0f}"
    if "code" in label_lower or "postal" in label_lower or "registration year" in label_lower:
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)
    return display_value(value)


def claim_context_sections(claim_id: str, source: pd.DataFrame) -> tuple[list[tuple[str, dict[str, Any]]], bool]:
    row = claim_frame(claim_id, source).iloc[0]
    identity: dict[str, Any] = {"Claim ID": claim_id}
    identity_labels = {
        "customer_name": "Customer name",
        "policyholder_name": "Policyholder name",
        "customer_id": "Customer ID",
        "policyholder_id": "Policyholder ID",
        "policy_id": "Policy ID",
        "claim_reference": "Claim reference",
    }
    has_identity = False
    for field in OPTIONAL_IDENTITY_FIELDS:
        if field in row.index and present_value(row[field]):
            identity[identity_labels[field]] = row[field]
            has_identity = True

    groups = [
        (
            "Policyholder profile",
            {
                context_label(field): row[field]
                for field in (
                    "tipocliente",
                    "tomadornivel",
                    "sexo",
                    "tipodocumento",
                    "tomadorexperian",
                    "edad_conductor1",
                )
                if field in row.index
            },
        ),
        (
            "Policy & location",
            {
                context_label(field): row[field]
                for field in (
                    "garantia",
                    "garantia_agrupada",
                    "antiguedad_poliza",
                    "tomadorcodigopostal",
                    "tomadormunicipioid",
                    "tomadorprovinciaid",
                    "tomadornacionalidadid",
                    "canalnombre",
                )
                if field in row.index
            },
        ),
        (
            "Vehicle",
            {
                context_label(field): row[field]
                for field in (
                    "tipo",
                    "categoria",
                    "anyomatricula",
                    "valorvehiculomercado",
                    "valorvehiculofabrica",
                    "potencia",
                )
                if field in row.index
            },
        ),
        (
            "Claim context",
            {
                context_label(field): row[field]
                for field in (
                    "dias_notificacion",
                    "zona",
                    "comarcaid",
                    "formapago",
                    "subramo",
                    "situacionvts",
                )
                if field in row.index
            },
        ),
    ]
    return [("Case reference", identity)] + groups, has_identity


def render_context_sections(sections: list[tuple[str, dict[str, Any]]]) -> None:
    section_html: list[str] = []
    for title, values in sections:
        cells = "".join(
            f'<div class="context-fact"><span>{escape(label)}</span><strong>{escape(context_display_value(label, value))}</strong></div>'
            for label, value in values.items()
            if present_value(value)
        )
        if cells:
            section_html.append(
                f'<section class="context-group"><h3>{escape(title)}</h3><div class="context-facts">{cells}</div></section>'
            )
    st.markdown(
        '<div class="context-intro"><h2>Customer &amp; case context</h2></div>'
        f'<div class="context-sections">{"".join(section_html)}</div>',
        unsafe_allow_html=True,
    )


def format_updated(value: str) -> str:
    if not value:
        return "Not reviewed"
    try:
        parsed = pd.Timestamp(value)
        return parsed.strftime("%d %b %Y %H:%M UTC")
    except Exception:
        return value


def render_case_hero(
    result: dict[str, Any],
    review: dict[str, str],
    *,
    capacity: int,
) -> None:
    st.markdown(
        f"""
        <section class="case-hero">
          <div class="case-hero-top">
            <div>
              <div class="case-id">{escape(result['claim_id'])}</div>
              <div class="case-decision">{escape(result['classification_label'])}</div>
            </div>
            <div>
              <div class="risk-score">{probability_text(result['fraud_probability'])}</div>
              <div class="risk-caption"><span>Fraud probability</span><span>Threshold {probability_text(result['threshold'])}</span></div>
            </div>
          </div>
          <div class="case-meta">
            <div><div class="meta-label">Review status</div><div class="meta-value">{escape(review['status'])}</div></div>
            <div><div class="meta-label">Investigation capacity</div><div class="meta-value">Top {capacity:,}</div></div>
            <div><div class="meta-label">Updated</div><div class="meta-value">{escape(format_updated(review['reviewed_at_utc']))}</div></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_shap_chart(
    factors: list[dict[str, Any]],
    *,
    title: str,
    absolute: bool = False,
) -> None:
    if not factors:
        st.markdown(
            '<div class="empty-state"><h3>No explanation available</h3>'
            '<p>This claim does not have a model explanation in the active batch.</p></div>',
            unsafe_allow_html=True,
        )
        return
    contributions = [
        float(item.get("mean_abs_shap", item.get("contribution", 0.0)))
        for item in factors
    ]
    has_positive = any(value > 0 for value in contributions)
    has_negative = any(value < 0 for value in contributions)
    mixed = has_positive and has_negative and not absolute
    maximum = max(abs(value) for value in contributions) or 1.0
    rows: list[str] = []
    for item, contribution in zip(factors, contributions):
        width = max(2.0, abs(contribution) / maximum * (43.0 if mixed else 78.0))
        label = item.get("label", item.get("feature", "Feature"))
        raw_value = "" if absolute else display_value(item.get("value", ""))
        direction = "negative" if contribution < 0 else "positive"
        mixed_class = " mixed" if mixed and contribution >= 0 else ""
        if mixed and contribution < 0:
            bar_style = f"width:{width:.2f}%;right:50%;"
            number_style = f"right:{50 + width:.2f}%;transform:translateX(-6px);"
        elif mixed:
            bar_style = f"width:{width:.2f}%;left:50%;"
            number_style = f"left:{50 + width + 1:.2f}%;"
        else:
            bar_style = f"width:{width:.2f}%;left:0;"
            number_style = f"left:{min(width + 1.5, 86):.2f}%;"
        zero = (
            '<span style="position:absolute;left:50%;top:0;bottom:0;border-left:1px solid #93a6c0"></span>'
            if mixed
            else ""
        )
        number = f"{contribution:+.2f}" if not absolute else compact_number(contribution)
        tooltip_text = (
            f"{label}: mean absolute SHAP {contribution:.6f}"
            if absolute
            else f"{label}: {raw_value}; SHAP {contribution:+.6f}"
        )
        feature_meta = f"<span>{escape(raw_value)}</span>" if raw_value else ""
        rows.append(
            dedent(
                f"""
            <div class="shap-row" title="{escape(tooltip_text)}">
              <div class="shap-feature"><strong>{escape(label)}</strong>{feature_meta}</div>
              <div class="shap-track" style="{'border-left:0' if mixed else ''}">
                {zero}<span class="shap-bar {direction}{mixed_class}" style="{bar_style}"></span>
                <span class="shap-number" style="{number_style}">{escape(number)}</span>
              </div>
            </div>
            """
            )
        )
    if mixed:
        ticks = [f"-{compact_number(maximum)}", "0", compact_number(maximum)]
    else:
        ticks = [
            "0",
            compact_number(maximum / 3),
            compact_number(maximum * 2 / 3),
            compact_number(maximum),
        ]
    axis_label = "Mean absolute SHAP value" if absolute else "SHAP value"
    note = (
        "Influence describes model behaviour, not evidence of fraud."
        if absolute
        else "Positive values raise the model score; negative values lower it. Model behaviour is not evidence of fraud."
    )
    chart_html = (
        f'<div class="shap-title">{escape(title)}</div>'
        f'<div class="shap-chart">{"".join(rows)}</div>'
        '<div class="shap-axis"><div></div><div>'
        f'<div class="axis-line">{"".join(f"<span>{escape(tick)}</span>" for tick in ticks)}</div>'
        f'<div class="axis-label">{escape(axis_label)}</div></div></div>'
        f'<div class="model-note">{escape(note)}</div>'
    )
    st.markdown(chart_html, unsafe_allow_html=True)


def render_detail_grid(summary: dict[str, Any], class_name: str = "detail-grid") -> None:
    cells = "".join(
        f'<div><div class="detail-label">{escape(label)}</div>'
        f'<div class="detail-value">{escape(display_value(value))}</div></div>'
        for label, value in summary.items()
    )
    st.markdown(f'<div class="{escape(class_name)}">{cells}</div>', unsafe_allow_html=True)


def operational_queue_export(
    scores: pd.DataFrame,
    source: pd.DataFrame,
    *,
    capacity: int,
    batch_name: str,
) -> pd.DataFrame:
    ranked = scores.sort_values("fraud_probability", ascending=False).copy()
    ranked["review_rank"] = range(1, len(ranked) + 1)
    queue = ranked[ranked["flagged"]].head(capacity).copy()
    queue["fraud_probability"] = queue["fraud_probability"].astype(float)
    queue["batch_name"] = batch_name
    queue["model_track"] = backend.config["model_track"]
    queue["q_star_threshold"] = backend.threshold
    queue["exported_at_utc"] = utc_timestamp()
    core_columns = [
        "review_rank",
        "claim_id",
        "fraud_probability",
        "classification_label",
        "status",
        "notes",
        "reviewed_at_utc",
        "batch_name",
        "model_track",
        "q_star_threshold",
        "exported_at_utc",
    ]
    source_fields = [
        field
        for field in (*OPTIONAL_IDENTITY_FIELDS, *INVESTIGATION_CONTEXT_FIELDS)
        if field in source.columns
    ]
    safe_source = source[["claim_id", *source_fields]].drop_duplicates(
        "claim_id", keep="first"
    )
    merged = queue[core_columns].merge(safe_source, on="claim_id", how="left")
    rename_map = {
        field: context_label(field)
        for field in source_fields
    }
    merged = merged.rename(columns=rename_map)
    readable_context = [rename_map[field] for field in source_fields]
    return merged[[*core_columns, *readable_context]]


def render_queue_table(display: pd.DataFrame) -> None:
    if display.empty:
        st.markdown(
            '<div class="empty-state"><h3>No cases match these filters</h3>'
            '<p>Clear the search or choose another review status.</p></div>',
            unsafe_allow_html=True,
        )
        return

    rows: list[str] = []
    for item in display.to_dict(orient="records"):
        claim_id = str(item["Claim ID"])
        status_class = re.sub(r"[^a-z0-9]+", "-", str(item["Review status"]).lower()).strip("-")
        rows.append(
            dedent(
                f"""
                <tr>
                  <td class="queue-rank">{escape(item['Rank'])}</td>
                  <td class="queue-claim">{escape(claim_id)}</td>
                  <td class="queue-customer">{escape(item['Customer context'])}</td>
                  <td class="queue-probability">{escape(item['Fraud probability'])}</td>
                  <td>{escape(item['Model decision'])}</td>
                  <td><span class="queue-status status-{escape(status_class)}">{escape(item['Review status'])}</span></td>
                  <td class="queue-action"><a href="?case={quote(claim_id, safe='')}" target="_self" aria-label="View details for {escape(claim_id)}">View details</a></td>
                </tr>
                """
            )
        )
    st.markdown(
        dedent(
            f"""
            <div class="queue-scroll-hint">Swipe horizontally to view all columns</div>
            <div class="queue-table-shell" role="region" aria-label="Review queue table; swipe horizontally to view all columns" tabindex="0">
              <table class="queue-table">
                <caption>Cases selected for investigation</caption>
                <thead><tr>
                  <th scope="col">Rank</th>
                  <th scope="col">Claim ID</th>
                  <th scope="col">Customer context</th>
                  <th scope="col">Fraud probability</th>
                  <th scope="col">Model decision</th>
                  <th scope="col">Review status</th>
                  <th scope="col" class="queue-action">Details</th>
                </tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    batch_name, raw_scores, source, timestamp, is_demo = active_batch()
    scores = with_review_state(raw_scores)
    flagged = scores[scores["flagged"]].sort_values(
        "fraud_probability", ascending=False
    )
    flagged_total = DEMO_FLAGGED_TOTAL if is_demo else int(len(flagged))
    capacity = min(int(st.session_state["review_capacity"]), len(flagged))
    selected = flagged.head(capacity)
    reviewed = int((selected["status"] != "Pending").sum())
    progress = min(100, round(100 * reviewed / max(1, capacity)))
    remaining = max(0, capacity - reviewed)
    outside_cycle = max(0, flagged_total - capacity)
    top_claims = flagged.head(5).copy()
    peak_risk = float(top_claims["fraud_probability"].max()) if not top_claims.empty else 1.0
    source_lookup = source.copy()
    source_lookup["claim_id"] = source_lookup["claim_id"].astype(str)
    source_lookup = source_lookup.drop_duplicates("claim_id").set_index("claim_id")

    with st.container(key="overview_hero"):
        story_col, visual_col = st.columns([1.08, 0.92], gap="large", vertical_alignment="center")
        with story_col:
            st.markdown(
                """
                <div class="overview-story">
                  <h1>Review today’s flagged claims.</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="overview-summary-copy"><strong>{flagged_total:,}</strong> claims are flagged in {escape(batch_name)}. '
                f'<strong>{capacity:,}</strong> are in this review cycle and <strong>{reviewed:,}</strong> have been completed.</p>',
                unsafe_allow_html=True,
            )
            primary_col, secondary_col, spacer_col = st.columns([1.05, 1, 0.55], gap="small")
            with primary_col:
                st.button(
                    "Open review queue",
                    type="primary",
                    width="stretch",
                    on_click=navigate,
                    args=("Review queue",),
                )
            with secondary_col:
                st.button(
                    "Upload claims",
                    width="stretch",
                    on_click=navigate,
                    args=("Data uploads",),
                )
        with visual_col:
            st.markdown(
                f"""
                <div class="overview-visual" aria-label="Active queue snapshot">
                  <div class="overview-visual-head">
                    <span>Current data</span>
                    <strong>{escape(batch_name)}</strong>
                  </div>
                  <div class="overview-data-total">
                    <strong>{flagged_total:,}</strong>
                    <span>flagged claims</span>
                  </div>
                  <div class="overview-cycle-head"><span>Current review cycle</span><strong>{progress}% complete</strong></div>
                  <div class="overview-cycle-track" aria-label="{progress}% of the current review cycle completed">
                    <i style="--progress-width:{progress}%"></i>
                  </div>
                  <div class="overview-data-grid">
                    <div><span>Selected</span><strong>{capacity:,}</strong></div>
                    <div><span>Reviewed</span><strong>{reviewed:,}</strong></div>
                    <div><span>Remaining</span><strong>{remaining:,}</strong></div>
                  </div>
                  <div class="overview-data-foot">
                    <span>{outside_cycle:,} outside this cycle</span>
                    <span>Threshold {probability_text(backend.threshold)}</span>
                    <span>{escape(timestamp)}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    metrics = (
        ("Flagged claims", f"{flagged_total:,}"),
        ("Review capacity", f"{capacity:,}"),
        ("Reviewed", f"{reviewed:,}"),
        ("Active model", "XGBoost baseline"),
    )
    metric_cells = "".join(
        f'<div class="overview-metric"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in metrics
    )
    with st.container(key="overview_metrics"):
        st.markdown(
            f'<section class="overview-metrics" aria-label="Operations summary">{metric_cells}</section>',
            unsafe_allow_html=True,
        )

    claim_rows: list[str] = []
    for rank, item in enumerate(top_claims.to_dict(orient="records"), start=1):
        claim_id = str(item["claim_id"])
        probability = float(item["fraud_probability"])
        relative_width = max(9.0, 100 * probability / max(peak_risk, 0.000001))
        context = "Profile not supplied"
        if claim_id in source_lookup.index:
            context = policyholder_context(source_lookup.loc[claim_id])
        claim_rows.append(
            f"""
            <a class="overview-claim-row" href="?case={quote(claim_id, safe='')}" target="_self" aria-label="View details for {escape(claim_id)}">
              <span class="overview-claim-rank">{rank:02d}</span>
              <span class="overview-claim-name"><strong>{escape(claim_id)}</strong><small>{escape(context)}</small></span>
              <span class="overview-claim-track"><i style="--risk-width:{relative_width:.2f}%"></i></span>
              <span class="overview-claim-risk">{probability_text(probability)}</span>
            </a>
            """
        )

    with st.container(key="overview_modules"):
        queue_col, governance_col = st.columns([1.25, 0.75], gap="large")
        with queue_col:
            st.markdown(
                f"""
                <section class="overview-panel overview-panel--queue">
                  <div class="overview-panel-head">
                    <h2>Top 5 by risk</h2>
                  </div>
                  <div class="overview-claim-list">{''.join(claim_rows)}</div>
                </section>
                """,
                unsafe_allow_html=True,
            )
        with governance_col:
            with st.container(key="overview_governance"):
                st.markdown(
                    f"""
                    <section class="overview-panel overview-panel--governance">
                      <div>
                        <h2>Review progress</h2>
                        <div class="overview-progress-number">{reviewed}<span> / {capacity}</span></div>
                        <div class="overview-progress-track" aria-label="{progress}% of selected cases reviewed">
                          <i style="--progress-width:{progress}%"></i>
                        </div>
                        <p>{remaining:,} cases remain in the current cycle.</p>
                      </div>
                      <div class="overview-governance-line">
                        <span>Model updates</span>
                        <strong>Manual retraining</strong>
                        <small>Validation required before approval</small>
                      </div>
                    </section>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "Open model updates",
                    width="stretch",
                    on_click=navigate,
                    args=("Model updates",),
                )


def render_review_queue() -> None:
    notice = st.session_state.pop("activation_notice", None)
    if notice:
        st.success(notice)

    batch_name, raw_scores, source, _timestamp, is_demo = active_batch()
    scores = with_review_state(raw_scores)
    flagged_total = int(scores["flagged"].sum())
    max_capacity = flagged_total
    capacity = min(int(st.session_state["review_capacity"]), max_capacity)
    st.session_state["review_capacity"] = capacity

    export_frame = operational_queue_export(
        scores, source, capacity=capacity, batch_name=batch_name
    )
    title_col, export_col = st.columns([6.35, 1.15], gap="medium")
    with title_col:
        transition_class = consume_page_transition_class()
        st.markdown(
            f"""
            <div class="page-heading page-heading--compact{transition_class}"><h1>Review queue</h1></div>
            <div class="batch-line"><strong>{escape(batch_name)}</strong></div>
            """,
            unsafe_allow_html=True,
        )
    with export_col:
        st.download_button(
            "Export queue",
            dataframe_to_csv_bytes(export_frame),
            file_name=f"{safe_filename(batch_name)}_review_queue.csv",
            mime="text/csv",
            width="stretch",
        )

    copy_col, input_col, total_col = st.columns([0.23, 0.72, 3.49], gap="small")
    with copy_col:
        st.markdown('<div class="capacity-copy">Top</div>', unsafe_allow_html=True)
    with input_col:
        stored_capacity = int(st.session_state.get("capacity_input", capacity))
        if stored_capacity < 0 or stored_capacity > max_capacity:
            st.session_state["capacity_input"] = capacity
        capacity = int(
            st.number_input(
                "Investigation capacity",
                min_value=0,
                max_value=max_capacity,
                value=capacity,
                step=25 if max_capacity >= 25 else 1,
                label_visibility="collapsed",
                key="capacity_input",
            )
        )
        st.session_state["review_capacity"] = capacity
    with total_col:
        st.markdown(
            f'<div class="capacity-copy">of {flagged_total:,} flagged claims</div>',
            unsafe_allow_html=True,
        )

    ranked = scores.sort_values("fraud_probability", ascending=False).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    selected_pool = ranked[ranked["flagged"]].head(capacity).copy()
    customer_context = source[["claim_id"]].copy()
    customer_context["claim_id"] = customer_context["claim_id"].astype(str)
    customer_context["Customer context"] = source.apply(policyholder_context, axis=1)
    customer_context = customer_context.drop_duplicates("claim_id", keep="first")
    selected_pool["claim_id"] = selected_pool["claim_id"].astype(str)
    selected_pool = selected_pool.merge(customer_context, on="claim_id", how="left")
    with st.container(key="queue_filters"):
        search_col, status_col, sort_col = st.columns([1.5, 1.15, 1.15])
        with search_col:
            search = st.text_input(
                "Search claim ID", placeholder="Search claim ID", label_visibility="collapsed"
            )
        with status_col:
            status_filter = st.selectbox(
                "Review status",
                ("All review statuses",) + REVIEW_STATUSES,
                label_visibility="collapsed",
            )
        with sort_col:
            sort_order = st.selectbox(
                "Sort",
                ("Risk high to low", "Risk low to high"),
                label_visibility="collapsed",
            )

    visible = selected_pool.copy()
    if status_filter != "All review statuses":
        visible = visible[visible["status"] == status_filter]
    if search:
        visible = visible[
            visible["claim_id"].astype(str).str.contains(search, case=False, na=False)
        ]
    visible = visible.sort_values(
        "fraud_probability", ascending=sort_order == "Risk low to high"
    )
    display = pd.DataFrame(
        {
            "Rank": visible["rank"].astype(int),
            "Claim ID": visible["claim_id"].astype(str),
            "Customer context": visible["Customer context"].fillna("Profile not supplied"),
            "Fraud probability": visible["fraud_probability"].map(probability_text),
            "Model decision": visible["classification_label"],
            "Review status": visible["status"],
        }
    ).reset_index(drop=True)

    page_size = 100
    total_matches = len(display)
    total_pages = max(1, (total_matches + page_size - 1) // page_size)
    current_page = min(
        max(1, int(st.session_state.get("queue_page", 1))), total_pages
    )
    if int(st.session_state.get("queue_page", 1)) != current_page:
        st.session_state["queue_page"] = current_page
    page_copy, page_control = st.columns([4.5, 1], vertical_alignment="bottom")
    with page_copy:
        st.caption(
            f"{total_matches:,} matching {'case' if total_matches == 1 else 'cases'} · "
            f"showing up to {page_size:,} per page"
        )
    with page_control:
        current_page = int(
            st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                step=1,
                key="queue_page",
            )
        )
    page_start = (current_page - 1) * page_size
    display = display.iloc[page_start : page_start + page_size].reset_index(drop=True)

    selected_claim = st.session_state.get("selected_claim_id")
    source_ids = set(source["claim_id"].astype(str))
    if selected_claim and selected_claim not in source_ids:
        clear_case_selection()

    render_queue_table(display)


def render_case_page() -> None:
    claim_id = st.session_state.get("selected_claim_id")
    if not claim_id:
        render_page_heading("Case", "Select a claim from the review queue first.")
        st.button("Back to review queue", on_click=navigate, args=("Review queue",))
        return
    batch_name, _scores, source, _timestamp, is_demo = active_batch()
    if str(claim_id) not in set(source["claim_id"].astype(str)):
        if "case" in st.query_params:
            del st.query_params["case"]
        clear_case_selection()
        render_page_heading("Case unavailable", "This claim is not part of the active batch.")
        st.button("Back to review queue", on_click=navigate, args=("Review queue",))
        return
    if is_demo:
        source = load_demo_claim(str(claim_id))
    frame = claim_frame(claim_id, source)
    result = backend.score_claim(frame, top_n=8)
    review = review_store.get(claim_id)
    title_col, back_col = st.columns([5, 1.2], gap="medium")
    with title_col:
        transition_class = consume_page_transition_class()
        st.markdown(
            f'<div class="page-heading page-heading--case{transition_class}"><h1>Case investigation</h1>'
            f'<div class="batch-line"><strong>{escape(batch_name)}</strong><span>{escape(claim_id)}</span></div></div>',
            unsafe_allow_html=True,
        )
    with back_col:
        st.button(
            "Back to queue",
            width="stretch",
            on_click=navigate,
            args=("Review queue",),
        )
    render_case_hero(result, review, capacity=int(st.session_state["review_capacity"]))

    evidence_col, review_col = st.columns([1.65, 0.75], gap="large")
    with evidence_col:
        render_shap_chart(
            result["top_factors"],
            title="What influenced this score",
        )
    with review_col:
        st.markdown(
            '<div class="review-panel-intro"><div class="section-title">Review decision</div></div>',
            unsafe_allow_html=True,
        )
        status_index = REVIEW_STATUSES.index(review["status"])
        with st.form(f"review_form_{claim_id}"):
            review_status = st.selectbox(
                "Review status", REVIEW_STATUSES, index=status_index
            )
            review_notes = st.text_area(
                "Review notes",
                value=review["notes"],
                height=170,
                placeholder="Record evidence checked, questions, and next action.",
            )
            saved = st.form_submit_button("Save review", type="primary", width="stretch")
            if saved:
                review_store.save(claim_id, review_status, review_notes)
                st.cache_data.clear()
                st.success("Review saved locally.")
                st.rerun()
    context_sections, has_identity = claim_context_sections(claim_id, source)
    render_context_sections(context_sections)


def render_upload_results(result: dict[str, Any]) -> None:
    scores: pd.DataFrame = result["scores"]
    source: pd.DataFrame = result["source"]
    validation = result["validation"]
    batch_name = result["batch_name"]
    flagged_count = int(scores["flagged"].sum())
    _current_name, _current_scores, current_source, _timestamp, _is_demo = active_batch()
    current_ids = (
        set()
        if _is_demo
        else set(current_source["claim_id"].astype(str))
    )
    upload_ids = set(source["claim_id"].astype(str))
    duplicate_ids = sorted(upload_ids & current_ids)
    preferred_mode = result.get(
        "upload_mode", st.session_state.get("upload_mode", "Append new cases")
    )
    st.markdown(
        f"""
        <section class="result-summary" aria-label="Scoring complete">
          <div><span>Accepted</span><strong>{len(scores):,}</strong></div>
          <div><span>Flagged</span><strong>{flagged_count:,}</strong></div>
          <div><span>Review rate</span><strong>{scores['flagged'].mean():.1%}</strong></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if result.get("label_status") == "Outcomes present":
        st.markdown(
            '<p class="upload-role"><strong>Training-ready outcomes detected.</strong> Queue scoring remains available, and this labelled schema can also be used in Model updates.</p>',
            unsafe_allow_html=True,
        )
    elif result.get("target_present"):
        st.markdown(
            '<p class="upload-role"><strong>Scoring only.</strong> The outcome column contains missing or invalid values, so these rows will not be counted as training-ready.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="upload-role"><strong>Scoring only.</strong> No fraud outcome column was supplied, which is valid for new claims.</p>',
            unsafe_allow_html=True,
        )
    if duplicate_ids:
        st.warning(
            "Duplicate claim IDs found. Choose Replace to continue."
        )
    operation = "append" if preferred_mode == "Append new cases" else "replace"
    action_label = (
        "Start current queue"
        if _is_demo
        else "Add to queue"
        if operation == "append"
        else "Replace queue"
    )
    if st.button(
        action_label,
        type="primary",
        disabled=bool(duplicate_ids) and operation == "append",
    ):
        ok, message = activate_upload(operation)
        if ok:
            st.rerun()
        st.error(message)

    with st.expander("Preview scored claims"):
        preview = scores.sort_values("fraud_probability", ascending=False).copy()
        preview["Fraud probability"] = preview["fraud_probability"].map(probability_text)
        st.dataframe(
            preview[["claim_id", "Fraud probability", "classification_label"]].head(100),
            hide_index=True,
            width="stretch",
            height=320,
            column_config={
                "claim_id": "Claim ID",
                "classification_label": "Model decision",
            },
        )

    with st.expander("Export results and audit files"):
        max_capacity = max(1, flagged_count)
        export_capacity = int(
            st.number_input(
                "Cases selected for review",
                min_value=1,
                max_value=max_capacity,
                value=min(int(st.session_state["review_capacity"]), max_capacity),
                step=25 if max_capacity >= 25 else 1,
                key="upload_export_capacity",
            )
        )
        export_frame = prepare_scored_export(
            source,
            scores,
            selected_count=export_capacity,
            batch_name=batch_name,
            model_track=backend.config["model_track"],
            threshold=backend.threshold,
            scored_at_utc=result["scored_at_utc"],
        )
        scope_col, format_col = st.columns(2)
        with scope_col:
            scope = st.selectbox(
                "Export scope", ("Selected review queue", "All scored records")
            )
        with format_col:
            export_format = st.selectbox("File format", ("CSV", "Parquet"))
        export_data = (
            export_frame[export_frame["selected_for_review"]].copy()
            if scope == "Selected review queue"
            else export_frame
        )
        if export_format == "CSV":
            payload = dataframe_to_csv_bytes(export_data)
            suffix, mime = "csv", "text/csv"
        else:
            payload = dataframe_to_parquet_bytes(export_data)
            suffix, mime = "parquet", "application/octet-stream"
        stem = safe_filename(batch_name)
        scope_name = "review_queue" if scope == "Selected review queue" else "all_scores"
        manifest = {
            "batch_name": batch_name,
            "source_file": result["source_file"],
            "model_family": backend.config["model_family"],
            "model_track": backend.config["model_track"],
            "q_star_threshold": backend.threshold,
            "approved_feature_count": len(backend.approved_features),
            "rows_scored": len(scores),
            "rows_flagged": flagged_count,
            "selected_for_review": export_capacity,
            "scored_at_utc": result["scored_at_utc"],
            "target_removed_from_export": True,
            "explanation_method": backend.config["explanation_method"],
        }
        audit_zip = build_audit_package(
            export_frame, validation=validation, manifest=manifest
        )
        export_col, audit_col = st.columns(2)
        with export_col:
            st.download_button(
                f"Download {scope_name.replace('_', ' ')}",
                payload,
                file_name=f"{stem}_{scope_name}.{suffix}",
                mime=mime,
                type="primary",
                width="stretch",
            )
        with audit_col:
            st.download_button(
                "Download audit package",
                audit_zip,
                file_name=f"{stem}_audit_package.zip",
                mime="application/zip",
                width="stretch",
            )
        retained_extra = [
            column
            for column in validation["extra_columns"]
            if column not in LEAKAGE_SOURCE_FIELDS
        ]
        removed_leakage = [
            column
            for column in validation["extra_columns"]
            if column in LEAKAGE_SOURCE_FIELDS
        ]
        if retained_extra:
            st.caption(
                "Extra columns were kept in the export and ignored by the model: "
                + ", ".join(retained_extra[:20])
            )
        if removed_leakage:
            st.caption(
                "Legacy leakage fields were removed from the queue and export: "
                + ", ".join(removed_leakage)
            )


def render_upload_history() -> None:
    events = data_upload_store.list_all()
    st.markdown('<div class="section-title">Upload history</div>', unsafe_allow_html=True)
    if notice := st.session_state.pop("upload_undo_notice", None):
        st.success(notice)
    if events.empty:
        st.markdown(
            '<div class="compact-empty"><strong>No queue updates yet</strong>'
            '<p>Files appear here after a queue update is confirmed.</p></div>',
            unsafe_allow_html=True,
        )
        return

    history = events.copy()
    history["activated_at_utc"] = history["activated_at_utc"].map(format_updated)
    st.dataframe(
        history[
            [
                "batch_name",
                "operation",
                "row_count",
                "flagged_count",
                "label_status",
                "status",
                "activated_at_utc",
            ]
        ].head(50),
        hide_index=True,
        width="stretch",
        column_config={
            "batch_name": "Batch",
            "operation": "Queue action",
            "row_count": "Rows",
            "flagged_count": "Flagged",
            "label_status": "Outcome labels",
            "status": "Status",
            "activated_at_utc": "Updated",
        },
    )
    active_events = events.loc[events["status"].eq("Active")]
    if active_events.empty:
        return
    latest = active_events.iloc[0]
    upload_id = str(latest["upload_id"])
    can_undo = operational_queue_store.can_undo(upload_id)
    action_col, context_col = st.columns([0.9, 2.2], vertical_alignment="center")
    with action_col:
        undo_requested = st.button(
            "Undo latest queue update",
            disabled=not can_undo,
            width="stretch",
        )
    with context_col:
        if can_undo:
            st.caption(
                f"Restores the queue used before {latest['batch_name']} and removes its rows from the backend active queue."
            )
        else:
            st.caption(
                "This older record has no reversible queue snapshot. New updates are fully reversible."
            )
    if undo_requested:
        ok, message = undo_latest_upload(upload_id)
        if ok:
            st.session_state["upload_undo_notice"] = message
            st.rerun()
        st.error(message)


def render_data_uploads() -> None:
    render_page_heading("Data uploads")
    _active_name, _active_scores, _active_source, _active_timestamp, queue_is_demo = (
        active_batch()
    )

    template_columns = [
        "claim_id",
        *backend.approved_features,
        backend.config["target_column"],
    ]
    template_csv = pd.DataFrame(columns=template_columns).to_csv(index=False).encode("utf-8-sig")
    with st.container(key="upload_header"):
        upload_title_col, template_col = st.columns([4.5, 1], vertical_alignment="bottom")
        with upload_title_col:
            st.markdown(
                '<div class="upload-heading-line"><div class="section-title section-title--flush">Upload file</div><span>Fraud outcome optional</span></div>',
                unsafe_allow_html=True,
            )
        with template_col:
            st.download_button(
                "Download template",
                template_csv,
                file_name="claim_upload_template.csv",
                mime="text/csv",
                width="stretch",
            )

    with st.container(key="upload_workspace"):
        uploaded = st.file_uploader(
            "File",
            type=["csv", "parquet"],
            key="claims_upload",
            on_change=clear_upload_result,
            label_visibility="collapsed",
        )

    with st.container(key="upload_actions"):
        action_col, name_col, score_col = st.columns(
            [0.85, 1.45, 1], gap="medium", vertical_alignment="bottom"
        )
        with action_col:
            if queue_is_demo:
                upload_action = st.selectbox(
                    "Queue action",
                    ("Start new queue",),
                    key="upload_queue_action_demo",
                )
            else:
                upload_action = st.selectbox(
                    "Queue action",
                    ("Add", "Replace"),
                    index=0
                    if st.session_state.get("upload_mode", "Append new cases")
                    == "Append new cases"
                    else 1,
                    key="upload_queue_action",
                )
        normalized_mode = (
            "Append new cases"
            if upload_action == "Add"
            else "Replace active queue"
        )
        st.session_state["upload_mode"] = normalized_mode
        with name_col:
            batch_name = st.text_input(
                "Batch name",
                value=st.session_state.get("upload_batch_name", "Claims_2026_08"),
                key="upload_batch_name_field",
            )
            st.session_state["upload_batch_name"] = batch_name
        with score_col:
            score_requested = st.button(
                "Validate and score",
                type="primary",
                disabled=uploaded is None,
                width="stretch",
            )

    existing_result = st.session_state.get("upload_result")
    if existing_result:
        existing_result["upload_mode"] = normalized_mode
        existing_result["batch_name"] = batch_name.strip() or existing_result["batch_name"]
    if score_requested:
        try:
            uploaded_frame = read_upload(uploaded)
            if "claim_id" not in uploaded_frame:
                if normalized_mode == "Append new cases":
                    raise ValueError(
                        "Append uploads need a claim_id column so new cases can be matched safely."
                    )
                uploaded_frame.insert(
                    0,
                    "claim_id",
                    [f"UPLOAD-{index:06d}" for index in range(len(uploaded_frame))],
                )
            if uploaded_frame["claim_id"].astype(str).duplicated().any():
                raise ValueError("Claim IDs must be unique within an uploaded batch.")
            validation = backend.validate_claims(uploaded_frame)
            if validation["missing_features"]:
                raise ValueError(
                    "Missing required features: "
                    + ", ".join(validation["missing_features"][:20])
                )
            if validation["unknown_categories"]:
                unknown = "; ".join(
                    f"{feature}: {values[:5]}"
                    for feature, values in validation["unknown_categories"].items()
                )
                raise ValueError(
                    "The upload contains categorical values that were not present in training. "
                    + unknown
                )
            scored_at = utc_timestamp()
            scores = backend.score_claims(uploaded_frame).sort_values(
                "fraud_probability", ascending=False
            )
            target_column = backend.config["target_column"]
            if target_column in uploaded_frame:
                uploaded_labels = normalize_outcome_labels(uploaded_frame[target_column])
                labels_valid = bool(
                    len(uploaded_labels)
                    and uploaded_labels.notna().all()
                    and uploaded_labels.isin([0, 1]).all()
                )
                invalid_label_rows = int(
                    (uploaded_labels.isna() | ~uploaded_labels.isin([0, 1])).sum()
                )
            else:
                uploaded_labels = pd.Series(dtype="float64")
                labels_valid = False
                invalid_label_rows = 0
            safe_source = uploaded_frame.drop(
                columns=[target_column, *LEAKAGE_SOURCE_FIELDS], errors="ignore"
            )
            st.session_state["upload_result"] = {
                "batch_name": batch_name.strip() or Path(uploaded.name).stem,
                "source_file": uploaded.name,
                "source": safe_source,
                "scores": scores,
                "validation": validation,
                "scored_at_utc": scored_at,
                "upload_mode": normalized_mode,
                "labelled_rows": len(uploaded_frame) if labels_valid else 0,
                "fraud_labels": int((uploaded_labels == 1).sum())
                if labels_valid
                else 0,
                "label_status": "Outcomes present"
                if labels_valid
                else "Outcomes invalid"
                if target_column in uploaded_frame
                else "Labels not included",
                "invalid_label_rows": invalid_label_rows,
                "target_present": target_column in uploaded_frame,
            }
            if labels_valid:
                st.success(
                    "File scored. Valid fraud outcomes were detected, so this schema is also eligible for retraining."
                )
            elif target_column in uploaded_frame:
                st.warning(
                    f"File scored as unlabelled because {invalid_label_rows:,} outcome values are missing or invalid. Queue scoring is still available."
                )
            else:
                st.success(
                    "File scored without fraud outcomes. Labels are not required for queue scoring."
                )
        except Exception as exc:
            st.error(f"The batch could not be scored. {exc}")

    result = st.session_state.get("upload_result")
    render_upload_history()
    if result:
        render_upload_results(result)


def render_monitoring() -> None:
    render_page_heading("Monitoring")
    batch_name, raw_scores, _source, _timestamp, _is_demo = active_batch()
    scores = with_review_state(raw_scores)
    reviews = review_store.list_all()
    reviewed = int((scores["status"] != "Pending").sum())
    render_stat_strip(
        (
            ("Active batch", batch_name),
            ("Flagged", f"{int(scores['flagged'].sum()):,}"),
            ("Reviewed", f"{reviewed:,}"),
            ("Review completion", f"{reviewed / max(1, len(scores)):.1%}"),
        )
    )
    st.markdown('<div class="section-title">Investigation outcomes</div>', unsafe_allow_html=True)
    counts = (
        scores["status"]
        .value_counts()
        .reindex(REVIEW_STATUSES, fill_value=0)
        .rename_axis("Review status")
        .reset_index(name="Cases")
    )
    st.dataframe(counts, hide_index=True, width="stretch")
    st.markdown('<div class="section-title">Recent review activity</div>', unsafe_allow_html=True)
    if reviews.empty:
        st.markdown(
            '<div class="empty-state"><h3>No completed review actions</h3>'
            '<p>Activity appears here after an investigator changes a case status or saves notes.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        recent = reviews.sort_values("reviewed_at_utc", ascending=False).copy()
        recent["reviewed_at_utc"] = recent["reviewed_at_utc"].map(format_updated)
        st.dataframe(
            recent[["claim_id", "status", "reviewed_at_utc", "notes"]].head(50),
            hide_index=True,
            width="stretch",
            column_config={
                "claim_id": "Claim ID",
                "status": "Review status",
                "reviewed_at_utc": "Updated",
                "notes": "Investigator notes",
            },
        )
        st.download_button(
            "Export review log",
            dataframe_to_csv_bytes(recent),
            file_name="fraud_review_activity.csv",
            mime="text/csv",
        )


def render_model_update_log() -> None:
    update_events = model_update_store.list_events()
    open_runs = model_update_store.open_runs()
    if open_runs:
        latest_by_run = update_events.drop_duplicates("run_id", keep="first").set_index(
            "run_id"
        )
        latest_run = open_runs[0]
        latest_status = str(latest_by_run.loc[latest_run, "status"])
        worker_status = training_service.status(latest_run)
        st.markdown(
            f"""
            <section class="active-model-run" aria-label="Active training run">
              <div><span>Active run</span><strong>{escape(latest_run)}</strong></div>
              <div><span>Status</span><strong>{escape(latest_status)}</strong></div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        if latest_status == "Training requested":
            progress_value = max(0, min(100, int(worker_status.get("percent", 0))))
            st.progress(progress_value, text=str(worker_status.get("message", "Training")))
            if st.button("Refresh training status", width="stretch"):
                st.rerun()

        candidate = training_service.candidate(latest_run)
        if candidate and latest_status in {"Candidate ready", "Validation passed"}:
            candidate_profile = candidate["profile"]
            incumbent = candidate.get("incumbent_on_candidate_holdout", {})
            comparable_incumbent = incumbent if not incumbent.get("error") else {}
            st.markdown('<div class="section-title">Candidate comparison</div>', unsafe_allow_html=True)
            st.caption("Both models are evaluated on this run’s grouped holdout.")
            comparison = pd.DataFrame(
                [
                    {
                        "Model": "Current",
                        "ROC-AUC": comparable_incumbent.get("roc_auc"),
                        "PR-AUC": comparable_incumbent.get("pr_auc"),
                        "Fraud detected": comparable_incumbent.get("tp"),
                        "Fraud missed": comparable_incumbent.get("fn"),
                        "Legitimate flagged": comparable_incumbent.get("fp"),
                    },
                    {
                        "Model": "Candidate",
                        "ROC-AUC": candidate_profile["roc_auc"],
                        "PR-AUC": candidate_profile["pr_auc"],
                        "Fraud detected": candidate_profile["fraud_detected"],
                        "Fraud missed": candidate_profile["fraud_missed"],
                        "Legitimate flagged": candidate_profile["legitimate_flagged"],
                    },
                ]
            )
            if not comparable_incumbent:
                st.warning(
                    "The active model could not be rescored on this holdout. Review the candidate artefacts before approval."
                )
            st.dataframe(
                comparison,
                hide_index=True,
                width="stretch",
                column_config={
                    "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
                    "PR-AUC": st.column_config.NumberColumn(format="%.4f"),
                },
            )

        if latest_status == "Candidate ready":
            with st.form("candidate_validation_form"):
                st.markdown('<div class="form-title">Validation decision</div>', unsafe_allow_html=True)
                decision = st.radio(
                    "Decision",
                    ("Approve candidate", "Reject candidate"),
                    horizontal=True,
                )
                decision_note = st.text_area(
                    "Validation note",
                    placeholder="Record the data, performance, and governance checks completed.",
                    height=90,
                )
                checks_complete = st.checkbox(
                    "I reviewed the holdout results and candidate artefacts."
                )
                save_decision = st.form_submit_button(
                    "Record decision", type="primary", width="stretch"
                )
                if save_decision:
                    if not decision_note.strip() or not checks_complete:
                        st.error("Complete the validation check and add an audit note.")
                    else:
                        status = (
                            "Validation passed"
                            if decision == "Approve candidate"
                            else "Rejected"
                        )
                        model_update_store.record_event(
                            latest_run,
                            status,
                            decision_note.strip(),
                            actor="Model reviewer",
                            metadata=model_update_store.latest_metadata(latest_run),
                        )
                        st.rerun()

        if latest_status == "Validation passed":
            with st.form("candidate_deployment_form"):
                st.markdown('<div class="form-title">Deploy candidate</div>', unsafe_allow_html=True)
                st.caption(
                    "Deployment switches the active model atomically. The previous configuration is retained for rollback."
                )
                confirmed = st.checkbox(
                    "Activate this candidate for scoring and investigation."
                )
                deploy = st.form_submit_button(
                    "Deploy approved model", type="primary", width="stretch"
                )
                if deploy:
                    if not confirmed:
                        st.error("Confirm the production switch before deployment.")
                    else:
                        try:
                            training_service.deploy(latest_run, model_update_store)
                        except Exception as exc:
                            st.error(f"Deployment did not complete. {exc}")
                        else:
                            load_backend.clear()
                            load_demo_scores.clear()
                            load_global_shap.clear()
                            for key in list(st.session_state):
                                if key.startswith("active_batch_"):
                                    del st.session_state[key]
                            st.success("The approved model is now active.")
                            st.rerun()

        if latest_status == "Validation failed":
            st.error(str(worker_status.get("message", "The training run failed.")))
            if st.button("Close failed run", width="stretch"):
                model_update_store.record_event(
                    latest_run,
                    "Rejected",
                    "Closed after the training or validation failure.",
                    actor="Model reviewer",
                    metadata=model_update_store.latest_metadata(latest_run),
                )
                st.rerun()

    if update_events.empty:
        return

    with st.expander("Update history"):
        display_events = update_events.copy()
        display_events["recorded_at_utc"] = display_events["recorded_at_utc"].map(
            format_updated
        )
        st.dataframe(
            display_events[
                ["recorded_at_utc", "run_id", "status", "summary", "actor"]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "recorded_at_utc": "Recorded",
                "run_id": "Run ID",
                "status": "Status",
                "summary": "Update note",
                "actor": "Source",
            },
        )
        st.download_button(
            "Export update log",
            dataframe_to_csv_bytes(update_events.drop(columns=["metadata_json"])),
            file_name="model_update_log.csv",
            mime="text/csv",
        )

    deployment = training_service.latest_deployment()
    if deployment and not deployment.get("rolled_back_at"):
        with st.expander("Rollback last deployment"):
            st.caption(
                f"Restore the model used before {deployment['deployment_id']}. This action is logged."
            )
            rollback_confirmed = st.checkbox(
                "I understand that scoring will return to the previous model.",
                key="rollback_confirmation",
            )
            if st.button(
                "Restore previous model",
                disabled=not rollback_confirmed,
                width="stretch",
            ):
                try:
                    training_service.rollback_latest(model_update_store)
                except Exception as exc:
                    st.error(f"Rollback did not complete. {exc}")
                else:
                    load_backend.clear()
                    load_demo_scores.clear()
                    load_global_shap.clear()
                    for key in list(st.session_state):
                        if key.startswith("active_batch_"):
                            del st.session_state[key]
                    st.success("The previous model has been restored.")
                    st.rerun()


def model_update_uploads(profile: dict[str, Any]) -> pd.DataFrame:
    uploads = data_upload_store.list_all()
    if not uploads.empty:
        uploads = uploads.loc[
            uploads["model_version"].astype(str).eq(str(profile["model_version"]))
            & uploads["status"].astype(str).eq("Active")
        ].copy()

    current_result = st.session_state.get("upload_result")
    if current_result:
        scored_at = str(current_result.get("scored_at_utc", ""))
        already_recorded = bool(
            not uploads.empty
            and (
                uploads["source_file"].astype(str).eq(
                    str(current_result.get("source_file", ""))
                )
                & uploads["scored_at_utc"].astype(str).eq(scored_at)
            ).any()
        )
        if not already_recorded:
            active_timestamp = str(st.session_state.get("active_batch_timestamp", ""))
            is_active = bool(scored_at and scored_at == active_timestamp)
            operation = (
                "Add"
                if is_active
                and st.session_state.get("upload_mode") == "Append new cases"
                else "Replace"
                if is_active
                else "Scored only"
            )
            scores = current_result["scores"]
            session_row = pd.DataFrame(
                [
                    {
                        "batch_name": current_result["batch_name"],
                        "source_file": current_result["source_file"],
                        "operation": operation,
                        "row_count": int(len(scores)),
                        "flagged_count": int(scores["flagged"].sum()),
                        "labelled_rows": int(current_result.get("labelled_rows", 0)),
                        "fraud_labels": int(current_result.get("fraud_labels", 0)),
                        "label_status": str(
                            current_result.get("label_status", "Labels needed")
                        ),
                        "scored_at_utc": scored_at,
                        "activated_at_utc": scored_at,
                        "model_version": str(profile["model_version"]),
                    }
                ]
            )
            uploads = pd.concat([session_row, uploads], ignore_index=True)
    return uploads


def render_model_data_change_summary(profile: dict[str, Any]) -> None:
    uploads = model_update_uploads(profile)
    if uploads.empty:
        st.markdown(
            f"""
            <section class="model-data-empty" aria-label="New data since current model">
              <span>New data since {escape(profile['model_version'])}</span>
              <h2>No new uploaded batches</h2>
              <p>Operational uploads will appear here with their row counts and label readiness.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    total_rows = int(uploads["row_count"].sum())
    total_flagged = int(uploads["flagged_count"].sum())
    labelled_rows = int(uploads["labelled_rows"].sum())
    fraud_labels = int(uploads["fraud_labels"].sum())
    historical_rows = int(
        profile.get(
            "source_rows",
            int(profile["training_rows"]) + int(profile["evaluation_rows"]),
        )
    )
    proposed_rows = historical_rows + labelled_rows
    label_message = (
        f"{labelled_rows:,} rows already include valid outcomes."
        if labelled_rows
        else "Outcome labels are still required before these rows can train a model."
    )
    st.markdown(
        f"""
        <section class="model-data-summary" aria-label="New data since current model">
          <div class="model-data-summary-heading">
            <div><span>Since {escape(profile['model_version'])}</span><h2>New uploaded data</h2></div>
            <strong>{len(uploads):,} {'batch' if len(uploads) == 1 else 'batches'}</strong>
          </div>
          <div class="model-data-summary-grid">
            <div><span>Rows uploaded</span><strong>{total_rows:,}</strong></div>
            <div><span>Flagged by current model</span><strong>{total_flagged:,}</strong></div>
            <div><span>Rows with outcomes</span><strong>{labelled_rows:,}</strong></div>
            <div><span>Fraud outcomes</span><strong>{fraud_labels:,}</strong></div>
          </div>
          <p>{escape(label_message)}</p>
        </section>
        <section class="model-change-preview" aria-label="Changes made by retraining">
          <div><span>Current model source</span><strong>{historical_rows:,} rows</strong></div>
          <div><span>After eligible additions</span><strong>{proposed_rows:,} rows</strong></div>
          <div><span>Retraining rebuilds</span><strong>Grouped split + five folds</strong></div>
          <div><span>Production model</span><strong>Unchanged until approval</strong></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    batch_table = uploads.copy()
    batch_table["activated_at_utc"] = batch_table["activated_at_utc"].map(
        format_updated
    )
    st.dataframe(
        batch_table[
            [
                "batch_name",
                "operation",
                "row_count",
                "flagged_count",
                "label_status",
                "activated_at_utc",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "batch_name": "Batch",
            "operation": "Queue action",
            "row_count": "Rows",
            "flagged_count": "Flagged",
            "label_status": "Training labels",
            "activated_at_utc": "Uploaded",
        },
    )


def render_model_data() -> None:
    render_page_heading("Model updates")
    st.markdown(
        """
        <p class="model-lede">Train a new model from labelled monthly data. The current model changes only after validation and approval.</p>
        """,
        unsafe_allow_html=True,
    )

    profile = current_model_profile()
    render_model_data_change_summary(profile)

    if started_run := st.session_state.pop("training_run_started", None):
        st.success(f"{started_run} is training in the background. The current model remains active.")

    active_runs = model_update_store.open_runs()
    if st.button(
        "Start retraining",
        type="primary",
        disabled=bool(active_runs),
        key="start_retraining",
    ):
        st.session_state["show_retraining_form"] = True
    if active_runs:
        st.caption("Finish or close the active run before starting another retraining job.")

    if st.session_state.get("show_retraining_form"):
        st.markdown('<div class="form-title">New retraining run</div>', unsafe_allow_html=True)
        training_file = st.file_uploader(
            "Labelled training data",
            type=["csv", "parquet"],
            key="retraining_labelled_file",
            help="Required: the 144 approved model features and respuesta_dicot_c. The three legacy leakage fields are optional and removed when present.",
        )
        training_frame: pd.DataFrame | None = None
        validation: dict[str, Any] | None = None
        historical_rows = int(
            profile.get(
                "source_rows",
                int(profile["training_rows"]) + int(profile["evaluation_rows"]),
            )
        )

        if training_file is not None:
            try:
                training_frame = read_upload(training_file)
                validation = validate_training_upload(training_frame)
            except Exception as exc:
                st.error(f"The training file could not be read. {exc}")
            else:
                if not validation["valid"]:
                    problems: list[str] = []
                    if validation["missing"]:
                        problems.append("missing " + ", ".join(validation["missing"][:6]))
                    if validation["extra"]:
                        problems.append("unexpected " + ", ".join(validation["extra"][:6]))
                    if validation["invalid_labels"]:
                        problems.append(
                            f"{validation['invalid_labels']:,} invalid target values"
                        )
                    st.error(
                        "Training file validation failed: "
                        + "; ".join(problems)
                        + "."
                    )
                else:
                    combined_rows = historical_rows + validation["row_count"]
                    st.markdown(
                        f"""
                        <section class="retraining-summary" aria-label="Retraining change summary">
                          <h3>Confirm this update</h3>
                          <p>{escape(validation['schema'])} detected. The new labelled rows will be appended to the current deployed data snapshot.</p>
                          <div class="retraining-summary-grid">
                            <div><span>New file</span><strong>{escape(training_file.name)}</strong></div>
                            <div><span>Rows being added</span><strong>{validation['row_count']:,}</strong></div>
                            <div><span>Fraud labels</span><strong>{validation['fraud_rows']:,}</strong></div>
                            <div><span>Combined source</span><strong>{combined_rows:,} rows</strong></div>
                          </div>
                          <div class="retraining-change-list">
                            <div><strong>Data preparation</strong>Use respuesta_dicot_c as the target and remove legacy leakage fields when they are present.</div>
                            <div><strong>Dataset rebuild</strong>Regenerate the grouped 80/20 train and holdout split from the combined source.</div>
                            <div><strong>Model training</strong>Recreate five grouped folds and train XGBoost with the same 144 approved features.</div>
                            <div><strong>Validation output</strong>Create a candidate model, holdout metrics, predictions, SHAP output, checksums, and an audit manifest.</div>
                          </div>
                          <div class="retraining-guardrail"><strong>No immediate production change.</strong> {escape(profile['model_version'])} remains active for scoring until the candidate is reviewed, approved, and deployed.</div>
                        </section>
                        """,
                        unsafe_allow_html=True,
                    )

        file_is_valid = bool(validation and validation["valid"] and training_frame is not None)
        if file_is_valid:
            with st.form("retraining_request_form"):
                request_note = st.text_area(
                    "Reason for update",
                    placeholder="For example: August outcomes are complete and drift review is due.",
                    height=90,
                )
                change_confirmed = st.checkbox(
                    "I confirm that these rows should be appended and the grouped datasets rebuilt."
                )
                submit_col, cancel_col = st.columns([1, 0.55])
                with submit_col:
                    requested = st.form_submit_button(
                        "Confirm and start retraining", type="primary", width="stretch"
                    )
                with cancel_col:
                    cancelled = st.form_submit_button("Cancel", width="stretch")
                if cancelled:
                    st.session_state["show_retraining_form"] = False
                    st.session_state.pop("retraining_labelled_file", None)
                    st.rerun()
                if requested:
                    if not request_note.strip():
                        st.error("Add a reason so the update can be audited later.")
                    elif not change_confirmed:
                        st.error("Confirm the listed data and model changes before retraining.")
                    else:
                        metadata = {
                            "source_file": training_file.name,
                            "source_rows": validation["row_count"],
                            "source_columns": validation["source_columns"],
                            "fraud_rows": validation["fraud_rows"],
                            "removed_fields": validation["leakage_fields_present"],
                            "training_features": len(backend.approved_features),
                            "historical_source_rows": historical_rows,
                            "combined_source_rows": historical_rows
                            + validation["row_count"],
                        }
                        run_id = model_update_store.start_run(
                            (
                                f"Added {validation['row_count']:,} labelled rows "
                                f"({validation['fraud_rows']:,} fraud). "
                                f"{request_note.strip()}"
                            ),
                            actor="Model owner",
                            metadata=metadata,
                        )
                        try:
                            upload_path = training_service.create_input(
                                run_id,
                                training_frame,
                                original_name=training_file.name,
                            )
                            process_id = training_service.launch(run_id, upload_path)
                        except Exception as exc:
                            model_update_store.record_event(
                                run_id,
                                "Validation failed",
                                f"The training worker could not start: {exc}",
                                actor="Training service",
                                metadata=metadata,
                            )
                            st.error(f"The training run could not start. {exc}")
                        else:
                            st.session_state["show_retraining_form"] = False
                            st.session_state.pop("retraining_labelled_file", None)
                            st.session_state["training_run_started"] = run_id
                            st.session_state["training_process_id"] = process_id
                            st.rerun()
        elif st.button("Cancel", key="cancel_retraining_without_valid_file"):
            st.session_state["show_retraining_form"] = False
            st.session_state.pop("retraining_labelled_file", None)
            st.rerun()

    render_model_update_log()

    profile = current_model_profile()
    st.markdown(
        f"""
        <section class="current-model-heading">
          <div><h2>Current model</h2><p>{escape(profile['training_snapshot'])}</p></div>
          <strong>Deployed</strong>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_stat_strip(
        (
            ("Version", str(profile["model_version"])),
            ("Training rows", f"{int(profile['training_rows']):,}"),
            ("Evaluation rows", f"{int(profile['evaluation_rows']):,}"),
            ("Model features", f"{len(backend.approved_features):,}"),
        )
    )
    info_col, shap_col = st.columns([0.9, 1.25], gap="large")
    with info_col:
        st.markdown('<div class="section-title" style="margin-top:.2rem">Operating point</div>', unsafe_allow_html=True)
        render_detail_grid(
            {
                "Threshold": f"{float(profile['threshold']):.10f}",
                "Fraud detected": (
                    f"{int(profile['fraud_detected']):,} of "
                    f"{int(profile['evaluation_fraud']):,}"
                ),
                "Fraud missed": f"{int(profile['fraud_missed']):,}",
                "Legitimate claims flagged": f"{int(profile['legitimate_flagged']):,}",
                "ROC-AUC": (
                    "Not recorded"
                    if profile["roc_auc"] is None
                    else f"{float(profile['roc_auc']):.4f}"
                ),
                "PR-AUC": (
                    "Not recorded"
                    if profile["pr_auc"] is None
                    else f"{float(profile['pr_auc']):.4f}"
                ),
            },
            class_name="detail-grid model-detail-grid",
        )
    with shap_col:
        importance = load_global_shap()
        render_shap_chart(
            importance,
            title="Feature influence",
            absolute=True,
        )

render_sidebar()
render_mobile_navigation()
page = st.session_state["active_page"]
if page == "Overview":
    render_overview()
elif page == "Review queue":
    render_review_queue()
elif page == "Case":
    render_case_page()
elif page == "Data uploads":
    render_data_uploads()
elif page == "Monitoring":
    render_monitoring()
elif page == "Model updates":
    render_model_data()
else:
    navigate("Overview")
    st.rerun()
