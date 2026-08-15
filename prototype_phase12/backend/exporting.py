from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Any

import pandas as pd


EXPORT_COLUMNS = [
    "claim_id",
    "review_rank",
    "fraud_probability",
    "threshold",
    "flagged",
    "classification_label",
    "selected_for_review",
    "batch_name",
    "model_track",
    "scored_at_utc",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prepare_scored_export(
    source_frame: pd.DataFrame,
    scored_frame: pd.DataFrame,
    *,
    selected_count: int,
    batch_name: str,
    model_track: str,
    threshold: float,
    scored_at_utc: str | None = None,
) -> pd.DataFrame:
    """Join model output to user-safe inputs and add reproducible review metadata."""
    if "claim_id" not in source_frame or "claim_id" not in scored_frame:
        raise ValueError("Both source and score frames must contain claim_id.")

    scored_at = scored_at_utc or utc_timestamp()
    safe_source = source_frame.drop_duplicates("claim_id", keep="first").copy()
    ranked = scored_frame.sort_values("fraud_probability", ascending=False).copy()
    ranked["review_rank"] = range(1, len(ranked) + 1)
    selected_limit = max(0, min(int(selected_count), int(ranked["flagged"].sum())))
    ranked["selected_for_review"] = (
        ranked["flagged"] & (ranked["review_rank"] <= selected_limit)
    )
    ranked["batch_name"] = str(batch_name)
    ranked["model_track"] = str(model_track)
    ranked["threshold"] = float(threshold)
    ranked["scored_at_utc"] = scored_at

    score_columns = [column for column in EXPORT_COLUMNS if column in ranked.columns]
    merged = safe_source.merge(ranked[score_columns], on="claim_id", how="left")
    ordered = score_columns + [
        column for column in merged.columns if column not in score_columns
    ]
    return merged[ordered]


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def build_audit_package(
    export_frame: pd.DataFrame,
    *,
    validation: dict[str, Any],
    manifest: dict[str, Any],
) -> bytes:
    """Create a portable ZIP with all scores, the active queue, and provenance."""
    buffer = io.BytesIO()
    queue = export_frame[export_frame["selected_for_review"]].copy()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("all_scored_results.csv", dataframe_to_csv_bytes(export_frame))
        archive.writestr("selected_review_queue.csv", dataframe_to_csv_bytes(queue))
        archive.writestr(
            "validation_report.json",
            json.dumps(validation, indent=2, ensure_ascii=False, default=str),
        )
        archive.writestr(
            "run_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        )
    return buffer.getvalue()
