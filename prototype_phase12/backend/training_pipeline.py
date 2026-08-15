from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .review_store import ModelUpdateStore


TARGET_COLUMN = "respuesta_dicot_c"
LEAKAGE_FIELDS = ("situacionvts", "scoreada", "respuesta_dicot1")
N_SPLITS = 5
RANDOM_STATE = 42
TEST_FOLD = 0
VALIDATION_FOLD = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_approved_features(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("approved_features")
    if not isinstance(features, list) or len(features) != 144:
        raise ValueError("The approved feature file must contain exactly 144 features.")
    if len(features) != len(set(features)) or TARGET_COLUMN in features:
        raise ValueError("The approved feature list is invalid.")
    return [str(feature) for feature in features]


def normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        target = pd.to_numeric(series, errors="coerce")
    else:
        target = (
            series.astype("string")
            .str.strip()
            .str.upper()
            .map({"NO FRAUDE": 0, "FRAUDE": 1, "0": 0, "1": 1})
        )
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("The target must contain only 0/1 or FRAUDE/NO FRAUDE.")
    return target.astype("int8")


def prepare_uploaded_source(frame: pd.DataFrame, approved: list[str]) -> pd.DataFrame:
    required = set(approved) | {TARGET_COLUMN}
    optional = {*LEAKAGE_FIELDS, "claim_id", "row_index"}
    missing = sorted(required - set(frame.columns))
    extra = sorted(set(frame.columns) - required - optional)
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing[:10]))
        if extra:
            details.append("unexpected " + ", ".join(extra[:10]))
        raise ValueError("The labelled training schema is invalid: " + "; ".join(details))
    prepared = frame[approved + [TARGET_COLUMN]].copy()
    prepared[TARGET_COLUMN] = normalize_target(prepared[TARGET_COLUMN])
    if prepared.isna().any().any():
        raise ValueError("The uploaded training data contains missing values.")
    return prepared


def load_base_source(
    train_path: Path,
    test_path: Path,
    approved: list[str],
    source_path: Path | None = None,
) -> pd.DataFrame:
    required = approved + [TARGET_COLUMN]
    if source_path is not None:
        source = pd.read_parquet(source_path, columns=required)
        source[TARGET_COLUMN] = normalize_target(source[TARGET_COLUMN])
        return source
    train = pd.read_parquet(train_path, columns=required)
    test = pd.read_parquet(test_path, columns=required)
    source = pd.concat([train, test], ignore_index=True)
    source[TARGET_COLUMN] = normalize_target(source[TARGET_COLUMN])
    return source


def feature_groups(frame: pd.DataFrame, approved: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(frame[approved], index=False)


def grouped_split(
    frame: pd.DataFrame,
    approved: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    groups = feature_groups(frame, approved)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = list(
        splitter.split(
            np.zeros(len(frame), dtype=np.int8),
            frame[TARGET_COLUMN],
            groups,
        )
    )
    train_index, test_index = splits[TEST_FOLD]
    train = frame.iloc[train_index].reset_index(drop=True)
    test = frame.iloc[test_index].reset_index(drop=True)
    train_groups = set(groups.iloc[train_index].astype("uint64"))
    test_groups = set(groups.iloc[test_index].astype("uint64"))
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"{len(overlap)} feature groups cross train and test.")
    return train, test, {
        "method": "StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "test_fold": TEST_FOLD,
        "random_state": RANDOM_STATE,
        "grouping_rule": "pandas_hash_of_all_approved_features",
        "unique_groups": int(groups.nunique()),
        "cross_split_group_overlap": 0,
    }


def create_fold_assignments(
    train: pd.DataFrame,
    approved: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = feature_groups(train, approved)
    target = train[TARGET_COLUMN]
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold = np.full(len(train), -1, dtype=np.int8)
    for fold_index, (_, validation_index) in enumerate(
        splitter.split(np.zeros(len(train), dtype=np.int8), target, groups)
    ):
        fold[validation_index] = fold_index
    if (fold < 0).any():
        raise AssertionError("At least one row was not assigned to a fold.")
    max_folds = (
        pd.DataFrame({"group": groups, "fold": fold})
        .groupby("group", sort=False)["fold"]
        .nunique()
        .max()
    )
    if int(max_folds) != 1:
        raise AssertionError("Identical approved feature vectors cross CV folds.")
    assignment = pd.DataFrame(
        {"row_index": np.arange(len(train), dtype=np.int64), "fold": fold}
    )
    summary = (
        pd.DataFrame({"fold": fold, "target": target.to_numpy()})
        .groupby("fold", sort=True)["target"]
        .agg(rows="size", fraud_cases="sum", fraud_rate="mean")
        .reset_index()
    )
    if len(summary) != N_SPLITS or (summary["fraud_cases"] == 0).any():
        raise ValueError("The generated folds do not contain enough fraud cases.")
    return assignment, summary


def categorical_columns(frame: pd.DataFrame, approved: list[str]) -> list[str]:
    return frame[approved].select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()


def prepare_categories(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
) -> dict[str, list[Any]]:
    category_map: dict[str, list[Any]] = {}
    for column in categorical:
        train[column] = train[column].astype("category")
        categories = [_json_safe(value) for value in train[column].cat.categories]
        category_map[column] = categories
        test[column] = pd.Categorical(test[column], categories=train[column].cat.categories)
    return category_map


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "gmean": float(np.sqrt(sensitivity * specificity)),
        "false_positive_rate": float(1.0 - specificity),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def global_shap_importance(
    model: Any,
    features: pd.DataFrame,
    probabilities: np.ndarray,
    approved: list[str],
    *,
    top_rows: int = 500,
) -> pd.DataFrame:
    import xgboost as xgb

    top_index = np.argsort(probabilities)[::-1][: min(top_rows, len(features))]
    top = features.iloc[top_index]
    matrix = xgb.DMatrix(top, enable_categorical=True)
    contributions = np.asarray(
        model.get_booster().predict(matrix, pred_contribs=True),
        dtype=np.float64,
    )[:, :-1]
    return (
        pd.DataFrame(
            {
                "feature": approved,
                "mean_abs_shap": np.mean(np.abs(contributions), axis=0),
                "positive_share": np.mean(contributions > 0, axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def run_grouped_retraining(
    *,
    run_id: str,
    run_dir: Path,
    upload_path: Path,
    historical_train_path: Path,
    historical_test_path: Path,
    approved_features_path: Path,
    database_path: Path,
    historical_source_path: Path | None = None,
    active_artifact_dir: Path | None = None,
    max_estimators: int = 2000,
    progress: Callable[[str, int, str], None] | None = None,
) -> dict[str, Any]:
    import xgboost as xgb

    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    store = ModelUpdateStore(database_path)

    def update(stage: str, percent: int, message: str) -> None:
        payload = {
            "run_id": run_id,
            "stage": stage,
            "percent": int(percent),
            "message": message,
            "updated_at": utc_now(),
        }
        atomic_json(status_path, payload)
        if progress:
            progress(stage, percent, message)

    try:
        update("Validating data", 5, "Checking the 148-column labelled file")
        approved = load_approved_features(approved_features_path)
        upload = pd.read_parquet(upload_path)
        new_data = prepare_uploaded_source(upload, approved)

        update("Creating snapshot", 15, "Combining approved history and new labels")
        history = load_base_source(
            historical_train_path,
            historical_test_path,
            approved,
            source_path=historical_source_path,
        )
        source = pd.concat([history, new_data], ignore_index=True)
        snapshot_dir = run_dir / "snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        source_path = snapshot_dir / "source_model_data.parquet"
        source.to_parquet(source_path, compression="snappy", index=False)

        update("Creating snapshot", 25, "Building the grouped train and test split")
        train, test, split_context = grouped_split(source, approved)
        train_path = snapshot_dir / "train_model_dataset.parquet"
        test_path = snapshot_dir / "test_model_dataset.parquet"
        train.to_parquet(train_path, compression="snappy", index=False)
        test.to_parquet(test_path, compression="snappy", index=False)

        update("Creating folds", 35, "Assigning grouped five-fold validation")
        folds, fold_summary = create_fold_assignments(train, approved)
        folds_path = snapshot_dir / "fold_assignments.parquet"
        folds.to_parquet(folds_path, compression="snappy", index=False)
        fold_summary.to_csv(snapshot_dir / "fold_summary.csv", index=False)

        update("Training", 45, "Training the grouped XGBoost candidate")
        target_train = train[TARGET_COLUMN].copy()
        target_test = test[TARGET_COLUMN].copy()
        train_features = train[approved].copy()
        test_features = test[approved].copy()
        categorical = categorical_columns(train, approved)
        category_map = prepare_categories(train_features, test_features, categorical)
        validation_mask = folds["fold"].to_numpy() == VALIDATION_FOLD
        model = xgb.XGBClassifier(
            n_estimators=max_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            early_stopping_rounds=50 if max_estimators > 50 else None,
            enable_categorical=True,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(
            train_features.loc[~validation_mask],
            target_train.loc[~validation_mask],
            eval_set=[
                (
                    train_features.loc[validation_mask],
                    target_train.loc[validation_mask],
                )
            ],
            verbose=False,
        )
        candidate_dir = run_dir / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        model_path = candidate_dir / "xgboost_grouped.json"
        model.save_model(model_path)
        atomic_json(
            candidate_dir / "category_map.json",
            {"categories": category_map},
        )

        update("Evaluating", 75, "Scoring the grouped holdout and calculating q-star")
        probabilities = model.predict_proba(test_features)[:, 1]
        q_star = float(target_train.mean())
        metrics = evaluate_predictions(target_test.to_numpy(), probabilities, q_star)
        pd.DataFrame(
            {"y_true": target_test.to_numpy(), "y_proba": probabilities}
        ).to_parquet(
            candidate_dir / "xgboost_grouped_test_predictions.parquet",
            index=False,
        )

        incumbent_evaluation: dict[str, Any] = {}
        if active_artifact_dir is not None:
            try:
                from .service import FraudScoringBackend

                incumbent = FraudScoringBackend(active_artifact_dir)
                incumbent_validation = incumbent.validate_claims(test_features)
                incumbent_scores = incumbent.score_claims(test_features)
                incumbent_evaluation = {
                    **evaluate_predictions(
                        target_test.to_numpy(),
                        incumbent_scores["fraud_probability"].to_numpy(),
                        incumbent.threshold,
                    ),
                    "model_version": incumbent.config.get("model_profile", {}).get(
                        "model_version", incumbent.config.get("model_track", "Active model")
                    ),
                    "unknown_category_features": sorted(
                        incumbent_validation.get("unknown_categories", {})
                    ),
                }
            except Exception as exc:
                incumbent_evaluation = {"error": str(exc)}

        update("Explaining", 88, "Calculating grouped-model feature influence")
        shap = global_shap_importance(
            model,
            test_features,
            probabilities,
            approved,
        )
        shap.to_parquet(candidate_dir / "global_shap.parquet", index=False)
        shap.to_csv(candidate_dir / "global_shap.csv", index=False)

        version = f"XGB-{run_id.removeprefix('RUN-')}"
        profile = {
            "model_version": version,
            "training_snapshot": run_id,
            "training_rows": int(len(train)),
            "training_fraud": int(target_train.sum()),
            "evaluation_rows": int(len(test)),
            "evaluation_fraud": int(target_test.sum()),
            "fraud_detected": metrics["tp"],
            "fraud_missed": metrics["fn"],
            "legitimate_flagged": metrics["fp"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "threshold": q_star,
            "best_iteration": int(getattr(model, "best_iteration", max_estimators - 1)),
            "new_rows": int(len(new_data)),
            "new_fraud": int(new_data[TARGET_COLUMN].sum()),
            "source_rows": int(len(source)),
            "source_snapshot_path": str(source_path.resolve()),
            "train_snapshot_path": str(train_path.resolve()),
            "test_snapshot_path": str(test_path.resolve()),
            "fold_assignments_path": str(folds_path.resolve()),
            "categorical_features": categorical,
        }
        atomic_json(
            candidate_dir / "metrics.json",
            {
                "profile": profile,
                **metrics,
                "incumbent_on_candidate_holdout": incumbent_evaluation,
            },
        )
        manifest = {
            "run_id": run_id,
            "created_at": utc_now(),
            "method": "grouped_xgboost_report_pipeline",
            "split": split_context,
            "leakage_fields_removed": list(LEAKAGE_FIELDS),
            "target": TARGET_COLUMN,
            "approved_features": len(approved),
            "input_upload": {
                "path": str(upload_path),
                "rows": int(len(upload)),
                "sha256": sha256_file(upload_path),
            },
            "historical_source": (
                str(historical_source_path)
                if historical_source_path is not None
                else "canonical_report_train_plus_test"
            ),
            "artifacts": {
                "model_sha256": sha256_file(model_path),
                "train_sha256": sha256_file(train_path),
                "test_sha256": sha256_file(test_path),
                "folds_sha256": sha256_file(folds_path),
            },
            "profile": profile,
            "incumbent_on_candidate_holdout": incumbent_evaluation,
        }
        atomic_json(run_dir / "manifest.json", manifest)
        store.record_event(
            run_id,
            "Candidate ready",
            (
                f"Grouped XGBoost candidate {version} completed: "
                f"ROC-AUC {metrics['roc_auc']:.4f}, PR-AUC {metrics['pr_auc']:.4f}."
            ),
            actor="Training service",
            metadata={**profile, "run_dir": str(run_dir)},
        )
        update("Candidate ready", 100, "Candidate training and evaluation completed")
        return manifest
    except Exception as exc:
        error_payload = {
            "run_id": run_id,
            "stage": "Failed",
            "percent": 100,
            "message": str(exc),
            "updated_at": utc_now(),
            "traceback": traceback.format_exc(),
        }
        atomic_json(status_path, error_payload)
        store.record_event(
            run_id,
            "Validation failed",
            f"Training failed: {exc}",
            actor="Training service",
            metadata={"run_dir": str(run_dir), "error": str(exc)},
        )
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, Path)):
        return str(value)
    if pd.isna(value):
        return None
    return value
