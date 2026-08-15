from __future__ import annotations

import argparse
from pathlib import Path

from prototype_phase12.backend.training_pipeline import run_grouped_retraining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a versioned grouped XGBoost retraining job."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--upload", type=Path, required=True)
    parser.add_argument("--historical-train", type=Path, required=True)
    parser.add_argument("--historical-test", type=Path, required=True)
    parser.add_argument("--historical-source", type=Path)
    parser.add_argument("--approved-features", type=Path, required=True)
    parser.add_argument("--active-artifacts", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--max-estimators", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_grouped_retraining(
        run_id=args.run_id,
        run_dir=args.run_dir,
        upload_path=args.upload,
        historical_train_path=args.historical_train,
        historical_test_path=args.historical_test,
        historical_source_path=args.historical_source,
        approved_features_path=args.approved_features,
        active_artifact_dir=args.active_artifacts,
        database_path=args.database,
        max_estimators=args.max_estimators,
    )


if __name__ == "__main__":
    main()
