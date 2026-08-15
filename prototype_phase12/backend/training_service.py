from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .review_store import ModelUpdateStore
from .training_pipeline import atomic_json, sha256_file


class TrainingRunService:
    """Create, monitor, approve, deploy, and roll back local training runs."""

    def __init__(
        self,
        *,
        project_root: Path,
        artifact_dir: Path,
        runtime_dir: Path,
        historical_train_path: Path,
        historical_test_path: Path,
        database_path: Path,
    ) -> None:
        self.project_root = Path(project_root)
        self.artifact_dir = Path(artifact_dir)
        self.runtime_dir = Path(runtime_dir)
        self.run_root = self.runtime_dir / "training_runs"
        self.deployment_root = self.runtime_dir / "deployments"
        self.historical_train_path = Path(historical_train_path)
        self.historical_test_path = Path(historical_test_path)
        self.database_path = Path(database_path)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.deployment_root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.run_root / str(run_id)

    def create_input(
        self,
        run_id: str,
        frame: pd.DataFrame,
        *,
        original_name: str,
    ) -> Path:
        run_dir = self.run_dir(run_id)
        if run_dir.exists():
            raise FileExistsError(f"Training directory already exists: {run_id}")
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=False)
        upload_path = input_dir / "labelled_source.parquet"
        frame.to_parquet(upload_path, compression="snappy", index=False)
        upload_sha = sha256_file(upload_path)
        events = ModelUpdateStore(self.database_path).list_events()
        deployed_runs = (
            set(events.loc[events["status"].eq("Deployed"), "run_id"].astype(str))
            if not events.empty
            else set()
        )
        for manifest_path in self.run_root.glob("*/input/upload_manifest.json"):
            if manifest_path.parent.parent.name not in deployed_runs:
                continue
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior.get("sha256") == upload_sha:
                raise ValueError(
                    "This labelled batch is already included in a deployed model."
                )
        atomic_json(
            input_dir / "upload_manifest.json",
            {
                "run_id": run_id,
                "original_name": original_name,
                "stored_name": upload_path.name,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "sha256": upload_sha,
                "stored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        return upload_path

    def launch(
        self,
        run_id: str,
        upload_path: Path,
        *,
        max_estimators: int = 2000,
    ) -> int:
        run_dir = self.run_dir(run_id)
        stdout_path = run_dir / "training.stdout.log"
        stderr_path = run_dir / "training.stderr.log"
        command = [
            sys.executable,
            "-m",
            "prototype_phase12.scripts.run_grouped_retraining",
            "--run-id",
            run_id,
            "--run-dir",
            str(run_dir),
            "--upload",
            str(upload_path),
            "--historical-train",
            str(self.historical_train_path),
            "--historical-test",
            str(self.historical_test_path),
            "--approved-features",
            str(self.artifact_dir / "approved_features.json"),
            "--active-artifacts",
            str(self.artifact_dir),
            "--database",
            str(self.database_path),
            "--max-estimators",
            str(int(max_estimators)),
        ]
        active_config = json.loads(
            (self.artifact_dir / "model_config.json").read_text(encoding="utf-8")
        )
        active_snapshot = active_config.get("model_profile", {}).get(
            "source_snapshot_path"
        )
        if active_snapshot and Path(active_snapshot).exists():
            command.extend(["--historical-source", str(active_snapshot)])
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
            )
        atomic_json(
            run_dir / "process.json",
            {
                "pid": process.pid,
                "command": command,
                "launched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        return int(process.pid)

    def status(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / "status.json"
        if not path.exists():
            return {
                "run_id": run_id,
                "stage": "Queued",
                "percent": 0,
                "message": "Waiting for the training worker",
            }
        return json.loads(path.read_text(encoding="utf-8"))

    def candidate(self, run_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        metrics_path = run_dir / "candidate" / "metrics.json"
        manifest_path = run_dir / "manifest.json"
        if not metrics_path.exists() or not manifest_path.exists():
            return {}
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {**metrics, "manifest": manifest}

    def deploy(self, run_id: str, store: ModelUpdateStore) -> dict[str, Any]:
        candidate = self.candidate(run_id)
        if not candidate:
            raise FileNotFoundError("Candidate artifacts are not ready.")
        events = store.list_events()
        run_events = events.loc[events["run_id"] == run_id]
        if run_events.empty or "Validation passed" not in set(run_events["status"]):
            raise PermissionError("The candidate must pass validation before deployment.")

        run_dir = self.run_dir(run_id)
        source_model = run_dir / "candidate" / "xgboost_grouped.json"
        source_categories = run_dir / "candidate" / "category_map.json"
        profile = dict(candidate["profile"])
        version = _safe_name(str(profile["model_version"]))
        model_dir = self.artifact_dir / "models"
        category_dir = self.artifact_dir / "category_maps"
        model_dir.mkdir(parents=True, exist_ok=True)
        category_dir.mkdir(parents=True, exist_ok=True)
        deployed_model = model_dir / f"{version}.json"
        deployed_categories = category_dir / f"{version}.json"
        if deployed_model.exists() or deployed_categories.exists():
            raise FileExistsError(f"Model version already exists: {version}")

        config_path = self.artifact_dir / "model_config.json"
        current_config = json.loads(config_path.read_text(encoding="utf-8"))
        deployment_id = datetime.now(timezone.utc).strftime("DEPLOY-%Y%m%d-%H%M%S")
        deployment_dir = self.deployment_root / deployment_id
        deployment_dir.mkdir(parents=True, exist_ok=False)
        (deployment_dir / "previous_model_config.json").write_text(
            json.dumps(current_config, indent=2) + "\n",
            encoding="utf-8",
        )

        shutil.copy2(source_model, deployed_model)
        shutil.copy2(source_categories, deployed_categories)
        new_config = dict(current_config)
        new_config.update(
            {
                "model_track": "xgboost_grouped_retrained",
                "model_type": "single fitted model",
                "model_paths": [deployed_model.relative_to(self.artifact_dir).as_posix()],
                "category_map_path": deployed_categories.relative_to(
                    self.artifact_dir
                ).as_posix(),
                "q_star_threshold": float(profile["threshold"]),
                "categorical_features": list(profile["categorical_features"]),
                "model_profile": {
                    key: value
                    for key, value in profile.items()
                    if key not in {"threshold", "categorical_features"}
                },
                "explanation_method": "XGBoost native TreeSHAP via pred_contribs=True",
            }
        )
        atomic_json(config_path, new_config)
        deployment_manifest = {
            "deployment_id": deployment_id,
            "run_id": run_id,
            "deployed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "previous_config": str(deployment_dir / "previous_model_config.json"),
            "new_model": str(deployed_model),
            "new_model_sha256": sha256_file(deployed_model),
            "new_category_map": str(deployed_categories),
            "profile": profile,
        }
        atomic_json(deployment_dir / "deployment.json", deployment_manifest)
        store.record_event(
            run_id,
            "Deployed",
            f"Activated model {profile['model_version']}.",
            actor="Deployment service",
            metadata={**profile, "deployment_id": deployment_id},
        )
        return deployment_manifest

    def latest_deployment(self) -> dict[str, Any]:
        manifests = sorted(
            self.deployment_root.glob("*/deployment.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not manifests:
            return {}
        return json.loads(manifests[0].read_text(encoding="utf-8"))

    def rollback_latest(self, store: ModelUpdateStore) -> dict[str, Any]:
        deployment = self.latest_deployment()
        if not deployment:
            raise FileNotFoundError("No deployment is available to roll back.")
        if deployment.get("rolled_back_at"):
            raise RuntimeError("The latest deployment has already been rolled back.")
        previous_path = Path(deployment["previous_config"])
        if not previous_path.exists():
            raise FileNotFoundError("The previous model configuration is missing.")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        atomic_json(self.artifact_dir / "model_config.json", previous)
        deployment["rolled_back_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        atomic_json(
            self.deployment_root
            / deployment["deployment_id"]
            / "deployment.json",
            deployment,
        )
        store.record_event(
            deployment["run_id"],
            "Rejected",
            f"Rolled back deployment {deployment['deployment_id']}.",
            actor="Deployment service",
            metadata={"rollback_of": deployment["deployment_id"]},
        )
        return deployment


def _safe_name(value: str) -> str:
    cleaned = "".join(character for character in value if character.isalnum() or character in "-_")
    return cleaned or "xgboost-grouped"
