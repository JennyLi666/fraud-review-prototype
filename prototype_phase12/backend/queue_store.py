from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


class OperationalQueueStore:
    """Version the local active queue so the latest upload can be reversed."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.active_dir = self.root / "active"
        self.history_dir = self.root / "history"
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def load_active(self) -> dict[str, Any] | None:
        source_path = self.active_dir / "source.parquet"
        scores_path = self.active_dir / "scores.parquet"
        metadata_path = self.active_dir / "metadata.json"
        if not source_path.exists() or not scores_path.exists() or not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            **metadata,
            "source": pd.read_parquet(source_path),
            "scores": pd.read_parquet(scores_path),
        }

    def can_undo(self, upload_id: str) -> bool:
        snapshot_dir = self.history_dir / str(upload_id)
        return all(
            (snapshot_dir / name).exists()
            for name in ("source.parquet", "scores.parquet", "metadata.json")
        )

    def commit(
        self,
        upload_id: str,
        *,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> None:
        snapshot_dir = self.history_dir / str(upload_id)
        if snapshot_dir.exists():
            raise FileExistsError(f"Queue snapshot already exists: {upload_id}")
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        self._write_state(snapshot_dir, previous)
        self._replace_active(current)

    def undo(self, upload_id: str) -> dict[str, Any]:
        snapshot_dir = self.history_dir / str(upload_id)
        if not snapshot_dir.exists():
            raise FileNotFoundError("The previous queue snapshot is unavailable.")
        previous = self._read_state(snapshot_dir)
        if previous.get("is_demo"):
            self._clear_active()
        else:
            self._replace_active(previous)
        shutil.rmtree(snapshot_dir)
        return previous

    def _replace_active(self, state: dict[str, Any]) -> None:
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self._write_state(self.active_dir, state)

    def _clear_active(self) -> None:
        for name in ("source.parquet", "scores.parquet", "metadata.json"):
            path = self.active_dir / name
            if path.exists():
                path.unlink()

    @staticmethod
    def _write_state(directory: Path, state: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        source_path = directory / "source.parquet"
        scores_path = directory / "scores.parquet"
        metadata_path = directory / "metadata.json"
        source_temp = directory / "source.parquet.tmp"
        scores_temp = directory / "scores.parquet.tmp"
        metadata_temp = directory / "metadata.json.tmp"
        state["source"].to_parquet(source_temp, index=False, compression="snappy")
        state["scores"].to_parquet(scores_temp, index=False, compression="snappy")
        metadata = {
            "name": str(state["name"]),
            "timestamp": str(state["timestamp"]),
            "is_demo": bool(state.get("is_demo", False)),
            "upload_count": int(state.get("upload_count", 1)),
        }
        metadata_temp.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        source_temp.replace(source_path)
        scores_temp.replace(scores_path)
        metadata_temp.replace(metadata_path)

    @staticmethod
    def _read_state(directory: Path) -> dict[str, Any]:
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        return {
            **metadata,
            "source": pd.read_parquet(directory / "source.parquet"),
            "scores": pd.read_parquet(directory / "scores.parquet"),
        }
