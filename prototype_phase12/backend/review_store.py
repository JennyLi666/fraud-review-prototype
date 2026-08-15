from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


REVIEW_STATUSES = ("Pending", "In review", "Confirmed fraud", "Legitimate")


class ReviewStore:
    """Small local review log for the prototype human-in-the-loop workflow."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    claim_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    reviewed_at_utc TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def get(self, claim_id: str) -> dict[str, str]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT claim_id, status, notes, reviewed_at_utc FROM reviews WHERE claim_id = ?",
                (str(claim_id),),
            ).fetchone()
        if row is None:
            return {
                "claim_id": str(claim_id),
                "status": "Pending",
                "notes": "",
                "reviewed_at_utc": "",
            }
        return dict(row)

    def list_all(self) -> pd.DataFrame:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT claim_id, status, notes, reviewed_at_utc FROM reviews"
            ).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["claim_id", "status", "notes", "reviewed_at_utc"]
            )
        return pd.DataFrame([dict(row) for row in rows])

    def save(self, claim_id: str, status: str, notes: str) -> None:
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Unsupported review status: {status}")
        reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO reviews (claim_id, status, notes, reviewed_at_utc)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(claim_id) DO UPDATE SET
                    status = excluded.status,
                    notes = excluded.notes,
                    reviewed_at_utc = excluded.reviewed_at_utc
                """,
                (str(claim_id), status, notes.strip(), reviewed_at),
            )
            connection.commit()


MODEL_UPDATE_STATUSES = (
    "Training requested",
    "Candidate ready",
    "Validation passed",
    "Validation failed",
    "Deployed",
    "Rejected",
)


class DataUploadStore:
    """Persistent record of operational data added after a model deployment."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS data_upload_events (
                    upload_id TEXT PRIMARY KEY,
                    batch_name TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    flagged_count INTEGER NOT NULL,
                    labelled_rows INTEGER NOT NULL DEFAULT 0,
                    fraud_labels INTEGER NOT NULL DEFAULT 0,
                    label_status TEXT NOT NULL,
                    scored_at_utc TEXT NOT NULL,
                    activated_at_utc TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'Active',
                    undone_at_utc TEXT NOT NULL DEFAULT ''
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(data_upload_events)"
                ).fetchall()
            }
            if "status" not in columns:
                connection.execute(
                    "ALTER TABLE data_upload_events "
                    "ADD COLUMN status TEXT NOT NULL DEFAULT 'Active'"
                )
            if "undone_at_utc" not in columns:
                connection.execute(
                    "ALTER TABLE data_upload_events "
                    "ADD COLUMN undone_at_utc TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_data_upload_activated "
                "ON data_upload_events (activated_at_utc)"
            )
            connection.commit()

    def record_activation(
        self,
        *,
        upload_id: str | None = None,
        batch_name: str,
        source_file: str,
        operation: str,
        row_count: int,
        flagged_count: int,
        labelled_rows: int,
        fraud_labels: int,
        label_status: str,
        scored_at_utc: str,
        model_version: str,
    ) -> str:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        upload_id = upload_id or uuid4().hex
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO data_upload_events (
                    upload_id, batch_name, source_file, operation, row_count,
                    flagged_count, labelled_rows, fraud_labels, label_status,
                    scored_at_utc, activated_at_utc, model_version, status,
                    undone_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    batch_name,
                    source_file,
                    operation,
                    int(row_count),
                    int(flagged_count),
                    int(labelled_rows),
                    int(fraud_labels),
                    label_status,
                    scored_at_utc,
                    timestamp,
                    model_version,
                    "Active",
                    "",
                ),
            )
            connection.commit()
        return upload_id

    def undo_latest(self, upload_id: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as connection:
            latest = connection.execute(
                """
                SELECT upload_id
                FROM data_upload_events
                WHERE status = 'Active'
                ORDER BY activated_at_utc DESC, rowid DESC
                LIMIT 1
                """
            ).fetchone()
            if latest is None or str(latest["upload_id"]) != str(upload_id):
                raise ValueError("Only the latest active queue update can be undone.")
            connection.execute(
                """
                UPDATE data_upload_events
                SET status = 'Undone', undone_at_utc = ?
                WHERE upload_id = ?
                """,
                (timestamp, str(upload_id)),
            )
            connection.commit()

    def reactivate(self, upload_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE data_upload_events
                SET status = 'Active', undone_at_utc = ''
                WHERE upload_id = ?
                """,
                (str(upload_id),),
            )
            connection.commit()

    def list_all(self) -> pd.DataFrame:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT upload_id, batch_name, source_file, operation, row_count,
                       flagged_count, labelled_rows, fraud_labels, label_status,
                       scored_at_utc, activated_at_utc, model_version, status,
                       undone_at_utc
                FROM data_upload_events
                ORDER BY activated_at_utc DESC, rowid DESC
                """
            ).fetchall()
        columns = [
            "upload_id",
            "batch_name",
            "source_file",
            "operation",
            "row_count",
            "flagged_count",
            "labelled_rows",
            "fraud_labels",
            "label_status",
            "scored_at_utc",
            "activated_at_utc",
            "model_version",
            "status",
            "undone_at_utc",
        ]
        return pd.DataFrame([dict(row) for row in rows], columns=columns)


class ModelUpdateStore:
    """Persistent audit log for manually governed model update runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS model_update_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(model_update_events)"
                ).fetchall()
            }
            if "metadata_json" not in columns:
                connection.execute(
                    "ALTER TABLE model_update_events "
                    "ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_model_update_run "
                "ON model_update_events (run_id, recorded_at_utc)"
            )
            connection.commit()

    def start_run(
        self,
        summary: str,
        actor: str = "Manual entry",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        run_id = f"RUN-{datetime.now(timezone.utc):%Y%m%d}-{uuid4().hex[:6].upper()}"
        self.record_event(run_id, "Training requested", summary, actor, metadata)
        return run_id

    def record_event(
        self,
        run_id: str,
        status: str,
        summary: str,
        actor: str = "Manual entry",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if status not in MODEL_UPDATE_STATUSES:
            raise ValueError(f"Unsupported model update status: {status}")
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO model_update_events (
                    event_id, run_id, status, summary, actor, recorded_at_utc,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    str(run_id),
                    status,
                    summary.strip(),
                    actor.strip() or "Manual entry",
                    timestamp,
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            connection.commit()

    def list_events(self) -> pd.DataFrame:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT run_id, status, summary, actor, recorded_at_utc,
                       metadata_json
                FROM model_update_events
                ORDER BY recorded_at_utc DESC, rowid DESC
                """
            ).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "run_id",
                    "status",
                    "summary",
                    "actor",
                    "recorded_at_utc",
                    "metadata_json",
                ]
            )
        return pd.DataFrame([dict(row) for row in rows])

    def latest_metadata(self, run_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT metadata_json
                FROM model_update_events
                WHERE run_id = ?
                ORDER BY recorded_at_utc DESC, rowid DESC
                """,
                (str(run_id),),
            ).fetchall()
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            if metadata:
                return metadata
        return {}

    def latest_deployed_metadata(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT run_id, metadata_json, recorded_at_utc
                FROM model_update_events
                WHERE status = 'Deployed'
                ORDER BY recorded_at_utc DESC, rowid DESC
                """
            ).fetchall()
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            if metadata.get("model_version"):
                return {
                    **metadata,
                    "run_id": row["run_id"],
                    "deployed_at": row["recorded_at_utc"],
                }
        return {}

    def open_runs(self) -> list[str]:
        events = self.list_events()
        if events.empty:
            return []
        terminal = {"Deployed", "Rejected"}
        latest = events.drop_duplicates("run_id", keep="first")
        return latest.loc[~latest["status"].isin(terminal), "run_id"].tolist()
