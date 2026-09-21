"""
Ingest Service — Event normalisation and persistent storage.

Provides a SQLite-backed TrajectoryStore for persisting and querying
captured agent trajectories. The IngestService normalises raw events
into the common Trajectory schema before storage.

Usage:
    store = TrajectoryStore("behaviour_eval.db")
    store.save(trajectory)
    trajectories = store.list_trajectories(project_id="my-project")
    trajectory = store.get(trajectory_id)
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from behaviour_eval.trajectory import Trajectory

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("behaviour_eval_data.db")


class TrajectoryStore:
    """
    SQLite-backed storage for agent trajectories.

    Stores full trajectory data as JSON blobs with indexed columns
    for efficient filtering by project, model, outcome, and time range.

    Args:
        db_path: Path to the SQLite database file
    """

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialise the database schema."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    user_task TEXT NOT NULL,
                    outcome TEXT,
                    latency_ms REAL,
                    timestamp TEXT NOT NULL,
                    scenario_id TEXT,
                    has_reasoning INTEGER DEFAULT 0,
                    data JSON NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_project ON trajectories(project_id);
                CREATE INDEX IF NOT EXISTS idx_model ON trajectories(model_id);
                CREATE INDEX IF NOT EXISTS idx_outcome ON trajectories(outcome);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON trajectories(timestamp);
                CREATE INDEX IF NOT EXISTS idx_scenario ON trajectories(scenario_id);

                CREATE TABLE IF NOT EXISTS scores (
                    trajectory_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    judge_id TEXT DEFAULT 'system',
                    timestamp TEXT NOT NULL,
                    PRIMARY KEY (trajectory_id, metric_name, judge_id),
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id)
                );

                CREATE INDEX IF NOT EXISTS idx_scores_trajectory ON scores(trajectory_id);
                CREATE INDEX IF NOT EXISTS idx_scores_metric ON scores(metric_name);

                CREATE TABLE IF NOT EXISTS comparison_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    scenario_set TEXT NOT NULL,
                    models JSON NOT NULL,
                    results JSON,
                    recommendation TEXT,
                    status TEXT DEFAULT 'pending'
                );
            """)
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Database initialised at {self.db_path}")

    def save(self, trajectory: Trajectory) -> str:
        """
        Save a trajectory to the store.

        Args:
            trajectory: The Trajectory object to save

        Returns:
            The trajectory_id of the saved trajectory
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO trajectories
                (trajectory_id, project_id, model_id, user_task, outcome,
                 latency_ms, timestamp, scenario_id, has_reasoning, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trajectory.trajectory_id,
                    trajectory.project_id,
                    trajectory.model_id,
                    trajectory.user_task,
                    trajectory.outcome,
                    trajectory.latency_ms,
                    trajectory.timestamp,
                    trajectory.scenario_id,
                    1 if trajectory.has_reasoning else 0,
                    trajectory.to_json(),
                ),
            )
            # Save scores separately for efficient querying
            for metric_name, score in trajectory.scores.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO scores
                    (trajectory_id, metric_name, score, judge_id, timestamp)
                    VALUES (?, ?, ?, 'system', ?)
                    """,
                    (trajectory.trajectory_id, metric_name, score, trajectory.timestamp),
                )
            conn.commit()
            logger.debug(f"Saved trajectory {trajectory.trajectory_id}")
            return trajectory.trajectory_id
        finally:
            conn.close()

    def get(self, trajectory_id: str) -> Optional[Trajectory]:
        """
        Retrieve a single trajectory by ID.

        Args:
            trajectory_id: The trajectory ID

        Returns:
            The Trajectory object, or None if not found
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT data FROM trajectories WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
            if row:
                return Trajectory.from_json(row["data"])
            return None
        finally:
            conn.close()

    def list_trajectories(
        self,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        outcome: Optional[str] = None,
        scenario_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trajectory]:
        """
        List trajectories with optional filtering.

        Args:
            project_id: Filter by project
            model_id: Filter by model
            outcome: Filter by outcome
            scenario_id: Filter by scenario
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of Trajectory objects
        """
        conn = self._get_conn()
        try:
            conditions = []
            params = []

            if project_id:
                conditions.append("project_id = ?")
                params.append(project_id)
            if model_id:
                conditions.append("model_id = ?")
                params.append(model_id)
            if outcome:
                conditions.append("outcome = ?")
                params.append(outcome)
            if scenario_id:
                conditions.append("scenario_id = ?")
                params.append(scenario_id)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"""
                SELECT data FROM trajectories
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [Trajectory.from_json(row["data"]) for row in rows]
        finally:
            conn.close()

    def get_stats(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregate statistics for a project.

        Returns:
            Dictionary with counts, averages, and breakdowns
        """
        conn = self._get_conn()
        try:
            where = "WHERE project_id = ?" if project_id else ""
            params = [project_id] if project_id else []

            # Overall counts
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN outcome = 'success' THEN 1 END) as success_count,
                    COUNT(CASE WHEN outcome = 'failure' THEN 1 END) as failure_count,
                    COUNT(CASE WHEN outcome = 'error' THEN 1 END) as error_count,
                    COUNT(CASE WHEN outcome = 'partial' THEN 1 END) as partial_count,
                    AVG(latency_ms) as avg_latency,
                    COUNT(CASE WHEN has_reasoning = 1 THEN 1 END) as with_reasoning
                FROM trajectories {where}
                """,
                params,
            ).fetchone()

            total = row["total"]

            # Per-model breakdown
            model_rows = conn.execute(
                f"""
                SELECT model_id,
                    COUNT(*) as total,
                    COUNT(CASE WHEN outcome = 'success' THEN 1 END) as success_count,
                    AVG(latency_ms) as avg_latency
                FROM trajectories {where}
                GROUP BY model_id
                """,
                params,
            ).fetchall()

            # Average scores per metric
            score_rows = conn.execute(
                f"""
                SELECT s.metric_name, AVG(s.score) as avg_score, COUNT(*) as count
                FROM scores s
                JOIN trajectories t ON s.trajectory_id = t.trajectory_id
                {where.replace('project_id', 't.project_id')}
                GROUP BY s.metric_name
                """,
                params,
            ).fetchall()

            # Trend data (last 30 entries grouped by day)
            trend_rows = conn.execute(
                f"""
                SELECT
                    DATE(timestamp) as day,
                    COUNT(*) as total,
                    COUNT(CASE WHEN outcome = 'success' THEN 1 END) as successes,
                    AVG(latency_ms) as avg_latency
                FROM trajectories {where}
                GROUP BY DATE(timestamp)
                ORDER BY day DESC
                LIMIT 30
                """,
                params,
            ).fetchall()

            return {
                "total_trajectories": total,
                "success_count": row["success_count"],
                "failure_count": row["failure_count"],
                "error_count": row["error_count"],
                "partial_count": row["partial_count"],
                "success_rate": row["success_count"] / total if total > 0 else 0,
                "avg_latency_ms": row["avg_latency"],
                "with_reasoning_pct": row["with_reasoning"] / total * 100 if total > 0 else 0,
                "models": [
                    {
                        "model_id": r["model_id"],
                        "total": r["total"],
                        "success_count": r["success_count"],
                        "success_rate": r["success_count"] / r["total"] if r["total"] > 0 else 0,
                        "avg_latency_ms": r["avg_latency"],
                    }
                    for r in model_rows
                ],
                "avg_scores": {
                    r["metric_name"]: round(r["avg_score"], 4) for r in score_rows
                },
                "trends": [
                    {
                        "day": r["day"],
                        "total": r["total"],
                        "successes": r["successes"],
                        "success_rate": r["successes"] / r["total"] if r["total"] > 0 else 0,
                        "avg_latency": r["avg_latency"],
                    }
                    for r in trend_rows
                ],
            }
        finally:
            conn.close()

    def save_scores(self, trajectory_id: str, scores: Dict[str, float], judge_id: str = "system"):
        """Save evaluation scores for a trajectory."""
        conn = self._get_conn()
        try:
            from datetime import datetime
            ts = datetime.utcnow().isoformat()
            for metric_name, score in scores.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO scores
                    (trajectory_id, metric_name, score, judge_id, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (trajectory_id, metric_name, score, judge_id, ts),
                )
            # Also update the trajectory data blob
            row = conn.execute(
                "SELECT data FROM trajectories WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
            if row:
                traj_data = json.loads(row["data"])
                traj_data.setdefault("scores", {}).update(scores)
                conn.execute(
                    "UPDATE trajectories SET data = ? WHERE trajectory_id = ?",
                    (json.dumps(traj_data), trajectory_id),
                )
            conn.commit()
        finally:
            conn.close()

    def save_comparison_run(self, run_id: str, scenario_set: str, models: List[str],
                           results: Optional[Dict] = None, recommendation: str = "",
                           status: str = "pending"):
        """Save an offline comparison run."""
        conn = self._get_conn()
        try:
            from datetime import datetime
            conn.execute(
                """
                INSERT OR REPLACE INTO comparison_runs
                (run_id, timestamp, scenario_set, models, results, recommendation, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.utcnow().isoformat(),
                    scenario_set,
                    json.dumps(models),
                    json.dumps(results) if results else None,
                    recommendation,
                    status,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_comparison_runs(self) -> List[Dict[str, Any]]:
        """Get all comparison runs."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM comparison_runs ORDER BY timestamp DESC"
            ).fetchall()
            return [
                {
                    "run_id": r["run_id"],
                    "timestamp": r["timestamp"],
                    "scenario_set": r["scenario_set"],
                    "models": json.loads(r["models"]),
                    "results": json.loads(r["results"]) if r["results"] else None,
                    "recommendation": r["recommendation"],
                    "status": r["status"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def count(self, project_id: Optional[str] = None) -> int:
        """Count total trajectories."""
        conn = self._get_conn()
        try:
            if project_id:
                row = conn.execute(
                    "SELECT COUNT(*) as c FROM trajectories WHERE project_id = ?",
                    (project_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as c FROM trajectories").fetchone()
            return row["c"]
        finally:
            conn.close()

    def clear(self):
        """Clear all data (for testing)."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                DELETE FROM scores;
                DELETE FROM trajectories;
                DELETE FROM comparison_runs;
            """)
            conn.commit()
        finally:
            conn.close()


class IngestService:
    """
    Service that receives raw agent events, normalises them into
    the Trajectory schema, and stores them.

    This is the entry point for the SDK to push data to the platform.
    """

    def __init__(self, store: TrajectoryStore):
        self.store = store

    def ingest(self, raw_event: Dict[str, Any]) -> str:
        """
        Ingest a raw event and normalise it into a Trajectory.

        Args:
            raw_event: Dictionary with raw event data

        Returns:
            The trajectory_id of the ingested trajectory
        """
        trajectory = Trajectory.from_dict(raw_event)
        return self.store.save(trajectory)

    def ingest_batch(self, raw_events: List[Dict[str, Any]]) -> List[str]:
        """
        Ingest a batch of raw events.

        Args:
            raw_events: List of raw event dictionaries

        Returns:
            List of trajectory_ids
        """
        ids = []
        for event in raw_events:
            tid = self.ingest(event)
            ids.append(tid)
        logger.info(f"Ingested batch of {len(ids)} trajectories")
        return ids
