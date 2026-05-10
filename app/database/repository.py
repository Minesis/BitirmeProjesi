"""
app/database/repository.py
---------------------------
Data access layer. All DB interactions go through this class.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine, event, func
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, ShelfVisit


class AnalyticsRepository:
    """
    Handles all database read/write operations for the retail analytics system.

    Usage:
        repo = AnalyticsRepository("data/analytics.db")
        visit_id = repo.open_visit(camera_id="cam_1", shelf_name="Snacks", visitor_id=42)
        repo.close_visit(visit_id)
    """

    def __init__(self, db_path: str = "data/analytics.db") -> None:
        db_url = f"sqlite:///{db_path}"
        self._engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False, "timeout": 30},
            echo=False,
        )

        # Improve read/write concurrency for multi-camera usage.
        # (SQLite still has a single-writer constraint, so writes should be serialized upstream.)
        def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA busy_timeout=30000;")
            finally:
                cursor.close()

        event.listen(self._engine, "connect", _set_sqlite_pragmas)

        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info(f"Database initialised at {db_path}")

    # ------------------------------------------------------------------ #
    #  Session context manager
    # ------------------------------------------------------------------ #

    @contextmanager
    def _session(self) -> Generator[Session, None, None]:
        session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    #  Write operations
    # ------------------------------------------------------------------ #

    def open_visit(
        self,
        camera_id: str,
        shelf_name: str,
        visitor_id: int,
        gender: str | None = None,
        age_group: str | None = None,
        enter_time: datetime | None = None,
    ) -> int:
        """Create a new shelf visit row. Returns the new row id."""
        with self._session() as s:
            row = ShelfVisit(
                camera_id=camera_id,
                shelf_name=shelf_name,
                visitor_id=visitor_id,
                gender=gender,
                age_group=age_group,
                enter_time=enter_time or datetime.utcnow(),
            )
            s.add(row)
            s.flush()
            return int(row.id)

    def close_visit(self, visit_id: int, exit_time: datetime | None = None) -> None:
        """Fill in exit_time and duration_seconds for an open visit."""
        with self._session() as s:
            row = s.get(ShelfVisit, visit_id)
            if row and row.exit_time is None:
                now = exit_time or datetime.utcnow()
                row.exit_time = now
                row.duration_seconds = (now - row.enter_time).total_seconds()

    def update_demographics(self, visit_id: int, gender: str, age_group: str) -> None:
        """Update demographics for a visit row (typically while it's active)."""
        with self._session() as s:
            row = s.get(ShelfVisit, visit_id)
            if row:
                row.gender = gender
                row.age_group = age_group

    def close_open_visits_for_camera(self, camera_id: str) -> int:
        """Close any visits still open for a camera (e.g. on worker shutdown)."""
        count = 0
        with self._session() as s:
            open_rows = (
                s.query(ShelfVisit)
                .filter(ShelfVisit.camera_id == camera_id)
                .filter(ShelfVisit.exit_time.is_(None))
                .all()
            )
            now = datetime.utcnow()
            for row in open_rows:
                row.exit_time = now
                row.duration_seconds = (now - row.enter_time).total_seconds()
                count += 1
        if count:
            logger.info(f"Closed {count} open visits for camera {camera_id}")
        return count

    # ------------------------------------------------------------------ #
    #  Read operations
    # ------------------------------------------------------------------ #

    def get_all_visits(
        self,
        limit: int = 1000,
        offset: int = 0,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        camera_id: str | None = None,
        shelf_name: str | None = None,
    ) -> list[dict]:
        with self._session() as s:
            q = s.query(ShelfVisit)
            if from_dt:
                q = q.filter(ShelfVisit.enter_time >= from_dt)
            if to_dt:
                q = q.filter(ShelfVisit.enter_time <= to_dt)
            if camera_id:
                q = q.filter(ShelfVisit.camera_id == camera_id)
            if shelf_name:
                q = q.filter(ShelfVisit.shelf_name == shelf_name)
            q = q.order_by(ShelfVisit.enter_time.desc()).limit(limit).offset(offset)
            return [e.to_dict() for e in q.all()]

    def get_active_visits(self, camera_id: str | None = None, shelf_name: str | None = None) -> list[dict]:
        """Return currently active visits (exit_time is NULL)."""
        with self._session() as s:
            q = s.query(ShelfVisit).filter(ShelfVisit.exit_time.is_(None))
            if camera_id:
                q = q.filter(ShelfVisit.camera_id == camera_id)
            if shelf_name:
                q = q.filter(ShelfVisit.shelf_name == shelf_name)
            q = q.order_by(ShelfVisit.enter_time.desc())
            return [r.to_dict() for r in q.all()]

    def get_statistics(
        self,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        camera_id: str | None = None,
        shelf_name: str | None = None,
    ) -> dict:
        """Aggregate shelf-based statistics for the dashboard."""
        with self._session() as s:
            q = s.query(ShelfVisit)
            if from_dt:
                q = q.filter(ShelfVisit.enter_time >= from_dt)
            if to_dt:
                q = q.filter(ShelfVisit.enter_time <= to_dt)
            if camera_id:
                q = q.filter(ShelfVisit.camera_id == camera_id)
            if shelf_name:
                q = q.filter(ShelfVisit.shelf_name == shelf_name)

            total_visits = q.count()
            if total_visits == 0:
                return self._empty_stats()

            unique_visitors = (
                q.with_entities(ShelfVisit.camera_id, ShelfVisit.visitor_id)
                .distinct()
                .count()
            )

            # Overall distributions
            gender_rows = (
                q.with_entities(ShelfVisit.gender, func.count(ShelfVisit.id))
                .group_by(ShelfVisit.gender)
                .all()
            )
            gender_distribution = {(g or "Unknown"): int(c) for g, c in gender_rows}

            age_rows = (
                q.with_entities(ShelfVisit.age_group, func.count(ShelfVisit.id))
                .group_by(ShelfVisit.age_group)
                .all()
            )
            age_group_distribution = {(a or "Unknown"): int(c) for a, c in age_rows}

            # Per-shelf counts and avg durations (completed only)
            counts_rows = (
                q.with_entities(
                    ShelfVisit.camera_id,
                    ShelfVisit.shelf_name,
                    func.count(ShelfVisit.id),
                )
                .group_by(ShelfVisit.camera_id, ShelfVisit.shelf_name)
                .all()
            )
            visit_counts = {(cid, shelf): int(c) for cid, shelf, c in counts_rows}

            avg_rows = (
                q.filter(ShelfVisit.duration_seconds.isnot(None))
                .with_entities(
                    ShelfVisit.camera_id,
                    ShelfVisit.shelf_name,
                    func.avg(ShelfVisit.duration_seconds),
                )
                .group_by(ShelfVisit.camera_id, ShelfVisit.shelf_name)
                .all()
            )
            avg_durations = {
                (cid, shelf): round(float(avg or 0.0), 2)
                for cid, shelf, avg in avg_rows
            }

            # Active visitors (no time filter; current state)
            active_q = (
                s.query(
                    ShelfVisit.camera_id,
                    ShelfVisit.shelf_name,
                    func.count(ShelfVisit.id),
                )
                .filter(ShelfVisit.exit_time.is_(None))
            )
            if camera_id:
                active_q = active_q.filter(ShelfVisit.camera_id == camera_id)
            if shelf_name:
                active_q = active_q.filter(ShelfVisit.shelf_name == shelf_name)
            active_rows = active_q.group_by(ShelfVisit.camera_id, ShelfVisit.shelf_name).all()
            active_counts = {(cid, shelf): int(c) for cid, shelf, c in active_rows}

            shelf_keys = sorted(set(visit_counts.keys()) | set(active_counts.keys()))
            shelf_stats = [
                {
                    "camera_id": cid,
                    "shelf_name": shelf,
                    "visit_count": visit_counts.get((cid, shelf), 0),
                    "avg_duration_seconds": avg_durations.get((cid, shelf), 0.0),
                    "active_visitors": active_counts.get((cid, shelf), 0),
                }
                for (cid, shelf) in shelf_keys
            ]

            overall_avg = (
                q.filter(ShelfVisit.duration_seconds.isnot(None))
                .with_entities(func.avg(ShelfVisit.duration_seconds))
                .scalar()
            )

            return {
                "total_visits": int(total_visits),
                "unique_visitors": int(unique_visitors),
                "gender_distribution": gender_distribution,
                "age_group_distribution": age_group_distribution,
                "shelf_stats": shelf_stats,
                "overall_avg_duration_seconds": round(float(overall_avg or 0.0), 2),
            }

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "total_visits": 0,
            "unique_visitors": 0,
            "gender_distribution": {},
            "age_group_distribution": {},
            "shelf_stats": [],
            "overall_avg_duration_seconds": 0.0,
        }
