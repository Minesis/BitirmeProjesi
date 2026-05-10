"""
app/services/dashboard_service.py
----------------------------------
Dashboard data access helpers for Streamlit UI.

This service reads from the SQLite database via AnalyticsRepository and exposes
convenient methods for UI pages (stats, active visitors, recent visits).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.database.repository import AnalyticsRepository


@dataclass(frozen=True)
class TimeRange:
    from_dt: datetime | None = None
    to_dt: datetime | None = None


class DashboardService:
    def __init__(self, db_path: str) -> None:
        self._repo = AnalyticsRepository(db_path)

    def get_statistics(
        self,
        *,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        camera_id: str | None = None,
        shelf_name: str | None = None,
    ) -> dict[str, Any]:
        return self._repo.get_statistics(from_dt=from_dt, to_dt=to_dt, camera_id=camera_id, shelf_name=shelf_name)

    def get_active_visits(self, *, camera_id: str | None = None, shelf_name: str | None = None) -> list[dict[str, Any]]:
        return self._repo.get_active_visits(camera_id=camera_id, shelf_name=shelf_name)

    def get_recent_visits(
        self,
        *,
        limit: int = 200,
        offset: int = 0,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        camera_id: str | None = None,
        shelf_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._repo.get_all_visits(
            limit=limit,
            offset=offset,
            from_dt=from_dt,
            to_dt=to_dt,
            camera_id=camera_id,
            shelf_name=shelf_name,
        )

