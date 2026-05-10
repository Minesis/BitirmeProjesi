"""
app/api/routes.py
------------------
FastAPI router definitions.

Endpoints:
  GET  /visits          – paginated raw shelf visits (alias: /events)
  GET  /active          – currently active visitors per camera
  GET  /statistics      – aggregated analytics grouped by shelf/camera
  GET  /health          – health check
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from app.database.repository import AnalyticsRepository

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  Dependency: shared repository instance
# ─────────────────────────────────────────────────────────────────────────────

_repo: AnalyticsRepository | None = None


def get_repo() -> AnalyticsRepository:
    if _repo is None:
        raise HTTPException(status_code=503, detail="Database not initialised")
    return _repo


def init_repo(db_path: str) -> None:
    global _repo
    _repo = AnalyticsRepository(db_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/visits")
@router.get("/events")  # backward compatibility
def get_visits(
    limit: int = Query(default=100, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    from_dt: Optional[str] = Query(default=None, description="ISO format: 2024-01-01T00:00:00"),
    to_dt: Optional[str] = Query(default=None),
    camera_id: Optional[str] = Query(default=None, description="Filter by camera id"),
    repo: AnalyticsRepository = Depends(get_repo),
):
    """Return paginated shelf visits."""
    try:
        from_parsed = datetime.fromisoformat(from_dt) if from_dt else None
        to_parsed = datetime.fromisoformat(to_dt) if to_dt else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date format: {exc}")

    visits = repo.get_all_visits(
        limit=limit,
        offset=offset,
        from_dt=from_parsed,
        to_dt=to_parsed,
        camera_id=camera_id,
    )
    return {"count": len(visits), "visits": visits, "events": visits}


@router.get("/active")
def get_active(
    camera_id: Optional[str] = Query(default=None, description="Filter by camera id"),
    repo: AnalyticsRepository = Depends(get_repo),
):
    """Return currently active shelf visits (exit_time is NULL)."""
    visits = repo.get_active_visits(camera_id=camera_id)

    # Add real-time duration for active visits.
    now = datetime.utcnow()
    for v in visits:
        if v.get("duration_seconds") is None and v.get("enter_time"):
            try:
                enter_dt = datetime.fromisoformat(str(v["enter_time"]))
                v["duration_seconds"] = (now - enter_dt).total_seconds()
            except Exception:
                pass
    active_counts: dict[str, int] = {}
    for v in visits:
        cid = str(v.get("camera_id", "unknown"))
        active_counts[cid] = active_counts.get(cid, 0) + 1
    return {"count": len(visits), "active_visits": visits, "active_counts": active_counts}


@router.get("/statistics")
def get_statistics(
    from_dt: Optional[str] = Query(default=None),
    to_dt: Optional[str] = Query(default=None),
    repo: AnalyticsRepository = Depends(get_repo),
):
    """Return aggregated shelf analytics statistics."""
    try:
        from_parsed = datetime.fromisoformat(from_dt) if from_dt else None
        to_parsed = datetime.fromisoformat(to_dt) if to_dt else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date format: {exc}")

    stats = repo.get_statistics(from_dt=from_parsed, to_dt=to_parsed)
    return stats
