"""
app/database/models.py
-----------------------
SQLAlchemy ORM models for the retail analytics database.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ShelfVisit(Base):
    """
    One row = one visitor's dwell session in a single camera/shelf.

    Each camera represents exactly one shelf. A visitor can create multiple
    rows over time if they leave and re-enter (new track).
    """

    __tablename__ = "shelf_visits"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Camera / shelf identity
    camera_id = Column(String(64), nullable=False, index=True)
    shelf_name = Column(String(128), nullable=False, index=True)

    # Visitor identity (unique only within a camera)
    visitor_id = Column(Integer, nullable=False, index=True)       # Deep SORT assigned ID

    # Demographics
    gender = Column(String(10), nullable=True)                     # "Male" | "Female" | "Unknown"
    age_group = Column(String(20), nullable=True)                  # configurable via config.yaml | "Unknown"

    # Timing
    enter_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)              # filled on exit

    def __repr__(self) -> str:
        return (
            f"<ShelfVisit(id={self.id}, camera={self.camera_id}, visitor={self.visitor_id}, "
            f"shelf={self.shelf_name}, gender={self.gender}, age={self.age_group})>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "shelf_name": self.shelf_name,
            "visitor_id": self.visitor_id,
            "gender": self.gender,
            "age_group": self.age_group,
            "enter_time": self.enter_time.isoformat() if self.enter_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "duration_seconds": self.duration_seconds,
        }
