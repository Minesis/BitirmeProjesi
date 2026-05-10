"""
app/vision/analytics_engine.py
--------------------------------
Per-camera analytics core for the multi-camera shelf analytics system.

Key idea:
  - Each camera represents exactly one shelf (no polygon/zones).
  - "Visibility inside the camera" is treated as customer interacting with that shelf.

Responsibilities:
  - YOLOv8 person detection
  - Deep SORT tracking (stable visitor_id per camera)
  - Age/gender inference (best-effort)
  - Dwell time tracking
  - DB persistence via EventAggregator (single writer)
  - Produces annotated frames for UI preview
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import cv2
import numpy as np
from loguru import logger

import config
from app.vision.detector import PersonDetector
from app.vision.event_aggregator import EventAggregator
from app.vision.inference_service import InferenceService
from app.vision.tracker import PersonTracker, TrackedPerson


Mode = Literal["analytics", "test"]


@dataclass
class TrackState:
    visitor_id: int
    enter_time: datetime
    visit_id: int | None = None

    gender: str = "Unknown"
    age_group: str = "Unknown"
    gender_conf: float = 0.0
    age_conf: float = 0.0
    gender_locked: bool = False
    age_locked: bool = False

    gender_vote_sum: dict[str, float] = field(default_factory=dict)
    gender_vote_count: dict[str, int] = field(default_factory=dict)
    age_vote_sum: dict[str, float] = field(default_factory=dict)
    age_vote_count: dict[str, int] = field(default_factory=dict)
    frame_counter: int = 0


class AnalyticsEngine:
    def __init__(
        self,
        *,
        camera_id: str,
        shelf_name: str,
        aggregator: EventAggregator,
        mode: Mode = "analytics",
    ) -> None:
        self.camera_id = camera_id
        self.shelf_name = shelf_name
        self._aggregator = aggregator
        self._mode: Mode = mode

        self._detector = PersonDetector()
        self._tracker = PersonTracker()
        self._inference = InferenceService()

        self._states: dict[int, TrackState] = {}
        self._active_ids: set[int] = set()

        self._demog_interval = int(config.get("age_gender.infer_interval", default=15))
        self._demog_lock_threshold = float(config.get("age_gender.lock_threshold", default=0.80))
        self._demog_min_votes = int(config.get("age_gender.min_votes", default=3))
        self._vote_min_conf = float(config.get("age_gender.vote_min_conf", default=0.55))

    # ------------------------------------------------------------------ #
    #  Main API
    # ------------------------------------------------------------------ #

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Process a BGR frame.

        Returns:
            annotated_frame_bgr, active_visitors_count
        """
        annotated = frame_bgr.copy()

        detections = self._detector.detect_for_tracker(frame_bgr)
        tracks = self._tracker.update(detections, frame_bgr)

        current_ids = {t.track_id for t in tracks}
        lost_ids = self._active_ids - current_ids
        for lost_id in lost_ids:
            self._close_visit(lost_id)
            self._states.pop(lost_id, None)
        self._active_ids = current_ids

        now = datetime.utcnow()

        for person in tracks:
            state = self._get_or_create_state(person.track_id, now)
            state.frame_counter += 1

            self._maybe_update_demographics(frame_bgr, person, state)
            self._draw_person(annotated, person, state, now)

        # Debug overlay
        cv2.putText(
            annotated,
            f"{self.camera_id} | {self.shelf_name} | active: {len(self._states)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return annotated, len(self._states)

    def shutdown(self) -> None:
        """Close any active visits on engine shutdown (analytics mode only)."""
        for visitor_id in list(self._states.keys()):
            self._close_visit(visitor_id)
        self._states.clear()
        self._active_ids.clear()
        logger.info(f"{self.camera_id}: AnalyticsEngine shutdown complete")

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _get_or_create_state(self, visitor_id: int, now: datetime) -> TrackState:
        if visitor_id in self._states:
            return self._states[visitor_id]

        visit_id: int | None = None
        if self._mode == "analytics":
            visit_id = int(
                self._aggregator.call(
                    lambda repo: repo.open_visit(
                        camera_id=self.camera_id,
                        shelf_name=self.shelf_name,
                        visitor_id=visitor_id,
                        gender="Unknown",
                        age_group="Unknown",
                        enter_time=now,
                    ),
                    timeout=10.0,
                )
            )

        state = TrackState(visitor_id=visitor_id, enter_time=now, visit_id=visit_id)
        self._states[visitor_id] = state
        return state

    def _close_visit(self, visitor_id: int) -> None:
        state = self._states.get(visitor_id)
        if not state:
            return
        if self._mode != "analytics":
            return
        if state.visit_id is None:
            return
        self._aggregator.submit(lambda repo: repo.close_visit(int(state.visit_id)))

    def _maybe_update_demographics(self, frame: np.ndarray, person: TrackedPerson, state: TrackState) -> None:
        if state.gender_locked and state.age_locked:
            return
        if self._demog_interval > 1 and (state.frame_counter % self._demog_interval) != 0:
            return

        demo = self._inference.infer_from_bbox(frame, person.face_crop_bbox)

        prev_gender, prev_age = state.gender, state.age_group

        if not state.gender_locked and demo.gender != "Unknown" and demo.gender_conf >= self._vote_min_conf:
            state.gender_vote_sum[demo.gender] = state.gender_vote_sum.get(demo.gender, 0.0) + float(demo.gender_conf)
            state.gender_vote_count[demo.gender] = state.gender_vote_count.get(demo.gender, 0) + 1
            g_label, g_conf, g_label_votes = self._best_vote(state.gender_vote_sum, state.gender_vote_count)
            if g_label_votes > 0:
                state.gender, state.gender_conf = g_label, g_conf
            if g_label_votes >= self._demog_min_votes and state.gender_conf >= self._demog_lock_threshold:
                state.gender_locked = True

        if not state.age_locked and demo.age_group != "Unknown" and demo.age_conf >= self._vote_min_conf:
            state.age_vote_sum[demo.age_group] = state.age_vote_sum.get(demo.age_group, 0.0) + float(demo.age_conf)
            state.age_vote_count[demo.age_group] = state.age_vote_count.get(demo.age_group, 0) + 1
            a_label, a_conf, a_label_votes = self._best_vote(state.age_vote_sum, state.age_vote_count)
            if a_label_votes > 0:
                state.age_group, state.age_conf = a_label, a_conf
            if a_label_votes >= self._demog_min_votes and state.age_conf >= self._demog_lock_threshold:
                state.age_locked = True

        changed = (state.gender != prev_gender) or (state.age_group != prev_age)
        if changed and self._mode == "analytics" and state.visit_id is not None:
            self._aggregator.submit(lambda repo: repo.update_demographics(int(state.visit_id), state.gender, state.age_group))

    @staticmethod
    def _best_vote(vote_sum: dict[str, float], vote_count: dict[str, int]) -> tuple[str, float, int]:
        """
        Return (label, avg_conf, label_votes) for the best label.

        Label is chosen by highest accumulated confidence (sum of confs).
        avg_conf is the mean confidence for that label (0..1).
        """
        if not vote_count:
            return "Unknown", 0.0, 0

        best_label = "Unknown"
        best_sum = -1.0
        best_count = 0
        for label, c in vote_count.items():
            c = int(c)
            if c <= 0:
                continue
            s = float(vote_sum.get(label, 0.0))
            if s > best_sum:
                best_label, best_sum, best_count = label, s, c

        if best_label == "Unknown" or best_count <= 0:
            return "Unknown", 0.0, 0

        best_avg = float(best_sum) / float(best_count)
        return best_label, float(round(best_avg, 3)), int(best_count)

    @staticmethod
    def _draw_person(frame: np.ndarray, person: TrackedPerson, state: TrackState, now: datetime) -> None:
        x1, y1, x2, y2 = person.bbox
        color = (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        dwell_s = max(0.0, (now - state.enter_time).total_seconds())
        label = (
            f"ID:{person.track_id} "
            f"{state.gender[0] if state.gender != 'Unknown' else '?'} "
            f"{state.age_group if state.age_group != 'Unknown' else '?'} "
            f"{dwell_s:0.1f}s"
        )
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
