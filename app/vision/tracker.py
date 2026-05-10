"""
app/vision/tracker.py
----------------------
Deep SORT tracker wrapper that maintains unique person IDs across frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from loguru import logger

import config


@dataclass
class TrackedPerson:
    track_id: int
    bbox: list[int]          # [x1, y1, x2, y2]
    confidence: float
    is_confirmed: bool
    center: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def face_crop_bbox(self) -> list[int]:
        """
        Rough head/face region: top part of the bounding box, narrowed to the center.

        This is intentionally conservative to:
          - increase face-detector hit-rate
          - reduce background in the crop
        """
        x1, y1, x2, y2 = self.bbox
        h = max(1, y2 - y1)
        w = max(1, x2 - x1)

        face_y2 = y1 + int(h * 0.50)
        cx = x1 + (w // 2)
        half_w = int(w * 0.35)
        fx1 = max(x1, cx - half_w)
        fx2 = min(x2, cx + half_w)

        return [int(fx1), int(y1), int(fx2), int(face_y2)]


class PersonTracker:
    """
    Wraps Deep SORT to produce stable per-person IDs.

    Usage:
        tracker = PersonTracker()
        tracked = tracker.update(detections_for_tracker, frame)
    """

    def __init__(self) -> None:
        cfg = config.get("tracking", default={})
        self._tracker = DeepSort(
            max_age=cfg.get("max_age", 30),
            n_init=cfg.get("n_init", 3),
            max_cosine_distance=cfg.get("max_cosine_distance", 0.4),
            nn_budget=cfg.get("nn_budget", 100),
            override_track_class=None,
            embedder="mobilenet",
            half=False,
            bgr=True,
            embedder_gpu=False,
        )
        logger.info("PersonTracker (Deep SORT) initialised")

    def update(
        self,
        detections: list[tuple],   # ([x,y,w,h], conf, "person")
        frame: np.ndarray,
    ) -> list[TrackedPerson]:
        """
        Update tracker and return confirmed tracks.

        Args:
            detections : output of PersonDetector.detect_for_tracker()
            frame      : current BGR frame (needed by Deep SORT for embedding)

        Returns:
            List of TrackedPerson objects.
        """
        tracks = self._tracker.update_tracks(detections, frame=frame)
        results: list[TrackedPerson] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = (int(v) for v in ltrb)
            results.append(
                TrackedPerson(
                    track_id=int(track.track_id),
                    bbox=[x1, y1, x2, y2],
                    confidence=track.det_conf if track.det_conf is not None else 0.0,
                    is_confirmed=True,
                )
            )
        return results
