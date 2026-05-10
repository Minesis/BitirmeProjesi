"""
app/vision/inference_service.py
--------------------------------
Face-crop demographics inference service used by multi-camera workers.

This module purposefully keeps inference concerns (cropping, bounds checks,
model invocation) separate from tracking/dwell logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

import config
from app.model.inference import AgeGenderPredictor


@dataclass(frozen=True)
class Demographics:
    gender: str
    age_group: str
    gender_conf: float
    age_conf: float


class InferenceService:
    """Single-model inference wrapper with safe cropping utilities."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        min_face_size: int | None = None,
    ) -> None:
        model_path = model_path or config.get("age_gender.model_path", default="models/age_gender_model.pth")
        device = device or config.get("age_gender.device", default="cpu")
        self._min_face_size = int(
            min_face_size
            if min_face_size is not None
            else config.get("age_gender.min_face_size", default=28)
        )
        self._face_detector = str(config.get("age_gender.face_detector", default="none") or "none").lower().strip()
        self._face_margin = float(config.get("age_gender.face_margin", default=0.15))

        self._haar: cv2.CascadeClassifier | None = None
        if self._face_detector == "haar":
            self._haar = self._init_haar()
            if self._haar is None:
                logger.warning("Haar face detector failed to load; falling back to head crop.")

        self._predictor = AgeGenderPredictor(model_path, device=device)
        logger.info(f"InferenceService ready (device={device})")

    def infer_from_bbox(self, frame_bgr: np.ndarray, bbox_xyxy: list[int]) -> Demographics:
        """
        Infer demographics from a BGR frame and a head bbox in xyxy format.

        If enabled, a face detector runs inside the head region and refines the crop.
        Returns Unknown values if crop is invalid.
        """
        refined_bbox = self._refine_face_bbox(frame_bgr, bbox_xyxy)
        crop = self._safe_crop(frame_bgr, refined_bbox)
        if crop is None:
            return Demographics("Unknown", "Unknown", 0.0, 0.0)

        res = self._predictor.predict(crop)
        return Demographics(
            gender=res.get("gender", "Unknown"),
            age_group=res.get("age_group", "Unknown"),
            gender_conf=float(res.get("gender_conf", 0.0)),
            age_conf=float(res.get("age_conf", 0.0)),
        )

    def _init_haar(self) -> cv2.CascadeClassifier | None:
        try:
            cascade_path = str(cv2.data.haarcascades) + "haarcascade_frontalface_default.xml"
            clf = cv2.CascadeClassifier(cascade_path)
            if clf.empty():
                raise RuntimeError(f"Empty classifier: {cascade_path}")
            logger.info(f"Haar face detector loaded: {cascade_path}")
            return clf
        except Exception as exc:
            logger.warning(f"Failed to init haar face detector: {exc}")
            return None

    def _refine_face_bbox(self, frame_bgr: np.ndarray, bbox_xyxy: list[int]) -> list[int]:
        if self._haar is None:
            return bbox_xyxy
        if frame_bgr is None or frame_bgr.size == 0:
            return bbox_xyxy
        if not bbox_xyxy or len(bbox_xyxy) != 4:
            return bbox_xyxy

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return bbox_xyxy

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return bbox_xyxy

        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self._haar.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self._min_face_size, self._min_face_size),
            )
        except Exception:
            return bbox_xyxy

        if faces is None or len(faces) == 0:
            return bbox_xyxy

        # Pick the largest detected face (best heuristic for a single person track).
        fx, fy, fw, fh = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        fx1, fy1 = x1 + int(fx), y1 + int(fy)
        fx2, fy2 = fx1 + int(fw), fy1 + int(fh)

        # Expand crop a bit to include chin/forehead, clamp to frame.
        mx = int(fw * self._face_margin)
        my = int(fh * self._face_margin)
        fx1 = max(0, fx1 - mx)
        fy1 = max(0, fy1 - my)
        fx2 = min(w, fx2 + mx)
        fy2 = min(h, fy2 + my)
        if fx2 <= fx1 or fy2 <= fy1:
            return bbox_xyxy
        return [int(fx1), int(fy1), int(fx2), int(fy2)]

    def _safe_crop(self, frame_bgr: np.ndarray, bbox_xyxy: list[int]) -> np.ndarray | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        if not bbox_xyxy or len(bbox_xyxy) != 4:
            return None

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if crop.shape[0] < self._min_face_size or crop.shape[1] < self._min_face_size:
            return None
        return crop
