"""
app/vision/detector.py
-----------------------
YOLOv8 person detector wrapper.

Returns bounding boxes in [x1, y1, x2, y2] format for class 0 (person).
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from ultralytics import YOLO

import config


class PersonDetector:
    """
    Wraps YOLOv8 for efficient human detection.

    Args:
        model_name : e.g. "yolov8n.pt" (auto-downloaded if absent)
        confidence : minimum detection confidence
        iou_threshold : NMS IoU threshold
        device     : "cpu" or "cuda"
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str | None = None,
        confidence: float | None = None,
        iou_threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        cfg_det = config.get("detection", default={})
        self._conf = confidence or cfg_det.get("confidence", 0.45)
        self._iou = iou_threshold or cfg_det.get("iou_threshold", 0.45)
        self._device = device or cfg_det.get("device", "cpu")

        model_name = model_name or cfg_det.get("model", "yolov8n.pt")
        self._model = YOLO(model_name)
        logger.info(f"PersonDetector: {model_name} on {self._device}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a BGR frame.

        Returns:
            List of dicts with keys:
                bbox   : [x1, y1, x2, y2]  (ints)
                conf   : float
                class_id : int (always 0)
        """
        results = self._model.predict(
            frame,
            conf=self._conf,
            iou=self._iou,
            classes=[self.PERSON_CLASS_ID],
            device=self._device,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append({"bbox": [x1, y1, x2, y2], "conf": conf, "class_id": 0})
        return detections

    def detect_for_tracker(self, frame: np.ndarray) -> list[tuple]:
        """
        Format detections for deep_sort_realtime.DeepSort.update_tracks().

        Returns list of ([x1,y1,w,h], conf, "person") tuples.
        """
        raw = self.detect(frame)
        out = []
        for d in raw:
            x1, y1, x2, y2 = d["bbox"]
            w, h = x2 - x1, y2 - y1
            out.append(([x1, y1, w, h], d["conf"], "person"))
        return out
