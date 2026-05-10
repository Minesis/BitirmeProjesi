"""
app/vision/stream_manager.py
-----------------------------
In-memory latest-frame store for Streamlit live previews.

Each camera worker pushes its latest annotated frame here.
The Streamlit UI pulls JPEG bytes from here to render thumbnails/live views.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class StreamFrame:
    jpeg_bytes: bytes
    width: int
    height: int
    timestamp: float


class StreamManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, StreamFrame] = {}

    def update_frame(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        *,
        max_width: int | None = None,
        jpeg_quality: int = 80,
    ) -> None:
        if frame_bgr is None or frame_bgr.size == 0:
            return

        frame = frame_bgr
        h, w = frame.shape[:2]
        if max_width and w > max_width:
            scale = float(max_width) / float(w)
            new_w = int(w * scale)
            new_h = max(1, int(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            return

        payload = StreamFrame(
            jpeg_bytes=bytes(buf.tobytes()),
            width=int(w),
            height=int(h),
            timestamp=time.time(),
        )
        with self._lock:
            self._frames[camera_id] = payload

    def get_latest_frame(self, camera_id: str) -> StreamFrame | None:
        with self._lock:
            return self._frames.get(camera_id)

    def clear(self, camera_id: str) -> None:
        with self._lock:
            self._frames.pop(camera_id, None)

