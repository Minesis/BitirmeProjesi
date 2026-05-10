"""
app/vision/camera_worker.py
----------------------------
Background camera worker controlled by the Streamlit UI.

This worker:
  - opens a camera source (webcam / RTSP / video)
  - runs AnalyticsEngine on each frame
  - pushes annotated frames to StreamManager for UI previews
  - writes analytics events to DB (analytics mode)

It is designed to be managed by CameraManager (start/stop/restart per camera).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import sys
import threading
import time
from typing import Any, Literal

import cv2
from loguru import logger

import config
from app.vision.analytics_engine import AnalyticsEngine, Mode
from app.vision.event_aggregator import EventAggregator
from app.vision.stream_manager import StreamManager


Status = Literal["offline", "running", "stopped", "error"]


@dataclass
class CameraStatus:
    camera_id: str
    shelf_name: str
    source: str
    mode: Mode

    status: Status = "offline"
    fps: float = 0.0
    active_visitors: int = 0
    last_error: str | None = None
    last_frame_time: float | None = None
    last_status_time: float | None = None


class CameraWorker:
    def __init__(
        self,
        *,
        camera_id: str,
        shelf_name: str,
        source: int | str,
        aggregator: EventAggregator,
        stream_manager: StreamManager,
        stop_event: threading.Event | None = None,
        capture_overrides: dict[str, Any] | None = None,
        mode: Mode = "analytics",
        preview_max_width: int = 960,
    ) -> None:
        self.camera_id = camera_id
        self.shelf_name = shelf_name
        self.source = source
        self.mode: Mode = mode

        self._aggregator = aggregator
        self._stream_manager = stream_manager
        self._stop_event = stop_event or threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera_{camera_id}", daemon=True)

        self._engine = AnalyticsEngine(
            camera_id=camera_id,
            shelf_name=shelf_name,
            aggregator=aggregator,
            mode=mode,
        )

        self._cap_lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None

        cap_cfg = config.get("capture", default={}) or {}
        self._capture_width = int(cap_cfg.get("width", 1280))
        self._capture_height = int(cap_cfg.get("height", 720))
        self._set_capture_resolution = bool(cap_cfg.get("set_capture_resolution", not sys.platform.startswith("win")))
        self._reconnect_backoff_seconds = float(cap_cfg.get("reconnect_backoff_seconds", 5.0))
        self._failure_sleep_seconds = float(cap_cfg.get("failure_sleep_seconds", 1.0))

        if capture_overrides:
            self._capture_width = int(capture_overrides.get("width", self._capture_width))
            self._capture_height = int(capture_overrides.get("height", self._capture_height))
            if "set_capture_resolution" in capture_overrides:
                self._set_capture_resolution = bool(capture_overrides["set_capture_resolution"])

        self._preview_max_width = int(preview_max_width)

        self._status = CameraStatus(
            camera_id=camera_id,
            shelf_name=shelf_name,
            source=str(source),
            mode=mode,
            status="offline",
        )
        self._status_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._thread.start()
        self._set_status(status="running", last_error=None)
        logger.info(f"{self.camera_id}: worker started (mode={self.mode}, source={self.source})")

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        # Best-effort: release capture to help unblock cap.read() on some drivers.
        with self._cap_lock:
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass
        self._thread.join(timeout=timeout)
        self._engine.shutdown()
        self._set_status(status="stopped")

        # Ensure no dangling open visits remain if we stopped mid-stream.
        if self.mode == "analytics":
            self._aggregator.submit(lambda repo: repo.close_open_visits_for_camera(self.camera_id))

        logger.info(f"{self.camera_id}: worker stopped")

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def get_status(self) -> CameraStatus:
        with self._status_lock:
            return CameraStatus(**self._status.__dict__)

    # ------------------------------------------------------------------ #
    #  Worker loop
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        fps_ema = 0.0
        last_tick = time.time()

        while not self._stop_event.is_set():
            cap = self._open_capture()
            if cap is None:
                self._set_status(status="offline", last_error=self._status.last_error)
                time.sleep(self._reconnect_backoff_seconds)
                continue

            try:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._set_status(status="offline", last_error="No frames received")
                    time.sleep(self._failure_sleep_seconds)
                    continue

                needs_resize = (frame.shape[1] != self._capture_width) or (frame.shape[0] != self._capture_height)
                self._set_status(status="running", last_error=None)

                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        self._set_status(status="offline", last_error="Stream ended/disconnected")
                        break

                    if needs_resize:
                        frame = cv2.resize(frame, (self._capture_width, self._capture_height), interpolation=cv2.INTER_LINEAR)

                    annotated, active_visitors = self._engine.process_frame(frame)

                    # FPS estimate (EMA)
                    now_t = time.time()
                    dt = max(1e-6, now_t - last_tick)
                    inst_fps = 1.0 / dt
                    fps_ema = (0.9 * fps_ema) + (0.1 * inst_fps) if fps_ema > 0.0 else inst_fps
                    last_tick = now_t

                    self._stream_manager.update_frame(
                        self.camera_id,
                        annotated,
                        max_width=self._preview_max_width,
                        jpeg_quality=80,
                    )

                    self._set_status(
                        status="running",
                        fps=float(round(fps_ema, 2)),
                        active_visitors=int(active_visitors),
                        last_frame_time=now_t,
                    )

            except Exception as exc:
                logger.exception(f"{self.camera_id}: worker error: {exc}")
                self._set_status(status="error", last_error=str(exc))
                time.sleep(self._failure_sleep_seconds)
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
                with self._cap_lock:
                    if self._cap is cap:
                        self._cap = None

        self._set_status(status="stopped")

    # ------------------------------------------------------------------ #
    #  Capture helpers
    # ------------------------------------------------------------------ #

    def _open_capture(self) -> cv2.VideoCapture | None:
        source: int | str = self.source

        is_windows = sys.platform.startswith("win")
        if isinstance(source, str):
            s = source.strip()
            if s.isdigit():
                source = int(s)

        try:
            if is_windows and isinstance(source, int):
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(source)
            elif isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source)
        except Exception as exc:
            self._set_status(status="error", last_error=f"Failed to open capture: {exc}")
            return None

        if not cap.isOpened():
            self._set_status(status="offline", last_error=f"Cannot open source: {source}")
            try:
                cap.release()
            except Exception:
                pass
            return None

        if self._set_capture_resolution:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._capture_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._capture_height)
            except Exception:
                pass

        try:
            backend_name = cap.getBackendName()
        except Exception:
            backend_name = "unknown"

        self._set_status(status="running", last_error=None)
        logger.info(f"{self.camera_id}: capture opened (backend={backend_name}, source={source})")
        with self._cap_lock:
            self._cap = cap
        return cap

    # ------------------------------------------------------------------ #
    #  Status helpers
    # ------------------------------------------------------------------ #

    def _set_status(
        self,
        *,
        status: Status | None = None,
        fps: float | None = None,
        active_visitors: int | None = None,
        last_error: str | None = None,
        last_frame_time: float | None = None,
    ) -> None:
        now_t = time.time()
        with self._status_lock:
            if status is not None:
                self._status.status = status
            if fps is not None:
                self._status.fps = fps
            if active_visitors is not None:
                self._status.active_visitors = active_visitors
            if last_error is not None:
                self._status.last_error = last_error
            if last_frame_time is not None:
                self._status.last_frame_time = last_frame_time
            self._status.last_status_time = now_t
