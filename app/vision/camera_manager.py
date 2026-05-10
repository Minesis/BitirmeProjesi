"""
app/vision/camera_manager.py
-----------------------------
Multi-camera orchestration layer (UI-controlled).

Loads camera configurations from YAML and allows starting/stopping/restarting
individual camera workers from the Streamlit UI.

All workers share:
  - EventAggregator (single DB writer thread)
  - StreamManager (latest annotated frames for UI previews)
"""

from __future__ import annotations

import threading
import time
from typing import Any

from loguru import logger

import config
from app.vision.analytics_engine import Mode
from app.vision.camera_worker import CameraWorker
from app.vision.event_aggregator import EventAggregator
from app.vision.stream_manager import StreamManager


class CameraManager:
    def __init__(
        self,
        *,
        show: bool = False,
        camera_configs: list[dict[str, Any]] | None = None,
        camera_id_filter: str | None = None,
    ) -> None:
        self._show = show
        self._stop_event = threading.Event()

        db_path = config.get("database.path", default="data/analytics.db")
        self._aggregator = EventAggregator(db_path)
        self._stream_manager = StreamManager()

        self._camera_cfgs: list[dict[str, Any]] = list(camera_configs or (config.get("cameras", default=[]) or []))
        if camera_id_filter:
            self._camera_cfgs = [c for c in self._camera_cfgs if str(c.get("id", "")).strip() == camera_id_filter]
        self._workers: dict[str, CameraWorker] = {}

        self._aggregator.start()

    def start_all(self, *, mode: str = "analytics") -> None:
        """Start all configured cameras (useful for non-UI runs)."""
        if not self._camera_cfgs:
            raise ValueError("No cameras configured. Add a 'cameras:' list to config/config.yaml.")
        for cam in self._camera_cfgs:
            cid = str(cam.get("id", "")).strip()
            if cid:
                self.start_camera(cid, mode=mode)

    def run_forever(self, poll_seconds: float = 2.0, restart_dead_workers: bool = True) -> None:
        """
        Blocks the main thread; useful for CLI runs.
        The UI typically does NOT call this.
        """
        self.start_all(mode="analytics")
        try:
            while True:
                if restart_dead_workers:
                    self.watchdog_restart_dead_workers()
                time.sleep(poll_seconds)
        finally:
            self.shutdown()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop all workers and the aggregator (best-effort)."""
        self.stop_all(timeout=timeout)
        self._aggregator.stop(timeout=timeout)
        logger.info("CameraManager shutdown complete")

    def get_stream_manager(self) -> StreamManager:
        return self._stream_manager

    # ------------------------------------------------------------------ #
    #  Config / discovery
    # ------------------------------------------------------------------ #

    def list_camera_configs(self) -> list[dict[str, Any]]:
        return list(self._camera_cfgs)

    def get_camera_config(self, camera_id: str) -> dict[str, Any] | None:
        cid = str(camera_id).strip()
        for c in self._camera_cfgs:
            if str(c.get("id", "")).strip() == cid:
                return c
        return None

    def reload_config(self) -> None:
        """Reload YAML config and update internal camera list."""
        try:
            # clear cached config
            try:
                from config.loader import load_config
                load_config.cache_clear()
            except Exception:
                pass

            self._camera_cfgs = list(config.get("cameras", default=[]) or [])
            logger.info(f"CameraManager reloaded config ({len(self._camera_cfgs)} cameras)")
        except Exception as exc:
            logger.warning(f"Config reload failed: {exc}")

    # ------------------------------------------------------------------ #
    #  Worker management
    # ------------------------------------------------------------------ #

    def get_worker(self, camera_id: str) -> CameraWorker | None:
        return self._workers.get(str(camera_id).strip())

    def get_status(self, camera_id: str) -> dict[str, Any]:
        worker = self.get_worker(camera_id)
        if worker is None:
            cfg = self.get_camera_config(camera_id) or {}
            return {
                "camera_id": str(camera_id),
                "shelf_name": str(cfg.get("shelf_name", "")),
                "source": str(cfg.get("source", "")),
                "status": "offline",
                "mode": "analytics",
                "fps": 0.0,
                "active_visitors": 0,
                "last_error": None,
            }
        st = worker.get_status()
        return {
            "camera_id": st.camera_id,
            "shelf_name": st.shelf_name,
            "source": st.source,
            "status": st.status,
            "mode": st.mode,
            "fps": st.fps,
            "active_visitors": st.active_visitors,
            "last_error": st.last_error,
            "last_frame_time": st.last_frame_time,
        }

    def start_camera(self, camera_id: str, *, mode: str = "analytics") -> None:
        cam_cfg = self.get_camera_config(camera_id)
        if not cam_cfg:
            raise ValueError(f"Unknown camera id: {camera_id}")

        existing = self.get_worker(camera_id)
        if existing and existing.is_alive():
            # If mode differs, restart in requested mode.
            if existing.mode != mode:
                self.restart_camera(camera_id, mode=mode)
            return

        self._start_worker(cam_cfg, mode=mode)

    def stop_camera(self, camera_id: str, *, timeout: float = 10.0) -> None:
        cid = str(camera_id).strip()
        worker = self._workers.get(cid)
        if not worker:
            return
        try:
            worker.stop(timeout=timeout)
        finally:
            self._workers.pop(cid, None)
            self._stream_manager.clear(cid)

    def restart_camera(self, camera_id: str, *, mode: str = "analytics") -> None:
        self.stop_camera(camera_id)
        time.sleep(0.2)
        self.start_camera(camera_id, mode=mode)

    def test_camera(self, camera_id: str, *, seconds: float = 10.0) -> None:
        """
        Start the camera in 'test' mode (no DB writes) for a short period.
        The UI can open the live view during this time.
        """
        self.start_camera(camera_id, mode="test")

        def _auto_stop() -> None:
            time.sleep(max(1.0, float(seconds)))
            # Only auto-stop if still in test mode.
            worker = self.get_worker(camera_id)
            if worker and worker.is_alive() and worker.mode == "test":
                self.stop_camera(camera_id)

        t = threading.Thread(target=_auto_stop, name=f"auto_stop_{camera_id}", daemon=True)
        t.start()

    def stop_all(self, timeout: float = 10.0) -> None:
        for cid in list(self._workers.keys()):
            try:
                self.stop_camera(cid, timeout=timeout)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _start_worker(self, cam_cfg: dict[str, Any], *, mode: str = "analytics") -> None:
        camera_id = str(cam_cfg.get("id", "")).strip()
        shelf_name = str(cam_cfg.get("shelf_name", "")).strip()
        source = cam_cfg.get("source", None)

        if not camera_id or not shelf_name or source is None:
            logger.warning(f"Skipping invalid camera config: {cam_cfg}")
            return

        capture_overrides = cam_cfg.get("capture", None)
        if capture_overrides is not None and not isinstance(capture_overrides, dict):
            logger.warning(f"{camera_id}: 'capture' overrides must be a dict. Ignoring.")
            capture_overrides = None

        safe_mode: Mode = "test" if str(mode).lower().strip() == "test" else "analytics"
        worker = CameraWorker(
            camera_id=camera_id,
            shelf_name=shelf_name,
            source=source,
            aggregator=self._aggregator,
            stream_manager=self._stream_manager,
            stop_event=None,
            capture_overrides=capture_overrides,
            mode=safe_mode,
        )
        worker.start()
        self._workers[camera_id] = worker

    def watchdog_restart_dead_workers(self) -> None:
        """Optional: restart workers that died unexpectedly."""
        for cam_cfg in self._camera_cfgs:
            cid = str(cam_cfg.get("id", "")).strip()
            if not cid:
                continue
            worker = self._workers.get(cid)
            if worker is None:
                continue
            if not worker.is_alive():
                logger.warning(f"{cid}: worker thread died; restarting")
                self._workers.pop(cid, None)
                self._start_worker(cam_cfg, mode="analytics")
