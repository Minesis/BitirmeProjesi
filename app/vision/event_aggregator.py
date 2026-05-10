"""
app/vision/event_aggregator.py
-------------------------------
Centralized DB write queue for multi-camera execution.

SQLite performs best when writes are serialized. Each camera worker submits
visit open/close/update tasks to this aggregator.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable
import threading

from loguru import logger

from app.database.repository import AnalyticsRepository


@dataclass(frozen=True)
class _DBTask:
    fn: Callable[[AnalyticsRepository], Any]
    result_queue: Queue | None = None


class EventAggregator:
    """Serializes DB writes behind a single worker thread."""

    def __init__(self, db_path: str) -> None:
        self._repo = AnalyticsRepository(db_path)
        self._queue: Queue[_DBTask | None] = Queue()
        self._thread = threading.Thread(target=self._run, name="event_aggregator", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._thread.start()
        self._started = True
        logger.info("EventAggregator started")

    def stop(self, timeout: float = 10.0) -> None:
        if not self._started:
            return
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        logger.info("EventAggregator stopped")

    def submit(self, fn: Callable[[AnalyticsRepository], Any]) -> None:
        """Fire-and-forget DB task."""
        self._queue.put(_DBTask(fn=fn, result_queue=None))

    def call(self, fn: Callable[[AnalyticsRepository], Any], timeout: float = 10.0) -> Any:
        """Submit a DB task and wait for its result."""
        res_q: Queue = Queue(maxsize=1)
        self._queue.put(_DBTask(fn=fn, result_queue=res_q))
        status, payload = res_q.get(timeout=timeout)
        if status == "err":
            raise payload
        return payload

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                try:
                    result = task.fn(self._repo)
                    if task.result_queue is not None:
                        task.result_queue.put(("ok", result))
                except Exception as exc:
                    logger.exception(f"DB task failed: {exc}")
                    if task.result_queue is not None:
                        task.result_queue.put(("err", exc))
            finally:
                self._queue.task_done()

