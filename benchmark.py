"""Utilities for tracking inference performance."""

from __future__ import annotations

import collections
import logging
import time
from typing import Deque, Optional

LOGGER = logging.getLogger(__name__)


class FPSMeter:
    """Tracks instantaneous and rolling average FPS and logs periodically."""

    def __init__(self, window_size: int = 60, log_interval: float = 5.0) -> None:
        self.window: Deque[float] = collections.deque(maxlen=max(2, window_size))
        self.last_stamp: Optional[float] = None
        self.log_interval = max(0.5, log_interval)
        self._last_log_ts = time.time()

    def tick(self) -> float:
        now = time.perf_counter()
        fps = 0.0
        if self.last_stamp is not None:
            delta = now - self.last_stamp
            if delta > 0:
                fps = 1.0 / delta
                self.window.append(fps)
        self.last_stamp = now
        self._maybe_log()
        return fps

    def average(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    def _maybe_log(self) -> None:
        current_time = time.time()
        if current_time - self._last_log_ts >= self.log_interval and self.window:
            LOGGER.info("FPS current=%.2f avg=%.2f", self.window[-1], self.average())
            self._last_log_ts = current_time
