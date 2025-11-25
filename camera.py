"""CSI camera wrapper built on JetCam for Jetson-class devices."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
from jetcam.csi_camera import CSICamera

LOGGER = logging.getLogger(__name__)


class Camera:
    """Thread-safe convenience wrapper around JetCam's CSICamera."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        device: int = 0,
        warmup_frames: int = 5,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.warmup_frames = max(0, warmup_frames)
        self._camera: Optional[CSICamera] = None
        self._frame_ready = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._warmup_remaining = self.warmup_frames

    def start(self) -> None:
        if self._camera is not None:
            return
        LOGGER.info(
            "Starting CSI camera (device=%s, %sx%s@%sfps)",
            self.device,
            self.width,
            self.height,
            self.fps,
        )
        self._camera = CSICamera(
            width=self.width,
            height=self.height,
            capture_fps=self.fps,
            capture_device=self.device,
        )
        self._camera.observe(self._on_frame, names="value")
        self._camera.running = True

    def _on_frame(self, change) -> None:
        frame = change.get("new")
        if frame is None:
            return
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            return
        self._latest_frame = frame.copy()
        self._frame_ready.set()

    def read(self, timeout: float = 1.0) -> np.ndarray:
        if self._camera is None:
            self.start()
        if not self._frame_ready.wait(timeout=timeout):
            raise TimeoutError("Camera frame timeout")
        self._frame_ready.clear()
        if self._latest_frame is None:
            raise RuntimeError("Camera returned empty frame")
        return self._latest_frame

    def stop(self) -> None:
        if self._camera is None:
            return
        LOGGER.info("Stopping CSI camera")
        self._camera.running = False
        self._camera.unobserve(self._on_frame, names="value")
        self._camera = None
        self._latest_frame = None
        self._frame_ready.clear()

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
