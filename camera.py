"""Camera module for Jetson CSI camera."""

from __future__ import annotations

import time

import cv2
import numpy as np
from jetcam.csi_camera import CSICamera


class JetCamera:
    """Wrapper for Jetson CSI Camera."""

    def __init__(self, width: int = 320, height: int = 240, fps: int = 30, device: int = 0) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.camera: CSICamera | None = None
        self.running = False
        print(f"Initializing camera {device}: {width}x{height} @ {fps}fps")

    def start(self) -> None:
        """Start camera capture."""

        if self.running:
            print("Camera already running")
            return

        try:
            self.camera = CSICamera(
                width=self.width,
                height=self.height,
                capture_fps=self.fps,
                capture_device=self.device,
            )
            self.running = True
            print("\u2713 Camera started successfully")
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            print(f"\u2717 Failed to start camera: {exc}")
            raise

    def read(self) -> np.ndarray | None:
        """Read a frame from camera."""

        if not self.running or self.camera is None:
            return None
        try:
            frame = self.camera.read()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as exc:  # noqa: BLE001
            print(f"Error reading frame: {exc}")
            return None

    def stop(self) -> None:
        """Stop camera and release resources."""

        if self.camera is not None:
            try:
                self.camera.release()
                print("\u2713 Camera stopped")
            except Exception as exc:  # noqa: BLE001
                print(f"Error stopping camera: {exc}")
        self.running = False
        self.camera = None

    def __enter__(self) -> "JetCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False
