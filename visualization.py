"""Display helpers for JetRacer minimal prototype."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class Visualizer:
    """Renders camera frames with optional overlay and FPS text."""

    def __init__(
        self,
        enable_ui: bool = True,
        show_fps: bool = True,
        overlay_mask: bool = True,
        window_name: str = "JetRacer-Minimal",
    ) -> None:
        self.enable_ui = enable_ui
        self.show_fps = show_fps
        self.overlay_mask = overlay_mask
        self.window_name = window_name
        self.mask_color = (0, 255, 0)

    def render(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        if frame is None:
            raise ValueError("frame is required")
        output = frame.copy()
        if self.overlay_mask and mask is not None:
            overlay = np.zeros_like(output)
            overlay[mask > 0] = self.mask_color
            output = cv2.addWeighted(output, 1.0, overlay, 0.4, 0.0)
        if self.show_fps and fps:
            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
        if self.enable_ui:
            cv2.imshow(self.window_name, output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                LOGGER.info("'q' pressed - exiting display loop")
                raise KeyboardInterrupt
        return output

    def close(self) -> None:
        if self.enable_ui:
            cv2.destroyWindow(self.window_name)
