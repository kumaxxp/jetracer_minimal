"""Main application for JetRacer minimal segmentation prototype."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml

from benchmark import PerformanceMonitor
from camera import JetCamera
from segmentation import SegmentationModel


running = True


def signal_handler(sig, frame):  # type: ignore[unused-argument]
    """Handle Ctrl+C for graceful shutdown."""

    global running
    print("\n\nShutting down...")
    running = False


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as cfg:
        return yaml.safe_load(cfg)


def create_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create visualization with mask overlay."""

    colored_mask = np.zeros_like(frame)
    colored_mask[mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(frame, 1.0, colored_mask, alpha, 0)
    return overlay


def draw_stats(frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """Draw performance statistics on frame."""

    cv2.rectangle(frame, (5, 5), (280, 85), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference: {stats['inference_ms']:.1f}ms", (10, 50), font, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Memory: {stats['memory_mb']:.0f}MB", (10, 75), font, 0.6, (0, 255, 0), 2)
    return frame


def main() -> None:
    global running

    signal.signal(signal.SIGINT, signal_handler)

    print("Loading configuration...")
    config = load_config()

    print("\nInitializing camera...")
    camera = JetCamera(
        width=config["camera"]["width"],
        height=config["camera"]["height"],
        fps=config["camera"]["fps"],
        device=config["camera"].get("device", 0),
    )

    print("\nLoading segmentation model...")
    model = SegmentationModel(
        model_path=config["model"]["path"],
        input_size=tuple(config["model"]["input_size"]),
        road_classes=config["segmentation"]["road_classes"],
    )

    print("\nInitializing performance monitor...")
    monitor = PerformanceMonitor()

    camera.start()

    window_name = "JetRacer Segmentation - Press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "=" * 50)
    print("System running. Press 'Q' to quit.")
    print("=" * 50 + "\n")

    last_print_time = time.time()

    try:
        while running:
            frame = camera.read()
            if frame is None:
                print("Warning: Failed to read frame")
                time.sleep(0.1)
                continue

            start_time = time.time()
            mask = model.inference(frame)
            inference_time = time.time() - start_time

            monitor.update(inference_time)

            overlay = create_overlay(frame, mask, alpha=config["display"]["overlay_alpha"])
            stats = monitor.get_stats()
            display_frame = draw_stats(overlay, stats)

            cv2.imshow(window_name, display_frame)

            if time.time() - last_print_time >= 1.0:
                monitor.print_stats()
                last_print_time = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    except Exception as exc:  # noqa: BLE001
        print(f"\nError in main loop: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        camera.stop()
        cv2.destroyAllWindows()
        print("\u2713 Shutdown complete")


if __name__ == "__main__":
    main()
