"""Entry point for the JetRacer minimal segmentation prototype."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from benchmark import FPSMeter
from camera import Camera
from segmentation import SegmentationModel
from visualization import Visualizer

LOGGER = logging.getLogger("jetracer_minimal")


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    return parser


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def main() -> None:
    args = build_argparser().parse_args()
    setup_logging()
    cfg = load_config(args.config)
    camera_cfg = cfg.get("camera", {})
    model_cfg = cfg.get("model", {})
    segmentation_cfg = cfg.get("segmentation", {})
    display_cfg = cfg.get("display", {})

    segmentation_model = SegmentationModel(
        model_path=model_cfg["path"],
        input_size=model_cfg["input_size"],
        road_classes=segmentation_cfg.get("road_classes", (0,)),
        threshold=segmentation_cfg.get("threshold", 0.5),
    )

    visualizer = Visualizer(
        enable_ui=display_cfg.get("enable_ui", True),
        show_fps=display_cfg.get("show_fps", True),
        overlay_mask=display_cfg.get("overlay_mask", True),
    )
    fps_meter = FPSMeter()

    try:
        with Camera(
            width=camera_cfg.get("width", 320),
            height=camera_cfg.get("height", 240),
            fps=camera_cfg.get("fps", 30),
            device=camera_cfg.get("device", 0),
        ) as camera:
            LOGGER.info("Starting segmentation loop. Press Ctrl+C to stop.")
            while True:
                frame = camera.read(timeout=1.0)
                mask, road_prob, latency = segmentation_model.infer(frame)
                fps = fps_meter.tick()
                LOGGER.debug("Latency: %.2f ms", latency * 1000)
                visualizer.render(frame, mask if display_cfg.get("overlay_mask", True) else None, fps)
    except KeyboardInterrupt:
        LOGGER.info("Shutting down gracefully (Ctrl+C)")
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
