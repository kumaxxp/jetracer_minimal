"""Main application with NiceGUI web interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from nicegui import ui
import yaml

from benchmark import PerformanceMonitor
from camera import JetCamera
from segmentation import SegmentationModel
from web_ui import WebUI


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as cfg:
        return yaml.safe_load(cfg)


def main() -> None:
    print("Loading configuration...")
    config = load_config()

    print("Initializing camera...")
    camera = JetCamera(
        width=config["camera"]["width"],
        height=config["camera"]["height"],
        fps=config["camera"]["fps"],
        device=config["camera"]["device"],
    )

    print("Loading segmentation model...")
    model = SegmentationModel(
        model_path=config["model"]["path"],
        input_size=tuple(config["model"]["input_size"]),
        road_classes=config["segmentation"]["road_classes"],
    )

    print("Initializing performance monitor...")
    monitor = PerformanceMonitor()

    print("Setting up web interface...")
    display_config = dict(config.get("display", {}))
    segmentation_cfg = config.get("segmentation", {})
    if "interval" in segmentation_cfg and "segmentation_interval" not in display_config:
        display_config["segmentation_interval"] = segmentation_cfg["interval"]
    web_ui = WebUI(camera, model, monitor, display_config)
    web_ui.setup_ui()

    print("\n" + "=" * 60)
    print("Web UI started!")
    print("Access from browser: http://<JETSON_IP>:8080")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    ui.run(
        host="0.0.0.0",
        port=8080,
        title="JetRacer Segmentation",
        reload=False,
        show=False,
    )


if __name__ == "__main__":
    main()
