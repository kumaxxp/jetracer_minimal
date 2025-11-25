"""Optimized Web UI with single canvas display and frame rate control."""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from nicegui import ui
import yaml


class WebUI:
    """Optimized NiceGUI web interface."""

    def __init__(
        self,
        camera,
        model,
        monitor,
        display_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.camera = camera
        self.model = model
        self.monitor = monitor
        self.display_config = dict(display_config or {})
        config = self.display_config
        self.running = False
        self.stream_only_mode = bool(config.get("stream_only_mode", True))
        overlay_default = bool(config.get("overlay_mask", False))
        self.overlay_enabled = overlay_default and not self.stream_only_mode
        self.overlay_alpha = float(config.get("overlay_alpha", 0.4))
        self.frame_send_count = 0
        self.process_task: asyncio.Task | None = None

        # Frame rate control
        default_fps = float(config.get("target_fps", config.get("display_fps", 10)) or 10)
        self.target_display_fps = max(5.0, min(20.0, default_fps))
        self.last_send_time = 0.0
        self.min_send_interval = 1.0 / self.target_display_fps

        # JPEG quality control
        self.jpeg_quality = int(config.get("jpeg_quality", 80))
        self.jpeg_quality = min(95, max(50, self.jpeg_quality))
        self.segmentation_interval = max(1, int(config.get("segmentation_interval", 10)))
        self.frame_process_count = 0
        self.last_mask: np.ndarray | None = None
        self._force_segmentation = False

        # UI components
        self.video_container = None
        self.start_button = None
        self.fps_slider = None
        self.fps_label = None
        self.inference_label = None
        self.memory_label = None
        self.frames_label = None
        self.status_label = None
        self.quality_slider = None
        self.segmentation_slider = None
        self.overlay_switch = None
        self.stream_only_checkbox = None
        self._suppress_overlay_event = False
        self._canvas_script_added = False

        print(
            "[WebUI] Initialized "
            f"(display FPS: {self.target_display_fps}, JPEG: {self.jpeg_quality})"
        )

    def setup_ui(self) -> None:
        """Setup optimized NiceGUI interface."""
        ui.page_title("JetRacer Segmentation")

        with ui.header().classes("items-center justify-between"):
            ui.label("JetRacer Minimal - Phase 0").classes("text-2xl")
            ui.label("Optimized Streaming").classes("text-sm text-gray-400")

        with ui.column().classes("w-full items-center gap-4 p-4"):
            # Single video display
            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Camera Feed").classes("text-xl mb-2")

                self.video_container = ui.html(
                    """
                    <div style="position: relative; width: 100%; max-width: 640px; margin: 0 auto;">
                        <canvas id="videoCanvas" 
                                width="640" 
                                height="480" 
                                style="width: 100%; border: 2px solid #333; background: #000; display: block;">
                        </canvas>
                        <div id="frameInfo" 
                             style="position: absolute; top: 10px; left: 10px; 
                                    color: #0f0; font-family: monospace; font-size: 14px;
                                    text-shadow: 1px 1px 2px #000; background: rgba(0,0,0,0.5);
                                    padding: 5px; border-radius: 3px;">
                            Waiting...
                        </div>
                    </div>
                    """,
                    sanitize=False,
                ).classes("w-full")
                self._ensure_canvas_script()

            # Controls
            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Controls").classes("text-xl mb-2")

                with ui.row().classes("gap-4 items-center flex-wrap"):
                    self.start_button = ui.button(
                        "\u25b6 Start",
                        on_click=self.toggle_processing,
                    ).props("color=green size=lg")

                    self.stream_only_checkbox = ui.checkbox(
                        "Stream Only (No Segmentation)",
                        value=self.stream_only_mode,
                        on_change=lambda e: self.set_stream_only(e.value),
                    )

                    ui.separator().props("vertical")

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Display FPS:")
                        self.fps_slider = ui.slider(
                            min=5,
                            max=20,
                            step=5,
                            value=self.target_display_fps,
                            on_change=self.update_target_fps,
                        ).props("label").classes("w-40")
                        ui.label(f"{int(self.target_display_fps)} FPS").bind_text_from(
                            self.fps_slider,
                            "value",
                            backward=lambda v: f"{int(v)} FPS",
                        )

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Seg Interval:")
                        self.segmentation_slider = (
                            ui.slider(
                                min=1,
                                max=30,
                                step=1,
                                value=self.segmentation_interval,
                                on_change=self.update_segmentation_interval,
                            )
                            .props("label")
                            .classes("w-40")
                        )
                        ui.label("Every 10 frames").bind_text_from(
                            self.segmentation_slider,
                            "value",
                            backward=lambda v: "Every frame"
                            if int(v) == 1
                            else f"Every {int(v)} frames",
                        )

                    ui.separator().props("vertical")

                    self.overlay_switch = ui.switch(
                        "Segmentation Overlay",
                        value=self.overlay_enabled,
                        on_change=lambda e: self._handle_overlay_change(e.value),
                    )

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Alpha:")
                        ui.slider(
                            min=0.0,
                            max=1.0,
                            step=0.1,
                            value=self.overlay_alpha,
                            on_change=lambda e: setattr(self, "overlay_alpha", e.value),
                        ).props("label").classes("w-32")

                    with ui.row().classes("items-center gap-2"):
                        ui.label("JPEG Quality:")
                        self.quality_slider = (
                            ui.slider(
                                min=50,
                                max=95,
                                step=5,
                                value=self.jpeg_quality,
                                on_change=lambda e: setattr(self, "jpeg_quality", int(e.value)),
                            )
                            .props("label")
                            .classes("w-32")
                        )
                        ui.label(str(self.jpeg_quality)).bind_text_from(
                            self.quality_slider,
                            "value",
                            backward=lambda v: str(int(v)),
                        )

            # Performance stats
            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Performance").classes("text-xl mb-2")

                with ui.grid(columns=5).classes("gap-4"):
                    with ui.card().tight().classes("p-4"):
                        ui.label("Processing FPS").classes("text-xs text-gray-400")
                        self.fps_label = ui.label("0.0").classes("text-3xl font-bold text-green-500")

                    with ui.card().tight().classes("p-4"):
                        ui.label("Inference").classes("text-xs text-gray-400")
                        self.inference_label = ui.label("0 ms").classes("text-3xl font-bold text-blue-500")

                    with ui.card().tight().classes("p-4"):
                        ui.label("Memory").classes("text-xs text-gray-400")
                        self.memory_label = ui.label("0 MB").classes("text-3xl font-bold text-purple-500")

                    with ui.card().tight().classes("p-4"):
                        ui.label("Frames Sent").classes("text-xs text-gray-400")
                        self.frames_label = ui.label("0").classes("text-3xl font-bold text-orange-500")

                    with ui.card().tight().classes("p-4"):
                        ui.label("Status").classes("text-xs text-gray-400")
                        self.status_label = ui.label("Ready").classes("text-3xl font-bold text-gray-500")

        print("[WebUI] UI setup complete")

    def update_target_fps(self, e: Any) -> None:
        """Update target display FPS."""
        value = max(5.0, min(20.0, float(e.value)))
        self.target_display_fps = value
        self.min_send_interval = 1.0 / self.target_display_fps
        print(f"[WebUI] Display FPS: {self.target_display_fps}")

    def update_segmentation_interval(self, e: Any) -> None:
        """Update how often segmentation runs."""
        self.segmentation_interval = max(1, int(e.value))
        self._force_segmentation = True
        print(f"[WebUI] Segmentation interval: {self.segmentation_interval}")

    def toggle_processing(self) -> None:
        """Toggle start/stop."""
        if not self.running:
            self.start_processing()
        else:
            self.stop_processing()

    def start_processing(self) -> None:
        """Start processing."""
        print("[WebUI] Starting...")

        success = self.camera.start()
        if not success:
            print("[WebUI] ERROR: Camera failed")
            ui.notify("Camera start failed", type="negative")
            return

        self.running = True
        self.frame_send_count = 0
        self.last_send_time = 0.0
        self.frame_process_count = 0
        self.last_mask = None
        self._force_segmentation = True

        if self.start_button is not None:
            self.start_button.props("color=red")
            self.start_button.text = "\u23f9 Stop"
        if self.status_label is not None:
            self.status_label.text = "Running"
            self.status_label.classes("text-3xl font-bold text-green-500")

        print("[WebUI] \u2713 Started")
        ui.notify("Started", type="positive")

        self.process_task = asyncio.create_task(self.process_loop())

    def stop_processing(self) -> None:
        """Stop processing."""
        print("[WebUI] Stopping...")
        self.running = False

        if self.process_task:
            self.process_task.cancel()
            self.process_task = None

        self.camera.stop()

        if self.start_button is not None:
            self.start_button.props("color=green")
            self.start_button.text = "\u25b6 Start"
        if self.status_label is not None:
            self.status_label.text = "Stopped"
            self.status_label.classes("text-3xl font-bold text-gray-500")

        print("[WebUI] \u2713 Stopped")
        ui.notify("Stopped", type="warning")

    def set_overlay(self, enabled: bool) -> None:
        """Toggle overlay."""
        if self.overlay_switch is not None and self.overlay_switch.value != enabled:
            self._suppress_overlay_event = True
            self.overlay_switch.value = enabled
            self._suppress_overlay_event = False
        self.overlay_enabled = enabled
        if enabled:
            self.stream_only_mode = False
            if self.stream_only_checkbox is not None:
                self.stream_only_checkbox.value = False
            self._force_segmentation = True
        else:
            self.last_mask = None
        print(f"[WebUI] Overlay: {enabled}")

    def set_stream_only(self, enabled: bool) -> None:
        """Toggle stream-only mode."""
        self.stream_only_mode = enabled
        if enabled and self.overlay_enabled:
            self.set_overlay(False)
        if not enabled:
            print("[WebUI] Stream-only mode OFF (processing enabled)")
        else:
            print("[WebUI] Stream-only mode ON (max performance)")

    def _handle_overlay_change(self, value: bool) -> None:
        if self._suppress_overlay_event:
            return
        self.set_overlay(value)

    def _ensure_canvas_script(self) -> None:
        if self._canvas_script_added:
            return
        ui.add_body_html(
            """
            <script>
                (function () {
                    if (window.__jetVideoInit) {
                        return;
                    }
                    window.__jetVideoInit = true;
                    let frameCount = 0;
                    let lastUpdate = Date.now();

                    function ensureContext() {
                        const canvas = document.getElementById('videoCanvas');
                        if (!canvas) {
                            return null;
                        }
                        return canvas.getContext('2d', { alpha: false });
                    }

                    window.updateVideoFrame = function (base64Data) {
                        const ctx = ensureContext();
                        if (!ctx) {
                            console.error('Canvas context not available');
                            return;
                        }
                        const img = new Image();
                        img.onload = function () {
                            ctx.drawImage(img, 0, 0, 640, 480);
                            frameCount += 1;
                            const now = Date.now();
                            const elapsed = now - lastUpdate;
                            const fps = elapsed > 0 ? (1000 / elapsed).toFixed(1) : '0.0';
                            lastUpdate = now;
                            const infoDiv = document.getElementById('frameInfo');
                            if (infoDiv) {
                                infoDiv.textContent = `Frame ${frameCount} | Display: ${fps} FPS`;
                            }
                        };
                        img.onerror = function () {
                            console.error('Failed to load frame image');
                        };
                        img.src = 'data:image/jpeg;base64,' + base64Data;
                    };
                    console.log('Video canvas script injected');
                })();
            </script>
            """
        )
        self._canvas_script_added = True

    async def process_loop(self) -> None:
        """Main processing loop with conditional segmentation."""
        print("[WebUI] Loop started")
        error_count = 0

        try:
            while self.running:
                try:
                    frame = self.camera.read()
                    if frame is None:
                        error_count += 1
                        if error_count % 30 == 1:
                            print(f"[WebUI] Frame read failed (count: {error_count})")
                        await asyncio.sleep(0.05)
                        continue

                    error_count = 0
                    self.frame_process_count += 1

                    run_inference = (
                        self.overlay_enabled
                        and (
                            self._force_segmentation
                            or self.frame_process_count % self.segmentation_interval == 0
                        )
                    )

                    inference_time: float | None = None
                    if run_inference:
                        start_time = time.time()
                        self.last_mask = self.model.inference(frame)
                        inference_time = time.time() - start_time
                        self._force_segmentation = False

                    self.monitor.update(inference_time if run_inference else 0.0)

                    current_time = time.time()
                    time_since_last = current_time - self.last_send_time

                    if time_since_last >= self.min_send_interval:
                        if self.overlay_enabled and self.last_mask is not None:
                            display_frame = self.create_overlay(frame, self.last_mask)
                        else:
                            display_frame = frame

                        await self.send_frame(display_frame)
                        self.last_send_time = current_time
                        self.frame_send_count += 1

                        stats = self.monitor.get_stats()
                        if self.fps_label is not None:
                            self.fps_label.text = f"{stats['fps']:.1f}"
                        if self.inference_label is not None:
                            if self.overlay_enabled:
                                self.inference_label.text = f"{stats['inference_ms']:.1f} ms"
                            else:
                                self.inference_label.text = "OFF"
                        if self.memory_label is not None:
                            self.memory_label.text = f"{stats['memory_mb']:.0f} MB"
                        if self.frames_label is not None:
                            self.frames_label.text = str(self.frame_send_count)

                    await asyncio.sleep(0.001)

                except asyncio.CancelledError:
                    print("[WebUI] Loop cancelled")
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"[WebUI] ERROR: {exc}")
                    await asyncio.sleep(0.5)

        finally:
            print("[WebUI] Loop ended")
            self.process_task = None

    def create_overlay(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create visualization with overlay."""
        h, w = frame.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = [0, 255, 0]

        return cv2.addWeighted(frame, 1.0, colored_mask, self.overlay_alpha, 0)

    async def send_frame(self, frame: np.ndarray) -> None:
        """Send frame to browser (optimized)."""
        if self.video_container is None:
            return
        try:
            if frame.shape[1] == 640 and frame.shape[0] == 480:
                display_frame = frame
            else:
                display_frame = cv2.resize(frame, (640, 480))

            quality = min(95, max(50, int(self.jpeg_quality)))
            img_base64 = await asyncio.to_thread(
                self._encode_frame,
                display_frame,
                quality,
            )
            await asyncio.wait_for(
                self.video_container.client.run_javascript(
                    f'window.updateVideoFrame("{img_base64}")'
                ),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            print("[WebUI] Send timeout (browser slow)")
        except Exception as exc:  # noqa: BLE001
            if self.frame_send_count % 100 == 0:
                print(f"[WebUI] Send error: {exc}")

    def _encode_frame(self, frame: np.ndarray, quality: int) -> str:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, buffer = cv2.imencode(".jpg", frame, params)
        if not success:
            raise RuntimeError("JPEG encoding failed")
        raw_bytes = buffer.tobytes() if hasattr(buffer, "tobytes") else bytes(buffer)
        return base64.b64encode(raw_bytes).decode("ascii")


def _load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with path.open("r", encoding="utf-8") as cfg:
        return yaml.safe_load(cfg)


def _run_standalone() -> None:
    import argparse

    from benchmark import PerformanceMonitor
    from camera import JetCamera
    from segmentation import SegmentationModel

    parser = argparse.ArgumentParser(description="Run WebUI standalone")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for NiceGUI")
    parser.add_argument("--port", type=int, default=8080, help="Port for NiceGUI server")
    args = parser.parse_args()

    config = _load_config(args.config)
    camera_cfg = config.get("camera", {})
    model_cfg = config.get("model", {})
    segmentation_cfg = config.get("segmentation", {})
    display_cfg = config.get("display", {})

    camera = JetCamera(
        width=int(camera_cfg.get("width", 640)),
        height=int(camera_cfg.get("height", 480)),
        fps=int(camera_cfg.get("fps", 30)),
        device=int(camera_cfg.get("device", 0)),
    )

    input_size = model_cfg.get("input_size", [camera_cfg.get("width", 640), camera_cfg.get("height", 480)])
    model = SegmentationModel(
        model_path=model_cfg.get("path", "models/dummy_segmentation_640x480.onnx"),
        input_size=tuple(input_size),
        road_classes=segmentation_cfg.get("road_classes", [0, 1]),
    )

    monitor = PerformanceMonitor()

    if "interval" in segmentation_cfg and "segmentation_interval" not in display_cfg:
        display_cfg["segmentation_interval"] = segmentation_cfg["interval"]

    web_ui = WebUI(camera, model, monitor, display_cfg)
    web_ui.setup_ui()

    print("\n" + "=" * 60)
    print("Web UI started!")
    print(f"Access from browser: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    ui.run(host=args.host, port=args.port, title="JetRacer Segmentation", reload=False, show=False)


if __name__ == "__main__":
    _run_standalone()