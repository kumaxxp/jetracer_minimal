"""Optimized Web UI with single canvas display and frame rate control."""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import cv2
import numpy as np
from nicegui import ui


class WebUI:
    """Optimized NiceGUI web interface."""

    def __init__(self, camera, model, monitor) -> None:
        self.camera = camera
        self.model = model
        self.monitor = monitor
        self.running = False
        self.overlay_enabled = True
        self.overlay_alpha = 0.4
        self.frame_send_count = 0
        self.process_task: asyncio.Task | None = None

        # Frame rate control
        self.target_display_fps = 20
        self.last_send_time = 0.0
        self.min_send_interval = 1.0 / self.target_display_fps

        # UI components
        self.video_container = None
        self.start_button = None
        self.fps_slider = None
        self.fps_label = None
        self.inference_label = None
        self.memory_label = None
        self.frames_label = None
        self.status_label = None
        self._canvas_script_added = False

        print(f"[WebUI] Initialized (target display FPS: {self.target_display_fps})")

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

                    ui.separator().props("vertical")

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Display FPS:")
                        self.fps_slider = ui.slider(
                            min=5,
                            max=30,
                            step=5,
                            value=20,
                            on_change=self.update_target_fps,
                        ).props("label").classes("w-40")
                        ui.label("20 FPS").bind_text_from(
                            self.fps_slider,
                            "value",
                            backward=lambda v: f"{int(v)} FPS",
                        )

                    ui.separator().props("vertical")

                    ui.switch(
                        "Segmentation Overlay",
                        value=True,
                        on_change=lambda e: self.set_overlay(e.value),
                    )

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Alpha:")
                        ui.slider(
                            min=0.0,
                            max=1.0,
                            step=0.1,
                            value=0.4,
                            on_change=lambda e: setattr(self, "overlay_alpha", e.value),
                        ).props("label").classes("w-32")

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
        self.target_display_fps = max(1, float(e.value))
        self.min_send_interval = 1.0 / self.target_display_fps
        print(f"[WebUI] Display FPS: {self.target_display_fps}")

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
        self.overlay_enabled = enabled
        print(f"[WebUI] Overlay: {enabled}")

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
        """Main processing loop with frame rate control."""
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

                    start_time = time.time()
                    mask = self.model.inference(frame)
                    inference_time = time.time() - start_time

                    self.monitor.update(inference_time)

                    current_time = time.time()
                    time_since_last = current_time - self.last_send_time

                    if time_since_last >= self.min_send_interval:
                        if self.overlay_enabled:
                            display_frame = self.create_overlay(frame, mask)
                        else:
                            display_frame = frame

                        await self.send_frame(display_frame)
                        self.last_send_time = current_time
                        self.frame_send_count += 1

                        stats = self.monitor.get_stats()
                        if self.fps_label is not None:
                            self.fps_label.text = f"{stats['fps']:.1f}"
                        if self.inference_label is not None:
                            self.inference_label.text = f"{stats['inference_ms']:.1f} ms"
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
            print("[WebUI] ERROR: Video container not initialized")
            return
        try:
            display_frame = cv2.resize(frame, (640, 480))
            success, buffer = cv2.imencode(
                ".jpg",
                display_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )

            if not success:
                return

            img_base64 = base64.b64encode(buffer).decode("utf-8")
            await self.video_container.client.run_javascript(
                f'window.updateVideoFrame("{img_base64}")'
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WebUI] Send error: {exc}")