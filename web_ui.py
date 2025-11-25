"""Web UI with enhanced debugging for frame streaming."""

from __future__ import annotations

import asyncio
import base64
import time

import cv2
import numpy as np
from nicegui import ui, app  # noqa: F401


class WebUI:
    """NiceGUI web interface with debugging."""

    def __init__(self, camera, model, monitor) -> None:
        self.camera = camera
        self.model = model
        self.monitor = monitor
        self.running = False
        self.overlay_enabled = True
        self.overlay_alpha = 0.4
        self.frame_send_count = 0

        self.video_canvas = None
        self.test_image = None
        self.start_button = None
        self.test_button = None
        self.fps_label = None
        self.inference_label = None
        self.frames_sent_label = None
        self.status_label = None
        self.debug_log = None
        self.process_task: asyncio.Task | None = None

        print("[WebUI] Initialized")

    def setup_ui(self) -> None:
        ui.page_title("JetRacer Segmentation - Debug")

        with ui.header().classes("items-center justify-between"):
            ui.label("JetRacer Minimal - Phase 0 (Debug Mode)").classes("text-2xl")

        with ui.column().classes("w-full items-center gap-4 p-4"):
            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Camera Feed").classes("text-xl mb-2")
                self.test_image = ui.image().classes("w-full")
                self.video_canvas = ui.html(
                    """
                    <canvas id="videoCanvas" 
                            width="640" 
                            height="480" 
                            style="width:100%; max-width:640px; border:2px solid #333; background:#000;"></canvas>
                    <div id="debugInfo" style="color:white; font-size:12px; margin-top:10px;">
                        Waiting for frames...
                    </div>
                    """,
                    sanitize=False,
                ).classes("w-full")

            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Controls").classes("text-xl mb-2")
                with ui.row().classes("gap-4"):
                    self.start_button = ui.button(
                        "Start Camera & Processing",
                        on_click=self.toggle_processing,
                    ).props("color=green")
                    self.test_button = ui.button(
                        "Test Single Frame",
                        on_click=self.test_single_frame,
                    ).props("color=blue")
                    ui.switch(
                        "Segmentation Overlay",
                        value=True,
                        on_change=lambda e: self.set_overlay(e.value),
                    )

            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Performance & Debug").classes("text-xl mb-2")
                with ui.grid(columns=4).classes("gap-4"):
                    with ui.card().tight():
                        ui.label("FPS").classes("text-sm")
                        self.fps_label = ui.label("0.0").classes("text-2xl font-bold")
                    with ui.card().tight():
                        ui.label("Inference").classes("text-sm")
                        self.inference_label = ui.label("0 ms").classes("text-2xl font-bold")
                    with ui.card().tight():
                        ui.label("Frames Sent").classes("text-sm")
                        self.frames_sent_label = ui.label("0").classes("text-2xl font-bold")
                    with ui.card().tight():
                        ui.label("Status").classes("text-sm")
                        self.status_label = ui.label("Ready").classes("text-2xl font-bold text-gray-500")

            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Debug Log").classes("text-xl mb-2")
                self.debug_log = ui.log(max_lines=20).classes("w-full h-48")

        print("[WebUI] UI setup complete")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        if self.debug_log is not None:
            self.debug_log.push(f"[{timestamp}] {message}")
        print(f"[WebUI] {message}")

    async def test_single_frame(self) -> None:
        self.log("Testing single frame capture...")
        if not self.camera.running:
            success = self.camera.start()
            if not success:
                self.log("ERROR: Failed to start camera")
                ui.notify("Camera start failed", type="negative")
                return
        frame = self.camera.read()
        if frame is None:
            self.log("ERROR: Read returned None")
            ui.notify("Frame read failed", type="negative")
            return
        self.log(f"\u2713 Frame captured: {frame.shape}, {frame.dtype}")
        try:
            await self.send_frame_to_canvas(frame)
            await self.update_image_element(frame)
            self.log("\u2713 Frame sent to browser")
            ui.notify("Frame test successful", type="positive")
        except Exception as exc:  # noqa: BLE001
            self.log(f"ERROR sending frame: {exc}")
            ui.notify(f"Frame send failed: {exc}", type="negative")

    def toggle_processing(self) -> None:
        if not self.running:
            self.start_processing()
        else:
            self.stop_processing()

    def start_processing(self) -> None:
        self.log("Starting processing...")
        success = self.camera.start()
        if not success:
            self.log("ERROR: Camera failed to start")
            ui.notify("Camera start failed", type="negative")
            return
        self.running = True
        self.start_button.props("color=red")
        self.start_button.text = "Stop"
        self.status_label.text = "Running"
        self.status_label.classes("text-2xl font-bold text-green-500")
        self.log("\u2713 Processing started")
        ui.notify("Processing started", type="positive")
        self.process_task = asyncio.create_task(self.process_loop())

    def stop_processing(self) -> None:
        self.log("Stopping processing...")
        self.running = False
        if self.process_task is not None:
            self.process_task.cancel()
            self.process_task = None
        self.camera.stop()
        self.start_button.props("color=green")
        self.start_button.text = "Start Camera & Processing"
        self.status_label.text = "Stopped"
        self.status_label.classes("text-2xl font-bold text-gray-500")
        self.log("\u2713 Processing stopped")
        ui.notify("Stopped", type="warning")

    def set_overlay(self, enabled: bool) -> None:
        self.overlay_enabled = enabled
        self.log(f"Overlay: {'enabled' if enabled else 'disabled'}")

    async def process_loop(self) -> None:
        self.log("Process loop started")
        error_count = 0
        while self.running:
            try:
                frame = self.camera.read()
                if frame is None:
                    error_count += 1
                    if error_count % 10 == 1:
                        self.log(f"WARNING: Frame read failed (error count: {error_count})")
                    await asyncio.sleep(0.1)
                    continue
                error_count = 0
                start_time = time.time()
                mask = self.model.inference(frame)
                inference_time = time.time() - start_time
                self.monitor.update(inference_time)
                if self.overlay_enabled:
                    display_frame = self.create_overlay(frame, mask)
                else:
                    display_frame = frame
                await self.send_frame_to_canvas(display_frame)
                await self.update_image_element(display_frame)
                self.frame_send_count += 1
                stats = self.monitor.get_stats()
                self.fps_label.text = f"{stats['fps']:.1f}"
                self.inference_label.text = f"{stats['inference_ms']:.1f} ms"
                self.frames_sent_label.text = str(self.frame_send_count)
                if self.frame_send_count % 100 == 0:
                    self.log(f"Processed {self.frame_send_count} frames, FPS: {stats['fps']:.1f}")
                await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                self.log("Process loop cancelled")
                break
            except Exception as exc:  # noqa: BLE001
                self.log(f"ERROR in loop: {exc}")
                import traceback

                traceback.print_exc()
                await asyncio.sleep(0.5)
        self.log("Process loop exited")
        self.process_task = None

    def create_overlay(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = [0, 255, 0]
        return cv2.addWeighted(frame, 1.0, colored_mask, self.overlay_alpha, 0)

    async def send_frame_to_canvas(self, frame: np.ndarray) -> None:
        display_frame = cv2.resize(frame, (640, 480))
        success, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            self.log("ERROR: JPEG encoding failed")
            return
        if self.video_canvas is None:
            self.log("ERROR: Video canvas not initialized")
            return
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        await self.video_canvas.client.run_javascript(
            """
            const canvas = document.getElementById('videoCanvas');
            const debugDiv = document.getElementById('debugInfo');
            if (canvas) {
                const ctx = canvas.getContext('2d');
                const img = new Image();
                img.onload = function() {
                    ctx.drawImage(img, 0, 0, 640, 480);
                    if (debugDiv) {
                        debugDiv.textContent = 'Frame updated: ' + new Date().toLocaleTimeString();
                    }
                };
                img.onerror = function() {
                    if (debugDiv) {
                        debugDiv.textContent = 'ERROR: Failed to load image';
                    }
                };
                img.src = 'data:image/jpeg;base64:%s';
            } else {
                console.error('Canvas element not found');
            }
            """ % img_base64
        )

    async def update_image_element(self, frame: np.ndarray) -> None:
        display_frame = cv2.resize(frame, (640, 480))
        success, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        if self.test_image is not None:
            self.test_image.source = f"data:image/jpeg;base64,{img_base64}"