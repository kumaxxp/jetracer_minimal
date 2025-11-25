"""Interactive camera testing UI with real-time performance monitoring."""

from __future__ import annotations

import asyncio
import base64
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
from nicegui import ui

from camera import JetCamera


class CameraTestUI:
    """Interactive camera testing interface."""

    def __init__(self) -> None:
        self.camera: JetCamera | None = None
        self.running = False
        self.process_task: asyncio.Task | None = None

        self.width = 320
        self.height = 240
        self.fps = 30
        self.jpeg_quality = 85
        self.display_fps = 20

        self.metrics_window = 60
        self.read_times: deque[float] = deque(maxlen=self.metrics_window)
        self.encode_times: deque[float] = deque(maxlen=self.metrics_window)
        self.send_times: deque[float] = deque(maxlen=self.metrics_window)
        self.frame_intervals: deque[float] = deque(maxlen=self.metrics_window)

        self.last_frame_time = 0.0
        self.last_send_time = 0.0
        self.min_send_interval = 1.0 / self.display_fps

        self.total_frames = 0
        self.sent_frames = 0
        self.dropped_frames = 0

        self.video_container = None
        self.start_button = None
        self.status_label = None
        self.resolution_select = None
        self.fps_select = None
        self.capture_fps_label = None
        self.display_fps_label = None
        self.frames_label = None
        self.dropped_label = None
        self.read_time_label = None
        self.encode_time_label = None
        self.send_time_label = None
        self.chart = None
        self.chart_option = {
            "animation": False,
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Read", "Encode", "Send"]},
            "grid": {"left": 40, "right": 16, "top": 24, "bottom": 30},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": list(range(60)),
            },
            "yAxis": {
                "type": "value",
                "min": 0,
                "max": 50,
            },
            "series": [
                {"name": "Read", "type": "line", "data": [0] * 60, "smooth": True},
                {"name": "Encode", "type": "line", "data": [0] * 60, "smooth": True},
                {"name": "Send", "type": "line", "data": [0] * 60, "smooth": True},
            ],
        }

    def setup_ui(self) -> None:
        """Setup interactive testing UI."""
        ui.page_title("Jetson Camera Test")

        with ui.header().classes("items-center justify-between"):
            ui.label("Jetson Camera Performance Tester").classes("text-2xl")

        with ui.column().classes("w-full items-center gap-4 p-4"):
            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Camera Feed").classes("text-xl mb-2")
                self.video_container = ui.html(
                    """
                    <div style="position: relative; display: inline-block; margin: 0 auto; border: 2px solid #333; background: #000;">
                        <canvas id="videoCanvas" width="640" height="480" style="display: block; image-rendering: pixelated;"></canvas>
                        <div id="frameInfo"
                             style="position: absolute; top: 10px; left: 10px;
                                    color: #0f0; font-family: monospace; font-size: 12px;
                                    text-shadow: 1px 1px 2px #000; background: rgba(0,0,0,0.7);
                                    padding: 5px; border-radius: 3px;">
                            Waiting...
                        </div>
                    </div>
                    """,
                    sanitize=False,
                ).classes("w-full text-center")
                self._inject_canvas_script()

            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Camera Settings").classes("text-xl")
                ui.label("Base profile: 320x240 @ 15-30 FPS, JPEG 85").classes(
                    "text-sm text-gray-500 mb-2"
                )

                with ui.grid(columns=2).classes("gap-4 w-full"):
                    with ui.column().classes("gap-2"):
                        ui.label("Resolution")
                        self.resolution_select = ui.select(
                            options={
                                "320x240": "320x240 (QVGA)",
                                "640x480": "640x480 (VGA)",
                                "1280x720": "1280x720 (720p)",
                                "1920x1080": "1920x1080 (1080p)",
                            },
                            value="320x240",
                            on_change=self.update_resolution,
                        ).classes("w-full")

                    with ui.column().classes("gap-2"):
                        ui.label("Capture FPS")
                        self.fps_select = ui.select(
                            options={
                                "15": "15 FPS",
                                "30": "30 FPS",
                                "60": "60 FPS",
                                "120": "120 FPS",
                            },
                            value="30",
                            on_change=self.update_fps,
                        ).classes("w-full")

                    with ui.column().classes("gap-2"):
                        ui.label("Display Update FPS")
                        ui.slider(min=5, max=60, step=5, value=20, on_change=self.update_display_fps).props(
                            "label"
                        )

                    with ui.column().classes("gap-2"):
                        ui.label("JPEG Quality")
                        ui.slider(min=50, max=95, step=5, value=85, on_change=self.update_jpeg_quality).props(
                            "label"
                        )

                with ui.row().classes("gap-4 mt-4"):
                    self.start_button = ui.button("\u25b6 Start Test", on_click=self.toggle_camera).props(
                        "color=green size=lg"
                    )
                    ui.button("Reset Stats", on_click=self.reset_stats).props("color=blue")

            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Real-time Performance").classes("text-xl mb-2")
                with ui.grid(columns=4).classes("gap-4"):
                    with ui.card().tight().classes("p-4"):
                        ui.label("Capture FPS").classes("text-xs")
                        self.capture_fps_label = ui.label("0.0").classes("text-3xl font-bold text-green-500")
                    with ui.card().tight().classes("p-4"):
                        ui.label("Display FPS").classes("text-xs")
                        self.display_fps_label = ui.label("0.0").classes("text-3xl font-bold text-blue-500")
                    with ui.card().tight().classes("p-4"):
                        ui.label("Total / Sent").classes("text-xs")
                        self.frames_label = ui.label("0 / 0").classes("text-3xl font-bold text-orange-500")
                    with ui.card().tight().classes("p-4"):
                        ui.label("Dropped").classes("text-xs")
                        self.dropped_label = ui.label("0").classes("text-3xl font-bold text-red-500")

                with ui.grid(columns=3).classes("gap-4 mt-4"):
                    with ui.card().tight().classes("p-4"):
                        ui.label("Avg Read Time").classes("text-xs")
                        self.read_time_label = ui.label("0.0 ms").classes("text-2xl font-bold")
                    with ui.card().tight().classes("p-4"):
                        ui.label("Avg Encode Time").classes("text-xs")
                        self.encode_time_label = ui.label("0.0 ms").classes("text-2xl font-bold")
                    with ui.card().tight().classes("p-4"):
                        ui.label("Avg Send Time").classes("text-xs")
                        self.send_time_label = ui.label("0.0 ms").classes("text-2xl font-bold")

            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Processing Time Chart (ms)").classes("text-xl mb-2")
                self.chart = ui.echart(self.chart_option).classes("w-full h-64")

            with ui.card().classes("w-full max-w-4xl"):
                ui.label("Status").classes("text-xl mb-2")
                self.status_label = ui.label("Ready to start").classes("text-lg")

    def _inject_canvas_script(self) -> None:
        ui.add_body_html(
            """
            <script>
                window.updateVideoFrame = function(base64Data) {
                    const canvas = document.getElementById('videoCanvas');
                    if (!canvas) return;
                    const ctx = canvas.getContext('2d', { alpha: false });
                    const img = new Image();
                    img.onload = function() {
                        const targetWidth = img.naturalWidth;
                        const targetHeight = img.naturalHeight;
                        if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
                            canvas.width = targetWidth;
                            canvas.height = targetHeight;
                            canvas.style.width = targetWidth + 'px';
                            canvas.style.height = targetHeight + 'px';
                        }
                        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
                        const infoDiv = document.getElementById('frameInfo');
                        if (infoDiv) {
                            infoDiv.textContent = `Streaming ${targetWidth}x${targetHeight}`;
                        }
                    };
                    img.src = 'data:image/jpeg;base64,' + base64Data;
                };
            </script>
            """
        )

    def update_resolution(self, e: Any) -> None:
        width, height = map(int, e.value.split("x"))
        self.width = width
        self.height = height
        print(f"[CamUI] Resolution: {self.width}x{self.height}")

    def update_fps(self, e: Any) -> None:
        self.fps = int(e.value)
        print(f"[CamUI] Capture FPS: {self.fps}")

    def update_display_fps(self, e: Any) -> None:
        self.display_fps = int(e.value)
        self.min_send_interval = 1.0 / self.display_fps
        print(f"[CamUI] Display FPS: {self.display_fps}")

    def update_jpeg_quality(self, e: Any) -> None:
        self.jpeg_quality = int(e.value)
        print(f"[CamUI] JPEG Quality: {self.jpeg_quality}")

    def reset_stats(self) -> None:
        self.read_times.clear()
        self.encode_times.clear()
        self.send_times.clear()
        self.frame_intervals.clear()
        self.total_frames = 0
        self.sent_frames = 0
        self.dropped_frames = 0
        print("[CamUI] Statistics reset")
        ui.notify("Statistics reset", type="info")

    def toggle_camera(self) -> None:
        if not self.running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self) -> None:
        print(f"[CamUI] Starting camera: {self.width}x{self.height} @ {self.fps}fps")
        try:
            self.camera = JetCamera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                device=0,
                capture_width=self.width,
                capture_height=self.height,
                capture_fps=self.fps,
            )

            if not self.camera.start():
                raise RuntimeError("Camera start failed")

            test_frame = self.camera.read()
            if test_frame is None:
                raise RuntimeError("Camera test read failed")

            self.running = True
            self.last_frame_time = time.time()
            self.last_send_time = time.time()
            if self.start_button:
                self.start_button.props("color=red")
                self.start_button.text = "\u23f9 Stop Test"
            if self.status_label:
                self.status_label.text = f"Running: {self.width}x{self.height} @ {self.fps}fps"

            ui.notify("Camera started", type="positive")
            self.process_task = asyncio.create_task(self.capture_loop())
            print("[CamUI] \u2713 Camera started")
        except Exception as exc:  # noqa: BLE001
            print(f"[CamUI] \u2717 Failed to start: {exc}")
            ui.notify(f"Failed to start: {exc}", type="negative")
            self.stop_camera()

    def stop_camera(self) -> None:
        print("[CamUI] Stopping camera...")
        self.running = False
        if self.process_task:
            self.process_task.cancel()
            self.process_task = None
        if self.camera:
            self.camera.stop()
            self.camera = None
        if self.start_button:
            self.start_button.props("color=green")
            self.start_button.text = "\u25b6 Start Test"
        if self.status_label:
            self.status_label.text = "Stopped"
        ui.notify("Camera stopped", type="warning")
        print("[CamUI] \u2713 Camera stopped")

    async def capture_loop(self) -> None:
        print("[CamUI] Capture loop started")
        try:
            while self.running and self.camera is not None:
                read_start = time.time()
                frame = self.camera.read()
                read_time = (time.time() - read_start) * 1000.0

                if frame is None:
                    self.dropped_frames += 1
                    await asyncio.sleep(0.01)
                    continue

                self.read_times.append(read_time)
                self.total_frames += 1

                current_time = time.time()
                if self.last_frame_time > 0:
                    interval = current_time - self.last_frame_time
                    self.frame_intervals.append(interval)
                self.last_frame_time = current_time

                time_since_send = current_time - self.last_send_time
                if time_since_send >= self.min_send_interval:
                    encode_start = time.time()
                    frame_bgr = frame
                    display_frame = frame_bgr
                    success, buffer = cv2.imencode(
                        ".jpg",
                        display_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                    encode_time = (time.time() - encode_start) * 1000.0

                    if success and self.video_container is not None:
                        self.encode_times.append(encode_time)
                        send_start = time.time()
                        img_base64 = base64.b64encode(buffer).decode("utf-8")
                        await self.video_container.client.run_javascript(
                            f'window.updateVideoFrame("{img_base64}")'
                        )
                        send_time = (time.time() - send_start) * 1000.0
                        self.send_times.append(send_time)
                        self.sent_frames += 1
                        self.last_send_time = current_time

                if self.total_frames % 10 == 0:
                    self.update_stats_ui()

                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            print("[CamUI] Capture loop cancelled")
        except Exception as exc:  # noqa: BLE001
            print(f"[CamUI] Error in capture loop: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            print("[CamUI] Capture loop ended")

    def update_stats_ui(self) -> None:
        capture_fps = 0.0
        if self.frame_intervals:
            avg_interval = float(np.mean(list(self.frame_intervals)))
            capture_fps = 1.0 / avg_interval if avg_interval > 0 else 0.0

        elapsed = sum(self.frame_intervals) if self.frame_intervals else 0.0
        display_fps = self.sent_frames / elapsed if elapsed > 0 else 0.0

        avg_read = float(np.mean(list(self.read_times))) if self.read_times else 0.0
        avg_encode = float(np.mean(list(self.encode_times))) if self.encode_times else 0.0
        avg_send = float(np.mean(list(self.send_times))) if self.send_times else 0.0

        if self.capture_fps_label:
            self.capture_fps_label.text = f"{capture_fps:.1f}"
        if self.display_fps_label:
            self.display_fps_label.text = f"{display_fps:.1f}"
        if self.frames_label:
            self.frames_label.text = f"{self.total_frames} / {self.sent_frames}"
        if self.dropped_label:
            self.dropped_label.text = str(self.dropped_frames)
        if self.read_time_label:
            self.read_time_label.text = f"{avg_read:.2f} ms"
        if self.encode_time_label:
            self.encode_time_label.text = f"{avg_encode:.2f} ms"
        if self.send_time_label:
            self.send_time_label.text = f"{avg_send:.2f} ms"

        self.update_chart()

    def update_chart(self) -> None:
        read_data = list(self.read_times)[-60:]
        encode_data = list(self.encode_times)[-60:]
        send_data = list(self.send_times)[-60:]

        while len(read_data) < 60:
            read_data.insert(0, 0)
        while len(encode_data) < 60:
            encode_data.insert(0, 0)
        while len(send_data) < 60:
            send_data.insert(0, 0)

        if self.chart is None:
            return

        self.chart.options["series"][0]["data"] = read_data
        self.chart.options["series"][1]["data"] = encode_data
        self.chart.options["series"][2]["data"] = send_data
        self.chart.update()


def main() -> None:
    test_ui = CameraTestUI()
    test_ui.setup_ui()
    print("\n" + "=" * 60)
    print("Jetson Camera Test UI")
    print("Access from browser: http://<JETSON_IP>:8080")
    print("=" * 60 + "\n")
    ui.run(host="0.0.0.0", port=8080, title="Jetson Camera Test", reload=False, show=False)


if __name__ == "__main__":
    main()
