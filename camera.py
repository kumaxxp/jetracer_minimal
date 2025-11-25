"""Camera module with enhanced debugging."""

from __future__ import annotations

import time

import cv2
import numpy as np
from jetcam.csi_camera import CSICamera

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    _GST_AVAILABLE = True
except Exception:  # noqa: BLE001
    Gst = None
    _GST_AVAILABLE = False


class JetCamera:
    """Wrapper for Jetson CSI Camera with debugging and fallback pipeline."""

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        fps: int = 30,
        device: int = 0,
        sensor_mode: int | None = None,
        capture_width: int | None = 1280,
        capture_height: int | None = 720,
        capture_fps: int | None = 60,
        flip_method: int = 0,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.sensor_mode = sensor_mode
        self.capture_width = capture_width or width
        self.capture_height = capture_height or height
        self.capture_fps = capture_fps or fps
        self.flip_method = flip_method
        self.camera: CSICamera | None = None
        self._gst_pipeline = None
        self._appsink = None
        self._use_fallback = False
        self.running = False
        self.frame_count = 0
        print(f"[Camera] Initializing: {width}x{height} @ {fps}fps, device={device}")

    def start(self) -> bool:
        if self.running:
            print("[Camera] Already running")
            return True
        if self._start_jetcam():
            return True
        print("[Camera] Falling back to custom GStreamer pipeline")
        return self._start_gstreamer_fallback()

    def _start_jetcam(self) -> bool:
        try:
            print("[Camera] Creating CSICamera instance...")
            self.camera = CSICamera(
                width=self.width,
                height=self.height,
                capture_fps=self.fps,
                capture_device=self.device,
            )
            print("[Camera] Waiting for camera to stabilize...")
            time.sleep(1.0)
            test_frame = self.camera.read()
            if test_frame is None:
                print("[Camera] ERROR: Test read returned None")
                return False
            print(
                f"[Camera] \u2713 Test read successful: shape={test_frame.shape}, dtype={test_frame.dtype}"
            )
            self.running = True
            self.frame_count = 0
            self._use_fallback = False
            print("[Camera] \u2713 Started successfully")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[Camera] \u2717 Failed to start: {exc}")
            import traceback

            traceback.print_exc()
            return False

    def _start_gstreamer_fallback(self) -> bool:
        if not _GST_AVAILABLE:
            print("[Camera] \u2717 PyGObject/GStreamer not available; cannot start fallback pipeline")
            return False
        self.camera = None
        pipeline_desc = self._build_gstreamer_pipeline()
        print(f"[Camera] GStreamer pipeline: {pipeline_desc}")
        try:
            pipeline = Gst.parse_launch(pipeline_desc)
        except Exception as exc:  # noqa: BLE001
            print(f"[Camera] \u2717 Failed to parse pipeline: {exc}")
            return False
        appsink = pipeline.get_by_name("jetcamera_sink")
        if appsink is None:
            print("[Camera] \u2717 appsink element not found in pipeline")
            pipeline.set_state(Gst.State.NULL)
            return False
        appsink.set_property("emit-signals", False)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
        appsink.set_property("sync", False)
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[Camera] \u2717 Failed to start GStreamer pipeline")
            pipeline.set_state(Gst.State.NULL)
            return False
        change, state, _ = pipeline.get_state(Gst.SECOND * 2)
        if change == Gst.StateChangeReturn.FAILURE or state != Gst.State.PLAYING:
            print("[Camera] \u2717 Pipeline did not reach PLAYING state")
            pipeline.set_state(Gst.State.NULL)
            return False
        self._gst_pipeline = pipeline
        self._appsink = appsink
        self._use_fallback = True
        self.running = True
        self.frame_count = 0
        print("[Camera] \u2713 Fallback pipeline running via GStreamer appsink")
        return True

    def read(self) -> np.ndarray | None:
        if not self.running:
            print("[Camera] ERROR: Camera not running")
            return None
        if self._use_fallback:
            return self._read_from_gstreamer()
        if self.camera is None:
            print("[Camera] ERROR: CSI camera instance missing")
            return None
        try:
            frame = self.camera.read()
            if frame is None:
                print(f"[Camera] WARNING: Read returned None (frame #{self.frame_count})")
                return None
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                print(f"[Camera] Frame #{self.frame_count}: shape={frame_bgr.shape}, dtype={frame_bgr.dtype}")
            return frame_bgr
        except Exception as exc:  # noqa: BLE001
            print(f"[Camera] ERROR reading frame: {exc}")
            return None

    def stop(self) -> None:
        if self.camera is not None and not self._use_fallback:
            try:
                self.camera.release()
                print(f"[Camera] \u2713 Stopped (total frames: {self.frame_count})")
            except Exception as exc:  # noqa: BLE001
                print(f"[Camera] Error stopping: {exc}")
        if self._gst_pipeline is not None:
            self._gst_pipeline.set_state(Gst.State.NULL)
            self._gst_pipeline = None
            self._appsink = None
            print(f"[Camera] \u2713 Fallback pipeline stopped (total frames: {self.frame_count})")
        self._use_fallback = False
        self.running = False
        self.camera = None

    def __enter__(self) -> "JetCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    def _build_gstreamer_pipeline(self) -> str:
        sensor_mode = f" sensor-mode={self.sensor_mode}" if self.sensor_mode is not None else ""
        return (
            "nvarguscamerasrc sensor-id={sensor_id}{sensor_mode} ! "
            "video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, format=(string)NV12, framerate=(fraction){capture_fps}/1 ! "
            "{nvvidconv}"
            "video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink name=jetcamera_sink"
        ).format(
            sensor_id=self.device,
            sensor_mode=sensor_mode,
            capture_fps=self.capture_fps,
            nvvidconv=self._nvvidconv_segment(),
            capture_width=self.capture_width,
            capture_height=self.capture_height,
            width=self.width,
            height=self.height,
        )

    def _nvvidconv_segment(self) -> str:
        if self.flip_method:
            return f"nvvidconv flip-method={self.flip_method} ! "
        return "nvvidconv ! "

    def _read_from_gstreamer(self) -> np.ndarray | None:
        if self._appsink is None:
            print("[Camera] ERROR: GStreamer appsink not initialized")
            return None
        timeout_ns = int(1e9 / max(self.fps, 1))
        sample = self._appsink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            print(f"[Camera] WARNING: No sample received (frame #{self.frame_count})")
            return None
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            print("[Camera] WARNING: Unable to map GStreamer buffer")
            return None
        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((self.height, self.width, 3)).copy()
        except ValueError as exc:
            print(f"[Camera] WARNING: Invalid frame shape: {exc}")
            frame = None
        finally:
            buffer.unmap(map_info)
        del sample
        if frame is not None:
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                print(f"[Camera] Frame #{self.frame_count}: shape={frame.shape}, dtype={frame.dtype}")
        return frame
