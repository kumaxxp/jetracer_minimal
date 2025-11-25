"""Performance monitoring utilities."""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional

import psutil


class PerformanceMonitor:
    """Monitor FPS and system metrics."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        self.inference_times: Deque[float] = deque(maxlen=window_size)
        self.last_update_time = time.time()
        self.frame_count = 0

    def update(self, inference_time: Optional[float] = None) -> None:
        current_time = time.time()
        if self.frame_count > 0:
            frame_time = current_time - self.last_update_time
            self.frame_times.append(frame_time)
        self.last_update_time = current_time
        self.frame_count += 1
        if inference_time is not None:
            self.inference_times.append(inference_time)

    def get_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_avg_inference_time(self) -> float:
        if not self.inference_times:
            return 0.0
        return (sum(self.inference_times) / len(self.inference_times)) * 1000.0

    def get_stats(self) -> dict:
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "fps": self.get_fps(),
            "inference_ms": self.get_avg_inference_time(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_mb": memory_info.rss / 1024 / 1024,
            "frame_count": self.frame_count,
        }

    def print_stats(self) -> None:
        stats = self.get_stats()
        print(
            f"FPS: {stats['fps']:.1f} | "
            f"Inference: {stats['inference_ms']:.1f}ms | "
            f"CPU: {stats['cpu_percent']:.1f}% | "
            f"Memory: {stats['memory_mb']:.0f}MB | "
            f"Frames: {stats['frame_count']}"
        )
