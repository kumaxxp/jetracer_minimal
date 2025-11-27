"""Performance monitoring module."""
import time
from collections import deque


class PerformanceMonitor:
    """Track inference timings and FPS with a rolling window."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.total_frames = 0
        self.start_time = time.time()

    def update(self, inference_time: float) -> None:
        """Record the inference time for the latest frame."""
        self.inference_times.append(inference_time)
        self.total_frames += 1

    def get_avg_inference_time(self) -> float:
        """Return rolling average inference time in milliseconds."""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times) * 1000

    def get_fps(self) -> float:
        """Calculate frames per second since the monitor started."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.total_frames / elapsed

    def reset(self) -> None:
        """Clear recorded metrics and restart the timer."""
        self.inference_times.clear()
        self.total_frames = 0
        self.start_time = time.time()
