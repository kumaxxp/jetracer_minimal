"""Comprehensive camera benchmark tool for Jetson."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from camera import JetCamera


class CameraBenchmark:
    """Benchmark camera performance across different modes."""

    SENSOR_MODES = [
        (1920, 1080, 30, None, "1080p @ 30fps"),
        (1640, 1232, 30, 3, "1640x1232 @ 30fps"),
        (1280, 720, 60, None, "720p @ 60fps"),
        (640, 480, 30, None, "VGA @ 30fps"),
        (320, 240, 30, None, "QVGA @ 30fps"),
    ]

    def __init__(self, device: int = 0) -> None:
        self.device = device
        self.results: list[dict[str, object]] = []

    def test_mode(
        self,
        width: int,
        height: int,
        fps: int,
        sensor_mode: int | None = None,
        duration: float = 5.0,
        save_sample: bool = False,
    ) -> dict[str, object]:
        """Test a specific camera mode and return metrics."""
        print("\n" + "=" * 60)
        print(f"Testing: {width}x{height} @ {fps}fps")
        if sensor_mode is not None:
            print(f"Sensor Mode: {sensor_mode}")
        print("=" * 60)

        result: dict[str, object] = {
            "width": width,
            "height": height,
            "fps": fps,
            "sensor_mode": sensor_mode,
            "success": False,
            "actual_fps": 0.0,
            "avg_read_time_ms": 0.0,
            "std_read_time_ms": 0.0,
            "max_read_time_ms": 0.0,
            "avg_encode_time_ms": 0.0,
            "std_encode_time_ms": 0.0,
            "total_frames": 0,
            "dropped_frames": 0,
        }

        camera: JetCamera | None = None
        try:
            print("Initializing camera (JetCamera wrapper)...")
            camera = JetCamera(
                width=width,
                height=height,
                fps=fps,
                device=self.device,
                sensor_mode=sensor_mode,
                capture_width=width,
                capture_height=height,
                capture_fps=fps,
            )

            if not camera.start():
                print("\u2717 Failed to start camera")
                return result

            test_frame = camera.read()
            if test_frame is None:
                print("\u2717 Failed to read test frame")
                camera.stop()
                return result

            print(f"\u2713 Camera initialized: {test_frame.shape}, {test_frame.dtype}")
            print(f"Running benchmark for {duration} seconds...")

            read_times: list[float] = []
            encode_times: list[float] = []
            frame_count = 0
            dropped = 0
            start_time = time.time()

            while time.time() - start_time < duration:
                read_start = time.time()
                frame = camera.read()
                read_time = time.time() - read_start

                if frame is None:
                    dropped += 1
                    continue

                read_times.append(read_time * 1000.0)

                encode_start = time.time()
                frame_bgr = frame
                _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                encode_time = time.time() - encode_start
                encode_times.append(encode_time * 1000.0)

                frame_count += 1

                if save_sample and frame_count == 30:
                    output_dir = Path("camera_test_samples")
                    output_dir.mkdir(exist_ok=True)
                    filename = f"sample_{width}x{height}_{fps}fps.jpg"
                    cv2.imwrite(str(output_dir / filename), frame_bgr)
                    print(f"  Sample saved: {filename}")

                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0.0
                    print(f"  Progress: {elapsed:.1f}s, {frame_count} frames, {current_fps:.1f} FPS")

            total_time = time.time() - start_time
            if frame_count > 0:
                result.update(
                    {
                        "success": True,
                        "actual_fps": frame_count / total_time,
                        "avg_read_time_ms": float(np.mean(read_times)),
                        "std_read_time_ms": float(np.std(read_times)),
                        "max_read_time_ms": float(np.max(read_times)),
                        "avg_encode_time_ms": float(np.mean(encode_times)),
                        "std_encode_time_ms": float(np.std(encode_times)),
                        "total_frames": frame_count,
                        "dropped_frames": dropped,
                    }
                )

                print("\n\u2713 Test Complete:")
                print(f"  Total Frames: {frame_count}")
                print(f"  Dropped Frames: {dropped}")
                print(f"  Actual FPS: {result['actual_fps']:.2f}")
                print(
                    "  Avg Read Time: "
                    f"{result['avg_read_time_ms']:.2f}ms (\u00b1{result['std_read_time_ms']:.2f}ms)"
                )
                print(f"  Max Read Time: {result['max_read_time_ms']:.2f}ms")
                print(
                    "  Avg Encode Time: "
                    f"{result['avg_encode_time_ms']:.2f}ms (\u00b1{result['std_encode_time_ms']:.2f}ms)"
                )
                print(
                    "  Total Processing: "
                    f"{result['avg_read_time_ms'] + result['avg_encode_time_ms']:.2f}ms/frame"
                )

        except Exception as exc:  # noqa: BLE001
            print(f"\u2717 Error: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            if camera is not None:
                try:
                    camera.stop()
                except Exception:  # noqa: BLE001
                    pass

        return result

    def run_all_tests(self, duration: float = 5.0, save_samples: bool = True) -> None:
        """Run tests for all predefined modes."""
        print("\n" + "=" * 60)
        print("JETSON CAMERA COMPREHENSIVE BENCHMARK")
        print("=" * 60)

        for width, height, fps, sensor_mode, desc in self.SENSOR_MODES:
            print(f"\n--- {desc} ---")
            result = self.test_mode(
                width,
                height,
                fps,
                sensor_mode=sensor_mode,
                duration=duration,
                save_sample=save_samples,
            )
            self.results.append(result)
            time.sleep(1.0)

        self.print_summary()

    def print_summary(self) -> None:
        """Print a summary of all benchmark runs."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(
            f"{'Resolution':<15} {'Target FPS':<12} {'Actual FPS':<12} "
            f"{'Read ms':<10} {'Encode ms':<10} {'Status'}"
        )
        print("-" * 60)

        for r in self.results:
            if r.get("success"):
                status = "\u2713 PASS"
                res_str = f"{r['width']}x{r['height']}"
                print(
                    f"{res_str:<15} {r['fps']:<12} {r['actual_fps']:<12.1f} "
                    f"{r['avg_read_time_ms']:<10.2f} {r['avg_encode_time_ms']:<10.2f} {status}"
                )
            else:
                res_str = f"{r['width']}x{r['height']}"
                print(f"{res_str:<15} {r['fps']:<12} {'N/A':<12} {'N/A':<10} {'N/A':<10} \u2717 FAIL")

        successful = [r for r in self.results if r.get("success")]
        if successful:
            best = max(successful, key=lambda x: x["actual_fps"])
            fastest = min(successful, key=lambda x: x["avg_read_time_ms"] + x["avg_encode_time_ms"])
            print("\nRecommendations:")
            print(f"  Best FPS: {best['width']}x{best['height']} @ {best['actual_fps']:.1f} FPS")
            print(
                "  Fastest Processing: "
                f"{fastest['width']}x{fastest['height']} "
                f"({fastest['avg_read_time_ms'] + fastest['avg_encode_time_ms']:.2f}ms/frame)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Jetson Camera Benchmark")
    parser.add_argument("--device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration per mode (seconds)")
    parser.add_argument("--no-samples", action="store_true", help="Skip saving sample frames")
    parser.add_argument("--mode", type=str, help="Specific mode WIDTHxHEIGHT@FPS")
    args = parser.parse_args()

    benchmark = CameraBenchmark(device=args.device)

    if args.mode:
        try:
            resolution, fps_str = args.mode.split("@")
            width_str, height_str = resolution.split("x")
            width, height = int(width_str), int(height_str)
            fps = int(fps_str)
        except ValueError:
            parser.error("Invalid mode format. Use WIDTHxHEIGHT@FPS")
            return

        result = benchmark.test_mode(
            width,
            height,
            fps,
            duration=args.duration,
            save_sample=not args.no_samples,
        )
        benchmark.results.append(result)
        benchmark.print_summary()
    else:
        benchmark.run_all_tests(duration=args.duration, save_samples=not args.no_samples)


if __name__ == "__main__":
    main()
