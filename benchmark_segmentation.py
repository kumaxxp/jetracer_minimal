"""Benchmark ONNX segmentation models with the 15 FPS target in mind."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from segmentation import SegmentationModel


def benchmark_model(model_path: str, num_frames: int = 100, target_fps: int = 15) -> dict[str, float]:
    """Benchmark segmentation inference latency and recommend an interval."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)

    print("\n" + "=" * 60)
    print(f"Benchmarking: {path}")
    print(f"Target: {target_fps} FPS overall")
    print(f"Test frames: {num_frames}")
    print("=" * 60 + "\n")

    frame_budget_ms = 1000.0 / target_fps
    print(f"Time budget per frame: {frame_budget_ms:.1f} ms")
    print("  Camera read: ~9 ms")
    print("  JPEG encode: ~14 ms")
    print(f"  Available for segmentation: ~{frame_budget_ms - 23:.1f} ms\n")

    print("Loading model...")
    model = SegmentationModel(str(path), input_size=(640, 480))

    test_images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(10)
    ]

    times: list[float] = []
    print("Running inference...\n")

    for i in range(num_frames):
        img = test_images[i % len(test_images)]
        start = time.perf_counter()
        _mask = model.inference(img)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

        if (i + 1) % 20 == 0:
            avg_so_far = float(np.mean(times))
            print(f"  {i + 1}/{num_frames} frames | Avg: {avg_so_far:.2f} ms")

    times_np = np.array(times)
    mean_ms = float(times_np.mean())
    median_ms = float(np.median(times_np))
    p95_ms = float(np.percentile(times_np, 95))

    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Mean:       {mean_ms:.2f} ms")
    print(f"  Median:     {median_ms:.2f} ms")
    print(f"  95th %ile:  {p95_ms:.2f} ms")
    print(f"  Std Dev:    {times_np.std():.2f} ms")
    print(f"  Min/Max:    {times_np.min():.2f} / {times_np.max():.2f} ms")
    pure_fps = 1000.0 / mean_ms
    print(f"  Pure FPS:   {pure_fps:.1f} (segmentation only)")
    print("=" * 60 + "\n")

    total_processing = 23 + mean_ms
    achievable_fps = 1000.0 / total_processing

    print("Analysis for 15 FPS Target:")
    print(f"  Total processing time: {total_processing:.1f} ms")
    print(f"  Achievable FPS: {achievable_fps:.1f}\n")

    if mean_ms < 25:
        print("\u2713\u2713 EXCELLENT: Run segmentation every frame")
        print(f"   Expected FPS: {achievable_fps:.1f} (exceeds 15 FPS target)")
        recommended_interval = 1
    elif mean_ms < 40:
        print("\u2713 GOOD: Run segmentation every frame")
        print(f"   Expected FPS: {achievable_fps:.1f} (meets 15 FPS target)")
        recommended_interval = 1
    elif mean_ms < 60:
        print("\u26a0 ACCEPTABLE: Run every 2 frames")
        print("   Segmentation: 7.5 Hz; Display: 15 FPS (smooth)")
        recommended_interval = 2
    elif mean_ms < 80:
        print("\u26a0 SLOW: Run every 3 frames")
        print("   Segmentation: 5 Hz; Display: 15 FPS (smooth)")
        recommended_interval = 3
    else:
        print("\u2717 TOO SLOW: Consider lighter model")
        recommended_interval = 5

    print("\nRecommended config.yaml setting:")
    print("  segmentation:")
    print(f"    interval: {recommended_interval}")
    print("=" * 60)

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "recommended_interval": recommended_interval,
    }


def compare_models() -> None:
    """Benchmark known model paths if present."""
    candidates = [
        "models/deeplabv3_resnet50_640x480.onnx",
        "models/mobilenetv2_deeplabv3_640x480.onnx",
    ]

    results: dict[str, dict[str, float]] = {}

    for candidate in candidates:
        if Path(candidate).exists():
            results[candidate] = benchmark_model(candidate)
        else:
            print(f"\n\u26a0 Model not found: {candidate}")

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        for model_file, stats in results.items():
            name = Path(model_file).stem
            print(f"\n{name}:")
            print(f"  Mean inference: {stats['mean_ms']:.2f} ms")
            print(f"  Recommended interval: {stats['recommended_interval']}")
        print("\nRecommendation: Favor the model meeting the 15 FPS target with best accuracy.")


def main() -> None:
    if len(sys.argv) > 1:
        benchmark_model(sys.argv[1])
    else:
        compare_models()


if __name__ == "__main__":
    main()
