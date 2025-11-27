"""Utility to download and export segmentation networks for Jetson deployment.

This script focuses on higher-quality models that still fit the 15 FPS
processing target described in Phase 1 of the segmentation rollout.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torchvision


def _export_model(model: torch.nn.Module, output_path: Path) -> None:
    """Export a torchvision segmentation model to ONNX and verify via OpenCV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, 480, 640)

    print("Exporting to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
            do_constant_folding=True,
        )

    print(f"\u2713 Model exported: {output_path}")

    import cv2  # Lazy import so OpenCV is only required when verifying.

    net = cv2.dnn.readNetFromONNX(str(output_path))
    _ = net
    print("\u2713 ONNX model verified with OpenCV DNN")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Model size: {size_mb:.1f} MB")


def download_deeplabv3_resnet50() -> Path:
    """Download DeepLabV3-ResNet50 (accuracy priority)."""
    print("Downloading DeepLabV3-ResNet50 (Cityscapes pretrain not available, using COCO weights)...")

    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights="COCO_WITH_VOC_LABELS_V1"
    )
    model.eval()

    output_path = Path("models/deeplabv3_resnet50_640x480.onnx")
    _export_model(model, output_path)
    return output_path


def download_mobilenetv2_deeplabv3() -> Path:
    """Download DeepLabV3-MobileNetV2 (speed priority)."""
    print("Downloading DeepLabV3-MobileNetV2 (torchvision MobileNetV3 Large weights)...")

    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights="COCO_WITH_VOC_LABELS_V1"
    )
    model.eval()

    output_path = Path("models/mobilenetv2_deeplabv3_640x480.onnx")
    _export_model(model, output_path)
    return output_path


def main() -> None:
    """Download both reference models for benchmarking."""
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("Segmentation Model Download")
    print("Target: 15 FPS overall processing (30-40 ms inference budget)")
    print("=" * 60 + "\n")

    resnet_path = None
    mobile_path = None

    try:
        resnet_path = download_deeplabv3_resnet50()
    except Exception as exc:  # noqa: BLE001 we want to report and keep going
        print(f"\n\u2717 ResNet50 export failed: {exc}")

    print("\n" + "-" * 60 + "\n")

    try:
        mobile_path = download_mobilenetv2_deeplabv3()
    except Exception as exc:  # noqa: BLE001
        print(f"\n\u2717 MobileNet export failed: {exc}")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Benchmark both models: python3 benchmark_segmentation.py")
    print("2. Choose model based on inference time vs. 15 FPS budget")
    print("3. Update config.yaml with the chosen model path and interval")
    print("=" * 60)

    if resnet_path:
        print(f"\n\u2713 ResNet50 model ready: {resnet_path}")
    if mobile_path:
        print(f"\u2713 MobileNetV3 model ready: {mobile_path}")


if __name__ == "__main__":
    main()
