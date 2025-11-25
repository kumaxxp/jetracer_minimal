"""Generate a dummy ONNX segmentation model for testing.

Output: 19-class segmentation (Cityscapes format)
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


class DummySegmentationNet(nn.Module):
    """Lightweight dummy segmentation network."""

    def __init__(self, num_classes: int = 19) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, num_classes, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.encoder(x)
        output = self.decoder(features)
        return output


def main() -> None:
    print("Creating dummy segmentation model...")
    os.makedirs("models", exist_ok=True)

    model = DummySegmentationNet(num_classes=19)
    model.eval()

    dummy_input = torch.randn(1, 3, 240, 320)
    output_path = "models/dummy_segmentation_320x240.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
            do_constant_folding=True,
        )

    print(f"\u2713 Model created: {output_path}")
    print("  Input shape: (1, 3, 240, 320) - NCHW format")
    print("  Output shape: (1, 19, 240, 320)")
    print("  Classes: 19 (Cityscapes)")

    import cv2

    net = cv2.dnn.readNetFromONNX(output_path)
    if net is None:
        raise RuntimeError("Failed to load model with OpenCV DNN")
    print("\u2713 Model can be loaded by OpenCV DNN")


if __name__ == "__main__":
    main()
