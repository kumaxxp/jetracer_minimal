"""Segmentation utilities implemented with OpenCV DNN."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


class SegmentationModel:
    """Load ONNX model with OpenCV DNN and produce binary masks."""

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (320, 240),
        road_classes: Iterable[int] | None = None,
    ) -> None:
        self.input_width, self.input_height = input_size
        self.road_classes = tuple(road_classes or (0, 1))
        if not self.road_classes:
            raise ValueError("road_classes must not be empty")
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Loading segmentation model from {path}...")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self._configure_backend()
        print("\u2713 Using OpenCV DNN backend")

    def _configure_backend(self) -> None:
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("\u2713 Using CUDA backend")
        except Exception:
            print("Using default CPU backend")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.ndim != 3:
            raise ValueError("image must be a BGR numpy array with shape (H, W, 3)")
        resized = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(rgb, scalefactor=1.0 / 255.0, size=(self.input_width, self.input_height))
        return blob

    def inference(self, image: np.ndarray) -> np.ndarray:
        blob = self.preprocess(image)
        self.net.setInput(blob)
        output = self.net.forward()
        return self.postprocess(output)

    def postprocess(self, output: np.ndarray) -> np.ndarray:
        if output.ndim == 4:
            logits = output[0]
        else:
            logits = output
        class_map = np.argmax(logits, axis=0)
        mask = np.isin(class_map, self.road_classes).astype(np.uint8)
        return mask * 255
