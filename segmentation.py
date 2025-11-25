"""ONNX Runtime based semantic segmentation helper."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

LOGGER = logging.getLogger(__name__)


class SegmentationModel:
    """Loads an ONNX model and produces binary masks for drivable areas."""

    def __init__(
        self,
        model_path: str,
        input_size: Sequence[int],
        road_classes: Sequence[int],
        threshold: float = 0.5,
        providers: Sequence[str | dict] | None = None,
    ) -> None:
        if len(input_size) != 2:
            raise ValueError("input_size must be [width, height]")
        self.input_width, self.input_height = int(input_size[0]), int(input_size[1])
        self.road_classes = tuple(int(idx) for idx in road_classes)
        self.threshold = float(threshold)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.providers = list(providers) if providers else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        LOGGER.info("Loading ONNX model from %s", self.model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self.providers,
            sess_options=sess_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits, axis=0, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        return exp / np.sum(exp, axis=0, keepdims=True)

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        input_tensor = self._preprocess(frame)
        start = time.perf_counter()
        logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0][0]
        latency = time.perf_counter() - start
        probs = self._softmax(logits)
        road_probs = probs[self.road_classes].max(axis=0)
        mask = (road_probs >= self.threshold).astype(np.uint8)
        return mask, road_probs, latency
