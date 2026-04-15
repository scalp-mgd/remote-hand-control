"""Gesture classifier — preprocesses keypoints and runs ONNX inference."""

import csv
import os
from typing import Optional

import numpy as np


def preprocess_landmarks(landmarks: list[tuple[float, float]]) -> np.ndarray:
    """Convert 21 landmarks to normalized 42-element feature vector.

    Approach (kinivi):
    1. Subtract wrist (landmark 0) from all points → relative coords
    2. Flatten to 42 values
    3. Normalize by max absolute value → all in [-1, 1]
    """
    wrist_x, wrist_y = landmarks[0]
    relative = []
    for x, y in landmarks:
        relative.append(x - wrist_x)
        relative.append(y - wrist_y)

    arr = np.array(relative, dtype=np.float32)
    max_val = np.max(np.abs(arr))
    if max_val > 0:
        arr /= max_val
    return arr


class GestureClassifier:
    def __init__(self, model_path: str, labels_path: str):
        self._session = None
        self._labels: dict[int, str] = {}
        self._model_path = model_path
        self._labels_path = labels_path
        self._load_labels()

    def _load_labels(self):
        if os.path.exists(self._labels_path):
            with open(self._labels_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        self._labels[int(row[0])] = row[1].strip()

    def _ensure_model(self):
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(self._model_path)

    @property
    def is_ready(self) -> bool:
        return os.path.exists(self._model_path)

    def classify(self, landmarks: list[tuple[float, float]]) -> Optional[tuple[int, str, float]]:
        """Classify hand gesture from landmarks.

        Returns (class_id, label, confidence) or None if model not available.
        """
        if not self.is_ready:
            return None

        self._ensure_model()
        features = preprocess_landmarks(landmarks)
        input_data = features.reshape(1, -1)

        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: input_data})

        logits = output[0][0]
        # softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        label = self._labels.get(class_id, f"class_{class_id}")

        return class_id, label, confidence
