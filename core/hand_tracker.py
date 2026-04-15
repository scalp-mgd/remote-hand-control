"""MediaPipe Hands wrapper — detects hand landmarks from camera frames."""

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass
class HandResult:
    """Result of hand detection on a single frame."""
    landmarks: list[tuple[float, float]]  # 21 (x_px, y_px) pairs
    landmarks_normalized: list[tuple[float, float]]  # 21 (x, y) in [0,1]
    handedness: str  # "Left" or "Right"


class HandTracker:
    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
        model_path: str = "models/hand_landmarker.task",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process(self, frame: np.ndarray) -> Optional[HandResult]:
        """Process a BGR frame, return HandResult or None if no hand detected."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_timestamp_ms += 33  # ~30fps increment

        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if not result.hand_landmarks:
            return None

        hand_lms = result.hand_landmarks[0]
        handedness = result.handedness[0][0].category_name

        landmarks_px = []
        landmarks_norm = []
        for lm in hand_lms:
            landmarks_norm.append((lm.x, lm.y))
            landmarks_px.append((lm.x * w, lm.y * h))

        return HandResult(
            landmarks=landmarks_px,
            landmarks_normalized=landmarks_norm,
            handedness=handedness,
        )

    def close(self):
        self._landmarker.close()
