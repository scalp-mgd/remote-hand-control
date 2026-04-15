"""Remote Hand Control — main application entry point."""

import logging
import signal
import sys
import threading
import time

import cv2
import yaml

from core.cursor_controller import CursorController
from core.gesture_classifier import GestureClassifier
from core.hand_tracker import HandTracker
from core.state_machine import StateMachine
from core.voice_recorder import RealtimeVoiceRecorder
from overlay.hud import draw_hud, help_tooltip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # --- Initialize components ---
    cam_cfg = config.get("camera", {})
    cap = cv2.VideoCapture(cam_cfg.get("index", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))

    if not cap.isOpened():
        logger.error("Cannot open camera")
        sys.exit(1)

    ht_cfg = config.get("hand_tracking", {})
    tracker = HandTracker(
        max_hands=ht_cfg.get("max_hands", 1),
        detection_confidence=ht_cfg.get("detection_confidence", 0.7),
        tracking_confidence=ht_cfg.get("tracking_confidence", 0.5),
    )

    g_cfg = config.get("gesture", {})
    classifier = GestureClassifier(
        model_path=g_cfg.get("model_path", "models/gesture_classifier.onnx"),
        labels_path=g_cfg.get("labels_path", "models/gesture_labels.csv"),
    )

    c_cfg = config.get("cursor", {})
    cursor = CursorController(
        smoothing_alpha=c_cfg.get("smoothing_alpha", 0.12),
        sensitivity=c_cfg.get("sensitivity", 1.5),
    )

    stt_cfg = config.get("stt", {})
    a_cfg = config.get("audio", {})
    recorder = RealtimeVoiceRecorder(
        model_path=stt_cfg.get("vosk", {}).get("model_path", "models/vosk-model-small-ru-0.22"),
        sample_rate=a_cfg.get("sample_rate", 16000),
    )

    act_cfg = config.get("actions", {})
    state_machine = StateMachine(
        cursor=cursor,
        recorder=recorder,
        action_mapping=act_cfg.get("mapping"),
        cooldown_frames=act_cfg.get("cooldown_frames", 15),
        debounce_frames=g_cfg.get("debounce_frames", 3),
        confidence_threshold=g_cfg.get("confidence_threshold", 0.8),
        pinch_threshold=act_cfg.get("pinch_threshold", 0.06),
    )

    overlay_cfg = config.get("overlay", {})

    # --- Graceful shutdown ---
    running = threading.Event()
    running.set()

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        running.clear()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # --- Main loop ---
    logger.info("Remote Hand Control started. Press 'q' to quit.")
    if not classifier.is_ready:
        logger.warning(
            "Gesture model not found at %s. "
            "Run training/collect_data.py and training/train_classifier.py first.",
            g_cfg.get("model_path"),
        )

    prev_time = time.time()
    fps = 0.0
    help_registered = False

    while running.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            break

        now = time.time()
        dt = now - prev_time
        fps = 1.0 / dt if dt > 0 else 0
        prev_time = now

        hand = tracker.process(frame)

        gesture = None
        landmarks = None
        landmarks_norm = None

        if hand is not None:
            landmarks = hand.landmarks
            landmarks_norm = hand.landmarks_normalized
            if classifier.is_ready:
                gesture = classifier.classify(landmarks)

        state_machine.update(gesture, landmarks_norm)

        gesture_label = gesture[1] if gesture else ""
        confidence = gesture[2] if gesture else 0.0

        draw_hud(
            frame,
            landmarks=landmarks,
            gesture_label=gesture_label,
            confidence=confidence,
            fps=fps,
            is_recording=recorder.is_recording,
            state=state_machine.state.value,
            show_landmarks=overlay_cfg.get("show_landmarks", True),
            status_message=state_machine.status_message,
            partial_text=recorder.partial_text,
            last_action=state_machine.last_action,
        )

        cv2.imshow("Remote Hand Control", frame)

        # Register help tooltip mouse callback (once, after window exists)
        if not help_registered:
            help_tooltip.register("Remote Hand Control")
            help_registered = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # --- Cleanup ---
    if recorder.is_recording:
        recorder.stop()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
