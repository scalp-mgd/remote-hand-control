"""Collect gesture training data — show hand + press number key to label."""

import csv
import os
import sys

import cv2
import yaml

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gesture_classifier import preprocess_landmarks
from core.hand_tracker import HandTracker
from overlay.hud import draw_hud


def load_labels(path: str) -> dict[int, str]:
    labels = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    labels[int(row[0])] = row[1].strip()
    return labels


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cam_cfg = config.get("camera", {})
    cap = cv2.VideoCapture(cam_cfg.get("index", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))

    ht_cfg = config.get("hand_tracking", {})
    tracker = HandTracker(
        max_hands=ht_cfg.get("max_hands", 1),
        detection_confidence=ht_cfg.get("detection_confidence", 0.7),
        tracking_confidence=ht_cfg.get("tracking_confidence", 0.5),
    )

    g_cfg = config.get("gesture", {})
    labels = load_labels(g_cfg.get("labels_path", "models/gesture_labels.csv"))
    output_path = "models/keypoints.csv"

    # Count existing samples
    existing_counts: dict[int, int] = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for row in csv.reader(f):
                if row:
                    cid = int(row[0])
                    existing_counts[cid] = existing_counts.get(cid, 0) + 1

    print("=== Gesture Data Collection ===")
    print("Show your hand and press a number key (0-9) to label the gesture.")
    print("Labels:", labels)
    print("Existing samples:", existing_counts)
    print("Press 'q' to quit.\n")

    collected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hand = tracker.process(frame)

        info_text = f"Collected: {collected}"
        if hand:
            draw_hud(frame, landmarks=hand.landmarks, show_landmarks=True)
            info_text += " | Hand detected - press 0-9 to save"
        else:
            info_text += " | No hand"

        # Show label hints at bottom
        y_pos = frame.shape[0] - 20
        for class_id, name in sorted(labels.items()):
            count = existing_counts.get(class_id, 0)
            hint = f"{class_id}: {name} ({count})"
            cv2.putText(frame, hint, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos -= 20

        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Collect Gesture Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Number keys 0-9
        if ord("0") <= key <= ord("9") and hand is not None:
            class_id = key - ord("0")
            features = preprocess_landmarks(hand.landmarks)
            row = [class_id] + features.tolist()

            with open(output_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            existing_counts[class_id] = existing_counts.get(class_id, 0) + 1
            collected += 1
            label_name = labels.get(class_id, f"class_{class_id}")
            print(f"  Saved: {label_name} (class {class_id}) — total {existing_counts[class_id]}")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Total collected this session: {collected}")
    print(f"Data saved to: {output_path}")
    print("Sample counts:", existing_counts)


if __name__ == "__main__":
    main()
