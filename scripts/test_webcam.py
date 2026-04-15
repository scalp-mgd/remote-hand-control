"""Test webcam — verify camera works and show FPS."""

import time
import cv2


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    print("Camera opened. Press 'q' to quit.")
    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        now = time.time()
        fps = 1.0 / (now - prev) if (now - prev) > 0 else 0
        prev = now

        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
