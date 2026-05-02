"""Download required models for Remote Hand Control.

Speech recognition is now done via Groq Whisper cloud API — no local STT model
to download. Just need the MediaPipe hand landmarker and your trained gesture
classifier.
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODELS = {
    "hand_landmarker.task": {
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "size_mb": 12,
    },
}


def download_file(url: str, dest: str, desc: str = ""):
    print(f"  Downloading {desc or os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"  Done ({size_mb:.1f} MB)")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    task_path = os.path.join(MODELS_DIR, "hand_landmarker.task")
    if not os.path.exists(task_path):
        print("[1/1] MediaPipe Hand Landmarker")
        download_file(MODELS["hand_landmarker.task"]["url"], task_path)
    else:
        print("[1/1] hand_landmarker.task — already exists")

    print()
    print("All models ready!")
    print()
    print("Next steps:")
    print("  1. Copy .env.example to .env and add your GROQ_API_KEY")
    print("     (get a free key at https://console.groq.com)")
    print("  2. Collect gesture data:  python training/collect_data.py")
    print("  3. Train classifier:      python training/train_classifier.py")
    print("  4. Run the app:           python main.py")


if __name__ == "__main__":
    main()
