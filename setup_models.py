"""Download required models for Remote Hand Control."""

import os
import sys
import urllib.request
import zipfile

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODELS = {
    "hand_landmarker.task": {
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "size_mb": 12,
    },
    "vosk-model-small-ru-0.22.zip": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
        "size_mb": 45,
        "extract": True,
    },
}


def download_file(url: str, dest: str, desc: str = ""):
    """Download a file with progress."""
    print(f"  Downloading {desc or os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"  Done ({size_mb:.1f} MB)")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Hand landmarker
    task_path = os.path.join(MODELS_DIR, "hand_landmarker.task")
    if not os.path.exists(task_path):
        print("[1/2] MediaPipe Hand Landmarker")
        download_file(MODELS["hand_landmarker.task"]["url"], task_path)
    else:
        print("[1/2] hand_landmarker.task — already exists")

    # Vosk model
    vosk_dir = os.path.join(MODELS_DIR, "vosk-model-small-ru-0.22")
    if not os.path.exists(vosk_dir):
        print("[2/2] Vosk Russian STT model")
        zip_path = os.path.join(MODELS_DIR, "vosk-model-small-ru-0.22.zip")
        download_file(MODELS["vosk-model-small-ru-0.22.zip"]["url"], zip_path, "vosk-model-small-ru-0.22")
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(MODELS_DIR)
        os.remove(zip_path)
        print("  Done")
    else:
        print("[2/2] vosk-model-small-ru-0.22 — already exists")

    print()
    print("All models ready!")
    print()
    print("Next steps:")
    print("  1. Collect gesture data:  python training/collect_data.py")
    print("  2. Train classifier:      python training/train_classifier.py")
    print("  3. Run the app:           python main.py")


if __name__ == "__main__":
    main()
