"""Test audio — verify microphone works by recording and playing back."""

import numpy as np
import sounddevice as sd
import soundfile as sf


def main():
    duration = 3  # seconds
    sample_rate = 16000

    print(f"Recording {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")

    audio = audio.flatten()
    peak = np.max(np.abs(audio))
    print(f"Peak amplitude: {peak:.4f}")

    if peak < 0.01:
        print("WARNING: Very low audio level — check microphone.")
    else:
        print("Audio level looks good.")

    # Save to WAV
    out_path = "test_recording.wav"
    sf.write(out_path, audio, sample_rate)
    print(f"Saved to {out_path}")

    # Playback
    print("Playing back...")
    sd.play(audio, sample_rate)
    sd.wait()
    print("Done.")


if __name__ == "__main__":
    main()
