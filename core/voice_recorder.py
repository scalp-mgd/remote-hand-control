"""Voice recorder with real-time Vosk streaming STT.

Final text is typed word-by-word as Vosk confirms phrases (~1-2 sec).
Partial (unconfirmed) text is shown in HUD only — no flickering.
"""

import json
import logging
import os
import queue
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


def add_punctuation(text: str) -> str:
    """Add basic punctuation to raw Vosk output.

    Simple rule-based: capitalize first word, add period at end,
    add commas before conjunctions.
    """
    if not text:
        return text

    # Conjunctions that often need a comma before them
    comma_words = {
        "но", "а", "однако", "хотя", "потому", "поэтому",
        "когда", "если", "чтобы", "который", "которая", "которое", "которые",
        "что", "где", "как", "так",
    }

    words = text.split()
    result = []

    for i, word in enumerate(words):
        if i > 0 and word.lower() in comma_words:
            # Add comma before conjunction (if previous char isn't already comma)
            if result and not result[-1].endswith(","):
                result[-1] = result[-1] + ","
        result.append(word)

    text = " ".join(result)

    # Capitalize first letter
    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    # Add period at end if no punctuation
    if text and text[-1] not in ".!?":
        text += "."

    return text


class RealtimeVoiceRecorder:
    """Records audio and streams it through Vosk for real-time transcription."""

    def __init__(
        self,
        model_path: str = "models/vosk-model-small-ru-0.22",
        sample_rate: int = 16000,
    ):
        self._sample_rate = sample_rate
        self._model_path = model_path
        self._recognizer = None
        self._stream: sd.RawInputStream | None = None
        self._recording = threading.Event()
        self._audio_queue: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread | None = None

        # Text state
        self._full_text: str = ""
        self._current_partial: str = ""  # shown in HUD only, not typed

        # Load Vosk model once at startup (avoids lag on first recording)
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Vosk model not found at {self._model_path}. "
                "Download from https://alphacephei.com/vosk/models"
            )
        logger.info("Loading Vosk model from %s...", self._model_path)
        self._model = Model(self._model_path)
        logger.info("Vosk model loaded.")

    @property
    def is_recording(self) -> bool:
        return self._recording.is_set()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def partial_text(self) -> str:
        """Current unconfirmed partial text (for HUD display)."""
        return self._current_partial

    def _audio_callback(self, indata, frames, time_info, status):
        if self._recording.is_set():
            self._audio_queue.put(bytes(indata))

    def start(self):
        """Start recording and real-time transcription."""
        from vosk import KaldiRecognizer

        self._recognizer = KaldiRecognizer(self._model, self._sample_rate)
        self._recognizer.SetWords(True)
        self._full_text = ""
        self._current_partial = ""

        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()

        self._recording.set()

        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=4000,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        )
        self._stream.start()

        self._worker_thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._worker_thread.start()

        logger.info("Real-time recording started")

    def _process_loop(self):
        """Process audio chunks — type only FINAL results."""
        from core.text_inserter import insert_text

        while self._recording.is_set():
            try:
                data = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._recognizer.AcceptWaveform(data):
                # FINAL result — Vosk is confident, type it!
                result = json.loads(self._recognizer.Result())
                final_text = result.get("text", "")

                if final_text:
                    # Add basic punctuation
                    punctuated = add_punctuation(final_text)
                    insert_text(punctuated + " ")
                    self._full_text += punctuated + " "
                    self._current_partial = ""
                    logger.info("Typed: %s", punctuated)
            else:
                # PARTIAL result — show in HUD only, don't type
                partial = json.loads(self._recognizer.PartialResult())
                self._current_partial = partial.get("partial", "")

    def stop(self) -> np.ndarray:
        """Stop recording. Process remaining audio."""
        self._recording.clear()

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        # Process remaining audio
        if self._recognizer is not None:
            while not self._audio_queue.empty():
                try:
                    data = self._audio_queue.get_nowait()
                    self._recognizer.AcceptWaveform(data)
                except queue.Empty:
                    break

            from core.text_inserter import insert_text
            result = json.loads(self._recognizer.FinalResult())
            final_text = result.get("text", "")

            if final_text:
                punctuated = add_punctuation(final_text)
                insert_text(punctuated + " ")
                self._full_text += punctuated + " "
                logger.info("Typed (final): %s", punctuated)

            self._current_partial = ""

        total = self._full_text.strip()
        logger.info("Total: %s", total[:100] if total else "(empty)")

        return np.array([], dtype=np.float32)
