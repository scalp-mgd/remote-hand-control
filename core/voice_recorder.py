"""Voice recorder + Groq Whisper transcription.

Flow:
  1. Finger-up gesture -> start() -> sounddevice records into a buffer
  2. Gesture changes / hand lost -> stop() returns the recorded audio
  3. main.py hands the audio to GroqTranscriber.transcribe() in a worker
     thread (so UI stays responsive while waiting ~0.5-1.5s for Groq)
  4. Returned text is pasted via text_inserter.

Whisper large-v3-turbo handles slang, mixed-language phrases, names, and
proper punctuation by itself — no rule-based post-processing needed.
"""

import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class VoiceRecorder:
    """Audio buffer recorder. Pure capture — transcription happens elsewhere."""

    def __init__(self, sample_rate: int = 16000):
        self._sample_rate = sample_rate
        self._stream: Optional[sd.RawInputStream] = None
        self._recording = threading.Event()
        self._chunks: list = []
        self._chunks_lock = threading.Lock()

        # State exposed to the HUD
        self._transcribing: bool = False  # True while waiting for Groq response

    @property
    def is_recording(self) -> bool:
        return self._recording.is_set()

    @property
    def is_transcribing(self) -> bool:
        return self._transcribing

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def partial_text(self) -> str:
        # Kept for backwards compat with HUD code. Always empty in Groq mode.
        return ""

    def set_transcribing(self, value: bool):
        self._transcribing = value

    def _audio_callback(self, indata, frames, time_info, status):
        if self._recording.is_set():
            # indata is a (frames, channels) int16 numpy array (RawInputStream gives bytes,
            # InputStream gives ndarray). We use InputStream for nicer typing.
            with self._chunks_lock:
                self._chunks.append(indata.copy())

    def start(self):
        """Begin recording into the in-memory buffer."""
        if self._recording.is_set():
            logger.warning("start() called while already recording")
            return

        with self._chunks_lock:
            self._chunks.clear()

        self._recording.set()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=2048,
            dtype="float32",
            channels=1,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Recording started")

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as float32 mono."""
        if not self._recording.is_set():
            return np.array([], dtype=np.float32)

        self._recording.clear()

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Error closing stream: %s", e)
            self._stream = None

        with self._chunks_lock:
            if not self._chunks:
                logger.info("Recording stopped (no audio captured)")
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._chunks, axis=0).flatten().astype(np.float32)
            self._chunks.clear()

        duration = len(audio) / self._sample_rate
        logger.info("Recording stopped (%.2fs captured)", duration)
        return audio
