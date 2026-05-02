"""STT transcriber — sends recorded audio to Groq Whisper large-v3-turbo.

Whisper large-v3-turbo handles slang, mixed-language phrases, and produces
output with proper punctuation. Latency is typically ~0.5–1.5s for clips
under 30s. Free tier: 30 requests/min — plenty for this use case.
"""

import io
import logging
import os
import wave

import numpy as np

logger = logging.getLogger(__name__)


# Whisper hallucinations on silence/noise — model was trained on YouTube and
# learned to emit these phrases when audio is empty or unintelligible.
_HALLUCINATIONS = {
    "продолжение следует...",
    "продолжение следует.",
    "продолжение следует",
    "субтитры сделал dimatorzok",
    "субтитры подогнал dimatorzok",
    "редактор субтитров а.семкин",
    "субтитры сделал",
    "thanks for watching!",
    "thank you for watching",
    "♪",
    "...",
}


def _is_hallucination(text: str) -> bool:
    norm = text.strip().lower()
    if norm in _HALLUCINATIONS:
        return True
    # Sometimes Whisper emits "Продолжение следует..." plus stray punctuation
    for h in _HALLUCINATIONS:
        if norm.startswith(h) and len(norm) < len(h) + 5:
            return True
    return False


def _audio_loud_enough(audio: np.ndarray, threshold: float = 0.005) -> bool:
    """Quick RMS-based VAD — skip Groq call if audio is essentially silence."""
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
    return rms >= threshold


class GroqTranscriber:
    """Lightweight Groq Whisper client."""

    def __init__(self, api_key: str | None = None, model: str = "whisper-large-v3-turbo", language: str = "ru"):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env or export it as an "
                "environment variable. Get a free key at https://console.groq.com"
            )

        # Lazy import — only loaded when transcriber is constructed.
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._model = model
        self._language = language

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a numpy float32 mono audio array to Russian text.

        Returns empty string on failure or if audio is too short.
        """
        if audio is None or audio.size == 0:
            return ""

        duration = len(audio) / sample_rate
        if duration < 0.3:
            logger.info("Audio too short (%.2fs), skipping transcription", duration)
            return ""

        if not _audio_loud_enough(audio):
            logger.info("Audio too quiet (silence), skipping transcription")
            return ""

        wav_bytes = self._to_wav_bytes(audio, sample_rate)
        logger.info("Sending %.1fs / %d KB audio to Groq (%s)...",
                    duration, len(wav_bytes) // 1024, self._model)

        try:
            result = self._client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model=self._model,
                language=self._language,
                response_format="text",
            )
            text = (result if isinstance(result, str) else getattr(result, "text", "")).strip()
            if _is_hallucination(text):
                logger.info("Filtered hallucination: %r", text)
                return ""
            logger.info("Transcribed: %s", text[:120])
            return text
        except Exception as e:
            logger.error("Groq transcription failed: %s", e)
            return ""

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        """Pack a float32 mono numpy array as 16-bit PCM WAV in-memory."""
        # Clamp + convert to int16
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()
