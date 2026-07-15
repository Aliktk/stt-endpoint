from pathlib import Path

from faster_whisper import WhisperModel

from app.errors import ProviderError
from app.providers.base import TranscriptionProvider
from app.schemas import Segment, TranscriptionResult


class LocalWhisperProvider(TranscriptionProvider):
    """Offline fallback backed by faster-whisper. Needs no API key.

    The model is loaded lazily on first use so importing the provider (and
    listing availability) stays cheap.
    """

    name = "local"

    def __init__(self, model_size: str, compute_type: str) -> None:
        self._model_size = model_size
        self._compute_type = compute_type
        self._model: WhisperModel | None = None

    def is_available(self) -> bool:
        return True

    def _load(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self._model_size, device="cpu", compute_type=self._compute_type
            )
        return self._model

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        try:
            segments_iter, info = self._load().transcribe(
                str(audio_path), vad_filter=True, language=language
            )
            segments = [
                Segment(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip())
                for s in segments_iter
            ]
        except Exception as exc:  # noqa: BLE001 - normalize any engine failure
            raise ProviderError(f"local whisper failed: {exc}") from exc

        return TranscriptionResult(
            language=info.language,
            text=" ".join(s.text for s in segments).strip(),
            segments=segments,
            provider=self.name,
            duration=round(getattr(info, "duration", 0.0), 2),
        )
