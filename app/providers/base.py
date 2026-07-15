from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas import TranscriptionResult


class TranscriptionProvider(ABC):
    """Contract every speech-to-text backend implements.

    A provider owns exactly one thing: turning an audio file into a normalized
    TranscriptionResult. Availability is decoupled from transcription so the
    fallback chain can skip a backend whose credentials are absent without
    treating that as an error.
    """

    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is usable right now (key present, model loadable)."""

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        """Transcribe an audio file into a normalized result."""
