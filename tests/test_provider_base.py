from pathlib import Path

import pytest

from app.providers.base import TranscriptionProvider
from app.schemas import Segment, TranscriptionResult


class FakeProvider(TranscriptionProvider):
    name = "fake"

    def is_available(self) -> bool:
        return True

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(
            language=language or "en",
            text="ok",
            segments=[Segment(start=0.0, end=1.0, text="ok")],
            provider=self.name,
            duration=1.0,
        )


def test_provider_contract_is_usable():
    provider = FakeProvider()
    result = provider.transcribe(Path("x.wav"))
    assert result.provider == "fake"
    assert provider.is_available() is True


def test_abstract_provider_cannot_instantiate():
    with pytest.raises(TypeError):
        TranscriptionProvider()  # type: ignore[abstract]
