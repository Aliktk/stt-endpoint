import pytest

from app.errors import AllProvidersFailedError, ProviderError
from app.providers.base import TranscriptionProvider
from app.schemas import Segment, TranscriptionResult
from app.services.transcription_service import TranscriptionService


class _Stub(TranscriptionProvider):
    def __init__(self, name, fail=False):
        self.name = name
        self._fail = fail

    def is_available(self):
        return True

    def transcribe(self, audio_path, language=None):
        if self._fail:
            raise ProviderError(f"{self.name} boom")
        return TranscriptionResult(
            language="en",
            text=self.name,
            segments=[Segment(start=0.0, end=1.0, text=self.name)],
            provider=self.name,
            duration=1.0,
        )


def _audio(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    return audio


def test_service_falls_back_to_next_provider(tmp_path):
    service = TranscriptionService(
        providers=[_Stub("elevenlabs", fail=True), _Stub("local")],
        chunk_threshold_seconds=600,
        max_upload_mb=100,
    )
    assert service.transcribe(_audio(tmp_path)).provider == "local"


def test_service_raises_when_all_fail(tmp_path):
    service = TranscriptionService(
        providers=[_Stub("elevenlabs", fail=True), _Stub("local", fail=True)],
        chunk_threshold_seconds=600,
        max_upload_mb=100,
    )
    with pytest.raises(AllProvidersFailedError):
        service.transcribe(_audio(tmp_path))


def test_service_raises_with_no_providers(tmp_path):
    service = TranscriptionService(providers=[], chunk_threshold_seconds=600, max_upload_mb=100)
    with pytest.raises(AllProvidersFailedError):
        service.transcribe(_audio(tmp_path))
