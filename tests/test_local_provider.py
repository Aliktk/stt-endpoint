from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.providers.local_whisper_provider import LocalWhisperProvider


def _fake_segments():
    return [
        SimpleNamespace(start=0.0, end=1.2, text=" hello"),
        SimpleNamespace(start=1.3, end=2.0, text=" world"),
    ]


@patch("app.providers.local_whisper_provider.WhisperModel")
def test_local_provider_normalizes_output(mock_model_cls):
    model = MagicMock()
    info = SimpleNamespace(language="en", duration=2.0)
    model.transcribe.return_value = (_fake_segments(), info)
    mock_model_cls.return_value = model

    provider = LocalWhisperProvider(model_size="base", compute_type="int8")
    result = provider.transcribe(Path("x.wav"))

    assert result.provider == "local"
    assert result.language == "en"
    assert result.text == "hello world"
    assert len(result.segments) == 2
    assert result.segments[0].start == 0.0


def test_local_provider_always_available():
    provider = LocalWhisperProvider(model_size="base", compute_type="int8")
    assert provider.is_available() is True
