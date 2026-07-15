from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.providers.elevenlabs_provider import ElevenLabsProvider


def _fake_response():
    words = [
        SimpleNamespace(text="hello", start=0.0, end=0.4, type="word"),
        SimpleNamespace(text="world", start=0.5, end=0.9, type="word"),
    ]
    return SimpleNamespace(language_code="en", text="hello world", words=words)


@patch("app.providers.elevenlabs_provider.ElevenLabs")
def test_elevenlabs_groups_words_into_segments(mock_client_cls):
    client = MagicMock()
    client.speech_to_text.convert.return_value = _fake_response()
    mock_client_cls.return_value = client

    provider = ElevenLabsProvider(api_key="k")
    with patch.object(Path, "open"):
        result = provider.transcribe(Path("x.mp3"))

    assert result.provider == "elevenlabs"
    assert result.language == "en"
    assert result.text == "hello world"
    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 0.9


@patch("app.providers.elevenlabs_provider.ElevenLabs")
def test_elevenlabs_omits_language_code_when_auto_detecting(mock_client_cls):
    client = MagicMock()
    client.speech_to_text.convert.return_value = _fake_response()
    mock_client_cls.return_value = client

    provider = ElevenLabsProvider(api_key="k")
    with patch.object(Path, "open"):
        provider.transcribe(Path("x.mp3"), language=None)

    # Scribe rejects an empty language_code, so it must not be sent at all.
    assert "language_code" not in client.speech_to_text.convert.call_args.kwargs


@patch("app.providers.elevenlabs_provider.ElevenLabs")
def test_elevenlabs_maps_iso_639_1_to_639_3(mock_client_cls):
    client = MagicMock()
    client.speech_to_text.convert.return_value = _fake_response()
    mock_client_cls.return_value = client

    provider = ElevenLabsProvider(api_key="k")
    with patch.object(Path, "open"):
        provider.transcribe(Path("x.mp3"), language="en")

    assert client.speech_to_text.convert.call_args.kwargs["language_code"] == "eng"


def test_elevenlabs_unavailable_without_key():
    assert ElevenLabsProvider(api_key=None).is_available() is False
