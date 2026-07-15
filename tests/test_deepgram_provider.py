from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.providers.deepgram_provider import DeepgramProvider


def _fake_response():
    utterances = [
        SimpleNamespace(transcript="hello world", start=0.0, end=1.0),
        SimpleNamespace(transcript="again", start=2.0, end=2.5),
    ]
    channel = SimpleNamespace(detected_language="en")
    results = SimpleNamespace(utterances=utterances, channels=[channel])
    return SimpleNamespace(results=results)


@patch("app.providers.deepgram_provider.DeepgramClient")
def test_deepgram_reads_utterances(mock_client_cls):
    client = MagicMock()
    client.listen.prerecorded.v.return_value.transcribe_file.return_value = _fake_response()
    mock_client_cls.return_value = client

    provider = DeepgramProvider(api_key="k")
    with patch.object(Path, "read_bytes", return_value=b"audio"):
        result = provider.transcribe(Path("x.mp3"))

    assert result.provider == "deepgram"
    assert result.language == "en"
    assert result.text == "hello world again"
    assert len(result.segments) == 2
    assert result.segments[1].start == 2.0


def test_deepgram_unavailable_without_key():
    assert DeepgramProvider(api_key=None).is_available() is False
