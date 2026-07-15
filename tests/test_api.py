import io
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import Segment, TranscriptionResult


def _client():
    return TestClient(create_app())


def test_health_ok():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_transcribe_happy_path():
    fake = TranscriptionResult(
        language="en",
        text="hello world",
        segments=[Segment(start=0.0, end=1.0, text="hello world")],
        provider="local",
        duration=1.0,
    )
    with patch("app.api.routes.TranscriptionService.transcribe", return_value=fake):
        files = {"file": ("clip.wav", io.BytesIO(b"0" * 32), "audio/wav")}
        resp = _client().post("/v1/transcribe", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "local"
    assert body["segments"][0]["end"] == 1.0


def test_transcribe_unsupported_format_returns_422():
    from app.errors import UnsupportedFormatError

    with patch(
        "app.api.routes.TranscriptionService.transcribe",
        side_effect=UnsupportedFormatError("nope"),
    ):
        files = {"file": ("notes.txt", io.BytesIO(b"hi"), "text/plain")}
        resp = _client().post("/v1/transcribe", files=files)
    assert resp.status_code == 422


def test_transcribe_all_providers_failed_returns_502():
    from app.errors import AllProvidersFailedError

    with patch(
        "app.api.routes.TranscriptionService.transcribe",
        side_effect=AllProvidersFailedError("all down"),
    ):
        files = {"file": ("clip.wav", io.BytesIO(b"0" * 32), "audio/wav")}
        resp = _client().post("/v1/transcribe", files=files)
    assert resp.status_code == 502
