import pytest
from pydantic import ValidationError

from app.schemas import Segment, TranscriptionResult


def test_transcription_result_roundtrips_to_dict():
    result = TranscriptionResult(
        language="en",
        text="hello world",
        segments=[Segment(start=0.0, end=1.5, text="hello world")],
        provider="local",
        duration=1.5,
    )
    dumped = result.model_dump()
    assert dumped["provider"] == "local"
    assert dumped["segments"][0]["start"] == 0.0
    assert dumped["text"] == "hello world"


def test_segment_rejects_end_before_start():
    with pytest.raises(ValidationError):
        Segment(start=2.0, end=1.0, text="bad")
