from app.schemas import Segment, TranscriptionResult
from app.services.audio import merge_chunk_results


def _result(segments, duration):
    return TranscriptionResult(
        language="en",
        text=" ".join(s.text for s in segments),
        segments=segments,
        provider="local",
        duration=duration,
    )


def test_merge_offsets_timestamps_across_chunks():
    chunk_a = _result([Segment(start=0.0, end=2.0, text="first")], duration=2.0)
    chunk_b = _result([Segment(start=0.0, end=1.5, text="second")], duration=1.5)

    merged = merge_chunk_results([chunk_a, chunk_b], offsets=[0.0, 2.0])

    assert merged.text == "first second"
    assert len(merged.segments) == 2
    assert merged.segments[1].start == 2.0
    assert merged.segments[1].end == 3.5
    assert merged.duration == 3.5


def test_merge_single_chunk_is_unchanged():
    chunk = _result([Segment(start=0.0, end=1.0, text="solo")], duration=1.0)
    merged = merge_chunk_results([chunk], offsets=[0.0])
    assert merged.segments[0].start == 0.0
    assert merged.duration == 1.0
