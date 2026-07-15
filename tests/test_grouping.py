from app.providers.grouping import group_words_into_segments


def test_words_split_on_long_pause():
    words = [
        {"text": "hello", "start": 0.0, "end": 0.4},
        {"text": "world", "start": 0.5, "end": 0.9},
        {"text": "again", "start": 3.0, "end": 3.4},  # >1s gap -> new segment
    ]
    segments = group_words_into_segments(words, pause_threshold=1.0)
    assert len(segments) == 2
    assert segments[0].text == "hello world"
    assert segments[0].start == 0.0
    assert segments[0].end == 0.9
    assert segments[1].text == "again"


def test_empty_words_yields_no_segments():
    assert group_words_into_segments([], pause_threshold=1.0) == []
