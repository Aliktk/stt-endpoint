from collections.abc import Sequence

from app.schemas import Segment


def group_words_into_segments(
    words: Sequence[dict], pause_threshold: float = 1.0
) -> list[Segment]:
    """Merge word-level timings into sentence-like segments.

    A new segment starts whenever the silence before a word exceeds
    ``pause_threshold`` seconds. Used for providers (ElevenLabs) that expose
    words but no native utterance grouping.
    """
    segments: list[Segment] = []
    buffer: list[dict] = []

    def flush() -> None:
        if not buffer:
            return
        segments.append(
            Segment(
                start=buffer[0]["start"],
                end=buffer[-1]["end"],
                text=" ".join(w["text"].strip() for w in buffer).strip(),
            )
        )

    for word in words:
        if buffer and word["start"] - buffer[-1]["end"] > pause_threshold:
            flush()
            buffer = []
        buffer.append(word)
    flush()
    return segments
