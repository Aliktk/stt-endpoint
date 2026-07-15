from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from app.errors import AudioValidationError, UnsupportedFormatError
from app.schemas import Segment, TranscriptionResult

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}


def validate_audio(path: Path, max_upload_mb: int) -> None:
    """Fail fast on missing, oversized, or unsupported audio before any decode."""
    if not path.is_file():
        raise AudioValidationError(f"audio file not found: {path.name}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"unsupported format: {path.suffix or 'none'}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_upload_mb:
        raise AudioValidationError(f"file too large: {size_mb:.1f} MB > {max_upload_mb} MB")


def probe_duration_seconds(path: Path) -> float:
    """Duration in seconds, decoded via pydub/ffprobe."""
    return len(AudioSegment.from_file(path)) / 1000.0


def split_on_silence(path: Path, max_chunk_seconds: int = 300) -> list[tuple[Path, float]]:
    """Split long audio into ``(chunk_path, offset_seconds)`` pairs.

    Cuts fall on silence boundaries so no word is split mid-utterance, and each
    chunk is capped near ``max_chunk_seconds`` to bound memory and allow retries
    per chunk. Returns the original file unchanged when no silence is detected.
    """
    audio = AudioSegment.from_file(path)
    nonsilent = detect_nonsilent(audio, min_silence_len=500, silence_thresh=audio.dBFS - 16)
    if not nonsilent:
        return [(path, 0.0)]

    chunks: list[tuple[Path, float]] = []
    cap_ms = max_chunk_seconds * 1000
    start_ms, end_ms = nonsilent[0]
    for seg_start, seg_end in nonsilent[1:]:
        if seg_end - start_ms > cap_ms:
            chunks.append(_export_chunk(audio, start_ms, end_ms, path))
            start_ms = seg_start
        end_ms = seg_end
    chunks.append(_export_chunk(audio, start_ms, end_ms, path))
    return chunks


def _export_chunk(
    audio: AudioSegment, start_ms: int, end_ms: int, source: Path
) -> tuple[Path, float]:
    out = source.with_name(f"{source.stem}_chunk_{start_ms}{source.suffix}")
    audio[start_ms:end_ms].export(out, format=source.suffix.lstrip("."))
    return out, start_ms / 1000.0


def merge_chunk_results(
    results: list[TranscriptionResult], offsets: list[float]
) -> TranscriptionResult:
    """Concatenate per-chunk results, shifting each segment by its chunk offset."""
    segments: list[Segment] = []
    for result, offset in zip(results, offsets, strict=True):
        for seg in result.segments:
            segments.append(
                Segment(
                    start=round(seg.start + offset, 2),
                    end=round(seg.end + offset, 2),
                    text=seg.text,
                )
            )
    duration = max((s.end for s in segments), default=0.0)
    return TranscriptionResult(
        language=results[0].language if results else "unknown",
        text=" ".join(s.text for s in segments).strip(),
        segments=segments,
        provider=results[0].provider if results else "unknown",
        duration=round(duration, 2),
    )
