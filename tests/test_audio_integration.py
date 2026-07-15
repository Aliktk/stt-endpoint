"""Integration tests that exercise real audio decoding via pydub/ffmpeg.

These synthesize a tone/silence/tone clip so the silence splitter has genuine
boundaries to cut on — no network or model inference involved.
"""

from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from app.services.audio import probe_duration_seconds, split_on_silence


def _clip_with_gap(path: Path) -> None:
    tone = Sine(220).to_audio_segment(duration=1500).apply_gain(-3)
    silence = AudioSegment.silent(duration=1200)
    (tone + silence + tone).export(path, format="wav")


def test_probe_duration_reads_real_length(tmp_path):
    clip = tmp_path / "clip.wav"
    _clip_with_gap(clip)
    duration = probe_duration_seconds(clip)
    assert 4.0 < duration < 4.4  # 1.5 + 1.2 + 1.5s


def test_split_on_silence_produces_offset_chunks(tmp_path):
    clip = tmp_path / "clip.wav"
    _clip_with_gap(clip)

    # A small cap forces a cut at the silent gap between the two tones.
    chunks = split_on_silence(clip, max_chunk_seconds=2)

    assert len(chunks) == 2
    paths = [p for p, _ in chunks]
    offsets = [o for _, o in chunks]
    assert offsets[0] == 0.0
    assert offsets[1] > 0.0  # second chunk starts after the first tone
    assert all(p.exists() for p in paths)
