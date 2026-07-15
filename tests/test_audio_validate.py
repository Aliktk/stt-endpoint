import pytest

from app.errors import AudioValidationError, UnsupportedFormatError
from app.services.audio import validate_audio


def test_validate_rejects_missing_file(tmp_path):
    with pytest.raises(AudioValidationError):
        validate_audio(tmp_path / "nope.wav", max_upload_mb=100)


def test_validate_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_bytes(b"hello")
    with pytest.raises(UnsupportedFormatError):
        validate_audio(bad, max_upload_mb=100)


def test_validate_rejects_oversized_file(tmp_path):
    big = tmp_path / "big.wav"
    big.write_bytes(b"0" * 2048)
    with pytest.raises(AudioValidationError):
        validate_audio(big, max_upload_mb=0)  # 0 MB limit forces failure


def test_validate_accepts_supported_file(tmp_path):
    good = tmp_path / "clip.wav"
    good.write_bytes(b"0" * 16)
    validate_audio(good, max_upload_mb=100)  # no exception
