import logging
from pathlib import Path

from app.errors import AllProvidersFailedError, ProviderError
from app.providers.base import TranscriptionProvider
from app.schemas import TranscriptionResult
from app.services.audio import (
    merge_chunk_results,
    probe_duration_seconds,
    split_on_silence,
    validate_audio,
)

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Runs audio through an ordered provider chain with graceful fallback,
    chunking long files first and merging the pieces back into one result."""

    def __init__(
        self,
        providers: list[TranscriptionProvider],
        chunk_threshold_seconds: int,
        max_upload_mb: int,
    ) -> None:
        self._providers = providers
        self._chunk_threshold = chunk_threshold_seconds
        self._max_upload_mb = max_upload_mb

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        validate_audio(audio_path, self._max_upload_mb)

        if self._is_long(audio_path):
            chunks = split_on_silence(audio_path, max_chunk_seconds=self._chunk_threshold // 2)
            results = [self._run_chain(path, language) for path, _ in chunks]
            return merge_chunk_results(results, offsets=[offset for _, offset in chunks])

        return self._run_chain(audio_path, language)

    def _is_long(self, audio_path: Path) -> bool:
        try:
            return probe_duration_seconds(audio_path) > self._chunk_threshold
        except Exception as exc:  # noqa: BLE001 - probing must never hard-fail the request
            logger.warning("duration probe failed, treating as short: %s", exc)
            return False

    def _run_chain(self, audio_path: Path, language: str | None) -> TranscriptionResult:
        if not self._providers:
            raise AllProvidersFailedError("no transcription providers are configured")
        errors: list[str] = []
        for provider in self._providers:
            try:
                return provider.transcribe(audio_path, language)
            except ProviderError as exc:
                logger.warning("provider %s failed, falling back: %s", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
        raise AllProvidersFailedError("; ".join(errors))
