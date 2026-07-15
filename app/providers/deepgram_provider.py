from pathlib import Path

from deepgram import DeepgramClient, PrerecordedOptions

from app.errors import ProviderError
from app.providers.base import TranscriptionProvider
from app.schemas import Segment, TranscriptionResult


class DeepgramProvider(TranscriptionProvider):
    """Secondary provider using Deepgram Nova with utterance-level segments."""

    name = "deepgram"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._client = DeepgramClient(api_key=api_key) if api_key else None

    def is_available(self) -> bool:
        return self._client is not None

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        if self._client is None:
            raise ProviderError("deepgram api key not configured")
        # A falsy language means auto-detect: enable detection and omit the hint.
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            utterances=True,
            detect_language=not language,
            language=language or None,
        )
        try:
            payload = {"buffer": audio_path.read_bytes()}
            response = self._client.listen.prerecorded.v("1").transcribe_file(payload, options)
            results = response.results
            segments = [
                Segment(start=round(u.start, 2), end=round(u.end, 2), text=u.transcript.strip())
                for u in results.utterances
            ]
            language_code = results.channels[0].detected_language or (language or "unknown")
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"deepgram failed: {exc}") from exc

        return TranscriptionResult(
            language=language_code,
            text=" ".join(s.text for s in segments).strip(),
            segments=segments,
            provider=self.name,
            duration=round(segments[-1].end, 2) if segments else 0.0,
        )
