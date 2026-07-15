from pathlib import Path

from elevenlabs.client import ElevenLabs

from app.errors import ProviderError
from app.providers.base import TranscriptionProvider
from app.providers.grouping import group_words_into_segments
from app.schemas import TranscriptionResult

_MODEL_ID = "scribe_v1"

# Scribe expects ISO 639-3 codes; callers (and other providers) commonly use
# 639-1. Map the languages the UI offers so a picked language works here too.
_ISO_639_1_TO_3 = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "hi": "hin",
    "ur": "urd",
    "ar": "ara",
    "zh": "zho",
    "ja": "jpn",
    "ko": "kor",
    "ru": "rus",
}


def _to_scribe_language(language: str | None) -> str | None:
    if not language:
        return None
    return _ISO_639_1_TO_3.get(language.lower(), language)


class ElevenLabsProvider(TranscriptionProvider):
    """Primary provider using ElevenLabs Scribe.

    Scribe returns word-level timing only, so words are grouped into segments
    on natural pauses to match the pipeline's segment contract.
    """

    name = "elevenlabs"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._client = ElevenLabs(api_key=api_key) if api_key else None

    def is_available(self) -> bool:
        return self._client is not None

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        if self._client is None:
            raise ProviderError("elevenlabs api key not configured")
        # Scribe rejects an empty language_code; omit it entirely to auto-detect.
        scribe_language = _to_scribe_language(language)
        extra = {"language_code": scribe_language} if scribe_language else {}
        try:
            with audio_path.open("rb") as handle:
                response = self._client.speech_to_text.convert(
                    file=handle, model_id=_MODEL_ID, **extra
                )
            words = [
                {"text": w.text, "start": w.start, "end": w.end}
                for w in response.words
                if getattr(w, "type", "word") == "word"
            ]
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"elevenlabs failed: {exc}") from exc

        segments = group_words_into_segments(words)
        return TranscriptionResult(
            language=response.language_code,
            text=response.text.strip(),
            segments=segments,
            provider=self.name,
            duration=round(segments[-1].end, 2) if segments else 0.0,
        )
