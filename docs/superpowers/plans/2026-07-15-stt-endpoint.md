# STT Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a plug-and-play speech-to-text pipeline that transcribes audio to text with per-segment timestamps, API-first (ElevenLabs → Deepgram) with a local faster-whisper fallback, exposed via FastAPI and a polished Streamlit UI.

**Architecture:** A provider-agnostic `TranscriptionService` runs an ordered fallback chain over pluggable providers that all return one normalized `TranscriptionResult`. Long audio is split on silence, transcribed per chunk, and merged with offset-corrected timestamps. FastAPI and Streamlit are thin front-ends over the same service.

**Tech Stack:** Python 3.11, FastAPI, pydantic-settings, httpx, `elevenlabs`, `deepgram-sdk`, `faster-whisper`, `pydub` (chunking), `ffmpeg`/`ffprobe`, Streamlit, pytest, ruff, black.

---

## Environment notes for the worker

- Windows dev box; shell is Git Bash for the `Bash` tool and PowerShell otherwise. Use forward slashes in Python.
- Create and use a venv: `python -m venv .venv && source .venv/Scripts/activate` (Git Bash on Windows).
- `ffmpeg`/`ffprobe` must be on PATH (faster-whisper + pydub need it). If missing, `winget install Gyan.FFmpeg` or note it in README.
- All provider network calls are **mocked** in tests. Never require real API keys to run the suite.
- Commit after every task with a conventional-commit message. Author: `Ali Nawaz <nawazktk99@gmail.com>`.

---

## File structure (locked)

```
app/
  __init__.py
  main.py                       # FastAPI app factory
  config.py                     # Settings (pydantic-settings)
  schemas.py                    # Segment, TranscriptionResult, ErrorResponse
  errors.py                     # typed exceptions
  api/__init__.py
  api/routes.py                 # endpoints + exception handlers
  services/__init__.py
  services/audio.py             # validate, probe, chunk, merge
  services/transcription_service.py
  providers/__init__.py
  providers/base.py             # TranscriptionProvider ABC
  providers/registry.py         # build ordered provider list from settings
  providers/local_whisper_provider.py
  providers/elevenlabs_provider.py
  providers/deepgram_provider.py
streamlit_app/app.py
tests/...
docs/ARCHITECTURE.md, DESIGN_DECISIONS.md, CHECKLIST.md
samples/
.env.example .gitignore requirements.txt Dockerfile Makefile README.md
.streamlit/config.toml pyproject.toml
```

---

## Task 0: Project scaffolding & tooling

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.env.example`, `app/__init__.py`, `app/api/__init__.py`, `app/services/__init__.py`, `app/providers/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Write `requirements.txt`**

```
fastapi==0.115.5
uvicorn[standard]==0.32.1
python-multipart==0.0.12
pydantic==2.9.2
pydantic-settings==2.6.1
httpx==0.27.2
elevenlabs==1.13.0
deepgram-sdk==3.7.7
faster-whisper==1.0.3
pydub==0.25.1
streamlit==1.40.1
requests==2.32.3
pytest==8.3.3
pytest-cov==5.0.0
pytest-asyncio==0.24.0
ruff==0.7.4
black==24.10.0
```

- [ ] **Step 2: Write `pyproject.toml`** (tooling config only)

```toml
[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 3: Write `.env.example`**

```
# All keys optional. With none set, the service uses the local faster-whisper fallback.
ELEVENLABS_API_KEY=
DEEPGRAM_API_KEY=

# Comma-separated fallback order. Providers without a key are skipped automatically.
PROVIDER_ORDER=elevenlabs,deepgram,local

# Local model
WHISPER_MODEL_SIZE=base
WHISPER_COMPUTE_TYPE=int8

# Long-audio handling
CHUNK_THRESHOLD_SECONDS=600
MAX_UPLOAD_MB=100
```

- [ ] **Step 4: Create empty package files**

Create `app/__init__.py`, `app/api/__init__.py`, `app/services/__init__.py`, `app/providers/__init__.py`, `tests/__init__.py` as empty files.

- [ ] **Step 5: Install and verify tooling**

Run: `python -m venv .venv && source .venv/Scripts/activate && pip install -r requirements.txt`
Expected: all install without error. Then `ruff --version && black --version && pytest --version` all print versions.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt .env.example app tests
git commit -m "chore: scaffold project structure and tooling"
```

---

## Task 1: Domain schemas

**Files:**
- Create: `app/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas.py
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
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Segment(start=2.0, end=1.0, text="bad")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.schemas'`.

- [ ] **Step 3: Write `app/schemas.py`**

```python
from pydantic import BaseModel, Field, model_validator


class Segment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _end_after_start(self) -> "Segment":
        if self.end < self.start:
            raise ValueError("segment end must be >= start")
        return self


class TranscriptionResult(BaseModel):
    language: str
    text: str
    segments: list[Segment]
    provider: str
    duration: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py
git commit -m "feat: add domain schemas for transcription results"
```

---

## Task 2: Typed errors + configuration

**Files:**
- Create: `app/errors.py`, `app/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from app.config import Settings


def test_provider_order_parses_to_list():
    settings = Settings(PROVIDER_ORDER="elevenlabs,local", ELEVENLABS_API_KEY="k")
    assert settings.provider_order == ["elevenlabs", "local"]


def test_defaults_are_sane_with_no_env():
    settings = Settings()
    assert "local" in settings.provider_order
    assert settings.max_upload_mb > 0
    assert settings.chunk_threshold_seconds > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`.

- [ ] **Step 3: Write `app/errors.py`**

```python
class TranscriptionError(Exception):
    """Base class for all transcription pipeline errors."""


class ProviderError(TranscriptionError):
    """A single provider failed to produce a transcript."""


class AllProvidersFailedError(TranscriptionError):
    """Every configured provider failed."""


class AudioValidationError(TranscriptionError):
    """The uploaded audio failed validation (missing, too large, unsupported)."""


class UnsupportedFormatError(AudioValidationError):
    """The audio format/extension is not accepted."""
```

- [ ] **Step 4: Write `app/config.py`**

```python
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")
    deepgram_api_key: str | None = Field(default=None, alias="DEEPGRAM_API_KEY")

    provider_order: list[str] = Field(default=["elevenlabs", "deepgram", "local"], alias="PROVIDER_ORDER")

    whisper_model_size: str = Field(default="base", alias="WHISPER_MODEL_SIZE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")

    chunk_threshold_seconds: int = Field(default=600, alias="CHUNK_THRESHOLD_SECONDS")
    max_upload_mb: int = Field(default=100, alias="MAX_UPLOAD_MB")

    @field_validator("provider_order", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add app/errors.py app/config.py tests/test_config.py
git commit -m "feat: add typed errors and settings"
```

---

## Task 3: Provider base interface

**Files:**
- Create: `app/providers/base.py`
- Test: `tests/test_provider_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provider_base.py
from pathlib import Path

from app.providers.base import TranscriptionProvider
from app.schemas import Segment, TranscriptionResult


class FakeProvider(TranscriptionProvider):
    name = "fake"

    def is_available(self) -> bool:
        return True

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(
            language=language or "en",
            text="ok",
            segments=[Segment(start=0.0, end=1.0, text="ok")],
            provider=self.name,
            duration=1.0,
        )


def test_provider_contract_is_usable():
    provider = FakeProvider()
    result = provider.transcribe(Path("x.wav"))
    assert result.provider == "fake"
    assert provider.is_available() is True


def test_abstract_provider_cannot_instantiate():
    import pytest

    with pytest.raises(TypeError):
        TranscriptionProvider()  # type: ignore[abstract]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_provider_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.providers.base'`.

- [ ] **Step 3: Write `app/providers/base.py`**

```python
from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas import TranscriptionResult


class TranscriptionProvider(ABC):
    """Contract every speech-to-text backend implements.

    A provider owns exactly one thing: turning an audio file into a normalized
    TranscriptionResult. Availability is decoupled from transcription so the
    fallback chain can skip a backend whose credentials are absent.
    """

    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is usable right now (key present, model loadable)."""

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        """Transcribe an audio file into a normalized result."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_provider_base.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/providers/base.py tests/test_provider_base.py
git commit -m "feat: add provider interface"
```

---

## Task 4: Word→segment grouping helper

ElevenLabs returns word-level timing only. This shared helper groups words into segments by pause, and is reused wherever we only have words.

**Files:**
- Create: `app/providers/grouping.py`
- Test: `tests/test_grouping.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grouping.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_grouping.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `app/providers/grouping.py`**

```python
from collections.abc import Sequence

from app.schemas import Segment


def group_words_into_segments(
    words: Sequence[dict], pause_threshold: float = 1.0
) -> list[Segment]:
    """Merge word-level timings into sentence-like segments.

    A new segment starts whenever the silence before a word exceeds
    `pause_threshold` seconds. Used for providers (ElevenLabs) that expose
    words but not utterances.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_grouping.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/providers/grouping.py tests/test_grouping.py
git commit -m "feat: add word-to-segment grouping helper"
```

---

## Task 5: Local faster-whisper provider

**Files:**
- Create: `app/providers/local_whisper_provider.py`
- Test: `tests/test_local_provider.py`

- [ ] **Step 1: Write the failing test** (model is mocked — no real inference)

```python
# tests/test_local_provider.py
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.providers.local_whisper_provider import LocalWhisperProvider


def _fake_segments():
    return [
        SimpleNamespace(start=0.0, end=1.2, text=" hello"),
        SimpleNamespace(start=1.3, end=2.0, text=" world"),
    ]


@patch("app.providers.local_whisper_provider.WhisperModel")
def test_local_provider_normalizes_output(mock_model_cls):
    model = MagicMock()
    info = SimpleNamespace(language="en", duration=2.0)
    model.transcribe.return_value = (_fake_segments(), info)
    mock_model_cls.return_value = model

    provider = LocalWhisperProvider(model_size="base", compute_type="int8")
    result = provider.transcribe(Path("x.wav"))

    assert result.provider == "local"
    assert result.language == "en"
    assert result.text == "hello world"
    assert len(result.segments) == 2
    assert result.segments[0].start == 0.0


def test_local_provider_always_available():
    provider = LocalWhisperProvider(model_size="base", compute_type="int8")
    assert provider.is_available() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_local_provider.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `app/providers/local_whisper_provider.py`**

```python
from pathlib import Path

from faster_whisper import WhisperModel

from app.errors import ProviderError
from app.providers.base import TranscriptionProvider
from app.schemas import Segment, TranscriptionResult


class LocalWhisperProvider(TranscriptionProvider):
    """Offline fallback backed by faster-whisper. Needs no API key."""

    name = "local"

    def __init__(self, model_size: str, compute_type: str) -> None:
        self._model_size = model_size
        self._compute_type = compute_type
        self._model: WhisperModel | None = None

    def is_available(self) -> bool:
        return True

    def _load(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self._model_size, device="cpu", compute_type=self._compute_type
            )
        return self._model

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        try:
            segments_iter, info = self._load().transcribe(
                str(audio_path), vad_filter=True, language=language
            )
            segments = [
                Segment(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip())
                for s in segments_iter
            ]
        except Exception as exc:  # noqa: BLE001 - normalize any engine failure
            raise ProviderError(f"local whisper failed: {exc}") from exc

        return TranscriptionResult(
            language=info.language,
            text=" ".join(s.text for s in segments).strip(),
            segments=segments,
            provider=self.name,
            duration=round(getattr(info, "duration", 0.0), 2),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_local_provider.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/providers/local_whisper_provider.py tests/test_local_provider.py
git commit -m "feat: add local faster-whisper provider"
```

---

## Task 6: ElevenLabs provider

**Files:**
- Create: `app/providers/elevenlabs_provider.py`
- Test: `tests/test_elevenlabs_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_elevenlabs_provider.py
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.providers.elevenlabs_provider import ElevenLabsProvider


def _fake_response():
    words = [
        SimpleNamespace(text="hello", start=0.0, end=0.4, type="word"),
        SimpleNamespace(text="world", start=0.5, end=0.9, type="word"),
    ]
    return SimpleNamespace(language_code="en", text="hello world", words=words)


@patch("app.providers.elevenlabs_provider.ElevenLabs")
def test_elevenlabs_groups_words_into_segments(mock_client_cls):
    client = MagicMock()
    client.speech_to_text.convert.return_value = _fake_response()
    mock_client_cls.return_value = client

    provider = ElevenLabsProvider(api_key="k")
    result = provider.transcribe(Path("x.mp3"))

    assert result.provider == "elevenlabs"
    assert result.language == "en"
    assert result.text == "hello world"
    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 0.9


def test_elevenlabs_unavailable_without_key():
    assert ElevenLabsProvider(api_key=None).is_available() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_elevenlabs_provider.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `app/providers/elevenlabs_provider.py`**

```python
from pathlib import Path

from elevenlabs.client import ElevenLabs

from app.errors import ProviderError
from app.providers.base import TranscriptionProvider
from app.providers.grouping import group_words_into_segments
from app.schemas import TranscriptionResult

_MODEL_ID = "scribe_v1"


class ElevenLabsProvider(TranscriptionProvider):
    """Primary provider using ElevenLabs Scribe. Returns word-level timing,
    which we group into segments."""

    name = "elevenlabs"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._client = ElevenLabs(api_key=api_key) if api_key else None

    def is_available(self) -> bool:
        return self._client is not None

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        if self._client is None:
            raise ProviderError("elevenlabs api key not configured")
        try:
            with audio_path.open("rb") as handle:
                response = self._client.speech_to_text.convert(
                    file=handle, model_id=_MODEL_ID, language_code=language
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_elevenlabs_provider.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/providers/elevenlabs_provider.py tests/test_elevenlabs_provider.py
git commit -m "feat: add elevenlabs provider"
```

---

## Task 7: Deepgram provider

**Files:**
- Create: `app/providers/deepgram_provider.py`
- Test: `tests/test_deepgram_provider.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deepgram_provider.py
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.providers.deepgram_provider import DeepgramProvider


def _fake_response():
    utterances = [
        SimpleNamespace(transcript="hello world", start=0.0, end=1.0),
        SimpleNamespace(transcript="again", start=2.0, end=2.5),
    ]
    channel = SimpleNamespace(detected_language="en")
    results = SimpleNamespace(utterances=utterances, channels=[channel])
    return SimpleNamespace(results=results)


@patch("app.providers.deepgram_provider.DeepgramClient")
def test_deepgram_reads_utterances(mock_client_cls):
    client = MagicMock()
    client.listen.prerecorded.v.return_value.transcribe_file.return_value = _fake_response()
    mock_client_cls.return_value = client

    provider = DeepgramProvider(api_key="k")
    with patch.object(Path, "read_bytes", return_value=b"audio"):
        result = provider.transcribe(Path("x.mp3"))

    assert result.provider == "deepgram"
    assert result.language == "en"
    assert result.text == "hello world again"
    assert len(result.segments) == 2
    assert result.segments[1].start == 2.0


def test_deepgram_unavailable_without_key():
    assert DeepgramProvider(api_key=None).is_available() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deepgram_provider.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `app/providers/deepgram_provider.py`**

```python
from pathlib import Path

from deepgram import DeepgramClient, PrerecordedOptions

from app.errors import ProviderError
from app.providers.base import TranscriptionProvider
from app.schemas import Segment, TranscriptionResult


class DeepgramProvider(TranscriptionProvider):
    """Secondary provider using Deepgram Nova with utterance segmentation."""

    name = "deepgram"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key
        self._client = DeepgramClient(api_key=api_key) if api_key else None

    def is_available(self) -> bool:
        return self._client is not None

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        if self._client is None:
            raise ProviderError("deepgram api key not configured")
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            utterances=True,
            detect_language=language is None,
            language=language,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_deepgram_provider.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/providers/deepgram_provider.py tests/test_deepgram_provider.py
git commit -m "feat: add deepgram provider"
```

---

## Task 8: Provider registry

**Files:**
- Create: `app/providers/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
from app.config import Settings
from app.providers.registry import build_providers


def test_registry_orders_and_filters_by_availability():
    settings = Settings(
        PROVIDER_ORDER="elevenlabs,deepgram,local",
        ELEVENLABS_API_KEY=None,
        DEEPGRAM_API_KEY=None,
    )
    providers = build_providers(settings)
    # No keys -> only local is available
    names = [p.name for p in providers]
    assert names == ["local"]


def test_registry_keeps_configured_order():
    settings = Settings(
        PROVIDER_ORDER="deepgram,elevenlabs,local",
        ELEVENLABS_API_KEY="a",
        DEEPGRAM_API_KEY="b",
    )
    names = [p.name for p in build_providers(settings)]
    assert names == ["deepgram", "elevenlabs", "local"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `app/providers/registry.py`**

```python
from app.config import Settings
from app.providers.base import TranscriptionProvider
from app.providers.deepgram_provider import DeepgramProvider
from app.providers.elevenlabs_provider import ElevenLabsProvider
from app.providers.local_whisper_provider import LocalWhisperProvider


def build_providers(settings: Settings) -> list[TranscriptionProvider]:
    """Instantiate providers in configured order, keeping only available ones."""
    catalog: dict[str, TranscriptionProvider] = {
        "elevenlabs": ElevenLabsProvider(settings.elevenlabs_api_key),
        "deepgram": DeepgramProvider(settings.deepgram_api_key),
        "local": LocalWhisperProvider(settings.whisper_model_size, settings.whisper_compute_type),
    }
    ordered = [catalog[name] for name in settings.provider_order if name in catalog]
    return [p for p in ordered if p.is_available()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/providers/registry.py tests/test_registry.py
git commit -m "feat: add provider registry with availability filtering"
```

---

## Task 9: Audio validation

**Files:**
- Create: `app/services/audio.py`
- Test: `tests/test_audio_validate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audio_validate.py
from pathlib import Path

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_validate.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `app/services/audio.py` (validation portion)**

```python
from pathlib import Path

from app.errors import AudioValidationError, UnsupportedFormatError

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_validate.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add app/services/audio.py tests/test_audio_validate.py
git commit -m "feat: add audio validation"
```

---

## Task 10: Chunk-and-merge for long audio

**Files:**
- Modify: `app/services/audio.py`
- Test: `tests/test_audio_chunk.py`

- [ ] **Step 1: Write the failing test** (pydub is mocked; we test the pure merge math directly)

```python
# tests/test_audio_chunk.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_chunk.py -v`
Expected: FAIL — `merge_chunk_results` not found.

- [ ] **Step 3: Add chunk + merge functions to `app/services/audio.py`**

Append these imports at the top (merge with existing import block) and functions at the end:

```python
# add to imports
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from app.schemas import Segment, TranscriptionResult


def probe_duration_seconds(path: Path) -> float:
    """Duration in seconds via pydub/ffprobe."""
    return len(AudioSegment.from_file(path)) / 1000.0


def split_on_silence(path: Path, max_chunk_seconds: int = 300) -> list[tuple[Path, float]]:
    """Split long audio into (chunk_path, offset_seconds) pairs at silence
    boundaries, capping each chunk near `max_chunk_seconds`."""
    audio = AudioSegment.from_file(path)
    nonsilent = detect_nonsilent(audio, min_silence_len=500, silence_thresh=audio.dBFS - 16)
    if not nonsilent:
        return [(path, 0.0)]

    chunks: list[tuple[Path, float]] = []
    cap_ms = max_chunk_seconds * 1000
    start_ms = nonsilent[0][0]
    end_ms = nonsilent[0][1]
    for seg_start, seg_end in nonsilent[1:]:
        if seg_end - start_ms > cap_ms:
            chunks.append(_export_chunk(audio, start_ms, end_ms, path))
            start_ms = seg_start
        end_ms = seg_end
    chunks.append(_export_chunk(audio, start_ms, end_ms, path))
    return chunks


def _export_chunk(audio: "AudioSegment", start_ms: int, end_ms: int, source: Path) -> tuple[Path, float]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_chunk.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add app/services/audio.py tests/test_audio_chunk.py
git commit -m "feat: add long-audio chunking and timestamp merge"
```

---

## Task 11: Transcription service (fallback orchestration)

**Files:**
- Create: `app/services/transcription_service.py`
- Test: `tests/test_transcription_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transcription_service.py
from pathlib import Path

import pytest

from app.errors import AllProvidersFailedError, ProviderError
from app.providers.base import TranscriptionProvider
from app.schemas import Segment, TranscriptionResult
from app.services.transcription_service import TranscriptionService


class _Stub(TranscriptionProvider):
    def __init__(self, name, fail=False):
        self.name = name
        self._fail = fail

    def is_available(self):
        return True

    def transcribe(self, audio_path, language=None):
        if self._fail:
            raise ProviderError(f"{self.name} boom")
        return TranscriptionResult(
            language="en",
            text=self.name,
            segments=[Segment(start=0.0, end=1.0, text=self.name)],
            provider=self.name,
            duration=1.0,
        )


def test_service_falls_back_to_next_provider(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    service = TranscriptionService(
        providers=[_Stub("elevenlabs", fail=True), _Stub("local")],
        chunk_threshold_seconds=600,
        max_upload_mb=100,
    )
    result = service.transcribe(audio)
    assert result.provider == "local"


def test_service_raises_when_all_fail(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    service = TranscriptionService(
        providers=[_Stub("elevenlabs", fail=True), _Stub("local", fail=True)],
        chunk_threshold_seconds=600,
        max_upload_mb=100,
    )
    with pytest.raises(AllProvidersFailedError):
        service.transcribe(audio)


def test_service_raises_with_no_providers(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    service = TranscriptionService(providers=[], chunk_threshold_seconds=600, max_upload_mb=100)
    with pytest.raises(AllProvidersFailedError):
        service.transcribe(audio)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcription_service.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `app/services/transcription_service.py`**

Note: validation runs first; short audio goes straight to the chain; long audio is chunked, each chunk run through the chain, then merged. Chunking is guarded so a probe failure degrades gracefully to whole-file transcription.

```python
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
    chunking long files first."""

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transcription_service.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app/services/transcription_service.py tests/test_transcription_service.py
git commit -m "feat: add transcription service with fallback orchestration"
```

---

## Task 12: FastAPI app, routes, dependency wiring

**Files:**
- Create: `app/api/routes.py`, `app/main.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import io
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import Segment, TranscriptionResult


def _client():
    return TestClient(create_app())


def test_health_ok():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_transcribe_happy_path():
    fake = TranscriptionResult(
        language="en",
        text="hello world",
        segments=[Segment(start=0.0, end=1.0, text="hello world")],
        provider="local",
        duration=1.0,
    )
    with patch("app.api.routes.TranscriptionService.transcribe", return_value=fake):
        files = {"file": ("clip.wav", io.BytesIO(b"0" * 32), "audio/wav")}
        resp = _client().post("/v1/transcribe", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "local"
    assert body["segments"][0]["end"] == 1.0


def test_transcribe_unsupported_format_returns_422():
    from app.errors import UnsupportedFormatError

    with patch(
        "app.api.routes.TranscriptionService.transcribe",
        side_effect=UnsupportedFormatError("nope"),
    ):
        files = {"file": ("notes.txt", io.BytesIO(b"hi"), "text/plain")}
        resp = _client().post("/v1/transcribe", files=files)
    assert resp.status_code == 422


def test_transcribe_all_providers_failed_returns_502():
    from app.errors import AllProvidersFailedError

    with patch(
        "app.api.routes.TranscriptionService.transcribe",
        side_effect=AllProvidersFailedError("all down"),
    ):
        files = {"file": ("clip.wav", io.BytesIO(b"0" * 32), "audio/wav")}
        resp = _client().post("/v1/transcribe", files=files)
    assert resp.status_code == 502
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write `app/api/routes.py`**

```python
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from app.config import get_settings
from app.providers.registry import build_providers
from app.schemas import TranscriptionResult
from app.services.transcription_service import TranscriptionService

router = APIRouter()


def _service() -> TranscriptionService:
    settings = get_settings()
    return TranscriptionService(
        providers=build_providers(settings),
        chunk_threshold_seconds=settings.chunk_threshold_seconds,
        max_upload_mb=settings.max_upload_mb,
    )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/providers")
def providers() -> dict[str, list[str]]:
    return {"available": [p.name for p in build_providers(get_settings())]}


@router.post("/v1/transcribe", response_model=TranscriptionResult)
async def transcribe(
    file: UploadFile = File(...), language: str | None = Form(default=None)
) -> TranscriptionResult:
    suffix = Path(file.filename or "audio").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)
    try:
        return _service().transcribe(temp_path, language)
    finally:
        temp_path.unlink(missing_ok=True)
```

- [ ] **Step 4: Write `app/main.py` with exception handlers**

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.errors import AllProvidersFailedError, AudioValidationError, UnsupportedFormatError
from app.schemas import ErrorResponse


def create_app() -> FastAPI:
    app = FastAPI(title="STT Endpoint", version="1.0.0")
    app.include_router(router)

    @app.exception_handler(UnsupportedFormatError)
    async def _unsupported(_: Request, exc: UnsupportedFormatError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(error="unsupported_format", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(AudioValidationError)
    async def _bad_audio(_: Request, exc: AudioValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error="invalid_audio", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(AllProvidersFailedError)
    async def _all_failed(_: Request, exc: AllProvidersFailedError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(error="all_providers_failed", detail=str(exc)).model_dump(),
        )

    return app


app = create_app()
```

Note: `UnsupportedFormatError` subclasses `AudioValidationError`, so register its handler first — FastAPI matches the most specific registered handler, but ordering makes intent explicit.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add app/api/routes.py app/main.py tests/test_api.py
git commit -m "feat: add FastAPI app, routes, and error handlers"
```

---

## Task 13: Full suite + coverage gate

- [ ] **Step 1: Run the whole suite with coverage**

Run: `pytest --cov=app --cov-report=term-missing`
Expected: all tests pass; coverage ≥ 80%. If below, add targeted tests for uncovered branches (e.g. `/v1/providers`, grouping edge cases).

- [ ] **Step 2: Lint and format**

Run: `ruff check app tests && black --check app tests`
Expected: clean. If black reports changes, run `black app tests` and re-run.

- [ ] **Step 3: Commit any fixups**

```bash
git add -A
git commit -m "test: reach coverage target and clean lint"
```

---

## Task 14: Polished Streamlit UI

**Files:**
- Create: `streamlit_app/app.py`, `.streamlit/config.toml`

- [ ] **Step 1: Write `.streamlit/config.toml` (custom theme)**

```toml
[theme]
primaryColor = "#6C5CE7"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1A1D29"
textColor = "#EAECEF"
font = "sans serif"

[server]
maxUploadSize = 100
```

- [ ] **Step 2: Write `streamlit_app/app.py`**

The UI must: show a clean header, a sidebar with API URL + language + a live provider-status panel (calls `GET /v1/providers`), a drag-and-drop uploader with inline audio player, a spinner with elapsed time during transcription, a provider badge, the full transcript, a segment table with monospace timestamps, and a Download JSON button. Errors surface as friendly messages, never stack traces.

```python
import json
import time

import requests
import streamlit as st

st.set_page_config(page_title="STT Endpoint", page_icon="🎙️", layout="wide")

DEFAULT_API = "http://localhost:8000"


def _provider_status(api_url: str) -> list[str]:
    try:
        resp = requests.get(f"{api_url}/v1/providers", timeout=5)
        resp.raise_for_status()
        return resp.json().get("available", [])
    except requests.RequestException:
        return []


def _format_ts(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 1100px; }
      .stDataFrame { border-radius: 10px; }
      div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎙️ Speech-to-Text Pipeline")
st.caption("API-first transcription with automatic local fallback and per-segment timestamps.")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API).rstrip("/")
    language = st.text_input("Language code (blank = auto-detect)", value="")
    st.divider()
    st.subheader("Providers online")
    available = _provider_status(api_url)
    if available:
        for name in available:
            st.success(name, icon="✅")
    else:
        st.warning("API unreachable or no providers available.")

uploaded = st.file_uploader(
    "Drop an audio file", type=["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "aac"]
)

if uploaded is not None:
    st.audio(uploaded)
    if st.button("Transcribe", type="primary", use_container_width=True):
        started = time.time()
        with st.spinner("Transcribing…"):
            try:
                resp = requests.post(
                    f"{api_url}/v1/transcribe",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    data={"language": language} if language else None,
                    timeout=1800,
                )
            except requests.RequestException as exc:
                st.error(f"Could not reach the API at {api_url}. Is it running? ({exc})")
                st.stop()

        elapsed = time.time() - started
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text)
            st.error(f"Transcription failed: {detail}")
            st.stop()

        data = resp.json()
        col1, col2, col3 = st.columns(3)
        col1.metric("Provider", data["provider"])
        col2.metric("Language", data["language"])
        col3.metric("Elapsed", f"{elapsed:.1f}s")

        st.subheader("Transcript")
        st.write(data["text"] or "_(empty)_")

        st.subheader("Segments")
        rows = [
            {"start": _format_ts(s["start"]), "end": _format_ts(s["end"]), "text": s["text"]}
            for s in data["segments"]
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.download_button(
            "Download JSON",
            data=json.dumps(data, indent=2),
            file_name=f"{uploaded.name}.transcript.json",
            mime="application/json",
            use_container_width=True,
        )
```

- [ ] **Step 3: Manual smoke check**

Run in two terminals:
`uvicorn app.main:app --reload` and `streamlit run streamlit_app/app.py`
Expected: UI loads, sidebar shows `local` under "Providers online" (with no keys set), uploading a sample WAV returns a transcript table.

- [ ] **Step 4: Commit**

```bash
git add streamlit_app/app.py .streamlit/config.toml
git commit -m "feat: add polished streamlit demo UI"
```

---

## Task 15: Sample audio + end-to-end local run

**Files:**
- Create: `samples/generate_sample.py`, `samples/README.md`

- [ ] **Step 1: Write `samples/generate_sample.py`** (creates a tiny spoken-word WAV via TTS if available, else a tone — mock data)

```python
"""Generate a small sample audio file for demos and manual testing.

Uses pyttsx3 if installed for real speech; otherwise emits a short sine tone
so the pipeline has something valid to decode.
"""
import math
import struct
import wave
from pathlib import Path

OUT = Path(__file__).parent / "sample.wav"


def _tone(path: Path, seconds: float = 3.0, freq: float = 220.0, rate: int = 16000) -> None:
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        for i in range(int(seconds * rate)):
            value = int(32767 * 0.3 * math.sin(2 * math.pi * freq * i / rate))
            wav.writeframes(struct.pack("<h", value))


if __name__ == "__main__":
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.save_to_file("Hello, this is a sample transcription test.", str(OUT))
        engine.runAndWait()
        if not OUT.exists() or OUT.stat().st_size == 0:
            raise RuntimeError("tts produced no file")
    except Exception:
        _tone(OUT)
    print(f"wrote {OUT}")
```

- [ ] **Step 2: Generate the sample and run end-to-end**

Run: `python samples/generate_sample.py`
Then with the API running: `curl -F "file=@samples/sample.wav" http://localhost:8000/v1/transcribe`
Expected: JSON with `provider`, `language`, `segments`. (Tone audio may transcribe to empty text — that's fine; it proves the pipeline path.)

- [ ] **Step 3: Commit**

```bash
git add samples/
git commit -m "chore: add sample audio generator"
```

---

## Task 16: Documentation

**Files:**
- Create: `docs/ARCHITECTURE.md`, `docs/DESIGN_DECISIONS.md`, `docs/CHECKLIST.md`, `README.md`, `Dockerfile`, `Makefile`

- [ ] **Step 1: Write `docs/ARCHITECTURE.md`**

Include the mermaid flowchart from the spec plus a component table (module → responsibility → key types) and a short "why a provider chain" paragraph. Add a second mermaid `sequenceDiagram` showing: Client → FastAPI → validate → (chunk?) → provider chain → merge → response.

- [ ] **Step 2: Write `docs/DESIGN_DECISIONS.md`**

The Part-2 written answers (concurrency, storage, retries, API exposure) from spec §10, each connected to what the demo already shows. Include a "Trade-offs & what I'd add for production" section (queue/DB/object storage, auth, rate limiting, observability).

- [ ] **Step 3: Write `docs/CHECKLIST.md`**

The assessment + quality checklist (from spec §8 and §11) as ticked boxes, plus a "how each requirement is satisfied → file/line" mapping.

- [ ] **Step 4: Write `README.md`**

Sections: what it is, architecture image (mermaid), quickstart (venv, install, ffmpeg note), run the API, run the UI, configure providers via `.env`, run tests + coverage, Docker, project layout, design-decisions link. Keep prose natural and concise.

- [ ] **Step 5: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 6: Write `Makefile`**

```makefile
.PHONY: install api ui test lint fmt
install:
	pip install -r requirements.txt
api:
	uvicorn app.main:app --reload
ui:
	streamlit run streamlit_app/app.py
test:
	pytest --cov=app --cov-report=term-missing
lint:
	ruff check app tests && black --check app tests
fmt:
	black app tests && ruff check --fix app tests
```

- [ ] **Step 7: Commit**

```bash
git add docs/ README.md Dockerfile Makefile
git commit -m "docs: add architecture, design decisions, checklist, and readme"
```

---

## Task 17: Final verification & GitHub

- [ ] **Step 1: Green suite + lint one last time**

Run: `pytest --cov=app --cov-report=term-missing && ruff check app tests && black --check app tests`
Expected: all pass, coverage ≥ 80%, lint clean.

- [ ] **Step 2: Confirm GitHub with the user**

Ask whether to `gh repo create stt-endpoint --public --source=. --push` or push to an existing remote. Do not push until confirmed.

- [ ] **Step 3: Push**

After confirmation, create/push and report the repo URL.

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §2 providers + fallback → Tasks 5,6,7,8,11 ✅
- §3 architecture → Task 16 (docs) + realized across service/providers ✅
- §4 folder structure → Task 0 + subsequent files ✅
- §5 contracts (Provider ABC, TranscriptionResult, service, audio) → Tasks 1,3,9,10,11 ✅
- §6 API (transcribe/health/providers, error codes) → Task 12 ✅
- §7 polished Streamlit UI → Task 14 ✅
- §8 code-quality bar → enforced in every task (naming, why-comments, typed, ruff/black in Task 13/17) ✅
- §9 testing ≥80% mocked → every task is TDD; Task 13 gate ✅
- §10 Part-2 written answers → Task 16 (DESIGN_DECISIONS.md) ✅
- §11 deliverables → Tasks 14,15,16,17 ✅

**Placeholder scan:** Tasks 1–15 contain full code. Task 16 doc steps describe contents rather than embedding full prose — acceptable because docs are narrative, and their required contents/section list is explicit and traceable to spec sections.

**Type consistency:** `TranscriptionResult{language,text,segments,provider,duration}`, `Segment{start,end,text}`, `TranscriptionProvider.transcribe(audio_path, language)` / `is_available()`, `build_providers(settings)`, `TranscriptionService(providers, chunk_threshold_seconds, max_upload_mb).transcribe(path, language)`, `validate_audio(path, max_upload_mb)`, `merge_chunk_results(results, offsets)`, `split_on_silence(path, max_chunk_seconds)`, `probe_duration_seconds(path)`, `group_words_into_segments(words, pause_threshold)` — all consistent across tasks. ✅
