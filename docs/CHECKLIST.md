# Checklist

## Assessment requirements → where they are satisfied

| Requirement | Status | Where |
|-------------|:------:|-------|
| Accepts an audio file (WAV/MP3/…) | ✅ | `POST /v1/transcribe`, `app/api/routes.py`; formats in `app/services/audio.py` |
| Transcribes speech to text | ✅ | `TranscriptionService` + providers |
| Returns timestamps per segment | ✅ | `Segment{start,end,text}` in every `TranscriptionResult` |
| Handles different audio formats | ✅ | ffmpeg decode + extension whitelist (`SUPPORTED_EXTENSIONS`) |
| Handles long audio files | ✅ | `split_on_silence` + `merge_chunk_results`; verified in `tests/test_audio_integration.py` |
| API-first with local fallback | ✅ | ElevenLabs → Deepgram → local chain (`registry.py`, `transcription_service.py`) |
| Plug-and-play (runs with no keys) | ✅ | `build_providers` availability filter; local model default |
| Demo UI | ✅ | `streamlit_app/app.py` |
| Concurrency — design | ✅ | `docs/DESIGN_DECISIONS.md` |
| Storage — design | ✅ | `docs/DESIGN_DECISIONS.md` |
| Retry/recovery — design | ✅ | `docs/DESIGN_DECISIONS.md` |
| API exposure — design | ✅ | `docs/DESIGN_DECISIONS.md` |
| README with design decisions | ✅ | `README.md`, `docs/` |
| Source in a git repo | ✅ | conventional commits throughout |

## Code quality

- [x] Meaningful names; functions read as verbs, values as nouns
- [x] Comments explain *why*, not *what*; docstrings only where the contract isn't obvious
- [x] Small, cohesive modules (one responsibility each, well under the 800-line cap)
- [x] Typed signatures throughout; `pathlib`, f-strings, early returns
- [x] Explicit typed exceptions; no bare `except` that swallows
- [x] No dead code, no speculative abstractions (YAGNI)
- [x] `ruff` clean, `black` formatted

## Testing

- [x] TDD throughout — tests written before implementation
- [x] Providers mocked; no network or API keys needed in CI
- [x] Unit coverage: schemas, config, grouping, providers, validation, chunk/merge, service, API
- [x] Real-audio integration test for silence chunking (`tests/test_audio_integration.py`)
- [x] Coverage ≥ 80% (currently ~95%)

## Run it

```bash
uv sync --extra dev --extra ui        # install
uv run pytest --cov=app               # tests + coverage
uv run ruff check app tests && uv run black --check app tests   # lint
uv run uvicorn app.main:app --reload --port 8080   # API on :8080
uv run streamlit run streamlit_app/app.py   # UI
```
