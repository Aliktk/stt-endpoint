# STT Endpoint

Plug-and-play speech-to-text pipeline. API-first (ElevenLabs → Deepgram) with an
automatic local **faster-whisper** fallback, exposed via a FastAPI service and a
polished Streamlit demo UI. Returns full transcripts with per-segment timestamps.

> Full documentation is added in the docs step of the build. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
> [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md).

## Quickstart

```bash
uv sync --extra dev --extra ui       # create env + install
uv run uvicorn app.main:app --reload # start the API
uv run streamlit run streamlit_app/app.py  # start the UI
uv run pytest --cov=app              # run tests
```

The service runs with **no API keys** (local model). Add keys in `.env` (copy
`.env.example`) to enable the cloud providers.
