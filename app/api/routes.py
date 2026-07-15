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
