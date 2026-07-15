from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.errors import AllProvidersFailedError, AudioValidationError, UnsupportedFormatError
from app.schemas import ErrorResponse


def create_app() -> FastAPI:
    app = FastAPI(title="STT Endpoint", version="1.0.0")
    app.include_router(router)

    # UnsupportedFormatError subclasses AudioValidationError; register it first
    # so the 422 handler wins over the generic 400 for format problems.
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
