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
