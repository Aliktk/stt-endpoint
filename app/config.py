from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")
    deepgram_api_key: str | None = Field(default=None, alias="DEEPGRAM_API_KEY")

    provider_order: list[str] = Field(
        default=["elevenlabs", "deepgram", "local"], alias="PROVIDER_ORDER"
    )

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
