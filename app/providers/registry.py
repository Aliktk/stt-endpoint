from app.config import Settings
from app.providers.base import TranscriptionProvider
from app.providers.deepgram_provider import DeepgramProvider
from app.providers.elevenlabs_provider import ElevenLabsProvider
from app.providers.local_whisper_provider import LocalWhisperProvider


def build_providers(settings: Settings) -> list[TranscriptionProvider]:
    """Instantiate providers in the configured order, keeping only available ones.

    A provider is "available" when its credentials are present (or, for the
    local model, always). This is what makes the pipeline plug-and-play: drop
    in a key and the provider joins the chain; remove it and the chain simply
    skips past.
    """
    catalog: dict[str, TranscriptionProvider] = {
        "elevenlabs": ElevenLabsProvider(settings.elevenlabs_api_key),
        "deepgram": DeepgramProvider(settings.deepgram_api_key),
        "local": LocalWhisperProvider(settings.whisper_model_size, settings.whisper_compute_type),
    }
    ordered = [catalog[name] for name in settings.provider_order if name in catalog]
    return [p for p in ordered if p.is_available()]
