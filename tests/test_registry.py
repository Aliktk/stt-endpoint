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
    assert [p.name for p in providers] == ["local"]


def test_registry_keeps_configured_order():
    settings = Settings(
        PROVIDER_ORDER="deepgram,elevenlabs,local",
        ELEVENLABS_API_KEY="a",
        DEEPGRAM_API_KEY="b",
    )
    assert [p.name for p in build_providers(settings)] == ["deepgram", "elevenlabs", "local"]
