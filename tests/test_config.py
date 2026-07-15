from app.config import Settings


def test_provider_order_parses_to_list():
    settings = Settings(PROVIDER_ORDER="elevenlabs,local", ELEVENLABS_API_KEY="k")
    assert settings.provider_order == ["elevenlabs", "local"]


def test_defaults_are_sane_with_no_env():
    settings = Settings()
    assert "local" in settings.provider_order
    assert settings.max_upload_mb > 0
    assert settings.chunk_threshold_seconds > 0
