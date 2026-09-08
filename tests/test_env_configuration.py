from pathlib import Path

from models.registry import ModelRegistry


def test_model_tags_can_be_overridden_from_environment(monkeypatch):
    monkeypatch.setenv("AEGIS_MODEL_CODER", "local-coder:test")
    monkeypatch.setenv("AEGIS_OLLAMA_BASE_URL", "http://127.0.0.1:9999")
    registry = ModelRegistry.from_yaml(Path("config/models.yaml"))

    assert registry.get_provider("qwen-coder").model_id == "local-coder:test"
    assert registry.get_provider("qwen-coder").base_url == "http://127.0.0.1:9999"
