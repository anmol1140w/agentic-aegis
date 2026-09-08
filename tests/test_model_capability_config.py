from pathlib import Path

from models.base import ModelCapability
from models.registry import ModelRegistry


def test_config_capabilities_are_supported_and_coder_is_not_a_calculator():
    config_path = Path(__file__).parents[1] / "config" / "models.yaml"
    registry = ModelRegistry.from_yaml(config_path)
    supported = {item.value for item in ModelCapability}
    for config in registry.list_models():
        assert {cap.value for cap in config.capabilities} <= supported
    coder = registry.get_provider("qwen-coder").config
    assert ModelCapability.CODING in coder.capabilities
    assert ModelCapability.CODE_EXECUTION in coder.capabilities
    assert ModelCapability.CALCULATION not in coder.capabilities


def test_embedding_model_is_not_silently_treated_as_chat_model():
    # The registry currently constructs chat providers for every entry. Keep
    # embedding models out until a typed embedding provider/config path exists.
    assert not hasattr(ModelCapability, "EMBEDDING")
