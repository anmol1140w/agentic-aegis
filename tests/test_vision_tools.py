import asyncio
from pathlib import Path

from PIL import Image

from tools.vision import VisionRuntime


class FakeVisionProvider:
    class Config:
        model = "qwen3-vl:8b"

    config = Config()

    def encode_images(self, paths):
        return ["encoded"] * len(paths)

    async def chat(self, messages, **kwargs):
        class Response:
            content = '{"observations":[{"label":"pump","text":"P-101","confidence":0.91}],"uncertain_text":[]}'
        return Response()


class SlowVisionProvider(FakeVisionProvider):
    async def chat(self, messages, **kwargs):
        await asyncio.sleep(0.05)
        return await super().chat(messages, **kwargs)


def test_local_image_returns_structured_observations(tmp_path: Path):
    image = tmp_path / "pid.png"
    Image.new("RGB", (300, 200), "white").save(image)
    result = asyncio.run(VisionRuntime(tmp_path, FakeVisionProvider(), timeout_seconds=2).analyze_image("pid.png", "inspect P&ID"))
    assert result["success"] is True
    assert result["observations"][0]["label"] == "pump"
    assert result["source"] == "pid.png"


def test_vision_path_escape_and_timeout_are_structured(tmp_path: Path):
    runtime = VisionRuntime(tmp_path, FakeVisionProvider(), timeout_seconds=1)
    assert asyncio.run(runtime.analyze_image("../secret.png"))["error"] == "OutsideWorkspace"
    image = tmp_path / "pid.png"
    Image.new("RGB", (20, 20), "white").save(image)
    result = asyncio.run(VisionRuntime(tmp_path, SlowVisionProvider(), timeout_seconds=0.01).analyze_image("pid.png"))
    assert result["success"] is False
    assert result["error"] == "vision_timeout"

