import asyncio
from pathlib import Path
from PIL import Image
from models.registry import ModelRegistry
from runtime.orchestrator import Orchestrator

async def main():
    registry = ModelRegistry.from_yaml("config/models.yaml")
    availability = await registry.check_availability()
    orchestrator = Orchestrator(registry, availability=availability)

    print("--- 1. Calculator ---")
    res1 = await orchestrator.ainvoke("calculate 5 * 12", "thread-1")
    print(f"Run ID: {res1.get('run_id')}")
    print(f"Output Dir: {res1.get('output_dir')}")

    print("\n--- 2. General Model ---")
    res2 = await orchestrator.ainvoke("Explain what a P&ID is in 1 sentence.", "thread-2")
    print(f"Run ID: {res2.get('run_id')}")
    print(f"Output Dir: {res2.get('output_dir')}")

    print("\n--- 3. Vision (Text-only) ---")
    res3 = await orchestrator.ainvoke("Reply only with: OK", "thread-3")
    print(f"Model: {res3.get('selected_model')}")
    print(f"Load Latency: {res3.get('image_load_ms', 0)}ms")
    print(f"Inference Latency: {res3.get('model_ms')}ms")

    print("\n--- 4. Vision (Image) ---")
    # create a dummy image if we don't have one
    img_path = Path("test_image.jpg")
    if not img_path.exists():
        img = Image.new('RGB', (7168, 4561), color = 'red')
        img.save(img_path)
        print("Created test image 7168x4561")

    res4 = await orchestrator.ainvoke(f"'{img_path.absolute()}' What color is this?", "thread-4")
    print(f"Model: {res4.get('selected_model')}")
    print(f"Load Latency: {res4.get('image_load_ms')}ms")
    print(f"Inference Latency: {res4.get('model_ms')}ms")
    print(f"Total Latency: {res4.get('total_ms')}ms")
    print(f"Run ID: {res4.get('run_id')}")
    print(f"Output Dir: {res4.get('output_dir')}")
    print(f"Output chars: len({len(res4.get('messages', [])[-1].content)})")

if __name__ == "__main__":
    asyncio.run(main())
