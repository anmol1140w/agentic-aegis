import asyncio
from pathlib import Path
from models.registry import ModelRegistry
from runtime.orchestrator import Orchestrator

async def main():
    registry = ModelRegistry.from_yaml("config/models.yaml")
    availability = await registry.check_availability()
    orchestrator = Orchestrator(registry, availability=availability)

    print("--- 1. Invalid image path ---")
    try:
        res = await orchestrator.ainvoke("'/nonexistent/image.jpg' describe this", "fail-1")
        print(f"  Status: {res.get('current_step')}")
        print(f"  Error: {res.get('errors', ['none'])}")
        print(f"  Run ID: {res.get('run_id')}")
        if res.get('output_dir'):
            trace_path = Path(res['output_dir']) / "trace.json"
            print(f"  trace.json exists: {trace_path.exists()}")
    except Exception as exc:
        print(f"  Exception: {exc}")

    print("\n--- 2. Empty expression calculator ---")
    try:
        res = await orchestrator.ainvoke("calculate", "fail-2")
        print(f"  Status: {res.get('current_step')}")
        print(f"  Run ID: {res.get('run_id')}")
    except Exception as exc:
        print(f"  Exception: {exc}")

    print("\n--- 3. Model unavailable (force) ---")
    forced_avail = {k: False for k in availability}
    orch2 = Orchestrator(registry, availability=forced_avail)
    try:
        res = await orch2.ainvoke("explain quantum computing", "fail-3")
        print(f"  Status: {res.get('current_step')}")
    except Exception as exc:
        print(f"  Exception (expected): {type(exc).__name__}: {exc}")

asyncio.run(main())
