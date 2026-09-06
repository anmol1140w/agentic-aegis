"""Thin HTTP boundary for the existing AEGIS orchestration runtime."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from models.registry import ModelRegistry
from runtime.orchestrator import Orchestrator
from storage.outputs import OutputStore


class TaskRequest(BaseModel):
    request: str = Field(min_length=1, max_length=20_000)
    files: list[dict[str, str]] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class WorkbenchService:
    def __init__(self) -> None:
        registry = ModelRegistry.from_yaml("config/models.yaml")
        self.registry = registry
        self.orchestrator = Orchestrator(registry)
        self.tasks: dict[str, dict[str, Any]] = {}

    async def health(self) -> dict[str, Any]:
        return {"status": "ok", "service": "aegis", "models": await self.registry.check_availability()}

    async def submit(self, payload: TaskRequest) -> dict[str, Any]:
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        events: list[dict[str, Any]] = [{"type": "task_started", "execution_id": execution_id}]
        self.tasks[execution_id] = {"execution_id": execution_id, "status": "queued", "events": events}
        asyncio.create_task(self._run(execution_id, payload))
        return {"execution_id": execution_id, "status": "queued"}

    async def _run(self, execution_id: str, payload: TaskRequest) -> None:
        record = self.tasks[execution_id]
        record["status"] = "running"
        record["stage"] = "master"

        def progress(event: dict[str, Any]) -> None:
            # Keep API events high-level; raw prompts and private reasoning are
            # deliberately not exposed.
            item = {"execution_id": execution_id, "type": str(event.get("event", "progress")).lower(),
                    "agent": event.get("agent"), "message": str(event.get("event", ""))}
            record["events"].append(item)

        try:
            result = await self.orchestrator.run_master(
                payload.request,
                context={"api_execution_id": execution_id, "files": payload.files, "options": payload.options},
                progress_callback=progress,
            )
            record.update({"status": "success" if not result.get("errors") else "failed", "stage": "final", "result": result})
            record["events"].append({"execution_id": execution_id,
                                     "type": "task_completed" if record["status"] == "success" else "task_failed"})
        except Exception as exc:
            record.update({"status": "failed", "stage": "final", "error": str(exc)[:500]})
            record["events"].append({"execution_id": execution_id, "type": "task_failed", "message": str(exc)[:300]})


service = WorkbenchService()
app = FastAPI(title="AEGIS API", version="0.1.0")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return await service.health()


@app.post("/api/tasks", status_code=202)
async def create_task(payload: TaskRequest) -> dict[str, Any]:
    return await service.submit(payload)


@app.get("/api/tasks/{execution_id}")
async def task_status(execution_id: str) -> dict[str, Any]:
    task = service.tasks.get(execution_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    return {k: v for k, v in task.items() if k != "events"}


@app.get("/api/tasks/{execution_id}/events")
async def task_events(execution_id: str) -> StreamingResponse:
    if execution_id not in service.tasks:
        raise HTTPException(status_code=404, detail="Execution not found")

    async def stream():
        sent = 0
        while True:
            task = service.tasks[execution_id]
            events = task["events"]
            while sent < len(events):
                yield f"data: {json.dumps(events[sent], ensure_ascii=False)}\n\n"
                sent += 1
            if task.get("status") in {"success", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream")
