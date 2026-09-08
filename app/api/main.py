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
from aegis.contracts import ExecutionPlan, PlanValidator, WorkflowRegistry
from aegis.governance import AuditChain, PolicyEngine, Principal
from aegis.memory import MemoryStore
from aegis.retrieval import HybridKnowledgeStore
from aegis.evaluation import EvaluationSuite
from aegis.sovereign import KubernetesPlanner, SovereignExecutor, VisualWorkflowEditor
from runtime.background import BackgroundJobManager


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
        self.workflows = WorkflowRegistry()
        self.plan_validator = PlanValidator()
        self.policy = PolicyEngine()
        self.audit = AuditChain("logs/aegis-chain.jsonl")
        self.memory = MemoryStore(".aegis/memory.sqlite3")
        self.knowledge = HybridKnowledgeStore()
        self.jobs = BackgroundJobManager()
        self.evaluations = EvaluationSuite(".aegis/evaluations.jsonl")
        # The API runs the deterministic capability route by default. RL remains
        # an explicitly enabled experimental strategy, never a hidden fallback.
        self.sovereign = SovereignExecutor(self.plan_validator, self.audit)

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


def _principal(subject: str = "local") -> Principal:
    # Authentication providers can replace this boundary with OIDC/LDAP/SCIM.
    return Principal(subject=subject, roles=frozenset({"admin"}))


@app.post("/api/workflows/validate")
async def validate_workflow(plan: ExecutionPlan) -> dict[str, Any]:
    errors = service.plan_validator.validate(plan)
    service.audit.append("workflow_validated", task_id=plan.task_id, valid=not errors, errors=errors)
    return {"valid": not errors, "errors": errors}


@app.post("/api/workflows")
async def register_workflow(name: str, plan: ExecutionPlan, version: str | None = None) -> dict[str, Any]:
    service.policy.check(_principal(), "workflow:run")
    service.plan_validator.require_valid(plan)
    saved = service.workflows.register(name, plan, version=version)
    service.audit.append("workflow_registered", workflow=name, version=saved.version)
    return {"name": name, "version": saved.version, "steps": len(saved.steps)}


@app.get("/api/workflows")
async def list_workflows() -> list[dict[str, str]]:
    return service.workflows.list()


@app.get("/api/workflows/{name}/{version}/visual")
async def visual_workflow(name: str, version: str) -> dict[str, Any]:
    return VisualWorkflowEditor.export(service.workflows.get(name, version))


@app.post("/api/workflows/execute")
async def execute_sovereign_workflow(plan: ExecutionPlan, approve_high_risk: bool = False,
                                     attestation: dict[str, str] | None = None) -> dict[str, Any]:
    """Execute an explicit typed-plan integration contract.

    Normal user tasks use ``POST /api/tasks`` and the production
    ``Orchestrator.run_master`` path. This endpoint is retained for typed-plan
    clients and uses only its deliberately narrow demo executor.
    """
    service.policy.check(_principal(), "workflow:run")

    def demo_executor(step):
        return {"ok": True, "status": "success", "tool": step.tool or step.capability.value,
                "evidence": {"mode": "api-demo", "step": step.id}}

    return await service.sovereign.execute(plan, executor=demo_executor,
                                           approval=lambda _risk, _step: approve_high_risk,
                                           attestation=attestation)


@app.get("/api/kubernetes/manifest")
async def kubernetes_manifest(name: str = "aegis-worker", image: str = "aegis:local",
                              replicas: int = 1, tee_required: bool = False) -> dict[str, Any]:
    return KubernetesPlanner.manifest(name, image, replicas=replicas, tee_required=tee_required)


@app.post("/api/knowledge")
async def add_knowledge(payload: dict[str, Any]) -> dict[str, Any]:
    service.policy.check(_principal(), "rag:read")
    ids = service.knowledge.add(str(payload.get("content", "")), str(payload.get("source", "api")), metadata=payload.get("metadata"))
    return {"chunk_ids": ids}


@app.post("/api/knowledge/search")
async def search_knowledge(payload: dict[str, Any]) -> list[dict[str, Any]]:
    service.policy.check(_principal(), "rag:read")
    results = await service.knowledge.search(str(payload.get("query", "")), top_k=int(payload.get("top_k", 5)))
    return [item.model_dump() for item in results]


@app.post("/api/memory")
async def save_memory(payload: dict[str, Any]) -> dict[str, Any]:
    service.policy.check(_principal(), "task:create")
    memory_id = service.memory.put(str(payload.get("text", "")), scope=str(payload.get("scope", "global")), metadata=payload.get("metadata"), ttl_seconds=payload.get("ttl_seconds"))
    return {"id": memory_id}


@app.get("/api/memory/search")
async def search_memory(query: str, scope: str | None = None) -> list[dict[str, Any]]:
    service.policy.check(_principal(), "task:read")
    return service.memory.search(query, scope=scope)


@app.get("/api/audit/verify")
async def verify_audit() -> dict[str, bool]:
    return {"valid": service.audit.verify()}


@app.post("/api/jobs")
async def create_background_job(payload: TaskRequest) -> dict[str, str]:
    service.policy.check(_principal(), "task:create")

    async def run() -> dict[str, Any]:
        execution_id = await service.submit(payload)
        return execution_id

    job = service.jobs.submit(run, retries=int(payload.options.get("retries", 1)))
    service.audit.append("background_job_created", job_id=job.id)
    return {"job_id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}")
async def background_job_status(job_id: str) -> dict[str, Any]:
    try:
        return service.jobs.status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.delete("/api/jobs/{job_id}")
async def cancel_background_job(job_id: str) -> dict[str, bool]:
    try:
        return {"cancelled": service.jobs.cancel(job_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/evaluations/summary")
async def evaluation_summary() -> dict[str, float]:
    return service.evaluations.summary()
