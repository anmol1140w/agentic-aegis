"""Bounded background-agent execution with status, cancellation and retry."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class Job:
    id: str
    status: str = "queued"
    attempts: int = 0
    result: Any = None
    error: str | None = None
    task: asyncio.Task[Any] | None = field(default=None, repr=False)


class BackgroundJobManager:
    def __init__(self, *, max_concurrency: int = 4) -> None:
        self.jobs: dict[str, Job] = {}
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def submit(self, operation: Callable[[], Awaitable[Any]], *, retries: int = 1) -> Job:
        job = Job(f"job_{uuid.uuid4().hex[:12]}")
        self.jobs[job.id] = job
        job.task = asyncio.create_task(self._run(job, operation, retries))
        return job

    async def _run(self, job: Job, operation: Callable[[], Awaitable[Any]], retries: int) -> None:
        async with self._semaphore:
            job.status = "running"
            for attempt in range(retries + 1):
                job.attempts = attempt + 1
                try:
                    job.result = await operation()
                    job.status = "success"
                    return
                except asyncio.CancelledError:
                    job.status = "cancelled"
                    raise
                except Exception as exc:
                    job.error = str(exc)
            job.status = "failed"

    def cancel(self, job_id: str) -> bool:
        job = self.jobs[job_id]
        if job.task and not job.task.done():
            job.task.cancel()
            return True
        return False

    def status(self, job_id: str) -> dict[str, Any]:
        job = self.jobs[job_id]
        return {"id": job.id, "status": job.status, "attempts": job.attempts, "result": job.result, "error": job.error}
