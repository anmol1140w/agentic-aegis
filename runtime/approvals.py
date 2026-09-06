"""Human-in-the-loop approval manager (terminal-based).

For the terminal agent, approval is a simple console prompt.
Future versions can replace this with an API-backed approval queue.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalRequest(BaseModel):
    """A pending approval request."""

    task_id: str
    action: str              # what the agent wants to do
    details: str = ""        # human-readable description
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: str | None = None


class ApprovalManager:
    """Manages human approval requests.

    Terminal mode:  blocks on console input.
    API mode (future):  stores request, returns pending, polls for response.
    """

    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}

    def request_approval(
        self,
        task_id: str,
        action: str,
        details: str = "",
    ) -> ApprovalRequest:
        """Create a new pending approval request."""
        req = ApprovalRequest(
            task_id=task_id, action=action, details=details,
        )
        self._requests[task_id] = req
        return req

    def approve(self, task_id: str) -> None:
        req = self._requests.get(task_id)
        if req:
            req.status = ApprovalStatus.APPROVED
            req.resolved_at = datetime.now(timezone.utc).isoformat()

    def reject(self, task_id: str) -> None:
        req = self._requests.get(task_id)
        if req:
            req.status = ApprovalStatus.REJECTED
            req.resolved_at = datetime.now(timezone.utc).isoformat()

    def get_status(self, task_id: str) -> ApprovalStatus | None:
        req = self._requests.get(task_id)
        return req.status if req else None

    def prompt_terminal(self, task_id: str, action: str, details: str) -> bool:
        """Synchronous terminal prompt.  Returns True if approved."""
        self.request_approval(task_id, action, details)
        # Rich console formatting is done by the caller (cli.py)
        try:
            response = input(f"\n⚠  Approve: {action}?\n   {details}\n   [y/N] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = ""
        if response in ("y", "Y", "Yes", "YES", "yes" ):
            self.approve(task_id)
            return True
        self.reject(task_id)
        return False
