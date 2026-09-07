"""Approval-gated official document workflow."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from pipeline.docx_writer import write_text_document
from routing.approval_note import ApprovalNote


class OfficialDocumentState(BaseModel):
    request_id: str
    status: str = "draft"
    output_path: str | None = None
    approval_required: bool = True
    human_approved: bool = False
    error: str | None = None


class OfficialDocumentWorkflow:
    """Prepare, approve, and generate a deliverable in separate stages."""

    def __init__(self, deliverables_root: str | Path,
                 approval_requester: Callable[[str, str, str], bool] | None = None) -> None:
        self.root = Path(deliverables_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.approval_requester = approval_requester
        self.states: dict[str, OfficialDocumentState] = {}

    def prepare(self, note: ApprovalNote, *, request_id: str | None = None) -> OfficialDocumentState:
        state = OfficialDocumentState(request_id=request_id or f"doc_{uuid.uuid4().hex[:12]}")
        self.states[state.request_id] = state
        return state

    def generate(self, note: ApprovalNote, *, request_id: str, approve: bool = False,
                 filename: str = "official_approval_note.docx") -> dict[str, Any]:
        state = self.states.get(request_id) or self.prepare(note, request_id=request_id)
        if not approve:
            state.status = "approval_required"
            if self.approval_requester and self.approval_requester(request_id, "write_official_document", note.subject):
                approve = True
            else:
                return {"ok": False, "status": "approval_required", "request_id": request_id,
                        "approval_required": True, "human_approved": False}
        if not filename.endswith(".docx") or Path(filename).name != filename:
            state.status = "failed"
            state.error = "invalid_deliverable_filename"
            return {"ok": False, "status": state.status, "error": state.error}
        output = self.root / filename
        content = "\n".join([
            "# Official Approval Note", f"## Subject\n{note.subject}",
            f"## Background\n{note.background}", f"## Proposal\n{note.proposal}",
            f"## Financial Implication\n{note.financial_implication}",
            f"## Delegation of Power Authority\n{note.dop_authority}",
            f"## Recommendation\n{note.recommendation}",
        ])
        try:
            write_text_document(content, output)
        except Exception as exc:
            state.status = "failed"
            state.error = type(exc).__name__
            return {"ok": False, "status": state.status, "error": state.error}
        state.status = "generated"
        state.human_approved = True
        state.output_path = str(output)
    return {"ok": True, "status": state.status, "request_id": request_id,
            "approval_required": True, "human_approved": True, "path": str(output)}
