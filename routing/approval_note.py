"""Deterministic PSU approval-note contract and grounded explanation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApprovalNote(BaseModel):
    """Canonical administrative approval-note fields for the MRPL workflow."""

    subject: str = ""
    background: str = ""
    proposal: str = ""
    financial_implication: str = ""
    dop_authority: str = ""
    recommendation: str = ""


APPROVAL_NOTE_FIELDS = tuple(ApprovalNote.model_fields)


def explain_approval_note() -> str:
    """Return the stable domain explanation used for explain-intent tasks."""
    return (
        "In the MRPL/PSU administrative context, a refinery approval note is an "
        "internal decision document seeking approval for a refinery work, purchase, "
        "repair, or project. It normally records the subject, background, proposal, "
        "financial implication, applicable Delegation of Power (DoP) authority, and "
        "recommendation. It is not a petroleum-product shipment or transaction authorization."
    )
