"""Deterministic decomposition for common SIH compound workflows."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CompoundStage:
    name: str
    capability: str
    objective: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decompose_task(request: str) -> list[CompoundStage]:
    text = request.casefold()
    stages: list[CompoundStage] = []
    wants_presentation = bool(re.search(r"\b(pptx?|powerpoint|presentation|slide deck|slides?)\b", text))
    wants_spreadsheet = bool(re.search(r"\b(xlsx|excel|spreadsheet|workbook|budget tracker|expense tracker)\b", text))
    if wants_spreadsheet:
        stages.append(CompoundStage("spreadsheet_generation", "spreadsheet_generation", "Create and validate the Excel workbook", "validated XLSX artifact"))
    if wants_presentation:
        stages.append(CompoundStage("presentation_generation", "presentation_generation", "Create and validate the PowerPoint presentation", "validated PPTX artifact"))
    if stages:
        return stages
    scanned = bool(re.search(r"\b(scanned|ocr|inspection report|pdf)\b", text))
    engineering = bool(re.search(r"\b(p&id|piping|valve|engineering drawing|inspection)\b", text))
    calculation = bool(re.search(r"\b(calculate|remaining life|corrosion rate|thickness|pressure)\b", text))
    approval = bool(re.search(r"\b(approval note|office note|official document|recommendation)\b", text))
    if scanned:
        stages.append(CompoundStage("extract", "ocr/layout_extraction", "Extract source-labelled evidence", "confidence and page metadata"))
    if engineering:
        stages.append(CompoundStage("interpret_visual", "vision/engineering_document", "Interpret engineering symbols and tags", "structured tags and citations"))
    if calculation:
        stages.append(CompoundStage("calculate", "calculation", "Perform the requested engineering calculation", "formula, inputs, and result"))
    if approval:
        stages.append(CompoundStage("draft_approval", "psu_approval_note", "Populate the canonical approval-note schema", "all required note fields"))
    if len(stages) > 1:
        stages.append(CompoundStage("verify", "verification", "Verify evidence, calculations, and citations", "deterministic verification result"))
    return stages
