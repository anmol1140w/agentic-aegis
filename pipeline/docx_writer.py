"""Deterministic DOCX approval-note writer using python-docx.

Generates a structured Word document from inspection findings JSON.
Labels output as 'Draft — Human Approval Required'.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


class DocxWriterError(Exception):
    pass


def write_text_document(content: str, output_path: Path | str) -> Path:
    """Write bounded plain/markdown-like content while preserving structure."""
    if not HAS_DOCX:
        raise DocxWriterError("python-docx is not installed")
    if not isinstance(content, str) or not content.strip():
        raise DocxWriterError("content_generation_failed: document content is empty")
    output_path = Path(output_path)
    doc = Document()
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        bullet = re.match(r"^[-*]\s+(.+)$", line)
        numbered = re.match(r"^\d+[.)]\s+(.+)$", line)
        if heading:
            doc.add_heading(heading.group(2).strip(), level=min(len(heading.group(1)), 3))
        elif bullet:
            doc.add_paragraph(bullet.group(1).strip(), style="List Bullet")
        elif numbered:
            doc.add_paragraph(numbered.group(1).strip(), style="List Number")
        else:
            doc.add_paragraph(line)
    if not any(p.text.strip() for p in doc.paragraphs):
        raise DocxWriterError("content_generation_failed: no document elements")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    return output_path


def _add_heading(doc: Any, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def _add_para(doc: Any, text: str, bold: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold


def write_approval_note(
    findings: dict[str, Any],
    output_path: Path | str,
) -> Path:
    """Generate an approval_note.docx from structured findings.

    Parameters
    ----------
    findings:
        The structured findings dict (matching the findings JSON schema).
    output_path:
        Where to write the .docx file.

    Returns
    -------
    Path to the written .docx file.

    Raises
    ------
    DocxWriterError
        If python-docx is unavailable or writing fails.
    """
    if not HAS_DOCX:
        raise DocxWriterError(
            "python-docx is not installed. Install with: pip install python-docx"
        )

    output_path = Path(output_path)
    try:
        doc = Document()

        # Title
        title = doc.add_heading("Inspection Report — Approval Note", level=0)
        # Draft watermark
        draft_para = doc.add_paragraph()
        draft_run = draft_para.add_run("DRAFT — HUMAN APPROVAL REQUIRED")
        draft_run.bold = True
        draft_run.font.size = Pt(14)
        draft_run.font.color.rgb = RGBColor(0xFF, 0x00, 0x00)
        draft_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

        doc.add_paragraph("")  # spacer

        # Report Metadata
        _add_heading(doc, "Report Metadata", level=1)
        meta = findings.get("report_metadata", {})
        meta_table = doc.add_table(rows=0, cols=2)
        meta_table.style = "Light List Accent 1"
        for key, label in [
            ("report_title", "Report Title"),
            ("inspection_date", "Inspection Date"),
            ("site", "Site"),
            ("unit", "Unit"),
            ("inspector", "Inspector"),
        ]:
            val = meta.get(key) or findings.get(key, "N/A")
            if val is None:
                val = "N/A"
            row = meta_table.add_row()
            row.cells[0].text = label
            row.cells[1].text = str(val)

        doc.add_paragraph("")  # spacer

        # Key Findings Table
        _add_heading(doc, "Key Findings", level=1)
        items = findings.get("findings", [])
        if items:
            ft = doc.add_table(rows=1, cols=6)
            ft.style = "Light List Accent 1"
            headers = ["ID", "Description", "Location/Asset", "Severity", "Evidence", "Page"]
            for i, h in enumerate(headers):
                ft.rows[0].cells[i].text = h

            for f in items:
                row = ft.add_row()
                row.cells[0].text = str(f.get("id", ""))
                row.cells[1].text = str(f.get("description", ""))
                row.cells[2].text = str(f.get("location_or_asset", "N/A"))
                row.cells[3].text = str(f.get("severity", "unknown"))
                row.cells[4].text = str(f.get("evidence", ""))[:200]
                row.cells[5].text = str(f.get("page", ""))
        else:
            _add_para(doc, "No findings extracted.", bold=True)

        doc.add_paragraph("")  # spacer

        # Risk / Severity Summary
        _add_heading(doc, "Risk and Severity Assessment", level=1)
        nonconformities = findings.get("nonconformities", [])
        if nonconformities:
            for nc in nonconformities:
                doc.add_paragraph(f"• {nc}", style="List Bullet")
        else:
            _add_para(doc, "No nonconformities identified.")

        doc.add_paragraph("")

        # Corrective Actions
        _add_heading(doc, "Required Corrective Actions", level=1)
        actions = findings.get("corrective_actions", [])
        if actions:
            for a in actions:
                doc.add_paragraph(f"• {a}", style="List Bullet")
        else:
            _add_para(doc, "No corrective actions specified.")

        doc.add_paragraph("")

        # Approval Recommendation
        _add_heading(doc, "Approval Recommendation", level=1)
        recommendation = findings.get("approval_recommendation", "Manual review required")
        _add_para(doc, str(recommendation))

        doc.add_paragraph("")

        # Assumptions and Uncertainties
        _add_heading(doc, "Assumptions and Uncertainties", level=1)
        uncertainties = findings.get("uncertainties", [])
        if uncertainties:
            for u in uncertainties:
                doc.add_paragraph(f"• {u}", style="List Bullet")

        missing = findings.get("missing_information", [])
        if missing:
            _add_heading(doc, "Missing or Unreadable Information", level=2)
            for m in missing:
                doc.add_paragraph(f"• {m}", style="List Bullet")

        doc.add_paragraph("")

        # Human Approval / Signature Section
        _add_heading(doc, "Human Approval", level=1)
        _add_para(doc, "This document requires human review and approval before any action is taken.")
        doc.add_paragraph("")
        sig_table = doc.add_table(rows=3, cols=2)
        sig_table.style = "Light List Accent 1"
        sig_table.rows[0].cells[0].text = "Reviewed by"
        sig_table.rows[0].cells[1].text = "________________________"
        sig_table.rows[1].cells[0].text = "Date"
        sig_table.rows[1].cells[1].text = "________________________"
        sig_table.rows[2].cells[0].text = "Signature"
        sig_table.rows[2].cells[1].text = "________________________"

        doc.add_paragraph("")
        footer_para = doc.add_paragraph()
        footer_run = footer_para.add_run(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
            "Automated extraction — Draft only"
        )
        footer_run.font.size = Pt(8)
        footer_run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

        doc.save(str(output_path))
        return output_path

    except Exception as exc:
        raise DocxWriterError(f"Failed to write approval note: {exc}") from exc
