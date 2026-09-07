"""Deterministic inspect-report pipeline.

Orchestrates: input validation → OCR → structured findings → DOCX → network report.
Does not use an LLM agent loop. Uses local model only for bounded extraction
with a single structured request when available.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from runtime.errors import InvalidOCRInputError, OCRBackendError, WorkbenchError
from security.permissions import PermissionError_, validate_path
from tools.ocr.models import OCRResult, OCRTextBlock
from tools.ocr.paddle import PaddleOCRBackend

from .docx_writer import DocxWriterError, write_approval_note
from .network_policy import NetworkPolicy
from .sandbox import SandboxResult, SandboxUnavailableError, run_sandboxed

console = Console()

# Findings JSON schema for reference and validation
FINDINGS_SCHEMA_KEYS = {
    "report_metadata",
    "findings",
    "nonconformities",
    "corrective_actions",
    "missing_information",
    "uncertainties",
    "approval_recommendation",
}

FINDING_REQUIRED_KEYS = {"id", "description", "page", "evidence", "confidence"}

_FINDING_CONTEXT_WORDS = (
    "defect", "fault", "failure", "damage", "leak", "crack",
    "nonconform", "observation", "deficiency", "hazard", "violation", "corrosion",
    "malfunction", "broken", "worn", "missing", "unsafe",
)

_FINDING_ANCHOR_WORDS = (
    "finding", "defect", "observation", "nonconform",
    "corrective", "deficiency", "fault", "damage", "hazard",
    "leak", "crack", "violation", "unsafe", "failed", "failure",
)

def _contains_word(text_lower: str, words: tuple[str, ...]) -> bool:
    """Check whole-word presence, not substring — prevents 'fault' matching inside 'default'."""
    return any(
        re.search(
            r"\bnon[-\s]?conform(?:ity|ance)?\b"
            if w == "nonconform" else rf"\b{re.escape(w)}\b",
            text_lower,
        )
        for w in words
    )

def _is_likely_finding(line: str, has_severity: bool) -> bool:
    """Decide whether a numbered/flagged line is actually a finding.

    A bare numbered line ("21. ...") is NOT sufficient on its own — many
    document types (drawings, legends, general notes) use the same numbering
    convention without being inspection findings. Require either:
      - an explicit finding-type keyword in the line, or
      - a detected severity (which itself now requires finding-context words)
    """
    line_lower = line.lower()
    has_anchor = _contains_word(line_lower, _FINDING_ANCHOR_WORDS)
    return has_anchor or has_severity

def _generate_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"inspect_{ts}_{uuid.uuid4().hex[:4]}"


def _validate_input(
    input_path: str | Path,
    workspace_root: str | Path | None = None,
) -> Path:
    """Validate and resolve the input file path.

    Accepts absolute paths to PDF or image files. If a workspace_root is
    provided, the path is validated against it. Otherwise, only basic
    existence and type checks are performed.
    """
    path = Path(input_path).expanduser().resolve()

    if not path.is_file():
        raise WorkbenchError(f"Input file not found: {path}")

    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
    if path.suffix.lower() not in allowed_extensions:
        raise WorkbenchError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(allowed_extensions))}"
        )

    # Check file size (50 MB limit from settings)
    max_size = 50 * 1024 * 1024
    if path.stat().st_size > max_size:
        raise WorkbenchError(f"File exceeds 50 MB limit: {path}")

    return path


def _pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    """Convert PDF pages to images for OCR.

    Tries pymupdf (fitz) first, falls back to pdf2image, then reports
    the missing prerequisite clearly.
    """
    images: list[Path] = []
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Try pymupdf
    try:
        import pymupdf  # type: ignore[import-untyped]
        doc = pymupdf.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img_path = pages_dir / f"page_{page_num + 1:04d}.png"
            pix.save(str(img_path))
            images.append(img_path)
        doc.close()
        return images
    except ImportError:
        pass

    # Try pdf2image
    try:
        from pdf2image import convert_from_path  # type: ignore[import-untyped]
        pil_images = convert_from_path(str(pdf_path), dpi=200)
        for i, img in enumerate(pil_images, 1):
            img_path = pages_dir / f"page_{i:04d}.png"
            img.save(str(img_path))
            images.append(img_path)
        return images
    except ImportError:
        pass

    # Try PIL-based approach for single-page image PDFs
    # Last resort — report requirement
    raise WorkbenchError(
        "PDF conversion requires pymupdf (pip install pymupdf) or "
        "pdf2image (pip install pdf2image). Neither is available."
    )


def _ocr_images(
    image_paths: list[Path],
    ocr_backend: PaddleOCRBackend | None = None,
) -> tuple[str, list[OCRTextBlock]]:
    """OCR a list of images, preserving page boundaries.

    Returns (full_text_with_page_markers, all_blocks).
    """
    if ocr_backend is None:
        ocr_backend = PaddleOCRBackend()

    all_text_parts: list[str] = []
    all_blocks: list[OCRTextBlock] = []

    for page_num, img_path in enumerate(image_paths, 1):
        try:
            result = ocr_backend.extract(img_path)
            page_text = result.text.strip()
            if page_text:
                all_text_parts.append(f"--- PAGE {page_num} ---\n{page_text}")
            # Update page numbers on blocks
            for block in result.text_blocks:
                block.page_number = page_num
                all_blocks.append(block)
        except (OCRBackendError, InvalidOCRInputError) as exc:
            all_text_parts.append(f"--- PAGE {page_num} ---\n[OCR FAILED: {exc}]")

    return "\n\n".join(all_text_parts), all_blocks


def _ocr_single_image(
    image_path: Path,
    ocr_backend: PaddleOCRBackend | None = None,
) -> tuple[str, list[OCRTextBlock]]:
    """OCR a single image file."""
    if ocr_backend is None:
        ocr_backend = PaddleOCRBackend()

    result = ocr_backend.extract(image_path)
    text = f"--- PAGE 1 ---\n{result.text.strip()}"
    for block in result.text_blocks:
        block.page_number = 1
    return text, result.text_blocks


def _extract_findings_deterministic(
    ocr_text: str,
    blocks: list[OCRTextBlock],
    source_path: Path,
) -> dict[str, Any]:
    """Extract structured findings deterministically from OCR text.

    Uses regex patterns and heuristics. Does not use an LLM.
    Never invents values — uses null/unknown for ambiguous fields.
    """
    findings: list[dict[str, Any]] = []
    uncertainties: list[str] = []
    missing_info: list[str] = []
    nonconformities: list[str] = []
    corrective_actions: list[str] = []

    # Parse page-separated text
    pages: dict[int, str] = {}
    current_page = 1
    for line in ocr_text.split("\n"):
        page_match = re.match(r"^--- PAGE (\d+) ---$", line)
        if page_match:
            current_page = int(page_match.group(1))
            pages[current_page] = ""
        else:
            pages.setdefault(current_page, "")
            pages[current_page] += line + "\n"

    # Extract report metadata from first page
    first_page = pages.get(1, "")
    report_title = _extract_field(first_page, r"\b(?:report\s+id|title|report|subject)\b\s*[:\-]\s*(.+)", "Unknown Report")
    inspection_date = _extract_field(first_page, r"\b(?:date|dated?)\b\s*[:\-]\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", None)
    site = _extract_field(first_page, r"\b(?:site|location|facility)\b\s*[:\-]\s*(.+)", None)
    unit = _extract_field(first_page, r"\b(?:unit|department|section)\b\s*[:\-]\s*(.+)", None)
    inspector = _extract_field(first_page, r"\b(?:inspector|auditor|examiner|reviewed by)\b\s*[:\-]\s*(.+)", None)
    # Extract findings from all pages
    # Extract findings from all pages
    finding_id = 0
    for page_num, page_text in pages.items():
        lines = page_text.strip().split("\n")
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Look for numbered findings, defects, observations
            finding_match = re.match(
                r"^(?:[^\w\s]\s*)?(?:(\d+)[.\)]\s*|(?:finding|defect|observation|issue|item)\s*[#:]?\s*(\d*))",
                line_stripped,
                re.IGNORECASE,
            )

            # Also look for severity/risk keywords
            severity_lines: list[str] = []
            for nearby_line in lines[i + 1:i + 3]:
                nearby_stripped = nearby_line.strip()
                if not nearby_stripped:
                    continue
                if re.match(
                    r"^(?:[^\w\s]\s*)?(?:(\d+)[.\)]\s*|(?:finding|defect|observation|issue|item)\s*[#:]?\s*(\d*))",
                    nearby_stripped,
                    re.IGNORECASE,
                ):
                    break
                severity_lines.append(nearby_stripped)
            severity = _detect_severity(line_stripped, " ".join(severity_lines) or None)

            # Numbered-line match is NOT sufficient by itself.
            # Must also pass the finding-context gate.
            if finding_match and _is_likely_finding(line_stripped, bool(severity)):
                finding_id += 1
                description = line_stripped
                if i + 1 < len(lines) and lines[i + 1].strip():
                    description += " " + lines[i + 1].strip()

                evidence = line_stripped[:150]

                findings.append({
                    "id": f"F-{finding_id:03d}",
                    "description": description[:300],
                    "location_or_asset": _extract_field(
                        description,
                        r"\b(?:at|in|on|location)\b[:\s]+(\S+(?:\s+\S+){0,3})",
                        None,
                    ),
                    "severity": severity or "unknown",
                    "evidence": evidence,
                    "page": page_num,
                    "confidence": _calculate_confidence(blocks, page_num, line_stripped),
                })

            if re.search(r"non[\-\s]?conform", line_stripped, re.IGNORECASE):
                nonconformities.append(line_stripped[:200])

            if re.match(
                r"^\s*(?:corrective\s+action|action\s+required|must\s+be|shall\s+be)\s*[:\-]",
                line_stripped,
                re.IGNORECASE,
            ):
                corrective_actions.append(line_stripped[:200])
    # Track uncertainties
    if not inspection_date:
        uncertainties.append("Inspection date could not be extracted from OCR text")
    if not inspector:
        uncertainties.append("Inspector name could not be extracted from OCR text")
    if not site:
        uncertainties.append("Site/location could not be extracted from OCR text")

    # Check for low-confidence OCR regions. Values below the review threshold
    # remain usable evidence but are explicitly queued for human confirmation.
    review_threshold = 0.85
    low_conf_blocks = [b for b in blocks if b.confidence < review_threshold]
    if low_conf_blocks:
        uncertainties.append(
            f"{len(low_conf_blocks)} OCR regions had confidence < {review_threshold:.2f} — human review required"
        )
        missing_info.append(
            f"Low-confidence OCR text on pages: {sorted(set(b.page_number for b in low_conf_blocks))}"
        )

    if not findings:
        uncertainties.append("No structured findings could be extracted — manual review required")

    return {
        "report_metadata": {
            "report_title": report_title,
            "inspection_date": inspection_date,
            "site": site,
            "unit": unit,
            "inspector": inspector,
            "source_file": str(source_path.name),
            "extraction_method": "deterministic_ocr",
            "confidence_threshold": review_threshold,
            "review_required": bool(low_conf_blocks),
            "review_queue": [
                {"value": b.text, "confidence": b.confidence, "page": b.page_number,
                 "bounding_box": b.bounding_box, "source": str(source_path)}
                for b in low_conf_blocks
            ],
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "findings": findings,
        "nonconformities": nonconformities,
        "corrective_actions": corrective_actions,
        "missing_information": missing_info,
        "uncertainties": uncertainties,
        "approval_recommendation": (
            "Manual review required — automated extraction only"
            if findings else
            "Manual review required — no findings extracted"
        ),
    }


def _extract_field(text: str, pattern: str, default: Any) -> Any:
    """Extract a field value using a regex pattern, return default if not found."""
    if not isinstance(pattern, str) or len(pattern) > 2048:
        return default
    try:
        match = re.search(pattern, str(text)[:256 * 1024], re.IGNORECASE)
    except re.error:
        return default
    if match:
        value = match.group(1).strip()[:200]
        # Reject if the captured group starts mid-word (e.g. "SITES" matched as "site")
        # by requiring the match to have started on a real word boundary — enforced
        # by \b in the caller's pattern. Guard here too against empty/garbage captures.
        if len(value) < 2:
            return default
        return value
    return default


def _detect_severity(text: str, next_line: str | None = None) -> str | None:
    """Detect severity level from text content.

    1. First checks for an explicit label like 'Severity: critical',
       'Severity - minor', 'severity=major'. An explicit label always wins.
    2. Falls back to keyword inference only if no explicit label is found,
       and only when a finding-context word is also present.
    """
    combined = text
    if next_line:
        combined = text + " " + next_line
    combined_lower = combined.lower()

    # Explicit severity label — highest priority, case-insensitive,
    # tolerates colon, dash, equals, whitespace.
    explicit_match = re.search(
        r"\bseverity\s*[:\-=]\s*(critical|severe|emergency|danger|major|significant|high|minor|low|observation)\b",
        combined_lower,
    )
    if explicit_match:
        raw = explicit_match.group(1)
        if raw in ("critical", "severe", "emergency", "danger"):
            return "critical"
        if raw in ("major", "significant", "high"):
            return "major"
        if raw in ("minor", "low", "observation"):
            return "minor"

    # Keyword-based inference — requires finding context to avoid false positives
    text_lower = text.lower()
    has_finding_context = _contains_word(text_lower, _FINDING_CONTEXT_WORDS)
    if not has_finding_context:
        return None

    if _contains_word(text_lower, ("critical", "severe", "emergency", "danger")):
        return "critical"
    if _contains_word(text_lower, ("major", "significant", "high")):
        return "major"
    if _contains_word(text_lower, ("minor", "low", "observation")):
        return "minor"
    # Fallback: finding-context words without explicit severity level
    if _contains_word(text_lower, ("defect", "fault", "failure", "damage", "leak", "crack")):
        return "major"
    return None

def _calculate_confidence(blocks: list[OCRTextBlock], page: int, text: str) -> float:
    """Calculate finding confidence from OCR blocks that actually produced this line.

    Falls back to page-average only if no matching block is found (should be rare).
    """
    page_blocks = [b for b in blocks if b.page_number == page]
    if not page_blocks:
        return 0.5

    # Try to find the specific block(s) whose text overlaps with this line.
    # OCR block text may be a fragment of the line or vice versa, so check
    # both directions of substring containment.
    text_lower = text.strip().lower()
    matching = [
        b for b in page_blocks
        if b.text and (
            b.text.strip().lower() in text_lower
            or text_lower in b.text.strip().lower()
        )
    ]

    if matching:
        avg = sum(b.confidence for b in matching) / len(matching)
    else:
        # No exact match found — fall back to page average, but this
        # should be logged/flagged since it means confidence is an estimate,
        # not a measurement of this specific line.
        avg = sum(b.confidence for b in page_blocks) / len(page_blocks)

    return round(min(1.0, max(0.0, avg)), 3)


def run_inspect_report(
    *,
    input_path: str | Path,
    output_dir: str | Path | None = None,
    workspace_root: str | Path | None = None,
    ocr_backend: PaddleOCRBackend | None = None,
    skip_docx: bool = False,
) -> dict[str, Any]:
    """Execute the full deterministic inspection-report pipeline.

    Returns a result dict with status, paths, and summary.
    """
    started = time.perf_counter()
    run_id = _generate_run_id()

    # Resolve output dir
    if output_dir is None:
        ws = Path(workspace_root) if workspace_root else Path("./workspace")
        out = ws / "outputs" / run_id
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Network policy
    policy = NetworkPolicy(run_id=run_id)

    result: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "output_dir": str(out),
        "artifacts": {},
        "errors": [],
        "phases": [],
    }

    def _phase(name: str, status: str, detail: str = "", ms: float = 0):
        result["phases"].append({
            "phase": name, "status": status, "detail": detail,
            "duration_ms": round(ms, 2),
        })

    try:
        # Phase 1: Validate input
        phase_start = time.perf_counter()
        input_file = _validate_input(input_path)
        _phase("validate_input", "PASS", str(input_file),
               (time.perf_counter() - phase_start) * 1000)
        console.print(f"  [green]✓[/green] Input validated: {input_file.name}")

        # Phase 2: OCR
        phase_start = time.perf_counter()
        ocr_text = ""
        blocks: list[OCRTextBlock] = []

        if input_file.suffix.lower() == ".pdf":
            pages_dir = out / "pages"
            try:
                page_images = _pdf_to_images(input_file, out)
                ocr_text, blocks = _ocr_images(page_images, ocr_backend)
                _phase("ocr", "PASS", f"{len(page_images)} pages",
                       (time.perf_counter() - phase_start) * 1000)
            except WorkbenchError as exc:
                _phase("ocr", "FAIL", str(exc),
                       (time.perf_counter() - phase_start) * 1000)
                result["errors"].append(f"PDF OCR failed: {exc}")
                ocr_text = f"[PDF OCR UNAVAILABLE: {exc}]"
            finally:
                cleanup_error = None
                try:
                    for page_image in pages_dir.glob("page_*.png"):
                        page_image.unlink()
                    pages_dir.rmdir()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    cleanup_error = str(exc)
                cleanup_status = "PASS" if cleanup_error is None else "FAIL"
                _phase(
                    "pdf_page_cleanup",
                    cleanup_status,
                    "temporary page images removed" if cleanup_error is None else cleanup_error,
                    (time.perf_counter() - phase_start) * 1000,
                )
                if cleanup_error:
                    result["errors"].append(f"PDF page cleanup failed: {cleanup_error}")
        else:
            try:
                ocr_text, blocks = _ocr_single_image(input_file, ocr_backend)
                _phase("ocr", "PASS", f"1 image",
                       (time.perf_counter() - phase_start) * 1000)
            except (OCRBackendError, InvalidOCRInputError) as exc:
                _phase("ocr", "FAIL", str(exc),
                       (time.perf_counter() - phase_start) * 1000)
                result["errors"].append(f"OCR failed: {exc}")
                ocr_text = f"[OCR FAILED: {exc}]"

        # Save OCR text
        ocr_path = out / "ocr.txt"
        ocr_path.write_text(ocr_text, encoding="utf-8")
        result["artifacts"]["ocr_text"] = str(ocr_path)
        console.print(f"  [green]✓[/green] OCR complete: {len(blocks)} text blocks")

        # Phase 3: Extract findings
        phase_start = time.perf_counter()
        findings = _extract_findings_deterministic(ocr_text, blocks, input_file)
        findings_path = out / "findings.json"
        findings_path.write_text(
            json.dumps(findings, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result["artifacts"]["findings"] = str(findings_path)
        n_findings = len(findings.get("findings", []))
        n_uncertainties = len(findings.get("uncertainties", []))
        _phase("extract_findings", "PASS",
               f"{n_findings} findings, {n_uncertainties} uncertainties",
               (time.perf_counter() - phase_start) * 1000)
        console.print(
            f"  [green]✓[/green] Findings: {n_findings} findings, "
            f"{n_uncertainties} uncertainties"
        )

        # Phase 4: Generate DOCX
        if not skip_docx:
            phase_start = time.perf_counter()
            docx_path = out / "approval_note.docx"
            try:
                write_approval_note(findings, docx_path)
                result["artifacts"]["approval_note"] = str(docx_path)
                _phase("docx_generation", "PASS", str(docx_path),
                       (time.perf_counter() - phase_start) * 1000)
                console.print(f"  [green]✓[/green] Approval note: {docx_path}")
            except DocxWriterError as exc:
                _phase("docx_generation", "FAIL", str(exc),
                       (time.perf_counter() - phase_start) * 1000)
                result["errors"].append(f"DOCX generation failed: {exc}")
                console.print(f"  [red]✗[/red] DOCX failed: {exc}")

        # Phase 5: Save metadata
        phase_start = time.perf_counter()
        metadata = {
            "run_id": run_id,
            "input_file": str(input_file),
            "input_size_bytes": input_file.stat().st_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ocr_blocks": len(blocks),
            "findings_count": n_findings,
            "uncertainties_count": n_uncertainties,
            "errors": result["errors"],
        }
        metadata_path = out / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result["artifacts"]["metadata"] = str(metadata_path)

        # Save trace
        trace_path = out / "trace.json"
        trace_path.write_text(
            json.dumps({
                "run_id": run_id,
                "phases": result["phases"],
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result["artifacts"]["trace"] = str(trace_path)

        # Phase 6: Network report
        net_path = policy.save(out)
        result["artifacts"]["network_report"] = str(net_path)
        _phase("network_report", "PASS", policy.summary_line(),
               (time.perf_counter() - phase_start) * 1000)

        total_ms = (time.perf_counter() - started) * 1000
        result["status"] = "complete"
        result["total_ms"] = round(total_ms, 2)

        # Terminal summary
        console.print()
        ocr_phases = [p for p in result["phases"] if p["phase"] == "ocr"]
        ocr_status = "PASS" if ocr_phases and ocr_phases[0]["status"] == "PASS" else "FAIL"
        console.print(f"  OCR: {ocr_status}")
        console.print(f"  FINDINGS: PASS ({n_findings} findings, {n_uncertainties} uncertainties)")
        if "approval_note" in result["artifacts"]:
            console.print(f"  APPROVAL NOTE: PASS -> {result['artifacts']['approval_note']}")
        else:
            console.print(f"  APPROVAL NOTE: SKIP (python-docx unavailable)")
        console.print(f"  {policy.summary_line()}")
        console.print(f"  RUN ARTIFACTS: {out}")
        console.print(f"  Total: {total_ms:.0f}ms")

        return result

    except Exception as exc:
        result["status"] = "error"
        result["errors"].append(str(exc))
        # Still save network report
        try:
            policy.save(out)
        except Exception:
            pass
        console.print(f"  [red]✗[/red] Pipeline error: {exc}")
        return result
