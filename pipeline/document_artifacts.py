"""Local PDF, Markdown, and TXT artifact creation and verification."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _normalized_format(value: str) -> str:
    fmt = value.lower().lstrip(".")
    if fmt == "md":
        return "markdown"
    return fmt


def create_document_artifact(content: str, output_format: str, output_path: str | Path) -> Path:
    fmt = _normalized_format(output_format)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content_generation_failed")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "markdown":
        path.write_text(content, encoding="utf-8")
    elif fmt == "txt":
        plain = re.sub(r"^#{1,6}\s+", "", content, flags=re.M)
        path.write_text(plain, encoding="utf-8")
    elif fmt == "pdf":
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
        from xml.sax.saxutils import escape
        styles = getSampleStyleSheet()
        story: list[Any] = []
        bullets: list[ListItem] = []
        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue
            bullet = re.match(r"^[-*]\s+(.+)$", line)
            if bullet:
                bullets.append(ListItem(Paragraph(escape(bullet.group(1)), styles["BodyText"])))
                continue
            if bullets:
                story.append(ListFlowable(bullets, bulletType="bullet")); bullets = []
            heading = re.match(r"^(#{1,3})\s+(.+)$", line)
            if heading:
                level = min(len(heading.group(1)), 3)
                style = styles["Title"] if level == 1 else styles["Heading2"] if level == 2 else styles["Heading3"]
                story.extend([Paragraph(escape(heading.group(2)), style), Spacer(1, 8)])
            else:
                story.extend([Paragraph(escape(line), styles["BodyText"]), Spacer(1, 6)])
        if bullets:
            story.append(ListFlowable(bullets, bulletType="bullet"))
        if not story:
            raise ValueError("invalid_document_content")
        SimpleDocTemplate(str(path), pagesize=letter).build(story)
    else:
        raise ValueError(f"unsupported_document_format: {output_format}")
    return path


def verify_artifact(path: str | Path, artifact_type: str, expected_content: str | None = None,
                    expected_topic: str | None = None) -> dict[str, Any]:
    target = Path(path)
    result: dict[str, Any] = {"exists": target.exists(), "readable": False, "format_valid": False,
                              "content_present": False, "text_length": 0}
    if not target.is_file() or target.stat().st_size == 0:
        return result
    try:
        if artifact_type == "pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(target))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                page_count = len(reader.pages)
            except ImportError:
                import pymupdf
                document = pymupdf.open(str(target))
                text = "\n".join(page.get_text() for page in document)
                page_count = document.page_count
                document.close()
            result.update({"readable": True, "format_valid": True, "page_count": page_count, "text_length": len(text), "text": text})
        else:
            text = target.read_text(encoding="utf-8")
            result.update({"readable": True, "format_valid": True, "text_length": len(text), "text": text})
        result["content_present"] = bool(text.strip())
        expected = expected_topic or expected_content
        result["expected_content_present"] = not expected or expected.lower() in text.lower()
        result["verified"] = all((result["exists"], result["readable"], result["format_valid"], result["content_present"], result["expected_content_present"]))
    except Exception as exc:
        result["error"] = str(exc)[:300]
        result["verified"] = False
    result.pop("text", None)
    return result
