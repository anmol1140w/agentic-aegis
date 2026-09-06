"""Fast, deterministic preprocessing for user and OCR text.

This module deliberately performs normalization and extraction only.  It does
not choose agents, execute tools, or perform retrieval.
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Correction:
    original: str
    normalized: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class NLPResult:
    original_text: str
    normalized_text: str
    enhanced_prompt: str
    intent: str
    domain: str
    modality: str
    entities: list[str]
    keywords: list[str]
    technical_terms: list[str]
    requested_actions: list[str]
    constraints: list[str]
    severity_terms: list[str]
    corrections: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NLPPreprocessor:
    """Apply conservative, auditable text cleanup in a single fast pass."""

    _OCR_FIXES = {
        "majr": ("major", 0.97),
        "minr": ("minor", 0.96),
        "inspction": ("inspection", 0.96),
        "inspec tion": ("inspection", 0.98),
        "instrumnts": ("instruments", 0.95),
        "anlyze": ("analyze", 0.96),
        "housc": ("house", 0.93),
    }

    def process(self, text: str, *, source: str = "user") -> NLPResult:
        started = time.perf_counter()
        original = text if isinstance(text, str) else str(text or "")
        corrections: list[Correction] = []
        value = unicodedata.normalize("NFKC", original)
        value = "".join(ch for ch in value if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n\s*", " ", value).strip()
        value = re.sub(r"([!?.,])\1{2,}", r"\1", value)

        def replace(pattern: str, replacement: str, confidence: float, reason: str) -> None:
            nonlocal value
            new_value, count = re.subn(pattern, replacement, value, flags=re.I)
            if count:
                corrections.append(Correction(pattern, replacement, confidence, reason))
                value = new_value

        for token, (replacement, confidence) in self._OCR_FIXES.items():
            replace(rf"(?<!\w){re.escape(token)}(?!\w)", replacement, confidence, "known OCR correction")
        replace(r"\bP\s*(?:&|and)\s*I\s*D\b", "P&ID", 0.99, "controlled technical terminology")
        replace(r"\bq\.?\s*c\.?\b", "quality control", 0.99, "controlled abbreviation")

        # Preserve ambiguous tags and make the uncertainty auditable.
        for match in re.finditer(r"\b[A-Za-z]{1,3}-\d{2}[Iil]\b", value):
            token = match.group(0)
            corrections.append(Correction(token, token, 0.42, "ambiguous technical identifier preserved"))

        tags = re.findall(r"\b(?:P|V|T|FT|PT|XV)-\d{3}\b", value, flags=re.I)
        ids = re.findall(r"\b(?:Report\s+ID\s*:\s*|ID\s*[:=]\s*)([A-Za-z0-9_.-]+)", value, flags=re.I)
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", value)
        paths = re.findall(r"(?:[A-Za-z]:)?(?:/[\w.\-]+)+|\b[\w.\-]+\.(?:pdf|docx?|py|json|png|jpg)\b", value, flags=re.I)
        entities = list(dict.fromkeys(tags + ids + dates + paths))
        severity_terms = [w for w in ("critical", "major", "minor", "high", "low") if re.search(rf"\b{w}\b", value, re.I)]
        actions = [w for w in ("analyze", "inspect", "extract", "find", "identify", "fix", "edit", "run", "test", "summarize") if re.search(rf"\b{w}\b", value, re.I)]
        constraints = re.findall(r"\b(?:do not|don't|read-only|without approval|only)\b[^.\n]*", value, flags=re.I)
        technical_terms = [w for w in ("P&ID", "inspection", "instruments", "quality control", "OCR") if re.search(re.escape(w), value, re.I)]
        lower = value.lower()
        if re.search(r"\b(pdf|docx?|inspection report|ocr|document)\b", lower):
            intent, domain, modality = "document_analysis", "industrial_inspection" if "inspection" in lower else "document", "document"
        elif re.search(r"\b(image|p&?id|diagram|visual|photo)\b", lower):
            intent, domain, modality = "vision_analysis", "industrial_engineering" if "p&id" in lower else "visual", "image"
        elif re.search(r"\b(code|python|function|script|pytest|bug|file|directory|git)\b", lower) and actions:
            intent, domain, modality = "coding", "software", "text"
        elif len(value.split()) <= 6:
            intent, domain, modality = "general", "general", "text"
        else:
            intent, domain, modality = "general", "general", "text"

        enhanced = value
        if re.fullmatch(r"(?:what is|show me) (?:this )?image\??", value, re.I):
            enhanced = "Analyze the provided image and identify its document or engineering context, key visible components, and notable findings."
        elapsed = (time.perf_counter() - started) * 1000
        metadata = {
            "source": source,
            "normalization_count": len(corrections),
            "entity_count": len(entities),
            "enhancement_applied": enhanced != value,
            "processing_duration_ms": round(elapsed, 3),
            "raw_preserved": True,
        }
        return NLPResult(original, value, enhanced, intent, domain, modality, entities,
                         list(dict.fromkeys(technical_terms + severity_terms)), technical_terms,
                         actions, constraints, severity_terms,
                         [asdict(c) for c in corrections], metadata)
