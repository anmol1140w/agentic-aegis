"""Small safeguards for regex operations over user/model/OCR text."""

from __future__ import annotations

import re

MAX_TEXT_CHARS = 256 * 1024
MAX_PATTERN_CHARS = 2 * 1024


def bounded_text(value: object, *, limit: int = MAX_TEXT_CHARS) -> str:
    """Normalize regex input and cap its size before scanning it."""
    text = value if isinstance(value, str) else str(value or "")
    return text[:max(1, limit)]


def compile_safe(pattern: str, flags: int = 0) -> re.Pattern[str]:
    """Compile a bounded pattern and turn malformed patterns into a clear error."""
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("regex pattern must be a non-empty string")
    if len(pattern) > MAX_PATTERN_CHARS:
        raise ValueError("regex pattern exceeds the safety limit")
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"invalid regex pattern: {exc}") from exc
