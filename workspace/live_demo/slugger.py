"""Small workspace demo module for URL-safe slug normalization."""

import re


def slugify(value: str) -> str:
    """Return a URL-safe lowercase slug."""
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return normalized.strip("-")
