"""Small deterministic fixture used by repository inspection examples."""


def health_check() -> dict[str, str]:
    """Return a stable local health status."""
    return {"status": "ok", "scope": "local"}
