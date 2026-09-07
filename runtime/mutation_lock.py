"""Per-workspace serialization for mutating operations."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading

_locks: dict[str, threading.RLock] = {}
_guard = threading.Lock()


@contextmanager
def workspace_mutation_lock(root: str | Path):
    key = str(Path(root).resolve())
    with _guard:
        lock = _locks.setdefault(key, threading.RLock())
    with lock:
        yield
