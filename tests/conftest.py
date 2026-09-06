"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
