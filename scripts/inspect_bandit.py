#!/usr/bin/env python3
"""Print the persisted contextual-bandit policy diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from routing.adaptive_router import ContextualBanditRouter, SQLiteBanditStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="workspace/.aegis/runs.db")
    args = parser.parse_args()
    store = SQLiteBanditStore(Path(args.db))
    try:
        print(json.dumps(ContextualBanditRouter(store=store).diagnostics(), indent=2))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
