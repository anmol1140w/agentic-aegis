#!/usr/bin/env python3
"""Run deterministic versus contextual-bandit routing evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.routing import evaluate_baseline_vs_adaptive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--actual", action="store_true", help="execute local specialist candidates")
    parser.add_argument("--max-actual-tasks", type=int, default=None)
    args = parser.parse_args()
    report = evaluate_baseline_vs_adaptive(repeats=args.repeats, actual=args.actual,
                                           max_actual_tasks=args.max_actual_tasks)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"baseline_rows", "adaptive_rows"}}, indent=2))
    return 0 if report["promotion"]["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
