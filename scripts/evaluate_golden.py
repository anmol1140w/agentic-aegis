#!/usr/bin/env python3
"""Run the deterministic pre-RL golden-task gate."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.golden import evaluate_golden_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = evaluate_golden_tasks(repeats=args.repeats)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    for item in report["results"]:
        if not item["passed"]:
            print(f"FAIL {item['name']}: {'; '.join(item['failures'])}")
    raise SystemExit(0 if report["accuracy"] == 1.0 else 1)


if __name__ == "__main__":
    main()
