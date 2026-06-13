#!/usr/bin/env python3
"""Original runner for the local Browser Use smoke task.

This mirrors the shape that external browser-use eval runners emit: final
result, steps, screenshots, duration, and pass/fail score.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def run_original(task_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    descriptor = json.loads((task_dir / "browser-use-task.json").read_text())
    expected = descriptor["expected_result"]
    html = (task_dir / "environment" / "browser_fixture" / "index.html").read_text()
    passed = expected in html
    final_result = expected if passed else ""
    return {
        "framework": "browser-use-smoke-original",
        "task_id": descriptor["task_id"],
        "final_result": final_result,
        "score": 1.0 if passed else 0.0,
        "steps": [
            {"action": "open", "url": descriptor["url"]},
            {"action": "extract_text", "value": final_result},
        ],
        "screenshots_b64": [],
        "num_steps": 2,
        "duration_sec": round(time.perf_counter() - started, 6),
        "error": None if passed else "expected status not found",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = run_original(args.task_dir)
    payload = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
