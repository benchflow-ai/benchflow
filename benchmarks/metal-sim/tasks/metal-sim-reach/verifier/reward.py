"""Turn a ``metal-sim-eval`` summary into BenchFlow's reward files.

Usage: reward.py <summary.json> <verifier log dir> [<eval stdout log>]
Writes reward.txt (scalar 0..1 = success rate over held-out scenes), reward.json and
reward-details.json. A missing or unreadable summary is a scored 0.0 with the reason recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(summary_path: str, out_dir: str, eval_log: str | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reward, details = 0.0, {}
    try:
        summary = json.loads(Path(summary_path).read_text())
        reward = float(summary.get("success_rate", 0.0))
        details = summary
    except Exception as exc:  # noqa: BLE001 - any failure is a 0.0 with a reason
        tail = ""
        if eval_log and Path(eval_log).exists():
            tail = Path(eval_log).read_text()[-4000:]
        details = {"error": f"could not read eval summary: {exc}", "eval_log_tail": tail}
    reward = max(0.0, min(1.0, reward))
    (out / "reward.txt").write_text(f"{reward:.4f}\n")
    (out / "reward.json").write_text(json.dumps({"reward": reward, "task_success": reward}, indent=2) + "\n")
    (out / "reward-details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(f"reward {reward:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
