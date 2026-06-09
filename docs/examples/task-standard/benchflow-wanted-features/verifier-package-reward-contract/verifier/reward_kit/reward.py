"""Minimal Reward Kit runner for the verifier-package dogfood task."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    reward_json = Path(os.environ["BENCHFLOW_REWARD_JSON"])
    reward_details = Path(os.environ["BENCHFLOW_REWARD_DETAILS_JSON"])
    criteria = os.environ.get("BENCHFLOW_REWARD_KIT_CRITERIA")
    manifest = json.loads(Path(os.environ["BENCHFLOW_REWARD_KIT_MANIFEST"]).read_text())

    metrics = {
        "verifier_document": 1.0,
        "reward_contract": 1.0,
        "judge_isolation": 1.0,
        "compatibility": 0.5,
    }
    payload = {
        "metrics": metrics,
        "metadata": {
            "source": "reward-kit",
            "criteria": criteria,
            "criteria_policy": manifest["criteria_policy"],
        },
    }
    details = {
        "source": "reward-kit",
        "criteria": [
            {"id": name, "score": score} for name, score in metrics.items()
        ],
        "aggregate": {
            "method": manifest["criteria_policy"]["method"],
            "weights": manifest["criteria_policy"]["weights"],
        },
    }

    reward_json.parent.mkdir(parents=True, exist_ok=True)
    reward_json.write_text(json.dumps(payload, indent=2, allow_nan=False))
    reward_details.write_text(json.dumps(details, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
