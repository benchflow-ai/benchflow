from __future__ import annotations

import json
import os
from pathlib import Path


def _contains(path: Path, needle: str) -> bool:
    return path.exists() and needle in path.read_text(errors="replace")


def main() -> None:
    repo = Path("/repo")
    manifest_path = Path(os.environ["BENCHFLOW_REWARD_KIT_MANIFEST"])
    manifest = json.loads(manifest_path.read_text())
    metrics = {
        "task_paths_entrypoint": float(
            _contains(
                repo / "src/benchflow/task/paths.py",
                "def has_verifier_entrypoint",
            )
            and _contains(
                repo / "src/benchflow/task/paths.py",
                "verifier.md",
            )
        ),
        "regression_tests": float(
            _contains(
                repo / "tests/test_verifier_document.py",
                "test_task_paths_accepts_selected_reward_kit_without_test_sh",
            )
            and _contains(
                repo / "tests/test_verifier_document.py",
                "test_task_paths_rejects_selected_reward_kit_without_runner",
            )
        ),
        "no_test_sh_dogfood": float(
            not (
                repo
                / "docs/examples/task-standard/benchflow-wanted-features/"
                "verifier-native-entrypoint/verifier/test.sh"
            ).exists()
        ),
    }
    payload = {
        "metrics": metrics,
        "metadata": {
            "source": "reward-kit",
            "strategy": manifest["strategy"]["name"],
            "criteria_policy": manifest["criteria_policy"],
        },
    }
    details = {
        "criteria": [
            {"id": key, "score": value} for key, value in metrics.items()
        ],
        "aggregate": {
            "method": manifest["criteria_policy"]["method"],
            "weights": manifest["criteria_policy"]["weights"],
        },
    }
    Path(os.environ["BENCHFLOW_REWARD_JSON"]).write_text(
        json.dumps(payload, indent=2, allow_nan=False)
    )
    Path(os.environ["BENCHFLOW_REWARD_DETAILS_JSON"]).write_text(
        json.dumps(details, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
