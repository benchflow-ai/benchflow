"""Parity entry points for the ASI-Bench to BenchFlow conversion.

No benchmark-wide parity claim is valid yet. Each entry point therefore
fails explicitly instead of treating an empty task set as a successful run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NoReturn

_HERE = Path(__file__).resolve().parent
PARITY_EXPERIMENT = _HERE / "parity_experiment.json"


class InsufficientParityEvidenceError(RuntimeError):
    """Raised until real source and converted evaluations have been compared."""


def _not_implemented(layer: str, tasks_dir: Path) -> NoReturn:
    raise InsufficientParityEvidenceError(
        f"ASI-Bench {layer} parity is not implemented; no evidence was recorded "
        f"from {tasks_dir}"
    )


def structural_parity(tasks_dir: Path) -> dict:
    """Validate generated task structure once conversion is implemented."""
    _not_implemented("structural", tasks_dir)


def eval_parity(tasks_dir: Path) -> dict:
    """Compare known outputs once the in-sandbox verifier is implemented."""
    _not_implemented("evaluation", tasks_dir)


def side_by_side_parity(tasks_dir: Path) -> dict:
    """Compare ASI-Bench and BenchFlow scores on identical artifacts."""
    _not_implemented("side-by-side", tasks_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASI-Bench parity test")
    parser.add_argument(
        "--mode",
        choices=["full", "eval-parity", "side-by-side"],
        default="full",
    )
    parser.add_argument("--tasks-dir", type=Path, default=_HERE / "tasks")
    args = parser.parse_args()

    try:
        if args.mode == "eval-parity":
            result = eval_parity(args.tasks_dir)
        elif args.mode == "side-by-side":
            result = side_by_side_parity(args.tasks_dir)
        else:
            result = structural_parity(args.tasks_dir)
    except InsufficientParityEvidenceError as exc:
        print(
            json.dumps(
                {
                    "benchmark": "asi-bench",
                    "status": "insufficient-evidence",
                    "mode": args.mode,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        raise SystemExit(2) from exc

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
