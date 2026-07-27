"""Run the medical-assistant benchmark via BenchFlow.

    set -a; . ~/sb-run.env; set +a       # provides DEEPSEEK_API_KEY for the proxy
    python benchmarks/medical-assistant/run_medical_assistant.py

    # or, equivalently, via the CLI:
    bench eval run --config benchmarks/medical-assistant/medical-assistant-deepseek.yaml

Every rollout runs in its own Docker sandbox (environment: docker). The
medical-assistant agent (a LangGraph supervisor->specialists graph) is installed
into the sandbox and its deepseek-v4-pro calls are routed through BenchFlow's
LiteLLM proxy, so usage/cost + the raw-LLM trajectory are captured per rollout.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical-assistant benchmark.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_SCRIPT_DIR / "medical-assistant-deepseek.yaml"),
        help="BenchFlow evaluation YAML config.",
    )
    return parser.parse_args()


async def main() -> None:
    from benchflow.evaluation import Evaluation

    args = _parse_args()
    job = Evaluation.from_yaml(args.config)
    job._tasks_dir = _SCRIPT_DIR / "tasks"  # absolute path → cwd-independent
    # Dedicated jobs_dir so we never resume into another agent's results.
    job._jobs_dir = _REPO_ROOT / "out" / "medical-bench" / "jobs"
    job._job_name = "medical-assistant-deepseek"
    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
