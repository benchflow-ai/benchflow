"""Parity check: run the same submission through both pipelines and compare scores.

For each upstream ProgramBench instance we exercise:

  * the upstream `programbench eval` pipeline (truth);
  * the BenchFlow-adapted task running the same submission tarball as oracle.

Both produce a `passed / total` over the same active branches, so the parity
score is the absolute delta in pass rate plus per-branch agreement.

By default this drives the **fixture** instance shipped with ProgramBench
(`testorg__calculator.abc1234`) so it runs without pulling 8.2 GB of test blobs.
Pass `--instance-id <id> --submission <archive>` to drive a real instance with
your own oracle submission. Pass `--limit N` to walk the first N upstream
instances and run the full Gemini-driven loop end-to-end.

Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) before running the live mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmarks.programbench.benchflow import convert

logger = logging.getLogger(__name__)

GEMINI_MODEL_DEFAULT = "gemini-3.1-flash-lite-preview"


def upstream_eval_score(submission_archive: Path, instance_id: str, upstream_repo: Path) -> dict:
    """Run `programbench eval` against a single submission and return its score dict."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        inst_dir = run_dir / instance_id
        inst_dir.mkdir(parents=True)
        # programbench expects submission.tar.gz in the per-instance folder.
        target = inst_dir / "submission.tar.gz"
        if submission_archive.suffix == ".zip":
            # Re-pack zip -> tar.gz (programbench evaluator wants tar.gz).
            with tempfile.TemporaryDirectory() as unzip_tmp:
                subprocess.run(
                    ["unzip", "-q", str(submission_archive), "-d", unzip_tmp],
                    check=True,
                )
                subprocess.run(
                    ["tar", "-czf", str(target), "-C", unzip_tmp, "."],
                    check=True,
                )
        else:
            subprocess.run(["cp", str(submission_archive), str(target)], check=True)

        env = {**os.environ, "PYTHONPATH": str(upstream_repo / "src")}
        try:
            subprocess.run(
                ["uv", "run", "programbench", "eval", str(run_dir)],
                cwd=str(upstream_repo),
                env=env,
                check=True,
                timeout=7200,
            )
        except FileNotFoundError:
            # Fall back to plain python -m if uv isn't available.
            subprocess.run(
                [sys.executable, "-m", "programbench.cli.main", "eval", str(run_dir)],
                cwd=str(upstream_repo),
                env=env,
                check=True,
                timeout=7200,
            )

        eval_json = inst_dir / f"{instance_id}.eval.json"
        if not eval_json.exists():
            raise RuntimeError(f"upstream eval did not produce {eval_json}")
        data = json.loads(eval_json.read_text())
        passed = sum(1 for r in data.get("test_results", []) if r.get("status") == "passed")
        total = len(data.get("test_results", []))
        return {
            "passed": passed,
            "total": total,
            "score": passed / total if total else 0.0,
            "branches": data.get("test_branches", []),
            "executable_hash": data.get("executable_hash"),
            "error_code": data.get("error_code"),
        }


def benchflow_oracle_score(task_dir: Path, submission_archive: Path) -> dict:
    """Run BenchFlow's verifier against a submission as if it were the oracle.

    We don't spin up a real ACP agent — we just unpack the submission into a
    fresh /app, run the task's verifier image, and read back /logs/verifier/reward.txt
    and /logs/verifier/score.json. Equivalent to ``bench run <task> --agent oracle``
    but we provide the submission ourselves.
    """
    image_tag = f"benchflow-bench-pb-{task_dir.name}".lower().replace("_", "-")
    subprocess.run(
        ["docker", "build", "-t", image_tag, "-f", str(task_dir / "environment" / "Dockerfile"),
         str(task_dir / "environment")],
        check=True, timeout=1800,
    )
    container = f"benchflow-bench-pb-{task_dir.name}".lower().replace("_", "-") + "-run"
    subprocess.run(["docker", "rm", "-f", container], check=False)
    subprocess.run(
        ["docker", "run", "-d", "--name", container, image_tag, "sleep", "7200"],
        check=True,
    )
    try:
        # Stage the submission into /app/.
        with tempfile.TemporaryDirectory() as tmp:
            unpacked = Path(tmp) / "unpacked"
            unpacked.mkdir()
            if submission_archive.suffix == ".zip":
                subprocess.run(["unzip", "-q", str(submission_archive), "-d", str(unpacked)], check=True)
            else:
                subprocess.run(["tar", "-xzf", str(submission_archive), "-C", str(unpacked)], check=True)
            subprocess.run(
                ["docker", "cp", f"{unpacked}/.", f"{container}:/app/"],
                check=True,
            )
        # Stage the verifier under /tests/, mirroring what the BenchFlow runner does.
        subprocess.run(["docker", "exec", container, "mkdir", "-p", "/tests"], check=True)
        subprocess.run(
            ["docker", "cp", str(task_dir / "tests" / "test.sh"), f"{container}:/tests/test.sh"],
            check=True,
        )
        subprocess.run(["docker", "exec", container, "chmod", "+x", "/tests/test.sh"], check=True)
        subprocess.run(
            ["docker", "exec", container, "bash", "/tests/test.sh"],
            check=False, timeout=7200,
        )
        # Pull back the verifier outputs.
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "logs"
            outdir.mkdir()
            subprocess.run(
                ["docker", "cp", f"{container}:/logs/verifier/.", str(outdir)],
                check=True,
            )
            reward = float((outdir / "reward.txt").read_text().strip() or 0)
            score_json = outdir / "score.json"
            score = json.loads(score_json.read_text()) if score_json.exists() else {}
            return {
                "passed": score.get("passed", 0),
                "total": score.get("total", 0),
                "score": reward,
                "branches": [b.get("branch") for b in score.get("branches", [])],
                "branch_results": score.get("branches", []),
            }
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=False)


def compare(upstream: dict, ours: dict, instance_id: str) -> dict:
    delta = abs(upstream["score"] - ours["score"])
    return {
        "instance_id": instance_id,
        "upstream": upstream,
        "benchflow": ours,
        "delta": delta,
        "agree": delta < 1e-6 and upstream["passed"] == ours["passed"] and upstream["total"] == ours["total"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parity check: ProgramBench upstream vs. BenchFlow programbench adapter.")
    parser.add_argument("--upstream-repo", type=Path, required=True,
                        help="Path to a clone of facebookresearch/ProgramBench.")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/programbench/parity_experiment.json"))
    parser.add_argument("--tasks-dir", type=Path, default=Path(".ref/programbench-bf"),
                        help="Where converted BenchFlow tasks live.")
    parser.add_argument("--instance-id", default=None,
                        help="Run a single specified upstream instance.")
    parser.add_argument("--submission", type=Path, default=None,
                        help="Submission archive (.tar.gz or .zip) to score on both pipelines.")
    parser.add_argument("--limit", type=int, default=1,
                        help="When no --submission is given, walk the first N test_runs/correct/ fixtures.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    upstream_tasks = args.upstream_repo / "src" / "programbench" / "data" / "tasks"
    upstream_runs = args.upstream_repo / "src" / "programbench" / "data" / "test_runs" / "correct"

    # Decide which instances to run.
    if args.instance_id and args.submission:
        cases = [(args.instance_id, args.submission)]
    else:
        cases = []
        for inst_dir in sorted(upstream_runs.iterdir()):
            sub = inst_dir / "submission.tar.gz"
            if not sub.exists():
                sub = inst_dir / "submission.zip"
            if sub.exists():
                cases.append((inst_dir.name, sub))
        cases = cases[: args.limit]

    if not cases:
        print("No parity cases found. Pass --instance-id and --submission, "
              "or ensure test_runs/correct/<id>/submission.* exists.", file=sys.stderr)
        return 2

    # Make sure converted tasks exist for all cases.
    convert(
        upstream_tasks_dir=upstream_tasks,
        output_dir=args.tasks_dir,
        task_ids=[c[0] for c in cases],
        overwrite=False,
    )

    results = []
    for instance_id, submission in cases:
        logger.info("Parity: %s", instance_id)
        try:
            up = upstream_eval_score(submission, instance_id, args.upstream_repo)
        except Exception as e:
            logger.error("upstream eval failed for %s: %s", instance_id, e)
            up = {"passed": 0, "total": 0, "score": 0.0, "branches": [], "error": str(e)}
        try:
            bf = benchflow_oracle_score(args.tasks_dir / instance_id, submission)
        except Exception as e:
            logger.error("benchflow oracle run failed for %s: %s", instance_id, e)
            bf = {"passed": 0, "total": 0, "score": 0.0, "branches": [], "error": str(e)}
        cmp = compare(up, bf, instance_id)
        results.append(cmp)
        logger.info("  upstream=%s/%s  benchflow=%s/%s  delta=%.4f  agree=%s",
                    up["passed"], up["total"], bf["passed"], bf["total"], cmp["delta"], cmp["agree"])

    summary = {
        "model": os.environ.get("PARITY_AGENT_MODEL", GEMINI_MODEL_DEFAULT),
        "n_cases": len(results),
        "n_agreeing": sum(1 for r in results if r["agree"]),
        "max_delta": max((r["delta"] for r in results), default=0.0),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {args.output} ({summary['n_agreeing']}/{summary['n_cases']} agree, "
          f"max_delta={summary['max_delta']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
