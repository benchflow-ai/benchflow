"""Parity helpers for the MLE-bench -> BenchFlow conversion."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
PARITY_EXPERIMENT = _HERE / "parity_experiment.json"

_REQUIRED_FILES = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "environment/instructions.txt",
    "environment/validate_submission.sh",
    "tests/test.sh",
    "tests/verify.py",
    "tests/mlebench_task.json",
)


def _task_dirs(tasks_dir: Path) -> list[Path]:
    if not tasks_dir.is_dir():
        return []
    return sorted(path for path in tasks_dir.iterdir() if path.is_dir())


def _summary(name: str, tasks_tested: int, issues: list[dict]) -> dict:
    passed = tasks_tested - len({issue["task_id"] for issue in issues})
    return {
        name: {
            "tasks_tested": tasks_tested,
            "passed": passed,
            "failed": tasks_tested - passed,
            "issues": issues,
        }
    }


def structural_parity(tasks_dir: Path) -> dict:
    """Check generated tasks preserve the expected BenchFlow file contract."""
    issues: list[dict] = []
    tasks = _task_dirs(tasks_dir)

    for task_dir in tasks:
        for rel in _REQUIRED_FILES:
            if not (task_dir / rel).is_file():
                issues.append(
                    {
                        "task_id": task_dir.name,
                        "path": rel,
                        "issue": "missing required file",
                    }
                )

        metadata_path = task_dir / "tests" / "mlebench_task.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("slug") != task_dir.name:
                issues.append(
                    {
                        "task_id": task_dir.name,
                        "path": "tests/mlebench_task.json",
                        "issue": "metadata slug does not match task directory",
                    }
                )

        if (task_dir / "environment" / "private-data").exists():
            issues.append(
                {
                    "task_id": task_dir.name,
                    "path": "environment/private-data",
                    "issue": "private grading data must not be agent-visible",
                }
            )

    return {
        "benchmark": "mle-bench",
        **_summary("structural_parity", len(tasks), issues),
    }


def eval_parity(tasks_dir: Path) -> dict:
    """Check the verifier-side files needed for upstream grading are present."""
    issues: list[dict] = []
    tasks = _task_dirs(tasks_dir)

    for task_dir in tasks:
        private_data = task_dir / "tests" / "private-data"
        mlebench_pkg = task_dir / "tests" / "mlebench"
        verify_py = task_dir / "tests" / "verify.py"
        if not private_data.exists():
            issues.append(
                {
                    "task_id": task_dir.name,
                    "path": "tests/private-data",
                    "issue": "missing verifier-only prepared private data",
                }
            )
        if not (mlebench_pkg / "grade.py").is_file():
            issues.append(
                {
                    "task_id": task_dir.name,
                    "path": "tests/mlebench/grade.py",
                    "issue": "missing upstream grading module",
                }
            )
        if verify_py.is_file():
            content = verify_py.read_text(encoding="utf-8")
            if "grade_csv" not in content or "any_medal" not in content:
                issues.append(
                    {
                        "task_id": task_dir.name,
                        "path": "tests/verify.py",
                        "issue": "verifier does not call upstream grade_csv any-medal path",
                    }
                )

    return {
        "benchmark": "mle-bench",
        **_summary("eval_parity", len(tasks), issues),
    }


def _load_metadata(task_dir: Path) -> dict[str, Any]:
    path = task_dir / "tests" / "mlebench_task.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_submission_paths(
    task_dir: Path,
    metadata: dict[str, Any],
    submissions_dir: Path | None,
) -> list[Path]:
    competition_id = str(metadata.get("competition_id") or task_dir.name)
    slug = str(metadata.get("slug") or task_dir.name)
    candidates: list[Path] = []
    if submissions_dir is not None:
        candidates.extend(
            [
                submissions_dir / f"{competition_id}.csv",
                submissions_dir / f"{slug}.csv",
                submissions_dir / competition_id / "submission.csv",
                submissions_dir / slug / "submission.csv",
            ]
        )
    dataset = metadata.get("dataset")
    if isinstance(dataset, dict):
        sample_submission = dataset.get("sample_submission")
        if sample_submission:
            candidates.append(
                task_dir / "environment" / "data" / Path(str(sample_submission)).name
            )
    candidates.extend(
        [
            task_dir / "environment" / "data" / "sample_submission.csv",
            task_dir / "environment" / "data" / "sampleSubmission.csv",
        ]
    )
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def _find_submission(
    task_dir: Path,
    metadata: dict[str, Any],
    submissions_dir: Path | None,
) -> Path | None:
    for candidate in _candidate_submission_paths(task_dir, metadata, submissions_dir):
        if candidate.is_file():
            return candidate
    return None


def _run_json_subprocess(code: str, args: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": (proc.stderr or proc.stdout).strip()[-4000:],
        }
    try:
        return {"ok": True, "report": json.loads(proc.stdout)}
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": f"could not parse JSON from subprocess: {exc}; stdout={proc.stdout[-1000:]}",
        }


def _run_original_grade(
    task_dir: Path,
    *,
    competition_id: str,
    submission: Path,
) -> dict[str, Any]:
    code = r"""
import json
import sys
from pathlib import Path

tests_dir = Path(sys.argv[1])
competition_id = sys.argv[2]
submission = Path(sys.argv[3])
data_dir = tests_dir / "private-data"
sys.path.insert(0, str(tests_dir))

from mlebench.grade import grade_csv
from mlebench.registry import registry

competition = registry.set_data_dir(data_dir).get_competition(competition_id)
print(json.dumps(grade_csv(submission, competition).to_dict(), sort_keys=True))
"""
    return _run_json_subprocess(
        code,
        [str(task_dir / "tests"), competition_id, str(submission)],
        cwd=task_dir,
    )


def _run_converted_verifier(
    task_dir: Path,
    *,
    competition_id: str,
    submission: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mle-bench-parity-") as tmp:
        tmp_path = Path(tmp)
        reward_file = tmp_path / "reward.txt"
        reward_json = tmp_path / "reward.json"
        report_file = tmp_path / "grading_report.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(task_dir / "tests" / "verify.py"),
                "--competition-id",
                competition_id,
                "--submission",
                str(submission),
                "--data-dir",
                str(task_dir / "tests" / "private-data"),
                "--reward-file",
                str(reward_file),
                "--reward-json",
                str(reward_json),
                "--report-file",
                str(report_file),
            ],
            cwd=str(task_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "error": (proc.stderr or proc.stdout).strip()[-4000:],
            }
        if not report_file.is_file() or not reward_file.is_file():
            return {
                "ok": False,
                "error": "converted verifier produced no report/reward",
            }
        try:
            return {
                "ok": True,
                "report": json.loads(report_file.read_text(encoding="utf-8")),
                "reward": float(reward_file.read_text(encoding="utf-8")),
            }
        except (ValueError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": f"converted output parse error: {exc}"}


def _verdict(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.10g}"
    return str(value)


def _criteria_for_reports(
    original: dict[str, Any],
    converted: dict[str, Any],
) -> list[dict[str, Any]]:
    criteria: list[dict[str, Any]] = []
    for field in (
        "submission_exists",
        "valid_submission",
        "any_medal",
        "gold_medal",
        "silver_medal",
        "bronze_medal",
        "above_median",
        "score",
    ):
        original_value = original.get(field)
        converted_value = converted.get(field)
        if isinstance(original_value, (float, int)) and isinstance(
            converted_value, (float, int)
        ):
            agreement = math.isclose(
                float(original_value), float(converted_value), abs_tol=1e-9
            )
        else:
            agreement = original_value == converted_value
        criteria.append(
            {
                "criterion_id": field,
                "original_verdict": _verdict(original_value),
                "adapted_verdict": _verdict(converted_value),
                "agreement": agreement,
            }
        )
    return criteria


def _reward_from_report(report: dict[str, Any]) -> float:
    return 1.0 if report.get("any_medal") else 0.0


def side_by_side_parity(
    tasks_dir: Path,
    *,
    submissions_dir: Path | None = None,
    limit: int | None = None,
) -> dict:
    """Run direct grade_csv and the converted verifier on identical submissions.

    By default this uses each task's copied sample submission. For a stronger
    parity run, pass ``--submissions-dir`` with paired agent/oracle
    submissions named ``<competition-id>.csv`` or ``<slug>.csv``.
    """
    tasks = _task_dirs(tasks_dir)
    if limit is not None:
        tasks = tasks[:limit]

    parity_tasks: list[dict[str, Any]] = []
    reward_samples: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for task_dir in tasks:
        metadata = _load_metadata(task_dir)
        competition_id = str(metadata.get("competition_id") or task_dir.name)
        submission = _find_submission(task_dir, metadata, submissions_dir)
        private_data = task_dir / "tests" / "private-data"
        if submission is None:
            skipped.append(
                {
                    "task_id": task_dir.name,
                    "reason": "no submission found",
                }
            )
            continue
        if not private_data.exists():
            skipped.append(
                {
                    "task_id": task_dir.name,
                    "reason": "missing tests/private-data",
                }
            )
            continue

        original = _run_original_grade(
            task_dir,
            competition_id=competition_id,
            submission=submission,
        )
        converted = _run_converted_verifier(
            task_dir,
            competition_id=competition_id,
            submission=submission,
        )
        if not original.get("ok") or not converted.get("ok"):
            parity_tasks.append(
                {
                    "task_id": task_dir.name,
                    "n_criteria": 1,
                    "criteria_results": [
                        {
                            "criterion_id": "grader-execution",
                            "original_verdict": original.get("error", "ok"),
                            "adapted_verdict": converted.get("error", "ok"),
                            "agreement": False,
                        }
                    ],
                }
            )
            continue

        original_report = original["report"]
        converted_report = converted["report"]
        criteria = _criteria_for_reports(original_report, converted_report)
        parity_tasks.append(
            {
                "task_id": task_dir.name,
                "submission": str(submission),
                "n_criteria": len(criteria),
                "criteria_results": criteria,
            }
        )

        legacy_reward = _reward_from_report(original_report)
        converted_reward = float(converted["reward"])
        reward_samples.append(
            {
                "task_id": task_dir.name,
                "legacy_reward": legacy_reward,
                "converted_reward": converted_reward,
                "reward_delta": abs(converted_reward - legacy_reward),
            }
        )

    status = "recorded" if parity_tasks or reward_samples else "insufficient-evidence"
    return {
        "experiment": "side-by-side-parity",
        "benchmark": "mle-bench",
        "status": status,
        "tasks_dir": str(tasks_dir),
        "conversion_parity": {
            "description": "Direct upstream grade_csv vs converted BenchFlow verifier on identical CSV submissions.",
            "tasks": parity_tasks,
        },
        "reward_distribution_parity": {
            "description": "Reward is 1.0 for any_medal and 0.0 otherwise.",
            "samples": reward_samples,
        },
        "skipped": skipped,
    }


def full_parity(
    tasks_dir: Path,
    *,
    submissions_dir: Path | None = None,
    limit: int | None = None,
) -> dict:
    structural = structural_parity(tasks_dir)["structural_parity"]
    eval_result = eval_parity(tasks_dir)["eval_parity"]
    side_by_side = side_by_side_parity(
        tasks_dir,
        submissions_dir=submissions_dir,
        limit=limit,
    )
    return {
        "benchmark": "mle-bench",
        "structural_parity": structural,
        "eval_parity": eval_result,
        "side_by_side_status": side_by_side.get("status", "recorded"),
        "conversion_parity": side_by_side.get("conversion_parity", {"tasks": []}),
        "reward_distribution_parity": side_by_side.get(
            "reward_distribution_parity",
            {"samples": []},
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MLE-bench parity checks")
    parser.add_argument(
        "--mode",
        choices=["full", "eval-parity", "side-by-side"],
        default="full",
    )
    parser.add_argument("--tasks-dir", type=Path, default=_HERE / "tasks")
    parser.add_argument("--submissions-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write the parity payload to parity_experiment.json.",
    )
    args = parser.parse_args()

    if args.mode == "eval-parity":
        result = eval_parity(args.tasks_dir)
    elif args.mode == "side-by-side":
        result = side_by_side_parity(
            args.tasks_dir,
            submissions_dir=args.submissions_dir,
            limit=args.limit,
        )
    else:
        result = full_parity(
            args.tasks_dir,
            submissions_dir=args.submissions_dir,
            limit=args.limit,
        )
    if args.record and args.mode in {"full", "side-by-side"}:
        PARITY_EXPERIMENT.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
