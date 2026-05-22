"""Review integration test results.

Reads jobs/<agent>/*/result.json and summary.json files produced by
``bench eval create`` and validates:

1. Every expected agent produced a jobs directory
2. Each trial has a valid result.json (schema check)
3. summary.json exists and has required fields
4. No infrastructure errors (agent_install, timeout, pipe)
5. Score table printed for human review

Usage::

    python tests/integration/check_results.py jobs/integration gemini pi-acp
    python tests/integration/check_results.py jobs/integration   # all subdirs

Guards: ENG-6 integration test plan (PR #255).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULT_REQUIRED = {"task_name", "agent", "rewards", "error", "verifier_error"}
SUMMARY_REQUIRED = {"total", "passed", "failed", "errored", "score"}
INFRA_ERRORS = {"agent_install", "timeout", "pipe_closed", "sandbox_setup"}


def load_results(agent_dir: Path) -> list[dict]:
    """Load the latest result.json per task from an agent's jobs directory."""
    latest_by_task: dict[str, tuple[float, dict]] = {}
    for rfile in sorted(agent_dir.rglob("result.json")):
        try:
            result = json.loads(rfile.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: bad result file {rfile}: {e}")
            continue
        task_name = result.get("task_name") or rfile.parent.name.rsplit("__", 1)[0]
        mtime = rfile.stat().st_mtime
        previous = latest_by_task.get(task_name)
        if previous is None or mtime > previous[0]:
            latest_by_task[task_name] = (mtime, result)
    return [result for _, result in latest_by_task.values()]


def check_agent(agent_dir: Path) -> dict:
    """Validate one agent's output. Returns a summary dict."""
    agent = agent_dir.name
    findings: dict = {"agent": agent, "ok": True, "issues": []}

    # Find the latest run directory
    run_dirs = sorted(
        (d for d in agent_dir.iterdir() if d.is_dir()),
        key=lambda d: d.name,
    )
    if not run_dirs:
        findings["ok"] = False
        findings["issues"].append("no run directory found")
        return findings

    latest = run_dirs[-1]
    results = load_results(latest)

    if not results:
        findings["ok"] = False
        findings["issues"].append("no result.json files")
        return findings

    # Schema check
    for r in results:
        missing = RESULT_REQUIRED - set(r.keys())
        if missing:
            findings["issues"].append(f"{r.get('task_name', '?')}: missing {missing}")
            findings["ok"] = False

    # Infrastructure errors
    infra_errors = []
    for r in results:
        err = r.get("error")
        if err and any(tag in str(err).lower() for tag in INFRA_ERRORS):
            infra_errors.append(f"{r.get('task_name', '?')}: {err}")
    if infra_errors:
        findings["issues"].extend(infra_errors)
        findings["ok"] = False

    # Summary.json — bench eval create writes it at the agent_dir root
    summary_path = agent_dir / "summary.json"
    if not summary_path.exists():
        summary_path = latest / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            missing = SUMMARY_REQUIRED - set(summary.keys())
            if missing:
                findings["issues"].append(f"summary.json missing: {missing}")
                findings["ok"] = False
            findings["summary"] = summary
        except json.JSONDecodeError:
            findings["issues"].append("summary.json: invalid JSON")
            findings["ok"] = False
    else:
        findings["issues"].append("summary.json not found")
        findings["ok"] = False

    # Stats
    findings["total"] = len(results)
    findings["passed"] = sum(
        1
        for r in results
        if (r.get("rewards") or {}).get("reward") == 1.0
        and not r.get("error")
        and not r.get("verifier_error")
    )
    findings["failed"] = sum(
        1
        for r in results
        if (r.get("rewards") or {}).get("reward") is not None
        and (r.get("rewards") or {}).get("reward") != 1.0
        and not r.get("error")
        and not r.get("verifier_error")
    )
    findings["errored"] = sum(1 for r in results if r.get("error"))
    findings["verifier_errored"] = sum(1 for r in results if r.get("verifier_error"))

    if summary := findings.get("summary"):
        for key in ("total", "passed", "failed", "errored", "verifier_errored"):
            if key in summary and summary[key] != findings[key]:
                findings["issues"].append(
                    f"summary.json {key}={summary[key]} but results imply {findings[key]}"
                )
                findings["ok"] = False

    return findings


def _is_rollout_artifact_root(path: Path) -> bool:
    """Return True when *path* is one completed bench eval artifact root."""
    if not (path / "summary.json").is_file():
        return False
    return any(path.rglob("result.json"))


def discover_agent_dirs(jobs_root: Path, agents: list[str] | None) -> list[Path]:
    """Discover result roots accepted by the audit CLI."""
    if agents:
        return [jobs_root / a for a in agents if (jobs_root / a).is_dir()]
    if _is_rollout_artifact_root(jobs_root):
        return [jobs_root]
    return sorted(
        d for d in jobs_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: check_results.py <jobs_root> [agent ...]")
        sys.exit(1)

    jobs_root = Path(sys.argv[1])
    agents = sys.argv[2:] if len(sys.argv) > 2 else None

    if not jobs_root.exists():
        print(f"ERROR: {jobs_root} does not exist")
        sys.exit(1)

    agent_dirs = discover_agent_dirs(jobs_root, agents)

    if not agent_dirs:
        print("ERROR: no agent directories found")
        sys.exit(1)

    all_findings = []
    for d in agent_dirs:
        findings = check_agent(d)
        all_findings.append(findings)

    # Print table
    print()
    print(
        f"{'Agent':<25} {'Score':>8} {'Pass':>5} {'Fail':>5} {'Err':>5} {'Status':>8}"
    )
    print("-" * 62)
    any_fail = False
    for f in all_findings:
        status = "OK" if f["ok"] else "FAIL"
        if not f["ok"]:
            any_fail = True
        total = f.get("total", 0)
        passed = f.get("passed", 0)
        score = f"{passed / total:.1%}" if total else "N/A"
        print(
            f"{f['agent']:<25} {score:>8} "
            f"{passed:>5} {f.get('failed', 0):>5} "
            f"{f.get('errored', 0):>5} {status:>8}"
        )
        for issue in f.get("issues", []):
            print(f"  ⚠ {issue}")

    print("-" * 62)

    if any_fail:
        print("\nSome agents had issues. See warnings above.")
        sys.exit(1)
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
