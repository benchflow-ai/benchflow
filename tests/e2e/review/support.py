"""Helpers for the review scenarios of the end-to-end tier.

Each scenario writes synthetic tasks, runs the real ``bench`` CLI against real
Daytona sandboxes and then reads the recorded trial files the way a reviewer
would: every number the SDK wrote must agree with the files it came from.

Environment (same switches as the rest of ``tests/e2e``):

- ``BENCHFLOW_E2E_SANDBOX=daytona`` with ``DAYTONA_API_KEY`` and
  ``BENCHFLOW_DAYTONA_OWNER`` turns the scenarios on; nothing runs otherwise.
- ``BENCHFLOW_E2E_OUT``: folder for tasks, jobs and CLI logs (a temporary
  folder by default).
- Scenarios that need a model are skipped unless their login is present:
  ``CODEX_AUTH_JSON`` (a ChatGPT-login ``auth.json``) or
  ``CLAUDE_CODE_OAUTH_TOKEN``. API keys are never used.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRUB = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def sandbox_or_reason() -> tuple[str | None, str]:
    if os.environ.get("BENCHFLOW_E2E_SANDBOX", "").strip().lower() != "daytona":
        return None, "end-to-end tier is off: set BENCHFLOW_E2E_SANDBOX=daytona"
    for key in ("DAYTONA_API_KEY", "BENCHFLOW_DAYTONA_OWNER"):
        if not os.environ.get(key):
            return None, f"BENCHFLOW_E2E_SANDBOX=daytona needs {key}"
    return "daytona", ""


def env(*, keep_logins: bool = False) -> dict[str, str]:
    out = {k: v for k, v in os.environ.items() if k not in _SCRUB}
    if not keep_logins:
        out.pop("CODEX_AUTH_JSON", None)
        out.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    out.pop("VIRTUAL_ENV", None)
    out.update(
        PYTHONDONTWRITEBYTECODE="1", BENCHFLOW_SKIP_UPDATE_CHECK="1", COLUMNS="200"
    )
    return out


@dataclass
class Run:
    args: list[str]
    returncode: int
    output: str

    def tail(self, n: int = 3000) -> str:
        return self.output[-n:]


def bench(
    *args: str | Path, log: Path, keep_logins: bool = False, timeout: int = 1800
) -> Run:
    exe = Path(sys.executable).with_name("bench")
    argv = [str(exe), *map(str, args)]
    proc = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=env(keep_logins=keep_logins),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = proc.stdout + proc.stderr
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(argv) + "\n\n" + output)
    return Run(argv, proc.returncode, output)


def write_task(
    root: Path,
    name: str,
    *,
    dockerfile: str,
    solve: str,
    test: str,
    toml_extra: str = "",
    instruction: str = "Do the task.\n",
) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True, exist_ok=True)
    (task / "solution").mkdir(exist_ok=True)
    (task / "tests").mkdir(exist_ok=True)
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    (task / "instruction.md").write_text(instruction)
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 120.0\n'
        "[agent]\ntimeout_sec = 300.0\n" + toml_extra
    )
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\n" + solve)
    (task / "tests" / "test.sh").write_text("#!/bin/bash\n" + test)
    for script in (task / "solution" / "solve.sh", task / "tests" / "test.sh"):
        script.chmod(0o755)
    return task


def reward_if(condition: str) -> str:
    return (
        f"if {condition}; then echo 1 > /logs/verifier/reward.txt; "
        "else echo 0 > /logs/verifier/reward.txt; fi\n"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def fresh_job(job: Path) -> Path:
    """Remove an earlier copy of ``job`` so the scenario runs it anew."""
    import shutil

    if job.exists():
        shutil.rmtree(job)
    return job


def trials(job: Path) -> dict[str, Path]:
    """Task name -> trial folder, for the job's own trials (reviewer runs are
    nested deeper and are not included)."""
    out: dict[str, Path] = {}
    for result in sorted(job.glob("*/result.json")):
        out[read_json(result)["task_name"]] = result.parent
    return out
