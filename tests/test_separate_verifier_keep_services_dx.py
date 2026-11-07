"""A task that keeps services cannot have a separate verifier (DF-4).

Dogfood p2-task-author (2026-10-18): ``tasks/svc`` keeps the agent's HTTP
server on port 8000 for grading. Under ``--verifier-sandbox separate`` the
verifier sandbox had no server, the oracle scored a plain 0, and ``bench
tasks check --verifier-sandbox separate`` said the switch "can run it". The
plan-gate and check parity is in ``test_sv_switch.py`` (``kept-services``);
these cover the task that declares the separate verifier itself.
"""

from __future__ import annotations

from pathlib import Path

from benchflow.task import Task
from benchflow.task.runtime_capabilities import validate_task_runtime_support

# runtime_capabilities.KEPT_SERVICES_ISSUE_PATH
KEPT_SERVICES_ISSUE_PATH = "metadata.x-keep-services"

_TASK_MD = """---
schema_version: "1.3"
metadata:
  x-keep-services:
    - port: 8000
      name: wordfreq-http
agent:
  timeout_sec: 300
  user: agent
verifier:
  timeout_sec: 120
{verifier}
---
Start an HTTP server on port 8000 that keeps running after you finish.
"""


def _task(root: Path, verifier: str = "") -> Path:
    task = root / "svc"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "verifier").mkdir()
    (task / "verifier" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "oracle").mkdir()
    (task / "oracle" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    (task / "task.md").write_text(_TASK_MD.format(verifier=verifier))
    return task


def test_a_task_declaring_both_is_refused_at_launch(tmp_path: Path) -> None:
    task = _task(tmp_path, "  sandbox_mode: separate")
    config = Task(task).config

    issues = validate_task_runtime_support(config, sandbox="docker", task_dir=task)

    kept = [issue for issue in issues if issue.path == KEPT_SERVICES_ISSUE_PATH]
    assert len(kept) == 1, issues
    message = kept[0].format()
    assert "port 8000 (wordfreq-http)" in message
    assert "a separate verifier sandbox has none of them" in message
    assert "remove x-keep-services" in message


def test_a_shared_verifier_keeps_its_services(tmp_path: Path) -> None:
    task = _task(tmp_path)
    config = Task(task).config

    issues = validate_task_runtime_support(config, sandbox="docker", task_dir=task)

    assert not [i for i in issues if i.path == KEPT_SERVICES_ISSUE_PATH]
