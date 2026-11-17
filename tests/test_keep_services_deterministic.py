"""A service the ACP agent leaves running is graded in audit (K2, deterministic).

``claude-agent-acp`` on the scripted fake model (the deterministic tier)
starts an HTTP service on port 5005 with its Bash tool, detached, and ends its
turn; the task names the port in ``metadata.x-keep-services``. The verifier
connects to it.

The run's integrity is the default, audit ((g2), F2b 19:37Z; the
faithful default, 09:58Z). S2 (12:36Z): audit runs no kill before grading,
so the service is alive while test.sh reads it, the trial scores 1, and no
``kill-before-grading.json`` is written. The rollout ends the agent's
processes after grading. That end is asserted on Docker in
``tests/test_keep_services_docker.py``, where the test wraps it; this trial
runs ``bench eval run`` in a subprocess, which a monkeypatch cannot reach.
Strict refuses a kept-services task before the agent's turn (DF-4).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
REQUESTED = os.environ.get(h.SANDBOX_ENV, "").strip().lower() in {"docker", "daytona"}

SERVICE_TEST_SH = """#!/bin/bash
# Reward 1 when the service the agent left running still answers.
out=$(curl -s --max-time 5 http://127.0.0.1:5005/index.html)
if [ "$out" = kept ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


def _materialize(root: Path) -> Path:
    tasks = root / "tasks"
    tasks.mkdir(parents=True)
    task = h.materialize_task(h.TaskVariant("keep-service", "keep-service"), tasks)
    text = (task / "task.md").read_text()
    marker = "metadata:\n  author_name: benchflow\n"
    assert marker in text
    (task / "task.md").write_text(
        text.replace(marker, marker + "  x-keep-services:\n  - port: 5005\n", 1)
    )
    test_sh = task / "tests" / "test.sh"
    test_sh.write_text(SERVICE_TEST_SH)
    test_sh.chmod(0o755)
    return task


def test_a_service_the_acp_agent_leaves_running_is_kept(tmp_path):
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    task = _materialize(tmp_path)
    fake = None
    if not h.proxy_runs_in_sandbox(SANDBOX):
        fake = h.host_fake_llm(tmp_path / "host-fake-llm.jsonl")
    url = fake.__enter__() if fake is not None else None
    try:
        run = h.run_bench(
            ["eval", "run"],
            tasks_dir=task,
            jobs_dir=tmp_path / "jobs",
            sandbox=SANDBOX,
            host_fake_url=url,
        )
    finally:
        if fake is not None:
            fake.__exit__(None, None, None)
    trial = run.trial("keep-service")
    result = json.loads((trial / "result.json").read_text())
    # test.sh read the service during grading, and the trial scores what it
    # wrote.
    assert (result.get("rewards") or {}).get("reward") == 1.0, run.output[-4000:]
    config = json.loads((trial / "config.json").read_text())
    assert config["integrity"] == "audit", config
    # No kill before grading in audit (S2), so no kill record. Its rows (the
    # kept process on port 5005, its command, "the task" as where the port
    # came from, kept_services_no_listener) and the release's check of the
    # links (kept_services_dropped) are strict's hardening (DF-4).
    assert not (trial / "kill-before-grading.json").exists()
