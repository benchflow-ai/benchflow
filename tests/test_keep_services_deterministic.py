"""A service the ACP agent leaves running is kept for grading (K2, deterministic).

``claude-agent-acp`` on the scripted fake model (the deterministic tier)
starts an HTTP service on port 5005 with its Bash tool, detached, and ends its
turn; the task names the port in ``metadata.x-keep-services``. The verifier
connects to it. Without the key the kill before grading ends the service and
the trial scores 0; with it the service is kept and the trial scores 1, and
the kill's record names the kept process and where the port came from.
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
    assert (result.get("rewards") or {}).get("reward") == 1.0, run.output[-4000:]
    record = json.loads((trial / "kill-before-grading.json").read_text())
    rows = record["kept_services"]
    assert [r["port"] for r in rows] == [5005], record
    assert "http.server 5005" in rows[0]["command"]
    assert rows[0]["source"] == "the task"
    assert record["kept_services_no_listener"] == []
    # The release found no way into /logs/verifier and could read the links
    # (as the agent user, since root in Docker cannot): nothing was dropped.
    assert "kept_services_dropped" not in record, record
