"""Provider admission of a kept-service task looks at the runner (K1-R11-2).

At 3d0845ba the plan gate refused every trial of a task that keeps services
on a sandbox with no freezer (direct Daytona, Modal), before it knew what
would run: so the root oracle and nop controls of such a task, which need no
freezer (nop runs nothing; a root oracle's service is never kept), could not
run there. Now nop is admitted, the oracle is admitted and judged by the
identity it really runs as before solve.sh (the freezer probe, which refuses
where there is none, tests/test_kept_freezer_probe_start.py). An agent is
refused too, but at setup, once DT2 has measured whether its verifier is
root: before that no spelling tells.
"""

from pathlib import Path

import pytest

from benchflow.task.plan_validation import validate_task_plan


def _task(root: Path) -> Path:
    task_dir = root / "keeps"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n'
        '[metadata]\n"x-keep-services" = [{ port = 5000, name = "api" }]\n'
        "[agent]\ntimeout_sec = 1\n"
        "[verifier]\ntimeout_sec = 1\n"
    )
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    return task_dir


@pytest.mark.parametrize("sandbox", ["daytona", "modal"])
@pytest.mark.parametrize("agent", ["nop", "oracle"])
def test_controls_are_admitted_where_no_freezer_is(tmp_path, sandbox, agent):
    validate_task_plan(
        _task(tmp_path), sandbox=sandbox, agent=agent, sandbox_user="agent"
    )


def test_an_agent_is_refused_at_setup_not_by_the_plan_gate(tmp_path):
    """(Modal runs no ACP agent at all, so direct Daytona is the case.) The
    refusal itself is tests/test_kept_freezer_probe_start.py's
    test_a_sandbox_with_no_freezer_refuses_only_a_measured_kept_agent."""
    validate_task_plan(
        _task(tmp_path),
        sandbox="daytona",
        agent="claude-agent-acp",
        sandbox_user="agent",
    )
