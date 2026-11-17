"""A reward test.sh did not write is never scored, in every mode (the lead, 19:40Z).

Audit and off run test.sh as written beside the agent's processes (the
faithful default, 09:58Z; S2, 12:36Z), so the hardened grader that
tests/test_grading_integrity_docker.py and tests/test_grader_access_docker.py
check is strict's. What every mode keeps is the reward's provenance: the
scored reward is the one test.sh wrote while it graded.

Here test.sh writes nothing. The agent, as the sandbox's agent user, plants
reward.txt = 0.5 and leaves a process that keeps rewriting it, or links
reward.txt to the host's ip_forward (which reads 1); each task has a
separate-verifier twin. The oracle, as root, plants 0.5 and leaves nothing
running. In audit and in off every trial is unscored ("No reward file
found"), never 0.5 or 1: the plant is cleared before test.sh, the agent user
cannot write the root 0755 folder while test.sh runs, and the link is gone
from the scored outputs.

A root process the agent leaves running can still write the reward while
test.sh runs in audit and off; the lead's 19:59Z ruling records that case
rather than refusing it, and it is not checked here.

Marked ``deterministic``: a real ``bench`` and a real Docker sandbox, no model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import test_grading_integrity_docker as gi

pytestmark = pytest.mark.deterministic

MODES = ("audit", "off")
SILENT_TEST_SH = "#!/bin/bash\n# Writes no reward.\nexit 0\n"
# The agent's plants, one class per task (GI_VECTOR, from the image).
PLANT_SH = r"""#!/bin/bash
echo 41 > "$PWD/answer.txt"
case "$GI_VECTOR" in
silent-file)
  echo 0.5 > /logs/verifier/reward.txt && echo "live reward-txt"
  ( while :; do echo 0.5 > /logs/verifier/reward.txt 2>/dev/null; sleep 0.1; done ) </dev/null >/dev/null 2>&1 &
  sleep 0.3; kill -0 $! 2>/dev/null && echo "live survivor"
  ;;
silent-link)
  ln -s /proc/sys/net/ipv4/ip_forward /logs/verifier/reward.txt && echo "live reward-link"
  ;;
esac
"""
# The oracle's, as root: a reward file and nothing left running.
ROOT_SOLVE_SH = (
    "#!/bin/bash\necho 41 > /app/answer.txt\necho 0.5 > /logs/verifier/reward.txt\n"
)
AGENT_TASKS = ("silent-file", "silent-file-sep", "silent-link", "silent-link-sep")
ROOT_TASKS = ("root-preplant",)


def _write_task(
    tasks: Path, name: str, vector: str, *, separate: bool, solve: str
) -> None:
    task = tasks / name
    for sub in ("environment", "verifier", "oracle"):
        (task / sub).mkdir(parents=True, exist_ok=True)
    (task / "environment" / "Dockerfile").write_text(
        gi.DOCKERFILE.format(tail="", workdir="/app", vector=vector)
    )
    extra = "  sandbox_mode: separate\n" if separate else ""
    (task / "task.md").write_text(
        gi.TASK_MD.format(verifier_extra=extra, workdir="/app", name=name)
    )
    for path, text in (
        (task / "verifier" / "test.sh", SILENT_TEST_SH),
        (task / "oracle" / "solve.sh", solve),
    ):
        path.write_text(text)
        path.chmod(0o755)


def _trials(jobs: Path) -> dict[str, Path]:
    return {
        path.parent.name.rsplit("__", 1)[0]: path.parent
        for path in jobs.glob("*/*__*/result.json")
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, dict[str, Path]]:
    """Per mode: one job of the planting agent, one of the root oracle."""
    if gi.SANDBOX != "docker":
        pytest.skip(gi.SKIP_REASON if gi.SANDBOX is None else "Docker only")
    root = tmp_path_factory.mktemp("reward-provenance")
    agent_tasks, root_tasks = root / "agent-tasks", root / "root-tasks"
    honest = "#!/bin/bash\necho 42 > /app/answer.txt\n"
    for name in AGENT_TASKS:
        vector = name.removesuffix("-sep")
        _write_task(
            agent_tasks, name, vector, separate=name.endswith("-sep"), solve=honest
        )
    for name in ROOT_TASKS:
        _write_task(root_tasks, name, name, separate=False, solve=ROOT_SOLVE_SH)
    plant = root / "plant.sh"
    plant.write_text(PLANT_SH)
    plant.chmod(0o755)
    found: dict[str, dict[str, Path]] = {}
    for mode in MODES:
        agent_jobs, root_jobs = root / f"jobs-agent-{mode}", root / f"jobs-root-{mode}"
        said = gi._run(agent_tasks, agent_jobs, f"script:{plant}", "--integrity", mode)
        said += gi._run(
            root_tasks,
            root_jobs,
            "oracle",
            "--integrity",
            mode,
            "--sandbox-user",
            "null",
        )
        found[mode] = {**_trials(agent_jobs), **_trials(root_jobs)}
        assert set(found[mode]) == set(AGENT_TASKS + ROOT_TASKS), said[-3000:]
    return found


@pytest.mark.parametrize("mode", MODES)
def test_the_agents_plants_were_made(runs, mode):
    """Each plant ran in the agent's own run (no fixture bug hides one)."""
    expected = {
        "silent-file": ("live reward-txt", "live survivor"),
        "silent-link": ("live reward-link",),
    }
    for name in AGENT_TASKS:
        said = (runs[mode][name] / "agent" / "script.txt").read_text()
        for marker in expected[name.removesuffix("-sep")]:
            assert marker in said, (mode, name, said)


@pytest.mark.parametrize("mode", MODES)
def test_a_reward_test_sh_did_not_write_is_never_scored(runs, mode):
    """Planted, rewritten during grading or linked, it is cleared or kept out:
    the trial is unscored, never scored 0.5 or 1."""
    for name, trial in sorted(runs[mode].items()):
        result = json.loads((trial / "result.json").read_text())
        assert result.get("rewards") is None, (mode, name, result.get("rewards"))
        assert "No reward file found" in (result.get("verifier_error") or ""), (
            mode,
            name,
            result.get("verifier_error"),
        )
        at_exit = trial / "verifier-at-exit" / "reward.txt"
        assert not (at_exit.exists() or at_exit.is_symlink()), (mode, name)
