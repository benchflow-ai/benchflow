"""#1247 parts 3 and 4 on real Docker: what a root ``solve.sh`` leaves running.

The oracle's processes live through grading in integrity off and audit. On
0.10.0 a root ``solve.sh`` that left a process read ``verified``, and the
names that process added to the mounted ``/logs/verifier`` once test.sh had
exited (the folder is the host user's again then; root writes it anyway)
stayed in the trial's ``verifier/`` unflagged: a file, a folder, a file in
test.sh's own folder, a link to the host's ``/etc/hostname`` and a
``reward.json`` link (measured on gp2, Docker 29.1.3).

- E1: each such name is gone from the trial's ``verifier/`` and named in
  ``integrity/verifier-inputs.json`` (audit; off's record says the detector
  did not run); test.sh's own names and BenchFlow's records stay; the reward
  is test.sh's.
- E2: the trial reads ``unverifiable`` with the process listed, its reward
  kept; an oracle that leaves nothing reads ``verified``.
- The task.md judge's own records after test.sh stay unflagged (on 0.10.0
  its ``reward.json`` and ``reward.txt`` were moved aside as planted).

Mutants: put back every name (E1); skip rule S for the oracle (E2).

Marked ``deterministic``: whole trials on real Docker, no model.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from tests.exec_diet_fakes import docker_available
from tests.test_acc_run2_docker import BASE

pytestmark = [pytest.mark.deterministic, pytest.mark.verifier_runtime]

MODES = ("audit", "off")
TEST_SH = """#!/bin/bash
echo ran > /logs/verifier/out.txt
mkdir -p /logs/verifier/logs && echo log > /logs/verifier/logs/a.txt
ln -s out.txt /logs/verifier/latest
echo 1 > /logs/verifier/reward.txt
"""
# A root process solve.sh leaves: once test.sh's reward is there and the
# folder is the host user's again (after the copy at exit), it plants names
# every 0.2 s until the sandbox goes.
PLANTER = r"""echo 42 > /app/answer.txt
cat > /tmp/planter.sh <<'P'
#!/bin/bash
while :; do
  if [ -e /logs/verifier/reward.txt ] && [ "$(stat -c %u /logs/verifier)" != 0 ]; then
    echo x > /logs/verifier/extra.txt 2>/dev/null
    mkdir -p /logs/verifier/newdir 2>/dev/null && echo y > /logs/verifier/newdir/f.txt 2>/dev/null
    ln -sfn /etc/hostname /logs/verifier/hostlink 2>/dev/null
    [ -e /logs/verifier/reward.json ] || ln -s /proc/sys/net/ipv4/ip_forward /logs/verifier/reward.json 2>/dev/null
    echo z > /logs/verifier/logs/planted.txt 2>/dev/null
    echo planted >> /logs/agent/planter.log
  fi
  sleep 0.2
done
P
chmod +x /tmp/planter.sh
setsid nohup /tmp/planter.sh </dev/null >/dev/null 2>&1 &
"""
PLANTED = ("extra.txt", "newdir", "hostlink", "reward.json", "logs/planted.txt")
LEFTOVER = (
    "echo 42 > /app/answer.txt\nsetsid nohup sleep 900 </dev/null >/dev/null 2>&1 &\n"
)
# The same as the task's [agent].user, which can't write the root-granted /app.
LEFTOVER_AS_USER = "setsid nohup sleep 900 </dev/null >/dev/null 2>&1 &\n"
PLAIN = "echo 42 > /app/answer.txt\n"
SOLVER_UID = 2345


@pytest.fixture(autouse=True)
def _needs_docker():
    if not docker_available():
        pytest.skip("needs a Docker daemon")


def _task(
    tmp_path: Path, name: str, *, solve: str, agent_user: str | None = None
) -> Path:
    task = tmp_path / name
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    agent = "[agent]\ntimeout_sec = 300.0\n"
    dockerfile = BASE
    if agent_user is not None:
        agent += f'user = "{agent_user}"\n'
        dockerfile += (
            f"RUN useradd -m -u {SOLVER_UID} {agent_user} && chown {agent_user} /app\n"
        )
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 300.0\n\n'
        f"{agent}\n[environment]\nbuild_timeout_sec = 900.0\n"
    )
    (task / "instruction.md").write_text("Write /app/answer.txt.\n")
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\nset -e\n" + solve)
    (task / "tests" / "test.sh").write_text(TEST_SH)
    return task


async def _run(task: Path, mode: str, sandbox_user: str | None):
    from benchflow.rollout import Rollout, RolloutConfig

    config = RolloutConfig(
        task_path=task,
        agent="oracle",
        environment="docker",
        sandbox_user=sandbox_user,
        jobs_dir=task.parent / "jobs",
        job_name=f"{task.name}-{mode}",
        rollout_name=f"{task.name}__{mode}",
        integrity=mode,
    )
    result = await Rollout(config).run()
    record = json.loads(
        (result.rollout_dir / "integrity" / "verifier-inputs.json").read_text()
    )
    return result, record


def _names(folder: Path) -> set[str]:
    found = set()
    for root, dirs, files in os.walk(folder):
        for name in dirs + files:
            found.add((Path(root) / name).relative_to(folder).as_posix())
    return found


@pytest.mark.parametrize("mode", MODES)
async def test_names_a_root_process_plants_after_test_sh_are_removed(tmp_path, mode):
    """E1 on Docker. The planter ran: its log is in the trial's agent/
    folder, which the host keeps (off records no flags, so without it the off
    row would pass with nothing planted)."""
    task = _task(tmp_path, f"planter-{mode}", solve=PLANTER)
    result, record = await _run(task, mode, "agent")
    assert result.verifier_error is None, result.verifier_error
    assert result.rewards == {"reward": 1.0}
    planter_log = result.rollout_dir / "agent" / "planter.log"
    assert planter_log.is_file() and planter_log.read_text().strip(), (
        "precondition: the planter planted after test.sh's exit"
    )
    verifier = result.rollout_dir / "verifier"
    names = _names(verifier)
    assert {"out.txt", "logs", "logs/a.txt", "latest", "reward.txt"} <= names, names
    assert os.readlink(verifier / "latest") == "out.txt"
    assert {"test-stdout.txt", "verifier-network.json"} <= names, names
    left = [n for n in PLANTED if n in names or (verifier / n).is_symlink()]
    assert not left, f"planted after test.sh and still in verifier/: {left}"
    if mode == "audit":
        flagged = {
            f["path"]
            for f in record.get("flags") or ()
            if f.get("where") == "verifier_output" and f.get("kind") == "planted"
        }
        assert {f"/logs/verifier/{n}" for n in PLANTED} <= flagged, record.get("flags")
        assert record["status"] == "flagged", record["status"]
        # Named while the sandbox lived, removed after its stop.
        rows = [
            f
            for f in record.get("flags") or ()
            if f.get("where") == "verifier_output" and f.get("kind") == "planted"
        ]
        assert all(f.get("removed") is True and "pending" not in f for f in rows), rows


@pytest.mark.parametrize("sandbox_user", ["agent", None], ids=["default", "null"])
@pytest.mark.parametrize("mode", MODES)
async def test_a_root_solve_sh_that_leaves_a_process_is_unverifiable(
    tmp_path, mode, sandbox_user
):
    """E2 on Docker: ``sleep 900`` left by a root solve.sh, listed; the
    reward stays 1.0."""
    task = _task(tmp_path, f"leftover-{mode}", solve=LEFTOVER)
    result, record = await _run(task, mode, sandbox_user)
    assert result.verifier_error is None, result.verifier_error
    assert result.rewards == {"reward": 1.0}
    found = record["provenance"]
    assert found["verdict"] == "unverifiable", found
    assert any("sleep 900" in str(p.get("argv") or "") for p in found["processes"]), (
        found
    )
    said = json.loads((result.rollout_dir / "result.json").read_text())
    assert any(
        "Reward provenance unverifiable" in w for w in said.get("warnings") or ()
    ), said.get("warnings")


@pytest.mark.parametrize("mode", MODES)
async def test_an_oracle_that_leaves_nothing_is_verified(tmp_path, mode):
    """Guard row."""
    task = _task(tmp_path, f"plain-{mode}", solve=PLAIN)
    result, record = await _run(task, mode, "agent")
    assert result.rewards == {"reward": 1.0}
    found = record["provenance"]
    assert found["verdict"] == "verified", found
    assert found["processes"] == [], found


async def test_an_oracle_run_as_its_agent_user_gets_rule_s(tmp_path):
    """E2, solve.sh as the task's ``[agent].user``: rule S finds its process
    alive at test.sh."""
    task = _task(
        tmp_path, "solver-leftover", solve=LEFTOVER_AS_USER, agent_user="solver"
    )
    result, record = await _run(task, "audit", "agent")
    assert result.rewards == {"reward": 1.0}
    found = record["provenance"]
    assert found["verdict"] == "unverifiable", found
    asked = found.get("alive_at_test") or {}
    assert asked.get("alive") is True and asked.get("uid") == SOLVER_UID, found


async def test_the_taskmd_judges_records_stay_unflagged(tmp_path):
    """E1 guard on Docker, the task.md judge (draft 2) as the oracle, audit:
    its reward.json, reward.txt and review.json stay in verifier/, nothing
    goes to verifier-after-exit/, and the record has no verifier_output flag.
    On 0.10.0: both reward files moved aside, two `planted` flags."""
    source = Path(__file__).parent / "fixtures" / "taskmd" / "examples" / "hello-world"
    task = tmp_path / "hello-world"
    shutil.copytree(source, task)
    result, record = await _run(task, "audit", "agent")
    assert result.rewards and result.rewards.get("reward") == 1.0, result.rewards
    verifier = result.rollout_dir / "verifier"
    assert {"reward.json", "reward.txt", "review.json"} <= _names(verifier)
    # The rollout's timed_out, added after the verify, stays too.
    assert json.loads((verifier / "reward.json").read_text())["timed_out"] is False
    assert not (result.rollout_dir / "verifier-after-exit").exists()
    assert not [
        f for f in record.get("flags") or () if f.get("where") == "verifier_output"
    ], record.get("flags")
