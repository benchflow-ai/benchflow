"""A criteria runner's reward is the one it wrote (read-back #1), on real Docker.

Deterministic tier: whole oracle trials, the sandbox user ``agent``. The
criteria-runner verifier cleared its reward folder to 0777 and never closed
it, and read reward.json live: a process of the agent's uid, alive through
grading in integrity off and audit, replaced the runner's reward.json with
its own. The folder is now closed to the agent while the runner runs, as
test.sh's is, and the reward is read from the copy taken at its exit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.exec_diet_fakes import docker_available
from tests.test_acc_run2_docker import BASE, _task
from tests.test_verifier_strategies import (
    _CRITERIA_RUNNER_VERIFIER_MD,
    _criteria_toml,
)

pytestmark = [pytest.mark.deterministic, pytest.mark.verifier_runtime]

# The runner: c1 fails; it stays a second after writing, long enough for a
# writer that polls every 50 ms to replace its reward.
RUNNER = """import json, os, time
with open(os.environ["BENCHFLOW_REWARD_JSON"], "w") as f:
    json.dump({"metrics": {"c1": 0.0}}, f)
time.sleep(1)
"""
# The agent's writer: its uid, a reward of its own over the runner's.
WRITER = """import json, os, pwd, time
agent = pwd.getpwnam("agent")
os.setgroups([])
os.setgid(agent.pw_gid)
os.setuid(agent.pw_uid)
said = {"ok": 0, "refused": 0}
end = time.time() + 240
while time.time() < end:
    try:
        with open("/logs/verifier/.forged", "w") as f:
            json.dump({"metrics": {"c1": 1.0}}, f)
        os.replace("/logs/verifier/.forged", "/logs/verifier/reward.json")
        said["ok"] += 1
    except OSError:
        said["refused"] += 1
    with open("/tmp/writer.json", "w") as f:
        json.dump(said, f)
    time.sleep(0.05)
"""
SOLVE = (
    "#!/bin/bash\n"
    "echo 41 > /app/answer.txt\n"
    "cat > /opt/writer.py <<'PY'\n" + WRITER + "PY\n"
    "setsid nohup python3 /opt/writer.py </dev/null >/dev/null 2>&1 &\n"
    "sleep 0.5\n"
)


@pytest.fixture(autouse=True)
def _needs_docker():
    if not docker_available():
        pytest.skip("needs a Docker daemon")


def _criteria_task(root: Path, name: str) -> Path:
    task = _task(
        root,
        name,
        dockerfile=BASE,
        solve=SOLVE.removeprefix("#!/bin/bash\n"),
        test="#!/bin/bash\nexit 0\n",
    )
    tests = task / "tests"
    (tests / "verifier.md").write_text(_CRITERIA_RUNNER_VERIFIER_MD)
    (tests / "criteria_runner").mkdir()
    (tests / "criteria_runner" / "reward.py").write_text(RUNNER)
    (tests / "rubrics").mkdir()
    (tests / "rubrics" / "criteria.toml").write_text(
        _criteria_toml("weighted_mean", {"c1": 1.0})
    )
    return task


@pytest.mark.parametrize("mode", ["audit", "off"])
async def test_a_criteria_runner_scores_its_own_reward(tmp_path, mode):
    from benchflow.rollout import Rollout, RolloutConfig

    task = _criteria_task(tmp_path, f"criteria-{mode}")
    result = await Rollout(
        RolloutConfig(
            task_path=task,
            agent="oracle",
            environment="docker",
            sandbox_user="agent",
            jobs_dir=tmp_path / "jobs",
            job_name=f"criteria-{mode}",
            rollout_name=f"criteria-{mode}__oracle",
            integrity=mode,
        )
    ).run()
    assert result.verifier_error is None, result.verifier_error
    # The runner's c1 = 0, never the agent's 1.
    assert result.rewards is not None and result.rewards["reward"] == 0.0, (
        result.rewards
    )
    at_exit = result.rollout_dir / "verifier-at-exit" / "reward.json"
    assert json.loads(at_exit.read_text()) == {"metrics": {"c1": 0.0}}
