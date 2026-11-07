"""Guards SDK-ENV-ARTIFACT-PATH, introduced on first-parent 546aca59f (#388).

At fcc135b4b Docker wrote reward.txt into the Environment's bind mount,
while the Rollout looked in a second directory and returned no reward.
Set BENCHFLOW_TB2_TASKS_DIR to also exercise the real TB2 fix-git task.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import benchflow as bf
from tests.exec_diet_fakes import docker_available

DOCKER_RUN_DEFAULTS = True
pytestmark = [
    pytest.mark.deterministic,
    pytest.mark.verifier_runtime,
    pytest.mark.skipif(not docker_available(), reason="needs a Docker daemon"),
]

UBUNTU = (
    "ubuntu@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55"
)


def _task(root: Path) -> Path:
    task = root / "task"
    source = os.environ.get("BENCHFLOW_TB2_TASKS_DIR")
    if source:
        shutil.copytree(Path(source) / "fix-git", task)
        return task
    for sub in ("environment", "solution", "tests"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec = 120.0\n'
        "[verifier]\ntimeout_sec = 120.0\n"
        "[environment]\nbuild_timeout_sec = 600.0\n"
    )
    (task / "instruction.md").write_text("Write solved into /app/answer.\n")
    (task / "environment/Dockerfile").write_text(f"FROM {UBUNTU}\nWORKDIR /app\n")
    (task / "solution/solve.sh").write_text("#!/bin/bash\necho solved > /app/answer\n")
    (task / "tests/test.sh").write_text(
        '#!/bin/bash\nif [ "$(cat /app/answer 2>/dev/null)" = solved ]; then\n'
        "echo 1 > /logs/verifier/reward.txt\n"
        "else\necho 0 > /logs/verifier/reward.txt\nfi\n"
    )
    return task


@pytest.mark.parametrize("agent,reward", [("oracle", 1.0), ("nop", 0.0)])
@pytest.mark.parametrize("caller_started", [False, True])
async def test_agent_environment_scores_from_its_verifier_mount(
    tmp_path, monkeypatch, agent, reward, caller_started
):
    """Guards 546aca59f: actual verifier output and result share one trial."""
    monkeypatch.chdir(tmp_path)
    if os.environ.get("BENCHFLOW_TB2_TASKS_DIR"):
        monkeypatch.delenv("BENCHFLOW_PLUGIN_RELEASES", raising=False)
    task = _task(tmp_path)
    env = bf.Environment.from_task(
        task, sandbox="docker", rollout_name=f"sdkenv-{agent}"
    )
    paths = env.inner.rollout_paths
    try:
        if caller_started:
            await env.start()
        result = await bf.run(
            bf.Agent(agent, model=""), env, bf.RuntimeConfig(timeout=900)
        )
        assert result.error is None, result.error
        assert result.verifier_error is None, result.verifier_error
        assert result.reward == reward
        assert result.rollout_dir == paths.rollout_dir
        assert float((paths.rollout_dir / "verifier/reward.txt").read_text()) == reward
        saved = json.loads((result.rollout_dir / "result.json").read_text())
        assert saved["rewards"] == {"reward": reward}
        assert env._started, "the caller still owns the environment"
    finally:
        await env.stop()
