"""Guards CONF2-rollout-2 against recovery identity loss from 941a0cc5."""

import asyncio
import json
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.task_authoring import task_digest
from benchflow.review.resume import resume_review
from benchflow.rollout import _verifier_recovery as recovery
from benchflow.rollout._setup import _publish_trajectory_for_verifier
from benchflow.task import Task
from tests.test_verifier_recovery import (
    _save_solver,
)
from tests.test_verifier_recovery import (
    child as child,
)
from tests.test_verifier_recovery import (
    trial as trial,
)


class LogTransport:
    """Run the real publication shell against a fresh dummy sandbox log tree."""

    def __init__(self, root):
        self.root = root

    async def exec(self, command, **kwargs):
        if "/logs/" not in command:
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        proc = await asyncio.create_subprocess_shell(
            command.replace("/logs/", str(self.root) + "/logs/"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return SimpleNamespace(
            return_code=proc.returncode, stdout=out.decode(), stderr=err.decode()
        )

    async def upload_file(self, source, target, **_kwargs):
        if target.startswith("/logs/"):
            shutil.copyfile(source, self.root / target.lstrip("/"))


@pytest.mark.parametrize("agent", ["opencode", "codex-acp", "oracle", "nop"])
@pytest.mark.parametrize(
    "backend,contract",
    [
        ("docker", "workspace"),
        ("daytona", "workspace"),
        ("remote-docker", "workspace"),
        ("modal", "workspace"),
        ("docker", "submission"),
    ],
)
@pytest.mark.parametrize("reply", ["42", ""])
@pytest.mark.parametrize("entry", ["direct", "score-resume"])
async def test_interrupted_resume_publishes_original_solver_reply(
    trial, child, monkeypatch, agent, backend, contract, reply, entry, tmp_path
):
    trial._config.agent = agent
    trial._config.scenes = []
    trial._config.sandbox_user = None
    trial._config.environment = backend
    if contract == "submission":
        task_file = trial._config.task_path / "task.toml"
        task_file.write_text(
            task_file.read_text().replace(
                "workspace_recovery = true", 'submission_files = ["/app/result.txt"]'
            )
        )
        trial._task = Task(trial._config.task_path)
        trial._config.task_digest = task_digest(trial._config.task_path)
        trial._config.sandbox_user = "65534"
        monkeypatch.setattr(recovery, "admitted_submission", lambda _r: None)
        monkeypatch.setattr(recovery, "restore_submission", AsyncMock())
    root = trial._require_rollout_dir()
    _save_solver(trial)
    source = json.loads((root / "solver.json").read_text())
    source["agent"] = agent
    (root / "solver.json").write_text(json.dumps(source))
    saved = (root / "solver.json").read_bytes()
    config = json.loads((root / "config.json").read_text())
    config.update(environment=backend, sandbox_user=trial._config.sandbox_user)
    (root / "config.json").write_text(json.dumps(config))
    trial._trajectory = [{"type": "agent_message", "text": reply}]
    (root / "trajectory/acp_trajectory.jsonl").write_text(
        json.dumps(trial._trajectory[0]) + "\n"
    )
    starts = 0

    async def start():
        nonlocal starts
        starts += 1
        sandbox = tmp_path / f"sandbox-{starts}"
        (sandbox / "logs/agent").mkdir(parents=True)
        child._env = LogTransport(sandbox)
        if agent in {"oracle", "nop"}:
            (sandbox / "logs/agent/output.txt").write_text("script-owned marker")

    child.start.side_effect = start
    monkeypatch.setattr(
        "benchflow.rollout._setup._publish_trajectory_for_verifier",
        _publish_trajectory_for_verifier,
    )

    async def verify(env, *args, **kwargs):
        if starts == 1:
            raise asyncio.CancelledError()
        output = env.root / "logs/agent/output.txt"
        expected = "script-owned marker" if agent in {"oracle", "nop"} else reply
        reward = float(output.is_file() and output.read_text().strip() == expected)
        return {"reward": reward}, None, None

    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    if backend not in {"docker", "daytona"}:
        rewards, error = await recovery.recover_verifier(trial)
        assert rewards is None and "owned Docker or Daytona" in error
        assert starts == 0
        result = await recovery.resume_verification(root, trial._config.task_path)
        assert result["rewards"] is None
        assert starts == 0
        return
    with pytest.raises(asyncio.CancelledError):
        await recovery.recover_verifier(trial)
    assert not (root / "verification.json").exists()
    result = (
        await recovery.resume_verification(root, trial._config.task_path)
        if entry == "direct"
        else await resume_review(root, tasks_root=trial._config.task_path)
    )
    assert result["rewards"] == {"reward": 1.0}
    assert (root / "solver.json").read_bytes() == saved
    assert child.factory.call_args.args[0].primary_agent == "oracle"
    child.run.assert_not_awaited()
    await recovery.resume_verification(root, trial._config.task_path)
    assert starts == 2


@pytest.mark.parametrize("identity", [None, "", " ", 42, [], {}])
async def test_resume_refuses_unknown_solver_identity(trial, child, identity):
    _save_solver(trial)
    root = trial._require_rollout_dir()
    source = json.loads((root / "solver.json").read_text())
    source["agent"] = identity
    (root / "solver.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="original solver identity"):
        await recovery.resume_verification(root, trial._config.task_path)
    child.factory.assert_not_called()
