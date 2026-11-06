"""Guards FIX-49 r6's private capture sibling gap at 0294404ae (Sol M1)."""

import contextlib
import errno

import pytest

from benchflow._utils import reachable_write as rw
from tests.test_verifier_recovery import child as child
from tests.test_verifier_recovery import trial as trial


def fail_destination(monkeypatch, kind):
    original = rw.create_private_file
    writes = []

    @contextlib.contextmanager
    def create(target):
        if target.name == "reward.txt":
            writes.append(target.name)
            raise OSError(getattr(errno, kind), "fixture private destination denied")
        with original(target) as stream:
            yield stream

    monkeypatch.setattr(rw, "create_private_file", create)
    return writes


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("kind", ["EACCES", "EIO", "ENOSPC"])
async def test_separate_capture_private_creation_is_unscored(
    tmp_path, monkeypatch, prior, kind
):
    from benchflow.rollout import _separate_verifier as separate
    from tests.test_separate_verifier import (
        FakeVerifierSandbox,
        _agent_trial,
        _fake_rollout,
        _task_dir,
    )

    _, workspace, root = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, root, task, str(workspace))
    box = FakeVerifierSandbox(tmp_path / "verifier-box")
    writes = []

    async def verify(env, task, paths, *args, **kwargs):
        env._benchflow_test_command_issued = prior
        paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        paths.reward_text_path.write_text("1")
        writes.append(fail_destination(monkeypatch, kind))
        return {"reward": 1.0}, None

    rewards, error = await separate.run_separate_verifier(
        rollout, create_environment=lambda *_: box, verify=verify
    )
    assert writes and writes[0] == ["reward.txt"]
    assert rewards is None and "private destination denied" in error


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("kind", ["EACCES", "EIO", "ENOSPC"])
async def test_recovery_private_creation_is_unscored(
    monkeypatch, trial, child, prior, kind
):
    from benchflow.rollout import _setup
    from benchflow.rollout import _verifier_recovery as recovery

    writes = []

    async def verify(env, task, paths, *args, **kwargs):
        env._benchflow_test_command_issued = prior
        paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        paths.reward_text_path.write_text("1")
        writes.append(fail_destination(monkeypatch, kind))
        return {"reward": 1.0}, None, None

    monkeypatch.setattr(_setup, "_verify_rollout", verify)
    rewards, error = await recovery.recover_verifier(trial)
    assert writes and writes[0] == ["reward.txt"]
    assert rewards is None and "private destination denied" in error
