"""A task.md verifier's ``network = "none"`` (the lead's 07:59Z
ruling): one reader. The plan writes it where the run reads every task's
verifier network (``verifier.network_mode``, read by
:func:`benchflow.sandbox.verifier_network.verifier_network_mode`), so
BenchFlow's helper holds it, for every process, as for any task. The task.md
verifier's own offline step defers to that hold (no root iptables in main).
Where the rules cannot be held, the trial ends "verifier network not held",
never open."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import verifier_network as vn
from benchflow.task.config import TaskConfig


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _draft2(root: Path, config: str):
    from benchflow.task import Task
    from benchflow.task.formats import materialize_task_dir

    _write(root / "task.md", f"Do it.\n\n```toml task\n{config}```\n")
    _write(root / "sandbox" / "Dockerfile", "FROM ubuntu:24.04\nWORKDIR /app\n")
    _write(
        root / "verifier" / "test.sh",
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
    )
    return Task(materialize_task_dir(root)).config


@pytest.mark.parametrize(
    ("config", "mode"),
    [
        ('[sandbox]\nnetwork = "open"\n\n[verifier]\nnetwork = "none"\n', "no-network"),
        (
            '[agent]\nnetwork = ["github.com"]\n\n[sandbox]\nnetwork = "open"\n\n'
            '[verifier]\nnetwork = "none"\n',
            "no-network",
        ),
        (
            '[sandbox]\nnetwork = ["pypi.org"]\n\n[verifier]\nnetwork = "none"\n',
            "no-network",
        ),
        # Public controls: an open verifier stays open.
        ('[sandbox]\nnetwork = "open"\n\n[verifier]\nnetwork = "open"\n', "public"),
        ('[sandbox]\nnetwork = "open"\n', "public"),
    ],
    ids=["none", "none-agent-allowlist", "none-sandbox-allowlist", "open", "unset"],
)
def test_draft_2_verifier_network_is_read_by_the_runs_one_reader(
    tmp_path, config, mode
):
    assert vn.verifier_network_mode(_draft2(tmp_path, config)) == mode


@pytest.mark.parametrize(
    ("toml", "mode"),
    [
        (
            '[environment]\nallow_internet = true\n[verifier]\nnetwork_mode = "no-network"\n',
            "no-network",
        ),
        (
            '[environment]\nallow_internet = true\n[verifier]\nnetwork_mode = "public"\n',
            "public",
        ),
        ("[environment]\nallow_internet = true\n", "public"),
    ],
    ids=["no-network", "public", "unset"],
)
def test_draft_1_verifier_network_reads_the_same(toml, mode):
    config = TaskConfig.model_validate_toml('version = "1.0"\n' + toml)
    assert vn.verifier_network_mode(config) == mode


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "held",
    [
        vn.VerifierNetwork(
            "no-network", "held", match="every process", via_helper=True
        ),
        vn.VerifierNetwork(
            "no-network", "no route, counted", match="every process", via_helper=True
        ),
    ],
    ids=["held", "no-route-counted"],
)
async def test_the_offline_step_defers_to_the_helpers_hold(monkeypatch, held):
    from benchflow.taskmd import verify

    root = AsyncMock(side_effect=AssertionError("no root iptables in main"))
    monkeypatch.setattr(verify, "root_exec", root)
    sandbox = SimpleNamespace(_benchflow_verifier_network=held)
    note = await verify.take_offline(sandbox)
    assert note.startswith("offline: BenchFlow's helper holds the verifier's network")
    root.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "armed",
    [
        None,
        vn.VerifierNetwork("public", "open"),
        # A hold that is not every process's is not taken for one.
        vn.VerifierNetwork("no-network", "held"),
    ],
    ids=["none-armed", "public", "not-every-process"],
)
async def test_without_that_hold_root_takes_it_offline_or_the_trial_is_unscored(
    monkeypatch, armed
):
    from benchflow.taskmd import verify

    root = AsyncMock(
        return_value=SimpleNamespace(
            return_code=1,
            stdout="online: the container reaches the internet and iptables is not available",
        )
    )
    monkeypatch.setattr(verify, "root_exec", root)
    sandbox = SimpleNamespace(_benchflow_verifier_network=armed)
    with pytest.raises(
        verify.TaskMdVerifierError, match="could not take the verifier offline"
    ):
        await verify.take_offline(sandbox)
    root.assert_awaited_once()


@pytest.mark.asyncio
async def test_rules_that_cannot_be_held_end_the_trial_never_open(monkeypatch):
    """The helper could not add the verifier's rules: test.sh does not run,
    and the ending is its own (not "verifier crashed")."""
    from benchflow.task.verifier_errors import VerifierNetworkUnheldError

    async def facts(env, *a, **k):
        return SimpleNamespace(stdout="", return_code=0)

    async def install(env, v4, v6, nat):
        return "BF verifier-network status=3\n", True, ([], [], [])

    monkeypatch.setattr(
        vn, "route_facts", lambda out: (2, ("172.18.0.0/16",), ("172.18.0.1",))
    )
    monkeypatch.setattr(vn, "_install_recorded", install)
    env = SimpleNamespace(exec=facts)
    task = SimpleNamespace(
        config=TaskConfig.model_validate_toml(
            'version = "1.0"\n[verifier]\nnetwork_mode = "no-network"\n'
        )
    )
    with pytest.raises(
        VerifierNetworkUnheldError, match="could not be enforced"
    ) as raised:
        await vn.arm_verifier_network(env, task, "no-network")
    assert isinstance(raised.value, RuntimeError)
    assert str(raised.value).startswith(
        "Verifier hardening failed: the verifier's network"
    )


@pytest.mark.asyncio
async def test_the_rollout_ends_it_verifier_network_not_held(tmp_path):
    from benchflow._utils.scoring import (
        VERIFIER_NETWORK_UNHELD,
        classify_verifier_error,
    )
    from benchflow.rollout._setup import _verify_rollout
    from benchflow.task.verifier_errors import VerifierNetworkUnheldError

    async def harden(env, task, sandbox_user, **kwargs):
        raise VerifierNetworkUnheldError(
            "Verifier hardening failed: the verifier's network (in-container under "
            "no-network: verifier.network_mode, else the sandbox's) could not be "
            "enforced for test.sh (its rules could not be added (status 3))"
        )

    def verifier(**kwargs):
        raise AssertionError("test.sh must not run")

    paths = SimpleNamespace(
        verifier_dir=tmp_path / "verifier",
        test_stdout_path=tmp_path / "verifier" / "test-stdout.txt",
        rollout_dir=tmp_path,
    )
    task = SimpleNamespace(
        name="t",
        config=TaskConfig.model_validate_toml(
            'version = "1.0"\n[verifier]\nnetwork_mode = "no-network"\n'
        ),
    )
    rewards, error, _ = await _verify_rollout(
        SimpleNamespace(),
        task,
        paths,
        {},
        SimpleNamespace(harden_before_verify=harden, verifier=verifier),
    )
    assert rewards is None
    assert error.startswith(vn.VERIFIER_NETWORK_UNHELD), error
    assert classify_verifier_error(error) == VERIFIER_NETWORK_UNHELD
    assert (
        classify_verifier_error(f"[solver-preserved] {error}")
        == VERIFIER_NETWORK_UNHELD
    )
