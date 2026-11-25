"""What the trial's last disconnect stops, in each separate-verifier order and
integrity mode, with the agent's sandbox stopped before cleanup or not.

Every cell runs the turn-end disconnect, then the verifier's path, then the
trial's end:

- ``overlap``: run()'s last verify on a backend that stops the agent's
  sandbox before the verifier (``prestart``: Docker, remote Docker;
  ``after_stop``: Daytona, Modal); the stop goes through.
- ``overlap-stop-failed``: the same order, with a stop that fails, so the
  sandbox is still up at cleanup.
- ``serial-env``: ``BENCHFLOW_VERIFIER_PREPARE=serial``, and ``outside-run``:
  a verify() outside run(); neither stops the sandbox before cleanup.

Strict stops the agent's processes at its turn's end, so the trial's end has
none left to stop. Audit and off keep them through grading and stop them at
the trial's end, unless the sandbox has already stopped: then no process is
left, the turn's end already removed the login, and nothing is exec'd into
the stopped sandbox (before, the stop ran there and failed the cleanup, and
the login's removal failed there and logged a warning). A root agent has no
sandbox user's processes to stop, and its turn's end removes its login for
good. The sandbox is a fake that answers like each backend once it has
stopped: Docker's compose says ``service "main" is not running``, Daytona's
and Modal's exec raise.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from benchflow.agents.registry import AGENTS
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout._separate_verifier import VERIFIER_PREPARE_ENV
from benchflow.sandbox.lockdown import _confirmed_kill_count

LOGIN = "/home/agent/.pi/agent/models.json"
ROOT_LOGIN = "/root/.pi/agent/models.json"
MUSE_LOGIN = "/home/agent/.config/muse/auth.json"
KILL = "bf-test-kill-sandbox-user"
CLEANUP_FAILED = "Credential cleanup at disconnect failed"

BACKENDS = {
    "docker": "prestart",
    "remote-docker": "prestart",
    "daytona": "after_stop",
    "modal": "after_stop",
}
PATHS = ["overlap", "overlap-stop-failed", "serial-env", "outside-run"]
MODES = ["strict", "audit", "off"]


class Sandbox:
    """Answers every command while up; once stopped, as the backend does."""

    def __init__(self, backend: str, *, stop_fails: bool = False) -> None:
        self.backend = backend
        self.stop_fails = stop_fails
        self.stopped = False
        self.commands: list[str] = []

    async def exec(self, command, *args, **kwargs):
        self.commands.append(command)
        if self.stopped:
            if self.backend in ("docker", "remote-docker"):
                return SimpleNamespace(
                    return_code=1,
                    stdout="",
                    stderr='service "main" is not running',
                )
            # Both drop their handle at stop(); exec then refuses.
            word = "build" if self.backend == "daytona" else "start"
            raise RuntimeError(
                f"Sandbox not found. Please {word} the environment first."
            )
        stdout = "BF killed 0 end\n" if command == KILL else ""
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    async def stop(self, delete: bool) -> None:
        if self.stop_fails:
            raise RuntimeError("the provider refused the stop")
        self.stopped = True


class Planes:
    """The process stop, run as a root exec whose receipt must count."""

    def __init__(self) -> None:
        self.quiesced = 0

    async def quiesce_agent(self, env, sandbox_user, **kwargs):
        self.quiesced += 1
        result = await env.exec(KILL, user="root")
        _confirmed_kill_count(result, result.stdout)


@pytest.fixture(autouse=True)
def no_kept_services(monkeypatch):
    monkeypatch.setattr(
        "benchflow.task.keep_services.trial_keep", lambda *a: None, raising=False
    )


def _trial(
    tmp_path,
    backend: str,
    integrity: str,
    sandbox: Sandbox,
    *,
    sandbox_user: str | None = "agent",
    login: str = LOGIN,
) -> Rollout:
    trial = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            agent="pi-acp",
            model=None,
            environment=backend,
            skip_agent_install=True,
            integrity=integrity,
            sandbox_user=sandbox_user,
        )
    )
    trial._rollout_dir = tmp_path
    trial._agent_launch = AGENTS["pi-acp"].launch_cmd
    trial._env = sandbox
    trial._task = SimpleNamespace(config=SimpleNamespace())
    trial._planes = Planes()
    trial._collect_native_acp_usage = lambda: None
    trial._capture_partial_acp_trajectory = lambda: None
    # As connect() leaves one ACP connection that staged one login file.
    trial._connection_needs_quiescence = True
    trial._agent_pkill_done = trial._agent_quiesced = False
    trial._credential_ledger = [login]
    return trial


def _verifier_prepare(monkeypatch, path: str) -> None:
    if path == "serial-env":
        monkeypatch.setenv(VERIFIER_PREPARE_ENV, "serial")
    else:
        monkeypatch.delenv(VERIFIER_PREPARE_ENV, raising=False)


async def _through_the_verifier(trial: Rollout, path: str) -> str | None:
    """The turn's end, then what the verifier's path does to the agent's
    sandbox; the order the trial took."""
    await trial.disconnect()
    trial._release_agent_before_verify = path != "outside-run"
    order = trial._separate_verifier_order()
    if order is not None:
        stops: list[str | None] = []
        await trial._stop_agent_sandbox_before_verify(
            SimpleNamespace(agent_stopped=stops.append)
        )
        assert len(stops) == 1
    return order


async def _at_the_trials_end(trial: Rollout, sandbox: Sandbox, caplog) -> dict:
    """What the trial's last disconnect stopped, exec'd, kept and logged."""
    turn_end_quiesces = trial._planes.quiesced
    sandbox.commands.clear()
    caplog.clear()
    raised = None
    with caplog.at_level(logging.WARNING):
        try:
            await trial._disconnect_at_trial_end()
        except Exception as exc:
            raised = f"{type(exc).__name__}: {exc}"
    return {
        "quiesce": trial._planes.quiesced > turn_end_quiesces,
        "pkill": any(c.startswith("pkill -f ") for c in sandbox.commands),
        "removal": any("rm -f -- " in c for c in sandbox.commands),
        "ledger": trial._credential_ledger,
        "warned": CLEANUP_FAILED in caplog.text,
        "raised": raised,
    }


def _expected(path: str, integrity: str) -> dict:
    """At the trial's end: the process stop, the pkill, the login's removal
    and what the ledger keeps."""
    if integrity == "strict":
        # The turn's end stopped them and removed the login.
        return {"quiesce": False, "pkill": False, "removal": False, "ledger": []}
    if path == "overlap":
        # Stopped with the sandbox: nothing to stop, and the turn's end
        # removed the login (root confirmed it gone) from a sandbox no exec
        # reaches now.
        return {"quiesce": False, "pkill": False, "removal": False, "ledger": []}
    return {"quiesce": True, "pkill": True, "removal": True, "ledger": []}


@pytest.mark.parametrize("integrity", MODES)
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("backend", sorted(BACKENDS))
async def test_the_trials_end_stops_only_what_is_still_running(
    tmp_path, monkeypatch, caplog, backend, path, integrity
):
    _verifier_prepare(monkeypatch, path)
    sandbox = Sandbox(backend, stop_fails=path == "overlap-stop-failed")
    trial = _trial(tmp_path, backend, integrity, sandbox)

    order = await _through_the_verifier(trial, path)

    assert order == (BACKENDS[backend] if path.startswith("overlap") else None)
    assert trial._agent_sandbox_stopped is (path == "overlap")
    assert sandbox.stopped is (path == "overlap")
    assert trial._planes.quiesced == (1 if integrity == "strict" else 0)

    got = await _at_the_trials_end(trial, sandbox, caplog)

    assert got == {**_expected(path, integrity), "warned": False, "raised": None}
    assert trial._connection_needs_quiescence is False
    if path == "overlap":
        # Nothing at all was exec'd into the stopped sandbox.
        assert sandbox.commands == []


@pytest.mark.parametrize("integrity", MODES)
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("backend", sorted(BACKENDS))
async def test_a_root_agents_trial_end_only_kills_in_a_sandbox_still_up(
    tmp_path, monkeypatch, caplog, backend, path, integrity
):
    """A root agent: no process stop at any point, its login removed for
    good at its turn's end, and at the trial's end only the agent's pkill,
    in a sandbox that is still up."""
    _verifier_prepare(monkeypatch, path)
    sandbox = Sandbox(backend, stop_fails=path == "overlap-stop-failed")
    trial = _trial(
        tmp_path, backend, integrity, sandbox, sandbox_user=None, login=ROOT_LOGIN
    )

    await _through_the_verifier(trial, path)

    assert trial._agent_sandbox_stopped is (path == "overlap")
    assert trial._planes.quiesced == 0
    assert trial._credential_ledger == []

    got = await _at_the_trials_end(trial, sandbox, caplog)

    assert got == {
        "quiesce": False,
        "pkill": path != "overlap",
        "removal": False,
        "ledger": [],
        "warned": False,
        "raised": None,
    }
    assert trial._planes.quiesced == 0


@pytest.mark.parametrize("integrity", ["audit", "off"])
@pytest.mark.parametrize("backend", sorted(BACKENDS))
async def test_a_muse_connection_has_nothing_pending_once_its_sandbox_stopped(
    tmp_path, monkeypatch, backend, integrity
):
    """A Muse connection's turn end stops its processes and removes its
    login in every mode, so the trial's end, after the stop before the
    verifier, has nothing to stop or remove (a failed Muse removal there
    would fail the trial)."""
    monkeypatch.delenv(VERIFIER_PREPARE_ENV, raising=False)
    sandbox = Sandbox(backend)
    trial = _trial(tmp_path, backend, integrity, sandbox)
    trial._credential_ledger = [MUSE_LOGIN]
    trial._muse_credentials_staged = True

    await _through_the_verifier(trial, "overlap")

    assert trial._planes.quiesced == 1
    assert trial._connection_needs_quiescence is False
    assert trial._credential_ledger == []
    assert trial._muse_credentials_staged is False
    assert trial._agent_sandbox_stopped is True
    sandbox.commands.clear()

    await trial._disconnect_at_trial_end()

    assert trial._planes.quiesced == 1
    assert sandbox.commands == []
