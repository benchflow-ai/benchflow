"""A trial that keeps services, on a sandbox whose /logs/verifier is the
host's own trial folder (``is_mounted``: local Docker's bind mount).

- the hardening's clear keeps the folder's owner, the host user, only
  there (``host_owned``); everywhere else it is root's, as #84 made it.
- an agent user with the uid BenchFlow runs as is refused before the
  agent starts. The release would make the folder root's for grading, and
  BenchFlow could then write nothing there until the kept services end.

Stand-ins only: nothing here runs a sandbox.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox.protocol import KeptServiceFreezerUnavailable
from benchflow.task import keep_services
from tests.test_preverify_order import _harden_fakes, _HardenSandbox, _task


class _Cleared(Exception):
    """Stops the hardening once the verifier folder's clear was issued."""


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [True, False, None])
async def test_the_clear_keeps_the_host_owner_only_on_a_mounted_folder(
    tmp_path, monkeypatch, mounted
):
    """``host_owned`` needs ``is_mounted`` to be True: with the folder in the
    sandbox (Daytona, remote Docker, Modal: False; a stand-in: missing) the
    clear makes it root's, as #84 made it (G69)."""
    _harden_fakes(monkeypatch, [])
    keep = SimpleNamespace(ports=(5000,))
    monkeypatch.setattr(keep_services, "trial_keep", lambda *a: keep)

    async def keeping(env, sandbox_user, ports, *, wait):
        return lockdown.KeepingKill(killed=0, kept=(), waited_sec=0.0)

    monkeypatch.setattr(lockdown, "_kill_sandbox_user_procs_keeping", keeping)
    monkeypatch.setattr(lockdown, "_record_kept", lambda *a, **k: None)
    issued: list[str] = []

    async def checked_exec(env, command, label, **kwargs):
        issued.append(command)
        if "/logs/verifier -mindepth 1 -exec rm -rf" in command:
            raise _Cleared
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(lockdown, "_checked_exec", checked_exec)
    env = _HardenSandbox()
    if mounted is not None:
        env.is_mounted = mounted
    with pytest.raises(_Cleared):
        await lockdown._harden_steps(
            env, _task(tmp_path), "agent", "/app", False, None, {}
        )
    clear = issued[-1]
    assert clear.endswith("chmod 777 /logs/verifier"), clear
    assert ("chown root:root /logs/verifier" in clear) is (mounted is not True), clear


def _identities(task, uid: int) -> None:
    agent = SimpleNamespace(uid=uid, gid=uid)
    verifier = SimpleNamespace(uid=0, gid=0)
    vars(task)["_benchflow_service_identities"] = (agent, verifier)


def _env(mounted):
    env = SimpleNamespace()
    if mounted is not None:
        env.is_mounted = mounted
    return env


def test_an_agent_user_with_the_host_uid_on_a_mounted_folder_is_refused(monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 1001)
    task = SimpleNamespace()
    _identities(task, 1001)
    with pytest.raises(KeptServiceFreezerUnavailable) as caught:
        keep_services.refuse_host_uid(task, _env(True))
    message = str(caught.value)
    assert "uid 1001" in message, message
    assert "the uid BenchFlow runs as on the host" in message, message
    assert "No agent code ran" in message, message
    assert "useradd -u" in message and "--sandbox remote-docker" in message, message


@pytest.mark.parametrize(
    ("mounted", "agent_uid", "host_uid"),
    [
        # The folder is the sandbox's own: BenchFlow downloads it.
        (False, 1001, 1001),
        (None, 1001, 1001),
        # A Mock's attribute is not True.
        ("mock", 1001, 1001),
        # Another uid: the folder keeps the host user's owner.
        (True, 1002, 1001),
        # BenchFlow as root writes a root folder.
        (True, 1001, 0),
    ],
)
def test_every_other_trial_that_keeps_services_runs(
    monkeypatch, mounted, agent_uid, host_uid
):
    monkeypatch.setattr(os, "getuid", lambda: host_uid)
    task = SimpleNamespace()
    _identities(task, agent_uid)
    env = MagicMock() if mounted == "mock" else _env(mounted)
    keep_services.refuse_host_uid(task, env)


def test_no_measured_identities_refuse_nothing(monkeypatch):
    """Before prepare_identities (or for a trial it skipped) there is no
    agent uid to compare; _needs_kept_freezer needs them first anyway."""
    monkeypatch.setattr(os, "getuid", lambda: 1001)
    keep_services.refuse_host_uid(SimpleNamespace(), _env(True))


def test_the_refusal_comes_before_the_freezer_probe():
    """Both run before solve.sh or the agent: the refusal first, right after
    the check that the trial keeps services (checked in the source)."""
    import inspect

    from benchflow.rollout import Rollout

    source = inspect.getsource(Rollout._prove_kept_freezer)
    needs = source.index("if not self._needs_kept_freezer():")
    refuse = source.index("keep_services.refuse_host_uid(self._task, self._env)")
    probe = source.index("probe_kept_service_freezer")
    assert needs < refuse < probe


def test_bench_explain_names_the_uid_and_the_remedy(monkeypatch):
    from benchflow import failures

    monkeypatch.setattr(os, "getuid", lambda: 1001)
    task = SimpleNamespace()
    _identities(task, 1001)
    with pytest.raises(KeptServiceFreezerUnavailable) as caught:
        keep_services.refuse_host_uid(task, _env(True))
    cause = failures.cause_of(
        {"rewards": None, "error": f"KeptServiceFreezerUnavailable: {caught.value}"}
    )
    assert cause is not None
    assert cause.key == "kept_service_freezer:host_uid", cause
    assert "useradd -u" in cause.next_step, cause
