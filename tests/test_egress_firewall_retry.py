"""Guards benchflow-ai/benchflow#1151: the egress firewall was one 120 s exec
that installed iptables and added the rules, so a slow package mirror or a
loaded sandbox failed the launch before the first prompt, with an error that
named no step. A timed-out exec now gets iptables installed on its own with
a longer limit and runs again; the rules are idempotent."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox.lockdown import enforce_sandbox_uid_egress
from benchflow.sandbox.protocol import ExecTimeoutError


def _env(*outcomes):
    """A sandbox whose execs answer *outcomes* in order (an exception is raised)."""
    calls: list[tuple[str, dict]] = []
    queue = list(outcomes)

    async def exec_(command, **kwargs):
        calls.append((command, kwargs))
        outcome = queue.pop(0) if queue else 0
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(return_code=outcome, stdout="", stderr="")

    return SimpleNamespace(exec=exec_), calls


@pytest.mark.parametrize(
    "timed_out",
    [
        ExecTimeoutError(120),
        # Docker's exec channel says so with a plain RuntimeError.
        RuntimeError("Command timed out after 120 seconds"),
        # Daytona runs the exec under `timeout`, which ends it with 124.
        124,
    ],
    ids=["exec-timeout", "docker-channel", "daytona-timeout-124"],
)
async def test_a_timed_out_firewall_installs_iptables_alone_and_runs_again(timed_out):
    env, calls = _env(timed_out, 0, 0)

    await enforce_sandbox_uid_egress(env, "agent")

    assert [kwargs["timeout_sec"] for _, kwargs in calls] == [120, 600, 120]
    assert all(kwargs["user"] == "root" for _, kwargs in calls)
    install = calls[1][0]
    assert "DPkg::Lock::Timeout" in install and "iptables -C" not in install
    assert calls[2][0] == calls[0][0]


async def test_a_firewall_that_keeps_timing_out_names_its_step():
    env, calls = _env(*[ExecTimeoutError(120), 0] * 2, ExecTimeoutError(120))

    with pytest.raises(RuntimeError, match="egress firewall: its exec timed out 3"):
        await enforce_sandbox_uid_egress(env, "agent")
    assert len(calls) == 5


async def test_a_failed_iptables_install_names_its_step():
    env, _ = _env(ExecTimeoutError(120), 86)

    with pytest.raises(RuntimeError, match="egress firewall: installing iptables"):
        await enforce_sandbox_uid_egress(env, "agent")


async def test_a_timed_out_install_on_its_own_names_its_step():
    env, _ = _env(ExecTimeoutError(120), 124)

    with pytest.raises(RuntimeError, match="installing iptables timed out"):
        await enforce_sandbox_uid_egress(env, "agent")


async def test_another_failure_is_not_retried():
    env, calls = _env(RuntimeError("dummy: the exec channel broke"))

    with pytest.raises(RuntimeError, match="the exec channel broke"):
        await enforce_sandbox_uid_egress(env, "agent")
    assert len(calls) == 1


def test_the_install_on_its_own_is_valid_shell():
    command = lockdown._IPTABLES_INSTALL_RETRY_CMD
    assert command.count("DPkg::Lock::Timeout=300") == 2
    # A timed-out apt can leave dpkg interrupted mid-unpack.
    assert "dpkg --configure -a" in command
    subprocess.run(["bash", "-n", "-c", command], check=True)
