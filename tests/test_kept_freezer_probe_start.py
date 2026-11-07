"""A trial that keeps services tries the freezer before any agent code
(K1-R9-1), decided from the identities setup measured.

At 16187b3e the only freezer check of a Docker or Daytona Compose trial ran
in the kill after the agent's turn, so on a sandbox that cannot hold kept
services (a cgroup v1 host, an Engine without writable-cgroups) the whole
turn ran and then the trial failed. install_agent now runs the probe
(probe_kept_service_freezer, see tests/test_kept_service_freezer.py) for a
trial that keeps services only, and times it (``kept_freezer_probe``).

Whether the trial keeps anything is DT2's measured decision
(``keep_services.prepare_identities``, then ``kept_by_identity``), which
raises when asked before the identities exist. So the probe runs right after
the measurement: after the agent's install and its account, before its
launch; for the oracle, before solve.sh. Which account the oracle runs as
(its own su, digits included) is DT2's and tested in
tests/test_keep_services_matrix.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils import timing_spans
from benchflow.task import keep_services
from tests._receipt_fakes import inner_command, receipt_aware
from tests.test_timing_spans import FakeClock, _install_trial, _quiet_planes

KEEP = keep_services.KeptServices(
    services=(keep_services.KeepService(port=5000),),
    source=keep_services.SOURCE_TASK,
)


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(timing_spans, "_clock", fake)
    return fake


def _measured(monkeypatch, kept: bool) -> list[str]:
    """The trial's measured decision: it keeps services (``kept``) or not.

    As DT2's ``kept_by_identity``, asking before ``prepare_identities``
    raises. Returns the order of the calls."""
    order: list[str] = []
    measured = {"done": False}

    async def prepare(env, task, sandbox_user, *, inherit=False):
        order.append("prepare_identities")
        measured["done"] = True

    def decided(task):
        if not measured["done"]:
            raise RuntimeError(
                "service retention: resolve the agent and verifier UID/GID "
                "before starting the agent (prepare_identities)"
            )
        order.append("decided")
        return KEEP if kept else None

    monkeypatch.setattr(keep_services, "prepare_identities", prepare)
    monkeypatch.setattr(keep_services, "kept_by_identity", decided)
    monkeypatch.setattr(
        keep_services,
        "trial_keep",
        lambda task, user: decided(task) if user else None,
    )
    return order


def _sandbox(freezer: bool):
    """An exec whose probe passes (cgroup v2) or fails as on cgroup v1."""

    async def run(command, **kwargs):
        if "BF freezer ok" in command:
            if freezer:
                return SimpleNamespace(exit_code=0, stdout="BF freezer ok\n", stderr="")
            return SimpleNamespace(
                exit_code=2,
                stdout="",
                stderr="Cannot hold the kept services in a frozen cgroup: "
                "/sys/fs/cgroup is not cgroup v2\n",
            )
        return SimpleNamespace(exit_code=0, stdout="/workspace\n", stderr="")

    return AsyncMock(side_effect=receipt_aware(run))


def _probed(trial):
    return any(
        "BF freezer ok" in inner_command(c.args[0])
        for c in trial._env.exec.await_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude-agent-acp", "oracle"])
async def test_the_decision_waits_for_the_measured_identities(
    tmp_path, monkeypatch, clock, agent
):
    """On 8a67dbda the probe decision ran first in install_agent,
    before prepare_identities, and DT2's check refused every trial of a task
    that keeps services."""
    order = _measured(monkeypatch, kept=True)
    trial = _install_trial(tmp_path, agent)
    _quiet_planes(monkeypatch, trial, clock)
    trial._env.exec = _sandbox(freezer=True)
    await trial.install_agent()
    assert order[:2] == ["prepare_identities", "decided"]
    assert _probed(trial)


@pytest.mark.asyncio
async def test_a_cgroup_v1_sandbox_is_refused_before_the_agent_launches(
    tmp_path, monkeypatch, clock
):
    """A trial that keeps services on a sandbox whose probe fails (here the
    probe's own cgroup v1 refusal) ends in install_agent once its account
    is measured, before its skills, lockdown or launch."""
    _measured(monkeypatch, kept=True)
    trial = _install_trial(tmp_path, "claude-agent-acp")
    _quiet_planes(monkeypatch, trial, clock)
    trial._env.exec = _sandbox(freezer=False)
    with pytest.raises(RuntimeError, match="not cgroup v2"):
        await trial.install_agent()
    trial._planes.setup_sandbox_user.assert_awaited_once()
    trial._planes.deploy_skills.assert_not_awaited()
    trial._planes.lockdown_paths.assert_not_awaited()
    assert "kept_freezer_probe" in trial.timing


@pytest.mark.asyncio
async def test_a_sandbox_that_holds_goes_on_and_times_the_probe(
    tmp_path, monkeypatch, clock
):
    _measured(monkeypatch, kept=True)
    trial = _install_trial(tmp_path, "claude-agent-acp")
    install = _quiet_planes(monkeypatch, trial, clock)
    trial._env.exec = _sandbox(freezer=True)
    await trial.install_agent()
    install.assert_awaited_once()
    assert set(trial.timing) == {"kept_freezer_probe", "install_agent"}


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude-agent-acp", "oracle", "nop"])
async def test_a_trial_that_keeps_nothing_runs_no_probe(
    tmp_path, monkeypatch, clock, agent
):
    """Measured as keeping nothing (a root oracle, a non-root verifier), and
    nop whatever it measures: no probe, no refusal on cgroup v1, as at
    16187b3e (K1-R10-1)."""
    _measured(monkeypatch, kept=agent == "nop")
    trial = _install_trial(tmp_path, agent)
    _quiet_planes(monkeypatch, trial, clock)
    trial._env.exec = _sandbox(freezer=False)
    await trial.install_agent()
    assert not _probed(trial)
    assert "kept_freezer_probe" not in trial.timing


@pytest.mark.asyncio
@pytest.mark.parametrize("freezer", [True, False], ids=["cgroup-v2", "cgroup-v1"])
async def test_a_non_root_oracle_is_probed_before_solve(
    tmp_path, monkeypatch, clock, freezer
):
    """An oracle measured as non-root beside a root verifier is probed:
    refused on cgroup v1 before the verifier's baseline or solve.sh, on its
    way on cgroup v2."""
    _measured(monkeypatch, kept=True)
    trial = _install_trial(tmp_path, "oracle")
    _quiet_planes(monkeypatch, trial, clock)
    trial._env.exec = _sandbox(freezer=freezer)
    if freezer:
        await trial.install_agent()
    else:
        with pytest.raises(RuntimeError, match="not cgroup v2"):
            await trial.install_agent()
        trial._planes.snapshot_build_config.assert_not_awaited()
    assert _probed(trial)
    assert "kept_freezer_probe" in trial.timing


# The refusals of a sandbox that cannot hold kept services moved
# from the plan gate and the sandbox's start to here, where the measured
# identities say whether the trial keeps anything.


@pytest.mark.asyncio
@pytest.mark.parametrize("kept", [True, False], ids=["kept", "keeps-nothing"])
async def test_a_sandbox_with_no_freezer_refuses_only_a_measured_kept_agent(
    tmp_path, monkeypatch, clock, kept
):
    """Modal gives root no cgroup of its own: an agent that keeps services
    is refused with what to run instead, before its launch and without a
    probe; one whose verifier is measured as non-root keeps nothing and
    goes on (DT2-R12-1)."""
    from benchflow.sandbox.protocol import KeptServiceFreezerUnavailable

    _measured(monkeypatch, kept=kept)
    trial = _install_trial(tmp_path, "claude-agent-acp", environment="modal")
    _quiet_planes(monkeypatch, trial, clock)
    trial._env.exec = _sandbox(freezer=True)
    if kept:
        with pytest.raises(KeptServiceFreezerUnavailable, match="--sandbox docker"):
            await trial.install_agent()
        trial._planes.deploy_skills.assert_not_awaited()
    else:
        await trial.install_agent()
    assert not _probed(trial)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude-agent-acp", "oracle"])
async def test_an_engine_without_the_freezer_refuses_a_measured_kept_agent(
    tmp_path, monkeypatch, clock, agent
):
    """The start recorded the Engine's reason and started without the
    freezer; a non-root agent that keeps services is refused with it, an
    oracle meets the probe (which here, as without writable-cgroups, fails)."""
    from benchflow.sandbox.protocol import KeptServiceFreezerUnavailable

    _measured(monkeypatch, kept=True)
    trial = _install_trial(tmp_path, agent)
    _quiet_planes(monkeypatch, trial, clock)
    trial._env.exec = _sandbox(freezer=False)
    trial._env.kept_service_freezer_refusal = (
        "Docker Engine 27.5.1 predates writable-cgroups (28.0)"
    )
    if agent == "oracle":
        with pytest.raises(RuntimeError, match="not cgroup v2"):
            await trial.install_agent()
        assert _probed(trial)
    else:
        with pytest.raises(KeptServiceFreezerUnavailable, match=r"27\.5\.1") as caught:
            await trial.install_agent()
        assert "No agent code ran" in str(caught.value)
        assert not _probed(trial)
