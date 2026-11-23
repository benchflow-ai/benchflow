"""The link check's gate holds back only a reward test.sh wrote that would
otherwise be scored. An attempt that ended earlier (a crash, a wedge, a
breach, a setup failure, a tools refusal, a timeout) keeps its own error,
diagnostic and category whether the check completed or not, so main's retry
and recovery run from it; the reward's provenance is recorded either way.
Every attempt that ends with test.sh's reward, a retry's and a recovery's
included, waits on the check."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.scoring import classify_verifier_error
from benchflow.rollout import _verifier_recovery as recovery
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox import _verifier_links
from benchflow.task import RolloutPaths
from benchflow.task.verifier_errors import (
    NetworkPolicyBreachedError,
    VerifierSetupError,
    VerifierToolsRefusedError,
)
from tests.test_verifier_recovery import child, trial  # noqa: F401  (fixtures)

AGENT = "claude-agent-acp"
#: A non-root agent's test.sh reward with a check that did not complete: 0,
#: counted (the lead's 02:38Z Q6 ruling).
ZERO = {"reward": 0.0}
PLANTED = {
    "path": "/app/grade.log",
    "target": "/logs/verifier/reward.txt",
    "resolved": "/logs/verifier/reward.txt",
    "removed": True,
}


def _task():
    return SimpleNamespace(
        name="provenance-task",
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(timeout_sec=0.2, reward_range=None, env={})
        ),
    )


def _check(monkeypatch, *, ran: bool, before=(), after=()):
    """The link walks' records: *ran* for both, *before* the links removed
    before test.sh, *after* those found after grading."""
    calls = []

    def record(links, step):
        async def fake(env, *, timeout_sec=None, **_kwargs):
            calls.append(step)
            return {
                "ran": ran,
                "why": None if ran else "the walk failed: forced",
                "engine": "bfx",
                "links": [dict(row) for row in links],
            }

        return fake

    monkeypatch.setattr(_verifier_links, "remove_turn_links", record(before, "before"))
    monkeypatch.setattr(_verifier_links, "links_after_grading", record(after, "after"))
    return calls


def _env(probe: str = "0\n", code: int = 0):
    return SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(stdout=probe, stderr="", return_code=code)
        ),
        _benchflow_verifier_inputs={"flags": [], "changes": []},
    )


def _planes(verifier, harden=None):
    return SimpleNamespace(
        harden_before_verify=harden or AsyncMock(),
        quiesce_agent=AsyncMock(),
        verifier=lambda **_: verifier,
    )


async def _returns(value):
    return SimpleNamespace(rewards={"reward": value})


async def _crash():
    raise ConnectionResetError("connection reset")


async def _wedge():
    await asyncio.sleep(3600)


async def _breach_at_test_end():
    raise NetworkPolicyBreachedError("the ipv6 rule is gone", record={"ipv6": "lost"})


def _ending(kind):
    """(env, planes, recovery_eligible) that end the attempt as *kind*."""
    harden = None
    verify = _returns
    env = _env()
    eligible = False
    receipt = None
    if kind == "crash":
        verify = _crash
    elif kind == "wedge":
        # No start receipt and a recovery contract: the wedge is left to
        # recovery with BenchFlow's own 0.
        verify, eligible = _wedge, True
        env = _env(code=1)
        receipt = ("/logs/verifier/s", "main")  # the start receipt is looked for
    elif kind == "timeout":
        verify = _wedge
        env = _env(probe="4242\n")  # output was produced: scored 0, timed out
    elif kind == "breach in hardening":
        harden = AsyncMock(
            side_effect=NetworkPolicyBreachedError(
                "the ipv6 rule is gone", record={"ipv6": "lost"}
            )
        )
    elif kind == "breach at test end":
        verify = _breach_at_test_end
    elif kind == "setup failure":
        harden = AsyncMock(side_effect=VerifierSetupError("chmod of test.sh failed"))
    elif kind == "tools refusal":
        harden = AsyncMock(
            side_effect=VerifierToolsRefusedError("pytest only in /home/agent/bin")
        )
    else:
        raise AssertionError(kind)

    async def run_verify():
        return await verify(1.0) if verify is _returns else await verify()

    verifier = SimpleNamespace(
        outputs_closed=True, verify=run_verify, execution_receipt=receipt
    )
    return env, _planes(verifier, harden), eligible


async def _attempt(env, planes, tmp_path, name, *, eligible=False, user="agent"):
    return await _verify_rollout(
        env,
        _task(),
        RolloutPaths(rollout_dir=tmp_path / name),
        {},
        planes,
        sandbox_user=user,
        integrity="audit",
        recovery_eligible=eligible,
        agent=AGENT,
    )


IN_HARDENING = ("breach in hardening", "setup failure", "tools refusal")
ENDINGS = (
    "crash",
    "wedge",
    "timeout",
    "breach in hardening",
    "breach at test end",
    "setup failure",
    "tools refusal",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ENDINGS)
async def test_an_earlier_ending_keeps_its_own_error_and_category(
    kind, tmp_path, monkeypatch
):
    """The same attempt with the link check complete and with it failed: the
    ending, its error, diagnostic and category are the same; the check's
    reason never replaces them; the provenance is recorded for both."""
    seen = []
    for ran in (True, False):
        _check(monkeypatch, ran=ran)
        env, planes, eligible = _ending(kind)
        rewards, error, diagnostic = await _attempt(
            env, planes, tmp_path, f"{ran}", eligible=eligible
        )
        provenance = env._benchflow_verifier_inputs.get("provenance")
        assert provenance is not None, kind
        # An ending in hardening comes before the walk before test.sh: that
        # check was not made, whatever the walks would have said.
        assert provenance["links_checked"] is (ran and kind not in IN_HARDENING)
        assert provenance["checks"]
        assert provenance["verdict"] != "verified" or ran
        seen.append(
            (
                rewards,
                error,
                type(diagnostic).__name__,
                classify_verifier_error(error),
            )
        )
    assert seen[0] == seen[1], (kind, seen)
    rewards, error, _, category = seen[1]
    assert error is None or "link check" not in error
    if kind == "wedge":  # BenchFlow's own 0, no diagnostic: no start receipt
        assert (rewards, error, seen[1][2]) == ({"reward": 0.0}, None, "NoneType")
    elif kind == "timeout":
        assert (rewards, seen[1][2]) == ({"reward": 0.0}, "VerifierTimeoutDiagnostic")
    else:
        assert rewards is None and error and category


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["agent", None])
async def test_test_sh_reward_waits_on_the_check(user, tmp_path, monkeypatch):
    """test.sh's own reward: a non-root agent's is unscored when the check
    did not complete, a root agent's is scored as written (unverifiable)."""
    _check(monkeypatch, ran=False)
    env, planes, _ = _ending("crash")
    planes.verifier = lambda **_: SimpleNamespace(
        outputs_closed=True, verify=lambda: _returns(1.0)
    )
    rewards, error, _ = await _attempt(env, planes, tmp_path, "a", user=user)
    if user is None:
        assert (rewards, error) == ({"reward": 1.0}, None)
    else:
        assert (rewards, error) == (ZERO, None)


@pytest.mark.asyncio
async def test_crash_then_a_retry_with_a_planted_link(tmp_path, monkeypatch, caplog):
    """Attempt 1 crashes and keeps the crash; attempt 2, on the same sandbox,
    removes and flags the planted link and scores test.sh's own value."""
    env = _env()
    _check(monkeypatch, ran=True)
    planes = _planes(SimpleNamespace(outputs_closed=True, verify=_crash))
    first = await _attempt(env, planes, tmp_path, "1")
    assert first[0] is None
    assert first[1] == "verifier crashed: ConnectionResetError: connection reset"
    calls = _check(monkeypatch, ran=True, before=[PLANTED])
    planes = _planes(
        SimpleNamespace(outputs_closed=True, verify=lambda: _returns(0.25))
    )
    rewards, error, _ = await _attempt(env, planes, tmp_path, "2")
    assert calls == ["before", "after"]
    assert (rewards, error) == ({"reward": 0.25}, None)
    flags = [
        f
        for f in env._benchflow_verifier_inputs["flags"]
        if f.get("category") == _verifier_links.CATEGORY
    ]
    assert [f["path"] for f in flags] == ["/app/grade.log"]
    assert "removed before test.sh" in caplog.text
    provenance = env._benchflow_verifier_inputs["provenance"]
    assert provenance["links_checked"] is True
    assert provenance["links_removed_before_test"]


@pytest.mark.asyncio
async def test_crash_then_a_retry_whose_check_cannot_complete(tmp_path, monkeypatch):
    """Attempt 1 crashes; attempt 2's test.sh reward, with a link check that
    could not complete, is unscored for a non-root agent."""
    env = _env()
    _check(monkeypatch, ran=True)
    first = await _attempt(
        env, _planes(SimpleNamespace(outputs_closed=True, verify=_crash)), tmp_path, "1"
    )
    assert first[1].startswith("verifier crashed: ")
    _check(monkeypatch, ran=False)
    planes = _planes(SimpleNamespace(outputs_closed=True, verify=lambda: _returns(1.0)))
    rewards, error, diagnostic = await _attempt(env, planes, tmp_path, "2")
    assert (rewards, error, diagnostic) == (ZERO, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ran", "expected"), [(True, ({"reward": 1.0}, None)), (False, (ZERO, None))]
)
async def test_in_place_retry_after_a_wedge_waits_on_the_check(
    ran, expected, tmp_path, monkeypatch
):
    """The in-place retry after a wedge (no recovery contract) scores
    test.sh's reward only when the check completed."""
    _check(monkeypatch, ran=ran)
    attempts = []

    async def verify():
        attempts.append(1)
        if len(attempts) == 1:
            await asyncio.sleep(3600)
        return SimpleNamespace(rewards={"reward": 1.0})

    planes = _planes(SimpleNamespace(outputs_closed=True, verify=verify))
    rewards, error, _ = await _attempt(_env(), planes, tmp_path, "w")
    assert len(attempts) == 2
    assert (rewards, error) == expected


@pytest.mark.asyncio
async def test_recovery_verify_runs_the_checks_for_the_trial_agent(
    trial,  # noqa: F811
    child,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """Recovery's verify gets the trial's agent: the link it left in the
    workspace is removed and flagged before test.sh, and test.sh's reward is
    scored only with the check complete."""
    calls = _check(monkeypatch, ran=True, before=[PLANTED])
    child._env._benchflow_verifier_inputs = {"flags": [], "changes": []}
    real = _verify_rollout
    seen = {}

    async def verify(env, task, paths, timing, planes, **kwargs):
        seen.update(kwargs)
        return await real(
            env,
            _task(),
            paths,
            timing,
            _planes(SimpleNamespace(outputs_closed=True, verify=lambda: _returns(0.5))),
            **kwargs,
        )

    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    rewards, error = await recovery.recover_verifier(trial)
    assert seen["agent"] == trial._config.primary_agent
    assert calls == ["before", "after"]
    assert (rewards, error) == ({"reward": 0.5}, None)
    flags = child._env._benchflow_verifier_inputs["flags"]
    assert [f["path"] for f in flags] == ["/app/grade.log"]


@pytest.mark.asyncio
async def test_recovery_verify_with_an_incomplete_check_is_not_scored(
    trial,  # noqa: F811
    child,  # noqa: F811
    monkeypatch,
):
    """Recovery's test.sh reward with a link check that could not complete
    is not scored for a non-root agent."""
    _check(monkeypatch, ran=False)
    child._env._benchflow_verifier_inputs = {"flags": [], "changes": []}
    real = _verify_rollout

    async def verify(env, task, paths, timing, planes, **kwargs):
        return await real(
            env,
            _task(),
            paths,
            timing,
            _planes(SimpleNamespace(outputs_closed=True, verify=lambda: _returns(1.0))),
            **kwargs,
        )

    monkeypatch.setattr("benchflow.rollout._setup._verify_rollout", verify)
    rewards, error = await recovery.recover_verifier(trial)
    assert (rewards, error) == (ZERO, None)
