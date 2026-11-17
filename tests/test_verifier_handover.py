"""The oracle's turn hands main to the verifier's network with no moment
unheld (RCI on 011d84c9f), and an internal network's every-process
proof (RCI's F4)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown, rule_ledger
from benchflow.sandbox import verifier_network as vn
from benchflow.task.config import TaskConfig


class Env:
    """A Docker sandbox as the handover sees it: its helper's answers."""

    rules_by_helper = True

    def __init__(self) -> None:
        self.helper: list[str] = []

    async def metadata_rules_install(self, command):  # the type has the helper
        raise AssertionError("answered by the patched _via_helper")

    async def exec(self, command, *args, **kwargs):
        return SimpleNamespace(stdout="", stderr="", return_code=0)


@pytest.fixture
def helper(monkeypatch):
    def install(env):
        async def via(env_, command):
            env_.helper.append(command)
            if "handover status" in command:
                return "BF verifier-network handover status=0 v6=1\n"
            if "released status" in command:
                return "BF verifier-network released status=0\n"
            if "BF rule-check" in command:
                return f"BF rule-check status=0 missing={env_.missing}\n"
            return ""

        monkeypatch.setattr(vn, "_via_helper", via)

    return install


@pytest.mark.asyncio
async def test_the_hold_goes_in_on_top_is_in_the_ledger_and_comes_out(helper):
    env = Env()
    helper(env)
    env._benchflow_oracle_egress = (object(), None)
    assert await vn.hold_for_handover(env) is True
    assert vn.HANDOVER_HOLD in env.helper[-1] and "-I OUTPUT 1" in env.helper[-1]
    # The TCP half goes in last, so it sits on top: a reset, never a retry.
    assert env.helper[-1].rindex("rt=") < env.helper[-1].rindex("$rt")
    assert "--reject-with tcp-reset" in vn.HANDOVER_HOLD_TCP
    (entry,) = rule_ledger.expected(env, "handover")
    assert entry.v4 == list(vn.HANDOVER_HOLDS) and entry.v6 == list(vn.HANDOVER_HOLDS)
    await vn.release_handover(env)
    assert "-D OUTPUT" in env.helper[-1]
    assert rule_ledger.expected(env, "handover") == []
    assert getattr(env, "_benchflow_handover_hold", None) is None


@pytest.mark.asyncio
async def test_no_hold_without_rules_to_hand_over(helper):
    env = Env()
    helper(env)
    assert await vn.hold_for_handover(env) is False
    assert env.helper == []


def _task(toml: str):
    return SimpleNamespace(
        config=TaskConfig.model_validate_toml('version = "1.0"\n' + toml)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("toml", "armed"),
    [
        (
            '[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
            True,
        ),
        (
            '[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n[verifier]\nnetwork_mode = "public"\n',
            False,
        ),
    ],
    ids=["verifier-allowlist", "verifier-public"],
)
async def test_the_old_rules_come_out_only_under_the_hold(monkeypatch, toml, armed):
    events: list[str] = []
    env = Env()

    async def hold(env_):
        events.append("hold in")
        env_._benchflow_handover_hold = True
        return True

    async def release(env_):
        if getattr(env_, "_benchflow_handover_hold", None):
            events.append("hold out")
            env_._benchflow_handover_hold = None

    async def offline_out(env_):
        events.append("offline rules out")

    async def oracle_out(env_):
        events.append("oracle rules out")

    async def arm(env_, task, mode, *, where):
        events.append("verifier rules in")
        await release(env_)
        return vn.VerifierNetwork(mode, "held", where=where)

    async def refused(env_):
        return None

    monkeypatch.setattr(vn, "hold_for_handover", hold)
    monkeypatch.setattr(vn, "release_handover", release)
    monkeypatch.setattr(vn, "remove_offline_agent_phase", offline_out)
    monkeypatch.setattr(lockdown, "_hand_over_oracle_egress", oracle_out)
    monkeypatch.setattr(vn, "arm_verifier_network", arm)
    monkeypatch.setattr(vn, "main_metadata_refused", refused)
    await lockdown._arm_verifier_network(env, _task(toml), {})
    tail = ["verifier rules in", "hold out"] if armed else ["hold out"]
    assert events == ["hold in", "offline rules out", "oracle rules out", *tail]


@pytest.mark.asyncio
@pytest.mark.parametrize(("missing", "held"), [(0, True), (1, False)])
async def test_an_internal_network_is_proven_by_its_rules_present(
    monkeypatch, helper, missing, held
):
    """RCI's F4: on an internal network root's TEST-NET connect is refused by
    the kernel (no route), before any rule can count. The proof there: every
    rule just added present, read from the helper, with no route needed."""
    env = Env()
    env.missing = missing
    helper(env)
    added = (["-o lo -j ACCEPT", "-j REJECT"], ["-j REJECT"], [])

    async def install(env_, v4, v6, nat):
        return "BF verifier-network status=0\n", True, added

    async def count(env_, via_helper):
        return "BF verifier-network refused 0 status=0\n"

    async def connect(env_, out):
        return "BF verifier-proof connect=unreachable\n", True

    monkeypatch.setattr(vn, "_install_recorded", install)
    monkeypatch.setattr(vn, "_count_root", count)
    monkeypatch.setattr(vn, "_shell_or_helper_connect", connect)
    if held:
        assert await vn.arm_policy_agent_phase(env, "allowlist") == added
    else:
        with pytest.raises(RuntimeError, match="not all present"):
            await vn.arm_policy_agent_phase(env, "allowlist")
