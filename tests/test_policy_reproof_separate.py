"""A separate verifier's agent sandbox, proven again before the transfer.

A shared sandbox's hardening proves the rules of sandbox start and of the
agent's turn again at the end of the turn. A separate verifier sandbox hardens
only itself, so a rule removed in the agent's sandbox during its turn passed to
a positive reward. The rules are now proven again once what crosses is captured
and while the agent's sandbox still runs: in the overlap order before the
sandbox is released, in the serial order when the transfer starts. A breach
ends the trial "network policy breached" with nothing packed and no verifier
sandbox used. The re-proof is lane C's (benchflow.sandbox.policy_reproof),
run here against the stub iptables of tests/test_policy_reproof.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchflow._utils import timing_spans
from benchflow._utils.scoring import classify_verifier_error
from benchflow.rollout import _separate_verifier as sv
from benchflow.rollout._separate_verifier import run_separate_verifier
from benchflow.sandbox import policy_reproof as pr
from tests.test_policy_reproof import Namespace, _start
from tests.test_separate_verifier import (
    FakeVerifierSandbox,
    _agent_trial,
    _fake_rollout,
    _task_dir,
)
from tests.test_sv_overlap import RECORD, FakeAgent, _scenario, _trial


class _Rules:
    """The agent sandbox's network namespace as BenchFlow's helper sees it:
    the rules sandbox start put in, recorded in the ledger."""

    def __init__(self, tmp_path: Path) -> None:
        self.ns = Namespace(tmp_path, "agent-main")
        self.start = _start()
        self.ns.add(self.start, packets=1)
        self.ledger = [self.start]
        self.runs: list[str] = []

    def remove_start_rule(self) -> None:
        """What a process holding NET_ADMIN in the namespace does mid-turn."""
        self.ns.ipt("-w", "-D", "OUTPUT", *self.start.v4[0].split())


class _RuledEnv:
    """An agent sandbox with a helper: only the re-proof's calls."""

    is_mounted = False

    def __init__(self, rules: _Rules) -> None:
        self.rules = rules
        self._benchflow_rule_ledger = rules.ledger

    async def metadata_rules_install(self, command: str) -> tuple[Any, str]:
        self.rules.runs.append(command)
        return SimpleNamespace(stdout=self.rules.ns.sh(command)), ""


class _RuledAgent(FakeAgent):
    """test_sv_overlap's agent sandbox, with the helper and the ledger."""

    def __init__(self, sc: Any, rules: _Rules) -> None:
        super().__init__(sc)
        self.rules = rules
        self._benchflow_rule_ledger = rules.ledger

    async def metadata_rules_install(self, command: str) -> tuple[Any, str]:
        # cleanup() reads main's metadata counters through the helper too.
        self.sc.log.append(
            "agent.reprove" if "BF rule-entry" in command else "agent.helper"
        )
        return SimpleNamespace(stdout=self.rules.ns.sh(command)), ""


# --- the serial order: the transfer's start ---------------------------------------


async def _serial(tmp_path: Path, rules: _Rules) -> tuple[Any, Any, list, list]:
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    rollout._env = _RuledEnv(rules)
    # What the agent's run can leave in the trial's verifier/ (on Docker, the
    # agent sandbox's live /logs/verifier mount).
    (trial / "verifier" / "reward.txt").write_text("1\n")
    box = FakeVerifierSandbox(tmp_path / "verifier-box")
    created: list[Any] = []
    verified: list[Any] = []

    def create(vtask, context_path, vpaths):
        created.append(vtask)
        return box

    async def verify(env, vtask, vpaths, timing, *, workspace, agent_paths):
        verified.append(env)
        vpaths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (vpaths.verifier_dir / "reward.txt").write_text("1\n")
        return {"reward": 1.0}, None

    outcome = await run_separate_verifier(
        rollout, create_environment=create, verify=verify
    )
    return rollout, outcome, created, verified


def _record(trial: Path) -> dict[str, Any]:
    return json.loads((trial / RECORD).read_text())


@pytest.mark.asyncio
async def test_a_rule_removed_in_the_agent_s_turn_ends_the_trial_before_the_transfer(
    tmp_path, monkeypatch
):
    rules = _Rules(tmp_path)
    rules.remove_start_rule()
    packed: list[Any] = []
    monkeypatch.setattr(
        sv, "build_transfer_payload", lambda *a, **k: packed.append(a) or {}
    )

    rollout, (rewards, error), created, verified = await _serial(tmp_path, rules)

    assert rewards is None
    assert error.startswith(pr.NETWORK_POLICY_BREACHED)
    assert classify_verifier_error(error) == "network_policy_breached"
    assert "main (start): 1 of" in error and pr.END_OF_AGENT in error
    # Nothing crossed and no verifier sandbox ran.
    assert packed == [] and created == [] and verified == []
    trial = rollout._rollout_dir
    record = _record(trial)
    assert record["status"] == "network_policy_breached"
    assert record["error"] == error
    policy = record["agent_sandbox_policy"]
    assert policy["state"] == "breached" and policy["at"] == pr.END_OF_AGENT
    # When the rule was last proven present: its proof at sandbox start.
    assert policy["last_proven_present"] == rules.start.proven_at
    # The trial's verifier/ holds the breach's record and nothing the agent's
    # run left there (its planted reward.txt was moved aside).
    written = json.loads((trial / "verifier" / pr.NETWORK_POLICY_RECORD).read_text())
    assert written["state"] == "breached"
    assert written["entries"][0]["missing"] == [f"v4: {rules.start.v4[0]}"]
    assert not (trial / "verifier" / "reward.txt").exists()
    assert timing_spans.AGENT_SANDBOX_REPROOF in rollout._timing


@pytest.mark.asyncio
async def test_a_clean_agent_turn_keeps_the_verifier_sandbox_s_reward(tmp_path):
    rules = _Rules(tmp_path)

    rollout, (rewards, error), created, verified = await _serial(tmp_path, rules)

    assert (rewards, error) == ({"reward": 1.0}, None)
    assert len(created) == 1 and len(verified) == 1
    record = _record(rollout._rollout_dir)
    assert record["status"] == "complete"
    policy = record["agent_sandbox_policy"]
    assert policy["state"] == "held"
    assert policy["rules"] == len(rules.start.v4) + len(rules.start.v6)
    # One helper run in the agent's namespace, before the transfer.
    assert len(rules.runs) == 1
    assert timing_spans.AGENT_SANDBOX_REPROOF in record["timing"]


@pytest.mark.asyncio
async def test_counters_zeroed_in_the_agent_s_turn_are_a_breach(tmp_path):
    """main's start rule was proven by a refused, counted connect
    (service-network.json): its counter back to 0 is a breach."""
    rules = _Rules(tmp_path)
    rules.ns.ipt("-w", "-Z", "OUTPUT")
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    rollout._env = _RuledEnv(rules)
    rollout._env._main_cover_record = {"proof": {"metadata_counted": 1}}

    rewards, error = await run_separate_verifier(rollout)

    assert rewards is None and "counters went back" in error
    assert _record(trial)["agent_sandbox_policy"]["entries"][0]["state"] == (
        "counters back"
    )


@pytest.mark.asyncio
async def test_a_sandbox_with_no_ledger_is_not_proven(tmp_path):
    """A provider's agent sandbox (no helper, no ledger): nothing to prove,
    no record, the verifier runs as before."""
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    rollout._env = SimpleNamespace(is_mounted=False)
    box = FakeVerifierSandbox(tmp_path / "verifier-box")

    async def verify(env, vtask, vpaths, timing, *, workspace, agent_paths):
        return {"reward": 1.0}, None

    rewards, error = await run_separate_verifier(
        rollout, create_environment=lambda *a: box, verify=verify
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    assert "agent_sandbox_policy" not in _record(trial)


@pytest.mark.asyncio
async def test_a_regrade_has_no_agent_sandbox_to_prove(tmp_path):
    """eval_regrade marks the agent's sandbox gone before finish(): a stored
    trial is not proven again, whatever the rollout's sandbox object holds."""
    rules = _Rules(tmp_path)
    rules.remove_start_rule()
    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    rollout._env = _RuledEnv(rules)
    box = FakeVerifierSandbox(tmp_path / "verifier-box")

    async def verify(env, vtask, vpaths, timing, *, workspace, agent_paths):
        return {"reward": 1.0}, None

    run = sv.VerifierSandboxRun(
        rollout, order=sv.SERIAL, create_environment=lambda *a: box
    )
    run.agent_stopped(None)
    assert await run.finish(verify) == ({"reward": 1.0}, None)
    assert rules.runs == []


# --- the overlap order: before the agent's sandbox is released ------------------


@pytest.mark.parametrize("environment", ["docker", "daytona"])
async def test_the_overlap_order_proves_the_agent_sandbox_before_its_release(
    tmp_path, monkeypatch, environment
):
    rules = _Rules(tmp_path)
    sc = await _scenario(tmp_path, monkeypatch, environment=environment)
    sc.rollout._env = _RuledAgent(sc, rules)

    rewards = await _trial(sc)

    assert rewards == {"reward": 1.0}
    log = sc.log
    # After everything that crosses was captured, before the release.
    assert log.index("capture") < log.index("agent.reprove"), log
    assert log.index("collect") < log.index("agent.reprove"), log
    assert log.index("agent.reprove") < log.index("skill_export"), log
    assert log.index("agent.reprove") < log.index("agent.stop"), log
    assert log.count("agent.reprove") == 1
    assert sc.record()["agent_sandbox_policy"]["state"] == "held"


@pytest.mark.parametrize("environment", ["docker", "daytona"])
async def test_the_overlap_order_ends_a_breached_trial_with_nothing_transferred(
    tmp_path, monkeypatch, environment
):
    rules = _Rules(tmp_path)
    rules.remove_start_rule()
    sc = await _scenario(tmp_path, monkeypatch, environment=environment)
    sc.rollout._env = _RuledAgent(sc, rules)

    rewards = await _trial(sc)

    assert rewards is None
    error = sc.rollout._verifier_error
    assert classify_verifier_error(error) == "network_policy_breached"
    log = sc.log
    assert "pack" not in log and "verifier.upload tar" not in log, log
    assert "verifier.test_sh" not in log, log
    # A verifier sandbox the prestart order began is deleted, once.
    assert log.count("verifier.deleted") == (1 if environment == "docker" else 0)
    # cleanup() released the agent's sandbox.
    assert sc.rollout._env.stops == [True]
    record = sc.record()
    assert record["status"] == "network_policy_breached"
    assert record["agent_sandbox_policy"]["state"] == "breached"
    trial = sc.rollout._rollout_dir
    written = json.loads((trial / "verifier" / pr.NETWORK_POLICY_RECORD).read_text())
    assert written["at"] == pr.END_OF_AGENT
