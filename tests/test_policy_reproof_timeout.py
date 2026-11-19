"""The end-of-test re-proof on a verifier timeout.

A test.sh still running at the verifier's deadline is killed and scored 0
(benchflow-ai/benchflow#1199), and the end-of-test re-proof used to be
skipped on that path: test.sh could write reward 1, remove a rule of the
verifier's network and outlive its timeout, and the trial ended a plain
scored 0 with no word of the breach. The rollout now proves the rules again
once the tests are stopped: a breach ends the trial "network policy breached"
(unscored, an error, never retried); a clean proof keeps the timeout's 0 and is
recorded. The rules and the helper are lane C's stubs
(tests/test_policy_reproof.py); the timeout is the rollout's own.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._utils.scoring import classify_verifier_error, is_network_policy_breach
from benchflow.rollout import _setup
from benchflow.sandbox import policy_reproof as pr
from benchflow.sandbox import verifier_network as vn
from tests.test_policy_reproof import (
    _held,
    _install,
    _sandbox,
    _start,
    _verifier,
    _verifier_entry,
)


def _timing_out(tmp_path, *, during_test=None, connects=None):
    """A trial whose test.sh writes reward 1 (and does *during_test* to main's
    namespace), then outlives the verifier's 1 s timeout."""
    box = _sandbox(tmp_path, connects=connects)
    _install(box, _start())
    entry = _verifier_entry()
    _install(box, entry)
    verifier, paths = _verifier(
        tmp_path,
        box,
        _held(),
        during_test=(lambda: during_test(box._main)) if during_test else None,
    )
    verifier._task.config.verifier.timeout_sec = 1.0
    inner = box.exec.side_effect

    async def exec_(command, *args, **kwargs):
        if command == _setup._VERIFIER_PROBE_SH:
            # test.sh's processes are alive: a plain timeout, no retry.
            return MagicMock(return_code=0, stdout="running\n")
        result = await inner(command, *args, **kwargs)
        if "> /logs/verifier/test-stdout.txt" in command:
            await asyncio.Event().wait()  # past the deadline
        return result

    box.exec = AsyncMock(side_effect=exec_)
    planes = SimpleNamespace(
        harden_before_verify=AsyncMock(), verifier=lambda **_kw: verifier
    )
    return box, entry, verifier, paths, planes


async def _score(verifier, paths, planes, box):
    return await _setup._verify_rollout(box, verifier._task, paths, {}, planes)


def _network_record(paths) -> dict:
    return json.loads((paths.verifier_dir / vn.VERIFIER_NETWORK_RECORD).read_text())


@pytest.mark.asyncio
async def test_a_rule_removed_by_a_test_sh_that_outlives_its_timeout_is_a_breach(
    tmp_path,
):
    box, entry, verifier, paths, planes = _timing_out(
        tmp_path, during_test=lambda ns: ns.ipt("-w", "-D", "OUTPUT", "-j", "REJECT")
    )

    rewards, error, diagnostic = await _score(verifier, paths, planes, box)

    assert rewards is None
    assert is_network_policy_breach(error)
    assert classify_verifier_error(error) == "network_policy_breached"
    assert pr.END_OF_TEST in error
    # Unscored, never this timeout's 0, and no timeout diagnostic: its
    # category would name the ending (rollout._results).
    assert diagnostic is None
    record = json.loads((paths.verifier_dir / pr.NETWORK_POLICY_RECORD).read_text())
    assert record["state"] == "breached" and record["at"] == pr.END_OF_TEST
    rows = {r["phase"]: r for r in record["entries"]}
    assert rows["verifier"]["missing"] == ["v4: -j REJECT"]
    assert record["last_proven_present"] == entry.proven_at
    # verifier-network.json says breached, never "held, refused 0".
    network = _network_record(paths)
    assert network["state"] == "breached" and "refused" not in network
    # The reward test.sh wrote before its deadline is kept, never a verdict.
    assert (paths.verifier_dir / "timeout-reward.txt").read_text() == "1"
    assert not paths.reward_text_path.exists()


@pytest.mark.asyncio
async def test_a_clean_timed_out_test_sh_keeps_its_0_and_records_the_proof(tmp_path):
    box, _entry, verifier, paths, planes = _timing_out(tmp_path)

    rewards, error, diagnostic = await _score(verifier, paths, planes, box)

    # Today's outcome: a scored 0, no error.
    assert (rewards, error) == ({"reward": 0.0}, None)
    assert diagnostic.scored is True
    assert (paths.verifier_dir / "timeout-reward.txt").read_text() == "1"
    assert not (paths.verifier_dir / pr.NETWORK_POLICY_RECORD).exists()
    # The proof ran and is on the record.
    network = _network_record(paths)
    assert network["state"] == "held" and network["timed_out"] is True
    assert network["policy"]["state"] == "held"
    (connect,) = [c for c in box._execs if "BF reproof-connect" in c]
    assert pr._METADATA in connect and pr._TEST_NET in connect


@pytest.mark.asyncio
async def test_counters_zeroed_by_a_timed_out_test_sh_are_a_breach(tmp_path):
    box, _entry, verifier, paths, planes = _timing_out(
        tmp_path, during_test=lambda ns: ns.ipt("-w", "-Z", "OUTPUT")
    )
    # A first read at the end of the agent's turn sets the counters' floors.
    box._main_cover_record = {"proof": {"metadata_counted": 1}}
    assert (await pr.require_held_at_end_of_agent(box)).held

    rewards, error, _ = await _score(verifier, paths, planes, box)

    assert rewards is None and is_network_policy_breach(error)
    assert "counters went back" in error


@pytest.mark.asyncio
async def test_a_connect_that_gets_through_after_a_timeout_is_a_breach(tmp_path):
    box, _entry, verifier, paths, planes = _timing_out(
        tmp_path, connects={pr._METADATA: "reached"}
    )

    rewards, error, diagnostic = await _score(verifier, paths, planes, box)

    assert rewards is None and is_network_policy_breach(error)
    assert f"connect to {pr._METADATA}:80" in error
    assert diagnostic is None


@pytest.mark.asyncio
async def test_a_verdict_that_fails_keeps_the_timeout_s_0(tmp_path, monkeypatch):
    """As after a test command that returned: BenchFlow's own failure in the
    verdict leaves the score it had, never a positive one."""
    box, _entry, verifier, paths, planes = _timing_out(tmp_path)

    async def broken(*_a, **_k):
        raise RuntimeError("dummy: the helper's docker call broke")

    monkeypatch.setattr(pr, "reprove", broken)

    rewards, error, diagnostic = await _score(verifier, paths, planes, box)

    assert (rewards, error) == ({"reward": 0.0}, None)
    assert diagnostic.scored is True


@pytest.mark.asyncio
async def test_an_unknown_test_end_is_not_judged(tmp_path, monkeypatch):
    """The sandbox did not answer whether the tests had ended: the timeout
    stays unscored as before, and nothing is asked of the helper."""
    box, _entry, verifier, paths, planes = _timing_out(tmp_path)

    async def unknown(*_a, **_k):
        raise _setup._TestEndUnknown("dummy: no answer")

    monkeypatch.setattr(_setup, "_await_verifier", unknown)
    runs_before = len(box._helper_runs)

    rewards, error, diagnostic = await _score(verifier, paths, planes, box)

    assert rewards is None and error.startswith("verifier timed out")
    assert diagnostic.scored is False
    assert len(box._helper_runs) == runs_before
