"""BenchFlow's network rules, proven again before a trial is scored.

The rule ledger records every rule BenchFlow's helper put in a namespace. At
the end of the agent's turn and at the end of test.sh the helper checks each
one again (``iptables -C``) and reads its counter, and test.sh's bounded
launch connects to the metadata server and TEST-NET. A rule gone, a counter
back, a check not made or a connect that got through ends the trial "network
policy breached": unscored, loud, with when the rules were last proven
present, and every reader counts it. The check script runs here under sh
against a stub iptables that keeps its rules in a file; the real-iptables runs
are in tests/test_policy_reproof_docker.py.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._utils.scoring import (
    VERIFIER_ERROR_CATEGORIES,
    classify_score_outcome,
    classify_verifier_error,
    score_summary_fields,
)
from benchflow.sandbox import policy_reproof as pr
from benchflow.sandbox import rule_ledger as rl
from benchflow.sandbox import service_cover as sc
from benchflow.sandbox import verifier_network as vn
from benchflow.task.verifier_errors import NetworkPolicyBreachedError

ROOT_HELD = (
    "BF verifier-launch uid=0 gid=0 CapBnd=00000000a80405fb "
    "CapEff=00000000a80405fb NoNewPrivs=1"
)

# --- a namespace's iptables, kept in a file -------------------------------------

_STUB = r"""#!/usr/bin/env python3
import json, os, re, sys
fam = "6" if "ip6tables" in os.path.basename(sys.argv[0]) else "4"
args = [a for a in sys.argv[1:] if a not in ("-w", "-x", "-n")]
verbose = "-v" in args
args = [a for a in args if a != "-v"]
table = "filter"
if "-t" in args:
    i = args.index("-t"); table = args[i + 1]; del args[i:i + 2]
path = os.environ["BF_STUB_STATE"]
state = json.load(open(path))
rules = state.setdefault(table + fam, [])
op, spec = args[0], " ".join(args[2:])
def shown(s):
    target = s.split("-j ", 1)[1].split()[0]
    return target, s.replace(" ", "_")
def save():
    json.dump(state, open(path, "w"))
if op == "-C":
    if not any(r[0] == spec for r in rules):
        sys.exit(1)
    if verbose:
        t, rest = shown(spec)
        print(f"{t}  all opt -- in * out *  0.0.0.0/0  -> 0.0.0.0/0   {rest}")
elif op == "-L":
    print("Chain OUTPUT (policy ACCEPT 0 packets, 0 bytes)")
    print("    pkts      bytes target     prot opt in     out     source               destination")
    for s, p in rules:
        t, rest = shown(s)
        print(f"{p:8d} 0 {t} all -- * * 0.0.0.0/0 0.0.0.0/0 {rest}")
elif op == "-S":
    print("-P OUTPUT ACCEPT" + (" -c 0 0" if verbose else ""))
    for s, p in rules:
        m, _, t = s.partition("-j ")
        m = re.sub(r"-d (\d+\.\d+\.\d+\.\d+) ", r"-d \1/32 ", m)
        print(f"-A OUTPUT {m}" + (f"-c {p} 0 " if verbose else "") + f"-j {t}")
elif op == "-Z":
    for r in rules:
        r[1] = 0
    save()
elif op == "-D":
    for i, r in enumerate(rules):
        if r[0] == spec:
            del rules[i]
            break
    else:
        sys.exit(1)
    save()
else:
    sys.exit(2)
"""


class Namespace:
    """One network namespace: its rules (spec, packets) by table."""

    def __init__(self, tmp: Path, name: str) -> None:
        self.dir = tmp / f"ns-{name}"
        self.bin = self.dir / "bin"
        self.bin.mkdir(parents=True)
        for tool in ("iptables", "ip6tables"):
            (self.bin / tool).write_text(_STUB)
            (self.bin / tool).chmod(0o755)
        self.state = self.dir / "state.json"
        self.state.write_text("{}")

    def add(self, entry: rl.InstalledRules, packets: int = 0) -> None:
        state = json.loads(self.state.read_text())
        for key, specs in (
            ("filter4", entry.v4),
            ("filter6", entry.v6),
            ("nat4", entry.nat),
        ):
            rows = state.setdefault(key, [])
            for spec in specs:
                if not any(r[0] == spec for r in rows):
                    rows.append(
                        [
                            spec,
                            packets if key == "filter4" and "-j REJECT" in spec else 0,
                        ]
                    )
        self.state.write_text(json.dumps(state))

    def ipt(self, *args: str) -> None:
        subprocess.run(
            [str(self.bin / "iptables"), *args],
            env={**os.environ, "BF_STUB_STATE": str(self.state)},
            check=False,
        )

    def hit(self, spec: str, packets: int = 1) -> None:
        state = json.loads(self.state.read_text())
        for row in state.get("filter4", []):
            if row[0] == spec:
                row[1] += packets
        self.state.write_text(json.dumps(state))

    def present(self, spec: str) -> bool:
        state = json.loads(self.state.read_text())
        return any(r[0] == spec for r in state.get("filter4", []))

    def sh(self, command: str) -> str:
        done = subprocess.run(
            ["sh", "-c", command],
            env={
                **os.environ,
                "PATH": f"{self.bin}:{os.environ['PATH']}",
                "BF_STUB_STATE": str(self.state),
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
        return done.stdout


class FakeSandbox(MagicMock):
    """A Docker sandbox as the re-proof sees it: BenchFlow's helper joined to
    main (and to each service), and root's execs in main."""

    async def metadata_rules_install(self, command):
        if self._helper_down:
            return None
        self._helper_runs.append(command)
        return SimpleNamespace(stdout=self._main.sh(command)), ""

    async def metadata_rules_run(self, command, *, timeout_sec, user=None):
        return SimpleNamespace(stdout=self._main.sh(command)), ""

    async def metadata_service_install(self, container, command):
        self._helper_runs.append(command)
        return SimpleNamespace(stdout=self._services[container].sh(command)), ""

    def metadata_service_close(self):
        return None


def _sandbox(tmp: Path, *, connects: dict[str, str] | None = None) -> FakeSandbox:
    box = FakeSandbox()
    box._helper_down = False
    box._helper_runs = []
    box._main = Namespace(tmp, "main")
    box._services = {}
    box._benchflow_rule_ledger = []
    box._connect_states = connects or {}
    box._execs = []

    async def exec_(command, *args, **kwargs):
        box._execs.append(command)
        if "BF reproof-connect" in command:
            out = "".join(
                f"BF reproof-connect {a} {box._connect_states.get(a, 'refused')}\n"
                for a in (pr._METADATA, pr._TEST_NET)
                if a in command
            )
            return MagicMock(return_code=0, stdout=out)
        return MagicMock(return_code=0, stdout="")

    box.exec = AsyncMock(side_effect=exec_)
    return box


def _start() -> rl.InstalledRules:
    m4, m6 = sc._metadata_rules()
    return rl.InstalledRules(
        where="main",
        phase="start",
        container="main-id",
        v4=m4,
        v6=m6,
        proven_at="2026-11-15T01:00:00.000+00:00",
    )


def _verifier_entry() -> rl.InstalledRules:
    v4, v6, nat = vn.everyone_rules(
        "allowlist", subnets=("172.20.0.0/16",), gateways=("172.20.0.1",)
    )
    return rl.InstalledRules(
        where="main",
        phase="verifier",
        v4=v4,
        v6=v6,
        nat=nat,
        proven_at="2026-11-15T01:00:30.000+00:00",
    )


def _install(box: FakeSandbox, entry: rl.InstalledRules, packets: int = 1) -> None:
    box._main.add(entry, packets)
    box._benchflow_rule_ledger.append(entry)


# --- the end of the agent's turn ------------------------------------------------


@pytest.mark.asyncio
async def test_a_clean_agent_turn_proves_every_rule_present(tmp_path):
    box = _sandbox(tmp_path)
    _install(box, _start())
    proof = await pr.require_held_at_end_of_agent(box)
    assert proof is not None and proof.held
    (row,) = proof.entries
    assert (row["state"], row["phase"]) == ("held", "start")
    # Proven now: the next check's "last proven present".
    assert row["last_proven_present"] == proof.checked_at
    # One helper run for main's namespace, no exec into main.
    assert len(box._helper_runs) == 1 and box._execs == []


@pytest.mark.asyncio
async def test_a_start_rule_removed_in_the_agent_s_turn_is_a_breach(tmp_path):
    box = _sandbox(tmp_path)
    start = _start()
    _install(box, start)
    box._main.ipt("-w", "-D", "OUTPUT", *start.v4[0].split())
    with pytest.raises(NetworkPolicyBreachedError) as caught:
        await pr.require_held_at_end_of_agent(box)
    record = caught.value.record
    assert record["state"] == "breached" and record["at"] == pr.END_OF_AGENT
    (row,) = record["entries"]
    assert row["state"] == "missing" and row["missing"] == [f"v4: {start.v4[0]}"]
    # Last proven present: its proof at sandbox start.
    assert record["last_proven_present"] == start.proven_at
    assert start.proven_at in str(caught.value)


@pytest.mark.asyncio
async def test_counters_zeroed_after_the_proof_are_a_breach(tmp_path):
    """main's start rule was proven by a refused, counted connect: a counter
    back to 0 means the counters were zeroed or the rule put in again."""
    box = _sandbox(tmp_path)
    _install(box, _start(), packets=1)
    box._main_cover_record = {"proof": {"metadata_counted": 1}}
    assert (await pr.require_held_at_end_of_agent(box)).held
    box._main.ipt("-w", "-Z", "OUTPUT")
    with pytest.raises(NetworkPolicyBreachedError) as caught:
        await pr.require_held_at_end_of_agent(box)
    (row,) = caught.value.record["entries"]
    assert row["state"] == "counters back"
    assert "(1 -> 0)" in row["counters_back"][0]


@pytest.mark.asyncio
async def test_counters_zeroed_before_the_first_check_are_a_breach(tmp_path):
    box = _sandbox(tmp_path)
    _install(box, _start(), packets=1)
    box._main_cover_record = {"proof": {"metadata_counted": 1}}
    box._main.ipt("-w", "-Z", "OUTPUT")
    with pytest.raises(NetworkPolicyBreachedError, match="counters went back"):
        await pr.require_held_at_end_of_agent(box)


@pytest.mark.asyncio
async def test_every_agent_phase_is_checked_whatever_its_name(tmp_path):
    """Lane A's agent-phase names (agent-uid, agent-guard, agent-metadata,
    agent-oracle) are checked without naming them; the verifier's are not
    there yet at the end of the agent's turn."""
    box = _sandbox(tmp_path)
    _install(box, _start())
    names = ("agent", "agent-uid", "agent-guard", "agent-metadata", "agent-oracle")
    for i, phase in enumerate(names):
        _install(
            box,
            rl.InstalledRules(
                where="main", phase=phase, v4=[f"-d 192.0.2.{10 + i}/32 -j REJECT"]
            ),
        )
    _install(
        box,
        rl.InstalledRules(
            where="main", phase="verifier", v4=["-d 192.0.2.99/32 -j REJECT"]
        ),
    )
    box._main.ipt("-w", "-D", "OUTPUT", "-d", "192.0.2.13/32", "-j", "REJECT")
    with pytest.raises(NetworkPolicyBreachedError) as caught:
        await pr.require_held_at_end_of_agent(box)
    rows = {r["phase"]: r["state"] for r in caught.value.record["entries"]}
    assert rows == {
        "start": "held",
        "agent": "held",
        "agent-uid": "held",
        "agent-guard": "held",
        "agent-metadata": "missing",
        "agent-oracle": "held",
    }


@pytest.mark.asyncio
async def test_a_retired_phase_is_not_expected_but_the_rule_it_shared_is(tmp_path):
    """FX598's matrix: a later phase's removal took out main's own start rule
    (an identical spec). The re-proof flags it as start's rule missing."""
    box = _sandbox(tmp_path)
    start = _start()
    _install(box, start)
    # The oracle phase recorded the same metadata spec (the old bug).
    oracle = rl.InstalledRules(where="main", phase="agent-oracle", v4=[start.v4[0]])
    box._benchflow_rule_ledger.append(oracle)
    # Its removal deletes the one copy there is, then it is retired.
    box._main.ipt("-w", "-D", "OUTPUT", *start.v4[0].split())
    rl.retire(box, "main", "agent-oracle")
    with pytest.raises(NetworkPolicyBreachedError) as caught:
        await pr.require_held_at_end_of_agent(box)
    (row,) = caught.value.record["entries"]
    assert (row["phase"], row["state"]) == ("start", "missing")
    assert row["missing"] == [f"v4: {start.v4[0]}"]


@pytest.mark.asyncio
async def test_a_check_the_helper_cannot_make_is_not_proven_never_a_breach(
    tmp_path, monkeypatch
):
    """The helper's check did not run, twice,
    and nothing shows a breach: the ending is "network policy not proven",
    unscored as BenchFlow's infrastructure, never "network policy breached"."""
    from benchflow._utils.scoring import VERIFIER_INFRA, is_network_policy_breach
    from benchflow.task.verifier_errors import NetworkPolicyUnprovenError

    monkeypatch.setattr(pr, "_RECHECK_SEC", 0)
    box = _sandbox(tmp_path)
    _install(box, _start())
    box._helper_down = True
    with pytest.raises(NetworkPolicyUnprovenError) as raised:
        await pr.require_held_at_end_of_agent(box)
    error = raised.value
    assert error.record["state"] == "not proven"
    assert "breaches" not in error.record and error.record["unproven"]
    ending = f"{pr.ending_head(error)}{error}"
    assert ending.startswith(pr.NETWORK_POLICY_UNPROVEN)
    assert not is_network_policy_breach(ending)
    assert classify_verifier_error(ending) == VERIFIER_INFRA
    assert "infrastructure failure, not the agent's" in str(error)


@pytest.mark.asyncio
async def test_a_check_made_on_its_second_try_proves_the_rules(tmp_path, monkeypatch):
    """A helper that could not run the check once (slow to start) runs it on
    the second try, and the rules are proven."""
    monkeypatch.setattr(pr, "_RECHECK_SEC", 0)
    box = _sandbox(tmp_path)
    _install(box, _start())
    tries = {"n": 0}
    real = FakeSandbox.metadata_rules_install

    async def flaky(self, command):
        tries["n"] += 1
        if tries["n"] == 1:
            return None
        return await real(self, command)

    monkeypatch.setattr(FakeSandbox, "metadata_rules_install", flaky)
    proof = await pr.require_held_at_end_of_agent(box)
    assert proof is not None and proof.proven and tries["n"] == 2


@pytest.mark.asyncio
async def test_a_rule_gone_is_a_breach_even_where_another_check_was_not_made(
    tmp_path, monkeypatch
):
    """Evidence wins: main's rule gone is a breach, whatever a service's
    unchecked namespace says."""
    from benchflow.task.verifier_errors import NetworkPolicyUnprovenError

    monkeypatch.setattr(pr, "_RECHECK_SEC", 0)
    box = _sandbox(tmp_path)
    start = _start()
    _install(box, start)
    box._main.ipt("-w", "-D", "OUTPUT", *start.v4[0].split())
    service = rl.InstalledRules(
        where="service db", phase="start", container="db-id", v4=["-j REJECT"]
    )
    box._benchflow_rule_ledger.append(service)

    async def down(container, command):
        return None

    box.metadata_service_install = down
    with pytest.raises(NetworkPolicyBreachedError) as raised:
        await pr.require_held_at_end_of_agent(box)
    assert not isinstance(raised.value, NetworkPolicyUnprovenError)
    assert raised.value.record["state"] == "breached"
    assert raised.value.record["unproven"]


@pytest.mark.asyncio
async def test_a_service_s_rules_are_checked_from_its_own_helper(tmp_path):
    box = _sandbox(tmp_path)
    _install(box, _start())
    side = Namespace(tmp_path, "side")
    box._services["side-id"] = side
    v4, v6 = sc.service_rules(metadata=True, egress="closed", subnets=(), gateways=())
    entry = rl.InstalledRules(
        where="service side", phase="start", container="side-id", v4=v4, v6=v6
    )
    side.add(entry)
    box._benchflow_rule_ledger.append(entry)
    assert (await pr.require_held_at_end_of_agent(box)).held
    side.ipt("-w", "-D", "OUTPUT", "-j", "REJECT")
    with pytest.raises(NetworkPolicyBreachedError, match=r"service side \(start\)"):
        await pr.require_held_at_end_of_agent(box)


@pytest.mark.asyncio
async def test_a_sandbox_with_no_ledger_is_not_checked(tmp_path):
    """A sandbox whose rules did not come from the helper (built directly, a
    provider): nothing to prove, no helper run, no exec, no alarm."""
    box = _sandbox(tmp_path)
    assert await pr.require_held_at_end_of_agent(box) is None
    assert await pr.reprove(box, pr.END_OF_TEST) is None
    assert box._helper_runs == [] and box._execs == []
    plain = SimpleNamespace(exec=AsyncMock())
    assert await pr.require_held_at_end_of_agent(plain) is None
    plain.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_hardening_ends_the_trial_before_any_verifier_step(tmp_path, monkeypatch):
    from benchflow.sandbox import lockdown

    box = _sandbox(tmp_path)
    start = _start()
    _install(box, start)
    box._main.ipt("-w", "-D", "OUTPUT", *start.v4[0].split())
    later = AsyncMock()
    monkeypatch.setattr(lockdown, "_cloud_metadata_verifier_applies", lambda *_: True)
    monkeypatch.setattr(lockdown, "cover_cloud_metadata_for_verifier", later)
    task = SimpleNamespace(
        task_dir=None,
        config=SimpleNamespace(verifier=SimpleNamespace(env={}, service="main")),
    )
    with pytest.raises(NetworkPolicyBreachedError):
        await lockdown.harden_before_verify(box, task, None, integrity="strict")
    later.assert_not_awaited()


# --- the end of test.sh -----------------------------------------------------------


def _verifier(
    tmp_path: Path,
    box: FakeSandbox,
    armed: vn.VerifierNetwork,
    *,
    reward: str = "1",
    exit_code: int = 0,
    refused_now: int = 0,
    during_test=None,
):
    from benchflow.task import RolloutPaths, Verifier
    from tests.test_llm_judge_verifier import _make_task

    task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
    task.paths.tests_dir = task.task_dir / "tests"
    task.paths.test_path = task.task_dir / "tests" / "test.sh"
    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    box.upload_dir = AsyncMock()
    box.is_mounted = True
    box._benchflow_verifier_network = armed
    written: dict[str, str] = {}
    base = box.exec.side_effect

    async def exec_(command, *args, **kwargs):
        if "BF verifier-network refused" in command and "BF rule-entry" not in command:
            return MagicMock(
                return_code=0, stdout=f"BF verifier-network refused {refused_now}\n"
            )
        if "IFS= read -r bf_l" in command and "verifier-launch-" in command:
            # The record is read with builtins (no program of the image).
            path = shlex.split(command.split("< ", 1)[1].split(";", 1)[0])[0]
            return MagicMock(return_code=0, stdout=written.get(path, ""))
        if "test-stdout" in command:
            path = re.search(r"> (/run/runtime/\S+?) 2>/dev/null", command)[1]
            written[path.strip("'")] = ROOT_HELD + "\n"
            paths.reward_text_path.write_text(reward)
            if during_test is not None:
                during_test()
            return MagicMock(return_code=exit_code, stdout="")
        return await base(command, *args, **kwargs)

    box.exec = AsyncMock(side_effect=exec_)
    return Verifier(task, paths, box), paths


def _held(baseline: int = 4) -> vn.VerifierNetwork:
    return vn.VerifierNetwork(
        "allowlist",
        "held",
        match="every process",
        baseline=baseline,
        metadata_refused_before=1,
    )


@pytest.mark.asyncio
async def test_a_clean_test_sh_keeps_its_reward_and_records_the_proof(tmp_path):
    box = _sandbox(tmp_path)
    _install(box, _start())
    _install(box, _verifier_entry())
    verifier, paths = _verifier(tmp_path, box, _held())
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}
    record = json.loads((paths.verifier_dir / vn.VERIFIER_NETWORK_RECORD).read_text())
    assert record["state"] == "held" and record["refused_packets"] == 0
    assert record["policy"]["state"] == "held"
    # Both connects were made from inside the bounded launch, after the count.
    (connect,) = [c for c in box._execs if "BF reproof-connect" in c]
    assert "--bounding-set=-net_admin,-net_raw,-sys_ptrace" in connect
    assert pr._METADATA in connect and pr._TEST_NET in connect


@pytest.mark.asyncio
async def test_a_rule_removed_during_test_sh_ends_network_policy_breached(tmp_path):
    box = _sandbox(tmp_path)
    _install(box, _start())
    entry = _verifier_entry()
    _install(box, entry)
    # Proven at the end of the agent's turn.
    agent_end = await pr.require_held_at_end_of_agent(box)
    last = agent_end.checked_at

    def remove():
        box._main.ipt("-w", "-D", "OUTPUT", "-j", "REJECT")

    verifier, paths = _verifier(tmp_path, box, _held(), during_test=remove)
    with pytest.raises(NetworkPolicyBreachedError) as caught:
        await verifier.verify()
    record = json.loads((paths.verifier_dir / vn.VERIFIER_NETWORK_RECORD).read_text())
    # Never "held, refused 0".
    assert record["state"] == "breached" and "refused" not in record
    assert "refused_packets" not in record
    policy = record["policy"]
    assert policy["at"] == pr.END_OF_TEST
    rows = {r["phase"]: r for r in policy["entries"]}
    assert rows["verifier"]["missing"] == ["v4: -j REJECT"]
    # Its last proof: the verifier's arm (the start rules were proven later,
    # at the end of the agent's turn, and still hold).
    assert policy["last_proven_present"] == entry.proven_at
    assert rows["start"]["state"] == "held"
    assert rows["start"]["last_proven_present"] > last
    assert "last proven present at " + entry.proven_at in str(caught.value)


@pytest.mark.asyncio
async def test_counters_zeroed_during_test_sh_end_network_policy_breached(tmp_path):
    """A failed test.sh with zeroed counters must not read "held, refused 0"."""
    box = _sandbox(tmp_path)
    _install(box, _start())
    _install(box, _verifier_entry(), packets=4)
    agent_end = await pr.require_held_at_end_of_agent(box)

    verifier, paths = _verifier(
        tmp_path,
        box,
        _held(baseline=4),
        reward="0",
        exit_code=1,
        during_test=lambda: box._main.ipt("-w", "-Z", "OUTPUT"),
    )
    with pytest.raises(NetworkPolicyBreachedError) as caught:
        await verifier.verify()
    policy = caught.value.record
    assert any("counters went back" in b for b in policy["breaches"])
    # main's start rule had been read at the end of the agent's turn.
    assert policy["last_proven_present"] == agent_end.checked_at
    record = json.loads((paths.verifier_dir / vn.VERIFIER_NETWORK_RECORD).read_text())
    assert record["state"] == "breached" and "refused_packets" not in record


@pytest.mark.asyncio
async def test_the_verifier_counter_below_its_arm_is_a_breach(tmp_path):
    """The final REJECT the arm proved (baseline 4), read lower at the end of
    test.sh: refusals hidden, never a count of 0."""
    box = _sandbox(tmp_path)
    _install(box, _verifier_entry(), packets=0)
    armed = _held(baseline=4)
    armed.via_helper = True
    verifier, _ = _verifier(tmp_path, box, armed)
    with pytest.raises(NetworkPolicyBreachedError, match="went back from 4 to 0"):
        await verifier.verify()


@pytest.mark.asyncio
@pytest.mark.parametrize("address", [pr._METADATA, pr._TEST_NET])
async def test_a_connect_that_gets_through_is_a_breach(tmp_path, address):
    box = _sandbox(tmp_path, connects={address: "reached"})
    _install(box, _start())
    _install(box, _verifier_entry())
    verifier, paths = _verifier(tmp_path, box, _held())
    with pytest.raises(NetworkPolicyBreachedError, match=f"connect to {address}:80"):
        await verifier.verify()
    record = json.loads((paths.verifier_dir / vn.VERIFIER_NETWORK_RECORD).read_text())
    assert record["policy"]["connects"][address] == "reached"


@pytest.mark.asyncio
async def test_the_breach_wins_over_a_pass_and_over_verifier_network_refused(tmp_path):
    box = _sandbox(tmp_path)
    _install(box, _start())
    _install(box, _verifier_entry())
    gone = lambda: box._main.ipt("-w", "-D", "OUTPUT", "-o", "lo", "-j", "ACCEPT")  # noqa: E731
    verifier, _ = _verifier(tmp_path, box, _held(), reward="1", during_test=gone)
    with pytest.raises(NetworkPolicyBreachedError):
        await verifier.verify()
    box2 = _sandbox(tmp_path / "b")
    _install(box2, _verifier_entry())
    gone2 = lambda: box2._main.ipt("-w", "-D", "OUTPUT", "-o", "lo", "-j", "ACCEPT")  # noqa: E731
    verifier2, _ = _verifier(
        tmp_path / "b",
        box2,
        _held(),
        reward="0",
        exit_code=1,
        refused_now=9,
        during_test=gone2,
    )
    with pytest.raises(NetworkPolicyBreachedError):
        await verifier2.verify()


# --- the ending and its readers ------------------------------------------------------

BREACHED = (
    "network policy breached: at the end of test.sh, BenchFlow's network rules "
    "for the trial no longer held (main (verifier): 1 of 9 rules missing); they "
    "were last proven present at 2026-11-15T01:00:30.000+00:00."
)


def _rows() -> list[dict]:
    return [
        {
            "task_name": "a",
            "rewards": {"reward": 1.0},
            "error": None,
            "verifier_error": None,
        },
        {
            "task_name": "b",
            "rewards": None,
            "error": None,
            "verifier_error": BREACHED,
            "verifier_error_category": classify_verifier_error(BREACHED),
        },
    ]


def test_the_ending_is_unscored_and_counts_0_in_mean_reward():
    from benchflow import scoring

    rows = _rows()
    assert classify_verifier_error(BREACHED) == "network_policy_breached"
    assert "network_policy_breached" in VERIFIER_ERROR_CATEGORIES
    assert classify_score_outcome(rows[1]) == "verifier_errored"
    record = scoring.trial_record(rows[1])
    assert (record["outcome"], record["error_class"]) == (
        "verifier_errored",
        "network_policy_breached",
    )
    summary = score_summary_fields(rows)
    assert summary["network_policy_breaches"] == 1
    assert summary["verifier_network_refusals"] == 0
    assert summary["mean_reward"] == pytest.approx(0.5)
    assert summary["verifier_error_categories"] == {"network_policy_breached": 1}


def test_the_ending_is_never_retried():
    from benchflow.rollout._verifier_recovery import (
        needs_verifier_recovery,
        verifier_error_reruns_trial,
    )

    assert not verifier_error_reruns_trial(BREACHED)
    assert not verifier_error_reruns_trial(BREACHED, on_resume=True)
    assert not needs_verifier_recovery(BREACHED)


def test_a_test_sh_cannot_write_the_ending():
    """Head-anchored: test.sh's own words after another error never read as it."""
    assert classify_verifier_error("verifier crashed: network policy breached: x") != (
        "network_policy_breached"
    )


def test_explain_names_the_cause_and_the_record():
    from benchflow.cli.explain import _BODIES
    from benchflow.failures import cause_of

    cause = cause_of(_rows()[1])
    assert cause is not None and cause.key == "network_policy_breached"
    assert "network-policy.json" in cause.next_step
    assert "last proven present" in _BODIES["network_policy_breached"]


def test_the_results_jsonl_catalog_keeps_the_category():
    from benchflow.diagnostics import RolloutDiagnostics

    block = RolloutDiagnostics().to_results_jsonl_block(
        error_category=None, verifier_error_category="network_policy_breached"
    )
    assert block == {
        "schema_version": 1,
        "verifier_error_category": "network_policy_breached",
    }


def test_the_run_card_counts_and_flags_the_job():
    from benchflow.run_checks import after_run_lines

    lines = after_run_lines({"network_policy_breaches": 1})
    assert any(
        '1 trial ended "network policy breached"' in ln and "flagged" in ln
        for ln in lines
    )
    assert not any("Network:" in ln for ln in after_run_lines({}))


def test_the_job_s_card_is_flagged_and_the_log_says_so(tmp_path, caplog):
    import logging

    from benchflow.evaluation import Evaluation

    ev = Evaluation.__new__(Evaluation)
    ev._run_card = {}
    with caplog.at_level(logging.ERROR):
        # An oracle job: no model calls, so the card's lines are not logged,
        # but a breach always is.
        ev._log_run_card_after(tmp_path, {}, network_policy_breaches=2)
    after = ev._run_card["after_run"]
    assert after["network_policy_breaches"] == 2
    assert "network_policy_breached" in after["warnings"]
    assert any(
        '2 trials ended "network policy breached"' in r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR
    )


def test_load_job_and_the_export_read_the_ending(tmp_path):
    import benchflow as bf
    from benchflow.job_export import job_export

    job = tmp_path / "job"
    for i, row in enumerate(_rows()):
        trial = job / f"{row['task_name']}__{i:08x}"
        trial.mkdir(parents=True)
        trial.joinpath("result.json").write_text(
            json.dumps({**row, "rollout_name": trial.name, "agent": "oracle"})
        )
    loaded = bf.load_job(job)
    breached = next(t for t in loaded.trials if t.raw.get("verifier_error"))
    assert breached.reward is None
    exported = job_export(loaded, include_verifier=False)
    (row,) = [t for t in exported.trials if t.verifier_error]
    assert row.verifier_error_category == "network_policy_breached"
    assert row.score_outcome == "verifier_errored"


@pytest.mark.asyncio
async def test_the_rollout_records_the_ending_and_its_record(tmp_path):
    """_verify_hardened: hardening's breach becomes the trial's verifier error
    in BenchFlow's own words, with verifier/network-policy.json written."""
    import contextlib

    from benchflow._utils.reachable_write import pin_destination
    from benchflow.rollout import _setup
    from benchflow.task import RolloutPaths

    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    # The trial folder is pinned before the agent runs, as a rollout pins it.
    pin_destination(paths.rollout_dir)
    record = {
        "state": "breached",
        "last_proven_present": "2026-11-15T01:00:00.000+00:00",
    }

    async def harden(*_a, **_k):
        raise NetworkPolicyBreachedError(
            "at the end of the agent's turn, ...", record=record
        )

    planes = SimpleNamespace(harden_before_verify=harden, verifier=MagicMock())
    task = SimpleNamespace(
        name="t",
        step_label=None,
        config=SimpleNamespace(verifier=SimpleNamespace(timeout_sec=60, env={})),
    )
    with contextlib.ExitStack() as stack:
        rewards, error, _ = await _setup._verify_hardened(
            SimpleNamespace(),
            task,
            paths,
            {},
            planes,
            None,
            None,
            recovery_eligible=False,
            agent_paths=None,
            root_path=stack,
        )
    assert rewards is None
    assert error.startswith(pr.NETWORK_POLICY_BREACHED)
    assert classify_verifier_error(error) == "network_policy_breached"
    written = json.loads((paths.verifier_dir / pr.NETWORK_POLICY_RECORD).read_text())
    assert written == record


@pytest.mark.parametrize("tag", ["", "[solver-preserved] "])
def test_a_preserved_solvers_network_endings_keep_their_category(tag):
    """RCI on lane C: a recovery-eligible task tags its ending
    ``[solver-preserved]``; the category is the ending's either way."""
    from benchflow._utils.scoring import (
        _NETWORK_POLICY_BREACHED_HEAD,
        _VERIFIER_LAUNCH_REFUSED_HEAD,
        _VERIFIER_NETWORK_REFUSED_HEAD,
        _VERIFIER_NETWORK_UNHELD_HEAD,
        NETWORK_POLICY_BREACHED,
        VERIFIER_LAUNCH_REFUSED,
        VERIFIER_NETWORK_REFUSED,
        VERIFIER_NETWORK_UNHELD,
        classify_verifier_error,
    )

    for head, category in (
        (_NETWORK_POLICY_BREACHED_HEAD, NETWORK_POLICY_BREACHED),
        (_VERIFIER_NETWORK_REFUSED_HEAD, VERIFIER_NETWORK_REFUSED),
        (_VERIFIER_LAUNCH_REFUSED_HEAD, VERIFIER_LAUNCH_REFUSED),
        (_VERIFIER_NETWORK_UNHELD_HEAD, VERIFIER_NETWORK_UNHELD),
    ):
        assert (
            classify_verifier_error(f"{tag}{head}main (start): 1 of 10 rules missing")
            == category
        )
