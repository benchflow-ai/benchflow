"""The re-proof against real iptables (lane C).

A container from BenchFlow's helper image (its pinned iptables) stands in for
main's namespace and its helper at once. The rules go in with BenchFlow's own
install scripts, the re-proof runs through :func:`policy_reproof.reprove`,
and the namespace is changed between proofs as a process holding NET_ADMIN
would: a rule deleted, the counters zeroed, the chain flushed, or another
phase's removal taking a rule that start put in.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import uuid
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from benchflow.sandbox import policy_reproof as pr
from benchflow.sandbox import rule_ledger as rl
from benchflow.sandbox import service_cover as sc
from benchflow.sandbox import verifier_network as vn


def _docker() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )


@pytest.fixture(scope="module")
def image():
    if not _docker():
        pytest.skip("Docker required")
    tag = f"bf-reproof-{uuid.uuid4().hex[:10]}"
    built = subprocess.run(
        ["docker", "build", "-q", "-t", tag, "-"],
        input=helper_mod.HELPER_DOCKERFILE,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if built.returncode != 0:
        pytest.skip(f"helper image build failed: {built.stderr[-300:]}")
    yield tag
    subprocess.run(["docker", "rmi", tag], capture_output=True)


class Namespace:
    """A fresh network namespace, with BenchFlow's helper joined to it."""

    def __init__(self, tag: str) -> None:
        self.name = f"bf-reproof-{uuid.uuid4().hex[:10]}"
        run = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.name,
                "--cap-add",
                "NET_ADMIN",
                "--cap-add",
                "NET_RAW",
                "--entrypoint",
                "/bin/sleep",
                tag,
                "300",
            ],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr
        self._benchflow_rule_ledger: list[rl.InstalledRules] = []

    def sh(self, script: str) -> str:
        return subprocess.run(
            ["docker", "exec", self.name, "/bin/sh", "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout

    async def metadata_rules_install(self, command: str):
        out = await asyncio.to_thread(self.sh, command)
        return SimpleNamespace(stdout=out), ""

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


@pytest.fixture
def ns(image):
    box = Namespace(image)
    try:
        yield box
    finally:
        box.close()


def _start(ns: Namespace) -> rl.InstalledRules:
    """main's start rule, as service_cover puts it in, proven by a connect."""
    m4, m6 = sc._metadata_rules()
    assert "BF service-rules status=0" in ns.sh(sc.install_cmd(m4, m6))
    ns.sh("nc -z -w 2 169.254.169.254 80")
    entry = rl.InstalledRules(where="main", phase="start", v4=m4, v6=m6)
    rl.record(ns, entry)
    return entry


def _verifier(ns: Namespace) -> rl.InstalledRules:
    """The verifier's allowlist rules, captured as installed."""
    v4, v6, nat = vn.everyone_rules(
        "allowlist", subnets=("172.20.0.0/16",), gateways=("172.20.0.1",)
    )
    out = ns.sh(rl.capture_cmd(vn.rules_cmd(v4, v6, nat)))
    assert "BF rules-installed status=0" in out, out
    a4, a6, anat = rl.added_rules(out)
    ns.sh("nc -z -w 2 192.0.2.1 80")
    entry = rl.InstalledRules(where="main", phase="verifier", v4=a4, v6=a6, nat=anat)
    rl.record(ns, entry)
    return entry


def _states(proof: pr.Reproof) -> dict[str, str]:
    return {r["phase"]: r["state"] for r in proof.entries}


@pytest.mark.asyncio
async def test_a_namespace_left_alone_holds_at_every_proof(ns):
    _start(ns)
    _verifier(ns)
    for _ in range(2):
        proof = await pr.reprove(ns, pr.END_OF_TEST)
        assert proof is not None and proof.held, proof.record()
    assert _states(proof) == {"start": "held", "verifier": "held"}


@pytest.mark.asyncio
async def test_the_counters_read_are_iptables_own(ns):
    _start(ns)
    proof = await pr.reprove(ns, pr.END_OF_AGENT)
    assert proof.held
    counts = ns._benchflow_rule_counts
    key = next(k for k in counts if k.endswith("|v4|0"))
    # The start proof's connect: one packet at the 169.254.169.254 rule.
    assert counts[key] == 1
    listed = ns.sh("iptables -w -L OUTPUT -v -x -n")
    assert any(
        line.split()[:1] == ["1"] and "169.254.169.254" in line
        for line in listed.splitlines()
    ), listed


@pytest.mark.asyncio
async def test_a_deleted_rule_is_named_missing(ns):
    _start(ns)
    verifier = _verifier(ns)
    assert (await pr.reprove(ns, pr.END_OF_AGENT)).held
    ns.sh("iptables -w -D OUTPUT -j REJECT")
    proof = await pr.reprove(ns, pr.END_OF_TEST)
    assert not proof.held
    row = next(r for r in proof.entries if r["phase"] == "verifier")
    assert row["state"] == "missing"
    final = ("-j REJECT", "-j REJECT --reject-with icmp-port-unreachable")
    assert row["missing"] == [f"v4: {s}" for s in verifier.v4 if s in final]


@pytest.mark.asyncio
async def test_zeroed_counters_are_a_breach(ns):
    _start(ns)
    _verifier(ns)
    assert (await pr.reprove(ns, pr.END_OF_AGENT)).held
    ns.sh("iptables -w -Z OUTPUT")
    proof = await pr.reprove(ns, pr.END_OF_TEST)
    assert not proof.held
    assert _states(proof)["start"] == "counters back"


@pytest.mark.asyncio
async def test_a_flushed_chain_is_a_breach_with_its_last_proof(ns):
    start = _start(ns)
    first = await pr.reprove(ns, pr.END_OF_AGENT)
    assert first.held
    ns.sh("iptables -w -F OUTPUT; ip6tables -w -F OUTPUT")
    proof = await pr.reprove(ns, pr.END_OF_TEST)
    (row,) = proof.entries
    assert row["state"] == "missing" and len(row["missing"]) == len(start.v4) + len(
        start.v6
    )
    assert proof.last_proven_present == first.checked_at


@pytest.mark.asyncio
async def test_another_phase_s_removal_of_a_shared_rule_flags_start(ns):
    """FX598's fx609 matrix: the oracle phase's removal of an identical spec
    took main's own start rule. Recorded as its own (the old way), its removal
    leaves start's rule missing and the re-proof says so; captured (only what
    it added), start's rule stays and the proof holds."""
    start = _start(ns)
    v4, v6, nat = vn.everyone_rules("denylist", subnets=("172.20.0.0/16",))
    # The old way: rules_cmd's -C || -I adds no second copy, the ledger
    # records the whole list anyway, and the removal deletes one copy each.
    assert "status=0" in ns.sh(vn.rules_cmd(v4, v6, nat))
    rl.record(
        ns, rl.InstalledRules(where="main", phase="agent-oracle", v4=v4, v6=v6, nat=nat)
    )
    ns.sh(vn.remove_rules_cmd(v4, v6, nat))
    rl.retire(ns, "main", "agent-oracle")
    proof = await pr.reprove(ns, pr.END_OF_AGENT, phases=pr.end_of_agent_phases)
    assert not proof.held
    (row,) = proof.entries
    assert (row["phase"], row["state"]) == ("start", "missing")
    assert f"v4: {start.v4[0]}" in row["missing"]


@pytest.mark.asyncio
async def test_a_captured_phase_removal_keeps_start_and_holds(ns):
    _start(ns)
    v4, v6, nat = vn.everyone_rules("denylist", subnets=("172.20.0.0/16",))
    out = ns.sh(rl.capture_cmd(vn.rules_cmd(v4, v6, nat)))
    a4, a6, anat = rl.added_rules(out)
    rl.record(
        ns,
        rl.InstalledRules(where="main", phase="agent-oracle", v4=a4, v6=a6, nat=anat),
    )
    assert (await pr.reprove(ns, pr.END_OF_AGENT)).held
    ns.sh(vn.remove_rules_cmd(a4, a6, anat))
    rl.retire(ns, "main", "agent-oracle")
    proof = await pr.reprove(ns, pr.END_OF_AGENT, phases=pr.end_of_agent_phases)
    assert proof.held, proof.record()
    assert _states(proof) == {"start": "held"}
