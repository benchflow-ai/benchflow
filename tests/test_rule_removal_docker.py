"""Removing a phase's rules never takes another phase's copy (EN-609, real iptables).

FXNAC on e154e42db: the root oracle's every-process rules went in with
``-C || -I``, so the metadata spec main already held from sandbox start was
not added a second time; but the ledger recorded it, and the oracle-phase
removal deleted main's own rule, leaving the metadata endpoints open until
hardening's cover (fx609 matrix6: 7 connects from planted binaries). Now an
install records only what it added, and only that is removed.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from benchflow.sandbox import rule_ledger as rl
from benchflow.sandbox import service_cover as sc
from benchflow.sandbox import verifier_network as vn


def _docker() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )


@pytest.fixture(scope="module")
def netns():
    """A container with its own network namespace, NET_ADMIN, and the
    helper's iptables (its pinned recipe)."""
    if not _docker():
        pytest.skip("Docker required")
    tag = f"bf-rule-removal-{uuid.uuid4().hex[:10]}"
    built = subprocess.run(
        ["docker", "build", "-q", "-t", tag, "-"],
        input=helper_mod.HELPER_DOCKERFILE,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if built.returncode != 0:
        pytest.skip(f"helper image build failed: {built.stderr[-300:]}")
    name = f"bf-rule-removal-{uuid.uuid4().hex[:10]}"
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
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

    def sh(script: str) -> str:
        done = subprocess.run(
            ["docker", "exec", name, "/bin/sh", "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return done.stdout

    try:
        yield sh
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "rmi", tag], capture_output=True)


def _present(sh, spec: str) -> bool:
    return "present" in sh(f"iptables -w -C OUTPUT {spec} && echo present")


def test_the_oracle_phase_removal_keeps_mains_own_metadata_rule(netns):
    sh = netns
    # Sandbox start: main's own metadata rule, every user.
    m4, m6 = sc._metadata_rules()
    assert "BF service-rules status=0" in sh(sc.install_cmd(m4, m6))
    assert all(_present(sh, spec) for spec in m4)
    # The oracle's turn: every-process rules, captured as installed.
    v4, v6, nat = vn.everyone_rules(
        "allowlist", subnets=("172.20.0.0/16",), gateways=("172.20.0.1",)
    )
    out = sh(rl.capture_cmd(vn.rules_cmd(v4, v6, nat)))
    assert "BF rules-installed status=0" in out, out
    a4, a6, anat = rl.added_rules(out)
    assert a4, out
    # The metadata spec main already held was not added again, so it is not
    # the oracle phase's to remove.
    assert not any("169.254.169.254" in spec and "-j REJECT" in spec for spec in a4), a4
    # The oracle phase's removal (one copy per spec it added).
    assert "removed status=0" in sh(vn.remove_rules_cmd(a4, a6, anat))
    # main's own rule is still there; the oracle phase's are gone.
    assert all(_present(sh, spec) for spec in m4)
    assert not any(_present(sh, spec) for spec in a4 if "--mark" in spec)


def test_the_ledger_check_reads_present_and_missing_with_real_iptables(netns):
    sh = netns
    entry = rl.InstalledRules(
        where="main", phase="start", v4=["-d 192.0.2.77/32 -j REJECT"]
    )
    sh("iptables -w -I OUTPUT 1 -d 192.0.2.77/32 -j REJECT")
    assert rl.missing_in(sh(rl.check_cmd(entry)), entry) == []
    sh("iptables -w -D OUTPUT -d 192.0.2.77/32 -j REJECT")
    assert rl.missing_in(sh(rl.check_cmd(entry)), entry) == [
        "v4: -d 192.0.2.77/32 -j REJECT"
    ]
