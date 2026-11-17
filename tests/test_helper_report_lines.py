"""Each helper script's real report line reads as reported.

BenchFlow's helper container runs a script and reads its ``status=`` from a
report line (``_metadata_helper._status``). On c572eb0de the compose services'
and the verifier network's scripts printed lines that reader did not accept,
so every one of their helper runs read as failed: a task with a networked
sidecar or a ``verifier.service`` was refused at start, and so was every
no-network task's verifier (RCI's matrix). Here each script runs for real,
under ``sh`` with stub ``iptables``, ``ip6tables`` and ``ip``, through the
helper's own exec and receipt path, in both directions: a script whose rules
went in is served, one whose rules failed is not.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from benchflow.sandbox import service_cover as sc
from benchflow.sandbox import verifier_network as vn
from tests.test_metadata_helper import MAIN, FakeDocker, _helper

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")

# iptables as the helper's scripts call it: -C and -D find no rule, -I/-A add one
# (or fail with STUB_FAIL set), -S OUTPUT -v lists counted rules.
STUB = """#!/bin/sh
case " $* " in
  *" -C "*|*" -D "*) exit 1;;
  *" -I "*|*" -A "*) [ -n "$STUB_FAIL" ] && exit 1; exit 0;;
  *" -S OUTPUT -v "*)
    if [ -n "$STUB_GROW" ]; then
      n=$(cat "$STUB_GROW" 2>/dev/null || echo 0); echo $((n + 1)) > "$STUB_GROW"
      echo "-A OUTPUT -c $n 0 -j REJECT"; exit 0
    fi
    echo "-P OUTPUT ACCEPT -c 0 0"
    echo "-A OUTPUT -d 169.254.169.254/32 -p tcp -c 2 120 -j REJECT --reject-with tcp-reset"
    echo "-A OUTPUT -c 3 180 -j REJECT";;
esac
exit 0
"""


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    folder = tmp_path / "bin"
    folder.mkdir()
    for name in ("iptables", "ip6tables", "ip"):
        (folder / name).write_text(STUB)
        (folder / name).chmod(0o755)
    return folder


class ShellDocker(FakeDocker):
    """FakeDocker whose ``exec`` runs the framed script in a real ``sh``."""

    def __init__(self, stubs: Path, *, fail: bool = False) -> None:
        super().__init__()
        self.env = {
            "PATH": f"{stubs}:/usr/bin:/bin",
            **({"STUB_FAIL": "1"} if fail else {}),
        }

    async def __call__(self, args, *, timeout, stdin=None):
        if args[:1] != ["exec"]:
            return await super().__call__(args, timeout=timeout, stdin=stdin)
        self.calls.append(list(args))
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            args[-1],
            env=self.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        return SimpleNamespace(
            stdout=out.decode(), stderr=err.decode(), return_code=proc.returncode
        )


def _service_rules() -> tuple[list[str], list[str]]:
    return sc.service_rules(
        metadata=True, egress="closed", subnets=("172.20.0.0/16",), gateways=()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("script", "line"),
    [
        (lambda: sc.install_cmd(*_service_rules()), "BF service-rules status=0"),
        (sc.counters_cmd, "BF service-counters metadata=2 egress=3 status=0"),
        (vn._no_route_rules_cmd, "BF verifier-network status=0"),
        (
            lambda: vn.rules_cmd(*vn.everyone_rules("no-network")[:2]),
            "BF verifier-network status=0",
        ),
        (vn.count_cmd, "BF verifier-network refused 3 status=0"),
        (
            lambda: vn.remove_rules_cmd(*vn.everyone_rules("no-network")[:2]),
            "BF verifier-network removed status=0",
        ),
    ],
    ids=[
        "service-rules",
        "service-counters",
        "no-route",
        "verifier-rules",
        "count",
        "remove",
    ],
)
async def test_each_scripts_real_output_is_served_by_the_helper(stubs, script, line):
    helper = _helper(ShellDocker(stubs))
    outcome = await helper.installed(MAIN, f": fresh; {script()}")
    assert outcome is not None, helper.unavailable
    result, failed = outcome
    assert failed == "" and line in result.stdout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "script",
    [
        lambda: sc.install_cmd(*_service_rules()),
        vn._no_route_rules_cmd,
    ],
    ids=["service-rules", "no-route"],
)
async def test_a_script_whose_rules_failed_is_not_served(stubs, script):
    helper = _helper(ShellDocker(stubs, fail=True))
    assert await helper.installed(MAIN, script()) is None
    assert "could not add the rules (status 1)" in helper.unavailable


@pytest.mark.asyncio
async def test_the_service_cover_reads_its_counts_through_the_helper(stubs):
    helper = _helper(ShellDocker(stubs))
    outcome = await helper.installed(MAIN, f": count; {sc.counters_cmd()}")
    assert outcome is not None
    assert sc._counters(outcome[0].stdout) == (2, 3)


def test_the_reader_wants_a_report_line():
    status = helper_mod._status
    assert status("BF metadata status=0 fw4=iptables\n") == "0"
    assert status("BF metadata fw4=iptables\n") == ""
    assert status("BF verifier-network routes 2\nBF verifier-network status=3\n") == "3"
    assert (
        status("BF service-rules status=1\nBF service-counters m=0 status=0\n") == "1"
    )
    # Another line, or another script's own report, is no report.
    assert status("BF daemon-guard ports= status=0\n") is None
    assert status("status=0\n") is None
    assert status("") is None


def test_no_helper_install_or_read_script_lacks_a_report_line():
    """The scripts that go through ``installed`` (a fresh helper run) each
    print a line ``_status`` reads; a new one must too."""
    produced = [
        sc.install_cmd(*_service_rules()),
        sc.counters_cmd(),
        vn._no_route_rules_cmd(),
        vn.rules_cmd(*vn.everyone_rules("no-network")[:2]),
        vn.count_cmd(),
        vn.remove_rules_cmd(*vn.everyone_rules("no-network")[:2]),
    ]
    for script in produced:
        assert "status=" in script and any(
            r.strip() in script for r in helper_mod._REPORT_LINES
        )
    assert os.environ.get("BENCHFLOW_METADATA_HELPER") != "0"


class HelperSandbox:
    """A Docker sandbox as the verifier network sees it: main's own execs
    (root, an image with no iptables) and BenchFlow's helper (ShellDocker)."""

    def __init__(self, helper) -> None:
        self.helper = helper
        self.main_commands: list[str] = []

    async def metadata_rules_install(self, command):
        return await self.helper.installed(MAIN, command)

    async def exec(self, command, *args, **kwargs):
        self.main_commands.append(command)
        if "BF verifier-network routes" in command:
            out = "gw 0100A8C0\nBF verifier-network routes 1\n"
        elif "BF verifier-proof connect" in command:
            out = "BF verifier-proof connect=refused\n"
        else:
            # The image has no iptables: a rules script in main reports 2.
            out = "BF verifier-network status=2\n"
        return SimpleNamespace(stdout=out, stderr="", return_code=0)


@pytest.mark.asyncio
async def test_an_offline_oracle_is_held_from_the_helper_when_main_has_no_iptables(
    stubs, tmp_path
):
    """FXPB on 6c05b15: on real ProgramBench images every offline oracle was
    refused ("its rules could not be added"): root's rules ran in main, whose
    image has no iptables. They now go in, are counted and come out from
    the helper."""
    docker = ShellDocker(stubs)
    docker.env["STUB_GROW"] = str(tmp_path / "count")
    env = HelperSandbox(_helper(docker))
    record = await vn.arm_offline_agent_phase(env)
    assert record["rules_from"] == "helper"
    assert record["proof"] == {"connect": "refused", "counted": 1}
    assert not any("-I OUTPUT" in c for c in env.main_commands)
    await vn.remove_offline_agent_phase(env)
    assert env._benchflow_offline_rules is None
    # The removal runs where the rules went in (the helper), for what the
    # install added (captured; this stub lists nothing, so nothing: the
    # real-iptables test covers the copies, test_rule_removal_docker).
    assert any(
        "BF verifier-network removed" in c[-1]
        for c in docker.calls
        if c[:1] == ["exec"]
    ), "the rules come out where they went in"
    from benchflow.sandbox import rule_ledger

    assert rule_ledger.expected(env, "agent") == []


@pytest.mark.asyncio
async def test_main_metadata_refusals_are_read_from_the_helper(stubs):
    """verifier-network.json's main_metadata_refused_before_verifier: what
    main's metadata rules refused during the agent's turn, read through a
    fresh helper run (the counters line carries its status)."""
    from benchflow.sandbox import rule_ledger
    from benchflow.sandbox.docker import DockerSandbox

    docker = ShellDocker(stubs)
    env = HelperSandbox(_helper(docker))
    calls = len(docker.calls)
    # No metadata rule of main's in the ledger: nothing to read, no helper run.
    assert await vn.main_metadata_refused(env) is None
    assert len(docker.calls) == calls
    rule_ledger.record(
        env, rule_ledger.InstalledRules(where="main", phase="start", container=MAIN)
    )
    assert await vn.main_metadata_refused(env) == 2
    assert await vn.main_metadata_refused(SimpleNamespace()) is None
    # A DockerSandbox built without __init__ (the container tests' hand-made
    # sandboxes) has no helper: the read is None, never an AttributeError.
    assert await vn.main_metadata_refused(DockerSandbox.__new__(DockerSandbox)) is None


class _RunsConnect:
    """A sandbox with BenchFlow's helper whose uid-0 connect answers *state*."""

    def __init__(self, state):
        self.state = state
        self.runs = []

    async def metadata_rules_install(self, command):
        return None

    async def metadata_rules_run(self, command, *, timeout_sec, user=None):
        self.runs.append((command, user))
        return SimpleNamespace(stdout=f"BF service-connect {self.state}\n"), ""


@pytest.mark.asyncio
async def test_a_shell_without_dev_tcp_gets_the_proof_connect_from_the_helper():
    """Alpine and BusyBox (fx609 launcher rows): sh has no /dev/tcp, so the
    proof connect read "failed" and hardening refused the verifier. The rules
    hold every process of main's namespace, so the helper's uid-0 connect
    there is the same proof."""
    env = _RunsConnect("refused")
    out, from_helper = await vn._shell_or_helper_connect(
        env, "BF verifier-proof connect=failed\nBF verifier-proof delete=skipped\n"
    )
    assert from_helper is True
    assert "BF verifier-proof connect=refused" in out and "connect=failed" not in out
    ((command, user),) = env.runs
    assert user == "0:0" and vn._PROOF_ADDRESS in command


@pytest.mark.asyncio
async def test_a_shell_that_made_the_connect_keeps_its_own_answer():
    env = _RunsConnect("reached")
    out, from_helper = await vn._shell_or_helper_connect(
        env, "BF verifier-proof connect=refused\n"
    )
    assert (out, from_helper, env.runs) == (
        "BF verifier-proof connect=refused\n",
        False,
        [],
    )


@pytest.mark.asyncio
async def test_without_a_helper_the_failed_connect_stands():
    out, from_helper = await vn._shell_or_helper_connect(
        SimpleNamespace(), "BF verifier-proof connect=failed\n"
    )
    assert (out, from_helper) == ("BF verifier-proof connect=failed\n", False)


@pytest.mark.asyncio
async def test_the_proxy_container_answers_its_health_from_inside(monkeypatch):
    """main's image may have no Python: the proxy's health is asked from its
    own container (which has python3)."""
    from benchflow.sandbox import _egress_container as ec
    from tests.test_metadata_helper import FakeDocker

    docker = FakeDocker()
    docker.exec_out = "BF egress-healthy\n"
    box = ec.EgressContainer(docker, daemon="d", name="t-bf-egress", labels={})
    box.container = "e" * 64
    await box.wait_healthy("http://127.0.0.1:18628/healthz", 5)
    (call,) = docker.of("exec")
    assert call[1] == "e" * 64 and "urllib.request" in call[-1]
    docker.exec_out = "nothing\n"
    with pytest.raises(ec.EgressContainerError, match="did not answer"):
        await box.wait_healthy("http://127.0.0.1:18628/healthz", 0)
