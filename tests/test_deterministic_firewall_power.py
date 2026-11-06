"""Deterministic tier: images that hand the agent the firewall, in real sandboxes.

Guards PY8 S3 end to end (the unit tests are tests/test_firewall_power.py).
The disarm/refuse fixtures explicitly give ``main`` NET_ADMIN for the cloud
metadata block (PY8 S2), so two images derived from the deterministic task
would hand the agent user the firewall that refuses the endpoints:

- ``fw-cap``: ``setcap cap_net_admin+ep`` on iptables (xtables-nft-multi).
  S3 disarms the launch, then S2's metadata inventory refuses the powerful
  file capability before the agent starts. ``fw-cap-denylist`` retains the
  live agent check under the denylist: its attempt to make a chain fails,
  and the trial scores as usual. ``fw-clean`` checks that an ordinary image
  still runs with the metadata block on.
- ``fw-setuid``: a setuid-root helper that runs iptables as root, and no
  setpriv in the image. Nothing can take NET_ADMIN away from the agent's
  commands, so the trial stops before the agent starts.
- ``fw-rootsh``: a setuid-root copy of bash, setpriv there. Without
  NET_ADMIN it would still make the agent uid 0, whose traffic none of the
  uid rules match (S3-R1-A-1); under the drop's no_new_privs it gives the
  agent nothing, so S3 disarms. Any process outside that launch could still
  run it as root, so S2's metadata inventory takes its setuid bit off for
  the agent's turn and puts it back before grading; the
  trial scores as usual. Earlier that inventory needed getcap, which
  this image does not have, and never looked.
- ``fw-pycap``: ``cap_setuid+ep`` on a copy of python3, then libcap's tools
  and setpriv removed (no getcap, no drop). Root's python3 reads the
  capability, and the trial stops before the agent starts (S3-R1-A-2).
  Root's reader is PT's protected kit Python, never the image's.
- ``fw-bbnnp``: BusyBox's setpriv in place of util-linux's (no ``--reuid``,
  no ``--bounding-set``, but ``--nnp``) and the setuid-root bash copy.
  S2-CONF-9 requires the bounding-set drop too, so this image is refused
  before the agent starts. The task's fake-provider service needs python3,
  so the image keeps it.

The agent's one Bash call tries to make (and delete) an empty chain of its
own with each tool and prints ``firewall <tool> changed`` or ``refused``,
and with the setuid shell prints ``rootsh euid <uid>``.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
REQUESTED = os.environ.get(h.SANDBOX_ENV, "").strip().lower() in {"docker", "daytona"}

FWHELPER_C = r"""#include <stdio.h>
#include <unistd.h>

/* A setuid-root firewall helper: runs iptables as root for whoever calls it
   (iptables itself refuses to run setuid). */
int main(int argc, char **argv) {
    (void)argc;
    if (setgid(0) != 0 || setuid(0) != 0) {
        perror("fwhelper");
        return 1;
    }
    argv[0] = "iptables";
    execv("/usr/sbin/iptables", argv);
    perror("fwhelper");
    return 127;
}
"""

CAP_LAYER = """
# PY8 S3 fixture: cap_net_admin on iptables, for every user who runs it.
RUN apt-get update -qq \\
    && apt-get install -y -qq --no-install-recommends iptables libcap2-bin \\
    && rm -rf /var/lib/apt/lists/* \\
    && setcap cap_net_admin+ep /usr/sbin/xtables-nft-multi
"""

SETUID_BUILD = """FROM ubuntu:24.04 AS fwhelper
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc libc6-dev
COPY fwhelper.c /fwhelper.c
RUN gcc -O2 -o /fwhelper /fwhelper.c

"""
SETUID_LAYER = """
# PY8 S3 fixture: a setuid-root firewall helper, and no setpriv.
RUN apt-get update -qq \\
    && apt-get install -y -qq --no-install-recommends iptables \\
    && rm -rf /var/lib/apt/lists/* /usr/bin/setpriv
COPY --from=fwhelper /fwhelper /usr/local/sbin/fwhelper
RUN chmod 4755 /usr/local/sbin/fwhelper
"""
ROOTSH_LAYER = """
# PY8 S3 fixture: a setuid-root copy of bash (setpriv stays).
RUN cp /bin/bash /usr/local/bin/rootsh && chmod 4755 /usr/local/bin/rootsh
"""

PYCAP_LAYER = """
# PY8 S3 fixture: cap_setuid on a copy of python3; no getcap, no setpriv.
RUN apt-get update -qq \\
    && apt-get install -y -qq --no-install-recommends libcap2-bin \\
    && mkdir -p /opt/py && cp /usr/bin/python3 /opt/py/python3 \\
    && setcap cap_setuid+ep /opt/py/python3 \\
    && apt-get purge -y -qq libcap2-bin \\
    && rm -rf /var/lib/apt/lists/* /usr/bin/setpriv
"""

BUSYBOX_BUILD = """FROM alpine:3.20 AS busybox
RUN apk add --no-cache busybox-static

"""
BBNNP_LAYER = """
# PY8 S3 fixture: BusyBox's setpriv in place of util-linux's (it can set
# no_new_privs, not switch users or drop from the bounding set), and a
# setuid-root copy of bash.
COPY --from=busybox /bin/busybox.static /usr/local/bin/busybox
RUN rm -f /usr/bin/setpriv \\
    && ln -s /usr/local/bin/busybox /usr/bin/setpriv \\
    && cp /bin/bash /usr/local/bin/rootsh && chmod 4755 /usr/local/bin/rootsh
"""

LAYERS = {
    "fw-cap": CAP_LAYER,
    "fw-cap-denylist": CAP_LAYER,
    "fw-clean": "",
    "fw-setuid": SETUID_LAYER,
    "fw-rootsh": ROOTSH_LAYER,
    "fw-pycap": PYCAP_LAYER,
    "fw-bbnnp": BBNNP_LAYER,
}
BUILDS = {"fw-setuid": SETUID_BUILD, "fw-bbnnp": BUSYBOX_BUILD}
FIXTURE_NAMES = tuple(LAYERS)


def _materialize(root: Path, names: tuple[str, ...] = FIXTURE_NAMES) -> Path:
    tasks = root / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    for name in names:
        dest = h.materialize_task(h.TaskVariant(name, "firewall-probe"), tasks)
        dockerfile = dest / "environment" / "Dockerfile"
        text = dockerfile.read_text()
        text, n = re.subn(r"(?m)^WORKDIR /app$", "", text)
        assert n == 1, "template Dockerfile must set WORKDIR /app"
        text = BUILDS.get(name, "") + text + LAYERS[name]
        if name == "fw-setuid":
            (dest / "environment" / "fwhelper.c").write_text(FWHELPER_C)
        dockerfile.write_text(text + "\nWORKDIR /app\n")
        if name == "fw-cap-denylist":
            task_md = dest / "task.md"
            task_text, count = re.subn(
                r"(?m)^  allow_internet: true$",
                "  allow_internet: true\n  network_mode: denylist\n"
                "  blocked_hosts:\n  - example.org",
                task_md.read_text(),
            )
            assert count == 1
            task_md.write_text(task_text)
    return tasks


@pytest.fixture(scope="module")
def firewall_job(tmp_path_factory) -> h.CliRun:
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    if SANDBOX != "docker":
        pytest.skip("the fixtures' images are built on Docker")
    root = tmp_path_factory.mktemp("firewall-power")
    tasks = _materialize(
        root, tuple(name for name in FIXTURE_NAMES if name != "fw-cap-denylist")
    )
    with (
        pytest.MonkeyPatch.context() as env,
        h.host_fake_llm(root / "host-fake-llm.jsonl") as url,
    ):
        # PERF-A b21700ed3 removes this power by default. Keep exercising S3's
        # disarm/refuse paths, and keep the helper record for the clear test.
        env.setenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", "1")
        env.setenv("BENCHFLOW_METADATA_HELPER", "1")
        env.setenv("BENCHFLOW_METADATA_HELPER_STATE_DIR", str(root / "metadata-helper"))
        return h.run_bench(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=root / "jobs",
            sandbox=SANDBOX,
            host_fake_url=url,
            extra=["--concurrency", "4", "--retry-attempts", "0"],
        )


@pytest.fixture(scope="module")
def firewall_denylist_job(tmp_path_factory) -> h.CliRun:
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    if SANDBOX != "docker":
        pytest.skip("the fixtures' images are built on Docker")
    root = tmp_path_factory.mktemp("firewall-power-denylist")
    tasks = _materialize(root, ("fw-cap-denylist",))
    # The denylist forces the proxy into the sandbox, where the manifest
    # starts the same scripted provider. A host loopback URL is not its peer.
    with pytest.MonkeyPatch.context() as env:
        env.setenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", "1")
        return h.run_bench(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=root / "jobs",
            sandbox=SANDBOX,
            host_fake_url=h.IN_SANDBOX_FAKE_URL,
            extra=["--concurrency", "1", "--retry-attempts", "0"],
        )


def _probes(trial: Path) -> dict[str, str]:
    for event in h.normalized_acp(trial):
        if event.get("type") == "tool_call" and event.get("tool_name") == "Bash":
            output = str(event.get("raw_output") or "")
            return dict(
                re.findall(
                    r"(?m)^(?:firewall|rootsh) (\S+) (changed|refused|\d+)$", output
                )
            )
    return {}


def _result(trial: Path) -> dict[str, Any]:
    return h.read_json(trial / "result.json")


def test_a_capability_on_iptables_is_disarmed_and_the_trial_runs(firewall_denylist_job):
    """Guards S3's live drop alongside S2 e81b486c's stricter metadata audit."""
    trial = firewall_denylist_job.trial("fw-cap-denylist")
    assert _probes(trial) == {"iptables": "refused"}, firewall_denylist_job.output[
        -3000:
    ]
    record = h.read_json(trial / "firewall-power.json")
    assert record["state"] == "disarmed", record
    assert [e["path"] for e in record["found"]] == ["/usr/sbin/xtables-nft-multi"]
    assert not int(record["agent_capabilities"]["bounding"], 16) >> 12 & 1
    result = _result(trial)
    assert result["rewards"]["reward"] == 1.0
    assert "firewall_power" in result["timing"]


def test_metadata_audit_refuses_the_disarmed_capability_image(firewall_job):
    """Guards the e81b486c merge: S3 disarming cannot waive S2's refusal."""
    trial = firewall_job.trial("fw-cap")
    record = h.read_json(trial / "firewall-power.json")
    assert record["state"] == "disarmed", record
    assert [e["path"] for e in record["found"]] == ["/usr/sbin/xtables-nft-multi"]
    assert not int(record["agent_capabilities"]["bounding"], 16) >> 12 & 1
    result = _result(trial)
    error = str(result.get("error"))
    assert "CloudMetadataPowerError: sandbox startup refused" in error
    assert "through a firewall tool (iptables" in error
    assert not result.get("rewards")
    assert _probes(trial) == {}


def test_an_ordinary_image_runs_with_the_metadata_block_on(firewall_job):
    """Guards the e81b486c merge's honest-image path as well as its refusals."""
    trial = firewall_job.trial("fw-clean")
    assert _result(trial)["rewards"]["reward"] == 1.0
    assert h.read_json(trial / "cloud-metadata.json")["state"] == "on"
    assert h.read_json(trial / "firewall-power.json")["state"] == "disarmed"


async def _read_helper_rules(sandbox) -> dict[str, Any]:
    """Read the real trial namespace, using the product's checked helper exec."""
    from benchflow.sandbox.cloud_metadata import CLOUD_METADATA_V4, CLOUD_METADATA_V6

    async def run(command):
        result, failed = await sandbox.metadata_rules_run(command, timeout_sec=30)
        return {
            "command": command,
            "return_code": getattr(result, "return_code", None),
            "stdout": getattr(result, "stdout", ""),
            "stderr": getattr(result, "stderr", ""),
            "failed": failed,
        }

    snapshot: dict[str, Any] = {
        "main": sandbox._metadata_helper.target,
        "helper": sandbox._metadata_helper.container,
        "ipv6_kernel": await run(
            "if [ -e /proc/net/if_inet6 ] || [ -d /proc/sys/net/ipv6 ]; "
            "then echo enabled; else echo absent; fi"
        ),
        "ipv6_disabled": await run(
            "for scope in all default; do "
            "printf '%s=' \"$scope\"; "
            'cat /proc/sys/net/ipv6/conf/"$scope"/disable_ipv6 || exit; done'
        ),
        "families": {},
    }
    for family, tool, addresses in (
        ("IPv4", "iptables", CLOUD_METADATA_V4),
        ("IPv6", "ip6tables", CLOUD_METADATA_V6),
    ):
        checks = {}
        for address in addresses:
            # Independent specification of every required rule: TCP except DNS,
            # non-root owners, REJECT with tcp-reset. -C handles canonical IPv6
            # rendering and implicit modules without fragile -S text matching.
            checks[address] = await run(
                f"{tool} -w -C OUTPUT -d {shlex.quote(address)} "
                "-p tcp ! --dport 53 -m owner ! --uid-owner 0 "
                "-j REJECT --reject-with tcp-reset"
            )
        snapshot["families"][family] = {
            "listing": await run(f"{tool} -w -S OUTPUT"),
            "checks": checks,
        }
    return snapshot


def _assert_helper_rules(snapshot: dict[str, Any]) -> None:
    assert snapshot["main"] and snapshot["helper"], snapshot
    for family, rules in snapshot["families"].items():
        listing = rules["listing"]
        assert not listing["failed"] and listing["return_code"] == 0, listing
        missing = [
            address
            for address, check in rules["checks"].items()
            if check["failed"] or check["return_code"] != 0
        ]
        assert not missing, f"missing {family} metadata rules: {missing}; {rules}"


def _helper_rule_trial_main(receipt: str, removal: str) -> None:
    """Test-only CLI launcher; production files and audit results stay intact."""
    from unittest.mock import patch

    from benchflow.cli.main import main
    from benchflow.sandbox.docker import DockerSandbox

    installed = DockerSandbox.metadata_rules_install
    stop = DockerSandbox.stop
    evidence: dict[str, Any] = {"removal": removal}

    async def install(sandbox, command):
        result = await installed(sandbox, command)
        if result is not None:
            evidence["before"] = await _read_helper_rules(sandbox)
            if removal != "none":
                command = "ip6tables -w -F OUTPUT"
                if removal == "all":
                    command = "iptables -w -F OUTPUT && " + command
                changed, failed = await sandbox.metadata_rules_run(
                    command, timeout_sec=30
                )
                evidence["mutation"] = {
                    "command": command,
                    "return_code": getattr(changed, "return_code", None),
                    "failed": failed,
                }
                assert not failed and changed.return_code == 0, evidence
        return result

    async def stop_and_read(sandbox, delete):
        try:
            # After grading (or startup refusal), before namespace teardown.
            # Do not reinstall: missing rules must remain visible.
            evidence["after"] = await _read_helper_rules(sandbox)
        finally:
            Path(receipt).write_text(json.dumps(evidence, indent=2))
            await stop(sandbox, delete)

    with (
        patch.object(DockerSandbox, "metadata_rules_install", install),
        # Keep the original helper alive until stop_and_read. Normal stop
        # still calls its real aclose(), removing it before main is torn down.
        patch.object(DockerSandbox, "metadata_rules_close", lambda sandbox: None),
        patch.object(DockerSandbox, "stop", stop_and_read),
    ):
        main()


@pytest.fixture
def helper_rule_job(firewall_job, tmp_path, monkeypatch, request):
    # The preceding real job built the helper. Reuse its recorded image so
    # this tests the ready-helper path even on a cold daemon.
    monkeypatch.setenv(
        "BENCHFLOW_METADATA_HELPER_STATE_DIR",
        str(firewall_job.jobs_dir.parent / "metadata-helper"),
    )
    monkeypatch.setenv("BENCHFLOW_METADATA_HELPER", "1")
    monkeypatch.delenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", raising=False)
    receipt = tmp_path / "helper-rules.json"
    launcher = tmp_path / "bench-rule-observer"
    launcher.write_text(
        f"#!{sys.executable}\n"
        f"import sys; sys.path.insert(0, {str(h.REPO_ROOT)!r})\n"
        "from tests.test_deterministic_firewall_power import _helper_rule_trial_main\n"
        f"_helper_rule_trial_main({str(receipt)!r}, {getattr(request, 'param', 'none')!r})\n"
    )
    launcher.chmod(0o755)
    monkeypatch.setattr(h, "bench_executable", lambda: str(launcher))
    tasks = _materialize(tmp_path, ("fw-clean",))
    with h.host_fake_llm(tmp_path / "host-fake-llm.jsonl") as url:
        run = h.run_bench(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=tmp_path / "jobs",
            sandbox=SANDBOX,
            host_fake_url=url,
            extra=["--concurrency", "1", "--retry-attempts", "0"],
        )
    evidence = h.read_json(receipt)
    for identity in ("main", "helper"):
        assert evidence["before"][identity] == evidence["after"][identity], evidence
    return run, evidence


def _assert_clear_trial(run):
    trial = run.trial("fw-clean")
    assert run.returncode == 0, run.output[-3000:]
    assert _result(trial)["rewards"]["reward"] == 1.0
    power = h.read_json(trial / "firewall-power.json")
    assert power["state"] == "clear", power
    metadata = h.read_json(trial / "cloud-metadata.json")
    assert metadata["state"] == "on", metadata
    assert metadata["rules_from"] == "helper", metadata
    assert metadata["main_net_admin"] is False, metadata


def test_helper_rules_leave_main_clear_without_net_admin(helper_rule_job):
    """Guards PERF-A b21700ed3 and review M1: both rule families must hold."""
    run, evidence = helper_rule_job
    _assert_clear_trial(run)
    # Unlike lockdown's network-wide allowlist/denylist (exit 86 when IPv6
    # is enabled without ip6tables), the metadata installer always requires
    # IPv6, even with absent proc entries. Never waive this half on the host.
    _assert_helper_rules(evidence["after"])


@pytest.mark.parametrize("helper_rule_job", ["ipv6"], indirect=True)
def test_helper_rule_assertion_catches_ipv6_loss(helper_rule_job):
    """Guards review M1 at 99b301442: a successful IPv4 proof is insufficient."""
    run, evidence = helper_rule_job
    _assert_clear_trial(run)
    _assert_helper_rules(evidence["before"])
    assert evidence["mutation"]["return_code"] == 0, evidence
    families = evidence["after"]["families"]
    assert (
        families["IPv4"]["listing"]["stdout"]
        == evidence["before"]["families"]["IPv4"]["listing"]["stdout"]
    ), evidence
    assert all(
        not check["failed"] and check["return_code"] == 0
        for check in families["IPv4"]["checks"].values()
    ), families
    assert all(
        not check["failed"] and check["return_code"] == 1
        for check in families["IPv6"]["checks"].values()
    ), families
    with pytest.raises(AssertionError, match="missing IPv6 metadata rules"):
        _assert_helper_rules(evidence["after"])


@pytest.mark.parametrize("helper_rule_job", ["all"], indirect=True)
def test_helper_complete_rule_loss_still_refuses_startup(helper_rule_job):
    """Guards 99b301442's complete-loss refusal while closing review M1."""
    run, evidence = helper_rule_job
    _assert_helper_rules(evidence["before"])
    trial = run.trial("fw-clean")
    result = _result(trial)
    assert not result.get("rewards"), result
    assert "CloudMetadataUnprotectedError" in str(result.get("error")), result
    assert h.read_json(trial / "cloud-metadata.json")["state"] == "unproven"
    assert all(
        not check["failed"] and check["return_code"] == 1
        for family in evidence["after"]["families"].values()
        for check in family["checks"].values()
    ), evidence
    with pytest.raises(AssertionError, match="missing IPv4 metadata rules"):
        _assert_helper_rules(evidence["after"])


def test_a_setuid_helper_without_setpriv_is_refused_before_the_agent(firewall_job):
    trial = firewall_job.trial("fw-setuid")
    result = _result(trial)
    error = str(result.get("error") or "")
    assert "sandbox startup refused" in error, result
    assert "the image has no setpriv" in error
    assert _probes(trial) == {}
    record = h.read_json(trial / "firewall-power.json")
    assert record["state"] == "refused", record
    assert record["drop"] is None


def test_a_setuid_root_shell_is_disarmed_for_the_agents_turn_and_put_back(
    firewall_job,
):
    """Option A against S2's fail-open (with no getcap in the
    image the metadata inventory never looked, and a setuid-root bash sat
    beside the agent) and against refusing the image: S3 disarms the
    agent's own launch, S2 takes rootsh's setuid bit off for the agent's
    turn and puts it back before grading, and the trial scores as usual."""
    trial = firewall_job.trial("fw-rootsh")
    probes = _probes(trial)
    assert probes.get("euid") not in (None, "0"), (probes, firewall_job.output[-3000:])
    record = h.read_json(trial / "firewall-power.json")
    assert record["state"] == "disarmed", record
    assert [e["path"] for e in record["found"]] == ["/usr/local/bin/rootsh"]
    metadata = h.read_json(trial / "cloud-metadata.json")
    assert metadata["state"] == "on", metadata
    assert metadata["delegate_inventory"]["python"] == "kit", metadata
    disarmed = {d["path"]: d["why"] for d in metadata["delegates_disarmed"]}
    assert disarmed["/usr/local/bin/rootsh"] == "setuid root", disarmed
    restored = {r["path"]: r["outcome"] for r in metadata["delegates_restored"]}
    assert restored["/usr/local/bin/rootsh"] == "restored", restored
    assert _result(trial)["rewards"]["reward"] == 1.0


def test_a_capability_without_getcap_is_read_by_python3_and_refused(firewall_job):
    trial = firewall_job.trial("fw-pycap")
    result = _result(trial)
    error = str(result.get("error") or "")
    assert "sandbox startup refused" in error, result
    assert _probes(trial) == {}
    record = h.read_json(trial / "firewall-power.json")
    assert record["state"] == "refused", record
    # Root reads file capabilities with PT's protected kit Python (never the
    # image's), so the capability on the image's copy of python3 is found.
    assert record["file_capabilities_checked"] is True
    assert record["found"] == [
        {
            "path": "/opt/py/python3",
            "kind": "file capability",
            "capabilities": "cap_setuid=ep",
        }
    ], record
    assert "/opt/py/python3 (a file capability: cap_setuid=ep)" in error
    assert record["drop"] is None


def test_busyboxs_limited_setpriv_refuses_before_a_setuid_root_shell(firewall_job):
    """Guards S2-CONF-9 / 70400f55: no nnp-only launch fallback remains."""
    trial = firewall_job.trial("fw-bbnnp")
    record = h.read_json(trial / "firewall-power.json")
    assert record["state"] == "refused", (record, firewall_job.output[-3000:])
    assert record["drop"] is None
    assert [e["path"] for e in record["found"]] == ["/usr/local/bin/rootsh"]
    error = str(_result(trial).get("error") or "")
    assert "sandbox startup refused" in error
    result = _result(trial)
    error = str(result.get("error") or "")
    assert "sandbox startup refused" in error, result
    # S2-CONF-9 refuses the limited BusyBox launcher before agent code.
    assert "Authentication required" not in error, error
    probes = _probes(trial)
    assert probes == {}, (probes, error)
