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
- ``fw-pycap-setpriv``: ``fw-pycap`` with setpriv kept, run only under the
  helper start: the metadata block can verify the agent's launch,
  so its own delegate inventory must find the capability.

The agent's one Bash call tries to make (and delete) an empty chain of its
own with each tool and prints ``firewall <tool> changed`` or ``refused``,
and with the setuid shell prints ``rootsh euid <uid>``.

Since its helper start (#266, ``DockerSandbox._refuse_main_net_admin``)
Docker refuses that override wherever main's namespace needs rules (a
denylist, an open network with the metadata block): main would hold
NET_ADMIN and could remove them. Docker's own trials now start main without
NET_ADMIN, so S3 records ``clear`` at its first step; S3's disarm and refuse
paths run where root does hold NET_ADMIN (Daytona's daemon guard). To keep
them under real images and kernels, the two fixture jobs run ``bench``
through a test-only launcher (:func:`_old_start_trial_main`) that lifts that
one refusal in the test's own process, and
``test_docker_refuses_the_net_admin_override`` guards the refusal itself
with the plain ``bench``, on every image, word for word.

Each image's outcome under the helper start, the start every Docker trial
gets, is asserted too (``helper_start_job`` and
``helper_start_denylist_job``). There BenchFlow's helper container adds
main's rules from outside it, so main holds neither NET_ADMIN nor NET_RAW
and nothing in an image can give the agent the firewall. S3's check is then
clear on every image, and the metadata block's own checks (S2) decide: they
refuse an image whose agent launch they cannot verify or whose firewall
tools they cannot rule out, and disarm setuid and capability delegates for
the agent's turn.
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

# The capability on python3 with setpriv kept: the metadata block can
# verify the agent's launch, so its own delegate inventory, run by PT's
# protected kit Python with no getcap in the image, must find the capability.
PYCAP_SETPRIV_LAYER = PYCAP_LAYER.replace(" /usr/bin/setpriv", "")

LAYERS = {
    "fw-cap": CAP_LAYER,
    "fw-cap-denylist": CAP_LAYER,
    "fw-clean": "",
    "fw-setuid": SETUID_LAYER,
    "fw-rootsh": ROOTSH_LAYER,
    "fw-pycap": PYCAP_LAYER,
    "fw-pycap-setpriv": PYCAP_SETPRIV_LAYER,
    "fw-bbnnp": BBNNP_LAYER,
}
BUILDS = {"fw-setuid": SETUID_BUILD, "fw-bbnnp": BUSYBOX_BUILD}
FIXTURE_NAMES = tuple(LAYERS)
# The old start's shared job (its launcher): every image but the
# denylist's own job and its helper-start-only image.
OLD_START_NAMES = tuple(
    name
    for name in FIXTURE_NAMES
    if name not in ("fw-cap-denylist", "fw-pycap-setpriv")
)
# The helper start's shared job: every image but the denylist's.
HELPER_START_NAMES = tuple(name for name in FIXTURE_NAMES if name != "fw-cap-denylist")


def _old_start_trial_main() -> None:
    """Test-only CLI launcher: the old start, where main holds NET_ADMIN.

    Lifts ``DockerSandbox._refuse_main_net_admin`` (#266) in this process
    only, so ``BENCHFLOW_METADATA_MAIN_NET_ADMIN=1`` starts main as it did
    before the helper start and S3's disarm/refuse paths run on Docker as
    they do where root holds NET_ADMIN. Every other check runs as shipped;
    ``test_docker_refuses_the_net_admin_override`` keeps the refusal guarded.
    """
    from unittest.mock import patch

    from benchflow.cli.main import main
    from benchflow.sandbox.docker import DockerSandbox

    with patch.object(
        DockerSandbox, "_refuse_main_net_admin", lambda sandbox, why: None
    ):
        main()


def _old_start_launcher(root: Path) -> str:
    launcher = root / "bench-old-start"
    launcher.write_text(
        f"#!{sys.executable}\n"
        f"import sys; sys.path.insert(0, {str(h.REPO_ROOT)!r})\n"
        "from tests.test_deterministic_firewall_power import _old_start_trial_main\n"
        "_old_start_trial_main()\n"
    )
    launcher.chmod(0o755)
    return str(launcher)


def _materialize(root: Path, names: tuple[str, ...] = FIXTURE_NAMES) -> Path:
    tasks = root / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    for name in names:
        # The denylist's model agent: the fake on the host, never declared
        # in main, and the task a single image (refuses a
        # compose project under the denylist with main holding NET_ADMIN).
        dest = h.materialize_task(
            h.TaskVariant(
                name, "firewall-probe", fake_on_host=name == "fw-cap-denylist"
            ),
            tasks,
        )
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
    tasks = _materialize(root, OLD_START_NAMES)
    with (
        pytest.MonkeyPatch.context() as env,
        h.host_fake_llm(root / "host-fake-llm.jsonl") as url,
    ):
        # PERF-A b21700ed3 removes this power by default. Keep exercising S3's
        # disarm/refuse paths, and keep the helper record for the clear test.
        # Docker refuses the override since #266; the launcher lifts that one
        # refusal for this job (_old_start_trial_main).
        env.setenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", "1")
        launcher = _old_start_launcher(root)
        env.setattr(h, "bench_executable", lambda: launcher)
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
def bridge_fake_url(tmp_path_factory):
    """The fake provider on the host, at the Docker bridge's host address:
    the denylist job's in-sandbox proxy (root, which the agent's network
    rules do not hold) reaches it there."""
    if SANDBOX != "docker":
        pytest.skip("the fixtures' images are built on Docker")
    root = tmp_path_factory.mktemp("bridge-fake")
    with h.host_fake_llm(
        root / "host-fake-llm.jsonl", host=h.docker_bridge_host()
    ) as url:
        yield url


@pytest.fixture(scope="module")
def firewall_denylist_job(tmp_path_factory, bridge_fake_url) -> h.CliRun:
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    if SANDBOX != "docker":
        pytest.skip("the fixtures' images are built on Docker")
    root = tmp_path_factory.mktemp("firewall-power-denylist")
    tasks = _materialize(root, ("fw-cap-denylist",))
    # The denylist forces the proxy into the sandbox, which reaches the same
    # scripted provider on the host at the Docker bridge's host address. A
    # host loopback URL is not its peer.
    with pytest.MonkeyPatch.context() as env:
        env.setenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", "1")
        launcher = _old_start_launcher(root)
        env.setattr(h, "bench_executable", lambda: launcher)
        return h.run_bench(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=root / "jobs",
            sandbox=SANDBOX,
            host_fake_url=bridge_fake_url,
            extra=["--concurrency", "1", "--retry-attempts", "0"],
        )


@pytest.fixture(scope="module")
def helper_start_job(tmp_path_factory) -> h.CliRun:
    """Every image but the denylist's under the helper start: main
    without NET_ADMIN, every rule from BenchFlow's helper."""
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    if SANDBOX != "docker":
        pytest.skip("the fixtures' images are built on Docker")
    root = tmp_path_factory.mktemp("firewall-power-helper-start")
    tasks = _materialize(root, HELPER_START_NAMES)
    with (
        pytest.MonkeyPatch.context() as env,
        h.host_fake_llm(root / "host-fake-llm.jsonl") as url,
    ):
        # Keep the helper record for the clear tests and helper_rule_job.
        env.delenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", raising=False)
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
def helper_start_denylist_job(tmp_path_factory, bridge_fake_url) -> h.CliRun:
    """``fw-cap-denylist`` under the helper start."""
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    if SANDBOX != "docker":
        pytest.skip("the fixtures' images are built on Docker")
    root = tmp_path_factory.mktemp("firewall-power-helper-start-denylist")
    tasks = _materialize(root, ("fw-cap-denylist",))
    # The denylist forces the proxy into the sandbox, which reaches the same
    # scripted provider on the host at the Docker bridge's host address, as
    # firewall_denylist_job's does (no fake declared in main beside a
    # model agent under the denylist). A host loopback URL is not its peer.
    with pytest.MonkeyPatch.context() as env:
        env.delenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", raising=False)
        env.setenv("BENCHFLOW_METADATA_HELPER", "1")
        return h.run_bench(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=root / "jobs",
            sandbox=SANDBOX,
            host_fake_url=bridge_fake_url,
            extra=["--concurrency", "1", "--retry-attempts", "0"],
        )


@pytest.fixture(scope="module")
def override_refused_job(tmp_path_factory) -> h.CliRun:
    """Every image with the plain ``bench`` and the old start's override,
    which #266 refuses before any agent code, whatever the image."""
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    if SANDBOX != "docker":
        pytest.skip("the refusal is DockerSandbox's")
    root = tmp_path_factory.mktemp("firewall-power-override")
    tasks = _materialize(root)
    with pytest.MonkeyPatch.context() as env:
        env.setenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", "1")
        return h.run_bench(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=root / "jobs",
            sandbox=SANDBOX,
            host_fake_url=h.IN_SANDBOX_FAKE_URL,
            extra=["--concurrency", "4", "--retry-attempts", "0"],
        )


# The metadata block's refusals, word for word.
PRIVILEGE_REFUSAL = (
    "CloudMetadataPowerError: sandbox startup refused: the agent privilege "
    "construction could not be verified from /proc as root. The shared launch "
    "must set NoNewPrivs, remove NET_ADMIN and NET_RAW from CapBnd, and clear "
    "CapAmb. Provide util-linux setpriv and a readable /proc."
)
FIREWALL_TOOL_REFUSAL = (
    "CloudMetadataPowerError: sandbox startup refused: cannot rule out that the "
    "agent user 'agent' can change the firewall through a firewall tool "
    "(iptables, iptables-nft, ip6tables, ip6tables-nft). BenchFlow grants "
    "NET_ADMIN only to refuse the cloud metadata endpoints; file capabilities "
    "and setuid/setgid firewall tools could let the agent remove that refusal. "
    "Remove the capabilities or setuid/setgid bits in the task's Dockerfile "
    "(setcap -r <file>, chmod ug-s <file>), or set allow_cloud_metadata = true "
    "in the task's [environment] table, or run with --allow-cloud-metadata, to "
    "let the agent reach them."
)
# S3's record on every image under the helper start.
CLEAR_WHY = (
    "no process in the sandbox can hold NET_ADMIN (it is not in root's bounding "
    "set), so nothing in the image can give the agent the firewall"
)


def _old_start_refusal(policy: str) -> str:
    return (
        f"RuntimeError: sandbox startup refused: the task's network policy ({policy}) "
        "needs BenchFlow's helper container to add main's rules from outside it; "
        "BENCHFLOW_METADATA_MAIN_NET_ADMIN=1 would leave main holding NET_ADMIN, able "
        "to remove its own rules"
    )


def _assert_clear(trial: Path, installing: list[str]) -> None:
    """S3 under the helper start: nothing to drop, no "holds" before the
    rules are in."""
    record = h.read_json(trial / "firewall-power.json")
    assert record["state"] == "clear", record
    assert record["why"] == CLEAR_WHY, record
    assert record["installing"] == installing, record
    assert "holds" not in record and not record.get("found"), record


UNPROVEN_REFUSAL = (
    "CloudMetadataUnprotectedError: sandbox startup refused: the agent would "
    "start with the cloud metadata endpoints open to it (the block is "
    "unproven: the uid firewall refuses TCP from every non-root user to "
    "169.254.169.254, 169.254.170.2, 169.254.170.23, 100.100.100.200, "
    "168.63.129.16, 169.254.0.23, fd00:ec2::254, fd00:ec2::23, fd20:ce::254, "
    "fd00:c1::a9fe:a9fe on every port but 53; not proven: the agent user's "
    "connect to 169.254.169.254:80 {}). The trial stops before the agent "
    "starts; set allow_cloud_metadata = true in the task's [environment] "
    "table, or run with --allow-cloud-metadata, to let the agent reach them."
)


def _assert_refused_by_the_block(trial: Path, error: str) -> None:
    result = _result(trial)
    assert result.get("error") == error, result
    assert not result.get("rewards"), result
    assert _probes(trial) == {}
    _assert_clear(trial, ["the cloud metadata block"])


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


def _why_unscored(run: h.CliRun, trial: Path) -> str:
    """What a trial left about why it did not score: its error and category,
    the phases it reached (the failing step is the next one), the two
    defenses' records, its traceback's tail and the job's output. A shared
    job whose one trial fails then names that step, where R4b-prime's gate saw
    only ``TypeError: 'NoneType' object is not subscriptable``."""
    result = _result(trial)
    lines = [
        f"trial {trial.name}: error_category={result.get('error_category')!r}",
        f"error: {result.get('error')!r}",
    ]
    lines += [
        f"{key}: {value!r}"
        for key, value in result.items()
        if (key.endswith("_info") or key == "verifier_error") and value
    ]
    lines.append("phases reached: " + (", ".join(result.get("timing") or {}) or "none"))
    for name in ("cloud-metadata.json", "firewall-power.json"):
        path = trial / name
        if not path.exists():
            lines.append(f"{name}: not written")
            continue
        record = h.read_json(path)
        lines.append(f"{name}: {json.dumps(record, default=str)[:2000]}")
    traceback = trial / "error-traceback.txt"
    if traceback.exists():
        lines.append(
            "traceback tail:\n" + traceback.read_text(errors="replace")[-2000:]
        )
    lines.append("job output tail:\n" + run.output[-3000:])
    return "\n".join(lines)


def _scored(run: h.CliRun, task: str) -> dict[str, Any]:
    """``task``'s result once its trial is known to have scored with no
    error; otherwise the test fails with why it did not (_why_unscored)."""
    trial = run.trial(task)
    result = _result(trial)
    assert not result.get("error") and result.get("rewards"), _why_unscored(run, trial)
    return result


def test_a_capability_on_iptables_is_disarmed_and_the_trial_runs(firewall_denylist_job):
    """Guards S3's live drop alongside S2 e81b486c's stricter metadata audit."""
    result = _scored(firewall_denylist_job, "fw-cap-denylist")
    trial = firewall_denylist_job.trial("fw-cap-denylist")
    assert _probes(trial) == {"iptables": "refused"}, firewall_denylist_job.output[
        -3000:
    ]
    record = h.read_json(trial / "firewall-power.json")
    assert record["state"] == "disarmed", record
    assert [e["path"] for e in record["found"]] == ["/usr/sbin/xtables-nft-multi"]
    assert not int(record["agent_capabilities"]["bounding"], 16) >> 12 & 1
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
    result = _scored(firewall_job, "fw-clean")
    assert result["rewards"]["reward"] == 1.0
    trial = firewall_job.trial("fw-clean")
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
        if result is not None and "before" not in evidence:
            # Since #266 the helper also installs main's every-uid rule at
            # sandbox start and runs the verifier network's commands; act
            # once, after the install that leaves the agent's rules (the
            # shape _read_helper_rules checks) in both families.
            snapshot = await _read_helper_rules(sandbox)
            try:
                _assert_helper_rules(snapshot)
            except AssertionError:
                return result
            evidence["before"] = snapshot
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
def helper_rule_job(helper_start_job, tmp_path, monkeypatch, request):
    # The preceding real job built the helper. Reuse its recorded image so
    # this tests the ready-helper path even on a cold daemon. That is the
    # helper start's job, whose trials all start with the helper.
    monkeypatch.setenv(
        "BENCHFLOW_METADATA_HELPER_STATE_DIR",
        str(helper_start_job.jobs_dir.parent / "metadata-helper"),
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
    evidence = h.read_json(receipt) if receipt.exists() else {}
    # A trial that ended before the observer's reads leaves no "before"
    # (R4a's no-rule gate: KeyError 'before'); say why it ended instead.
    assert "before" in evidence and "after" in evidence, _why_unscored(
        run, run.trial("fw-clean")
    )
    # The rules live in main's namespace. Since #266 each new helper command
    # runs in a new helper container (MetadataHelper._served), so the helper
    # that read "before" need not be the one that reads "after".
    assert evidence["before"]["main"] == evidence["after"]["main"], evidence
    assert evidence["before"]["helper"] and evidence["after"]["helper"], evidence
    return run, evidence


def _assert_clear_trial(run):
    assert _scored(run, "fw-clean")["rewards"]["reward"] == 1.0
    trial = run.trial("fw-clean")
    assert run.returncode == 0, run.output[-3000:]
    power = h.read_json(trial / "firewall-power.json")
    assert power["state"] == "clear", power
    assert power["why"] == CLEAR_WHY, power
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
    """Guards review M1 at 99b301442: a successful IPv4 proof is insufficient.
    Since its helper start the product catches it too: the end-of-turn
    re-proof finds the IPv6 rules gone and ends the trial "network policy
    breached" (the trial scored before, and only this test's
    reader saw the loss)."""
    run, evidence = helper_rule_job
    # The IPv4 proof alone would pass this trial. The end-of-turn re-proof
    # finds main's IPv6 rules gone and does not score it.
    trial = run.trial("fw-clean")
    result = _result(trial)
    assert result.get("verifier_error_category") == "network_policy_breached", (
        _why_unscored(run, trial)
    )
    assert not result.get("rewards"), result
    # The four IPv6 metadata rules of the block's ten (CLOUD_METADATA_V6),
    # word for word.
    assert re.fullmatch(
        r"network policy breached: at the end of the agent's turn, BenchFlow's "
        r"network rules for the trial no longer held \(main \(start\): 4 of 10 "
        r"rules missing\); they were last proven present at "
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00\. The run is not scored: "
        r"its tests ran, or would have run, without the task's network policy\. "
        r"See verifier/network-policy\.json",
        result.get("verifier_error") or "",
    ), result
    policy = h.read_json(trial / "verifier" / "network-policy.json")
    assert policy["state"] == "breached", policy
    missing = [rule for entry in policy["entries"] for rule in entry["missing"]]
    assert missing and all(rule.startswith("v6: ") for rule in missing), policy
    assert h.read_json(trial / "firewall-power.json")["state"] == "clear"
    metadata = h.read_json(trial / "cloud-metadata.json")
    assert metadata["state"] == "on", metadata
    assert metadata["rules_from"] == "helper", metadata
    assert metadata["main_net_admin"] is False, metadata
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
    # The block's proof is the agent uid's connect to 169.254.169.254:80. With
    # main's rules gone it reaches the host's metadata server, or, on a host
    # that refuses that itself (the DOCKER-USER rule), is refused
    # without this rule's counter rising. Either way it is unproven, word for
    # word.
    assert result.get("error") in {
        UNPROVEN_REFUSAL.format("was not refused by the rule (reached)"),
        UNPROVEN_REFUSAL.format(
            "was refused, but not by this rule (its packet counter did not rise)"
        ),
    }, result
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
    result = _scored(firewall_job, "fw-rootsh")
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
    assert result["rewards"]["reward"] == 1.0


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


# Each image under the helper start, the start every Docker trial gets
# . S3 is clear on every image; the metadata block decides.


def test_a_capability_on_iptables_cannot_change_the_firewall_and_the_trial_runs(
    helper_start_denylist_job,
):
    """The helper-start twin of
     test_a_capability_on_iptables_is_disarmed_and_the_trial_runs: no process
     in main holds NET_ADMIN, so the capability on iptables gives the agent
     nothing: its own attempt to make a chain is refused and the trial scores
    ."""
    result = _scored(helper_start_denylist_job, "fw-cap-denylist")
    trial = helper_start_denylist_job.trial("fw-cap-denylist")
    assert _probes(trial) == {"iptables": "refused"}, helper_start_denylist_job.output[
        -3000:
    ]
    _assert_clear(trial, ["the denylist's uid firewall"])
    assert result["rewards"]["reward"] == 1.0
    assert "firewall_power" in result["timing"]


def test_metadata_audit_refuses_the_capability_image(
    helper_start_job,
):
    """The helper-start twin of
    test_metadata_audit_refuses_the_disarmed_capability_image: S3 is clear,
    and S2 still refuses the image, word for word."""
    _assert_refused_by_the_block(
        helper_start_job.trial("fw-cap"), FIREWALL_TOOL_REFUSAL
    )


def test_an_ordinary_image_runs_with_the_metadata_block_on_under_the_helper_start(
    helper_start_job,
):
    """The e81b486c merge's honest-image path under the helper start: the
    rules come from the helper, main has no NET_ADMIN, S3 is clear."""
    result = _scored(helper_start_job, "fw-clean")
    assert result["rewards"]["reward"] == 1.0
    trial = helper_start_job.trial("fw-clean")
    metadata = h.read_json(trial / "cloud-metadata.json")
    assert metadata["state"] == "on", metadata
    assert metadata["rules_from"] == "helper", metadata
    assert metadata["main_net_admin"] is False, metadata
    _assert_clear(trial, ["the cloud metadata block"])


def test_a_setuid_helper_without_setpriv_is_refused_under_the_helper_start(
    helper_start_job,
):
    """S3 refuses it under the old start ("the image has no setpriv"); under
    the helper start the metadata block cannot verify the agent's launch
    without setpriv and refuses it, word for word."""
    _assert_refused_by_the_block(helper_start_job.trial("fw-setuid"), PRIVILEGE_REFUSAL)


def test_a_setuid_root_shell_is_disarmed_and_put_back_under_the_helper_start(
    helper_start_job,
):
    """S3 disarms the launch under the old start; under the helper start it
    is clear, and S2's inventory is what takes rootsh's bit off (and su's)
    for the agent's turn and puts them back before grading."""
    result = _scored(helper_start_job, "fw-rootsh")
    trial = helper_start_job.trial("fw-rootsh")
    probes = _probes(trial)
    assert probes.get("euid") not in (None, "0"), (
        probes,
        helper_start_job.output[-3000:],
    )
    _assert_clear(trial, ["the cloud metadata block"])
    metadata = h.read_json(trial / "cloud-metadata.json")
    assert metadata["state"] == "on", metadata
    assert metadata["delegate_inventory"]["python"] == "kit", metadata
    disarmed = {d["path"]: d["why"] for d in metadata["delegates_disarmed"]}
    assert disarmed == {
        "/usr/local/bin/rootsh": "setuid root",
        "/usr/bin/su": "setuid root",
    }, disarmed
    restored = {
        r["path"]: (r["outcome"], r["why"]) for r in metadata["delegates_restored"]
    }
    assert restored == {
        "/usr/local/bin/rootsh": ("restored", "4755:exact"),
        "/usr/bin/su": ("restored", "4755:exact"),
    }, restored
    assert result["rewards"]["reward"] == 1.0


def test_a_capability_image_without_setpriv_is_refused(helper_start_job):
    """S3 finds the capability under the old start
     (test_a_capability_without_getcap_is_read_by_python3_and_refused). Under
     the helper start this image (no setpriv) is refused first by the metadata
     block, word for word; the capability itself is read by the next test
    ."""
    _assert_refused_by_the_block(helper_start_job.trial("fw-pycap"), PRIVILEGE_REFUSAL)


def test_a_capability_without_getcap_is_read_by_kit_python_and_disarmed(
    helper_start_job,
):
    """The same capability with setpriv kept: the metadata block's delegate
    inventory, run by PT's protected kit Python (never the image's, and with
    no getcap here), finds cap_setuid on the image's copy of python3, takes
    it off for the agent's turn and puts it back before grading."""
    result = _scored(helper_start_job, "fw-pycap-setpriv")
    trial = helper_start_job.trial("fw-pycap-setpriv")
    _assert_clear(trial, ["the cloud metadata block"])
    metadata = h.read_json(trial / "cloud-metadata.json")
    assert metadata["state"] == "on", metadata
    assert metadata["delegate_inventory"]["python"] == "kit", metadata
    disarmed = {d["path"]: d["why"] for d in metadata["delegates_disarmed"]}
    assert disarmed == {
        "/opt/py/python3": "file capabilities cap_setuid=ep",
        "/usr/bin/su": "setuid root",
    }, disarmed
    restored = {
        r["path"]: (r["outcome"], r["why"]) for r in metadata["delegates_restored"]
    }
    assert restored == {
        "/opt/py/python3": ("restored", "755:exact"),
        "/usr/bin/su": ("restored", "4755:exact"),
    }, restored
    assert result["rewards"]["reward"] == 1.0


def test_busyboxs_limited_setpriv_refuses_under_the_helper_start(helper_start_job):
    """S2-CONF-9 refuses the limited BusyBox launcher before agent code: under
    the helper start, the metadata block's own words (S3 refuses it
    under the old start and finds rootsh)."""
    _assert_refused_by_the_block(helper_start_job.trial("fw-bbnnp"), PRIVILEGE_REFUSAL)


@pytest.mark.parametrize(
    ("task", "policy"),
    [
        (name, "denylist" if name == "fw-cap-denylist" else "public")
        for name in FIXTURE_NAMES
    ],
)
def test_docker_refuses_the_net_admin_override(override_refused_job, task, policy):
    """Guards #266: on Docker the override that keeps main's old
    start is refused wherever main's namespace needs rules, an open network
    with the metadata block included, before the agent starts. On
    every image, in its own words, and before S3 runs, so no image check
    answers first (S3's disarm and refuse paths only ever ran on that
    start)."""
    trial = override_refused_job.trial(task)
    result = _result(trial)
    assert result.get("error") == _old_start_refusal(policy), _why_unscored(
        override_refused_job, trial
    )
    assert not result.get("rewards"), result
    assert _probes(trial) == {}
    assert not (trial / "firewall-power.json").exists(), task


def test_an_unscored_trial_is_reported_with_its_error_and_phases(tmp_path):
    """R4b-prime's no-rule gate saw fw-clean's errored trial only as
    ``TypeError: 'NoneType' object is not subscriptable`` on its rewards,
    and the job folder was gone. The report names the error, its category,
    the phases the trial reached, the records it left and the job's output.
    No sandbox: a trial folder as a job leaves it."""
    trial = tmp_path / "jobs" / "job" / "fw-clean__0"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "rewards": None,
                "error": "ExecTimeoutError: Command timed out after 300 seconds",
                "error_category": "sandbox_setup",
                "sandbox_startup_info": {"phase": "cloud_metadata_block"},
                "timing": {"environment_setup": 3.2, "install_agent": 22.7},
            }
        )
    )
    (trial / "firewall-power.json").write_text(json.dumps({"state": "disarmed"}))
    (trial / "error-traceback.txt").write_text(
        "Traceback (most recent call last):\nExecTimeoutError: timed out\n"
    )
    run = h.CliRun(["bench"], 1, "job output: 1 errored", 1.0, tmp_path / "jobs")
    with pytest.raises(AssertionError) as failed:
        _scored(run, "fw-clean")
    report = str(failed.value)
    for part in (
        "error: 'ExecTimeoutError: Command timed out after 300 seconds'",
        "error_category='sandbox_setup'",
        "sandbox_startup_info: {'phase': 'cloud_metadata_block'}",
        "phases reached: environment_setup, install_agent",
        "cloud-metadata.json: not written",
        'firewall-power.json: {"state": "disarmed"}',
        "ExecTimeoutError: timed out",
        "job output: 1 errored",
    ):
        assert part in report, report
