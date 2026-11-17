"""Deterministic tier: the agent keeps its environment on images without
util-linux setpriv, in real sandboxes.

Guards PY8 SU end to end (the unit tests are tests/test_su_env.py). Where the
image has no util-linux setpriv, BenchFlow starts the agent through ``su``;
that was ``su -l``, whose login shell cleared the agent env the root shell had
just sourced, so ``claude-agent-acp`` stopped at "Authentication required".
Six images derived from the deterministic task, two references with
util-linux setpriv and four without it:

- ``su-deb-setpriv``: Debian 12, util-linux setpriv (the reference).
- ``su-deb-nosp``: the same with setpriv removed: util-linux's su, whose
  PAM session adds MAIL (pam_mail).
- ``su-alp-setpriv``: Alpine 3.21 with util-linux setpriv added (the
  reference for Alpine).
- ``su-alp-bb``: Alpine as shipped: BusyBox's setpriv and su, so S3's
  ``nnp`` drop (root sets no_new_privs, then su).
- ``su-alp-nosp``: Alpine with BusyBox's setpriv removed: BusyBox's su.
- ``su-alp-shadow``: Alpine with shadow's su and no setpriv (shadow's su
  resets PATH even when it keeps the environment).

Every trial starts as every real run does since #234: BenchFlow's
helper container adds main's rules from outside it, and main holds neither
NET_ADMIN nor NET_RAW. The suite set ``BENCHFLOW_METADATA_MAIN_NET_ADMIN=1``
before, to keep main's NET_ADMIN and so the firewall-power drop; #234 refuses
that old start, so the suite drives the helper start and pins the old one as
refused (``test_the_old_in_main_start_is_refused``). The unsupported
images are refused by the metadata block's check of the agent's launch, each
with that refusal's own text, never any refusal.

Alpine's images carry musl Node (Node's own builds are glibc), so BenchFlow's
JS agent install keeps it. Root's rc files export ``BF_ROOT_ONLY``, the
agent's home gets rc files (from /etc/skel) and ``BASH_ENV`` that append
their uid to /tmp/bf-rc.log, and the run passes ``BF_SU_MARK`` with
``--agent-env``. The agent's one Bash call finds the ``claude-agent-acp``
process and prints the names of its environment's variables (never a
value), which markers are set, a hash of its PATH, and the rc log.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
REQUESTED = os.environ.get(h.SANDBOX_ENV, "").strip().lower() in {"docker", "daytona"}

# Root's own rc files export a marker no path may pass; the agent's home
# (useradd -m and BusyBox adduser copy /etc/skel) and BASH_ENV log who ran
# them.
PROBE_LAYER = """
# PY8 SU fixture: a root-only marker, and rc files that log their uid.
RUN for f in /root/.profile /root/.bashrc /root/.bash_profile /root/.ashrc; do \\
      echo 'export BF_ROOT_ONLY=1' >> "$f"; done \\
    && mkdir -p /etc/skel \\
    && for f in .profile .bashrc .bash_profile .bash_login .ashrc; do \\
      printf 'echo "rc %s uid=$(id -u)" >> /tmp/bf-rc.log\\n' "$f" >> "/etc/skel/$f"; \\
    done \\
    && printf 'echo "bash_env uid=$(id -u)" >> /tmp/bf-rc.log\\n' > /etc/bf-bash-env
"""

DEBIAN = """FROM debian:12

# python3 runs the fake model provider (fake_llm/); curl is the environment
# plane's readiness probe; iptables the cloud metadata block's.
RUN apt-get update -qq \\
    && apt-get install -y -qq --no-install-recommends python3 curl ca-certificates iptables \\
    && rm -rf /var/lib/apt/lists/*
"""

ALPINE = """FROM node:22-alpine3.21 AS node
FROM alpine:3.21

# bash runs BenchFlow's commands; python3 the fake model provider; curl the
# readiness probe; iptables the cloud metadata block. Node's own builds are
# glibc, so the image carries musl Node where BenchFlow's JS agents expect it.
RUN apk add --no-cache bash python3 curl ca-certificates libstdc++ iptables ip6tables
COPY --from=node /usr/local/bin/node /opt/runtime/node/bin/node
COPY --from=node /usr/local/lib/node_modules /opt/runtime/node/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /opt/runtime/node/bin/npm \\
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /opt/runtime/node/bin/npx
"""

COMMON = """
COPY fake_llm /opt/fake_llm
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
"""

# Exercise the remaining identity-only su fallback under S2's full barrier.
# The launch resolves setpriv on its pinned system PATH, never /usr/local, so
# the wrapper takes the place of the setpriv found there (/usr/bin/setpriv on
# both images; the build fails if not). Its --help hides --reuid, so the
# launch drops through su; the real util-linux executable, moved aside, still
# applies every requested restriction. A call without --reuid (the su
# branch's) appends a line to the rc log the probe prints.
BOUNDED_SU = """
RUN test "$(PATH=/usr/sbin:/usr/bin:/sbin:/bin; command -v setpriv)" = /usr/bin/setpriv \\
    && mkdir -p /usr/lib/bf-setpriv && mv /usr/bin/setpriv /usr/lib/bf-setpriv/setpriv \\
    && printf '#!/bin/sh\\nr=/usr/lib/bf-setpriv/setpriv\\nif [ "$1" = --help ]; then "$r" --help | sed /reuid/d; exit 0; fi\\ncase " $* " in *" --reuid="*) ;; *) echo setpriv-call no-reuid >> /tmp/bf-rc.log && chmod 666 /tmp/bf-rc.log ;; esac\\nexec "$r" "$@"\\n' > /usr/bin/setpriv \\
    && chmod 755 /usr/bin/setpriv
"""

FIXTURES = {
    "su-deb-bounded": (DEBIAN, BOUNDED_SU),
    "su-alp-bounded": (ALPINE, "RUN apk add --no-cache setpriv\n" + BOUNDED_SU),
    "su-alp-shadow-bounded": (
        ALPINE,
        "RUN apk add --no-cache setpriv shadow-login\n" + BOUNDED_SU,
    ),
    "su-deb-setpriv": (DEBIAN, ""),
    "su-deb-nosp": (DEBIAN, "RUN rm -f /usr/bin/setpriv\n"),
    "su-alp-setpriv": (ALPINE, "RUN apk add --no-cache setpriv\n"),
    "su-alp-bb": (ALPINE, ""),
    "su-alp-nosp": (ALPINE, "RUN rm -f /bin/setpriv /usr/bin/setpriv\n"),
    "su-alp-shadow": (
        ALPINE,
        "RUN apk add --no-cache shadow-login && rm -f /bin/setpriv /usr/bin/setpriv\n",
    ),
}
REFERENCE = {
    "su-deb-bounded": "su-deb-setpriv",
    "su-alp-bounded": "su-alp-setpriv",
    "su-alp-shadow-bounded": "su-alp-setpriv",
    "su-deb-nosp": "su-deb-setpriv",
    "su-alp-bb": "su-alp-setpriv",
    "su-alp-nosp": "su-alp-setpriv",
    "su-alp-shadow": "su-alp-setpriv",
}
# What the setpriv path itself sets; the su path may differ only there.
SET_BY_THE_DROP = {"HOME", "USER", "LOGNAME", "SHELL", "PATH"}
AGENT_ENV = ["--agent-env", "BF_SU_MARK=1", "--agent-env", "BASH_ENV=/etc/bf-bash-env"]
UNSUPPORTED = {"su-deb-nosp", "su-alp-bb", "su-alp-nosp", "su-alp-shadow"}
# The images whose launch drops through su (BOUNDED_SU).
BOUNDED = {name for name in FIXTURES if name.endswith("-bounded")}
# Their one refusal, word for word: the metadata block cannot verify the
# agent's launch (no util-linux setpriv), so no agent code runs.
UNSUPPORTED_REFUSAL = (
    "CloudMetadataPowerError: sandbox startup refused: the agent privilege "
    "construction could not be verified from /proc as root. The shared launch "
    "must set NoNewPrivs, remove NET_ADMIN and NET_RAW from CapBnd, and clear "
    "CapAmb. Provide util-linux setpriv and a readable /proc."
)
# The old start the suite used (main keeping NET_ADMIN), as #234 refuses it.
OLD_START_REFUSAL = (
    "RuntimeError: sandbox startup refused: the task's network policy (public) "
    "needs BenchFlow's helper container to add main's rules from outside it; "
    "BENCHFLOW_METADATA_MAIN_NET_ADMIN=1 would leave main holding NET_ADMIN, able "
    "to remove its own rules"
)


def materialize(root: Path, names=tuple(FIXTURES)) -> Path:
    tasks = root / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    for name in names:
        base, layer = FIXTURES[name]
        dest = h.materialize_task(h.TaskVariant(name, "env-probe"), tasks)
        dockerfile = dest / "environment" / "Dockerfile"
        dockerfile.write_text(base + COMMON + PROBE_LAYER + layer + "\nWORKDIR /app\n")
    return tasks


@pytest.fixture(scope="module")
def su_job(tmp_path_factory) -> h.CliRun:
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    if SANDBOX != "docker":
        pytest.skip("the fixtures' images are built on Docker")
    root = tmp_path_factory.mktemp("su-env")
    tasks = materialize(root)
    with (
        pytest.MonkeyPatch.context() as env,
        h.host_fake_llm(root / "host-fake-llm.jsonl") as url,
    ):
        # The helper start: main without NET_ADMIN, every rule from
        # BenchFlow's helper. Never the old start, which #234 refuses.
        env.delenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", raising=False)
        env.setenv("BENCHFLOW_METADATA_HELPER", "1")
        return h.run_bench(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=root / "jobs",
            sandbox=SANDBOX,
            host_fake_url=url,
            extra=["--concurrency", "6", "--retry-attempts", "0", *AGENT_ENV],
        )


@pytest.fixture(scope="module")
def old_start_job(tmp_path_factory) -> h.CliRun:
    """The suite's old setup (main keeping NET_ADMIN), on one supported image."""
    if SANDBOX is None:
        (pytest.fail if REQUESTED else pytest.skip)(SKIP_REASON)
    if SANDBOX != "docker":
        pytest.skip("the fixtures' images are built on Docker")
    root = tmp_path_factory.mktemp("su-env-old-start")
    tasks = materialize(root, ("su-deb-setpriv",))
    with (
        pytest.MonkeyPatch.context() as env,
        h.host_fake_llm(root / "host-fake-llm.jsonl") as url,
    ):
        env.setenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", "1")
        return h.run_bench(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=root / "jobs",
            sandbox=SANDBOX,
            host_fake_url=url,
            extra=["--retry-attempts", "0", *AGENT_ENV],
        )


def probe(trial: Path) -> dict[str, Any]:
    """What the agent's Bash call printed: names and markers, no values."""
    for event in h.normalized_acp(trial):
        if event.get("type") == "tool_call" and event.get("tool_name") == "Bash":
            out = str(event.get("raw_output") or "")
            lines = re.findall(r"(?m)^envprobe (.*)$", out)
            keys = next((ln[5:].split() for ln in lines if ln.startswith("keys ")), [])
            return {
                "agent": next((ln[6:] for ln in lines if ln.startswith("agent ")), ""),
                "keys": set(keys),
                "vars": dict(
                    ln[4:].split(" ", 1) for ln in lines if ln.startswith("var ")
                ),
                "path": next(
                    (ln[9:] for ln in lines if ln.startswith("path-sha ")), ""
                ),
                "uid": next((ln[4:] for ln in lines if ln.startswith("uid ")), ""),
                "rc": [ln for ln in lines if ln.startswith(("rc ", "bash_env "))],
                "calls": [ln for ln in lines if ln.startswith("setpriv-call ")],
            }
    return {}


def _result(trial: Path) -> dict[str, Any]:
    return h.read_json(trial / "result.json")


def _assert_refused(trial: Path) -> None:
    """S2-CONF-9 at 85b87bc9 supersedes SU's unrestricted image routes: the
    metadata block's own refusal, word for word (any refusal passed
    here before, the old start's included)."""
    result = _result(trial)
    error = str(result.get("error") or "")
    assert error == UNSUPPORTED_REFUSAL, result
    assert not result.get("rewards"), result
    assert not probe(trial)


def _assert_helper_start(trial: Path) -> None:
    """The trial started as every real run does: the metadata block's rules
    from BenchFlow's helper, main without NET_ADMIN, nothing for the agent to
    take, and no record saying "holds" before the rules were in."""
    metadata = h.read_json(trial / "cloud-metadata.json")
    assert metadata["state"] == "on", metadata
    assert metadata["rules_from"] == "helper", metadata
    assert metadata["main_net_admin"] is False, metadata
    power = h.read_json(trial / "firewall-power.json")
    assert power["state"] == "clear", power
    assert power.get("drop") is None, power
    assert power["installing"] == ["the cloud metadata block"], power
    assert "holds" not in power, power


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_the_agent_starts_with_its_env_and_finishes(su_job, name):
    trial = su_job.trial(name)
    if name in UNSUPPORTED:
        _assert_refused(trial)
        return
    result = _result(trial)
    error = str(result.get("error") or "")
    assert "Authentication required" not in error, error
    found = probe(trial)
    assert found.get("agent") == "found", (found, su_job.output[-3000:])
    assert found["vars"]["BF_SU_MARK"] == "present", found
    assert "ANTHROPIC_API_KEY" in found["keys"] or "ANTHROPIC_BASE_URL" in found["keys"]
    assert found["uid"] not in ("", "0"), found
    assert result["rewards"]["reward"] == 1.0, result
    _assert_helper_start(trial)


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_the_su_path_env_is_the_setpriv_paths(su_job, name):
    if name in UNSUPPORTED:
        _assert_refused(su_job.trial(name))
        return
    found = probe(su_job.trial(name))
    reference = probe(su_job.trial(REFERENCE[name]))
    assert found and reference, (found, reference)
    assert (found["keys"] ^ reference["keys"]) <= SET_BY_THE_DROP, sorted(
        found["keys"] ^ reference["keys"]
    )
    # Root's PATH reaches the agent on both paths (shadow's su resets it).
    assert found["path"] == reference["path"]


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_nothing_of_roots_own_and_no_rc_file_as_root(su_job, name):
    if name in UNSUPPORTED:
        _assert_refused(su_job.trial(name))
        return
    found = probe(su_job.trial(name))
    assert found, su_job.output[-3000:]
    assert found["vars"]["BF_ROOT_ONLY"] == "absent", found
    assert found["vars"]["BENCHFLOW_SU_PATH"] == "absent", found
    assert found["vars"]["BENCHFLOW_SU_NAMES"] == "absent", found
    # No variable the agent starts with names BenchFlow: BenchFlow's own
    # arrive under RUNTIME_ names, and every carrier is gone.
    assert not [k for k in found["keys"] if "BENCHFLOW" in k], found
    assert any(k.startswith("RUNTIME_") for k in found["keys"]), found
    # BASH_ENV ran, as the agent; nothing of the agent's ran as root.
    assert any(ln.startswith("bash_env uid=") for ln in found["rc"]), found
    assert not [ln for ln in found["rc"] if ln.endswith(" uid=0")], found


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_the_bounded_images_drop_through_su(su_job, name):
    """Each "-bounded" image's launch takes the su branch: its setpriv on the
    launch's PATH logged that branch's calls (restrictions, then su). The
    wrapper sat at /usr/local/bin before, which the launch's pinned PATH
    never reads, so no image reached su and breaking su went unnoticed
    (R647). No other image has the wrapper, so none logs a call."""
    if name in UNSUPPORTED:
        _assert_refused(su_job.trial(name))
        return
    found = probe(su_job.trial(name))
    assert found, su_job.output[-3000:]
    if name in BOUNDED:
        assert found["calls"], found
        assert set(found["calls"]) == {"setpriv-call no-reuid"}, found
    else:
        assert found["calls"] == [], found


def test_the_helper_start_leaves_no_firewall_power_to_drop(su_job):
    """Was test_the_drop_is_unchanged (S3's drop: util-linux setpriv drops
    NET_ADMIN), which needed the old start. Under the helper start main holds
    no NET_ADMIN, so S3's check is clear on every image and drops nothing; the
    supported images score, and the unsupported ones meet the metadata
    block's own refusal."""
    records = {
        name: h.read_json(su_job.trial(name) / "firewall-power.json")
        for name in FIXTURES
    }
    for name, record in records.items():
        assert record["state"] == "clear", (name, record)
        assert record.get("drop") is None, (name, record)
        assert record["installing"] == ["the cloud metadata block"], (name, record)
        assert "holds" not in record, (name, record)
    for name in sorted(set(FIXTURES) - UNSUPPORTED):
        assert _result(su_job.trial(name))["rewards"]["reward"] == 1.0, name
        _assert_helper_start(su_job.trial(name))
    for name in UNSUPPORTED:
        _assert_refused(su_job.trial(name))


def test_the_old_in_main_start_is_refused(old_start_job):
    """The suite's old setup, BENCHFLOW_METADATA_MAIN_NET_ADMIN=1, keeps main's
    NET_ADMIN, which root there could use to remove its own rules: #234
    refuses it before any agent code, with its own words."""
    trial = old_start_job.trial("su-deb-setpriv")
    result = _result(trial)
    assert result.get("error") == OLD_START_REFUSAL, result
    assert not result.get("rewards"), result
    assert not probe(trial)
    assert not (trial / "firewall-power.json").exists()
