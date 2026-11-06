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
COPY --from=node /usr/local/bin/node /opt/benchflow/node/bin/node
COPY --from=node /usr/local/lib/node_modules /opt/benchflow/node/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /opt/benchflow/node/bin/npm \\
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /opt/benchflow/node/bin/npx
"""

COMMON = """
COPY fake_llm /opt/fake_llm
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
"""

# Exercise the remaining identity-only su fallback under S2's full barrier.
# The real util-linux executable still applies every requested restriction.
BOUNDED_SU = """
RUN printf '#!/bin/sh\\nif [ "$1" = --help ]; then /usr/bin/setpriv --help | sed /reuid/d; else exec /usr/bin/setpriv "$@"; fi\\n' > /usr/local/bin/setpriv \\
    && chmod 755 /usr/local/bin/setpriv
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
        # PERF-A b21700ed3 otherwise bypasses the drop this suite guards.
        env.setenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", "1")
        return h.run_bench(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=root / "jobs",
            sandbox=SANDBOX,
            host_fake_url=url,
            extra=["--concurrency", "6", "--retry-attempts", "0", *AGENT_ENV],
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
            }
    return {}


def _result(trial: Path) -> dict[str, Any]:
    return h.read_json(trial / "result.json")


def _assert_refused(trial: Path) -> None:
    """S2-CONF-9 at 85b87bc9 supersedes SU's unrestricted image routes."""
    result = _result(trial)
    error = str(result.get("error") or "")
    assert (
        "sandbox startup refused" in error or "requires util-linux setpriv" in error
    ), result
    assert "Authentication required" not in error
    assert not probe(trial)


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
    # BASH_ENV ran, as the agent; nothing of the agent's ran as root.
    assert any(ln.startswith("bash_env uid=") for ln in found["rc"]), found
    assert not [ln for ln in found["rc"] if ln.endswith(" uid=0")], found


def test_the_drop_is_unchanged(su_job):
    """S3's drop: util-linux setpriv drops NET_ADMIN; BusyBox's setpriv sets
    no_new_privs before su; without setpriv nothing found, nothing dropped."""
    records = {
        name: h.read_json(su_job.trial(name) / "firewall-power.json")
        for name in FIXTURES
    }
    for name in ("su-deb-setpriv", "su-alp-setpriv"):
        assert records[name]["state"] == "disarmed", records[name]
        assert records[name]["drop"] == "setpriv", records[name]
    for name in UNSUPPORTED:
        _assert_refused(su_job.trial(name))
        assert records[name]["state"] == "refused", records[name]
        assert records[name]["drop"] is None, records[name]
