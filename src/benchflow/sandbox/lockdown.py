"""Sandbox user setup, path lockdown, and verifier hardening.

Owns the "agent runs as non-root" lifecycle:
    - Creating the sandbox user and preparing minimal home state it needs
    - Building the privilege-drop wrapper (setpriv / su) for agent launch
    - Locking down solution/test paths so the sandbox user cannot read them
    - Hardening the environment before the verifier runs

Does not own:
    - Spawning the agent process — see _acp_run.py
    - Running the verifier itself — see SDK._verify
"""

import ast
import contextlib
import contextvars
import dataclasses
import hashlib
import hmac
import io
import ipaddress
import itertools
import json as _json
import logging
import os
import posixpath
import re
import secrets
import shlex
import string
import tokenize
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from benchflow._bfx.client import BfxError, BfxNotStarted, BfxProtocolError
from benchflow._bfx.locate import BfxUnavailable
from benchflow._image_python_paths import GUARD_ID_SH as IMAGE_PYTHON_GUARD_ID_SH
from benchflow._image_python_paths import (
    GUARD_SEARCH_RULE as IMAGE_PYTHON_GUARD_SEARCH_RULE,
)
from benchflow._image_python_paths import HOOK_RULE as IMAGE_PYTHON_HOOK_RULE
from benchflow._image_python_paths import IMAGE_PYTHON_GUARD as IMAGE_PYTHON_GUARD_FILE
from benchflow._image_python_paths import IMAGE_PYTHON_PATHS as IMAGE_PYTHON_PATHS_FILE
from benchflow._image_python_paths import (
    IMAGE_PYTHON_PATHS_EACH as IMAGE_PYTHON_PATHS_EACH_FILE,
)
from benchflow._image_python_paths import READ as READ_IMAGE_PYTHON_PATHS
from benchflow._image_python_paths import SEARCH_VALUES as IMAGE_PYTHON_SEARCH_VALUES
from benchflow._root_python import root_python_command, root_python_sh
from benchflow._runtime_tree import (
    KIT_BIN,
    KIT_PYTHON,
    KIT_RECORD,
    RUN_DIR,
    VERIFIER_DIR,
)
from benchflow._utils import kill_record as _kill_record
from benchflow._utils import timing_spans as _timing_spans
from benchflow._utils.trusted_path import (
    PATH_FIELD_ENCODE_SH,
    PATH_QUERY_CMD,
    PYTHONPATH_QUERY_CMD,
    TrustedPathReport,
    VerifierPathEvidenceError,
    decode_path_field,
    report_trusted_path,
)
from benchflow._verifier_env import PIP_CONFIG_FIX, pin_verifier_env
from benchflow.agents.registry import get_sandbox_home_dirs
from benchflow.sandbox import _hardening_python, _verifier_tool_state
from benchflow.sandbox._cache_reclaim import build_reclaim_caches_cmd
from benchflow.sandbox._dependency_options import (
    UnknownDependencyInputs,
    dependency_inputs,
    python_invocation,
    registry_requirement,
)
from benchflow.sandbox._exec_merge import (
    UID_COMMAND,
    MergedStep,
    StepAnswer,
    kept_default_identity,
    merged_script,
    new_tag,
    parse_merged,
)
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw
from benchflow.sandbox._exec_receipt import ReceiptFailure, receipt_exec
from benchflow.sandbox._hardening_python import (
    RUNTIME_PATH_PREFIXES,
    TRUSTED_SYSTEM_PATH,
    hardening_python_cmd,
)
from benchflow.sandbox._hook_cleanup import ROOT_MADE_EXCLUDED
from benchflow.sandbox._identities import Identities, kill_prelude
from benchflow.sandbox._identities import resolve as resolve_identity
from benchflow.sandbox._in_order import in_order, runs_steps_at_once
from benchflow.sandbox._locked_path_validation import _validate_locked_path
from benchflow.sandbox._mount_aliases import PathExclusions, blocked_mount_aliases
from benchflow.sandbox._task_route import as_task_command
from benchflow.sandbox.cloud_metadata import (
    CLOUD_METADATA_OPT_OUT_HINT,
    CLOUD_METADATA_V4,
    CLOUD_METADATA_V6,
    NOT_REACHABLE,
    PROVIDER_MANAGED,
    CloudMetadataPowerError,
    cloud_metadata_list,
    cloud_metadata_record,
)
from benchflow.sandbox.protocol import KeptServiceFreezerUnavailable

if TYPE_CHECKING:
    from benchflow.sandbox._root_launch import AgentLaunch
    from benchflow.task import Task
    from benchflow.task.keep_services import KeptServices

logger = logging.getLogger(__name__)


def _system_probe(script: str, *args: str, user: str | int | None = "root") -> str:
    """Use the image PATH only as data, never to find a probe's tools.

    The prefix consists only of shell builtins. Backend exec shells and their
    env-file bootstrap must also use trusted paths before reaching this point.
    *user* is the exec's user (:func:`_on_path`).
    """
    return _on_path(
        f"sh -c {shlex.quote(script)} " + " ".join(shlex.quote(arg) for arg in args),
        user=user,
    )


def _on_path(
    command: str, path: str | None = None, *, user: str | int | None = "root"
) -> str:
    """Run harness probes through the backend's shared system route.

    ``path`` is the verifier's judged task PATH, retained for callers from
    V3's earlier probes. It never selects root's runtime: PT owns that route.
    A root exec on the kit route keeps the kit's PATH, as every root step
    does (:func:`_keep_route_path`, batch 32b review A, M-1). An exec as a
    user named other than root carries that user's own PATH, so the host
    pins :data:`SYSTEM_PATH` for it outright (round 2, review A, M-1).
    """
    from benchflow.sandbox._base import is_root_user

    if user is not None and not is_root_user(user):
        from benchflow.sandbox._base import SYSTEM_PATH, system_shell_prefix

        return f"{system_shell_prefix(SYSTEM_PATH)}\n{command}"
    return _keep_route_path(command)


def _keep_route_path(command: str) -> str:
    """*command* with no loader or ``PYTHON*`` value, on root's PATH.

    On the kit route (C8), the host already set PATH for a root exec: the
    kit's bin first, then only the folders PT's proof found root's alone.
    That PATH is kept, as integ's preparation prelude kept it
    (PT-C8-R3-3); only the loader and ``PYTHON*`` names are cleared again.
    Every other exec pins :data:`SYSTEM_PATH`, as V3's system route did
    a root exec without a kit, and an exec as any other user,
    whose route carries that user's own PATH (an image PATH the agent may
    write, batch 32b round-2 review A, M-1). Which one is read from the PATH
    the host's route set, not from a file in the sandbox (round 2, L-1).
    """
    from benchflow._runtime_tree import KIT_BIN
    from benchflow.sandbox._base import SYSTEM_PATH, system_shell_prefix

    pinned = system_shell_prefix(SYSTEM_PATH)
    kept = pinned.replace(f"PATH={SYSTEM_PATH}; export PATH; ", "")
    if kept == pinned:  # pragma: no cover - the prefix's shape changed
        raise RuntimeError("system_shell_prefix no longer pins PATH as expected")
    kit = shlex.quote(KIT_BIN)
    return f'case "$PATH" in {kit}|{kit}:*) {kept};; *) {pinned};; esac\n{command}'


# The hook clean-up's search-path inputs (U5a-CONF-5), kept as inert shell
# data before the root route unsets PYTHONPATH and PYTHONHOME.
_HOOK_INPUTS = (
    # On PT's plumbing route the task's values arrive only as data
    # (``_BENCHFLOW_IMAGE_<name>``, sandbox._base.plumbing_env).
    "__bf_hook_pythonpath=${_BENCHFLOW_IMAGE_PYTHONPATH-${PYTHONPATH-}}; "
    "__bf_hook_pythonhome=${_BENCHFLOW_IMAGE_PYTHONHOME-${PYTHONHOME-}}; "
)


# The interpreter root's Python steps run during verifier preparation: the
# one the host judged before the first of them, or None.
_root_python: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "benchflow_root_python", default=None
)

# What hardening knows of this trial that two of its rules need (ACC-RUN2),
# set by harden_before_verify: whether the agent ran as a user other than
# root (the start-up hook clean-up's root-made rule), and the workspace (the
# verifier PATH's workspace-tool rule). Unset, both rules are as before.
_hook_agent_user: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "benchflow_hook_agent_user", default=False
)
_path_workspace: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "benchflow_path_workspace", default=None
)


def _prep_on_path(command: str) -> str:
    """*command* on :func:`_on_path`'s route, the clean-up's inputs saved first.

    Every verifier-preparation step runs as root on this one route, step by
    step or in U5a's one script. Wrap a command once: a second
    wrap would save the inputs the first one already cleared. The route also
    names the judged interpreter (``$__bf_root_python``, empty when none
    passed): the steps' Python runs only that file.
    """
    python = shlex.quote(_root_python.get() or "")
    return f"__bf_root_python={python}; " + _HOOK_INPUTS + _keep_route_path(command)


def _select_verifier_python(path: str) -> str:
    """Select a literal interpreter from the already judged PATH using builtins.

    Never export this PATH: shell utilities and receipts still use the shared
    system route. The selected filename is also the presence probe's identity,
    so a broken interpreter cannot be mistaken for a missing system Python.
    """
    folders = path.split(":")
    if any(not folder.startswith("/") for folder in folders):
        raise ValueError("verifier Python selection requires an absolute judged PATH")
    candidates = " ".join(
        shlex.quote(folder.rstrip("/") + "/python3") for folder in folders
    )
    return (
        "bf_python=; "
        f"for bf_candidate in {candidates}; do "
        'if [ -f "$bf_candidate" ] && [ -x "$bf_candidate" ]; then '
        "bf_python=$bf_candidate; break; fi; done; "
    )


# Path lockdown defaults and validation

# /testbed_verify is the root-owned pre-agent workspace snapshot seeded by
# _seed_verifier_workspace, which makes it world-readable (chmod -R o+rX) so the
# verifier can diff against it. That readability leaks grading-side state to the
# agent: an agent can read the snapshot's verifier/judge config (rubrics,
# expected outputs, judge credentials) and forge or reverse-engineer a reward.
# Seeding runs before lockdown in install_agent, so locking it here (chmod 700,
# root-owned) closes the read window before the agent starts. The verifier runs
# as root and so still reads /testbed_verify regardless of mode.
_DEFAULT_LOCKED = ["/oracle", "/solution", "/verifier", "/tests", "/testbed_verify"]
# The same pre-agent copy, named for the plugin guard.
_VERIFIER_WORKSPACE_COPY = "/testbed_verify"


def _resolve_locked_paths(
    sandbox_user: str | None,
    sandbox_locked_paths: list[str] | None,
) -> list[str]:
    """Resolve effective locked paths.

    - sandbox_user=None → [] (no lockdown)
    - sandbox_user set, paths=None → defaults (/oracle, /solution, /verifier,
      /tests, /testbed_verify)
    - sandbox_user set, paths=[] → [] (explicit opt-out)
    - sandbox_user set, paths=[...] → union of defaults + caller paths
    """
    if not sandbox_user:
        if sandbox_locked_paths:
            raise ValueError("sandbox_locked_paths requires sandbox_user")
        return []
    if sandbox_locked_paths is None:
        return list(_DEFAULT_LOCKED)
    if not sandbox_locked_paths:
        return []  # explicit opt-out
    return list(dict.fromkeys(_DEFAULT_LOCKED + sandbox_locked_paths))


# Sandbox user + privilege drop


def _allowlist_firewall_cmd(allow_networks: tuple[str, ...]) -> str:
    """Allow-mode additions to the uid firewall (``$agent_uid`` already set).

    DNS from the agent uid (UDP and TCP port 53, any destination) is
    redirected to the proxy's DNS filter, which answers only listed names.
    A kernel without the nat table fails closed (exit 86) instead of leaving
    DNS open unnoticed. Docker's embedded resolver (127.0.0.11, reached on a
    rewritten port) is refused outright. Listed IP and CIDR entries are
    reachable directly, for any protocol.
    """
    parsed = [ipaddress.ip_network(n, strict=False) for n in allow_networks]
    uid = '-m owner --uid-owner "$agent_uid"'
    parts = [
        "iptables -t nat -L OUTPUT -n >/dev/null 2>&1 || "
        "{ echo 'iptables nat table unavailable: network_mode=allowlist cannot "
        "filter DNS' >&2; exit 86; }; "
    ]
    for proto in ("udp", "tcp"):
        redirect = f"OUTPUT -p {proto} --dport 53 {uid} -j REDIRECT --to-ports {EGRESS_DNS_PORT}"
        parts.append(
            f"iptables -t nat -C {redirect} 2>/dev/null || "
            f"iptables -t nat -I {redirect.replace('OUTPUT', 'OUTPUT 1', 1)} || "
            "{ echo 'cannot install the allowlist DNS redirect' >&2; exit 86; }; "
        )
        reject = f"OUTPUT -o lo -p {proto} --dport 53 {uid} -j REJECT"
        parts.append(
            f"iptables -C {reject} 2>/dev/null || "
            f"iptables -I {reject.replace('OUTPUT', 'OUTPUT 1', 1)}; "
        )
        # After the REDIRECT the filter chain does not see the query as
        # "-o lo" (observed on Daytona), so admit the rewritten destination.
        admit = (
            f"OUTPUT -d 127.0.0.1 -p {proto} --dport {EGRESS_DNS_PORT} {uid} -j ACCEPT"
        )
        parts.append(
            f"iptables -C {admit} 2>/dev/null || "
            f"iptables -I {admit.replace('OUTPUT', 'OUTPUT 1', 1)}; "
        )
    docker_dns = f"OUTPUT -d 127.0.0.11 {uid} -j REJECT"
    parts.append(
        f"iptables -C {docker_dns} 2>/dev/null || "
        f"iptables -I {docker_dns.replace('OUTPUT', 'OUTPUT 1', 1)}; "
    )
    for net in parsed:
        if net.version != 4:
            continue
        rule = f"OUTPUT -d {net.compressed} {uid} -j ACCEPT"
        parts.append(
            f"iptables -C {rule} 2>/dev/null || "
            f"iptables -I {rule.replace('OUTPUT', 'OUTPUT 1', 1)}; "
        )
    v6 = []
    for proto in ("udp", "tcp"):
        reject = f"OUTPUT -o lo -p {proto} --dport 53 {uid} -j REJECT"
        v6.append(
            f"ip6tables -C {reject} 2>/dev/null || "
            f"ip6tables -I {reject.replace('OUTPUT', 'OUTPUT 1', 1)}; "
        )
    for net in parsed:
        if net.version != 6:
            continue
        rule = f"OUTPUT -d {net.compressed} {uid} -j ACCEPT"
        v6.append(
            f"ip6tables -C {rule} 2>/dev/null || "
            f"ip6tables -I {rule.replace('OUTPUT', 'OUTPUT 1', 1)}; "
        )
    parts.append("if [ -e /proc/net/if_inet6 ]; then " + "".join(v6) + "fi")
    return "".join(parts)


def _agent_egress_firewall_cmd(
    sandbox_user: str,
    *,
    allow_networks: tuple[str, ...] | None = None,
    root_gid: int | None = None,
    refuse_dns: bool = False,
) -> str:
    """The uid firewall: the agent's packets leave only on loopback.

    ``root_gid``: the agent runs as uid 0 with that primary group
    (:func:`build_root_no_web_launch`), and the rules match uid 0 *and* that
    group, so BenchFlow's own root processes (gid 0) are not held.

    ``refuse_dns`` (an offline agent): DNS on loopback is refused too. A
    resolver there answers from outside the sandbox (Docker's embedded one,
    127.0.0.11 on a rewritten port, forwards to the host's), so it would
    resolve any name and carry data out in the names asked."""
    user = shlex.quote(sandbox_user)
    owner = (
        f"-m owner --uid-owner 0 --gid-owner {int(root_gid)}"
        if root_gid is not None
        else '-m owner --uid-owner "$agent_uid"'
    )
    allow = (
        "; " + _allowlist_firewall_cmd(allow_networks)
        if allow_networks is not None
        else ""
    )
    dns4 = dns6 = ""
    if refuse_dns:
        # Above the loopback ACCEPT (each goes in at the top).
        rules4 = [
            f"OUTPUT -d 127.0.0.11 {owner} -j REJECT",
            *(
                f"OUTPUT -o lo -p {p} --dport 53 {owner} -j REJECT"
                for p in ("udp", "tcp")
            ),
        ]
        dns4 = "".join(
            f"iptables -C {r} 2>/dev/null || "
            f"iptables -I {r.replace('OUTPUT', 'OUTPUT 1', 1)}; "
            for r in rules4
        )
        dns6 = "".join(
            f"ip6tables -C {r} 2>/dev/null || "
            f"ip6tables -I {r.replace('OUTPUT', 'OUTPUT 1', 1)}; "
            for r in rules4[1:]
        )
    return (
        "set -e; "
        "if ! command -v iptables >/dev/null 2>&1; then "
        "if command -v apt-get >/dev/null 2>&1; then "
        "export DEBIAN_FRONTEND=noninteractive; "
        "apt-get update -qq && apt-get install -y -qq iptables >/dev/null; "
        "elif command -v dnf >/dev/null 2>&1; then "
        "dnf -y install iptables >/dev/null; "
        "elif command -v apk >/dev/null 2>&1; then "
        "apk add --no-cache iptables >/dev/null; "
        "else echo 'No supported iptables package manager' >&2; exit 86; fi; fi; "
        + (
            "agent_uid=0; "
            if root_gid is not None
            else f"agent_uid=$(id -u {user}) || exit 86; "
        )
        + f"iptables -C OUTPUT -o lo {owner} "
        "-j ACCEPT 2>/dev/null || "
        f"iptables -I OUTPUT 1 -o lo {owner} "
        "-j ACCEPT; "
        f"iptables -C OUTPUT {owner} "
        "-j REJECT 2>/dev/null || "
        f"iptables -A OUTPUT {owner} -j REJECT; "
        + dns4
        # procfs reports zero stat size even when this interface has content.
        # Install IPv6 rules whenever the kernel exposes the stack, including
        # before a non-loopback interface acquires an address.
        + "if [ -e /proc/net/if_inet6 ]; then "
        "command -v ip6tables >/dev/null 2>&1 || "
        "{ echo 'IPv6 enabled but ip6tables unavailable' >&2; exit 86; }; "
        f"ip6tables -C OUTPUT -o lo {owner} "
        "-j ACCEPT 2>/dev/null || "
        f"ip6tables -I OUTPUT 1 -o lo {owner} "
        "-j ACCEPT; "
        f"ip6tables -C OUTPUT {owner} "
        "-j REJECT 2>/dev/null || "
        f"ip6tables -A OUTPUT {owner} -j REJECT; " + dns6 + f"fi{allow}"
    )


#: Ports Daytona's sandbox daemon (process 1, running as root) answers on
#: without credentials; its toolbox API runs commands as root. An agent user
#: that reaches it reaches root (seen live: an agent installed packages as
#: root through ``POST /process/session`` on 127.0.0.1:2280).
SANDBOX_DAEMON_PORTS = (2280, 22220, 22222, 33333)

# Prints the TCP ports process 1 listens on, one per line, when process 1 is
# Daytona's daemon (its program's name says so); nothing otherwise, so a
# task whose own server is process 1 keeps it. POSIX sh (dash included).
# Root only reads another process's fds, so run it as root.
_DAEMON_PORTS_SH = r"""bf_daemon_ports() {
  exe=$(readlink "$P/1/exe" 2>/dev/null) || return 0
  case "${exe##*/}" in *daytona*) ;; *) return 0;; esac
  inodes=" "
  for f in "$P"/1/fd/*; do
    l=$(readlink "$f" 2>/dev/null) || continue
    case $l in socket:\[*\]) inodes="$inodes$(printf '%s' "$l" | tr -cd 0-9) ";; esac
  done
  for f in "$P/net/tcp" "$P/net/tcp6"; do
    [ -r "$f" ] || continue
    awk -v want="$inodes" 'FNR > 1 && $4 == "0A" && index(want, " " $10 " ") { n = split($2, a, ":"); print a[n] }' "$f"
  done | while read -r hex; do
    case $hex in *[!0-9A-Fa-f]*|'') continue;; esac
    printf '%d\n' "0x$hex"
  done | sort -un
}
"""


class SandboxDaemonGuardError(RuntimeError):
    """The agent user could not be kept away from Daytona's root daemon.

    The message starts with "sandbox startup refused", so the trial is
    classified as a sandbox-setup failure (the task image's, see the
    message) rather than run with an agent that can reach root.
    ``progress``: what the guard logged (its steps and the package
    manager's output), kept in the trial folder as ``daemon-guard.log``.
    """

    def __init__(self, message: str, *, progress: str = "") -> None:
        super().__init__(message)
        self.progress = progress


#: Where the daemon guard logs its steps as they happen: root's private
#: folder (made 700 by the protected runtime's build, before the guard).
DAEMON_GUARD_LOG = f"{RUN_DIR}/daemon-guard.log"
#: The guard's time limit, in the sandbox (its ``timeout``) and here.
DAEMON_GUARD_TIMEOUT_SEC = 180


def _daemon_guard_cmd(
    sandbox_user: str, proc: str = "/proc", log: str = DAEMON_GUARD_LOG
) -> str:
    """Drop every non-root uid's TCP connections to the sandbox daemon's
    ports, on every interface (``-o lo`` is not relied on), and nothing else
    (only the daemon and BenchFlow's own root processes use them, so a
    service user the agent drives cannot relay to the daemon). The rules sit in the mangle table's OUTPUT chain, which the kernel
    walks before the filter table's, so no ACCEPT there (the egress
    firewall's loopback ACCEPT) can let a packet past them, not even for a
    moment; running it again adds nothing. Uses ``iptables`` or, where that
    does not work on the kernel, ``iptables-legacy`` (and the same for
    ip6tables). Prints ``BF daemon-guard ports=... status=N``: 0 enforced,
    1 a rule failed, 2 no iptables and no package manager to install it,
    3 IPv6 without ip6tables, 4 no such user, 5 installing iptables failed
    (then ``BF daemon-guard install: <its last lines>``).

    Each step, with the seconds since the start, and the package manager's
    output go to *log* as they happen, so a guard its timeout stopped still
    says where it was (DFR-p3-4); no log where its folder does not exist.
    """
    user = shlex.quote(sandbox_user)
    floor = " ".join(str(p) for p in SANDBOX_DAEMON_PORTS)
    p = shlex.quote(proc)
    folder = shlex.quote(posixpath.dirname(log))
    return (
        "( "
        f"L={shlex.quote(log)}; "
        f'{{ [ -d {folder} ] && [ ! -L {folder} ] && : > "$L"; }} 2>/dev/null || L=/dev/null; '
        "T0=$(date +%s); "
        'bf_at() { printf \'+%ss %s\\n\' "$(( $(date +%s) - T0 ))" "$*" >> "$L"; }; '
        "bf_at start; "
        f"agent_uid=$(id -u {user} 2>/dev/null) || {{ echo 'BF daemon-guard ports= status=4'; exit 0; }}; "
        f"P={p}; "
        f"{_DAEMON_PORTS_SH}"
        f"ports=$(printf '%s\\n' {floor} $(bf_daemon_ports) | sort -un | tr '\\n' ' '); "
        'bf_at "ports $ports"; '
        'pick() { for c in "$1" "$1-legacy"; do '
        'command -v "$c" >/dev/null 2>&1 && "$c" -w -t mangle -S OUTPUT >/dev/null 2>&1 && '
        '{ echo "$c"; return 0; }; done; return 1; }; '
        "guard() { for p in $ports; do "
        # Every non-root uid: only the daemon and BenchFlow's own root
        # processes use these ports, so a service user the agent drives
        # (postgres, www-data) cannot relay to the daemon either.
        'r="OUTPUT -p tcp --dport $p -m owner ! --uid-owner 0 -j DROP"; '
        '"$1" -w -t mangle -C $r 2>/dev/null || '
        '"$1" -w -t mangle -I $r || return 1; done; }; '
        "status=0; out=; "
        f"need6=; [ -e {p}/net/if_inet6 ] && ! command -v ip6tables >/dev/null 2>&1 && need6=1; "
        'if ! command -v iptables >/dev/null 2>&1 || [ -n "$need6" ]; then '
        # The install's output reaches the log as it is printed (tee).
        "if command -v apt-get >/dev/null 2>&1; then "
        "out=$( { bf_at 'install iptables: apt-get install'; "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -q iptables || "
        "{ bf_at 'install iptables: apt-get update'; apt-get update -q && "
        "bf_at 'install iptables: apt-get install' && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -q iptables; }; } 2>&1 "
        '| tee -a "$L" ); '
        "elif command -v dnf >/dev/null 2>&1; then bf_at 'install iptables: dnf'; "
        'out=$(dnf -y install iptables 2>&1 | tee -a "$L"); '
        "elif command -v apk >/dev/null 2>&1; then bf_at 'install iptables: apk'; "
        'out=$(apk add --no-cache iptables ip6tables 2>&1 | tee -a "$L"); '
        "elif ! command -v iptables >/dev/null 2>&1; then status=2; fi; "
        "if [ $status = 0 ] && ! command -v iptables >/dev/null 2>&1; then status=5; "
        """echo "BF daemon-guard install: $(printf '%s' "$out" | tail -n 3 | tr '\\n' ' ' | cut -c1-300)"; fi; fi; """
        "if [ $status = 0 ]; then "
        "bf_at 'rules: iptables'; "
        'fw4=$(pick iptables) && { guard "$fw4" || status=1; } || status=1; '
        f"if [ -e {p}/net/if_inet6 ]; then "
        "if command -v ip6tables >/dev/null 2>&1; then "
        "bf_at 'rules: ip6tables'; "
        'fw6=$(pick ip6tables) && { guard "$fw6" || status=1; } || status=1; '
        "elif [ $status = 0 ]; then status=3; fi; fi; fi; "
        'bf_at "done status=$status"; '
        'echo "BF daemon-guard ports=$ports status=$status" '
        ")"
    )


def _guard_needed_here(env: Any) -> bool:
    """Whether the agent user shares a network namespace with Daytona's root
    daemon: a single-container Daytona sandbox. (A compose task on Daytona
    is guarded on its VM when it starts; other backends have no such
    daemon, and get no iptables install either.)"""
    return getattr(env, "sandbox_daemon_guard", None) == "container"


async def guard_sandbox_daemon(env: Any, sandbox_user: str) -> None:
    """Install the daemon guard for ``sandbox_user`` where it is needed,
    or raise :class:`SandboxDaemonGuardError`: an agent that can reach the
    daemon can rewrite its own grade, so the run is refused, never run
    unguarded."""
    if not _guard_needed_here(env):
        return
    with _timing_spans.span(_timing_spans.DAEMON_GUARD):
        result = await _read_exec(
            env,
            _daemon_guard_cmd(sandbox_user),
            user="root",
            timeout_sec=DAEMON_GUARD_TIMEOUT_SEC,
        )
    out = getattr(result, "stdout", "") or ""
    line = next((ln for ln in out.splitlines() if "daemon-guard ports" in ln), "")
    if _exec_return_code(result) == 0 and re.search(r"\bstatus=0\b", line):
        return
    # What the guard logged as it went: its last step names where a guard
    # its timeout stopped was (DFR-p3-4).
    progress = ""
    try:
        read = await _read_exec(
            env, f"cat {shlex.quote(DAEMON_GUARD_LOG)}", user="root", timeout_sec=30
        )
        if _exec_return_code(read) == 0:
            progress = getattr(read, "stdout", "") or ""
    except Exception:
        # Diagnostics must not replace the guard's original setup failure.
        logger.debug("daemon guard log could not be read", exc_info=True)
    from benchflow._diagnostics import render_diagnostic
    from benchflow.cli._errors import _secrets

    # Receipt data decides enforcement; only the diagnostic copy is redacted.
    # Protect complete fields before line splitting can expose a secret fragment.
    m = re.search(r"\bstatus=(\d)", line)
    status = m.group(1) if m else ""
    timed_out = not line and _exec_return_code(result) == 124

    def message(fields: Sequence[str]) -> str:
        out, progress = fields
        line = next((ln for ln in out.splitlines() if "daemon-guard ports" in ln), "")
        steps = [ln for ln in progress.splitlines() if re.match(r"\+\d+s ", ln)]
        install = next(
            (ln for ln in out.splitlines() if "daemon-guard install:" in ln), ""
        )
        image = (
            "The task's image needs iptables (or apt-get, dnf or apk to install "
            "it), and ip6tables when IPv6 is on; or run the agent as root (no "
            "sandbox user)."
        )
        reason, remedy = {
            "1": (
                "an iptables rule could not be added (no owner match or mangle "
                "table on this kernel, or no permission)",
                image,
            ),
            "2": (
                "the image has no iptables and no package manager (apt-get, dnf "
                "or apk) to install it",
                image,
            ),
            "3": ("the image has IPv6 but no ip6tables", image),
            "4": (f"the user {sandbox_user!r} does not exist", image),
            "5": (
                "installing iptables failed"
                + (f" ({install.split('install:', 1)[1].strip()})" if install else ""),
                "This may pass on a retry (a mirror or a lock); for a lasting fix, "
                "put iptables in the task's image.",
            ),
        }.get(status, ("the guard did not report", image))
        if timed_out:
            # The sandbox's timeout stopped it: say where it was.
            reason = f"the guard did not finish in {DAEMON_GUARD_TIMEOUT_SEC} s" + (
                f" (its last step: {steps[-1]})" if steps else ""
            )
            remedy = (
                "This may pass on a retry (a slow package mirror or network); for "
                "a lasting fix, put iptables (and ip6tables) in the task's image, "
                "so the guard installs nothing."
            )
        return (
            "sandbox startup refused: the agent user could not be kept away from "
            f"Daytona's root toolbox daemon: {reason} ({line or 'no report'}). "
            "Running anyway would let the agent reach root and rewrite its own "
            f"grade. {remedy}"
        )

    raise SandboxDaemonGuardError(
        render_diagnostic((out, progress), _secrets(), message)
        + _exec_failure_detail(result),
        progress=progress,
    )


#: Carry root's PATH and the names of root's variables across ``su`` on the
#: path without util-linux setpriv (``_su_launch``). The agent's shell uses
#: and unsets them before the launch, so the agent never sees them.
_SU_PATH_ENV = "BENCHFLOW_SU_PATH"
_SU_NAMES_ENV = "BENCHFLOW_SU_NAMES"
# The names of the variables in the environment, one per line.
_ENV_NAMES = "env | sed -n 's/^\\([A-Za-z_][A-Za-z0-9_]*\\)=.*/\\1/p'"
# Root's side, in front of ``exec [setpriv --no-new-privs --] su``: the shell
# su starts is bash, as on the setpriv path, or sh where the image has no
# bash (never root's $SHELL, which ``su -m`` would otherwise pick); root's
# PATH and variable names ride along in the environment.
_SU_EXEC = (
    "if [ -x /bin/bash ]; then bf_su_shell=/bin/bash; else bf_su_shell=/bin/sh; fi;"
    f" {_SU_NAMES_ENV}=\" $({_ENV_NAMES} | tr '\\n' ' ')\""
    f' {_SU_PATH_ENV}="$PATH" exec'
)
# The agent's side, first thing after su: drop every variable root did not
# have (pam_mail's MAIL, pam_env's additions) other than those the agent's
# own shell keeps (as ``bash -c`` does on the setpriv path), put root's PATH
# back (shadow's su and pam_env reset it), and forget the two carriers.
_SU_RESTORE = (
    f'bf_p=${_SU_PATH_ENV} bf_n="${{{_SU_NAMES_ENV}}}SHLVL PWD OLDPWD _ ";'
    f" for bf_v in $({_ENV_NAMES}); do"
    ' case $bf_n in *" $bf_v "*) ;; *) unset "$bf_v";; esac; done;'
    ' export PATH="$bf_p"; unset bf_p bf_n bf_v;'
)


def _su_launch(sandbox_user: str, inner: str, *, su_command: str = "su") -> str:
    """``su`` starting ``inner`` as ``sandbox_user`` with the environment the
    setpriv path gives it (PY8 SU); the caller puts ``_SU_EXEC`` in front.

    This was ``su -l``, whose login shell clears the environment: the agent
    env the root shell had just sourced (keys, proxy settings) was gone, so
    on every image without util-linux setpriv (Alpine, BusyBox, a Debian
    without it) the agent failed before doing anything ("Authentication
    required"). ``-m`` keeps the environment, and then BusyBox's,
    util-linux's and shadow's su set none of HOME, SHELL, USER or LOGNAME;
    what su's PAM session adds or resets is undone by ``_SU_RESTORE``, and
    ``inner`` sets HOME as on the setpriv path. Without ``-l`` no profile is
    read and the working directory stays, as with setpriv. Every su switches
    users before it starts the shell, so what that shell reads
    (``BASH_ENV``) runs as the agent, as on the setpriv path. Options come
    before the user name, which all three accept.
    """
    restored = f"{_SU_RESTORE} {inner}"
    return (
        f'{su_command} -m -s "$bf_su_shell" -c {shlex.quote(restored)} {sandbox_user}'
    )


def build_priv_drop_cmd(
    agent_launch: str,
    sandbox_user: str,
    *,
    drop_net_admin: bool = False,
    protected_tools: bool = False,
    root_tools: bool = False,
) -> "AgentLaunch":
    """Build a shell command that drops to sandbox_user via setpriv or su.

    setpriv (util-linux) execs directly; environment-preserving su is the
    identity-drop fallback under the same kernel restrictions.
    Both paths require no_new_privs and drop the firewall capabilities before
    any agent code or login profile runs. BusyBox setpriv cannot remove the
    bounding capabilities: refuse launch rather than silently weaken the drop.
    Every step before the drop uses the system PATH and no loader or start-up
    variable; the task's PATH and those variables come back only as
    sandbox_user (C8, see :mod:`benchflow.sandbox._root_launch`). A
    live-process backend renders it inside its own root bootstrap
    (``root_launch_script``); as a string it is a complete command for an
    ``exec`` as root.

    ``drop_net_admin`` remains accepted for older audit callers. The S2
    barrier now applies to every launch, regardless of that argument.
    ``protected_tools`` resolves every root-side tool through S2's custody
    walk (``tool``); ``root_tools`` uses the absolute paths the root audit
    bound.
    """
    # Imported here: _base imports benchflow.task, which imports this module.
    from benchflow.sandbox._root_launch import AgentLaunch

    return AgentLaunch(
        agent_launch,
        sandbox_user,
        root_tools=root_tools,
        protected_tools=protected_tools,
    )


#: The primary group an agent that runs as uid 0 gets under the no-web
#: policy. The uid firewall matches uid 0 *and* this group,
#: so it holds the agent's processes and none of BenchFlow's own root
#: processes, which keep gid 0: the in-sandbox model proxy (it must reach
#: the provider), Daytona's root daemon (its replies are uid 0's) and root's
#: reads. The agent cannot leave the group: SETGID and SETUID are out of its
#: bounding set, with NET_ADMIN and NET_RAW, so it cannot change the rules.
NO_WEB_ROOT_GID = 61997
_NO_WEB_ENV = "BENCHFLOW_DISALLOW_WEB_TOOLS"


def no_web_root_agent(sandbox_user: str | None, agent_env: Mapping[str, str]) -> bool:
    """Whether the agent runs as uid 0 under the no-web policy (held by
    :func:`build_root_no_web_launch` and the root variant of the uid
    firewall)."""
    return not sandbox_user and agent_env.get(_NO_WEB_ENV) == "1"


def build_root_no_web_launch(agent_launch: str) -> str:
    """Start an agent that runs as uid 0 under the no-web policy: uid 0
    with primary group :data:`NO_WEB_ROOT_GID` and no supplementary groups,
    no new privileges, and none of NET_ADMIN, NET_RAW, SETUID or SETGID in
    its bounding set (util-linux setpriv). An image without it refuses the
    launch with the reason; the agent never starts unheld."""
    inner = shlex.quote(f"export HOME=/root && {agent_launch}")
    return (
        "if setpriv --help 2>&1 | grep -q regid"
        " && setpriv --help 2>&1 | grep -q bounding-set; then"
        f" exec setpriv --regid={NO_WEB_ROOT_GID} --clear-groups --no-new-privs"
        " --bounding-set=-net_admin,-net_raw,-setuid,-setgid"
        f" --inh-caps=-all --ambient-caps=-all -- bash -c {inner};"
        " else echo 'sandbox startup refused: an agent that runs as root under"
        " the no-web policy needs util-linux setpriv for its restricted launch;"
        " install util-linux in the image or choose a non-root --sandbox-user'"
        " >&2; exit 1; fi"
    )


EGRESS_DENYLIST_ENV = "BENCHFLOW_EGRESS_DENYLIST"
#: Set with EGRESS_DENYLIST_ENV when the proxy runs in allow mode.
EGRESS_ALLOWLIST_ENV = "BENCHFLOW_EGRESS_ALLOWLIST"
#: Comma-separated IP/CIDR allowlist entries the uid firewall admits directly.
EGRESS_ALLOW_NETWORKS_ENV = "BENCHFLOW_EGRESS_ALLOW_NETWORKS"
#: Loopback port of the allow-mode DNS filter inside the sandbox.
EGRESS_DNS_PORT = 18653


def _is_loopback_http(url: str) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and parsed.port is not None
    )


async def enforce_agent_egress_firewall(
    env: Any,
    sandbox_user: str | None,
    agent_env: dict[str, str],
) -> None:
    """Block sandbox-user external egress after ACP bootstrap, before prompting.

    Armed by the no-web policy (model traffic must already use the sandbox-local
    proxy) or by the denylist mode (all traffic must already use the loopback
    egress proxy).
    """
    no_web = agent_env.get("BENCHFLOW_DISALLOW_WEB_TOOLS") == "1"
    denylist = agent_env.get(EGRESS_DENYLIST_ENV) == "1"
    if no_web and agent_env.get(EGRESS_ALLOWLIST_ENV) == "1":
        # The loopback proxy would still permit allowed HTTP destinations even
        # if we omitted the direct-IP rules. Refuse inconsistent inputs rather
        # than silently weaken the agent's no-web policy. Native model-only
        # transport carries no allowlist marker and remains valid here.
        raise RuntimeError("No-web agent cannot use an allowlist egress policy")
    if not (no_web or denylist):
        return
    if not sandbox_user and denylist:
        raise RuntimeError("network_mode='denylist' requires a sandbox_user")

    base_url = agent_env.get("BENCHFLOW_PROVIDER_BASE_URL") or agent_env.get(
        "LLM_BASE_URL", ""
    )
    if denylist:
        if not _is_loopback_http(agent_env.get("HTTPS_PROXY", "")):
            raise RuntimeError(
                "Denylist agent requires HTTPS_PROXY on an HTTP loopback port"
            )
        if base_url and not _is_loopback_http(base_url):
            raise RuntimeError(
                "Denylist agent requires an HTTP loopback provider base URL with a port"
            )
    elif not _is_loopback_http(base_url):
        raise RuntimeError(
            "No-web agent requires an HTTP loopback provider base URL with a port"
        )

    if not sandbox_user:
        # A uid-0 agent under no-web: its launch put it in
        # NO_WEB_ROOT_GID (build_root_no_web_launch); hold that.
        await enforce_sandbox_uid_egress(
            env, "root", root_gid=NO_WEB_ROOT_GID, refuse_dns=True
        )
        return
    allow_networks = None
    if denylist and agent_env.get(EGRESS_ALLOWLIST_ENV) == "1":
        allow_networks = tuple(
            n for n in agent_env.get(EGRESS_ALLOW_NETWORKS_ENV, "").split(",") if n
        )
    await enforce_sandbox_uid_egress(
        env, sandbox_user, allow_networks=allow_networks, refuse_dns=no_web
    )


async def enforce_sandbox_uid_egress(
    env: Any,
    sandbox_user: str,
    *,
    allow_networks: tuple[str, ...] | None = None,
    root_gid: int | None = None,
    refuse_dns: bool = False,
) -> None:
    """Install the shared loopback-only UID firewall without inference setup.

    ``allow_networks`` (allowlist mode) adds the DNS redirect and the direct
    rules for listed IP ranges; see ``_allowlist_firewall_cmd``.
    ``refuse_dns`` (an offline agent) refuses DNS on loopback too.
    """
    result = await exec_raw(
        env.exec,
        _agent_egress_firewall_cmd(
            sandbox_user,
            allow_networks=allow_networks,
            root_gid=root_gid,
            refuse_dns=refuse_dns,
        ),
        user="root",
        timeout_sec=120,
    )
    if _exec_return_code(result) != 0:
        detail = _exec_failure_detail(result)
        raise RuntimeError(f"Failed to enforce sandbox-user egress firewall.{detail}")
    logger.info(
        "Sandbox-user egress firewall active for %s (loopback allowed)",
        sandbox_user,
    )


# The ten endpoints, the decision and the record live in
# benchflow.sandbox.cloud_metadata; the rules root installs are here.
_CLOUD_METADATA_TOOLS4 = ("iptables", "iptables-nft", "iptables-legacy")
_CLOUD_METADATA_TOOLS6 = ("ip6tables", "ip6tables-nft", "ip6tables-legacy")
# The endpoint the agent user's test connect goes to (TCP connect only).
_CLOUD_METADATA_PROBE = "169.254.169.254"
# Seconds root's script gives an iptables install, well inside the exec's own
# limit (_CLOUD_METADATA_EXEC_SEC), so the package manager is stopped in the
# sandbox rather than left running when the exec gives up.
_CLOUD_METADATA_INSTALL_SEC = 240
_CLOUD_METADATA_EXEC_SEC = 300
# Root's scripts search only these folders, never the image's PATH: the block
# runs again on every connection, after the agent may have put an iptables in
# an agent-owned folder the image's PATH names first (/app/bin), which root
# would have run with NET_ADMIN. A folder a non-root user can
# change is left out, and a firewall tool runs only when every folder and
# link on its way belongs to these uids and no one else can write it.
_CLOUD_METADATA_ROOT_PATH = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
_CLOUD_METADATA_TRUSTED_UIDS = ("0",)


def _cloud_metadata_root_prelude() -> str:
    """The start of root's scripts: no loader or startup variables, a PATH
    of root's own folders, and ``tool <name>``, which prints the firewall
    tool's path only when nothing on its way is anyone else's to change.

    ``mine <path>``: the entry is owned by a trusted uid and not writable by
    its group or others (a link's own mode means nothing; a sticky folder
    such as /tmp is fine, since no one else can replace root's entries in
    it); an ACL (``+``) counts as writable. ``walk <path>`` follows the path
    one name at a time, every link included, checking each entry it meets.
    """
    uids = " ".join(_CLOUD_METADATA_TRUSTED_UIDS)
    return (
        # ls formats nothing by the caller's TIME_STYLE or QUOTING_STYLE:
        # walk reads link targets from its output.
        "unset LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT BASH_ENV ENV CDPATH "
        "TIME_STYLE QUOTING_STYLE; "
        f"U={shlex.quote(uids)}; "
        # Bootstrap beside the trusted /bin/sh, not through optional PATH
        # folders whose custody we have not established yet. /bin/ls also
        # reads each link's target ("<path> -> <target>"; the fields before
        # the path hold no "/"): Alpine has no /bin/readlink, only
        # /usr/bin/readlink, so naming /bin/readlink failed every link
        # (DET-ALPINE-SU).
        'mine() { set -f; set -- $(/bin/ls -ldn "$1" 2>/dev/null); set +f; '
        '[ -n "$1" ] || return 1; '
        'case " $U " in *" $3 "*) ;; *) return 1;; esac; '
        "case $1 in l*) return 0;; ??????????+*) return 1;; esac; "
        "case $1 in ?????w*|????????w*) "
        "case $1 in d????????[tT]*) return 0;; esac; return 1;; esac; }; "
        "walk() { _r=$1; _c=; _n=0; case $_r in /*) ;; *) return 1;; esac; "
        "mine / || return 1; "
        'while [ -n "$_r" ]; do _r=${_r#/}; [ -n "$_r" ] || break; '
        "_x=${_r%%/*}; case $_r in */*) _r=/${_r#*/};; *) _r=;; esac; "
        "case $_x in ''|.) continue;; ..) _c=${_c%/*}; continue;; esac; "
        '_k=$_c/$_x; mine "$_k" || return 1; '
        'if [ -L "$_k" ]; then _n=$((_n + 1)); [ $_n -le 40 ] || return 1; '
        '_t=$(/bin/ls -ldn "$_k") || return 1; '
        'case $_t in *" $_k -> "?*) _t=${_t#*" $_k -> "};; *) return 1;; esac; '
        "case $_t in /*) _c=; _r=$_t$_r;; *) _r=/$_t$_r;; esac; "
        "else _c=$_k; fi; done; W=${_c:-/}; }; "
        # Each folder of root's PATH, kept only when it is root's alone; the
        # others are remembered, so a tool left in one counts as untrusted.
        f"_p=; _bad=; _l={shlex.quote(_CLOUD_METADATA_ROOT_PATH + ':')}; "
        'while [ -n "$_l" ]; do d=${_l%%:*}; _l=${_l#*:}; [ -n "$d" ] || continue; '
        'if walk "$d" && [ -d "$W" ]; then _p="$_p:$d"; else _bad="$_bad$d:"; fi; done; '
        "PATH=${_p#:}; export PATH; "
        # 1: no such tool; 2: there is one, but someone else can change it.
        'tool() { _f=$(command -v "$1" 2>/dev/null) || return 1; '
        'walk "$_f" && [ -f "$W" ] && [ -x "$W" ] || return 2; echo "$_f"; }; '
        'untrusted() { for c in "$@"; do tool "$c" >/dev/null; [ $? = 2 ] && return 0; '
        '_l=$_bad; while [ -n "$_l" ]; do d=${_l%%:*}; _l=${_l#*:}; '
        '[ -e "$d/$c" ] && return 0; done; done; return 1; }; '
    )


def _protected_root_prelude() -> str:
    """S2's tool custody walk, bootstrapped off the materialized image PATH.

    Callers invoke external commands with ``run <name>`` or carry ``tool``'s
    checked absolute result directly into exec, never resolve it again.
    """
    return (
        f"PATH={shlex.quote(_CLOUD_METADATA_ROOT_PATH)}; export PATH; "
        + _cloud_metadata_root_prelude()
        + 'run() { _bf_tool=$(tool "$1") || return 127; shift; '
        '"$_bf_tool" "$@"; }; '
        'have() { tool "$1" >/dev/null 2>&1; }; '
    )


def _cloud_metadata_rule(address: str, *, cover_root: bool = False) -> str:
    """One refusal: TCP from any non-root user to ``address``, but DNS; from
    every user, root included, with ``cover_root`` (an agent that runs as
    root, in a ``main`` whose root cannot remove the rule)."""
    owner = "" if cover_root else "-m owner ! --uid-owner 0 "
    return f"-d {address} -p tcp ! --dport 53 {owner}-j REJECT --reject-with tcp-reset"


# The packets the IPv4 test endpoint's rule has refused, read from
# ``-S -v`` (``-c <packets> <bytes>``, as iptables 1.8 prints it for nft and
# legacy alike); 0 without the rule.
_CLOUD_METADATA_HITS_SH = (
    '"$fw4" -w -S OUTPUT -v 2>/dev/null | awk '
    "'/-d 169\\.254\\.169\\.254\\/32 / && /tcp-reset/ "
    '{ for (i = 1; i < NF; i++) if ($i == "-c") n += $(i + 1) } '
    "END { print n + 0 }'"
)


def _cloud_metadata_block_cmd(
    proc: str = "/proc",
    apt_lists: str = "/var/lib/apt/lists",
    *,
    cover_root: bool = False,
) -> str:
    """Root's script: install iptables if the image has none, then refuse
    the cloud metadata endpoints for every non-root user (every user, root
    included, with ``cover_root``: :func:`_cloud_metadata_rule`).

    POSIX sh (dash and busybox ash included). Picks, per family, the first
    of ``iptables``, ``iptables-nft`` and ``iptables-legacy`` that can list
    the filter table on this kernel (the legacy backend where nf_tables is
    missing). Without iptables, and only when root holds NET_ADMIN (without
    it no rule could be added), installs it with apt-get, apk, dnf, microdnf
    or yum, leaving apt's package lists as it found them (a solution may
    install from them without an update). The install gets
    ``_CLOUD_METADATA_INSTALL_SEC`` seconds in all (``timeout``, where the
    image has one that takes ``-k``, stops the package manager and its
    children then), so it ends well inside the exec's own limit and leaves
    nothing running. IPv6 rules are always required: even absent proc
    entries cannot rule out a later module load. Current interface settings
    cannot constrain a later link or a service enabling IPv6; no backend
    here guarantees IPv6 stays off throughout the protected lifetime.
    Tools are searched and checked as
    :func:`_cloud_metadata_root_prelude` says.
    Running it again adds nothing. Prints one line, ``BF metadata
    status=<n> fw4=<tool> fw6=<tool> v6=<absent|enabled|global>
    install=<none|ok|failed|timeout|unavailable|no-net-admin>
    install_t=<start>,<end> hits=<n>`` (``/proc/uptime`` seconds; ``-`` for
    none): status 0 the rules hold, 1 a rule could not be added, 2 no
    iptables and no package manager, 3 IPv6 can route out but no ip6tables,
    4 root lacks NET_ADMIN (no rule can be added), 5 installing iptables
    failed or ran out of time (then ``BF metadata-install: <its last
    lines>``), 6 the only firewall tool is one a non-root user can change.
    """
    rules4 = " ".join(
        shlex.quote(_cloud_metadata_rule(a, cover_root=cover_root))
        for a in CLOUD_METADATA_V4
    )
    rules6 = " ".join(
        shlex.quote(_cloud_metadata_rule(a, cover_root=cover_root))
        for a in CLOUD_METADATA_V6
    )
    apt = "bounded env DEBIAN_FRONTEND=noninteractive apt-get"
    apt_install = f"{apt} install -y -qq --no-install-recommends iptables"
    tools4 = " ".join(_CLOUD_METADATA_TOOLS4)
    tools6 = " ".join(_CLOUD_METADATA_TOOLS6)
    return (
        "( "
        + _cloud_metadata_root_prelude()
        + f"P={shlex.quote(proc)}; L={shlex.quote(apt_lists)}; "
        "now() { cut -d ' ' -f 1 \"$P/uptime\" 2>/dev/null || echo 0; }; "
        'has() { for c in "$@"; do command -v "$c" >/dev/null 2>&1 && return 0; done; return 1; }; '
        'pick() { for c in "$@"; do f=$(tool "$c") && '
        '"$f" -w -S OUTPUT >/dev/null 2>&1 && { echo "$f"; return 0; }; done; return 1; }; '
        "v6=absent; "
        'if [ -e "$P/net/if_inet6" ] || [ -d "$P/sys/net/ipv6" ]; then v6=enabled; '
        'awk \'$4 == "00" && $6 != "lo" { f = 1 } END { exit !f }\' "$P/net/if_inet6" && v6=global; '
        "fi; "
        "cap=$(awk '/^CapEff:/ { print $2 }' \"$P/self/status\" 2>/dev/null); "
        "case $cap in ''|*[!0-9a-fA-F]*) cap=0;; esac; "
        "netadmin=$(( 0x$cap >> 12 & 1 )); "
        "install=none; install_t=-; out=; irc=0; need4=; need6=; "
        f"has {tools4} || need4=1; "
        f"has {tools6} || need6=1; "
        'if [ -n "$need4$need6" ]; then '
        'if [ "$netadmin" != 1 ]; then install=no-net-admin; else '
        "t0=$(now); install=failed; "
        # One deadline for the whole install; each package-manager call gets
        # what is left of it.
        "secs() { now | cut -d . -f 1; }; "
        f"dl=$(( $(secs) + {_CLOUD_METADATA_INSTALL_SEC} )); to=; "
        "command -v timeout >/dev/null 2>&1 && timeout -k 5 5 true >/dev/null 2>&1 && to=1; "
        'bounded() { [ -n "$to" ] || { "$@"; return; }; '
        'left=$(( dl - $(secs) )); [ "$left" -gt 0 ] || return 124; '
        'timeout -k 10 "$left" "$@"; }; '
        "if command -v apt-get >/dev/null 2>&1; then "
        'keep=$(mktemp -d 2>/dev/null) || { keep=/tmp/.bf-apt-lists-$$; mkdir -p "$keep"; }; '
        'saved=; cp -a "${L:?}/." "$keep/" 2>/dev/null && saved=1; '
        f"out=$( {{ {apt_install} || {{ {apt} update -qq && {apt_install}; }}; }} 2>&1 ); irc=$?; "
        # Restore the lists only from a complete copy of them.
        'if [ -n "$saved" ]; then rm -rf "${L:?}"/* 2>/dev/null; cp -a "${keep:?}/." "$L/" 2>/dev/null; fi; '
        'rm -rf "${keep:?}"; '
        "elif command -v apk >/dev/null 2>&1; then out=$(bounded apk add --no-cache iptables ip6tables 2>&1); irc=$?; "
        "elif command -v dnf >/dev/null 2>&1; then out=$( { bounded dnf install -y iptables-nft || bounded dnf install -y iptables; } 2>&1 ); irc=$?; "
        "elif command -v microdnf >/dev/null 2>&1; then out=$( { bounded microdnf install -y iptables-nft || bounded microdnf install -y iptables; } 2>&1 ); irc=$?; "
        "elif command -v yum >/dev/null 2>&1; then out=$(bounded yum install -y iptables 2>&1); irc=$?; "
        "else install=unavailable; fi; "
        # ok only when what was missing is there now (an IPv4 iptables that
        # was already there proves nothing about an ip6tables install).
        'if [ "$install" = failed ]; then '
        f'{{ [ -z "$need4" ] || has {tools4}; }} && {{ [ -z "$need6" ] || has {tools6}; }} && install=ok; '
        '[ "$install" = failed ] && [ "$irc" = 124 ] && install=timeout; fi; '
        'install_t="$t0,$(now)"; fi; fi; '
        f"fw4=$(pick {tools4}) || fw4=; "
        f"fw6=$(pick {tools6}) || fw6=; "
        "status=0; "
        'if [ -z "$fw4" ]; then '
        f"if untrusted {tools4}; then status=6; "
        f"elif has {tools4}; then "
        'if [ "$netadmin" = 1 ]; then status=1; else status=4; fi; '
        'elif [ "$install" = no-net-admin ]; then status=4; '
        'elif [ "$install" = unavailable ]; then status=2; '
        "else status=5; fi; "
        "else "
        f"for r in {rules4}; do "
        '"$fw4" -w -C OUTPUT $r 2>/dev/null || "$fw4" -w -I OUTPUT 1 $r 2>/dev/null || status=1; done; '
        'if [ -n "$fw6" ]; then '
        f"for r in {rules6}; do "
        '"$fw6" -w -C OUTPUT $r 2>/dev/null || "$fw6" -w -I OUTPUT 1 $r 2>/dev/null || '
        "status=1; done; "
        "elif [ $status = 0 ]; then "
        f"if untrusted {tools6}; then status=6; else status=3; fi; fi; fi; "
        f'hits=0; [ -n "$fw4" ] && hits=$({_CLOUD_METADATA_HITS_SH}); '
        # The record names the tool, never its path (block_cloud_metadata
        # runs only names it knows).
        "n4=${fw4##*/}; n6=${fw6##*/}; "
        'echo "BF metadata status=$status fw4=${n4:--} fw6=${n6:--} v6=$v6 '
        'install=$install install_t=$install_t hits=${hits:-0}"; '
        'if [ $status != 0 ] && [ -n "$out" ]; then '
        """echo "BF metadata-install: $(printf '%s' "$out" | tail -n 3 | tr '\\n' ' ' | cut -c1-300)"; fi """
        ")"
    )


def _cloud_metadata_agent_check_cmd(
    probe: str, proc: str = "/proc", *, connect_probe: bool = True
) -> str:
    """The agent user's check, run as that user after root's rules.

    Whether the user holds NET_ADMIN (in its permitted, effective or ambient
    set), whether any firewall tool in the fixed system directories or any
    entry of its ``PATH`` carries file capabilities or a setuid/setgid bit,
    whether it can mutate the firewall, and a TCP connect, and
    nothing more, to 169.254.169.254:80, which the rule must refuse. Prints
    ``BF metadata-agent held=<0|1> power=<tools> connect=<refused|reached|
    timeout|failed|skipped>``; ``skipped`` without ``bash`` and ``timeout``,
    or when installation failed and only the privilege check is needed.
    File inspection must complete with trusted ``getcap`` and ``readlink``;
    failure of a mutation probe cannot establish absence of privilege (an
    ordinary nft can support ip/ip6 while the kernel lacks inet tables).
    Candidates and their cleanup run with the agent's PATH so wrappers can
    reach the same delegated executables; inspection keeps its trusted PATH.
    """
    p = shlex.quote(proc)
    q = shlex.quote(probe)
    tools = " ".join(_CLOUD_METADATA_TOOLS4 + _CLOUD_METADATA_TOOLS6 + ("nft",))
    connect = shlex.quote(f"exec 3<>/dev/tcp/{_CLOUD_METADATA_PROBE}/80")
    return (
        "( "
        # Keep candidate discovery separate from the inspection utilities:
        # the agent PATH is never a source for getcap/readlink.
        f"agent_path=${{PATH-}}; PATH={shlex.quote(_CLOUD_METADATA_ROOT_PATH)}; export PATH; "
        + _cloud_metadata_root_prelude()
        + f"P={p}; "
        # Missing/malformed capabilities must not become a successful zero
        # report. Read all three sets with shell builtins, including at retry.
        "held=0; count=0; seen=; "
        "while read -r k c rest; do case $k in CapPrm:|CapEff:|CapAmb:) "
        'case "|$seen|" in *"|$k|"*) exit 7;; esac; seen="$seen|$k"; '
        "case $c in ''|*[!0-9a-fA-F]*) exit 7;; esac; "
        '[ ${#c} = 16 ] && [ -z "$rest" ] || exit 7; '
        "count=$((count + 1)); [ $(( 0x$c >> 12 & 1 )) = 1 ] && held=1; "
        ';; esac; done < "$P/self/status"; [ "$count" = 3 ] || exit 7; '
        "lock=$(mktemp 2>/dev/null) || lock=/tmp/.bf-xtables-$$; power=; cap_reader=; link_reader=; "
        'mark() { case ",$power," in *",$1,"*) ;; *) power="${power:+$power,}$1";; esac; }; '
        # A trusted wrapper may delegate by name to a privileged executable
        # outside the firewall-name allowlist. Probe that delegation as the
        # agent would, in a subshell that cannot change the inspector's PATH.
        'candidate() ( PATH="$agent_path"; export PATH; "$@"; ); '
        # A first-match lookup misses off-PATH and shadowed system tools,
        # including tools root rejected as writable. Enumerate every pair.
        f'dirs={shlex.quote(_CLOUD_METADATA_ROOT_PATH)}:"$agent_path:"; '
        'while [ -n "$dirs" ]; do d=${dirs%%:*}; dirs=${dirs#*:}; d=${d:-.}; '
        f'for t in {tools}; do f="$d/$t"; [ -f "$f" ] && [ -x "$f" ] || continue; '
        # Positive privilege evidence is independent of kernel features,
        # table availability, lock contention and the candidate's exit code.
        # Refuse every file capability, including permitted/inheritable-only
        # sets; interpreting which combination can gain power is unnecessary.
        'if [ -u "$f" ] || [ -g "$f" ]; then mark "$t"; continue; fi; '
        '[ -n "$cap_reader" ] || cap_reader=$(tool getcap) || exit 7; '
        '[ -n "$link_reader" ] || link_reader=$(tool readlink) || exit 7; '
        'target=$("$link_reader" -f -- "$f") || exit 7; '
        "case $target in /*) ;; *) exit 7;; esac; "
        '[ -f "$target" ] && [ -x "$target" ] || exit 7; '
        # getcap does not follow a symlink itself. Its verbose output must
        # name the resolved regular file even when there are no attributes;
        # a missing/failed/empty inspection must not become a clean report.
        'caps=$("$cap_reader" -v "$target" 2>&1) || exit 7; '
        'if [ "$caps" != "$target" ]; then '
        'case $caps in "$target "*) mark "$t"; continue;; *) exit 7;; esac; fi; '
        'if [ "$t" = nft ]; then '
        f'if candidate "$f" add table inet {q} >/dev/null 2>&1; then mark "$t"; '
        f'candidate "$f" delete table inet {q} >/dev/null 2>&1; fi; '
        f'elif XTABLES_LOCKFILE="$lock" candidate "$f" -w -N {q} >/dev/null 2>&1; then mark "$t"; '
        f'XTABLES_LOCKFILE="$lock" candidate "$f" -w -X {q} >/dev/null 2>&1; fi; done; done; '
        'rm -f "$lock" 2>/dev/null; '
        "conn=skipped; "
        f"if {'true' if connect_probe else 'false'} && "
        "command -v bash >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then "
        f"err=$(timeout 5 bash -c {connect} 2>&1); rc=$?; "
        "case $rc in 0) conn=reached;; 124) conn=timeout;; "
        "*) case $err in *[Rr]efused*) conn=refused;; *) conn=failed;; esac;; esac; fi; "
        'echo "BF metadata-agent held=$held power=${power#,} connect=$conn" '
        ")"
    )


def _cloud_metadata_after_cmd(fw4: str, probe: str) -> str:
    """Root, after the agent user's check: the test endpoint rule's packet
    count (``BF metadata-hits <n>``), and the agent's probe chain or table
    removed if the agent could make one and left it. Each tool is found again
    as root's script finds it (:func:`_cloud_metadata_root_prelude`), never on
    the image's PATH."""
    tools = " ".join(_CLOUD_METADATA_TOOLS4 + _CLOUD_METADATA_TOOLS6)
    q = shlex.quote(probe)
    return (
        "( "
        + _cloud_metadata_root_prelude()
        + f'n4={shlex.quote(fw4)}; fw4=$(tool "$n4") || fw4=; '
        f'hits=0; [ -n "$fw4" ] && hits=$({_CLOUD_METADATA_HITS_SH}); '
        'echo "BF metadata-hits ${hits:-0}"; '
        f'for t in {tools}; do f=$(tool "$t") && "$f" -w -X {q} >/dev/null 2>&1; done; '
        f'f=$(tool nft) && "$f" delete table inet {q} >/dev/null 2>&1; '
        "true )"
    )


def _cloud_metadata_delegate_check_cmd(root: str = "/") -> str:
    """Inventory privilege independently of executable names or probe success.

    A trusted wrapper (including a native binary) may execute a delegate by
    an absolute path, through another wrapper, or only for an unprobed
    operation. Inspect every regular file, across mounts, as root: an agent
    can traverse a directory it cannot list. Never execute the listed files.
    Only /proc and /sys kernel interfaces are excluded, not /opt, /dev/shm,
    private folders or volumes.
    Only narrow distribution set-ID helpers are exempt. General command or
    identity launchers are not: legitimate sudoers, doas or authentication
    policy can delegate root without a hostile package, even when a firewall
    mutation probe fails. Do not infer absent authority from current policy.
    Other set-ID files and file capabilities refuse startup; narrow network
    tools may retain only their usual NET_RAW at fixed distribution paths.
    Enumeration/inspection failure is not a clean inventory.
    """
    # Exact paths, not basenames: /opt/su and /tmp/ping receive no exemption.
    # Deliberately narrower than S3's _STANDARD_SETUID policy: su, ksu, sudo,
    # sudoedit, doas, pkexec and BusyBox bbsuid can launch commands/identities under
    # image policy. A fixed distribution path cannot make that grant narrow.
    names = [
        "at",
        "chage",
        "chfn",
        "chsh",
        "dbus-daemon-launch-helper",
        "expiry",
        "fusermount",
        "fusermount3",
        "gpasswd",
        "mount",
        "mount.cifs",
        "mount.nfs",
        "newgidmap",
        "newgrp",
        "newuidmap",
        "passwd",
        "ping",
        "ping6",
        "polkit-agent-helper-1",
        "pt_chown",
        "ssh-keysign",
        "umount",
        "unix_chkpwd",
        "Xorg.wrap",
        # Standard setgid-only helpers (S3's setuid inventory did not need
        # these): authentication, terminal messages and lookup databases.
        "ssh-agent",
        "pam_extrausers_chkpwd",
        "crontab",
        "wall",
        "write",
        "bsd-write",
        "locate",
        "mlocate",
        "plocate",
    ]
    standard_paths = [
        f"{directory}/{name}"
        for directory in ("/usr/bin", "/usr/sbin", "/bin", "/sbin")
        for name in names
    ]
    standard_paths += [
        "/opt/google/chrome/chrome-sandbox",
        "/usr/lib/chromium-browser/chrome-sandbox",
        "/usr/lib/chromium/chrome-sandbox",
        "/usr/lib/aarch64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-ptp-helper",
        "/usr/lib/x86_64-linux-gnu/gstreamer-1.0/gstreamer-1.0/gst-ptp-helper",
        "/usr/libexec/gstreamer-1.0/gst-ptp-helper",
        "/usr/lib/dbus-1.0/dbus-daemon-launch-helper",
        "/usr/lib/openssh/ssh-keysign",
        "/usr/lib/polkit-1/polkit-agent-helper-1",
        "/usr/libexec/dbus-daemon-launch-helper",
        "/usr/libexec/openssh/ssh-keysign",
        "/usr/libexec/polkit-agent-helper-1",
        "/usr/lib/xorg/Xorg.wrap",
        "/usr/lib/eject/dmcrypt-get-device",
        "/usr/sbin/pam_timestamp_check",
    ]
    standard = "|".join(map(shlex.quote, standard_paths))
    # Do not try to enumerate "dangerous" capability names: DAC_OVERRIDE,
    # FOWNER, SYS_MODULE, etc. can also lead to root. Only this narrow grant
    # on trusted distribution network tools is exempt, never NET_ADMIN.
    narrow_caps = "|".join(
        shlex.quote(f"{directory}/{name} cap_net_raw=ep")
        for directory in ("/usr/bin", "/usr/sbin", "/bin", "/sbin")
        for name in (
            "arping",
            "clockdiff",
            "fping",
            "fping6",
            "mtr-packet",
            "ping",
            "ping4",
            "ping6",
            "traceroute6.iputils",
        )
    )
    # Do not print filenames: even an agent-created filename is report data,
    # and cannot become a shell program or forge the final inventory marker.
    classify = (
        f"for f do case \"$f\" in {standard}) ;; *) printf 'unsafe\\n';; esac; done"
    )
    scan = (
        f'"$finder" {shlex.quote(root)} '
        r"\( -path /proc -o -path /sys \) -prune -o -type f "
    )
    return (
        "( "
        + _cloud_metadata_root_prelude()
        + "finder=$(tool find) && reader=$(tool getcap) || exit 7; "
        # GNU/BusyBox find propagate failures from the batched -exec. Unlike
        # a pipe to grep, the command substitution retains the scan's status.
        + f'caps=$({scan}-exec "$reader" {{}} +) || exit 7; '
        + "power=0; "
        + 'cap_power=$(printf "%s\\n" "$caps" | while IFS= read -r line; do '
        + f"case $line in ''|{narrow_caps}) ;; *) printf unsafe;; esac; done); "
        + '[ -z "$cap_power" ] || power=1; '
        + f"setids=$({scan}\\( -perm -4000 -o -perm -2000 \\) "
        + f"-exec /bin/sh -c {shlex.quote(classify)} sh {{}} +) || exit 7; "
        + '[ -z "$setids" ] || power=1; '
        + 'printf "BF metadata-delegates power=%s\\n" "$power"; )'
    )


def _proc_status_reader() -> str:
    """A shell-builtin status read shared by the barrier and inventories.

    Exactly one field is required; missing, duplicate or unreadable fields
    never turn into an empty capability set. No optional awk is involved.
    """
    return (
        "read_status() { _bf_want=$1; _bf_count=0; _bf_value=; "
        "while read -r _bf_key _bf_line; do "
        'if [ "$_bf_key" = "$_bf_want:" ]; then '
        "_bf_count=$((_bf_count + 1)); _bf_value=$_bf_line; fi; "
        'done < "$2" || return 1; '
        '[ "$_bf_count" = 1 ] || return 1; '
        'printf "%s\\n" "$_bf_value"; }; '
    )


def _cloud_metadata_barrier_verified(output: str, return_code: object) -> bool:
    """The one acceptance rule for S2's live construction receipt."""
    barrier = re.fullmatch(
        r"BF metadata-construction uid=([0-9]+) nnp=1 "
        r"bnd=([0-9a-fA-F]{16}) amb=0000000000000000\n",
        output,
    )
    return (
        type(return_code) is int
        and return_code == 0
        and barrier is not None
        and int(barrier[1]) > 0
        and not int(barrier[2], 16) & ((1 << 12) | (1 << 13))
    )


def _cloud_metadata_construction_cmd(
    sandbox_user: str, *, protected_tools: bool = False
) -> str:
    """Read the shared agent launch's kernel state as root, not via getcap.

    Root owns both FIFO names; the dropped shell inherits only their open
    descriptors. It waits until root has read /proc/<pid>/status, so neither
    a short-lived process nor a file-attribute inventory stands in for the
    exec barrier. The actual agent uses this same privilege-drop helper.
    """
    launch = build_priv_drop_cmd(
        "exec /bin/sh -c "
        + shlex.quote('printf "%s\\n" "$$" >&3; IFS= read -r done <&4'),
        sandbox_user,
        **({"protected_tools": True} if protected_tools else {}),
    )
    # Root's PATH is the exec route's (the protected kit's once the sandbox
    # has one): this command sets none of its own (C8, PT-C8-R3-3).

    # The supplemental caller also protects this proof's tooling after its
    # backend has materialized persistent env, just like every audit stage.
    def root_tool(name: str) -> str:
        return f'"$(tool {name})"' if protected_tools else name

    return (
        "( "
        + (_protected_root_prelude() if protected_tools else "")
        + _proc_status_reader()
        + f"uid=$({root_tool('id')} -u {shlex.quote(sandbox_user)}) || exit 7; "
        '[ "$uid" -gt 0 ] || exit 7; '
        f"d=$({root_tool('mktemp')} -d /tmp/.bf-metadata-barrier-XXXXXXXX) || exit 7; "
        f'trap \'{root_tool("rm")} -f "$d/pid" "$d/done"; {root_tool("rmdir")} "$d"\' EXIT; '
        f'{root_tool("mkfifo")} "$d/pid" "$d/done" || exit 7; '
        'exec 5<>"$d/pid" 6<>"$d/done"; '
        'exec 3>"$d/pid" 4<"$d/done"; '
        f'( ( {launch} ) || printf "failed\\n" >&3 ) 5>&- 6>&- & child=$!; '
        "exec 3>&- 4<&-; IFS= read -r pid <&5; "
        "case $pid in ''|*[!0-9]*) exit 7;; esac; "
        'valid=1; status="/proc/$pid/status"; '
        'uids=$(read_status Uid "$status") || valid=0; '
        "set -- $uids; "
        '[ "$#" = 4 ] && [ "$1" = "$uid" ] && [ "$2" = "$uid" ] && '
        '[ "$3" = "$uid" ] && [ "$4" = "$uid" ] || valid=0; '
        'nnp=$(read_status NoNewPrivs "$status") || valid=0; '
        'bnd=$(read_status CapBnd "$status") || valid=0; '
        'amb=$(read_status CapAmb "$status") || valid=0; '
        'printf "done\\n" >&6; wait "$child" || valid=0; '
        '[ "$valid" = 1 ] || exit 7; '
        'printf "BF metadata-construction uid=%s nnp=%s bnd=%s amb=%s\\n" '
        '"$uid" "$nnp" "$bnd" "$amb"; )'
    )


def _report_fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"\b(\w+)=(\S*)", line))


_CLOUD_METADATA_NOT_INSTALLED = {
    "1": "an iptables rule could not be added (no owner match or tcp-reset "
    "on this kernel, or no permission)",
    "2": "the image has no iptables and no package manager (apt-get, apk, "
    "dnf, microdnf or yum) to install it",
    "3": "IPv6 cannot be guaranteed unavailable for the sandbox lifetime and "
    "no working ip6tables could be installed (the IPv4 endpoints are refused)",
    "4": "the sandbox gives root no NET_ADMIN, so no firewall rule can be added",
    "5": "installing iptables failed",
    "6": "the only firewall tool is reached through a folder or link a non-root "
    "user can change, so root does not run it",
}


async def _cloud_metadata_exec(
    env: Any, command: str, *, user: str, timeout_sec: int
) -> tuple[Any, str]:
    """One exec of the block; ``(result, "")`` or ``(None, why)`` when the
    exec itself failed (it ran out of time, or the sandbox did not answer).

    The caller decides whether failure is optional evidence (installation
    or counters) or a mandatory privilege audit that must refuse startup.
    """
    try:
        # A sandbox that can start /bin/sh by path does so for root: the
        # image's PATH may name a folder the agent has written to by now.
        system = user == "root" and getattr(env, "supports_system_shell", False) is True
        kwargs: dict[str, Any] = {"system_shell": True} if system else {}
        result = await _read_exec(
            env, command, user=user, timeout_sec=timeout_sec, **kwargs
        )
    except Exception as e:  # recorded, and the trial goes on
        detail = str(e).strip().splitlines()[0][:300] if str(e).strip() else ""
        return None, f"{type(e).__name__}{': ' + detail if detail else ''}"
    return result, ""


async def _cloud_metadata_agent_exec(
    env: Any, command: str, sandbox_user: str, *, timeout_sec: int
) -> tuple[Any, str]:
    """Run probes through the same verified privilege drop as the agent.

    A bare backend exec(user=agent) does not establish no_new_privs or remove
    bounding capabilities. Agent-written audit candidates could then invoke
    a set-ID delegate before the audit could refuse startup.
    """
    return await _cloud_metadata_exec(
        env,
        build_priv_drop_cmd(command, sandbox_user),
        user="root",
        timeout_sec=timeout_sec,
    )


_ROOT_CAP_SETS = ("CapBnd", "CapPrm", "CapInh", "CapEff", "CapAmb")
# NET_ADMIN (12) changes the firewall; NET_RAW (13) opens packet sockets,
# whose frames never pass the OUTPUT chain the rules sit in.
_ROOT_CAP_BITS = {12: "NET_ADMIN", 13: "NET_RAW"}


def _cloud_metadata_root_caps_cmd(proc: str = "/proc") -> str:
    """Root's capability sets, read with shell builtins only:
    ``BF metadata-root CapBnd=<hex> CapPrm=<hex> ...``."""
    status = shlex.quote(f"{proc}/self/status")
    reads = "".join(
        f'v=$(read_status {name} {status}) || exit 7; line="$line {name}=$v"; '
        for name in _ROOT_CAP_SETS
    )
    return (
        "( "
        + _proc_status_reader()
        + "line=; "
        + reads
        + 'printf "BF metadata-root%s\\n" "$line" )'
    )


async def _cloud_metadata_root_caps(env: Any) -> tuple[dict[str, str], str]:
    """Root's capability sets in main, and why they cannot hold a root agent
    ("" when they can): NET_ADMIN or NET_RAW in its bounding, permitted or
    inheritable set (the effective and ambient sets are within those), or
    sets that could not be read. Every process the container runtime starts
    in main has these sets, and none can gain beyond its bounding set."""
    result, failed = await _cloud_metadata_exec(
        env, _cloud_metadata_root_caps_cmd(), user="root", timeout_sec=60
    )
    out = getattr(result, "stdout", "") or ""
    line = next(
        (
            ln
            for ln in (out if isinstance(out, str) else "").splitlines()
            if ln.startswith("BF metadata-root ")
        ),
        "",
    )
    sets = {
        k: v
        for k, v in _report_fields(line).items()
        if k in _ROOT_CAP_SETS and re.fullmatch(r"[0-9a-fA-F]{16}", v)
    }
    code = getattr(result, "return_code", None)
    if failed or type(code) is not int or code != 0 or len(sets) != len(_ROOT_CAP_SETS):
        why = failed or "the read did not report every set"
        return sets, f"root's capability sets could not be read ({why})"
    held = [
        name
        for bit, name in _ROOT_CAP_BITS.items()
        if any(int(sets[k], 16) >> bit & 1 for k in ("CapBnd", "CapPrm", "CapInh"))
    ]
    if held:
        return sets, (
            f"root in main holds {' and '.join(held)} (CapBnd {sets['CapBnd']})"
        )
    return sets, ""


def _cloud_metadata_connect_cmd(*, connect_probe: bool = True) -> str:
    """A TCP connect, and nothing more, to 169.254.169.254:80 (``bash`` and
    ``timeout`` from the image): ``BF metadata-connect <refused|reached|
    timeout|failed|skipped>``; ``skipped`` without either, or without
    ``connect_probe`` (root could not install the IPv4 rules)."""
    connect = shlex.quote(f"exec 3<>/dev/tcp/{_CLOUD_METADATA_PROBE}/80")
    return (
        "conn=skipped; "
        + (
            "if command -v bash >/dev/null 2>&1 && "
            "command -v timeout >/dev/null 2>&1; then "
            f"err=$(timeout 5 bash -c {connect} 2>&1); rc=$?; "
            "case $rc in 0) conn=reached;; 124) conn=timeout;; "
            "*) case $err in *[Rr]efused*) conn=refused;; *) conn=failed;; "
            "esac;; esac; fi; "
            if connect_probe
            else ""
        )
        + 'printf "BF metadata-connect %s\\n" "$conn"'
    )


async def _cloud_metadata_test_connect(
    env: Any, sandbox_user: str | None, *, connect_probe: bool
) -> str:
    """The test connect's result, run as ``sandbox_user`` through the agent's
    own launch, or as root for an agent that runs as root (None); "" when
    its exec failed."""
    command = _cloud_metadata_connect_cmd(connect_probe=connect_probe)
    if sandbox_user is None:
        tcp, _ = await _cloud_metadata_exec(env, command, user="root", timeout_sec=10)
    else:
        tcp, _ = await _cloud_metadata_agent_exec(
            env, command, sandbox_user, timeout_sec=10
        )
    tcp_out = getattr(tcp, "stdout", "")
    tcp_match = re.fullmatch(
        r"BF metadata-connect (refused|reached|timeout|failed|skipped)\n",
        tcp_out if isinstance(tcp_out, str) else "",
    )
    if type(getattr(tcp, "return_code", None)) is int and tcp.return_code == 0:
        return tcp_match[1] if tcp_match else ""
    return ""


def _cloud_metadata_ids_cmd(proc: str = "/proc") -> str:
    """The uid and gid a process started the agent's way runs as, read with
    shell builtins: ``BF metadata-ids <uid> <gid>``. It exits 7 unless the
    real, effective, saved and file-system ids agree."""
    status = shlex.quote(f"{proc}/self/status")
    return (
        "( uid=; gid=; "
        "while read -r k a b c d rest; do case $k in Uid:|Gid:) "
        "case $a in ''|*[!0-9]*) exit 7;; esac; "
        '[ "$a" = "$b" ] && [ "$a" = "$c" ] && [ "$a" = "$d" ] || exit 7; '
        "case $k in Uid:) uid=$a;; *) gid=$a;; esac;; esac; "
        f"done < {status}; "
        '[ -n "$uid" ] && [ -n "$gid" ] || exit 7; '
        'printf "BF metadata-ids %s %s\\n" "$uid" "$gid" )'
    )


def _cloud_metadata_helper_connect_cmd() -> str:
    """The test connect from BenchFlow's metadata helper, a TCP connect and
    nothing more to 169.254.169.254:80: ``BF metadata-connect <refused|
    reached|timeout|failed>``.

    BusyBox's ``nc`` prints nothing when a connect is refused, so the helper
    uses BusyBox's ``wget`` on an ``ftp://`` URL instead (no proxy, an empty
    environment). An FTP client waits for the server's greeting before it
    sends anything, and an HTTP metadata server never sends one: a connect
    that gets through sends no byte and ends ``download timed out`` (or
    ``error getting response`` on a close). A refusal reads
    ``Connection refused``."""
    return (
        "conn=failed; "
        "err=$(timeout 5 env -i /usr/bin/wget -q -Y off -T 3 -O /dev/null "
        f"ftp://{_CLOUD_METADATA_PROBE}:80/ 2>&1); rc=$?; "
        "case $err in *[Rr]efused*) conn=refused;; "
        "*'timed out'*) conn=timeout;; "
        "*'error getting response'*|*'unexpected server response'*) conn=reached;; "
        "*) case $rc in 0) conn=reached;; 124|143) conn=timeout;; esac;; esac; "
        'printf "BF metadata-connect %s\\n" "$conn"'
    )


async def _cloud_metadata_helper_test_connect(
    env: Any, rules: Any, sandbox_user: str | None
) -> tuple[str, str]:
    """The test connect for an image without ``bash`` and ``timeout``, made
    from the metadata helper, which shares main's network namespace, as the
    uid and gid the agent's own launch gives (root's for a root agent): the
    rules match that uid there exactly as for the agent. ``(result, ids)``;
    result "" when it could not be made."""
    if not sandbox_user or sandbox_user in {"root", "0"}:
        ids = "0:0"
    else:
        read, _ = await _cloud_metadata_agent_exec(
            env, _cloud_metadata_ids_cmd(), sandbox_user, timeout_sec=30
        )
        out = getattr(read, "stdout", "")
        found = re.fullmatch(
            r"BF metadata-ids ([0-9]{1,10}) ([0-9]{1,10})\n",
            out if isinstance(out, str) else "",
        )
        code = getattr(read, "return_code", None)
        if found is None or type(code) is not int or code != 0:
            return "", ""
        ids = f"{found[1]}:{found[2]}"
    tcp, failed = await rules.metadata_rules_run(
        _cloud_metadata_helper_connect_cmd(), timeout_sec=30, user=ids
    )
    tcp_out = getattr(tcp, "stdout", "")
    tcp_match = re.fullmatch(
        r"BF metadata-connect (refused|reached|timeout|failed)\n",
        tcp_out if isinstance(tcp_out, str) else "",
    )
    code = getattr(tcp, "return_code", None)
    if failed or tcp_match is None or type(code) is not int or code != 0:
        return "", ids
    return tcp_match[1], ids


#: The ports the reachability proof connects to on every endpoint: the
#: metadata services' (HTTP) and HTTPS. Only a connect that times out or
#: finds no route counts as not reachable.
_CLOUD_METADATA_REACH_PORTS = (80, 443)
_CLOUD_METADATA_UNREACHED = frozenset({"timeout", "unreachable"})


def _cloud_metadata_reach_cmd() -> str:
    """A TCP connect, and nothing more, to every cloud metadata endpoint on
    :data:`_CLOUD_METADATA_REACH_PORTS`, all at once (``bash`` and
    ``timeout`` from the image): one ``BF metadata-reach <address> <port>
    <timeout|unreachable|refused|reached|failed>`` line per target, or
    ``BF metadata-reach skipped`` without either tool. "unreachable" is the
    kernel's no-route answer (ENETUNREACH or EHOSTUNREACH)."""
    targets = " ".join(
        f"{address}/{port}"
        for address in (*CLOUD_METADATA_V4, *CLOUD_METADATA_V6)
        for port in _CLOUD_METADATA_REACH_PORTS
    )
    # BusyBox timeout returns the monitored process's signal status, whereas
    # GNU timeout returns 124. Use a dedicated deadline signal and a marker
    # emitted by our wrapper, not arbitrary SIGTERM/143 as reachability proof.
    # Bash's wait is interruptible: the handler also kills and reaps the socket
    # child when BusyBox signals only the wrapper instead of its process group.
    probe = (
        "child=; "
        'trap \'if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null; '
        'wait "$child" 2>/dev/null; fi; '
        'printf "BF metadata-deadline expired\\n" >&2; exit 124\' USR2; '
        'bash -c \'exec 3<>/dev/tcp/"$1"\' bf-connect "$1" & child=$!; '
        'wait "$child"'
    )
    one = (
        f'err=$(timeout -s USR2 4 bash -c {shlex.quote(probe)} bf-probe "$t" 2>&1); rc=$?; '
        "case $rc in 0) r=reached;; "
        "124) case $err in *'BF metadata-deadline expired') r=timeout;; "
        "*) r=failed;; esac;; "
        "*) case $err in *[Rr]efused*) r=refused;; "
        "*[Uu]nreachable*|*'No route to host'*) r=unreachable;; *) r=failed;; "
        "esac;; esac; "
        'printf "BF metadata-reach %s %s %s\\n" "${t%/*}" "${t##*/}" "$r"'
    )
    return (
        "if command -v bash >/dev/null 2>&1 && "
        "command -v timeout >/dev/null 2>&1; then "
        f"for t in {targets}; do ( {one} ) & done; wait; "
        'else printf "BF metadata-reach skipped\\n"; fi'
    )


def _reach_target(address: str, port: int | str) -> str:
    """How the record names one target: ``a.b.c.d:80``, ``[v6::addr]:80``."""
    return f"[{address}]:{port}" if ":" in address else f"{address}:{port}"


def _cloud_metadata_reach_results(out: Any) -> dict[str, str] | None:
    """``{target: result}`` for every target of :func:`_cloud_metadata_reach_cmd`
    (named by :func:`_reach_target`), or None when the report is not complete
    (a tool missing, a line lost)."""
    wanted = {
        _reach_target(address, port)
        for address in (*CLOUD_METADATA_V4, *CLOUD_METADATA_V6)
        for port in _CLOUD_METADATA_REACH_PORTS
    }
    found: dict[str, str] = {}
    for line in (out if isinstance(out, str) else "").splitlines():
        match = re.fullmatch(
            r"BF metadata-reach (\S+) (\d+) "
            r"(timeout|unreachable|refused|reached|failed)",
            line,
        )
        if match:
            found[_reach_target(match[1], match[2])] = match[3]
    return found if set(found) == wanted else None


async def _cloud_metadata_reach_record(
    env: Any, sandbox_user: str | None, *, root_agent: bool, started: float
) -> dict[str, Any]:
    """The record of a sandbox whose provider gives it no route to the cloud
    metadata endpoints (``cloud_metadata_unreachable``): ``not_reachable``
    when a connect to every endpoint and port, as ``sandbox_user`` through the
    agent's own launch (as root for a root agent), timed out or found no
    route; ``unproven`` (which stops the trial) for any other outcome, a
    missing tool or a failed exec. Nothing is installed either way."""
    command = _cloud_metadata_reach_cmd()
    if root_agent:
        result, failed = await _cloud_metadata_exec(
            env, command, user="root", timeout_sec=60
        )
    else:
        assert sandbox_user is not None
        result, failed = await _cloud_metadata_agent_exec(
            env, command, sandbox_user, timeout_sec=60
        )
    out = getattr(result, "stdout", "")
    code = getattr(result, "return_code", None)
    probes = _cloud_metadata_reach_results(out)
    who = "root" if root_agent else f"the agent's user ({sandbox_user})"
    extra: dict[str, Any] = {"root_agent": True} if root_agent else {}
    seconds = round(_timing_spans.clock() - started, 3)
    if failed or type(code) is not int or code != 0 or probes is None:
        why = failed or (
            "the image has no bash and timeout to connect with"
            if "BF metadata-reach skipped" in (out if isinstance(out, str) else "")
            else "the connects did not all report"
        )
        return cloud_metadata_record(
            "unproven",
            f"the sandbox's provider gives it no route to the cloud metadata "
            f"endpoints, but this trial could not prove it ({why}), so the "
            f"agent could reach {cloud_metadata_list()}",
            **extra,
            seconds=seconds,
        )
    reached = sorted(t for t, r in probes.items() if r not in _CLOUD_METADATA_UNREACHED)
    if reached:
        return cloud_metadata_record(
            "unproven",
            "the sandbox's provider is known to give it no route to the cloud "
            f"metadata endpoints, but a connect as {who} got an answer from "
            f"{', '.join(f'{t} ({probes[t]})' for t in reached)}, so the agent "
            f"could reach {cloud_metadata_list()}",
            probes=probes,
            **extra,
            seconds=seconds,
        )
    return cloud_metadata_record(
        NOT_REACHABLE,
        "the sandbox's provider gives it no route to the cloud metadata "
        f"endpoints: a TCP connect as {who} to each of them on ports "
        f"{' and '.join(map(str, _CLOUD_METADATA_REACH_PORTS))} timed out or "
        "found no route, so no block is needed",
        probes=probes,
        **extra,
        seconds=seconds,
    )


async def block_cloud_metadata(env: Any, sandbox_user: str | None) -> dict[str, Any]:
    """Refuse the cloud metadata endpoints to ``sandbox_user`` (and every
    other non-root user; to root too for an agent that runs as root, None)
    before the agent starts; return the record
    (:func:`_block_cloud_metadata`). A sandbox's metadata helper, when it
    has one, is removed once the block is done, whatever its outcome."""
    try:
        return await _block_cloud_metadata(env, sandbox_user)
    finally:
        rules = _metadata_rules(env)
        if rules is not None:
            rules.metadata_rules_close()


def _metadata_rules(env: Any) -> Any:
    """The sandbox when it can install the block's rules from a helper
    (``metadata_rules_install``, Docker), else None. Looked up on its type,
    so a mocked sandbox never grows one."""
    install = getattr(type(env), "metadata_rules_install", None)
    return env if callable(install) else None


async def _block_cloud_metadata(env: Any, sandbox_user: str | None) -> dict[str, Any]:
    """Refuse the cloud metadata endpoints to ``sandbox_user`` (and every
    other non-root user) before the agent starts; return the record.

    Root's rules go into main's network namespace. On a sandbox with a
    metadata helper (Docker: ``sandbox._metadata_helper``), root's script runs
    in BenchFlow's helper container that shares that namespace and brings its
    own iptables, so main gets nothing installed; elsewhere, or when the
    helper cannot run, it runs in main and installs iptables there. The
    rule's packet counter is read where the rules were added. The record's
    ``rules_from`` says which (``helper`` or ``sandbox``).

    It is ``on`` only when the agent user's test connect to
    169.254.169.254:80 was refused and the rule counted it, so a host's own
    block never passes for this one; ``unproven`` otherwise, and
    ``not_installed`` (with why) when the rules could not go in. The rollout
    starts the agent only on ``on`` unless the run opted out.

    An agent that runs as root (no ``sandbox_user``, or root) is held too:
    the rules then refuse root as well, and they hold only where root in
    main has neither NET_ADMIN, which could remove them, nor NET_RAW, whose
    packet sockets send around them. Root's capability sets are read first;
    a main whose root holds either is ``not_installed``. The proof is root's
    own test connect. (Docker's helper adds the rules from outside such a
    main; elsewhere root holds NET_ADMIN, and a root agent is not held.)

    Raises :class:`CloudMetadataPowerError` when the agent user holds
    NET_ADMIN, a firewall tool lets it change the rules, or the kernel
    privilege construction cannot be verified. Missing optional inventory
    tools are recorded rather than preventing an honest image from starting.
    """
    started = _timing_spans.clock()
    root_agent = not sandbox_user or sandbox_user in {"root", "0"}
    if getattr(type(env), "cloud_metadata_unreachable", False) is True:
        # The provider's sandboxes have no route to the endpoints (Modal,
        # measured): no rule is installed, and the trial proves it.
        reach = await _cloud_metadata_reach_record(
            env, sandbox_user, root_agent=root_agent, started=started
        )
        # A root agent whose proof did not hold keeps the provider's own
        # policy below, as before; a non-root agent is refused.
        if reach["state"] == NOT_REACHABLE or not root_agent:
            return reach
    if root_agent and _metadata_rules(env) is None:
        if getattr(type(env), "cloud_metadata_provider_managed", False) is not True:
            # Not one of the providers BenchFlow manages (an SDK caller's own
            # sandbox class, say): no helper, so root keeps the power to
            # remove any rule, and no provider's policy is known to apply.
            return cloud_metadata_record(
                "not_installed",
                "the agent runs as root on a sandbox that has no BenchFlow "
                f"metadata helper and is not a known provider "
                f"({type(env).__name__}), so root could remove any rule; none "
                f"is installed, and the agent can reach {cloud_metadata_list()}",
                root_agent=True,
                seconds=round(_timing_spans.clock() - started, 3),
            )
        # A sandbox provider (Daytona, Modal, AgentCore, Apple): no helper can
        # add rules a root agent cannot touch, and the host and its metadata
        # service are the provider's.
        return cloud_metadata_record(
            PROVIDER_MANAGED,
            "the agent runs as root on a sandbox provider whose host and "
            "metadata service are the provider's; its own metadata policy "
            "applies, and BenchFlow installs no rule",
            root_agent=True,
            seconds=round(_timing_spans.clock() - started, 3),
        )
    restarts = getattr(type(env), "metadata_main_restarts", None)
    if _metadata_rules(env) is not None or callable(restarts):
        # The rules live in main's network namespace, and a restarted main
        # gets a new one without them: the block holds only a main that
        # cannot restart (docker-compose-no-restart.yaml). Docker and
        # Daytona DinD read the policy; a helper sandbox must.
        why_restarts = (
            await env.metadata_main_restarts()
            if callable(restarts)
            else "main's restart policy cannot be read on this sandbox"
        )
        if why_restarts:
            return cloud_metadata_record(
                "not_installed",
                f"{why_restarts}, so no rule could hold, and the agent can "
                f"reach {cloud_metadata_list()}",
                **({"root_agent": True} if root_agent else {}),
                seconds=round(_timing_spans.clock() - started, 3),
            )
    root_caps: dict[str, str] = {}
    if root_agent:
        root_caps, unfit = await _cloud_metadata_root_caps(env)
        if unfit:
            helper = _metadata_rules(env)
            helper_why = helper.metadata_rules_unavailable if helper else ""
            if helper_why:
                # Main kept NET_ADMIN because no helper could add the rules.
                unfit += f" (BenchFlow's metadata helper is unavailable: {helper_why})"
            return cloud_metadata_record(
                "not_installed",
                f"the agent runs as root, and {unfit}, so it could remove or "
                "go around any rule; none is installed, and the agent can "
                f"reach {cloud_metadata_list()}",
                root_agent=True,
                root_capabilities=root_caps,
                seconds=round(_timing_spans.clock() - started, 3),
            )
    else:
        assert sandbox_user is not None
        construction, construction_failed = await _cloud_metadata_exec(
            env,
            _cloud_metadata_construction_cmd(sandbox_user),
            user="root",
            timeout_sec=60,
        )
        construction_out = getattr(construction, "stdout", "")
        construction_code = getattr(construction, "return_code", None)
        if construction_failed or not _cloud_metadata_barrier_verified(
            construction_out if isinstance(construction_out, str) else "",
            construction_code,
        ):
            raise CloudMetadataPowerError(
                "sandbox startup refused: the agent privilege construction could "
                "not be verified from /proc as root. The shared launch must set "
                "NoNewPrivs, remove NET_ADMIN and NET_RAW from CapBnd, and clear "
                "CapAmb. Provide util-linux setpriv and a readable /proc."
            )
    block_cmd = _cloud_metadata_block_cmd(cover_root=root_agent)
    rules = _metadata_rules(env)
    from_helper = (
        await rules.metadata_rules_install(block_cmd) if rules is not None else None
    )
    helper_failed = ""
    while True:
        if from_helper is not None:
            result, failed = from_helper
        else:
            result, failed = await _cloud_metadata_exec(
                env,
                block_cmd,
                user="root",
                timeout_sec=_CLOUD_METADATA_EXEC_SEC,
            )
        out = getattr(result, "stdout", "") or ""
        out = out if isinstance(out, str) else ""
        line = next(
            (ln for ln in out.splitlines() if ln.startswith("BF metadata ")), ""
        )
        fields = _report_fields(line)
        status = fields.get("status", "")
        if from_helper is None or (not failed and status == "0"):
            break
        # Never a helper's failed install: only one whose script added every
        # rule serves the block; otherwise root's script runs in main, with
        # the image's own tools, as before.
        helper_failed = failed or (
            "the metadata firewall helper could not add the rules "
            f"(status {status or 'missing'})"
        )
        from_helper = None
    # Only names this module knows are ever run as root later.
    fw4 = fields.get("fw4", "")
    fw6 = fields.get("fw6", "")
    extra: dict[str, Any] = {
        "iptables": fw4 if fw4 in _CLOUD_METADATA_TOOLS4 else None,
        "ip6tables": fw6 if fw6 in _CLOUD_METADATA_TOOLS6 else None,
        "iptables_installed": fields.get("install") == "ok",
    }
    if root_agent:
        # Read before the install: neither NET_ADMIN nor NET_RAW.
        extra["root_agent"] = True
        extra["root_capabilities"] = root_caps
    if rules is not None:
        # Not where the helper adds the rules from outside main.
        extra["main_net_admin"] = not getattr(
            getattr(env, "task_env_config", None), "cloud_metadata_by_helper", False
        )
    if from_helper is not None:
        extra["rules_from"] = "helper"
    elif rules is not None and (helper_failed or rules.metadata_rules_unavailable):
        # The helper could not serve the block: root installed in main, as before.
        extra["rules_from"] = "sandbox"
        extra["helper_unavailable"] = helper_failed or rules.metadata_rules_unavailable
    seconds = r"([0-9]+(?:\.[0-9]+)?)"
    span = re.fullmatch(f"{seconds},{seconds}", fields.get("install_t", ""))
    if span:
        extra["install_seconds"] = round(float(span[2]) - float(span[1]), 2)
    # NET_ADMIN was granted to main before installation. A rejected tool
    # (status 6), a failed install or even a root exec timeout does not revoke
    # it: the image can still pass it to the agent (S2-CONF-3 at a070066d).
    # Always check that power before returning, without a TCP probe when
    # root could not install the IPv4 rules.
    probe = f"BFMD{secrets.token_hex(4)}"
    recount_failed = ""
    if from_helper is not None and status in ("0", "3"):
        # The helper added the rules early, while the rest of the sandbox
        # started; count from here, so only the agent user's test connect
        # below can raise the counter the proof reads, as when root installs
        # right before it. Without that count the block is not proven.
        recount, recount_failed = await rules.metadata_rules_run(
            _cloud_metadata_after_cmd(extra["iptables"] or "iptables", probe),
            timeout_sec=60,
        )
        counted = re.search(
            r"BF metadata-hits (\d+)", getattr(recount, "stdout", "") or ""
        )
        if counted:
            fields["hits"] = counted.group(1)
        else:
            recount_failed = recount_failed or "the helper printed no count"
    check: Any = None
    if root_agent:
        # Root's own test connect. Its sets, read before the install, hold
        # neither NET_ADMIN nor NET_RAW: no tool or delegate can give root
        # the power to change or go around the rules.
        agent: dict[str, str] = {
            "connect": await _cloud_metadata_test_connect(
                env, None, connect_probe=status in ("0", "3")
            )
        }
        extra["agent_privilege_audit"] = "complete"
    else:
        assert sandbox_user is not None
        check, check_failed = await _cloud_metadata_agent_exec(
            env,
            _cloud_metadata_agent_check_cmd(probe, connect_probe=status in ("0", "3")),
            sandbox_user,
            timeout_sec=60,
        )
        check_out = getattr(check, "stdout", "") or ""
        check_out = check_out if isinstance(check_out, str) else ""
        agent_lines = [
            ln for ln in check_out.splitlines() if ln.startswith("BF metadata-agent ")
        ]
        # This is a safety prerequisite, not optional proof of the TCP block.
        # Exactly one complete report and a successful exec are required; a
        # late timeout cannot erase a tool's earlier positive power result.
        tool_names = "|".join(
            _CLOUD_METADATA_TOOLS4 + _CLOUD_METADATA_TOOLS6 + ("nft",)
        )
        agent_match = re.fullmatch(
            rf"BF metadata-agent held=(?P<held>[01]) "
            rf"power=(?P<power>(?:{tool_names})(?:,(?:{tool_names}))*)? "
            r"connect=(?P<connect>refused|reached|timeout|failed|skipped)",
            agent_lines[0] if len(agent_lines) == 1 else "",
        )
        agent = agent_match.groupdict(default="") if agent_match else {}
        check_code = getattr(check, "return_code", None)
        audit_ok = (
            not check_failed
            and type(check_code) is int
            and check_code == 0
            and agent_match is not None
        )
        extra["agent_privilege_audit"] = "complete" if audit_ok else "unavailable"
        if not audit_ok:
            # The construction above prevents exec-time privilege gain. Missing
            # optional inspection tools must not suppress the independent TCP
            # proof, or make an ordinary image fail startup.
            agent["connect"] = await _cloud_metadata_test_connect(
                env, sandbox_user, connect_probe=status in ("0", "3")
            )
    if (
        agent.get("connect") == "skipped"
        and status in ("0", "3")
        and from_helper is not None
    ):
        # No bash and timeout in the image (Alpine, BusyBox): the helper
        # makes the test connect, as the agent's uid, in main's namespace.
        agent["connect"], ids = await _cloud_metadata_helper_test_connect(
            env, rules, sandbox_user
        )
        extra["test_connect_from"] = "helper"
        if ids:
            extra["test_connect_ids"] = ids
    after_cmd = _cloud_metadata_after_cmd(extra["iptables"] or "iptables", probe)
    if from_helper is not None:
        # Where the rules are: the helper's tools, in main's namespace.
        after, after_failed = await rules.metadata_rules_run(after_cmd, timeout_sec=60)
    else:
        after, after_failed = await _cloud_metadata_exec(
            env, after_cmd, user="root", timeout_sec=60
        )
    if recount_failed and not after_failed:
        after_failed = f"the count before the test connect: {recount_failed}"
    after_out = getattr(after, "stdout", "") or ""
    after_out = after_out if isinstance(after_out, str) else ""
    m = re.search(r"BF metadata-hits (\d+)", after_out)
    before = fields.get("hits", "")
    hits = int(m.group(1)) - (int(before) if before.isdecimal() else 0) if m else 0
    if agent.get("held") == "1" or agent.get("power"):
        what = (
            f"a firewall tool ({agent['power'].replace(',', ', ')})"
            if agent.get("power")
            else "its own process"
        )
        raise CloudMetadataPowerError(
            "sandbox startup refused: cannot rule out that the agent user "
            f"{sandbox_user!r} can change the firewall through {what}. "
            "BenchFlow grants NET_ADMIN only "
            "to refuse the cloud metadata endpoints; file capabilities and "
            "setuid/setgid firewall tools could let the agent remove that "
            "refusal. Remove the capabilities or setuid/setgid bits in the "
            "task's Dockerfile (setcap -r <file>, chmod ug-s <file>), or "
            f"{CLOUD_METADATA_OPT_OUT_HINT}."
        )
    # Named probes cannot cover an arbitrary trusted delegation graph. Its
    # privilege sources must be absent even when every mutation probe fails.
    # A root agent is root already, and nothing adds to a bounding set that
    # holds neither NET_ADMIN nor NET_RAW: no delegate to look for.
    if root_agent:
        extra["delegate_privilege_audit"] = "not_needed"
    else:
        delegates, delegates_failed = await _cloud_metadata_exec(
            env, _cloud_metadata_delegate_check_cmd(), user="root", timeout_sec=120
        )
        delegate_out = getattr(delegates, "stdout", "")
        delegate_code = getattr(delegates, "return_code", None)
        delegate_ok = (
            not delegates_failed
            and type(delegate_code) is int
            and delegate_code == 0
            and delegate_out
            in ("BF metadata-delegates power=0\n", "BF metadata-delegates power=1\n")
        )
        extra["delegate_privilege_audit"] = "complete" if delegate_ok else "unavailable"
        if delegate_out == "BF metadata-delegates power=1\n":
            raise CloudMetadataPowerError(
                "sandbox startup refused: a firewall delegate may acquire root "
                "or NET_ADMIN, or its file privilege inventory did not finish. "
                "Provide trusted find and getcap utilities, remove powerful file "
                "capabilities and setuid/setgid command or identity launchers "
                "(including sudo, doas, su and pkexec) and other nonstandard "
                f"privileged helpers from the image, or {CLOUD_METADATA_OPT_OUT_HINT}."
            )
    extra["seconds"] = round(_timing_spans.clock() - started, 3)
    if result is None:
        return cloud_metadata_record(
            "not_installed",
            f"root's firewall script did not finish in {_CLOUD_METADATA_EXEC_SEC} s "
            f"or its exec failed ({failed}), so the agent can reach "
            f"{cloud_metadata_list()}",
            **extra,
        )
    if status not in ("0", "3"):
        why = _CLOUD_METADATA_NOT_INSTALLED.get(status, "the firewall did not report")
        if fields.get("install") == "timeout":
            why = (
                "installing iptables did not finish in "
                f"{_CLOUD_METADATA_INSTALL_SEC} s and was stopped"
            )
        install = next(
            (ln for ln in out.splitlines() if ln.startswith("BF metadata-install:")), ""
        )
        if status == "5" and install:
            why += f" ({install.split(':', 1)[1].strip()})"
        if not line:
            why += _exec_failure_detail(result)
        if extra.get("helper_unavailable"):
            why += (
                " (BenchFlow's metadata helper could not add them either: "
                f"{extra['helper_unavailable']})"
            )
        return cloud_metadata_record(
            "not_installed",
            f"{why}, so the agent can reach {cloud_metadata_list()}",
            **extra,
        )
    who = "every user, root included," if root_agent else "every non-root user"
    what = (
        f"the uid firewall refuses TCP from {who} to "
        f"{cloud_metadata_list()} on every port but 53"
    )
    if status == "3":
        return cloud_metadata_record(
            "not_installed",
            f"{_CLOUD_METADATA_NOT_INSTALLED['3']}, so the agent can reach "
            f"{', '.join(CLOUD_METADATA_V6)}",
            **extra,
        )
    connect = agent.get("connect", "")
    if after_failed:
        return cloud_metadata_record(
            "unproven",
            f"{what}; not proven: reading the rule's counter did not finish "
            f"({after_failed})",
            **extra,
        )
    actor = "root's" if root_agent else "the agent user's"
    if connect == "refused" and hits > 0:
        return cloud_metadata_record(
            "on",
            f"{what}; proof: {actor} connect to "
            f"{_CLOUD_METADATA_PROBE}:80"
            + (
                f" (made from BenchFlow's helper as {extra['test_connect_ids']})"
                if extra.get("test_connect_ids")
                else ""
            )
            + " was refused and the rule counted "
            f"{hits} packet{'' if hits == 1 else 's'}",
            **extra,
        )
    if connect == "skipped":
        reason = (
            f"the image has no bash and timeout for {actor} test connect, "
            "and no BenchFlow metadata helper served the rules to make it"
        )
    elif connect == "refused":
        reason = (
            f"{actor} connect to {_CLOUD_METADATA_PROBE}:80 was refused, "
            "but not by this rule (its packet counter did not rise)"
        )
    elif connect:
        reason = (
            f"{actor} connect to {_CLOUD_METADATA_PROBE}:80 was not "
            f"refused by the rule ({connect})"
        )
    elif root_agent:
        reason = "root's test connect did not report"
    else:
        reason = "the agent user's check did not report" + _exec_failure_detail(check)
    return cloud_metadata_record("unproven", f"{what}; not proven: {reason}", **extra)


def _legacy_root_tool_link_cmd(source: str, dest: str) -> str:
    """Link legacy root-only tool dirs into the sandbox home when needed."""
    src = shlex.quote(source)
    dst = shlex.quote(dest)
    parent = shlex.quote(str(Path(dest).parent))
    return (
        f"if [ -e {src} ] && [ ! -L {dst} ]; then "
        f"mkdir -p {parent} && "
        f"rmdir {dst} 2>/dev/null || true; "
        f"[ -e {dst} ] || ln -s {src} {dst}; "
        "fi"
    )


# The standard system folders, appended after the image's PATH when the
# folders on it are locked: a root command that the image's PATH does not
# resolve falls back to these (Rust 0.8.2, 68f20d2c).
_SYSTEM_PATH_FOLDERS = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Temp folders on PATH are left alone (anyone may write them by design), as
# the verifier's PATH leaves them out.
_PATH_LOCK_SKIP = ("/tmp", "/var/tmp", "/dev/shm")
# The walk up from each folder stops here (tests point it at their temp root).
_PATH_LOCK_TOP = "/"


def _lock_system_path_cmd(workspace: str, *, dry: bool = False) -> str:
    """Root: lock the folders on PATH (and the folders above them), and the
    programs in them, that someone other than root could write, before the
    agent starts.

    Every command BenchFlow sends as root resolves ``sh`` and the tools it
    calls through this PATH, and the verifier's PATH keeps the image's
    folders root alone can write (``_trusted_verifier_path``). An agent user
    that can write one of them could replace a program there and have root
    run it, or (for a folder the image left open) find the verifier without
    the tools the task put there. The image's last PATH folder and the
    appended system folders are split as one string, so neither stays glued
    to the other (dash and current bash split only what an expansion gives).
    Folders in the workspace, homes and temp folders are left alone, as the
    verifier's PATH leaves them out.

    A program directly in a PATH folder is locked too when group or others
    may write it (the official rust image's 0777 ``cargo``) or a user other
    than root owns it (a tool unpacked as root from an archive keeps the
    archive's uid, often 1000, the agent user's): a locked folder would
    otherwise still hold a program the agent can rewrite in place. A link is
    judged by the file it resolves to, and the folders each hop of its chain
    sits in (``py -> /opt/venv/bin/python -> python3``: ``/opt/venv/bin``
    too) are locked like PATH's own; a hop in the workspace, a home or a temp
    folder cannot be locked and is reported, and so is a link to a file that
    does not exist where the agent could make it. Prints ``BF locked-path
    <path>`` for each folder or program it locked (a link as ``<link> ->
    <file>``) and ``BF unlockable-path <path>`` for one it could not lock.
    POSIX ``sh`` (dash, busybox): no Python. *dry*: change nothing, and
    print ``BF locked-path`` and its device and inode for each one it would
    lock. An unreadable identity makes the early check defer to the seal.
    """
    skip = "".join(f"|{shlex.quote(p)}|{shlex.quote(p)}/*" for p in _PATH_LOCK_SKIP)
    return (
        f"w={shlex.quote(workspace.rstrip('/') or '/')}\n"
        f"top={shlex.quote(_PATH_LOCK_TOP)}\n"
        # bf_done: report one lock, or a lock whose chown or chmod failed.
        'bf_done() { if [ -n "$2" ]; then echo "BF locked-path $1"; '
        'else echo "BF unlockable-path $1"; fi; }\n'
        # bf_skip: the workspace, homes and temp folders, left alone.
        f'bf_skip() {{ case "$1" in "$w"|"$w"/*|/home/*{skip}) return 0;; '
        "esac; return 1; }\n"
        # bf_bad: group or others may write it, or a user other than root owns it.
        'bf_bad() { m=$(stat -c %A "$1" 2>/dev/null); '
        'u=$(stat -c %u "$1" 2>/dev/null)\n'
        + (
            '  [ -n "$m" ] && [ -n "$u" ] || '
            '{ echo "BF lock-identity unknown"; return 1; }\n'
            if dry
            else ""
        )
        + '  case "$m" in ?????w????|????????w?) return 0;; esac\n'
        '  [ -n "$u" ] && [ "$u" != 0 ]; }\n'
        # bf_lock <path> <name>: root's, and only root may write it.
        + (
            'bf_lock() { bf_i=$(stat -Lc "%d:%i" "$1" 2>/dev/null) || bf_i=unknown; '
            'printf "BF lock-identity %s\\n" "$bf_i"; bf_done "$2" 1; }\n'
            if dry
            else "bf_lock() { ok=1\n"
            '  if [ "$(stat -c %u "$1" 2>/dev/null)" != 0 ]; then '
            'chown root:root "$1" 2>/dev/null || ok=; fi\n'
            '  chmod go-w "$1" 2>/dev/null || ok=; bf_done "$2" "$ok"; }\n'
        )
        # bf_walk <folder> [link]: lock it and the folders above it. A folder
        # a link resolves into that is left alone is reported when it is bad.
        + "bf_walk() { bd=$1\n"
        "  while :; do\n"
        '    if [ -L "$bd" ]; then bf_done "$bd" ""\n'
        '    elif bf_skip "$bd"; then\n'
        '      [ -n "$2" ] && bf_bad "$bd" && bf_done "$bd" ""\n'
        '    elif bf_bad "$bd"; then bf_lock "$bd" "$bd"; fi\n'
        '    [ "$bd" = / ] || [ "$bd" = "$top" ] && break\n'
        '    bd=$(dirname "$bd")\n'
        "  done; }\n"
        # bf_res <path>: the path with every link in it resolved, the last
        # parts allowed to be missing (busybox has no readlink -m).
        'bf_res() { readlink -m "$1" 2>/dev/null || readlink -f "$1" 2>/dev/null '
        '|| { bh=$(bf_have "$1"); bt=$(readlink -f "$bh" 2>/dev/null) || return 1; '
        'printf \'%s%s\\n\' "$bt" "${1#"$bh"}"; }; }\n'
        # bf_have <path>: the path, or its first folder above that exists.
        'bf_have() { r=$1; while [ ! -e "$r" ] && [ "$r" != / ]; do '
        "r=${r%/*}; r=${r:-/}; done; printf '%s\\n' \"$r\"; }\n"
        # bf_hops <link> <its real folder> <depth>: the real folder of every
        # hop on the link's chain, one hop at a time (whoever may write a
        # middle hop's folder can put another file in the chain), the first
        # folder that exists for one that does not, and a link among the
        # folders a hop's path names is followed the same way. A chain that
        # ends in nothing, where the agent could make the file (the
        # workspace, a home, a temp folder), prints "!<link> -> <file>".
        # A hop's folder is resolved once (bf_lp, bf_lf): the 300 links in a
        # busybox image's /bin all point into one folder, and a fork per
        # link per step would add seconds.
        "bf_hops() { hc=$1; hf=$2; hn=0\n"
        '  while [ -L "$hc" ] && [ "$hn" -lt 40 ]; do hn=$((hn + 1))\n'
        '    hx=$(readlink "$hc" 2>/dev/null) || break\n'
        '    case "$hx" in\n'
        "      /*) ;;\n"
        "      */*) hx=${hf%/}/$hx;;\n"
        "      *) hc=${hf%/}/$hx; continue;;\n"
        "    esac\n"
        "    hp=${hx%/*}; hp=${hp:-/}\n"
        '    if [ "$hp" != "$bf_lp" ]; then bf_lp=$hp; bf_lf=$(bf_res "$hp")\n'
        '      [ "$bf_lf" = "$hp" ] || bf_dirs "$hp" "$3"\n'
        '      bf_have "$bf_lf"\n'
        "    fi\n"
        "    hf=$bf_lf; hc=${hf%/}/${hx##*/}\n"
        "  done\n"
        '  if [ "$hn" -gt 0 ] && [ ! -e "$hc" ]; then ha=$(bf_have "$hc")\n'
        '    bf_skip "$ha" && bf_bad "$ha" && printf \'!%s -> %s\\n\' "$1" "$hc"\n'
        "  fi; true; }\n"
        # bf_dirs <path> <depth>: the links among the folders a path names,
        # each followed by bf_hops (a few levels deep at most). A link past
        # that depth is reported, never passed over: the folder holding it
        # is not checked, so the route cannot be called locked (PT-CONF-4).
        "bf_dirs() { da=$1\n"
        '  while [ -n "$da" ] && [ "$da" != / ]; do\n'
        '    if [ -L "$da" ] && [ "$2" -ge 4 ]; then printf \'!%s\\n\' "$da"\n'
        '    elif [ -L "$da" ]; then dp=${da%/*}; dp=$(bf_res "${dp:-/}")\n'
        '      bf_have "$dp"; (bf_hops "$da" "$dp" $(($2 + 1)))\n'
        "    fi\n"
        "    da=${da%/*}\n"
        "  done; }\n"
        "IFS=:\n"
        f'bf_p="${{_BENCHFLOW_IMAGE_PATH-${{PATH:-}}}}:{_SYSTEM_PATH_FOLDERS}"\n'
        "bf_seen=; bf_lp=; bf_lf=\n"
        "for d in $bf_p; do\n"
        '  case "$d" in /*) ;; *) continue;; esac\n'
        '  [ -d "$d" ] || continue\n'
        '  bf_dirs "$d" 0 | sort -u | while IFS= read -r td; do\n'
        '    case "$td" in "!"*) bf_done "${td#!}" "";; *) bf_walk "$td" link;; esac\n'
        "  done\n"
        '  d=$(readlink -f "$d" 2>/dev/null || printf %s "$d")\n'
        '  case "$bf_seen" in *"|$d|"*) continue;; esac\n'
        '  bf_seen="$bf_seen|$d|"\n'
        '  bf_walk "$d" ""\n'
        '  bf_skip "$d" && continue\n'
        # The programs in the PATH folder itself, each link judged by its
        # file (find -L): one find per folder (a stat per file would take
        # seconds in /usr/bin).
        '  find -L "$d" -maxdepth 1 -type f \\( ! -user 0 -o -perm -0002 '
        "-o -perm -0020 \\) 2>/dev/null | while IFS= read -r e; do\n"
        '    t=$(readlink -f "$e" 2>/dev/null) || continue\n'
        '    if [ "$t" = "$e" ]; then n=$e; else n="$e -> $t"; fi\n'
        '    if bf_skip "$t"; then bf_done "$n" ""; else bf_lock "$t" "$n"; fi\n'
        "  done\n"
        # The folders every hop of the links in it sits in: whoever may write
        # one can put another file in a hop's place.
        '  find "$d" -maxdepth 1 -type l 2>/dev/null | while IFS= read -r e; do\n'
        '    bf_hops "$e" "$d" 0\n'
        "  done | sort -u | while IFS= read -r td; do\n"
        '    case "$td" in "!"*) bf_done "${td#!}" "";; "$d") ;; '
        '*) bf_walk "$td" link;; esac\n'
        "  done\n"
        "done\n" + ('echo "BF dry-path-complete"' if dry else "true")
    )


def _path_lock_report(stdout: str) -> tuple[list[str], list[str]]:
    """The folders ``_lock_system_path_cmd`` locked, and the ones it could not."""
    locked: list[str] = []
    unlockable: list[str] = []
    for line in stdout.splitlines():
        for prefix, into in (
            ("BF locked-path ", locked),
            ("BF unlockable-path ", unlockable),
        ):
            if line.startswith(prefix):
                folder = line[len(prefix) :].strip()
                if folder and folder not in into:
                    into.append(folder)
    return locked, unlockable


async def lock_system_path(env, workspace: str, *, timeout_sec: int = 120) -> None:
    """Lock the system folders on PATH to root (see ``_lock_system_path_cmd``).

    A folder whose lock failed is logged and recorded ``failed``; the trial
    goes on, as for the other pre-agent defenses.
    """
    from benchflow._utils.path_lock_record import record_path_lock
    from benchflow.sandbox._base import require_system_shell

    # The check's own tools come from the system shell (C8): run on the
    # image's PATH, the agent's ``readlink`` would judge its own folder.
    # Its report is read from a receipt, as every lockdown read is.
    require_system_shell(env, "The check of the folders on PATH")
    result = await _read_exec(
        env,
        _lock_system_path_cmd(workspace),
        user="root",
        system_shell=True,
        timeout_sec=timeout_sec,
    )
    locked, unlockable = _path_lock_report(getattr(result, "stdout", "") or "")
    if _exec_return_code(result) != 0 and not (locked or unlockable):
        detail = _exec_failure_detail(result)
        logger.warning(f"Could not check the folders on PATH.{detail}")
        unlockable = ["(the check did not run)"]
    if locked:
        logger.info(
            "Locked to root before the agent ran (the image let others write them): %s",
            ", ".join(locked),
        )
    if unlockable:
        logger.warning(
            "Folders on PATH the agent user may write, which could not be "
            "locked to root: %s",
            ", ".join(unlockable),
        )
    record_path_lock(locked, unlockable)
    # Root's plumbing pins /usr/local/sbin and /usr/local/bin (C8) only while
    # nothing says the agent user can write them: the lock left them open, it
    # did not run, or they are in the workspace, which the lock leaves alone.
    ws = workspace.rstrip("/") or "/"
    distrusted = [
        d
        for d in ("/usr/local/sbin", "/usr/local/bin")
        if d in unlockable
        or "(the check did not run)" in unlockable
        or ws == "/"
        or d == ws
        or d.startswith(ws + "/")
    ]
    distrust = getattr(env, "distrust_root_path_folders", None)
    if distrusted and callable(distrust):
        logger.warning(
            "BenchFlow's root steps leave %s off their PATH from now on",
            ", ".join(distrusted),
        )
        distrust(distrusted)


class LoaderCustodyError(RuntimeError):
    """The task's image lets a user other than root change a library root's
    steps load (a system loader file or folder that is not root's alone).

    The message starts with "sandbox startup refused", so the trial is a
    sandbox-setup failure of the image's, not run with a verifier whose
    libraries the agent could swap.
    """


#: What to do about an image the loader check refuses.
_LOADER_NEXT = (
    "make that path root's in the task's image (in its Dockerfile, "
    "RUN chown -R root:root <path> && chmod -R go-w <path>), or run the "
    "agent as root with --sandbox-user none"
)


def _loader_refusal(result: Any) -> str | None:
    """The loader proof's refusal in *result*'s stderr, up to "which is not
    root's alone", or None when the failure is another one."""
    found = re.search(
        r"benchflow: (the system loader (?:reads|searches|loads) .*?), which is "
        r"not root's alone",
        getattr(result, "stderr", "") or "",
    )
    return found.group(1) if found else None


def _own_fault(loader: str) -> str | None:
    """The path a loader refusal names, when its own owner or mode is what
    fails (not a folder above it, nor a link's target); else None."""
    found = re.fullmatch(
        r"the system loader (?:reads|searches|loads) (/.*) \(owner (\d+):\d+, "
        r"mode ([-a-zA-Z]{10})\S*\)",
        loader,
    )
    if found is None:
        return None
    path, uid, mode = found.groups()
    if uid != "0" or (mode[0] != "l" and "w" in (mode[5], mode[8])):
        return path
    return None


async def check_loader_custody(env, workspace: str, *, timeout_sec: int = 120) -> None:
    """Refuse, before the agent's install, an image whose system loader
    files or folders a user other than root owns or may write, which the
    seal (:func:`prepare_runtime_tree`) would refuse after the install.

    Runs the seal's loader proof on the image as built
    (:func:`benchflow._runtime_tree.loader_check_command`). It refuses only
    what the seal is sure to refuse too: a path whose own owner or mode
    fails, which the PATH lock (it runs before the seal, and repairs the
    folders on PATH, the folders above them and their links' folders)
    would not repair, by its own report in a run that changes nothing.
    Anything else, and a check that cannot run, decides nothing here; the
    seal decides after the workspace grant."""
    from benchflow._runtime_tree import loader_check_command

    if (
        getattr(env, "supports_runtime_kit", False) is not True
        or getattr(env, "supports_system_shell", False) is not True
    ):
        return
    result = await _read_exec(
        env, loader_check_command(), user="root", timeout_sec=timeout_sec
    )
    if _exec_return_code(result) == 0:
        return
    loader = _loader_refusal(result)
    path = _own_fault(loader) if loader is not None else None
    if path is None:
        return
    q = shlex.quote(path)
    try:
        real = await _read_exec(
            env, f"readlink -f {q} && stat -Lc '%d:%i' {q}", user="root", timeout_sec=30
        )
        dry = await _read_exec(
            env,
            _lock_system_path_cmd(workspace, dry=True),
            user="root",
            system_shell=True,
            timeout_sec=timeout_sec,
        )
    except Exception:
        # An unavailable repair report cannot justify an early refusal.
        logger.debug("loader repair report could not be read", exc_info=True)
        return
    if _exec_return_code(real) != 0 or _exec_return_code(dry) != 0:
        return
    spelling = (getattr(real, "stdout", "") or "").splitlines()
    report = getattr(dry, "stdout", "") or ""
    identities = [
        line.removeprefix("BF lock-identity ")
        for line in report.splitlines()
        if line.startswith("BF lock-identity ")
    ]
    # A partial or ambiguous report cannot prove that the lock will leave
    # the loader entry alone. In that case only the final seal decides.
    if (
        len(spelling) != 2
        or report.splitlines()[-1:] != ["BF dry-path-complete"]
        or bool(getattr(dry, "stderr", ""))
        or not re.fullmatch(r"\d+:\d+", spelling[-1])
        or any(not re.fullmatch(r"\d+:\d+", identity) for identity in identities)
        or len(identities)
        != sum(line.startswith("BF locked-path ") for line in report.splitlines())
    ):
        return
    repaired = {entry.rsplit(" -> ", 1)[-1] for entry in _path_lock_report(report)[0]}
    if {path, spelling[0]} & repaired or spelling[1] in identities:
        return
    raise LoaderCustodyError(
        f"sandbox startup refused: in the task's image, {loader}, which is not "
        "root's alone, so BenchFlow will not run root steps (the verifier's) "
        "with a sandbox user and did not install the agent.\n"
        f"next: {_LOADER_NEXT}"
    )


async def prepare_runtime_tree(
    env,
    *,
    phase: str = "all",
    timeout_sec: int = 300,
    service: str = "main",
    transfer_roots: tuple[str, ...] = (),
) -> bool:
    """Make (or keep) root's protected runtime kit in *env*, before the
    agent's turn, and route root's steps through it (C8).

    See :mod:`benchflow._runtime_tree`. Runs as root on the plumbing route
    while the image is still the task's own. A sandbox without the kit
    machinery (a test's stand-in) is left alone. Raises with what to do
    when the kit cannot be made: no root step may run on the image's tools
    once the agent has had a turn. *phase*: ``"build"`` before the
    workspace grant, ``"seal"`` after it and the PATH lock (see
    :func:`benchflow._runtime_tree.kit_prepare_command`). *service*: the
    compose service whose own kit this is (PY8-PT-R8-6); a service other
    than ``main`` that cannot have one is recorded without one, and root's
    plumbing there is refused after the agent's turn. *transfer_roots*: a
    separate verifier's (:func:`prepare_separate_verifier_runtime`).
    """
    from benchflow._runtime_tree import kit_prepare_command, parse_kit_record

    if getattr(env, "supports_runtime_kit", False) is not True:
        return False
    # The record is read from a receipt, as every lockdown read is.
    if service == "main":
        result = None
        miss = None
        if phase == "build":
            result, miss = await _restore_cached_kit(env, timeout_sec)
        if result is None:
            result = await _read_exec(
                env,
                kit_prepare_command(phase, transfer_roots=transfer_roots),
                user="root",
                timeout_sec=timeout_sec,
            )
            lines = (getattr(result, "stdout", "") or "").splitlines()
            if (
                miss is not None
                and _exec_return_code(result) == 0
                and "BF kit-built" in lines
                # A kit whose Python copy failed twice is not kept: the next
                # trial on this image builds it again (ACC-RUN2).
                and not any(line.startswith("BF kit-python-failed ") for line in lines)
            ):
                # Kept as the build left it, before the grant can touch anything.
                await env.kit_cache_store(miss)
    else:
        result = await _read_exec(
            env,
            kit_prepare_command(phase),
            user="root",
            timeout_sec=timeout_sec,
            service=service,
        )
    stdout = getattr(result, "stdout", "") or ""
    kit = parse_kit_record(stdout)
    built = phase == "build" and "BF kit-built" in stdout.splitlines()
    if _exec_return_code(result) == 0 and built:
        return True
    if service != "main" and (_exec_return_code(result) != 0 or kit is None):
        logger.warning(
            "The compose service %s has no protected runtime, so BenchFlow will "
            "run no root step there after the agent's turn.%s",
            service,
            _exec_failure_detail(result),
        )
        env.set_runtime_kit(None, service)
        return False
    if _exec_return_code(result) != 0 or kit is None:
        loader = _loader_refusal(result)
        if loader is not None:
            raise LoaderCustodyError(
                "sandbox startup refused: BenchFlow could not seal its protected "
                f"runtime: {loader}, which is not root's alone, so it will not "
                "run root steps (the verifier's) after the agent's turn.\n"
                "next: if the task's workdir holds that path, give the task a "
                f"workdir of its own; otherwise {_LOADER_NEXT}"
            )
        stderr = getattr(result, "stderr", "") or ""
        if (
            isinstance(result, ReceiptFailure)
            and _exec_return_code(result) in (1, 2)
            and re.search(
                r"(?im)^[^\n]*\b(?:sh|bash|dash|ash):[^\n]*"
                r"(?:syntax error|unexpected EOF)",
                stderr,
            )
        ):
            # A checked receipt can carry a child tool's status 2 and shell
            # error. Only a failed outer exec/read supports parser blame;
            # old Bash can report that failure with status 1 as well as 2.
            # Diagnostic only: failure still refuses all root work. Probe the
            # executing shell through a separate, small receipt so a parse
            # error in our setup cannot also prevent its identification.
            shell = "/bin/sh (version unavailable)"
            try:
                probe = await _read_exec(
                    env,
                    "printf '/bin/sh -> '; readlink /proc/$$/exe; "
                    'if [ -n "${BASH_VERSION-}" ]; then '
                    "printf 'bash %s\\n' \"$BASH_VERSION\"; "
                    "elif command -v busybox >/dev/null 2>&1; then "
                    "busybox 2>&1 | head -n 1; "
                    "elif command -v dpkg-query >/dev/null 2>&1; then "
                    "dpkg-query -W dash; else printf 'version unavailable\\n'; fi",
                    user="root",
                    timeout_sec=min(timeout_sec, 10),
                    **({"service": service} if service != "main" else {}),
                )
                if _exec_return_code(probe) == 0 and probe.stdout:
                    display = diagnostic_result(probe)
                    # Redact complete fields before stripping or clipping.
                    # Keep the raw receipt for every enforcement decision.
                    shell = display.stdout.render().strip()[:300]
            except Exception:
                # An unavailable diagnostic must never mask the setup failure.
                pass
            raise RuntimeError(
                "BenchFlow could not run its protected-runtime setup script "
                f"in this image's shell ({shell}): shell syntax error in "
                "BenchFlow's setup script. It will not run root steps after "
                f"the agent's turn.{_exec_failure_detail(result)}"
            )
        raise RuntimeError(
            "BenchFlow could not prepare its protected runtime "
            "(/opt/benchflow/rt) in the sandbox, so it will not run root steps "
            "after the agent's turn. The image needs a root-owned /bin/sh, "
            "sha256sum and /opt as a folder only root (or, sticky, anyone) may "
            f"write.{_exec_failure_detail(result)}"
        )
    if service != "main":
        env.set_runtime_kit(kit, service)
    else:
        env.set_runtime_kit(kit)
    for line in stdout.splitlines():
        if line.startswith("BF kit-python-failed ") and not kit.python:
            # The image has a root-only python3 that could not be copied,
            # twice (ACC-RUN2): say which step.
            logger.warning(
                "Protected runtime in %s holds no Python: %s",
                service,
                line.removeprefix("BF kit-python-failed "),
            )
    logger.debug(
        "Protected runtime ready in %s: python=%s bash=%s system folders=%s",
        service,
        "yes" if kit.python else "no",
        "yes" if kit.bash else "no",
        ":".join(kit.trusted) or "none",
    )
    return True


async def _restore_cached_kit(env, timeout_sec: int) -> tuple[Any, Any]:
    """(the restore's result, None) when main's kit came from this host's
    cache (PERF-SBX, :mod:`benchflow.sandbox._kit_cache`); (None, entry) when
    main may use the cache but this host has no kit for it yet (the build
    then keeps its kit); (None, None) otherwise. A restore that fails leaves
    the build to run as before: it starts by removing the kit folder."""
    from benchflow._runtime_tree import kit_prepare_command
    from benchflow.sandbox._hooks import sandbox_hook

    lookup = sandbox_hook(env, "kit_cache_lookup")
    if lookup is None:
        return None, None
    entry = await lookup()
    if entry is None:
        return None, None
    if entry.tar is None:
        return None, entry
    stage = await env.kit_cache_stage(entry)
    if stage is None:
        return None, None
    result = await _read_exec(
        env,
        kit_prepare_command("restore", stage=stage),
        user="root",
        timeout_sec=timeout_sec,
    )
    if (
        _exec_return_code(result) == 0
        and "BF kit-built" in (getattr(result, "stdout", "") or "").splitlines()
    ):
        logger.info("Protected runtime kit restored from this host's kit cache")
        return result, None
    logger.warning(
        "The kept protected runtime kit could not be placed, so it is built again.%s",
        _exec_failure_detail(result),
    )
    return None, None


async def _kit_services(env) -> list[str]:
    """The compose services other than ``main`` (none on a single-container
    backend)."""
    if getattr(env, "supports_runtime_kit", False) is not True:
        return []
    services = getattr(env, "services", None)
    if not callable(services):
        return []
    try:
        names = await services()
    except Exception:
        return []
    return [name for name in names if isinstance(name, str) and name != "main"]


# The first uid useradd hands out (UID_MIN), and the bound below Daytona's
# 0-65535 user namespace and nobody's 65534 that a fresh uid stays under.
_SANDBOX_UID_MIN = 1000
_SANDBOX_UID_LIMIT = 60000

# Every uid and gid the image already uses: the ids /etc/passwd and /etc/group
# name, and every file's owner and group, including image VOLUMEs and task
# mounts. Only virtual/runtime trees are pruned, never a filesystem boundary.
# Keep each step's status: a pipeline ending in tr used to hide find errors.
_USED_IDS_CMD = r"""
bf_accounts=$(cut -d: -f3 /etc/passwd /etc/group) || exit 1
if find / -maxdepth 0 -printf '' >/dev/null 2>&1; then
  bf_files=$(find / \( -path /proc -o -path /sys -o -path /dev -o -path /run \) -prune -o -printf '%U\n%G\n') || exit 1
else
  bf_files=$(find / \( -path /proc -o -path /sys -o -path /dev -o -path /run \) -prune -o -exec /bin/sh -c '
    stat -c "%u
%g" "$@" || printf "BF_SCAN_FAILED\n"
  ' bf-stat {} +) || exit 1
fi
case "$bf_files" in *BF_SCAN_FAILED*) exit 1 ;; esac
bf_ids=$(sort -un <<BF_IDS
$bf_accounts
$bf_files
BF_IDS
) || exit 1
printf '%s\n' "$bf_ids" | tr '\n' ' '
""".strip()


def _create_sandbox_user_cmd(sandbox_user: str) -> str:
    """Create *sandbox_user* with a uid and gid that own nothing in the image.

    ``useradd`` takes the first uid /etc/passwd does not name, which can be
    one the image's own files belong to: an image whose ``uv`` is uid 1001's
    (autocodebench's ``/root/.local/bin``, left by a ``docker commit``) hands
    those files to the agent user, which can then rewrite a program the
    verifier runs (PLAN P5, Rust P1.2c's finding 2). So the uid, used as the
    gid too, is the first from 1000 that no passwd or group entry names and
    no file's owner or group is. A user the image already has is kept as it
    is: the task chose it. busybox images (Alpine) have adduser, not useradd.
    """
    user = shlex.quote(sandbox_user)
    return (
        f"{{ id -u {user} >/dev/null 2>&1 || {{ "
        f'bf_ids=" $({_system_probe(_USED_IDS_CMD, "bf-used-ids")})" || '
        '{ echo "sandbox uid/gid inventory could not complete" >&2; exit 1; }; '
        f"bf_uid={_SANDBOX_UID_MIN}; "
        f'while case "$bf_ids" in *" $bf_uid "*) true;; *) false;; esac; '
        f"do bf_uid=$((bf_uid + 1)); done; "
        f'[ "$bf_uid" -lt {_SANDBOX_UID_LIMIT} ] && '
        f"if command -v useradd >/dev/null 2>&1; then "
        f'groupadd -g "$bf_uid" {user} && '
        f'useradd -m -u "$bf_uid" -g "$bf_uid" -s /bin/bash {user}; '
        f'else addgroup -g "$bf_uid" {user} && '
        f'adduser -D -u "$bf_uid" -G {user} -s /bin/bash {user}; fi; }}; }}'
    )


async def setup_sandbox_user(
    env,
    sandbox_user: str,
    workspace: str,
    *,
    timeout_sec: int = 120,
    guard_daemon: bool = True,
) -> str:
    """Create non-root sandbox user, grant workspace access. Return agent_cwd.

    The grant never covers BenchFlow's runtime tree (C8): a
    workspace of ``/``, inside ``/opt/benchflow`` or holding a system folder
    is refused, and one above the tree (``/opt``) stays root's and sticky
    (:func:`benchflow._runtime_tree.workspace_grant_command`). Last, root's
    protected runtime kit is made (:func:`prepare_runtime_tree`), before
    the agent's turn.
    """
    from benchflow._runtime_tree import (
        workspace_grant_command,
        workspace_grant_problem,
    )

    if not re.match(r"^[a-z_][a-z0-9_-]*$", sandbox_user):
        raise ValueError(
            f"Invalid sandbox_user: {sandbox_user!r} (must be alphanumeric)"
        )
    problem = workspace_grant_problem(workspace)
    if problem is not None:
        raise ValueError(
            f"BenchFlow will not give the sandbox user {sandbox_user!r} its "
            f"workspace: {problem}"
        )
    # Scripted controls run as root and use no privileged agent tools; the
    # physical check protects an untrusted user turn, not an oracle workspace.
    if guard_daemon:
        # Resolve aliases in the sandbox, where its symlinks live, before the
        # grant: a workspace that is a link into the runtime tree or above a
        # system folder is refused by where it leads (FF-install-C1).
        # Read from a receipt, as every lockdown read is. A failed exec stops
        # the setup; a receipt that is missing (a stand-in sandbox) leaves the
        # path as written, already checked above.
        from benchflow.sandbox._exec_receipt import RECEIPT_FAILED

        resolved = await _read_exec(
            env, f"cd {shlex.quote(workspace)} && pwd -P", timeout_sec=timeout_sec
        )
        status = _exec_return_code(resolved)
        if status not in (0, RECEIPT_FAILED):
            raise RuntimeError(
                "could not resolve the sandbox workspace before user setup"
            )
        physical = (getattr(resolved, "stdout", "") or "").rstrip("\n")
        if status != 0:
            physical = ""
        if physical:
            problem = workspace_grant_problem(physical)
            if problem is not None:
                raise ValueError(
                    f"BenchFlow will not give the sandbox user {sandbox_user!r} "
                    f"its workspace {workspace!r}, which resolves to "
                    f"{physical!r}: {problem}"
                )
            workspace = physical
    logger.info(f"Setting up sandbox user: {sandbox_user}")
    # Every compose service's kit is built, then main's, while the images are
    # the task's own and before the workspace grant can hand any of them
    # over. Each is sealed only after the grant (PT-C8-R9-1): a grant that
    # reaches a sidecar's kit through a shared mount is undone by its seal,
    # and one that takes a folder above it refuses the task.
    services = [
        service
        for service in await _kit_services(env)
        if await prepare_runtime_tree(
            env, phase="build", timeout_sec=max(timeout_sec, 300), service=service
        )
    ]
    await prepare_runtime_tree(env, phase="build", timeout_sec=max(timeout_sec, 300))
    home = f"/home/{sandbox_user}"
    home_dirs = sorted(d for d in get_sandbox_home_dirs() if d != ".local")
    result = await exec_raw(
        env.exec,
        f"{_create_sandbox_user_cmd(sandbox_user)} && "
        f"{_legacy_root_tool_link_cmd('/root/.local/bin', f'{home}/.local/bin')} && "
        f"{_legacy_root_tool_link_cmd('/root/.nvm', f'{home}/.nvm')} && "
        f"for d in {' '.join(home_dirs)}; do "
        f"mkdir -p {home}/$d && "
        f"if [ -d /root/$d ]; then "
        f"cp -a /root/$d/. {home}/$d/ 2>/dev/null || true; fi; done && "
        f"chown -R {sandbox_user}:{sandbox_user} {home} && "
        f"{workspace_grant_command(workspace, sandbox_user)} && "
        f"for d in /output /outputs; do "
        f'if [ -d "$d" ] && [ ! -L "$d" ]; then '
        f'chown -R {sandbox_user}:{sandbox_user} "$d"; fi; done',
        timeout_sec=timeout_sec,
    )
    if _exec_return_code(result) != 0:
        # A later step (a chown on a read-only mount) may fail once the user
        # exists, as before; a missing user would only surface much later.
        exists = await exec_raw(
            env.exec, f"id -u {sandbox_user} >/dev/null 2>&1", timeout_sec=30
        )
        detail = _exec_failure_detail(result)
        if _exec_return_code(exists) != 0:
            raise RuntimeError(
                f"could not create sandbox user {sandbox_user!r}: the image "
                "has neither a working useradd and groupadd nor busybox "
                "adduser and addgroup; run as "
                f"root with --sandbox-user none or add one to the image.{detail}"
            )
        logger.warning(f"Sandbox user {sandbox_user} setup step failed.{detail}")
    # Root's commands and the verifier find their programs on PATH: keep the
    # agent user from changing them. The oracle and nop get it too, so the
    # verifier keeps the same tool folders whoever ran (Rust 0.8.2: the
    # oracle is judged by the agent user's rule).
    await lock_system_path(env, workspace, timeout_sec=timeout_sec)
    # Each sidecar's kit is sealed after the grant and before main's kit is
    # published: once main has one, a service without its own refuses
    # root's steps, its own seal included (PT-C8-R10-1).
    for service in services:
        if not await prepare_runtime_tree(
            env, phase="seal", timeout_sec=max(timeout_sec, 300), service=service
        ):
            raise RuntimeError(
                f"the compose service {service!r} had its protected runtime "
                "before the workspace grant and lost it after: the grant "
                "reaches it through a shared mount, so BenchFlow will not run "
                "the agent; keep that service's /opt out of every mount the "
                "agent's workspace shares"
            )
    # From here on root's steps run on the protected runtime kit (C8).
    await prepare_runtime_tree(env, phase="seal", timeout_sec=max(timeout_sec, 300))
    # Daytona's daemon runs commands as root for anyone who reaches it;
    # keep the agent user away from it in every network mode, or refuse.
    # A run that executes nothing as the sandbox user (oracle, nop) passes
    # guard_daemon=False: there is nobody to keep away.
    if guard_daemon:
        await guard_sandbox_daemon(env, sandbox_user)
    logger.info(f"Sandbox user {sandbox_user} ready (workspace={workspace})")
    return workspace


async def lockdown_paths(env, paths: list[str]) -> None:
    """Lock directories so the sandbox user cannot access them.

    Runs after root-level setup but before agent launch.
    Uses chown-then-chmod ordering to prevent TOCTOU window.
    Rejects symlinks and validates path patterns against injection.
    """
    if not paths:
        return

    for p in paths:
        _validate_locked_path(p)

    # Each match must succeed. A loop's last successful match must not hide
    # an earlier failure, nor may a backend's check=False permit agent launch.
    parts = []
    for p in paths:
        parts.append(
            f"for d in {p}; do "
            f'  [ ! -L "$d" ] || {{ echo "Cannot lock symlink $d" >&2; exit 1; }}; '
            f'  [ -e "$d" ] || continue; '
            f'  chown root:root "$d" && chmod 700 "$d" || '
            '{ echo "Cannot lock $d: make the mount writable during setup "'
            '"or protect the path in the image and remove it from locked paths" >&2; exit 1; }; '
            f"done"
        )
    cmd = " && ".join(parts)
    await _checked_exec(
        env, cmd, "Locking sandbox paths failed", user="root", timeout_sec=30
    )


# Build-config snapshot / restore (Tier 2)

# Files snapshotted before agent runs and restored before verification.
# Covers common build backends to prevent setup.py / pyproject.toml hijacks.
_BUILD_CONFIG_FILES = (
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "noxfile.py",
    "hatch.toml",
    "flit.ini",
    "MANIFEST.in",
    # Non-build files that control how tests install/run — must be snapshotted
    # and restored so an agent cannot inject malicious packages or override
    # test targets via set-e + early-exit tricks.
    "requirements.txt",
    "requirements-dev.txt",
    "Makefile",
)
# chmod 700: root-only so sandbox_user cannot read or overwrite the snapshot.
# A fixed private root path keeps our pre-agent copy outside workspace and
# runtime input trees. A /tmp copy otherwise taints even an unchanged project
# because it has no workspace-relative image counterpart (V2-R12-CONF-5).
_SNAPSHOT_DIR = "/.benchflow_build_snapshot"
_SNAPSHOT_MANIFEST = f"{_SNAPSHOT_DIR}/manifest.json"
# The second step of _snapshot_build_config's exec (PY8 U12): the manifest
# the host builds from the copy's output (json.dumps of {file: present} in
# _BUILD_CONFIG_FILES' order; a file is present when the copy printed the
# word <file>=present), written as the host's echo writes it, then printed
# back. The file names are plain words, so they need no quoting here.
assert all(re.fullmatch(r"[A-Za-z0-9._-]+", f) for f in _BUILD_CONFIG_FILES)
_SNAPSHOT_MANIFEST_WRITE = (
    "set -f; __bf_m=''; "
    f"for __bf_f in {' '.join(_BUILD_CONFIG_FILES)}; do "
    "__bf_v=false; "
    "for __bf_t in $__bf_out0; do "
    '[ "$__bf_t" = "$__bf_f=present" ] && __bf_v=true; done; '
    '__bf_m="$__bf_m${__bf_m:+, }\\"$__bf_f\\": $__bf_v"; '
    "done; "
    '__bf_m="{$__bf_m}"; '
    f"printf '%s\\n' \"$__bf_m\" > {_SNAPSHOT_MANIFEST} && "
    "printf '%s' \"$__bf_m\""
)


async def _snapshot_build_config(env, workspace: str) -> None:
    """Snapshot build-config files before the agent runs.

    Absence/presence is recorded in manifest.json rather than embedding a
    sentinel string in captured files — prevents an agent from forging
    "this file was absent" by planting a magic string in setup.py.

    ORDERING INVARIANT: must be called before agent launch. The agent owns
    workspace files (chown'd by setup_sandbox_user) and could modify them
    immediately on start.

    All files are probed and copied in one exec (one line ``<file>=present``
    or ``<file>=absent`` each): on a remote sandbox every exec is a network
    round trip, so a per-file loop was slow.
    A file whose copy fails prints nothing and is recorded absent, as before.

    The manifest is written in the same exec (PY8 U12): the script keeps the
    copy's output and writes the manifest the host would have built from it,
    then prints what it wrote. The host builds the manifest itself, as
    before, and writes it with its own exec when the script's differs, its
    write failed or its answer is unreadable; when the copy's own answer is
    unreadable, both execs run again as before (the copy only re-copies the
    same files before the agent starts).
    """
    verifier_service = getattr(env, "_benchflow_verifier_service", "main")
    verifier_service = verifier_service if isinstance(verifier_service, str) else "main"
    sandbox_user = getattr(env, "_benchflow_sandbox_user", None)
    sandbox_user = sandbox_user if isinstance(sandbox_user, str) else None
    probes = [
        f"if [ -f {workspace}/{fname} ]; then "
        f"cp --preserve=all {workspace}/{fname} {_SNAPSHOT_DIR}/{fname} "
        f"&& echo {fname}=present; "
        f"else echo {fname}=absent; fi"
        for fname in _BUILD_CONFIG_FILES
    ]
    copy = (
        f"mkdir -p {_SNAPSHOT_DIR} && chmod 700 {_SNAPSHOT_DIR} && "
        f"{{ {'; '.join(probes)}; }}"
    )
    tag = new_tag()
    # The copy and the manifest write ride in front of the image inspectors'
    # root exec (_prepare_bash_env_inspector's prefix, V7), as the copy alone
    # does without U12. The inspectors run after them whatever they answer,
    # unchanged, with their own environment and budget; their "ready" line
    # comes after the steps' end markers, outside every answer. The manifest
    # step uses shell builtins only.
    merged = await _prepare_bash_env_inspector(
        env,
        blocked=_blocked_verifier_path_prefixes(sandbox_user, workspace),
        prefix_command=merged_script(
            [MergedStep(copy, capture=True), MergedStep(_SNAPSHOT_MANIFEST_WRITE)], tag
        ),
    )
    if verifier_service != "main":
        await _prepare_bash_env_inspector(
            env,
            service=verifier_service,
            blocked=_blocked_verifier_path_prefixes(sandbox_user, workspace),
        )
    answers = parse_merged(getattr(merged, "stdout", None), tag, 2)
    if 0 in answers:
        copied = answers[0].output
    else:
        result = await _read_exec(env, copy, user="root")
        copied = result.stdout or ""
    reported = set(copied.split())
    manifest = {fname: f"{fname}=present" in reported for fname in _BUILD_CONFIG_FILES}
    manifest_json = _json.dumps(manifest)
    written = answers.get(1)
    if written is None or (written.return_code, written.output) != (0, manifest_json):
        await exec_raw(
            env.exec,
            f"echo {shlex.quote(manifest_json)} > {_SNAPSHOT_MANIFEST}",
            user="root",
        )


async def _restore_build_config(env, workspace: str) -> None:
    """Restore build-config files to their pre-agent state.

    Files that existed pre-agent are restored from the snapshot; files that
    didn't are removed if the agent created them.
    """
    result = await _read_exec(env, _on_path(f"cat {_SNAPSHOT_MANIFEST}"), user="root")
    manifest: dict[str, bool] = _json.loads(result.stdout or "")
    for fname in _BUILD_CONFIG_FILES:
        src = f"{_SNAPSHOT_DIR}/{fname}"
        dst = f"{workspace}/{fname}"
        if manifest.get(fname):
            # File existed pre-agent: restore from snapshot.
            # rm -f first to sever any symlink the agent may have planted at dst.
            cmd = (
                f"rm -f {dst} && "
                f"cp --preserve=timestamps {src} {dst} "
                f"&& chown root:root {dst} && chmod 644 {dst}"
            )
        else:
            # File did not exist pre-agent: remove anything the agent created.
            cmd = f"rm -f {dst}"
        await exec_raw(env.exec, _on_path(cmd), user="root")


def _log_dir_cmds(sandbox_user: str | None) -> list[str]:
    return [
        # Lock /logs/ parent: sandbox_user cannot rename /logs/verifier/ out.
        "chown root:root /logs && chmod 755 /logs",
        # Grant sandbox user write access to agent-writable log dirs so tasks
        # that write answers to /logs/artifacts/ (e.g. infinitebench) work.
        *(
            [f"chown {sandbox_user}:{sandbox_user} /logs/agent /logs/artifacts"]
            if sandbox_user
            else []
        ),
    ]


async def _prepare_log_dirs(env, sandbox_user: str | None = None) -> None:
    """The /logs ownership part of _seed_verifier_workspace, for a sandbox
    started from a branch snapshot, whose pre-agent /testbed_verify copy and
    build-config baseline already exist and must not be re-captured."""
    await exec_raw(env.exec, " && ".join(_log_dir_cmds(sandbox_user)), user="root")


# Both exist once _snapshot_build_config and _seed_verifier_workspace ran.
VERIFIER_BASELINE_PROBE = (
    f"test -f {_SNAPSHOT_MANIFEST} && test -d /testbed_verify && echo baseline"
)


async def _seed_verifier_workspace(
    env, workspace: str = "/testbed", sandbox_user: str | None = None
) -> None:
    """Seed /testbed_verify as root-owned pre-agent snapshot used by harden_before_verify."""
    cmds = [
        # Lock /logs/ parent: sandbox_user cannot rename /logs/verifier/ out.
        "chown root:root /logs && chmod 755 /logs",
        # Grant sandbox user write access to agent-writable log dirs so tasks
        # that write answers to /logs/artifacts/ (e.g. infinitebench) work.
        *(
            [f"chown {sandbox_user}:{sandbox_user} /logs/agent /logs/artifacts"]
            if sandbox_user
            else []
        ),
        # Seed root-owned readable workspace copy from the actual workspace
        # (may differ from /testbed for tasks with WORKDIR=/app etc.).
        f"rm -rf /testbed_verify && cp -a {shlex.quote(workspace)} /testbed_verify && "
        f"chown -R root:root /testbed_verify && chmod -R o+rX /testbed_verify",
    ]
    # One exec (PY8 U12): each command in its own subshell, the next one run
    # whatever the last one's status, as the separate execs were (their
    # status was never read).
    await exec_raw(env.exec, merged_script(cmds, new_tag()), user="root")
    env._benchflow_verifier_workspace = workspace


async def _refresh_verifier_workspace(env, workspace: str) -> None:
    """Copy restored build-config files into the read-only verifier workspace.

    Called after _restore_build_config so /testbed_verify reflects the
    canonical pre-agent build-config state.
    """
    for fname in _BUILD_CONFIG_FILES:
        src = f"{workspace}/{fname}"
        dst = f"/testbed_verify/{fname}"
        cmd = (
            f"if [ -f {src} ]; then "
            f"  rm -f {dst} && "
            f"  cp --preserve=timestamps {src} {dst} "
            f"  && chown root:root {dst} && chmod 644 {dst}; "
            f"else "
            f"  rm -f {dst}; "
            f"fi"
        )
        await exec_raw(env.exec, _on_path(cmd), user="root")


# Verifier hardening

# Trusted env vars for verifier execution — override any agent pollution.
# Intentionally omitted (negative guard in test_verify.py explains why):
# PYTHONHOME.
VERIFIER_ENV: dict[str, str] = {
    "PATH": TRUSTED_SYSTEM_PATH,
    "PYTEST_ADDOPTS": (
        "-c /dev/null "  # block pyproject.toml/pytest.ini/tox.ini/setup.cfg discovery
        "--confcutdir=/tests "  # block conftest.py walk-up beyond /tests
        "-p no:cacheprovider"
        # --rootdir is injected dynamically by _build_pytest_addopts() based on
        # the task's actual workspace, so it works for both /app (task.toml/SWE-bench)
        # and /root (SkillsBench) conventions.
    ),
    # Block pytest11 entry-point plugins. An agent can modify a pre-installed
    # package's plugin source to forge a reward; -c /dev/null does not block
    # entry-point registration. Tasks that need specific plugins declare them
    # in task config [verifier] pytest_plugins = [...].
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    # Redirect .pyc cache reads/writes to a non-existent directory so
    # agent-pre-compiled __pycache__ bytecode is never found or executed.
    "PYTHONPYCACHEPREFIX": "/nonexistent",
    "PYTHONPATH": "",
    "PYTHONSTARTUP": "",
    "LD_PRELOAD": "",
    "LD_LIBRARY_PATH": "",
    # Prevent pip from writing to user site-packages during pip install -e .
    "PYTHONNOUSERSITE": "1",
    "PIP_USER": "0",
    # pip loads no configuration file at all when this is os.devnull: not
    # ~/.config/pip/pip.conf or ~/.pip/pip.conf under the HOME below, not
    # /etc/pip.conf, not a venv's own. It is pinned after the task's verifier
    # env and never replaced (pin_verifier_env, on every verifier path), so
    # every pip the verifier runs reads none.
    "PIP_CONFIG_FILE": "/dev/null",
    # git reads no ~/.gitconfig or ~/.config/git/config (git >= 2.32); the
    # system file /etc/gitconfig still applies.
    "GIT_CONFIG_GLOBAL": "/dev/null",
    # The freeze makes the workspace's repositories root's, so root's git
    # trusts what the agent wrote in them. At command-line scope (git >= 2.31)
    # git runs no hook, fsmonitor, external diff or signing program of theirs;
    # a test.sh that wants one names it with -c, which comes later. Drivers and
    # includes named in the repository's config are purged with the freeze.
    "GIT_CONFIG_COUNT": "6",
    "GIT_CONFIG_KEY_0": "core.hooksPath",
    "GIT_CONFIG_VALUE_0": "/dev/null",
    "GIT_CONFIG_KEY_1": "core.fsmonitor",
    "GIT_CONFIG_VALUE_1": "false",
    "GIT_CONFIG_KEY_2": "diff.external",
    "GIT_CONFIG_VALUE_2": "",
    "GIT_CONFIG_KEY_3": "commit.gpgSign",
    "GIT_CONFIG_VALUE_3": "false",
    "GIT_CONFIG_KEY_4": "tag.gpgSign",
    "GIT_CONFIG_VALUE_4": "false",
    "GIT_CONFIG_KEY_5": "log.showSignature",
    "GIT_CONFIG_VALUE_5": "false",
    # PEP-668 base images (Fedora, recent Debian) refuse pip installs into
    # system-site without this flag. Verifier runs as root and system-site is
    # root-owned, so allowing it is safe; without it, tasks that pip-install
    # pytest in test.sh either fail outright or fall back to a user-site path
    # that PYTHONNOUSERSITE=1 hides at import time.
    "PIP_BREAK_SYSTEM_PACKAGES": "1",
    # Root's home. When it is the agent's workspace (WORKDIR /root) or the
    # agent could otherwise write it, hardening gives the verifier a new one
    # (_verifier_tool_state.home_overrides); Python's user site and
    # pip's and git's per-user config are switched off above either way.
    "HOME": "/root",
    # Disable breakpoint() — any other value imports an arbitrary callable.
    "PYTHONBREAKPOINT": "0",
    # Prevent coverage.py from importing a config file as Python on startup.
    "COVERAGE_PROCESS_START": "",
    # Prevent Django/Celery from importing an agent-controlled module at startup.
    "DJANGO_SETTINGS_MODULE": "",
    "CELERY_CONFIG_MODULE": "",
}

_SAFE_VERIFIER_PATH = VERIFIER_ENV["PATH"]
_SAFE_VERIFIER_PATH_PARTS = tuple(_SAFE_VERIFIER_PATH.split(":"))
_RUNTIME_PATH_PREFIXES = RUNTIME_PATH_PREFIXES

# The file-mode creation mask the verifier's commands run under. A command
# otherwise inherits the runtime's: ``docker exec`` gives 0022 on Docker's own
# daemon (29.1.3, runc 1.3.4) but 0000 on Docker-in-Docker (29.8.1, runc
# 1.5.1). Under 0000 everything test.sh installs, uv's cache or a pip install
# into site-packages, came out group- and world-writable, the plugin guard
# refused the verifier's own ctrf, and every ``--ctrf`` task went unscored,
# the correct oracle included. The mask is set inside the command string, so
# it holds on every backend (Docker, remote Docker, Daytona, Modal, ...).
VERIFIER_UMASK = "022"


def with_verifier_umask(command: str) -> str:
    """*command* run under :data:`VERIFIER_UMASK`, whatever the runtime's mask."""
    return f"umask {VERIFIER_UMASK} && {command}"


_DEFAULT_ROOTDIR = "/app"
_LEGACY_VERIFIER_CONFCUTDIR = "/tests"


@dataclass(frozen=True)
class ImagePytestIni:
    """The image's own pytest configuration file, unchanged by the agent.

    *path* is the file in the workspace; *addopts* its ``addopts`` with every
    ``-p`` taken out (``None`` when it has none), and *plugins* the names
    those ``-p`` load, which the verifier passes after the plugin guard so
    the guard judges them (pytest imports an ini's ``-p`` before
    PYTEST_ADDOPTS', so left in the ini they would load before the guard).
    """

    path: str
    addopts: str | None = None
    plugins: tuple[str, ...] = ()


def _build_pytest_addopts(
    workspace: str | None = None,
    plugin_flags: str = "",
    *,
    verifier_confcutdir: str = _LEGACY_VERIFIER_CONFCUTDIR,
    ini: ImagePytestIni | None = None,
) -> str:
    """Build PYTEST_ADDOPTS with a dynamic --rootdir based on the workspace.

    Without an explicit --rootdir, -c /dev/null causes pytest to fall back to
    /dev as rootdir, producing broken test node IDs (../dev/::test_foo).
    The rootdir must point to a directory that actually exists in the container.

    *ini* replaces ``/dev/null`` with the image's own configuration file
    see :func:`_image_pytest_ini`.
    """
    rootdir = workspace or _DEFAULT_ROOTDIR
    base_addopts = VERIFIER_ENV["PYTEST_ADDOPTS"].replace(
        f"--confcutdir={_LEGACY_VERIFIER_CONFCUTDIR}",
        f"--confcutdir={shlex.quote(verifier_confcutdir)}",
    )
    if ini is not None:
        config = f"-c {shlex.quote(ini.path)}"
        if ini.addopts is not None:
            config += f" -o {shlex.quote('addopts=' + ini.addopts)}"
        base_addopts = base_addopts.replace("-c /dev/null", config, 1)
    addopts = f"{base_addopts} --rootdir={shlex.quote(rootdir)}"
    if plugin_flags:
        addopts += f" {plugin_flags}"
    if ini is not None and ini.plugins:
        addopts += "".join(f" -p {shlex.quote(name)}" for name in ini.plugins)
    return addopts


# The configuration files pytest looks for in a folder, in its order.
_PYTEST_CONFIG_NAMES = (
    "pytest.ini",
    ".pytest.ini",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
)
# A configuration file larger than this is not read: the verifier keeps
# ``-c /dev/null``.
_PYTEST_CONFIG_MAX_BYTES = 1 << 20

# Both copies of every configuration file pytest might read from the
# workspace: root's pre-agent copy in /testbed_verify (as the plugin guard
# uses it) and the workspace as the agent left it. $1 is the
# workspace, $2 a marker the host chose after the agent stopped. Each copy is
# a line "<marker> <name> image|work <kind>", then, for kind "file", the file
# and a line break. Kinds: "file" (a regular file BenchFlow may read), "link",
# "other" (not a regular file), "big", and "open" for an image copy the agent
# user could have written: /testbed_verify must be root's and closed to group
# and others, or the copy itself root's and not writable by them. The host
# decides which file pytest would read and whether the agent changed what
# pytest reads of it (:func:`_choose_unchanged_pytest_config`).
_IMAGE_PYTEST_INI_SCRIPT = r"""
w=$1
m=$2
c=/testbed_verify
[ -d "$c" ] && [ ! -L "$c" ] || exit 0
set -- $(ls -ldn "$c" 2>/dev/null)
[ "${3-}" = 0 ] || exit 0
case ${1-} in
  d???------*) closed=1 ;;
  d????-??-?*) closed=0 ;;
  *) exit 0 ;;
esac
for f in pytest.ini .pytest.ini pyproject.toml tox.ini setup.cfg; do
  for side in image work; do
    if [ $side = image ]; then p=$c/$f; else p=$w/$f; fi
    if [ -L "$p" ]; then k=link
    elif [ -f "$p" ]; then
      k=file
      if [ $side = image ] && [ $closed = 0 ]; then
        set -- $(ls -ln "$p" 2>/dev/null)
        case ${1-} in -????-??-?*) [ "${3-}" = 0 ] || k=open ;; *) k=open ;; esac
      fi
      s=$(wc -c < "$p" 2>/dev/null | tr -dc 0-9)
      [ "$k" != file ] || { [ -n "$s" ] && [ "$s" -le MAXBYTES ]; } || k=big
    elif [ -e "$p" ]; then k=other
    else continue
    fi
    printf '%s %s %s %s\n' "$m" "$f" "$side" "$k"
    if [ "$k" = file ]; then cat "$p" && printf '\n'; fi
  done
done
""".strip().replace("MAXBYTES", str(_PYTEST_CONFIG_MAX_BYTES))


def _iniconfig_sections(text: str) -> dict[str, dict[str, str]]:
    """*text* read as pytest's ``iniconfig`` reads an ini file.

    Comment lines start with ``#`` or ``;``; a section header may carry a
    comment, and a ``[`` line without its ``]`` continues the value before it;
    ``name = value`` or ``name: value``; an indented line continues the value
    before it, joined with a line break. No inline comments. ``ValueError``
    wherever iniconfig refuses the file (a line it cannot read, a value
    outside a section, a duplicate section or name), so a file this accepts
    pytest reads the same way.
    """
    sections: dict[str, dict[str, str]] = {}
    section: str | None = None
    name: str | None = None
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip()[:1] in ("#", ";"):
            continue
        line = line.rstrip()
        if not line:
            continue
        if line[0] == "[":
            header = line
            for char in "#;":
                header = header.split(char)[0].rstrip()
            if header.endswith("]"):
                section, name = header[1:-1], None
                if not section or section in sections:
                    raise ValueError(f"line {number}: empty or duplicate section")
                sections[section] = {}
                continue
            data = line.strip()
        elif not line[0].isspace():
            key, sep, value = line.partition("=")
            if not sep or ":" in key:
                key, sep, value = line.partition(":")
            if not sep:
                raise ValueError(f"line {number}: unexpected line")
            if section is None:
                raise ValueError(f"line {number}: no section header")
            name = key.strip()
            if name in sections[section]:
                raise ValueError(f"line {number}: duplicate name")
            sections[section][name] = value.strip()
            continue
        else:
            data = line.strip()
        if section is None or name is None:
            raise ValueError(f"line {number}: unexpected value continuation")
        before = sections[section][name]
        sections[section][name] = f"{before}\n{data}" if before else data
    return sections


def _pytest_section(name: str, text: str) -> Any:
    """What pytest reads of the configuration file *name*: its pytest table
    (pyproject.toml) or section (``None`` when it has none).

    ``ValueError`` when pytest could not read the file at all.
    """
    if name == "pyproject.toml":
        import tomllib

        tool = tomllib.loads(text).get("tool", {})
        return tool.get("pytest") if isinstance(tool, dict) else None
    section = "tool:pytest" if name == "setup.cfg" else "pytest"
    return _iniconfig_sections(text).get(section)


def _ini_addopts(name: str, text: str) -> list[str] | None:
    """The ``addopts`` tokens of the configuration file *name* (``[]`` if none)."""
    options = _pytest_section(name, text) or {}
    if name == "pyproject.toml" and isinstance(options, dict):
        options = options.get("ini_options", options)
    value = options.get("addopts", []) if isinstance(options, dict) else []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        tokens = [v for v in value if isinstance(v, str)]
        if len(tokens) == len(value):
            return tokens
    return None


def _split_ini_plugins(tokens: list[str]) -> tuple[list[str], list[str]] | None:
    """*tokens* without their ``-p NAME``, and the NAMEs (``no:`` ones stay).

    ``None`` for a token that may hide a ``-p`` in a bundle of short options
    (``-vp name``): rather than guess, the image's file is not used.
    """
    rest: list[str] = []
    plugins: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token == "-p" or (token.startswith("-p") and len(token) > 2):
            if token == "-p":
                if index == len(tokens):
                    return None
                name = tokens[index]
                index += 1
            else:
                name = token[2:]
            if name.startswith("no:"):
                rest += ["-p", name]
            else:
                plugins.append(name)
            continue
        if token.startswith("-") and not token.startswith("--"):
            for char in token[1:]:
                if char == "p":
                    return None
                # The rest of the token is this option's value.
                if char in "kmrWoc":
                    break
        rest.append(token)
    return rest, plugins


def _parse_pytest_config_probe(
    stdout: str, marker: str
) -> dict[tuple[str, str], tuple[str, str]]:
    """The probe's records: ``(name, side) -> (kind, text)``."""
    records: dict[tuple[str, str], tuple[str, str]] = {}
    if not stdout.startswith(marker + " "):
        return records
    parts = ("\n" + stdout).split("\n" + marker + " ")[1:]
    for index, part in enumerate(parts):
        header, _, text = part.partition("\n")
        fields = header.split(" ")
        if len(fields) != 3 or fields[0] not in _PYTEST_CONFIG_NAMES:
            return {}
        name, side, kind = fields
        if side not in ("image", "work") or (name, side) in records:
            return {}
        if index == len(parts) - 1 and text.endswith("\n"):
            text = text[:-1]
        records[name, side] = (kind, text if kind == "file" else "")
    return records


def _choose_unchanged_pytest_config(
    records: dict[tuple[str, str], tuple[str, str]],
) -> tuple[str, str] | None:
    """The configuration file pytest would read in the image's workspace and
    its image text, when the agent left what pytest reads of it unchanged.

    pytest takes the first of :data:`_PYTEST_CONFIG_NAMES` that is pytest.ini
    or .pytest.ini (even empty) or has a pytest table or section. That choice
    is made on the image's copies, so a file only the agent added is never
    read. The workspace's copy must give pytest the same table or section:
    an edit anywhere else in the file (a dependency in pyproject.toml, tox's
    own sections, setup.cfg's [options]) does not matter, as pytest reads
    nothing else, but a copy pytest could not parse does. ``None`` keeps
    ``-c /dev/null``.
    """
    for name in _PYTEST_CONFIG_NAMES:
        image = records.get((name, "image"))
        if image is None:
            continue
        kind, text = image
        if kind != "file" or "\ufffd" in text:
            return None
        try:
            section = _pytest_section(name, text)
        except ValueError:
            return None
        if section is None and name not in ("pytest.ini", ".pytest.ini"):
            continue
        work = records.get((name, "work"))
        if work is None or work[0] != "file" or "\ufffd" in work[1]:
            return None
        try:
            if _pytest_section(name, work[1]) != section:
                return None
        except ValueError:
            return None
        return name, text
    return None


# A literal absolute path to a Python file in a verifier script.
_ABSOLUTE_PY_PATH_RE = re.compile(r"(?<![\w.$}/:~-])(/[\w.+@%/-]*\.py)\b")


def _pytest_may_read_the_workspace_config(task: "Task", workspace: str) -> bool:
    """Whether the verifier's pytest runs on the workspace's own tests, where
    pytest itself would read the workspace's configuration file.

    pytest picks its configuration file by walking up from the paths it is
    given, not from the folder it runs in. ``cd /app && pytest
    /tests/test_outputs.py`` never reads /app's file, and its tests were
    written and checked under ``-c /dev/null``: forcing /app's ``addopts =
    --cov``, ``python_functions`` or ``filterwarnings = error`` on them would
    fail a correct answer (V3-R1-A-1). So the image's file is used only when
    the verifier package holds no Python file (its tests are not in /tests or
    /verifier) and no verifier script names a Python file by an absolute path
    outside the workspace (a test file test.sh writes to /tmp): test.sh then
    runs the project's own tests in the workspace, as SWE-bench-style tasks
    do. A package BenchFlow cannot read keeps ``-c /dev/null``.
    """
    verifier_dir = getattr(getattr(task, "paths", None), "tests_dir", None)
    if not isinstance(verifier_dir, Path) or not verifier_dir.is_dir():
        return False
    try:
        for _root, _dirs, names in os.walk(verifier_dir):
            if any(name.endswith(".py") for name in names):
                return False
    except OSError:
        return False
    texts = _verifier_script_texts(task)
    if texts is None:
        return False
    inside = workspace.rstrip("/") + "/"
    for text in texts:
        for match in _ABSOLUTE_PY_PATH_RE.finditer(_SCRIPT_COMMENT_RE.sub("", text)):
            if not match.group(1).startswith(inside):
                return False
    return True


async def _image_pytest_ini(env, workspace: str) -> ImagePytestIni | None:
    """The image's pytest configuration file, when the agent left what pytest
    reads of it unchanged.

    The verifier runs pytest with ``-c /dev/null`` so that a configuration file
    the agent wrote or edited is never read. That also hid the image's own
    one, and with it options its tests need: pytest-dev's tox.ini loads
    ``-p pytester``, without which every ``testdir`` test errors.
    The workspace's file is passed as it is (relative ``pythonpath`` entries
    resolve from its folder); what pytest reads of it is the image's.
    """
    marker = f"bf-pytest-config-{secrets.token_hex(16)}"
    result = await _read_exec(
        env,
        _system_probe(_IMAGE_PYTEST_INI_SCRIPT, "bf-pytest-ini", workspace, marker),
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    if _exec_return_code(result) != 0:
        return None
    records = _parse_pytest_config_probe(result.stdout or "", marker)
    chosen = _choose_unchanged_pytest_config(records)
    if chosen is None:
        if records:
            logger.info(
                "The verifier keeps -c /dev/null: the workspace's pytest "
                "configuration is not the image's, or cannot be read"
            )
        return None
    name, text = chosen
    path = f"{workspace.rstrip('/')}/{name}"
    try:
        tokens = _ini_addopts(name, text)
    except Exception:
        logger.debug("pytest configuration %s could not be read", path, exc_info=True)
        return None
    if tokens is not None and any(token.startswith("@") for token in tokens):
        raise RuntimeError(
            f"verifier refused pytest argument files (@file) in {path} addopts: "
            "their contents are not covered by the image configuration snapshot"
        )
    split = _split_ini_plugins(tokens) if tokens is not None else None
    if split is None:
        logger.info(
            "The verifier keeps -c /dev/null: the image's %s has addopts "
            "BenchFlow cannot read for -p plugins",
            path,
        )
        return None
    rest, plugins = split
    if not plugins:
        return ImagePytestIni(path=path)
    return ImagePytestIni(
        path=path, addopts=shlex.join(rest), plugins=tuple(dict.fromkeys(plugins))
    )


def _uses_native_verifier_dir(task: "Task") -> bool:
    paths = getattr(task, "paths", None)
    return getattr(paths, "uses_native_verifier_dir", False) is True


def _verifier_confcutdir(task: "Task") -> str:
    from benchflow.task.paths import SandboxPaths

    if _uses_native_verifier_dir(task):
        return str(SandboxPaths.verifier_code_dir)
    return str(SandboxPaths.tests_dir)


# Container-side script to enumerate pre-installed pytest11 entry points.
# Adapted from PR #1117 (tulerfeng);
# additionally check registration provenance, parent directories, and duplicate names.
#
# Discovery decides what goes on the verifier's ``-p`` allowlist, so a plugin
# reaching this list executes *inside* pytest with the power to rewrite a test
# report. A pytest11 entry point is therefore only trusted when the code it
# resolves to is root-owned and outside every agent-writable tree; both halves
# are needed, and neither is implied by the other:
#
#   forged   — ``importlib.metadata`` reads ``*.dist-info`` straight off
#              ``sys.path``, needing neither pip nor root, so the sandbox user
#              can register a plugin by creating a directory in its workspace.
#   editable — a root-installed editable project puts a root-owned dist-info in
#              site-packages while its *source* stays in the workspace, and
#              adds that workspace path to ``sys.path`` via a ``.pth``. Judging
#              such a plugin by its dist-info alone admits agent-owned code.
#
# Resolving the module covers both: a forged registration resolves into the
# workspace, and an editable project resolves to its workspace source. Three
# things about how that resolution is done are load-bearing:
#
#   * It goes through ``PathFinder``, which locates modules without importing
#     them — importing here would run the very code this is deciding whether
#     to trust.
#   * The first component is resolved against the full ``sys.path``, not a
#     pre-filtered one, because the question is which file pytest will
#     actually import, not which one it ideally would: a workspace module
#     shadowing a system plugin of the same name is what gets loaded.
#   * A dotted target is walked one component at a time, each step searching
#     only the locations its parent declares. Checking the top level alone
#     would clear ``trusted_pkg.evil`` on the strength of ``trusted_pkg``,
#     and a pkgutil-style namespace package widens its ``__path__`` at import
#     time to every same-named directory on ``sys.path`` — the workspace
#     included — which ``find_spec`` never sees. A component living only in
#     the workspace then resolves nowhere and the entry point is refused.
#
# Which ``sys.path`` that is differs by phase. Pre-verifier discovery only
# proposes ``-p`` names, so it starts from the verifier's base path (``/`` as
# cwd, trusted PYTHONPATH, no user site) and ignores registrations inside
# agent-writable trees: planted files can neither add a name nor drop or veto
# a protected one. The runtime guard then checks every ``-p`` name, discovered
# or declared, against the path pytest actually runs with, where a planted
# duplicate or shadowing module is refused.
#
# ``argv[1]`` is the JSON list of agent-writable path prefixes.
_DISCOVER_PYTEST_PLUGINS_SCRIPT = (
    Path(__file__).with_name("_pytest_plugin_guard.py").read_text()
)


def _blocked_verifier_path_prefixes(
    sandbox_user: str | None, workspace: str | None
) -> tuple[str, ...]:
    """Paths that must never be preserved as verifier PATH extras."""
    prefixes = list(_RUNTIME_PATH_PREFIXES)
    if workspace:
        prefixes.append(workspace)
    if sandbox_user:
        prefixes.append(f"/home/{sandbox_user}")
    return tuple(dict.fromkeys(prefixes))


async def _mount_provenance(
    env, blocked: tuple[str, ...], *, require_inspector: bool = False
) -> tuple[str, tuple[str, ...]]:
    """Read mount coordinates and physical blocked paths before ownership changes."""
    result = await _checked_read(
        env,
        (
            _path_inspector_prefix(env)
            if require_inspector or _path_inspector(env)
            else ""
        )
        + "[ -r /proc/self/mountinfo ] || exit 1; "
        "while IFS= read -r line; do printf '%s\\n' \"$line\"; done "
        "</proc/self/mountinfo; printf '# end\\n'; "
        f"for path in {' '.join(shlex.quote(p) for p in blocked)}; do "
        'if [ -d "$path" ]; then '
        '(cd -P "$path" && printf \'%s\\000\' "$PWD") || exit 1; '
        'elif [ -e "$path" ] || [ -L "$path" ]; then '
        'physical=$(readlink -f -- "$path" && printf .) || exit 1; '
        'physical=${physical%?}; physical=${physical%"\n"}; '
        "printf '%s\\000' \"$physical\"; fi; done; "
        "printf '\\000'",
        "Verifier hardening failed: reading workspace mount aliases",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    output = result.stdout or ""
    mountinfo, complete, physical = output.partition("# end\n")
    if not complete or not physical.endswith("\0"):
        raise RuntimeError("Verifier hardening failed: incomplete workspace mountinfo")
    prefixes = tuple(dict.fromkeys((*blocked, *filter(None, physical.split("\0")))))
    return mountinfo, prefixes


async def _blocked_before_freeze(env, blocked: tuple[str, ...]) -> tuple[str, ...]:
    """Capture mount provenance before chown can turn an alias into root's files."""
    mountinfo, prefixes = await _mount_provenance(env, blocked)
    try:
        return blocked_mount_aliases(mountinfo, prefixes)
    except ValueError as exc:
        raise RuntimeError(
            f"Verifier hardening failed: workspace mountinfo: {exc}"
        ) from exc


async def _service_exclusions_before_freeze(
    env,
    blocked: tuple[str, ...],
    service: str,
    *,
    target_seeds: tuple[str, ...] | None = None,
) -> tuple[PathExclusions, PathExclusions]:
    """Keep provenance in both namespaces, before shared files look root-owned."""
    main_info, main_prefixes = await _mount_provenance(
        env, blocked, require_inspector=True
    )
    try:
        main = await _index_exclusions(
            env,
            dataclasses.replace(
                PathExclusions.capture(main_info, main_prefixes),
                inodes=await _excluded_inodes(env, main_prefixes),
                inspector=_path_inspector(env),
            ),
        )
        in_target = _InService(env, service)
        target_info, target_prefixes = await _mount_provenance(
            in_target,
            blocked if target_seeds is None else target_seeds,
            require_inspector=True,
        )
        target = main.for_target(target_info, target_prefixes)
        return main, await _index_exclusions(
            in_target,
            dataclasses.replace(
                target,
                inodes=target.inodes
                | await _excluded_inodes(in_target, target_prefixes),
                inspector=_path_inspector(in_target),
            ),
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Verifier hardening failed: workspace mountinfo: {exc}"
        ) from exc


async def _path_exclusions_before_freeze(
    env, blocked: tuple[str, ...]
) -> PathExclusions:
    mountinfo, prefixes = await _mount_provenance(env, blocked, require_inspector=True)
    try:
        return await _index_exclusions(
            env,
            dataclasses.replace(
                PathExclusions.capture(mountinfo, prefixes),
                inodes=await _excluded_inodes(env, prefixes),
                inspector=_path_inspector(env),
            ),
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Verifier hardening failed: workspace mountinfo: {exc}"
        ) from exc


def _path_inspector(env) -> str:
    if isinstance(env, _InService):
        owner, service = env._env, env._service
    else:
        owner, service = env, "main"
    saved = getattr(owner, "_benchflow_path_inspectors", None)
    folder = saved.get(service, "") if isinstance(saved, dict) else ""
    return folder if isinstance(folder, str) else ""


def _path_inspector_prefix(env) -> str:
    folder = _path_inspector(env)
    if not folder:
        # No command from an unrecorded checkpoint may bootstrap this check.
        raise RuntimeError(
            "Verifier hardening failed: missing pre-agent PATH inspector; recreate this sandbox with a fresh baseline"
        )
    return _on_path(f"PATH={shlex.quote(folder)}; export PATH; ")


def _path_inspector_command(folder: str, script: str, *args: str) -> str:
    if not folder:
        return "printf 'missing pre-agent PATH inspector' >&2; exit 125"
    return _on_path(
        f"PATH={shlex.quote(folder)}; export PATH; "
        f"{shlex.quote(folder + '/sh')} -c {shlex.quote(script)} " + shlex.join(args)
    )


async def _excluded_inodes(
    env, prefixes: tuple[str, ...]
) -> frozenset[tuple[int, int]]:
    """Read physical identities without following links out of excluded trees.

    File contents and permissions are immaterial. Keep every inode, including
    non-executables that become executable later and hard-linked symlinks.
    ``find -H`` follows only the seed; it crosses nested mounts like freeze.
    """
    result = await _checked_read(
        env,
        _path_inspector_prefix(env) + "for p in " + shlex.join(prefixes) + "; do "
        'if [ -e "$p" ] || [ -L "$p" ]; then '
        'find -H "$p" ! -type d -exec stat -c "%d:%i" -- {} + || exit 1; '
        "fi; done; "
        'if [ -f "$PATH/before" ]; then cat "$PATH/before" || exit 1; fi; '
        'if [ -d "$PATH/before-links" ]; then '
        'find "$PATH/before-links" ! -type d -exec stat -c "%d:%i" -- {} + || exit 1; fi; '
        "printf 'complete\\n'",
        "Verifier hardening failed: capturing excluded inode identities",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    lines = (result.stdout or "").splitlines()
    if (
        not lines
        or lines[-1] != "complete"
        or any(not re.fullmatch(r"[0-9]+:[0-9]+", line) for line in lines[:-1])
    ):
        raise RuntimeError("Verifier hardening failed: incomplete inode identities")
    return frozenset(
        (int(dev), int(ino)) for line in lines[:-1] for dev, ino in [line.split(":")]
    )


async def _index_exclusions(env, captured: PathExclusions) -> PathExclusions:
    """Materialize numeric identities without an unbounded exec argument.

    The index is private to this container's pre-agent inspector. No source
    pathname is carried to a target, only device/inode pairs in bounded batches.
    """
    if not captured.inodes:
        return captured
    directory = captured.inspector + "/inodes-" + uuid.uuid4().hex
    prefix = _path_inspector_prefix(env)
    await _checked_exec(
        env,
        prefix + f"mkdir -m 700 {shlex.quote(directory)}",
        "Verifier hardening failed: creating inode exclusion index",
        user="root",
    )
    identities = [f"{dev}:{ino}" for dev, ino in sorted(captured.inodes)]
    for start in range(0, len(identities), 1000):
        await _checked_exec(
            env,
            prefix
            + "for i in "
            + " ".join(identities[start : start + 1000])
            + f'; do : > {shlex.quote(directory)}/"$i" || exit 1; done',
            "Verifier hardening failed: writing inode exclusion index",
            user="root",
        )
    return dataclasses.replace(captured, inode_index=directory)


# A separate verifier sandbox never ran the agent. What came from the agent's
# run is what the transfer wrote (the frozen workspace and declared artifacts,
# ``agent_paths``) and what lands under /logs (the collected /logs/artifacts,
# the published trajectory). The plugin guard distrusts exactly those there.
_SEPARATE_VERIFIER_AGENT_PREFIXES = ("/logs",)


def _separate_verifier_blocked_prefixes(
    agent_paths: "tuple[str, ...] | list[str]",
) -> tuple[str, ...]:
    """The plugin guard's blocked prefixes in a separate verifier sandbox."""
    return tuple(dict.fromkeys([*agent_paths, *_SEPARATE_VERIFIER_AGENT_PREFIXES]))


def _blocked_verifier_pythonpath_prefixes(
    sandbox_user: str | None,
) -> tuple[str, ...]:
    """Paths blocked from verifier PYTHONPATH.

    Unlike PATH, the workspace is NOT blocked: PYTHONPATH entries like /app
    are set by the Dockerfile for project imports, and the workspace is
    already importable via CWD/pytest sys.path insertion regardless.
    """
    prefixes = list(_RUNTIME_PATH_PREFIXES)
    if sandbox_user:
        prefixes.append(f"/home/{sandbox_user}")
    return tuple(dict.fromkeys(prefixes))


def _merge_trusted_verifier_path(extras: list[str]) -> str:
    """Prepend validated image PATH entries to the verifier allowlist."""
    kept: list[str] = []
    seen: set[str] = set(_SAFE_VERIFIER_PATH_PARTS)
    for entry in extras:
        if entry and entry not in seen:
            seen.add(entry)
            kept.append(entry)
    return ":".join([*kept, *_SAFE_VERIFIER_PATH_PARTS])


# The verifier's trusted PATH, judged by root after the agent's turn, in
# /bin/sh: the check runs on images without Python too (#1174). The rule for
# root's folders also covers their routes and entries: a closed directory
# alone does not stop the agent rewriting a writable tool in it. A folder
# another user owns is kept only for a non-root agent
# user whose uid and groups ``id`` reads, and only when that user could never
# have changed it or what the verifier runs from it (Rust 0.8.2's P1.2c,
# decision G-9; another user's ~/.cargo/bin, say): the way to it from / (each
# link followed one name at a time, at most 40, never through /proc or
# /dev/fd) passes no link the agent user owns and no folder it owns, may write
# by the owner, group or other bits, or that carries a POSIX ACL (``+`` after
# ``ls -l``'s mode, or an entry ``getfacl`` lists where the image has it); no
# entry in it is the agent user's, carries an ACL or (unless a link) may be
# written by group or others; and every link in it leads, by the same walk,
# through and to what the agent user cannot change. Nothing runs as the
# agent user: a process it left could answer for it.
#
# A program in a judged folder whose route goes on into the workspace at its
# top stays, for a non-root agent user, only when its name is absent from the
# folder's pre-agent record ($5, _PATH_NAMES_RECORD): something other than the
# agent user added it during the turn, as root's `ln -s /app/sqlite/sqlite3
# /usr/local/bin/sqlite3` in an oracle (ACC-RUN2). A name the image shipped,
# a folder the record does not hold, or no record: left out as before. The
# folder's own route, and every route the root-interpreter judge walks, still
# refuse the workspace.
#
# $1: the entries to judge, one a line; $2: the agent user, or '' for root's
# folders only; $3: the projection root; $4: the workspace, or ''; $5: the
# pre-agent record of the PATH folders' names, or ''. The
# compiled predicate tests the captured filesystem coordinates (or an
# uncaptured, namespace-local policy). Prints, for each entry in order,
# ``K <entry>`` (kept: root's), ``O <entry>`` (kept: another user's) or
# ``D <entry>`` and then ``W <why>`` (left out); nothing for an entry that is
# no folder.
_TRUSTED_PATH_EXTRAS_SCRIPT = r"""
set -f
nl='
'
a=$2
bf_filter_root=${3-}
bf_count=0
au=
ag=
if [ -n "$a" ]; then
  au=$(id -u "$a" 2>/dev/null) || au=
  case "$au" in
    0) a=; au= ;;
    ''|*[!0-9]*) au= ;;
  esac
  if [ -n "$au" ]; then
    ag=$(id -G "$a" 2>/dev/null) || au=
    ag=" $ag "
  fi
fi
command -v getfacl >/dev/null 2>&1 && bf_acl=1 || bf_acl=
bf_ws=; bf_wsp=; bf_ws_tool=; bf_rec=; bf_slot=
if [ -n "$au" ] && [ -n "${4-}" ] && [ -n "${5-}" ] && [ -f "$5/complete" ] &&
  [ -f "$5/index" ]; then
  bf_rec=$5
  case "$4" in /*) bf_ws=$4; bf_wsp=$(cd -P -- "$4" 2>/dev/null && pwd -P) || bf_wsp= ;; esac
fi
# The folder's slot in the pre-agent record; fails when it holds none.
bf_slot_of() {
  while IFS= read -r bf_line; do
    [ "${bf_line#* }" = "$1" ] && { printf '%s\n' "${bf_line%% *}"; return 0; }
  done < "$bf_rec/index"
  return 1
}
__BF_EXCLUSIONS__
bf_may() {
  set -- "$1" $(ls -ldn -- "$1" 2>/dev/null)
  [ $# -ge 5 ] || { echo "$1 could not be read"; return 0; }
  if [ -z "$au" ] && [ "$4" != 0 ]; then
    echo "$1 is not root's"; return 0
  fi
  [ "$4" = "$au" ] && { echo "$1 is the agent user's"; return 0; }
  case "$2" in ????????w*) echo "$1 is writable by others"; return 0 ;; esac
  case "$2" in ?????w*)
    [ -n "$au" ] || { echo "$1 is writable by group"; return 0; }
    case "$ag" in *" $5 "*) echo "$1 is writable by a group of the agent user"; return 0 ;; esac ;;
  esac
  case "$2" in ??????????+*) echo "$1 carries a POSIX ACL"; return 0 ;; esac
  if [ -n "$bf_acl" ]; then
    acl=$(getfacl -sp -- "$1" 2>/dev/null) || { echo "$1 ACL could not be read"; return 0; }
    [ -z "$acl" ] || { echo "$1 carries a POSIX ACL"; return 0; }
  fi
  return 1
}
# Only the byte-preserving system identity walk accepts every pathname.
# Permission-checked extras admit no control or PATH separator in any hop.
bf_representable() {
  [ -z "$bf_filter_root" ] || [ -n "$bf_identity_only" ] && return 0
  case "$1" in *[[:cntrl:]:]*) return 1 ;; esac
  return 0
}
bf_quote() (
  value=$1
  printf "'"
  while :; do
    case "$value" in
      *"'"*) printf '%s%s' "${value%%\'*}" "'\\''"; value=${value#*\'} ;;
      *) printf "%s'" "$value"; break ;;
    esac
  done
)
bf_route() {
  rest=$1; cur=; n=0
  bf_representable "$rest" || return 2
  case "$rest" in /*) ;; *) echo "$rest is not an absolute path"; return 1 ;; esac
  b=$(bf_under /) && { echo "under $b"; return 1; }
  if [ -z "$bf_identity_only" ]; then
    why=$(bf_may /) && { echo "$why"; return 1; }
  fi
  while [ -n "$rest" ]; do
    c=${rest%%/*}
    case "$rest" in */*) rest=${rest#*/} ;; *) rest= ;; esac
    case "$c" in ''|.) continue ;; ..) cur=${cur%/*}; continue ;; esac
    p=$cur/$c
    # Check the identity of every component before following it, including
    # a file bind mount or a link which later returns to a protected tree.
    if b=$(bf_under "$p"); then
      echo "$p is under $b"
      # A tool's route that enters the workspace at its top (bf_ws_tool): 3,
      # which bf_tool keeps only for a name added during the turn.
      if [ -n "$bf_ws_tool" ] && [ "$p" = "$b" ] && [ -n "$bf_ws" ] &&
        { [ "$p" = "$bf_ws" ] || [ "$p" = "$bf_wsp" ]; }; then
        return 3
      fi
      return 1
    fi
    case "$p" in
      /proc|/proc/*|/dev/fd|/dev/fd/*|/dev/stdin|/dev/stdout|/dev/stderr)
        echo "the way goes through $p"; return 1 ;;
    esac
    if [ -L "$p" ]; then
      n=$((n + 1))
      [ "$n" -le 40 ] || { echo "the way passes more than 40 links"; return 1; }
      if [ -z "$bf_identity_only" ]; then
        set -- $(ls -ldn -- "$p" 2>/dev/null)
        [ $# -ge 4 ] || { echo "$p could not be read"; return 1; }
        if [ -z "$au" ] && [ "$3" != 0 ]; then
          echo "$p is not root's"; return 1
        fi
        [ "$3" = "$au" ] && { echo "$p is a link the agent user made"; return 1; }
      fi
      t=$(readlink -- "$p" && echo .) || { echo "$p could not be read"; return 1; }
      t=${t%"$nl."}
      [ -n "$t" ] || { echo "$p has an empty link target"; return 1; }
      bf_representable "$t" || return 2
      case "$t" in /*) cur= ;; esac
      rest=$t${rest:+/$rest}
      continue
    fi
    if [ -e "$p" ]; then
      if [ -z "$bf_identity_only" ]; then
        why=$(bf_may "$p") && { echo "$why"; return 1; }
      fi
    elif [ -n "$rest" ]; then
      echo "$p is missing"; return 1
    fi
    cur=$p
  done
  return 0
}
# A program's route in the folder being judged: bf_route's answer, except
# that a route into the workspace (3) is kept (0) only when the folder's
# pre-agent record ($bf_slot) does not hold the program's name.
bf_tool() {
  why=$(bf_ws_tool=1; bf_route "$1"); rc=$?
  if [ "$rc" = 3 ]; then
    [ -n "$bf_slot" ] && [ ! -e "$bf_rec/$bf_slot/${1##*/}" ] &&
      [ ! -L "$bf_rec/$bf_slot/${1##*/}" ] && return 0
    rc=1
  fi
  [ -z "$why" ] || printf '%s\n' "$why"
  return "$rc"
}
bf_judge() (
  d=$1
  bf_filtered=
  if [ -n "$bf_rec" ]; then bf_slot=$(bf_slot_of "$d") || bf_slot=; fi
  why=$(bf_route "$d"); rc=$?
  case "$rc" in
    0) ;;
    2) bf_filtered=1 ;;
    *) printf '%s\n' "$why"; exit 1 ;;
  esac
  set +f
  for e in "$d"/* "$d"/.[!.]* "$d"/..?*; do
    [ -e "$e" ] || [ -L "$e" ] || continue
    if ! bf_representable "$e"; then
      bf_filtered=1
      printf 'T '; bf_encode "$e"; printf '\n'
      continue
    fi
    if [ -z "$bf_identity_only" ]; then
      # Numeric metadata only: filenames and link targets are never records.
      set -- $(stat -c '%a %u' -- "$e" 2>/dev/null)
      [ $# -eq 2 ] || { echo "its entry metadata could not be read"; exit 1; }
      case "$1" in ''|*[!0-7]*) echo "invalid entry mode"; exit 1 ;; esac
      if [ "$2" = "$au" ]; then echo "it holds an entry the agent user owns"; exit 1; fi
      if [ ! -L "$e" ] && [ $((0$1 & 022)) -ne 0 ]; then
        echo "it holds an entry group or others may write"; exit 1
      fi
      set -- $(ls -ldn -- "$e" 2>/dev/null)
      [ $# -ge 4 ] || { echo "its entry ACL could not be read"; exit 1; }
      case "$1" in ??????????+*) echo "it holds an entry with a POSIX ACL"; exit 1 ;; esac
    fi
    # A closed root-owned folder can still hold a writable executable or a
    # link to one. Check regular executable routes too (including their ACLs).
    if [ -n "$bf_identity_only" ]; then
      # Only executable files are tools on the protected system route.
      # Trusted images can also hold directory links or dangling links here.
      [ -f "$e" ] && [ -x "$e" ] || continue
    elif [ ! -L "$e" ]; then
      [ -f "$e" ] && [ -x "$e" ] || continue
    fi
    why=$(bf_tool "$e"); rc=$?
    case "$rc" in
      0) ;;
      2)
        bf_filtered=1
        printf 'T '; bf_encode "$e"; printf '\n' ;;
      *)
        if [ -L "$e" ]; then
          printf '%s\n' "its link ${e##*/} leads where the agent user can change: $why"
        else
          printf '%s\n' "its executable ${e##*/} can be changed: $why"
        fi
        exit 1 ;;
    esac
  done
  if [ -n "$bf_filtered" ]; then
    # A fresh, root-owned projection cannot express an omitted route. Keep
    # sibling tools and PATH order without modifying the trusted image.
    # No permissive inherited umask may expose a writable launcher before
    # its final mode is set.
    umask 077
    projection=$bf_filter_root-$bf_count
    mkdir -m 755 -- "$projection" || exit 1
    for e in "$d"/* "$d"/.[!.]* "$d"/..?*; do
      [ -f "$e" ] && [ -x "$e" ] || continue
      bf_tool "$e" >/dev/null || continue
      # Execute the original pathname so scripts still find resources next
      # to $0. A symlink in this directory would change that script identity.
      { printf '#!/bin/sh\nexec '; bf_quote "$e"; printf ' "$@"\n'; } > "$projection/${e##*/}" || exit 1
      chmod 555 -- "$projection/${e##*/}" || exit 1
    done
    printf 'F '; bf_encode "$d"; printf '\nP '; bf_encode "$projection"; printf '\n'
  fi
  exit 0
)
bf_drop() {
  printf 'D '; bf_encode "$1"; printf '\nW '; bf_encode "$2"; printf '\n'
}
__BF_SYSTEM_EXCLUSIONS__
IFS=$nl
for d in __BF_ENTRIES__; do
  unset IFS
  bf_count=$((bf_count + 1))
  if b=$(bf_under "$d"); then bf_drop "$d" "under $b"; continue; fi
  real=$(cd -P "$d" 2>/dev/null && pwd -P && echo .) || continue
  real=${real%"$nl."}
  [ -n "$real" ] || continue
  if b=$(bf_under "$real"); then bf_drop "$d" "it leads to $real, under $b"; continue; fi
  set -- $(ls -ldn "$real" 2>/dev/null)
  if [ $# -lt 4 ]; then bf_drop "$d" "$real could not be read"; continue; fi
  if [ "$3" = 0 ]; then
    case "$1" in
      ?????w*|????????w*) bf_drop "$d" "root's, but writable by group or others" ;;
      *)
        if why=$(bf_judge "$d"); then [ -z "$why" ] || printf '%s\n' "$why"; printf 'K '; bf_encode "$d"; printf '\n'
        else bf_drop "$d" "root's, but $why"; fi ;;
    esac
    continue
  fi
  if [ -z "$au" ]; then
    if [ -n "$a" ]; then
      bf_drop "$d" "not root's (uid $3), and the agent user's uid could not be read"
    else
      bf_drop "$d" "not root's (uid $3)"
    fi
    continue
  fi
  if why=$(bf_judge "$d"); then
    [ -z "$why" ] || printf '%s\n' "$why"
    printf 'O '; bf_encode "$d"; printf '\n'
  else
    bf_drop "$d" "not root's (uid $3), and $why"
  fi
done
""".strip()


def _path_entries_to_judge(raw_path: str) -> list[str]:
    """The image's PATH entries the trusted-PATH program judges, in order.

    Absolute, without a NUL, each once, and none of the safe folders the
    verifier has anyway. Shell literals carry entry bytes, not line records.
    """
    entries: list[str] = []
    seen: set[str] = set(_SAFE_VERIFIER_PATH_PARTS)
    for entry in raw_path.split(":"):
        if "\x00" in entry:
            raise RuntimeError(
                "Verifier hardening failed: PATH entry contains a NUL: " + repr(entry)
            )
        if not entry or entry in seen or not entry.startswith("/"):
            continue
        seen.add(entry)
        entries.append(entry)
    return entries


def _relative_path_entries(raw_path: str) -> list[str]:
    """The image's relative PATH entries, each once, an empty one as ``.``.

     The verifier's PATH never keeps one (it names another folder in every
     working folder), but each is a folder left out like any other: dropping it
     silently made a program found only there look merely missing
    .
    """
    entries: list[str] = []
    for entry in raw_path.split(":"):
        if "\x00" in entry:
            raise RuntimeError(
                "Verifier hardening failed: PATH entry contains a NUL: " + repr(entry)
            )
        entry = entry or "."
        if entry.startswith("/"):
            continue
        if entry not in entries:
            entries.append(entry)
    return entries


def _trusted_path_extras_cmd(
    raw_path: str,
    blocked_prefixes: tuple[str, ...] | PathExclusions,
    agent: str | None = None,
    workspace: str | None = None,
    record: str | None = None,
) -> str:
    """Build the container-side command that judges verifier PATH extras.

    Runs ``_TRUSTED_PATH_EXTRAS_SCRIPT`` under ``/bin/sh``, so an image with
    no Python is judged too (#1174). *agent* is the sandbox user, whose uid
    the rule for folders root does not own needs; ``None`` (an agent that ran
    as root, or a PYTHONPATH) keeps root's folders only. *workspace* (by
    default the one hardening is judging, :data:`_path_workspace`), with a
    non-root *agent* and the pre-agent *record* of the PATH folders' names
    (by default the inspector's, :data:`_PATH_NAMES_RECORD`): a tool whose
    route enters it stays when the record does not hold its name (ACC-RUN2).
    """
    if workspace is None:
        workspace = _path_workspace.get()
    if record is None and isinstance(blocked_prefixes, PathExclusions):
        record = blocked_prefixes.inspector + "/" + _PATH_NAMES_DIR
    entries = "\n".join(_path_entries_to_judge(raw_path))
    predicate = _exclusion_predicate(blocked_prefixes)
    if (
        isinstance(blocked_prefixes, PathExclusions)
        and not blocked_prefixes.system_route_clear
    ):
        # These folders are always on the verifier's route. Image ownership
        # is trusted there, but a task's file bind mount or symlink can still
        # expose an excluded workspace object after freeze. Refuse that route
        # rather than silently restoring it as the fallback system PATH.
        folders = " ".join(shlex.quote(p) for p in _SAFE_VERIFIER_PATH_PARTS)
        system_check = (
            _known_identity_predicate(predicate, bool(blocked_prefixes.inodes))
            + "\n"
            + _SYSTEM_ROUTE_FAST_SH.replace("__BF_FOLDERS__", folders)
            + "\nbf_identity_only=1\n"
            # The fast walk only ever clears; on anything else the full walk
            # below decides and writes its message.
            + "if bf_fast_system; then bf_fast=1; else bf_fast=; fi; set -f\n"
            + 'if [ -z "$bf_fast" ]; then\nfor d in '
            + folders
            + '; do\n  [ -d "$d" ] || continue\n'
            + '  why=$(bf_judge "$d") || { printf \'excluded system PATH %s: %s\\n\' "$d" "$why" >&2; exit 1; }\n'
            + "done\nfi\nbf_identity_only=\n"
        )
    else:
        system_check = "bf_identity_only=\n"
    script = (
        PATH_FIELD_ENCODE_SH
        + "\n"
        + re.sub(
            r"__BF_(?:EXCLUSIONS|SYSTEM_EXCLUSIONS|ENTRIES)__",
            lambda match: (
                predicate
                if match[0] == "__BF_EXCLUSIONS__"
                else shlex.join(_path_entries_to_judge(raw_path))
                if match[0] == "__BF_ENTRIES__"
                else system_check
            ),
            _TRUSTED_PATH_EXTRAS_SCRIPT,
        )
    )
    if isinstance(blocked_prefixes, PathExclusions):
        # All dependencies, including the shell, come from the pre-agent copy.
        # Checking a live utility with another live utility is circular.
        return _path_inspector_command(
            blocked_prefixes.inspector,
            script,
            "bf-trusted-path",
            entries,
            agent or "",
            blocked_prefixes.inspector + "/path-" + uuid.uuid4().hex,
            workspace or "",
            record or "",
        )
    return _system_probe(
        script,
        "bf-trusted-path",
        entries,
        agent or "",
        "",
        workspace or "",
        record or "",
    )


# The system check's walk without a subshell per component and with one
# ``stat`` per folder (PERF-SBX): ``bf_route`` with ``bf_identity_only=1``
# for each system folder, then each executable's route continues from where
# the folder's ended (the same components in the same order: a route through
# "$d/$e" walks "$d" first; the folder's last component must exist, as the
# longer route requires). Every leaf identity comes from one ``stat -c
# %d:%i`` over the folder's tools, the same tool and format ``bf_under``
# runs per path; any mismatch in the count returns 1. It returns 0 only when
# every route passes, and 1 on anything else, so the full walk then runs.
_SYSTEM_ROUTE_FAST_SH = r"""
bf_fr() {
  bf_c=$1; bf_k=$2; bf_r=$3; bf_id=$4
  while [ -n "$bf_r" ]; do
    bf_x=${bf_r%%/*}
    case "$bf_r" in */*) bf_r=${bf_r#*/} ;; *) bf_r= ;; esac
    case "$bf_x" in ''|.) continue ;; ..) bf_c=${bf_c%/*}; continue ;; esac
    bf_p=$bf_c/$bf_x
    if [ -n "$bf_id" ]; then
      bf_under_known "$bf_p" "$bf_id" >/dev/null && return 1
      bf_id=
    else
      bf_under "$bf_p" >/dev/null && return 1
    fi
    case "$bf_p" in
      /proc|/proc/*|/dev/fd|/dev/fd/*|/dev/stdin|/dev/stdout|/dev/stderr) return 1 ;;
    esac
    if [ -L "$bf_p" ]; then
      bf_k=$((bf_k + 1))
      [ "$bf_k" -le 40 ] || return 1
      bf_t=$(readlink -- "$bf_p" && echo .) || return 1
      bf_t=${bf_t%"$nl."}
      [ -n "$bf_t" ] || return 1
      case "$bf_t" in /*) bf_c= ;; esac
      bf_r=$bf_t${bf_r:+/$bf_r}
      continue
    fi
    if [ ! -e "$bf_p" ] && [ -n "$bf_r" ]; then return 1; fi
    bf_c=$bf_p
  done
  return 0
}
bf_fast_system() {
  for bf_d in __BF_FOLDERS__; do
    [ -d "$bf_d" ] || continue
    case "$bf_d" in /*) ;; *) return 1 ;; esac
    bf_under / >/dev/null && return 1
    bf_fr "" 0 "$bf_d" "" || return 1
    [ -n "$bf_c" ] && [ -e "$bf_c" ] || return 1
    bf_dc=$bf_c; bf_dn=$bf_k
    set +f
    set --
    for bf_e in "$bf_d"/* "$bf_d"/.[!.]* "$bf_d"/..?*; do
      [ -e "$bf_e" ] || [ -L "$bf_e" ] || continue
      [ -f "$bf_e" ] && [ -x "$bf_e" ] || continue
      set -- "$@" "$bf_dc/${bf_e##*/}"
    done
    set -f
    [ "$#" -gt 0 ] || continue
    bf_ids=
    if [ -n "$bf_inodes" ]; then
      bf_ids=$(stat -c '%d:%i' -- "$@") || return 1
      bf_ids=$bf_ids$nl
    fi
    for bf_p in "$@"; do
      bf_id=
      if [ -n "$bf_inodes" ]; then
        bf_id=${bf_ids%%"$nl"*}
        bf_ids=${bf_ids#*"$nl"}
        case "$bf_id" in ''|*[!0-9:]*) return 1 ;; esac
      fi
      bf_fr "$bf_dc" "$bf_dn" "${bf_p##*/}" "$bf_id" || return 1
    done
    [ -z "$bf_ids" ] || return 1
  done
  return 0
}
""".strip()

_IDENTITY_STAT_LINE = (
    '    bf_inode=$(stat -c "%d:%i" -- "$1") || '
    '{ printf "unreadable identity"; return 0; }'
)


def _known_identity_predicate(predicate: str, inodes: bool) -> str:
    """``bf_under_known PATH IDENTITY``: :meth:`PathExclusions.shell_predicate`
    with the identity ``stat`` already read (the fast walk's batch), else
    the same function; and ``bf_inodes``, set when identities are checked."""
    if not inodes:
        return 'bf_inodes=\nbf_under_known() { bf_under "$1"; }'
    if predicate.count(_IDENTITY_STAT_LINE) != 1 or not predicate.startswith(
        "bf_under() {"
    ):
        raise RuntimeError("Verifier hardening failed: unexpected PATH predicate shape")
    return "bf_inodes=1\n" + predicate.replace(
        _IDENTITY_STAT_LINE, "    bf_inode=$2"
    ).replace("bf_under() {", "bf_under_known() {", 1)


def _exclusion_predicate(blocked_prefixes: tuple[str, ...] | PathExclusions) -> str:
    """The shell ``bf_under`` the route checks call for every component."""
    if isinstance(blocked_prefixes, PathExclusions):
        return blocked_prefixes.shell_predicate()
    # Uncaptured local policies (e.g. PYTHONPATH) have no other namespace.
    # Quote each prefix instead of a newline-delimited name protocol. Like
    # PathExclusions, print the left-out prefix that holds the path, the
    # longest first: "/tmp/tools: under /tmp", not "under /tmp/tools".
    stripped: set[str] = {p.rstrip("/") for p in blocked_prefixes}
    prefixes = sorted(stripped, key=lambda p: -len(p))
    arms = "".join(
        ("/*" if p == "" else f"{shlex.quote(p)}|{shlex.quote(p + '/')}*")
        + f") printf '%s\\n' {shlex.quote(p or '/')}; return 0 ;; "
        for p in prefixes
    )
    return (
        "bf_under() { return 1; }"
        if not arms
        else f'bf_under() {{ case "$1" in {arms}esac; return 1; }}'
    )


# The trusted-PATH program's route check (bf_may, bf_route), shared with the
# root-interpreter judge below: one rule for what root may start.
_PATH_ROUTE_SH = _TRUSTED_PATH_EXTRAS_SCRIPT.split("\nbf_judge() (", 1)[0]

# $1: candidate interpreters, one per line, in order; $2: the sandbox user.
# Prints "A <file>" for the first whose whole route passes, after an
# "R <file>" / "W <why>" pair for each present candidate that did not. Last,
# the file a rejected candidate's links resolve to is judged on its own path,
# but only one named python3 or python3.N: an image whose only interpreter a
# workspace venv links to keeps the freeze (V8-R2-A-1), while a link the
# agent rewrote can name nothing but another python3 that passes the rule.
_ROOT_PYTHON_JUDGE_SH = r"""
bf_identity_only=
bf_later=
IFS=$nl
for c in $1; do
  unset IFS
  [ -f "$c" ] && [ -x "$c" ] || continue
  if why=$(bf_route "$c"); then
    printf 'A '; bf_encode "$c"; printf '\n'; exit 0
  fi
  printf 'R '; bf_encode "$c"; printf '\nW '; bf_encode "$why"; printf '\n'
  t=$(readlink -f -- "$c" 2>/dev/null) || continue
  case "$t" in ''|"$c"|*"$nl"*) continue ;; esac
  case "${t##*/}" in python3|python3.[0-9]|python3.[0-9][0-9]) ;; *) continue ;; esac
  bf_later=$bf_later$t$nl
done
IFS=$nl
for c in $bf_later; do
  unset IFS
  [ -f "$c" ] && [ -x "$c" ] || continue
  if why=$(bf_route "$c"); then
    printf 'A '; bf_encode "$c"; printf '\n'; exit 0
  fi
  printf 'R '; bf_encode "$c"; printf '\nW '; bf_encode "$why"; printf '\n'
done
exit 0
""".strip()


@dataclass(frozen=True)
class RootPython:
    """The interpreter root's preparation steps may start, judged before any
    of them runs. ``path`` is None when no candidate passed;
    ``rejected`` lists the present candidates that did not, with why."""

    path: str | None
    rejected: tuple[tuple[str, str], ...] = ()
    # The sandbox holds PT's kit without a Python: root runs no image Python
    # at all, whatever V3's judge would accept.
    kit_without_python: bool = False

    def missing_text(self, image: str) -> str:
        """The reason no interpreter is accepted, naming *image*."""
        if self.kit_without_python:
            return (
                "BenchFlow's protected runtime in this sandbox holds no Python "
                f"(the image {image} had no root-owned python3 in /usr/bin, /bin "
                "or /usr/local/bin), and root runs no image Python after the "
                "agent's turn"
            )
        if not self.rejected:
            return f"the image {image} has no python3 on the safe or image PATH"
        why = "; ".join(f"{path}: {reason}" for path, reason in self.rejected)
        return (
            f"no python3 in the image {image} passed the trusted-route check "
            f"({why}). Install a root-owned python3 outside the workspace, in "
            "/usr/bin or /usr/local/bin"
        )


def _task_image(task: Any) -> str:
    """The task's image name for a message, or a neutral stand-in."""
    environment = getattr(getattr(task, "config", None), "environment", None)
    image = getattr(environment, "docker_image", None)
    return image if isinstance(image, str) and image else "of this task"


def _root_python_candidates(raw_path: str) -> list[str]:
    """``python3`` in each safe folder, then in each absolute image PATH entry."""
    folders: list[str] = [*_SAFE_VERIFIER_PATH_PARTS]
    folders += [p for p in raw_path.split(":") if p.startswith("/") and "\0" not in p]
    seen: list[str] = []
    for folder in folders:
        candidate = folder.rstrip("/") + "/python3"
        if "\n" not in candidate and candidate not in seen:
            seen.append(candidate)
    return seen


async def _judge_root_python(
    env: Any,
    blocked: tuple[str, ...] | PathExclusions,
    sandbox_user: str | None,
) -> RootPython:
    """Pick root's interpreter for the preparation steps.

    Safe folders first, then the image's PATH (read as data). A candidate is
    accepted only if every hop of its route passes V3's identity rule
    (``bf_route``: root's, closed to the agent user, no ACL, outside every
    excluded tree), checked with the pre-agent inspector copies when the
    exclusions carry them. It runs before the first root Python step, so no
    interpreter runs as root before it is judged (V3-INTEG-4).

    PT's kit comes first: when the sandbox's protected kit
    holds a Python, every root Python step runs it (``root_python_sh`` and
    the ``_hardening_python`` seam), so the image's ``python3`` is neither
    judged nor able to refuse grading.
    """
    import inspect

    from benchflow._runtime_tree import RuntimeKit, runtime_kit_python

    # The route lookup reads this generation's kit record first; a broken kit
    # raises here, as for the firewall reader.
    lookup = getattr(env, "runtime_kit_for_route", None)
    if inspect.iscoroutinefunction(lookup):
        await lookup()
    if runtime_kit_python(env) is not None:
        return RootPython(KIT_PYTHON)
    if isinstance(getattr(env, "runtime_kit", None), RuntimeKit):
        # A kit without a Python: no image interpreter is judged or run as
        # root; the steps that need one are skipped and say so.
        return RootPython(None, kit_without_python=True)
    result = await _checked_read(
        env,
        PATH_QUERY_CMD,
        "Verifier hardening failed: reading image PATH",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    raw_path = decode_path_field((result.stdout or "").removesuffix("\n"))
    candidates = "\n".join(_root_python_candidates(raw_path))
    script = (
        PATH_FIELD_ENCODE_SH
        + "\n"
        + _PATH_ROUTE_SH.replace("__BF_EXCLUSIONS__", _exclusion_predicate(blocked))
        + "\n"
        + _ROOT_PYTHON_JUDGE_SH
    )
    if isinstance(blocked, PathExclusions):
        command = _path_inspector_command(
            blocked.inspector, script, "bf-root-python", candidates, sandbox_user or ""
        )
    else:
        command = _system_probe(
            script, "bf-root-python", candidates, sandbox_user or ""
        )
    judged = await _checked_read(
        env,
        command,
        "Verifier hardening failed: judging root's python3",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    accepted: str | None = None
    rejected: list[tuple[str, str]] = []
    pending: str | None = None
    for line in (judged.stdout or "").splitlines():
        tag, _, value = line.partition(" ")
        if tag == "A":
            accepted = decode_path_field(value)
        elif tag == "R":
            pending = decode_path_field(value)
        elif tag == "W" and pending is not None:
            rejected.append((pending, decode_path_field(value)))
            pending = None
    for path, why in rejected:
        logger.warning("Root's preparation steps do not run %s: %s", path, why)
    return RootPython(accepted, tuple(rejected))


def _parse_trusted_path_output(
    stdout: str, entries: list[str]
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Read the program's ``K``/``O``/``D``+``W`` lines; only judged entries count."""
    asked = set(entries)
    kept: list[str] = []
    other: list[str] = []
    dropped: list[tuple[str, str]] = []
    pending: str | None = None
    for line in stdout.split("\n"):
        tag, _, value = line.partition(" ")
        value = decode_path_field(value)
        if tag == "W" and pending is not None:
            dropped.append((pending, value or "left out"))
            pending = None
            continue
        if pending is not None:
            dropped.append((pending, "left out"))
            pending = None
        if value not in asked:
            continue
        if tag in ("K", "O"):
            kept.append(value)
            if tag == "O":
                other.append(value)
        elif tag == "D":
            pending = value
    if pending is not None:
        dropped.append((pending, "left out"))
    return kept, other, dropped


def _extras_candidates(raw_path: str) -> list[str]:
    """The entries of *raw_path* the root-only check looks at, in order: the
    same pre-filter as ``_TRUSTED_PATH_EXTRAS_SCRIPT`` (absolute, no NUL or
    newline, not a safe-PATH folder, first occurrence only)."""
    seen: set[str] = set(_SAFE_VERIFIER_PATH_PARTS)
    out = []
    for entry in raw_path.split(":"):
        entry = entry.strip()
        if (
            not entry
            or entry in seen
            or not entry.startswith("/")
            or "\x00" in entry
            or "\n" in entry
        ):
            continue
        seen.add(entry)
        out.append(entry)
    return out


#: How many path components the kit-tools probe walks up at most.
_EXTRAS_PROBE_DEPTH = 64


def _extras_probe_cmd(entries: list[str]) -> str:
    """the facts the root-only PATH/PYTHONPATH check needs,
    read with the route's own ``realpath``, ``stat`` and ``dirname`` (C8: the
    protected kit's once the sandbox has one, else the system folders' by
    pinned PATH; never a Python), where no protected Python can run the
    check: a kit without one, or no kit and no ``python3`` on root's route.
    Exit 3: the kit has a Python (its check stands). Output, NUL
    separated: per entry ``E entry``, then ``X`` (does not resolve) or
    ``R realpath`` followed by ``R|L path stat`` triples for the resolved
    folder and each parent, then for the entry as written and each parent
    (``stat`` reports the link itself, not its target); ``?`` is an
    unreadable stat; ``END`` closes the output. ``realpath`` takes no
    option (busybox accepts none, PT-INTEG-R3B-1): the entries are absolute,
    and a missing folder's ``stat`` reads ``?`` and drops it."""
    from benchflow.sandbox._base import SYSTEM_PATH

    words = " ".join(shlex.quote(e) for e in entries)
    # $__bf_t is the tools' folder prefix: the kit's bin, or empty with PATH
    # pinned to the system folders.
    walk = (
        f"n=0; while [ $n -lt {_EXTRAS_PROBE_DEPTH} ]; do n=$((n+1)); "
        "s=$(\"${__bf_t}stat\" -c '%u %a %F' -- \"$p\" 2>/dev/null) || s='?'; "
        'printf \'%s\\0%s\\0%s\\0\' "$t" "$p" "$s"; '
        '[ "$p" = / ] && break; p=$("${__bf_t}dirname" -- "$p") || break; done; '
    )
    return (
        f"if [ -f {KIT_RECORD} ]; then [ -x {KIT_PYTHON} ] && exit 3; "
        f"__bf_t={KIT_BIN}/; else __bf_t=; PATH={SYSTEM_PATH}; export PATH; fi; "
        f"for e in {words}; do "
        "printf 'E\\0%s\\0' \"$e\"; "
        'r=$("${__bf_t}realpath" "$e" 2>/dev/null) || { printf \'X\\0\'; continue; }; '
        "printf 'R\\0%s\\0' \"$r\"; "
        f"t=R; p=$r; {walk}t=L; p=$e; {walk}"
        "done; printf 'END\\0'"
    )


def _decide_root_only_extras(
    entries: list[str], blocked: tuple[str, ...], stdout: str
) -> list[str]:
    """The rule on the kit-tools probe's facts: an entry is
    kept only when it resolves to a directory that, with every parent, is
    owned by root and not group- or world-writable, every component of the
    entry as written is owned by root (and, unless it is a link, not group-
    or world-writable), and neither form is under a blocked prefix.
    Anything unreadable, cut short or unknown is dropped (fail closed)."""

    def under(path: str, prefix: str) -> bool:
        prefix = prefix.rstrip("/")
        return path == prefix or path.startswith(prefix + "/")

    def root_only(chain: list[tuple[str, str]], first: str, written: bool) -> bool:
        if not chain or chain[0][0] != first or chain[-1][0] != "/":
            return False
        for _path, facts in chain:
            parts = facts.split(" ", 2)
            if len(parts) != 3 or parts[0] != "0":
                return False
            try:
                mode = int(parts[1], 8)
            except ValueError:
                return False
            if parts[2] == "symbolic link" and written:
                continue
            if parts[2] != "directory" or mode & 0o022:
                return False
        return True

    fields = stdout.split("\0")
    if len(fields) < 2 or fields[-2:] != ["END", ""]:
        return []
    fields = fields[:-2]
    kept: list[str] = []
    i = 0
    while i < len(fields):
        if fields[i] != "E" or i + 1 >= len(fields):
            return kept  # malformed: keep nothing more
        entry = fields[i + 1]
        i += 2
        if i < len(fields) and fields[i] == "X":
            i += 1
            continue
        if i + 1 >= len(fields) or fields[i] != "R":
            return kept
        real = fields[i + 1]
        i += 2
        chains: dict[str, list[tuple[str, str]]] = {"R": [], "L": []}
        while i + 2 < len(fields) and fields[i] in ("R", "L"):
            chains[fields[i]].append((fields[i + 1], fields[i + 2]))
            i += 3
        if (
            entry in entries
            and real.startswith("/")
            and not any(under(p, b) for p in (entry, real) for b in blocked)
            and root_only(chains["R"], real, written=False)
            and root_only(chains["L"], entry, written=True)
        ):
            kept.append(entry)
    return kept


async def _extras_without_kit_python(
    env, raw_path: str, blocked: tuple[str, ...]
) -> list[str] | None:
    """The trusted extras of *raw_path* where no protected Python can run the
    check: a kit without a Python, or no kit and no
    ``python3`` on root's route (the Python check exited 127). None when the
    kit has a Python: the Python check's own result stands."""
    entries = _extras_candidates(raw_path)
    if not entries:
        return []
    result = await _read_exec(
        env,
        _extras_probe_cmd(entries),
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    if result.return_code == 3:
        return None
    if result.return_code != 0:
        return []
    return _decide_root_only_extras(entries, blocked, result.stdout or "")


def _discover_pytest_plugins_cmd(
    blocked_prefixes: tuple[str, ...],
    pythonpath: str | None = None,
    *,
    ownership: bool = True,
    verifier_path: str | None = None,
    agent_gids: tuple[int, ...] | None = None,
) -> str:
    """Build the container-side pytest plugin discovery command.

    Runs on the isolated protected Python, from no working folder of its own
    import path (C8). The pre-agent image search paths and the verifier's
    trusted PYTHONPATH are passed to metadata discovery and PathFinder as
    data, never added to this helper's import path: that PYTHONPATH keeps the
    workspace (``_blocked_verifier_pythonpath_prefixes``), so as an import
    path it ran an agent's ``/app/json.py`` as root.
    *ownership* is the guard's rule for code outside
    the blocked prefixes (see ``_BENCHFLOW_OWNERSHIP`` in
    ``_pytest_plugin_guard.py``). *verifier_path* is the verifier's PATH: the
    search path read is that of the first Python on it with a pre-agent
    record (PT-R13-A-2), else root's python3's. *agent_gids* are the agent
    user's groups, the only ones whose write bit makes a path untrusted
    (#1176; None: every group's).
    """
    policy: dict[str, Any] = {"ownership": ownership, "pythonpath": pythonpath}
    if verifier_path is not None:
        policy["verifier_path"] = verifier_path
    if agent_gids is not None:
        policy["agent_gids"] = list(agent_gids)
    return root_python_sh(
        READ_IMAGE_PYTHON_PATHS + "\n" + _DISCOVER_PYTEST_PLUGINS_SCRIPT,
        _json.dumps(blocked_prefixes),
        "[]",
        _json.dumps(policy),
        raw_args='"${PYTHONPATH-}"' if pythonpath is None else "",
    )


# The protected kit's RECORD exists but it holds no Python (the image had none
# when the kit was built), so discovery's script cannot run (127). Its answer
# depends only on data: the verifier's trusted PYTHONPATH and the pre-agent
# record's search paths (PT-R13-A-2), which this reads with the kit's own cat,
# never running a Python. Exit 3: not this case (no kit, or a kit Python).
_KIT_WITHOUT_PYTHON_RECORD_CMD = (
    f"[ -f {KIT_RECORD} ] && [ ! -x {KIT_PYTHON} ] || exit 3; "
    'printf "%s\\0" "${PYTHONPATH-}"; '
    f"{KIT_BIN}/cat {IMAGE_PYTHON_PATHS_FILE} || exit 4; printf '\\0'; "
    f"if [ -e {IMAGE_PYTHON_PATHS_EACH_FILE} ]; then "
    f"{KIT_BIN}/cat {IMAGE_PYTHON_PATHS_EACH_FILE} || exit 4; fi"
)


async def _discover_without_kit_python(env, pythonpath: str | None) -> dict | None:
    """Discovery's answer where the protected kit holds no Python, or None
    when the kit has one or there is no kit (the caller's rules then apply).

    The discovery script searches only the verifier's trusted PYTHONPATH and
    the pre-agent record's search path, both data.
    When all of them are empty there is nothing to search: the script would
    answer no plugins and no rejections, and so does this, without any
    Python. Otherwise those folders could only be read by the Python the kit
    lacks, and this answers None, so the caller's 127 rules apply exactly as
    they did before this path existed (PT-INTEG-R3-1: raising here made an
    image whose only Python is outside the kit's candidates, conda or pyenv,
    unscored where both bases score it).
    """
    result = await _read_exec(
        env,
        _KIT_WITHOUT_PYTHON_RECORD_CMD,
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    if result.return_code == 3:
        return None
    if result.return_code != 0:
        raise ValueError(
            "the protected runtime holds no Python and its pre-agent Python "
            "record cannot be read" + _exec_failure_detail(result)
        )
    raw_pythonpath, _, rest = (result.stdout or "").partition("\0")
    record_text, sep, each_text = rest.partition("\0")
    if not sep:
        raise ValueError("incomplete pre-agent Python record")
    # The guard's own reading of the trusted PYTHONPATH (a string here).
    entries = [
        p
        for p in (raw_pythonpath if pythonpath is None else pythonpath).split(":")
        if p
    ]
    try:
        records = [_json.loads(record_text)] + [
            _json.loads(line) for line in each_text.splitlines() if line.strip()
        ]
    except ValueError as exc:
        raise ValueError(f"invalid pre-agent Python record: {exc}") from exc
    if not all(
        isinstance(r, dict) and isinstance(r.get("path", []), list) for r in records
    ):
        raise ValueError("invalid pre-agent Python record")
    folders = [*entries, *(p for r in records for p in r.get("path", []))]
    if folders:
        # Only a Python could read them: the caller's 127 rules decide, as
        # on both bases (PT-INTEG-R3-1).
        return None
    return {"plugins": [], "rejected": []}


def _plugin_flags(names: "list[str] | tuple[str, ...]") -> str:
    """``-p`` flags for *names*, in order, each once."""
    return " ".join(f"-p {shlex.quote(name)}" for name in dict.fromkeys(names))


async def _discover_pytest_plugin_flags(
    env,
    task: "Task",
    sandbox_user: str | None = None,
    workspace: str | None = None,
    pythonpath: str | None = None,
    *,
    blocked: tuple[str, ...] | None = None,
    ownership: bool = True,
    verifier_path: str | None = None,
    agent_gids: tuple[int, ...] | None = None,
) -> str:
    """Every plugin of :func:`_discover_pytest_plugins` as ``-p`` flags.

    For a verifier without the guard (one outside ``main``), where nothing
    else could load them.
    """
    required, optional = await _discover_pytest_plugins(
        env,
        task,
        sandbox_user,
        workspace,
        pythonpath,
        blocked=blocked,
        ownership=ownership,
        verifier_path=verifier_path,
        agent_gids=agent_gids,
    )
    return _plugin_flags([*optional, *required])


async def _discover_pytest_plugins(
    env,
    task: "Task",
    sandbox_user: str | None = None,
    workspace: str | None = None,
    pythonpath: str | None = None,
    *,
    blocked: tuple[str, ...] | None = None,
    ownership: bool = True,
    verifier_path: str | None = None,
    agent_gids: tuple[int, ...] | None = None,
) -> tuple[list[str], list[str]]:
    """Only enable plugins whose current registration and code are trusted.

    Returns ``(required, optional)``: the plugins the task declares
    (``[verifier] pytest_plugins``), which every pytest of the verifier must
    load, and those found in the image's Python or inferred from test.sh,
    which the guard loads only in a pytest that has them installed (#1175).
    Missing aliases are deferred to the protected guard in the final pytest
    interpreter, allowing trusted verifier scripts to install plugins with uvx.
    *blocked* and *ownership* default to the agent's own sandbox; a separate
    verifier sandbox passes its transferred paths and no ownership rule.
    *path* finds ``python3``: the verifier's judged PATH, never the image's.
    """
    if blocked is None:
        blocked = _blocked_verifier_path_prefixes(sandbox_user, workspace)
    declared = list(
        dict.fromkeys(
            name.strip() for name in task.config.verifier.pytest_plugins if name.strip()
        )
    )
    requested = list(
        dict.fromkeys(
            [
                *declared,
                *(
                    name.strip()
                    for name in _infer_pytest_plugins_from_test_script(task)
                    if name.strip()
                ),
            ]
        )
    )
    inferred = [name for name in requested if name not in declared]
    try:
        result = await _read_exec(
            env,
            _discover_pytest_plugins_cmd(
                blocked,
                pythonpath,
                ownership=ownership,
                verifier_path=verifier_path,
                agent_gids=agent_gids,
            ),
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
        discovered = None
        if _exec_return_code(result) == 127:
            # A protected kit without a Python (an image that had none when
            # the kit was built): the answer is decided by the pre-agent
            # record, read as data (PT-integ round 3).
            discovered = await _discover_without_kit_python(env, pythonpath)
        if discovered is None and _exec_return_code(result) == 127:
            # Minimal shell-only images need no Python. A trusted verifier may
            # also install its Python/uvx runtime later; the protected guard is
            # still installed below and validates plugins in that interpreter.
            # Do not confuse a broken interpreter/discovery with proven absence:
            # asked on the same route as discovery's own python3.
            available = await exec_raw(
                env.exec,
                "if command -v python3 >/dev/null 2>&1; then exit 0; else exit 1; fi",
                user="root",
                timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
            )
            if _exec_return_code(available) == 1:
                return declared, inferred
        if discovered is None and _exec_return_code(result) != 0:
            raise ValueError(
                "sandbox discovery command failed" + _exec_failure_detail(result)
            )
        if discovered is None:
            discovered = _json.loads(result.stdout or "")
        if not isinstance(discovered, dict) or any(
            not isinstance(discovered.get(key), list)
            or any(not isinstance(name, str) or not name for name in discovered[key])
            for key in ("plugins", "rejected")
        ):
            raise ValueError("invalid discovery response")
        plugins = discovered["plugins"]
        unsafe_requested = set(requested).intersection(discovered["rejected"])
        if unsafe_requested or set(plugins).intersection(discovered["rejected"]):
            raise ValueError(
                "untrusted, ambiguous, or unavailable requested plugins: "
                + ", ".join(sorted(unsafe_requested))
                + "; preinstall plugins in protected image paths before verification"
            )
    except Exception as exc:
        raise RuntimeError(
            "Verifier hardening failed: pytest plugin trust discovery: " + str(exc)
        ) from exc
    optional = dict.fromkeys([*plugins, *inferred])
    return declared, [name for name in optional if name not in declared]


# pytest imports the guard by module name (``-p <guard>``), so any interpreter
# that runs pytest without being able to import it aborts, exits 1, and test.sh
# records reward 0 for a correct solution. A copy therefore goes into the first
# writable site directory that each existing Python keeps on its isolated
# sys.path: every python/python3/python3.N on the verifier PATH plus each pytest
# script's shebang interpreter. That reaches ``python3 -I`` and a test.sh that
# replaces PYTHONPATH (``PYTHONPATH=/app pytest``). A Python that is not a venv
# takes its copy in its standard library directory first: every venv made from
# it later (``uv init && uv run pytest``, ``python3 -m venv``) keeps that
# directory on its sys.path, and its site-packages not, so DevEval's
# ``export PYTHONPATH=$(pwd)`` before ``uv run pytest`` still loads the guard
# (#1180). Anything that does not run as Python 3
# with ``-I`` is skipped; a Python that cannot take or resolve the copy fails
# hardening, a verifier error instead of a silent zero. Without a kit the copy
# is written by the candidate itself, from ``/``, as root, after solver
# quiescence; with one, by root in sh from the pre-agent record
# (_INSTALL_PYTEST_PLUGIN_GUARD_KIT_CMD).
#
# Only PYTHONPATH reaches a Python that did not exist on the verifier PATH at
# hardening (uvx, ``uv run``, a fresh venv, a distro package, a venv run by a
# path off PATH), so the guard directory goes there too, but only when the
# verifier may need it: when no Python took a copy, when the kit's pre-agent
# record does not vouch for one (PT-C8-CONF-2b), or when a verifier script
# names such a Python (``_verifier_may_run_uncopied_python``). Tasks read a set
# PYTHONPATH as a startup-injection hook (``os.environ.get("PYTHONPATH")`` in an
# anti-tamper preflight), so a verifier whose Pythons all hold a copy gets the
# image's trusted PYTHONPATH, as before the guard existed. A verifier that
# creates a Python in a way the scan does not recognise gets ``Error importing
# plugin "<guard>"``, which is a verifier error below, not a score.
#
# After the run the verifier turns a guard pytest could not load into a verifier
# error instead of scoring test.sh's 0. A guard pytest imported but never ran
# (pluggy refused its hooks) leaves its ``loading`` marker; that check needs no
# output. A guard pytest could not import at all is found only by pytest's
# message in a top-level verifier log (``verifier_scan._has_guard_load_failure``),
# and only when no pytest of this verification registered the guard: a solution
# the tests run can print that message, but cannot leave a ``registered``
# marker unregistered. A test.sh that discards pytest's output still scores
# that run 0. No sound
# output-independent check exists for it: the failure happens in an interpreter
# test.sh created itself (uvx, a fresh venv) run with ``-I`` or its own
# PYTHONPATH, where every file that executes (Python, its site-packages, pytest)
# was written after hardening and ``-I`` ignores every PYTHON* hook; pytest reads
# nothing before importing ``-p`` plugins except the ``-c`` file, whose access
# time is not updated on noatime mounts and which as a FIFO would hang pytest if
# its writer died. Nor can a verification with no guard marker at all count as
# a failure, because test.sh need not run pytest: a solution that makes test.sh
# stop before pytest would turn its own 0 into an unscored verifier error.
_PYTEST_PLUGIN_GUARD_PARENT = "/"
# The guard module is this prefix plus a random hex suffix chosen at hardening.
PYTEST_PLUGIN_GUARD_PREFIX = "_benchflow_guard_"
# The guard leaves ``<guard>.<token>.<kind>`` markers in ``<guard dir>/.markers``
# (mode 0700, the verifier's user's), not the world-writable /logs/verifier,
# and signs each with a key in ``<guard dir>/.key`` (mode 0400, the same
# user's). The key derives from a secret that never leaves this process, so
# the verifier recomputes it from the guard's name; it is never put in the
# environment test.sh and everything it runs inherits.
_PYTEST_PLUGIN_GUARD_MARKER_SECRET = secrets.token_bytes(32)


def _redraw_marker_secret() -> None:
    global _PYTEST_PLUGIN_GUARD_MARKER_SECRET
    _PYTEST_PLUGIN_GUARD_MARKER_SECRET = secrets.token_bytes(32)


# A forked process draws its own, as a fresh one does: the local runtime's
# CLI workers are forks of one pre-imported parent, and must not share it.
os.register_at_fork(after_in_child=_redraw_marker_secret)
PYTEST_PLUGIN_GUARD_MARKER_KINDS = (
    "loading",
    "registered",
    "crashed",
    "installed",
    "unlisted",
    "unproven",
    "unrecorded",
    "refused",
    "checking",
)
# At most this many verdict markers (every kind but ``registered``), each at
# most this big, are read back. More, or a bigger one, makes the read
# incomplete, and an incomplete read is never taken as "no marker"
# .
_GUARD_MARKER_MAX_COUNT = 256
# Every pytest process the guard loads in leaves a ``registered`` marker
# (pytest-xdist workers included) and only whether there is one matters, so
# the reader prints this many and skips the rest without calling the read
# incomplete (V2-R1-A-1).
_GUARD_REGISTERED_READ = 8
# The reader's own lines, which no marker name can be.
_GUARD_MARKERS_END = "!end"
_GUARD_MARKERS_PROBLEMS = {
    "!missing": "the marker directory is gone",
    "!truncated": "more markers than the verifier reads",
    "!oversize": "a marker bigger than the verifier reads",
}
_GUARD_MARKER_MAX_BYTES = 64 * 1024


def pytest_plugin_guard_key(guard: str) -> str:
    """The key the guard *guard* signs its markers with (hex)."""
    return hmac.new(
        _PYTEST_PLUGIN_GUARD_MARKER_SECRET, guard.encode(), hashlib.sha256
    ).hexdigest()


def pytest_plugin_guard_markers_dir(guard: str) -> str:
    """Where the guard *guard* leaves its markers in the sandbox."""
    return os.path.join(_PYTEST_PLUGIN_GUARD_PARENT, guard, ".markers")


def pytest_plugin_guard_key_file(guard: str) -> str:
    """Where the guard *guard* reads its marker key in the sandbox."""
    return os.path.join(_PYTEST_PLUGIN_GUARD_PARENT, guard, ".key")


def pytest_plugin_guard_record_file(guard: str) -> str:
    """Where hardening records the agent's trees for the guard *guard*."""
    return os.path.join(_PYTEST_PLUGIN_GUARD_PARENT, guard, ".record")


# What ``$BENCHFLOW_PYTEST`` names when no protected runner was prepared (the
# task's tests folder and ``[verifier] env`` never name it, so only a file in
# its image can call it): a root-owned stub in the guard's directory that
# prints why and exits 4, so no script falls back to the image's own pytest
# (REL r4 H1). It signals nothing to the verifier: the agent's code runs as
# test.sh's user and can run it too, so the trial is scored on what test.sh
# reports.
RUNNER_STUB_REASON = (
    "no protected test runner was prepared, because the task's test.sh (and "
    "the rest of its tests folder) and [verifier] env never name "
    "$BENCHFLOW_PYTEST. The task must name $BENCHFLOW_PYTEST in test.sh (or "
    "a [verifier] env value) for BenchFlow to prepare it."
)


def pytest_runner_stub_file(guard: str) -> str:
    """Where the guard *guard*'s directory holds the runner stub."""
    return os.path.join(_PYTEST_PLUGIN_GUARD_PARENT, guard, "benchflow-pytest")


def _pytest_runner_stub_source() -> str:
    return (
        "#!/bin/sh\n"
        f"printf 'benchflow-pytest: error: %s\\n' {shlex.quote(RUNNER_STUB_REASON)} >&2\n"
        "exit 4\n"
    )


# The identity (``st_dev:st_ino``) of every file and directory in the agent's
# trees (argv[2], a JSON list), never following a link, then ``end``; written
# to argv[1] after the agent stopped and the workspace froze, before the
# guard. A file test.sh installs later is new to it, and nothing the agent
# left becomes new by a chmod, a chown or a touch. Where statx(2)
# tells an inode's birth time, the line ends ``:<seconds>.<nanoseconds>`` of it,
# as the guard's ``_birth`` writes it: the filesystem reuses a deleted file's
# inode number for the next new one at once, and the birth time tells the two
# apart (V2-R1-A-2). Python 3.6+: an old image interpreter may run this.
# Without python3 the shell fallback writes the lines without birth times
# with find and stat; with neither there is no record, and the guard counts
# nothing in those trees as the verifier's install.
_RECORD_AGENT_TREES_SCRIPT = """\
import json, os, stat, struct, sys
out = open(sys.argv[1], 'x')
try:
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        statx = libc.statx
    except AttributeError:
        number = {'x86_64': 332, 'aarch64': 291}.get(os.uname().machine)
        statx = None
        if number is not None:
            def statx(*args):
                return libc.syscall(ctypes.c_long(number), *args)
    buffer = ctypes.create_string_buffer(256)
except Exception:
    statx = None
def birth(dirfd, name, st):
    if statx is None:
        return ''
    try:
        if statx(ctypes.c_int(dirfd), os.fsencode(name), ctypes.c_int(0x100),
                 ctypes.c_uint(0x800), buffer) != 0:
            return ''
        raw = buffer.raw
        mask, = struct.unpack_from('=I', raw, 0)
        ino, = struct.unpack_from('=Q', raw, 32)
        seconds, nanoseconds = struct.unpack_from('=qI', raw, 80)
    except Exception:
        return ''
    if not mask & 0x800 or ino != st.st_ino or not (seconds or nanoseconds):
        return ''
    return ':%d.%09d' % (seconds, nanoseconds)
def note(name, dirfd=-100):
    if dirfd == -100:
        st = os.lstat(name)
    else:
        st = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
    out.write('%d:%d%s\\n' % (st.st_dev, st.st_ino, birth(dirfd, name, st)))
    return st
FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
def same(fd, dev, ino):
    st = os.fstat(fd)
    if (st.st_dev, st.st_ino) != (dev, ino):
        raise OSError('a directory changed while it was recorded')
def walk(root):
    # Depth first through directory descriptors and names relative to them:
    # no path handed to the kernel is longer than one name, so a chain of
    # folders deeper than PATH_MAX cannot fail the record (V2-R1-B-1), and
    # at most two descriptors are open at once, however deep it goes. An
    # entry that vanishes before it is recorded or opened (a service's file
    # in /tmp, /var/tmp or /logs) is skipped: the agent's processes are
    # gone, so it was not the agent's doing, and a file no longer there
    # cannot be a plugin (V2-R2-A-1). A directory removed while it is
    # walked still leads back up: '..' of a removed directory is its old
    # parent, which `same` checks.
    try:
        fd = os.open(root, FLAGS)
    except FileNotFoundError:
        return
    try:
        names, stack = os.listdir(fd), []
        while True:
            if names:
                name = names.pop()
                try:
                    st = note(name, fd)
                    if not stat.S_ISDIR(st.st_mode):
                        continue
                    child = os.open(name, FLAGS, dir_fd=fd)
                except (FileNotFoundError, NotADirectoryError):
                    continue
                try:
                    same(child, st.st_dev, st.st_ino)
                    entries = os.listdir(child)
                    here = os.fstat(fd)
                except BaseException:
                    os.close(child)
                    raise
                stack.append((names, here.st_dev, here.st_ino))
                os.close(fd)
                fd, names = child, entries
            elif stack:
                names, dev, ino = stack.pop()
                parent = os.open('..', FLAGS, dir_fd=fd)
                os.close(fd)
                fd = parent
                same(fd, dev, ino)
            else:
                return
    finally:
        os.close(fd)
for top in json.loads(sys.argv[2]):
    try:
        st = note(top)
    except FileNotFoundError:
        continue
    # A root that is a link (/app -> /srv/app) is recorded as the link and
    # as the directory it resolves to, which is walked; links below it are
    # recorded, never followed.
    root = top
    if stat.S_ISLNK(st.st_mode):
        root = os.path.realpath(top)
        try:
            st = note(root)
        except FileNotFoundError:
            continue
    if stat.S_ISDIR(st.st_mode):
        walk(root)
# Only a walk that read every entry ends the record: any error above leaves
# it without `end`, and the guard ignores it.
out.write('end\\n')
out.close()
"""


def _record_agent_trees_cmd(record: str, prefixes: "tuple[str, ...]") -> str:
    """The root command writing the guard's record of *prefixes* to *record*.

    A failure leaves no ``end`` line, so the guard ignores the record; it
    never fails hardening. It is one subshell, so its ``|| true`` covers
    only the record, never a command chained before it (V2-R1-A-3).
    """
    quoted = " ".join(shlex.quote(prefix) for prefix in prefixes)
    target = shlex.quote(record)
    # The walk runs on root's protected Python (C8, PY8-PT-R8-5); a sandbox
    # with neither the kit's nor a system python3 takes the find walk.
    walk = root_python_sh(
        _RECORD_AGENT_TREES_SCRIPT, record, _json.dumps(list(prefixes))
    )
    return (
        f"( (umask 077; if [ -x {KIT_PYTHON} ] || command -v python3 "
        f">/dev/null 2>&1; then {walk}; "
        f"else set -o noclobber; {{ for p in {quoted}; do "
        '[ -e "$p" ] || [ -L "$p" ] || continue; '
        "find \"$p\" -maxdepth 0 -exec stat -c '%d:%i' {} + && "
        'if [ -d "$p" ]; then find "$p"/. -exec stat -c \'%d:%i\' {} +; fi '
        "|| exit 1; "
        f"done; echo end; }} > {target}; fi) 2>/dev/null || true )"
    )


# Where no kit exists (nobody but root ran there), the installer runs this on
# each interpreter, as before: GUARD_SEARCH_RULE, then the copy.
_INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT = (
    r"""
import sys
source_path, name = sys.argv[1], sys.argv[2]
with open(source_path, 'rb') as handle:
    source = handle.read()
"""
    + IMAGE_PYTHON_GUARD_SEARCH_RULE
    + r"""
from importlib.machinery import PathFinder
# Where the copy goes: the first folder of that search path it can write, so
# the copy written is the one the finder below resolves. The standard library
# comes first, which venvs made from this Python later (`uv run`) keep on
# their path (#1180); a folder it cannot write is skipped.
directories = [p for p in search if os.path.isdir(p)]
if not directories:
    sys.exit('no site directory on the search path of ' + sys.executable)
for directory in dict.fromkeys(directories):
    target = os.path.join(directory, name + '.py')
    try:
        with open(target, 'xb') as handle:
            handle.write(source)
        os.chmod(target, 0o444)
        break
    except FileExistsError:
        with open(target, 'rb') as handle:
            if handle.read() != source:
                sys.exit('a different module occupies ' + target)
        break
    except OSError:
        continue
else:
    sys.exit('cannot write the guard to any site directory of ' + sys.executable)
# Resolve it as the tests will: this interpreter's own finder over the
# whole search path above, not just the folder written to.
spec = PathFinder.find_spec(name, search)
if spec is None or os.path.realpath(spec.origin or '') != os.path.realpath(target):
    sys.exit('the guard does not resolve to ' + target + ' on the search path of ' + exe)
"""
).strip()

_INSTALL_PYTEST_PLUGIN_GUARD_CMD_TEMPLATE = r"""
# guard_agents <python>: 0 when it, or the file it resolves to, is under a
# prefix the transfer brought into a separate verifier (none elsewhere), or
# cannot be resolved while there are such prefixes. Without a kit such an
# interpreter is never started: it is printed as unrecorded, and the
# PYTHONPATH of the verifier carries its guard (PT-C8-CONF-2b round 3).
guard_agents() {
    guard_py=$1
    set -- __BLOCKED__
    [ $# -gt 0 ] || return 1
    guard_real=$(readlink -f "$guard_py" 2>/dev/null) || return 0
    [ -n "$guard_real" ] || return 0
    for guard_b in "$@"; do
        case "$guard_py" in "$guard_b"|"$guard_b"/*) return 0 ;; esac
        case "$guard_real" in "$guard_b"|"$guard_b"/*) return 0 ;; esac
    done
    return 1
}
install_guard() {
    if guard_agents "$1"; then
        printf 'unrecorded %s\n' "$1"
        printf 'benchflow: the pytest plugin guard reaches %s through PYTHONPATH only: it is among the files the agent could write, which hardening never starts\n' "$1"
        return 0
    fi
    "$1" -I -S -B -c 'import sys; sys.exit(sys.version_info[0] != 3)' \
        </dev/null >/dev/null 2>&1 || return 0
    if "$1" -I -S -B -c __SCRIPT__ __GUARD__ __NAME__ </dev/null >/dev/null; then
        printf 'guarded %s\n' "$1"
        return 0
    fi
    echo "Cannot install the pytest plugin guard for $1" >&2
    return 1
}
shebang_interpreter() {
    line=
    IFS= read -r line 2>/dev/null <"$1"
    case "$line" in '#!/'*) ;; *) return 0 ;; esac
    set -f
    set -- ${line#??}
    set +f
    case "${1##*/}" in python*) echo "$1" ;; esac
}
if [ -f __KIT_RECORD__ ]; then
__KIT_INSTALL__
fi
cd / || exit 1
verifier_path=__PATH__
IFS=:
set -f
set -- $verifier_path
set +f
unset IFS
for dir in "$@"; do
    case "$dir" in /*) ;; *) continue ;; esac
    for py in "$dir"/python "$dir"/python3 "$dir"/python3.[0-9] "$dir"/python3.[0-9][0-9]; do
        if [ -f "$py" ] && [ -x "$py" ]; then install_guard "$py" || exit 1; fi
    done
    for script in "$dir"/pytest "$dir"/py.test; do
        [ -f "$script" ] || continue
        py=$(shebang_interpreter "$script")
        if [ -n "$py" ] && [ -f "$py" ] && [ -x "$py" ]; then
            install_guard "$py" || exit 1
        fi
    done
done
""".strip()

# With a kit (PT-C8-CONF-2b): after the agent's turn hardening starts no
# image Python, as any uid, and the record taken before the agent's turn
# (IMAGE_PYTHON_GUARD: the same GUARD_SEARCH_RULE under the same -I -S -B)
# is the only source for where an interpreter looks. The copy is written and
# resolved in sh by the rules of _INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT; a
# recorded interpreter that cannot take or resolve it, or whose answer stops
# before ``end``, fails hardening, as before.
#
# An interpreter the record does not hold, or holds as another file, or
# whose id cannot be checked now (no stat), is not started and gets no
# copy: it is printed as ``unrecorded`` with a ``benchflow:`` line, and the
# verifier puts the guard folder on PYTHONPATH, which reaches it wherever
# its root run looks (CONF-2b round 2).
_INSTALL_PYTEST_PLUGIN_GUARD_KIT_CMD = r"""
guard_nl='
'
__GUARD_ID__
# guard_entry <python>: its record's py3/dir/end lines; 1 when the record
# has no entry for <python>, or that entry was another file, or the id of
# the file cannot be read now (no stat: a change could not be seen).
guard_entry() {
    [ -f __RECORD__ ] || return 1
    guard_want="$(bf_guard_id "$1")"
    case " $guard_want" in *" ?"*) return 1 ;; esac
    guard_hit=
    guard_out=
    while IFS= read -r guard_line; do
        case "$guard_line" in
        "exe "*)
            [ -z "$guard_hit" ] || break
            [ "${guard_line#exe }" != "$1" ] || guard_hit=exe
            ;;
        "id "*)
            if [ "$guard_hit" = exe ]; then
                [ "${guard_line#id }" = "$guard_want" ] || return 1
                guard_hit=id
            fi
            ;;
        *) [ "$guard_hit" != id ] || guard_out="$guard_out$guard_line$guard_nl" ;;
        esac
    done < __RECORD__
    [ "$guard_hit" = id ] || return 1
    printf '%s' "$guard_out"
}
install_guard() {
    if ! guard_got=$(guard_entry "$1"); then
        printf 'unrecorded %s\n' "$1"
        printf 'benchflow: the pytest plugin guard reaches %s through PYTHONPATH only: it is not in the pre-agent record as this file\n' "$1"
        return 0
    fi
    # Not a Python 3, or no answer (it did not start at capture): skipped,
    # as the old version check skipped it. Any other answer runs from py3 1
    # to end.
    case "$guard_got" in ""|"py3 0") return 0 ;; esac
    case "$guard_nl$guard_got" in "${guard_nl}py3 1$guard_nl"*"${guard_nl}end") ;; *)
        echo "Cannot install the pytest plugin guard for $1" >&2
        return 1
        ;;
    esac
    guard_dirs=
    IFS=$guard_nl
    set -f
    for guard_line in $guard_got; do
        case "$guard_line" in "dir "*) guard_dirs="$guard_dirs${guard_line#dir }$guard_nl" ;; esac
    done
    # The copy: the first folder of the search path it can write, one
    # holding the same bytes already included; a different module there
    # is an error.
    guard_target=
    guard_seen=$guard_nl
    guard_any=
    for guard_dir in $guard_dirs; do
        [ -d "$guard_dir" ] || continue
        case "$guard_seen" in *"$guard_nl$guard_dir$guard_nl"*) continue ;; esac
        guard_seen="$guard_seen$guard_dir$guard_nl"
        guard_any=1
        guard_file="$guard_dir/"__NAME__.py
        if [ -e "$guard_file" ] || [ -L "$guard_file" ]; then
            if [ "$(cat "$guard_file" 2>/dev/null; echo x)" = "$(cat __GUARD__; echo x)" ]; then
                guard_target=$guard_file
                break
            fi
            unset IFS
            set +f
            echo "Cannot install the pytest plugin guard for $1: a different module occupies $guard_file" >&2
            return 1
        fi
        (set -C; cat __GUARD__ > "$guard_file") 2>/dev/null || continue
        chmod 444 "$guard_file" 2>/dev/null || continue
        guard_target=$guard_file
        break
    done
    unset IFS
    set +f
    if [ -z "$guard_any" ]; then
        echo "Cannot install the pytest plugin guard for $1: no site directory on its search path" >&2
        return 1
    fi
    if [ -z "$guard_target" ]; then
        echo "Cannot install the pytest plugin guard for $1: cannot write the guard to any site directory" >&2
        return 1
    fi
    # Resolve it as the tests will, by the path finder's rules over the
    # whole search path: per folder a package (name/__init__.*), else an
    # extension module, the source, the bytecode.
    guard_want=$(realpath "$guard_target" 2>/dev/null) || guard_want=
    guard_found=
    IFS=$guard_nl
    set -f
    for guard_dir in $guard_dirs; do
        unset IFS
        set +f
        [ -d "$guard_dir" ] || continue
        if [ -d "$guard_dir/"__NAME__ ]; then
            for guard_f in "$guard_dir/"__NAME__/__init__.*; do
                if [ -f "$guard_f" ]; then guard_found=$guard_f; break; fi
            done
        fi
        if [ -z "$guard_found" ]; then
            for guard_f in "$guard_dir/"__NAME__.*.so "$guard_dir/"__NAME__.so \
                "$guard_dir/"__NAME__.py "$guard_dir/"__NAME__.pyc; do
                if [ -f "$guard_f" ]; then guard_found=$guard_f; break; fi
            done
        fi
        [ -z "$guard_found" ] || break
    done
    unset IFS
    set +f
    if [ -z "$guard_want" ] || [ -z "$guard_found" ] \
        || [ "$(realpath "$guard_found" 2>/dev/null)" != "$guard_want" ]; then
        echo "Cannot install the pytest plugin guard for $1: the guard does not resolve to $guard_target on its search path" >&2
        return 1
    fi
    printf 'guarded %s\n' "$1"
}
""".strip()


def _install_guard_kit_cmd(guard: str, name: str) -> str:
    """The kit branch of the guard installer for the guard file *guard*
    named *name* (:data:`_INSTALL_PYTEST_PLUGIN_GUARD_KIT_CMD`)."""
    return (
        _INSTALL_PYTEST_PLUGIN_GUARD_KIT_CMD.replace(
            "__GUARD_ID__", IMAGE_PYTHON_GUARD_ID_SH
        )
        .replace("__RECORD__", shlex.quote(IMAGE_PYTHON_GUARD_FILE))
        .replace("__GUARD__", shlex.quote(guard))
        .replace("__NAME__", shlex.quote(name))
    )


def _guard_blocked_words(blocked: "tuple[str, ...] | list[str]") -> str:
    """*blocked* as the no-kit installer's ``for`` words: each absolute
    prefix once, quoted, without a trailing ``/`` (``/`` itself is not a
    prefix the agent owns, and an empty word would match every path)."""
    prefixes = (p.rstrip("/") for p in blocked if p.startswith("/"))
    return " ".join(shlex.quote(p) for p in dict.fromkeys(p for p in prefixes if p))


def _runtime_guard_top(runtime: dict[str, Any] | None) -> tuple[str, ...]:
    """The top-level names the guard serves from the protected runtime: the
    whole closure (V2-R4-2: a plugin's dependency must not come from the
    workspace), but for the running pytest's own packages, which test.sh's
    pytest has already imported from where it lives."""
    from benchflow.sandbox._verifier_runtime import RUNNING_PYTEST

    names = (runtime or {}).get("protected_top") or (runtime or {}).get(
        "plugin_top", ()
    )
    return tuple(sorted(str(t) for t in names if str(t) not in RUNNING_PYTEST))


def _pytest_plugin_guard_source(
    name: str,
    blocked: tuple[str, ...],
    requested: list[str],
    markers_dir: str | None = None,
    *,
    runtime: "dict[str, Any] | None" = None,
    tests_dir: str = "",
    ownership: bool = True,
    key_file: str | None = None,
    optional: "list[str] | tuple[str, ...]" = (),
    agent_gids: tuple[int, ...] | None = None,
    workspace: str | None = None,
    snapshot: str | None = None,
    verifier_root: bool = False,
    record: str | None = None,
    named_inputs: "list[str] | tuple[str, ...] | None" = None,
) -> str:
    """Return the armed guard module *name*: its policy, then its load marker.

    *requested* plugins every pytest must load (``-p`` flags); *optional*
    ones the guard loads where they are installed (#1175). *runtime* is the
    manifest of BenchFlow's protected verifier runtime, whose
    plugins the guard registers from there; *tests_dir* the folder BenchFlow
    uploads the verifier's tests to. *ownership* False (a
    separate verifier sandbox) distrusts the *blocked* paths alone.
    *agent_gids* are the agent user's groups (#1176). A file in *workspace*
    holding the same bytes as its counterpart in *snapshot*, the root-only
    pre-agent copy, is the image's. *verifier_root* says test.sh
    runs as root, so write bits inside the agent's trees do not count against
    a file test.sh installed itself (V2-R1-A-2). *record* is the root-only
    record of the agent's trees taken before the guard; without
    it nothing there counts as test.sh's own install. Before test.sh runs a
    root-only proof freezes whether the agent left every file in its trees
    alone; *named_inputs*, the dependency files the parser finds named in the
    verifier's scripts, are checked first and can only make that proof fail
    sooner. Markers go to
    *markers_dir* and are signed with the key in *key_file* (by default the
    guard directory's ``.markers`` and ``.key``).
    """
    markers_dir = markers_dir or pytest_plugin_guard_markers_dir(name)
    key_file = key_file or pytest_plugin_guard_key_file(name)
    return (
        _DISCOVER_PYTEST_PLUGINS_SCRIPT
        + "\n_BENCHFLOW_BLOCKED = "
        + repr(blocked)
        + "\n_BENCHFLOW_REQUESTED = "
        + repr(requested)
        + "\n_BENCHFLOW_RUNTIME_SITE = "
        + repr(str(runtime["site"]) if runtime else "")
        + "\n_BENCHFLOW_RUNTIME_PLUGINS = "
        + repr({str(k): str(v) for k, v in (runtime or {}).get("plugins", {}).items()})
        + "\n_BENCHFLOW_RUNTIME_TOP = "
        + repr(_runtime_guard_top(runtime))
        + "\n_BENCHFLOW_TESTS_DIR = "
        + repr(tests_dir)
        + "\n_BENCHFLOW_OWNERSHIP = "
        + repr(bool(ownership))
        + "\n_BENCHFLOW_MARKERS = "
        + repr(os.path.join(markers_dir, name))
        + "\n_BENCHFLOW_KEY_FILE = "
        + repr(key_file)
        + "\n_BENCHFLOW_OPTIONAL = "
        + repr(list(optional))
        + "\n_BENCHFLOW_AGENT_GIDS = "
        + repr(None if agent_gids is None else tuple(agent_gids))
        + "\n_BENCHFLOW_WORKSPACE = "
        + repr(workspace if workspace and snapshot else "")
        + "\n_BENCHFLOW_SNAPSHOT = "
        + repr(snapshot if workspace and snapshot else "")
        + "\n_BENCHFLOW_VERIFIER_ROOT = "
        + repr(bool(verifier_root))
        + "\n_BENCHFLOW_RECORD = "
        + repr(record or "")
        + "\n_BENCHFLOW_NAMED_INPUTS = "
        + repr(None if named_inputs is None else tuple(named_inputs))
        + "\n_BENCHFLOW_INPUT_PROOF = "
        + repr((record + ".inputs") if record else "")
        + "\n_BENCHFLOW_ARMED = True"
        + "\nif _imported_by_pytest():\n    _mark('loading')\n"
    )


async def _install_pytest_plugin_guard(
    env,
    sandbox_user,
    workspace,
    plugin_flags,
    verifier_path=_SAFE_VERIFIER_PATH,
    *,
    runtime: "dict[str, Any] | None" = None,
    tests_dir: str = "",
    blocked: tuple[str, ...] | None = None,
    ownership: bool = True,
    verifier_uid: str | None = None,
    optional: "list[str] | tuple[str, ...]" = (),
    agent_gids: tuple[int, ...] | None = None,
    snapshot: str | None = None,
    verifier_root: bool | None = None,
    record: bool = False,
    named_inputs: "list[str] | tuple[str, ...] | None" = None,
    runner_stub: bool = False,
):
    """Create an unguessable protected bootstrap after solver quiescence.

     *runner_stub* also writes the runner stub
     (:func:`pytest_runner_stub_file`) into the guard's directory.
     Also creates the guard's marker directory and key file, handed to
     *verifier_uid* (the user test.sh runs as) when that is not root.
     *plugin_flags* are the plugins every pytest must load; *optional*,
     *agent_gids* and *snapshot* go to the guard's policy
     (:func:`_pytest_plugin_guard_source`). *verifier_root* (by default:
     *verifier_uid* is 0) is the one answer to whether test.sh runs as root
    . *record* first records the *blocked* trees for the guard
    ; hardening asks for it only when test.sh runs as root with
     the uv and pip configuration it protected.
     Returns the guard directory, the ``-p`` flags with the guard first, the
     Pythons that took a copy, and the Pythons the pre-agent record does not
     vouch for (PT-C8-CONF-2b), whose guard the verifier's PYTHONPATH carries.
    """
    name = PYTEST_PLUGIN_GUARD_PREFIX + uuid.uuid4().hex
    directory = os.path.join(_PYTEST_PLUGIN_GUARD_PARENT, name)
    guard = os.path.join(directory, name + ".py")
    if blocked is None:
        blocked = _blocked_verifier_path_prefixes(sandbox_user, workspace)
    source = _pytest_plugin_guard_source(
        name,
        blocked,
        shlex.split(plugin_flags)[1::2],
        runtime=runtime,
        tests_dir=tests_dir,
        ownership=ownership,
        optional=optional,
        agent_gids=agent_gids,
        workspace=workspace,
        snapshot=snapshot,
        verifier_root=(verifier_uid == "0" if verifier_root is None else verifier_root),
        record=pytest_plugin_guard_record_file(name) if record else None,
        named_inputs=named_inputs,
    )
    install_into_interpreters = (
        _INSTALL_PYTEST_PLUGIN_GUARD_CMD_TEMPLATE.replace(
            "__GUARD__", shlex.quote(guard)
        )
        .replace("__NAME__", shlex.quote(name))
        .replace("__PATH__", shlex.quote(verifier_path))
        .replace("__SCRIPT__", shlex.quote(_INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT))
        .replace("__KIT_RECORD__", shlex.quote(KIT_RECORD))
        # In a separate verifier only (no ownership): there the blocked
        # prefixes are what the transfer brought. A main sandbox without a
        # kit had no sandbox user: nobody but root ran there.
        .replace("__BLOCKED__", "" if ownership else _guard_blocked_words(blocked))
        .replace("__KIT_INSTALL__", _install_guard_kit_cmd(guard, name))
    )
    record_agent_trees = (
        _record_agent_trees_cmd(pytest_plugin_guard_record_file(name), blocked) + " && "
        if record
        else ""
    )
    proof_script = (
        "import runpy,sys; "
        "runpy.run_path(sys.argv[1])['_write_input_proof'](sys.argv[2])"
    )
    # The proof reads every recorded file of the agent's trees and its
    # pre-agent copy, in the exec below; it must end, tainted if need be, by
    # this many seconds after the exec started, leaving the rest of the exec's
    # timeout to the guard's install (V2-R1-A-2). The clock is the sandbox's.
    proof_budget = max(VERIFIER_SETUP_TIMEOUT_SEC - _INPUT_PROOF_RESERVE_SEC, 0)
    clock = "_bf_t0=$(date +%s 2>/dev/null || echo 0) && " if record else ""
    # On root's protected Python (C8, PY8-PT-R8-5).
    proof = root_python_sh(
        proof_script,
        guard,
        raw_args=f'"$((_bf_t0 + {proof_budget} - $(date +%s)))"',
    )
    input_proof = f"( (umask 077; {proof}) 2>/dev/null || true ) && " if record else ""
    markers = shlex.quote(pytest_plugin_guard_markers_dir(name))
    key_file = shlex.quote(pytest_plugin_guard_key_file(name))
    handover = (
        f"chown {shlex.quote(verifier_uid)} {markers} {key_file} && "
        if verifier_uid not in (None, "0")
        else ""
    )
    stub = shlex.quote(pytest_runner_stub_file(name))
    install_stub = (
        f"printf %s {shlex.quote(_pytest_runner_stub_source())} > {stub} && "
        f"chmod 555 {stub} && "
        if runner_stub
        else ""
    )
    result = await _checked_read(
        env,
        f"mkdir -m 755 {shlex.quote(directory)} && {clock}"
        f"{record_agent_trees}"
        f"printf %s {shlex.quote(source)} > {shlex.quote(guard)} && "
        f"chmod 444 {shlex.quote(guard)} && {install_stub}"
        f"{input_proof}"
        f"mkdir -m 700 {markers} && "
        f"(umask 077 && printf %s {pytest_plugin_guard_key(name)} > {key_file}) && "
        f"chmod 400 {key_file} && {handover}"
        f"{{\n{install_into_interpreters}\n}}",
        "Verifier hardening failed: installing protected pytest plugin guard",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    stdout = getattr(result, "stdout", "")
    lines = (stdout if isinstance(stdout, str) else "").splitlines()
    guarded = tuple(
        line.removeprefix("guarded ") for line in lines if line.startswith("guarded ")
    )
    unrecorded = tuple(
        line.removeprefix("unrecorded ")
        for line in lines
        if line.startswith("unrecorded ")
    )
    why: dict[str, str] = {}
    for line in lines:
        if line.startswith("benchflow: "):
            # An interpreter the guard reaches through PYTHONPATH only. The
            # verifier error that comes of it names it with this reason
            # (verifier_core), so the line is no console line (dogfood p1-5).
            logger.debug(line)
            named = _GUARD_NOTE_RE.match(line)
            if named:
                why[named.group(1)] = named.group(2)
    with contextlib.suppress(AttributeError):
        env._benchflow_guard_why = why
    return directory, f"-p {name} {plugin_flags}".strip(), guarded, unrecorded


# The install script's line for an interpreter that took no guard copy.
_GUARD_NOTE_RE = re.compile(
    r"benchflow: the pytest plugin guard reaches (\S+) through PYTHONPATH only: (.+)"
)


# Verifier scripts naming a Python that has no guard copy: one test.sh creates
# or installs after hardening, as shell commands or as a Python argv list.
# Matched outside comments, which often mention these tools in passing.
_UNCOPIED_PYTHON_RE = re.compile(
    r"""
    \buvx\b
    | \buv\W+(?:run|tool|venv|sync|python)\b
    | -m\W+(?:venv|virtualenv)\b
    | (?:^|[;&|(`])\s*virtualenv\b | \bvirtualenv\s+[-/.$~"'] | ["']virtualenv["']
    | \b(?:conda|mamba|micromamba|pixi|pipx|poetry|pdm|hatch|rye|pyenv)
      \W+(?:run|exec|create|install|shell|activate|env)\b
    | (?:^|[;&|(`])\s*(?:tox|nox)\b(?!\.) | ["'](?:tox|nox)["']
    | /bin/activate\b
    | \b(?:apt-get|apt|apk|yum|dnf|microdnf|zypper)\b[^\n]*
      \b(?:install|add)\b[^\n]*\bpy(?:thon|3-)
    """,
    re.VERBOSE | re.MULTILINE,
)
# A Python or pytest executable named by its path in a ``bin`` directory; a
# bare name resolves on the verifier PATH, whose Pythons hold a copy.
_PYTHON_BY_PATH_RE = re.compile(
    r"(?<![\w.$@{}~+/-])(?P<directory>(?:[\w.$@{}~+-]*/)*bin)"
    r"/(?:python(?:\d+(?:\.\d+)*)?|pytest|py\.test)(?=$|[\s\"'`;|&)<>])",
    re.MULTILINE,
)
# A shell PATH assignment; a directory it adds may hold a Python without a copy.
_PATH_ASSIGNMENT_RE = re.compile(
    r"(?<![\w$])PATH=(?P<value>\"[^\"\n]*\"|'[^'\n]*'|[^\s;&|)\"'`,]*)"
)
# A comment line (but not a shebang) or a trailing `` # comment``.
_SCRIPT_COMMENT_RE = re.compile(r"^\s*#(?!!).*$|\s#.*$", re.MULTILINE)
_VERIFIER_SCRIPT_SUFFIXES = frozenset({"", ".sh", ".bash", ".py"})
_VERIFIER_SCRIPT_MAX_BYTES = 1 << 20


def _verifier_script_sources(task: "Task") -> list[tuple[str, str]] | None:
    """Return readable verifier script text with its source suffix."""
    verifier_dir = getattr(getattr(task, "paths", None), "tests_dir", None)
    if not isinstance(verifier_dir, Path):
        return None
    texts = []
    try:
        for root, _, names in os.walk(verifier_dir):
            for name in sorted(names):
                path = Path(root, name)
                if path.suffix not in _VERIFIER_SCRIPT_SUFFIXES or not path.is_file():
                    continue
                if path.stat().st_size > _VERIFIER_SCRIPT_MAX_BYTES:
                    continue
                data = path.read_bytes()
                if b"\0" not in data:
                    texts.append((data.decode("utf-8", "replace"), path.suffix))
    except OSError:
        return None
    return texts or None


def _verifier_script_texts(task: "Task") -> list[str] | None:
    """Return the text of every script in the task's verifier dir, if readable."""
    sources = _verifier_script_sources(task)
    return [text for text, _ in sources] if sources is not None else None


# Dependency files a verifier script names: requirements and constraints
# files (by name, or after -r/-c and their long forms), pylock files and
# setup files. The plugin guard counts only these, besides the files uv's
# project discovery always reads, as the inputs of test.sh's install
# .
_NAMED_DEPENDENCY_FILE_RE = re.compile(
    r"""
    (?P<name>[\w.+/$@{}~-]*requirements[\w.+-]*\.(?:txt|in)
      | [\w.+/$@{}~-]*pylock(?:\.[\w.-]+)?\.toml
      | [\w.+/$@{}~-]*setup\.(?:py|cfg))\b
    """,
    re.VERBOSE | re.MULTILINE,
)
# What may stand right before an absolute path a script names: anything else
# (``$(dirname "$0")/requirements.txt``) leaves its directory unknown.
_PATH_BOUNDARY = frozenset(" \t\n=:,([{;|&<>")
# A quote before an absolute path counts only when it opens the word, that
# is when a boundary or the start of the text stands before it: the closing
# quote of ``"$APP"/requirements.txt`` ends a variable, so the directory is
# unknown (V2-R2-B-1).
_PATH_QUOTES = frozenset("\"'`")


def _opens_path(text: str, start: int) -> bool:
    """Whether the absolute path at *start* of *text* begins its shell word."""
    if start <= 0:
        return True
    before = text[start - 1]
    if before in _PATH_QUOTES:
        return start == 1 or text[start - 2] in _PATH_BOUNDARY
    return before in _PATH_BOUNDARY


# A shell word, including concatenated quoted and escaped pieces. This is
# used only inside an already separated logical header, never across a
# heredoc body. Delimiters need shell quote removal, including ANSI-C quotes.
_HEREDOC_WORD_RE = re.compile(
    r"""(?:\$'(?:[^'\\]|\\[\s\S])*'|[^\s;&|()<>'"\\]|\\[\s\S]|'[^']*'|"(?:[^"\\]|\\[\s\S])*")+"""
)


def _shell_quote_remove(word: str, *, literal: bool = False) -> str:
    """Remove shell quotes without evaluating verifier text.

    Heredoc delimiters do not expand. For argv/redirection operands, reject
    words needing expansion instead of treating their syntax as a value.
    ANSI-C and locale quotes are word pieces, including when concatenated.
    """
    parts = []
    index = 0
    escapes = dict(zip("abefnrtvE", "\a\b\x1b\f\n\r\t\v\x1b", strict=True))
    while index < len(word):
        ansi = word.startswith("$'", index)
        locale = word.startswith('$"', index)
        char = word[index]
        if ansi or locale or char in "'\"":
            quote = "'" if ansi else '"' if locale else char
            index += 2 if ansi or locale else 1
            while index < len(word) and word[index] != quote:
                char = word[index]
                if literal and quote == '"' and char in "$`":
                    raise ValueError("word needs expansion")
                if char == "\\" and (ansi or quote == '"'):
                    index += 1
                    if index >= len(word):
                        raise ValueError("incomplete escape")
                    char = word[index]
                    if ansi:
                        numeric = re.match(
                            r"[0-7]{1,3}|x[0-9a-fA-F]{1,2}|u[0-9a-fA-F]{1,4}|U[0-9a-fA-F]{1,8}",
                            word[index:],
                        )
                        if numeric:
                            value = numeric.group()
                            parts.append(
                                chr(
                                    int(value[1:], 16)
                                    if value[0] in "xuU"
                                    else int(value, 8)
                                )
                            )
                            index += len(value)
                            continue
                        if char in "xuU":
                            raise ValueError("incomplete numeric escape")
                        if char == "c":
                            index += 1
                            if index >= len(word):
                                raise ValueError("incomplete control escape")
                            char = chr(ord(word[index].upper()) ^ 64)
                        elif char in escapes:
                            char = escapes[char]
                        elif char not in "\\'\"?":
                            char = "\\" + char
                    elif char not in '$`"\\\n':
                        char = "\\" + char
                parts.append(char)
                index += 1
            if index >= len(word):
                raise ValueError("unclosed quote")
        elif char == "\\":
            index += 1
            if index >= len(word):
                raise ValueError("incomplete escape")
            parts.append(word[index])
        else:
            if literal and char in "$`":
                raise ValueError("word needs expansion")
            parts.append(char)
        index += 1
    result = "".join(parts)
    if "\0" in result:
        raise ValueError("NUL delimiter")
    return result


def _shell_descriptor_path(path: str) -> int | None:
    """The fixed shell/Linux aliases that refer to an already open descriptor."""
    # Collapse only lexical no-ops. '..' may cross a symlink and must never
    # manufacture authority; no filesystem lookup runs on verifier text.
    if (
        not path.startswith("/")
        or ".." in path.split("/")
        or path.endswith(("/", "/."))
    ):
        return None
    path = posixpath.normpath("/" + path.lstrip("/"))
    standard = {"/dev/stdin": 0, "/dev/stdout": 1, "/dev/stderr": 2}
    if path in standard:
        return standard[path]
    match = re.fullmatch(r"/(?:dev/fd|proc/(?:self|thread-self)/fd)/([0-9]+)", path)
    return int(match.group(1)) if match else None


_SHELL_COMMAND_PREFIXES = frozenset(
    {"if", "then", "elif", "else", "do", "while", "until", "!", "{", "}"}
)


def _heredoc_is_shell_script(header: str, target: int) -> bool:
    """Grant shell semantics only to the descriptor a known shell reads as code.

    Keep argv separate from ordered redirections. Wrappers can select the
    executable, but -c consumes its argument, never the inherited heredoc.
    Unknown syntax/option semantics cannot grant literal path precision.
    """
    tokens = []
    index = 0
    while index < len(header):
        if header[index].isspace():
            index += 1
            continue
        word = _HEREDOC_WORD_RE.match(header, index)
        if word:
            tokens.append((word.group(), index, word.end()))
            index = word.end()
        else:
            operator = re.match(r"<<<|<<-?|[<>]&|>>|<>|[;&|()<>]", header[index:])
            if operator is None:
                return False
            end = index + len(operator.group())
            tokens.append((operator.group(), index, end))
            index = end
    # Only the simple command owning this redirection can consume it.
    start = 0
    end = len(tokens)
    for i, (word, left, _) in enumerate(tokens):
        if word in (";", "&", "|", "(", ")"):
            if left < target:
                start = i + 1
            else:
                end = i
                break
    tokens = tokens[start:end]
    argv = []
    descriptors: dict[int, int | None] = {}
    index = 0
    while index < len(tokens):
        word, left, right = tokens[index]
        fd = None
        if (
            word.isdecimal()
            and index + 1 < len(tokens)
            and right == tokens[index + 1][1]
            and tokens[index + 1][0].startswith(("<", ">"))
        ):
            fd = int(word)
            index += 1
            word, left, right = tokens[index]
        if word.startswith(("<", ">")):
            if index + 1 >= len(tokens):
                return False
            operand = tokens[index + 1][0]
            fd = fd if fd is not None else (0 if word.startswith("<") else 1)
            if word in ("<<", "<<-"):
                descriptors[fd] = left
            else:
                try:
                    operand = _shell_quote_remove(operand, literal=True)
                except ValueError:
                    return False
                source = None
                move = False
                if word in ("<&", ">&"):
                    duplicate = re.fullmatch(r"([0-9]+)(-?)", operand)
                    if duplicate:
                        source = int(duplicate.group(1))
                        move = bool(duplicate.group(2))
                elif word in ("<", "<>"):
                    source = _shell_descriptor_path(operand)
                # Bind by value at this point in the redirection sequence:
                # later overwrites/closures cannot change an existing copy.
                descriptors[fd] = (
                    descriptors.get(source) if source is not None else None
                )
                if move and source is not None:
                    descriptors[source] = None
            index += 2
        else:
            if not argv and word in _SHELL_COMMAND_PREFIXES:
                index += 1
                continue
            try:
                value = _shell_quote_remove(word, literal=True)
            except ValueError:
                return False
            argv.append(value)
            index += 1
    while argv:
        executable = argv.pop(0)
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", executable, re.DOTALL):
            continue
        executable = executable.rsplit("/", 1)[-1]
        if executable not in {"env", "command", "exec"}:
            break
        while argv and argv[0].startswith("-"):
            option = argv.pop(0)
            if option == "--":
                break
            if executable == "env" and option in {"-u", "--unset", "-C", "--chdir"}:
                if not argv:
                    return False
                argv.pop(0)
            elif not (
                (
                    executable == "env"
                    and (
                        option in {"-i", "-", "--ignore-environment"}
                        or option.startswith(("--unset=", "--chdir="))
                    )
                )
                or (executable == "command" and option == "-p")
                or (executable == "exec" and option in {"-c", "-l"})
            ):
                return False
    else:
        return False
    if executable not in {"sh", "bash", "dash", "zsh", "ksh", "ash"}:
        return False
    stdin = False
    while argv and argv[0].startswith(("-", "+")):
        option = argv.pop(0)
        if option in {"--", "-"}:
            break
        if option in {"-o", "+o", "-O", "+O"}:
            if not argv:
                return False
            argv.pop(0)
        elif option in {"--noprofile", "--norc", "--posix"}:
            continue
        elif re.fullmatch(r"[-+][a-zA-Z]+", option):
            if "c" in option:
                return False
            if "s" in option:
                stdin = option.startswith("-")
        else:
            return False
    script_fd = 0
    if argv and not stdin:
        descriptor = _shell_descriptor_path(argv[0])
        if descriptor is None:
            return False
        script_fd = descriptor
    return descriptors.get(script_fd) == target


def _shell_header(text: str, start: int) -> tuple[str, int]:
    """Read one logical shell header, without ever reading its heredoc body."""
    result = []
    quote = ""
    ansi = False
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\" and (quote != "'" or ansi) and index + 1 < len(text):
            if text[index + 1] == "\n":
                index += 2
                continue
            if text[index + 1 : index + 3] == "\r\n":
                index += 3
                continue
            result.append(text[index : index + 2])
            index += 2
            continue
        if (
            char == "#"
            and not quote
            and (not result or result[-1][-1] in " \t\r\n;&|()<>")
        ):
            # A comment cannot open a redirection or continue the command.
            end = text.find("\n", index)
            return "".join(result) + "\n", len(text) if end < 0 else end + 1
        if char == quote:
            quote = ""
            ansi = False
        elif char in ("'", '"') and not quote:
            quote = char
            ansi = char == "'" and index > start and text[index - 1] == "$"
        result.append(char)
        index += 1
        if char == "\n" and not quote:
            break
    return "".join(result), index


def _shell_heredoc_bodies(text: str, depth: int) -> tuple[str, list[tuple[str, bool]]]:
    """Separate headers from FIFO heredoc bodies before parsing either.

    Only headers receive shell continuation/comment processing. Bodies
    retain their bytes and use the Python/conservative scanner unless the
    command explicitly names a shell, whose body is recursively separated.
    A body never acquires shell path precision from a Python fragment.
    An unsupported header or missing delimiter leaves the remaining text
    conservative too (V2-R6-CONF-1/2), rather than guessing a boundary.
    """
    shell = []
    bodies = []
    position = 0
    while position < len(text):
        start = position
        header, position = _shell_header(text, position)
        pending = []
        index = 0
        while index < len(header):
            if header.startswith("<<<", index):
                index += 3  # A here-string has no following body.
                continue
            if header.startswith("<<", index):
                left = index
                index += 2
                tabs = header[index : index + 1] == "-"
                index += int(tabs)
                while index < len(header) and header[index] in " \t":
                    index += 1
                word = _HEREDOC_WORD_RE.match(header, index)
                if word is None:
                    return "".join(shell), [*bodies, (text[start:], True)]
                try:
                    delimiter = _shell_quote_remove(word.group())
                except ValueError:
                    return "".join(shell), [*bodies, (text[start:], True)]
                index = word.end()
                pending.append((left, index, delimiter, tabs))
                continue
            word = _HEREDOC_WORD_RE.match(header, index)
            if word is not None:
                index = word.end()
            elif header[index] in "'\"\\":
                return "".join(shell), [*bodies, (text[start:], True)]
            else:
                index += 1
        for left, _, delimiter, tabs in pending:
            body = []
            while position < len(text):
                end = text.find("\n", position)
                end = len(text) if end < 0 else end + 1
                line = text[position:end]
                position = end
                if tabs:
                    line = line.lstrip("\t")
                if line.rstrip("\r\n") == delimiter:
                    break
                body.append(line)
            else:
                return "".join(shell), [*bodies, (text[start:], True)]
            source = "".join(body)
            if _heredoc_is_shell_script(header, left):
                bodies.extend(_dependency_script_bodies(source, ".sh", depth + 1))
            else:
                bodies.append((source, True))
        for left, right, _, _ in reversed(pending):
            header = header[:left] + " " * (right - left) + header[right:]
        shell.append(header)
    return "".join(shell), bodies


def _shell_continuations(text: str) -> str:
    """Remove shell escaped newlines outside comments and single quotes."""
    result = []
    quote = ""
    comment = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and not comment and quote != "'" and index + 1 < len(text):
            if text[index + 1] == "\n":
                index += 2
                continue
            if text[index + 1 : index + 3] == "\r\n":
                index += 3
                continue
            result.append(text[index : index + 2])
            index += 2
            continue
        if comment:
            if char == "\n":
                comment = False
        elif char == "#" and not quote:
            comment = True
        elif char == quote:
            quote = ""
        elif char in ("'", '"') and not quote:
            quote = char
        result.append(char)
        index += 1
    return "".join(result)


def _dependency_script_bodies(
    text: str, suffix: str, depth: int = 0
) -> list[tuple[str, bool]]:
    """Separate shell text from Python files, heredocs and -c payloads.

    Parse Python as Python even when its containing file is test.sh. Keep
    the shell portions separate so an ordinary absolute shell argument
    retains its directory (V2-R3-CONF-2 and V2-R3-CONF-3).
    """
    if (
        suffix == ".py"
        or (text.startswith("#!") and "python" in text.splitlines()[0])
        or depth >= 16
    ):
        return [(text, True)]
    text, bodies = _shell_heredoc_bodies(text, depth)
    # Shell punctuation separates commands even without whitespace. Track
    # shlex's one-character lookahead when masking a payload, so the next
    # command's first character is never consumed (V2-R4-CONF-2).
    try:
        stream = io.StringIO(text)
        lexer = shlex.shlex(stream, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lookahead = getattr(lexer, "_pushback_chars", None)
        if lookahead is None:
            return [*bodies, (text, True)]
        words = []
        spans = []
        while True:
            start = stream.tell() - len(lookahead)
            word = lexer.get_token()
            if word is None:
                break
            end = stream.tell() - len(lookahead)
            with contextlib.suppress(ValueError):
                word = _shell_quote_remove(text[start:end].strip(), literal=True)
            words.append(word)
            spans.append((start, end))
    except ValueError:
        # An unparsed Python payload must not gain absolute-path precision
        # from the shell fallback's view of a string fragment.
        return [*bodies, (text, True)]
    payloads = []
    for i, word in enumerate(words):
        executable = word.rsplit("/", 1)[-1]
        python = bool(re.fullmatch(r"python[\d.]*", executable))
        shell = bool(re.fullmatch(r"(?:(?:ba|da|z|k|c|tc|fi|a)?sh|su)", executable))
        if (
            not (python or shell)
            or word.startswith("-")
            or (i > 0 and words[i - 1] in ("--python", "-p"))
        ):
            continue
        if python:
            invocation = python_invocation(words[i + 1 :])
            if invocation is not None and invocation.kind == "c":
                bodies.append((invocation.value, True))
                payloads.append(spans[i + 1 + invocation.index])
            continue
        index = i + 1
        while index < len(words):
            option = words[index]
            # These interpreter options consume the next word; its contents
            # cannot be a command option. Stop at a script or module operand.
            takes_value = {"-o", "-O"}
            if option in takes_value:
                index += 2
                continue
            if option in ("--", "-m") or not option.startswith("-"):
                break
            if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", option):
                # Shells accept combined flags such as -ec; they carry a
                # separate command payload instead of reading a script.
                payload_index = index + 1
                if payload_index < len(words):
                    payload = words[payload_index]
                    bodies.extend(_dependency_script_bodies(payload, ".sh", depth + 1))
                    payloads.append(spans[payload_index])
                break
            index += 1
    for start, end in reversed(payloads):
        # shlex can include the separator newline in a word's source span.
        # Keep command boundaries when removing the language payload.
        text = text[:start] + re.sub(r"[^\r\n]", " ", text[start:end]) + text[end:]
    bodies.append((text, False))
    return bodies


def _python_dependency_commands(
    text: str, *, conservative: bool = False
) -> tuple[list[str], str]:
    """Render Python argv lists without treating expression fragments as paths.

    Only a complete literal argument of an installer argv has a known
    directory. Fragments of concatenations, joins and formatting keep only
    their basename. The remaining source is still scanned conservatively
    for dependency names; no verifier code is evaluated.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        # A failed heredoc boundary can leave shell and Python interleaved.
        # Tokenize each line independently so one malformed header cannot
        # hide a complete argv list/tuple. Parse balanced Python fragments
        # with the same AST rules, but never grant them absolute precision:
        # the surrounding shell/Python boundary is still unknown.
        tokens = []
        for line in text.splitlines(keepends=True):
            try:
                for token in tokenize.generate_tokens(io.StringIO(line).readline):
                    if token.type in {
                        tokenize.STRING,
                        tokenize.NAME,
                        tokenize.NUMBER,
                        tokenize.OP,
                    } or tokenize.tok_name[token.type].startswith("FSTRING_"):
                        tokens.append((token.type, token.string))
            except (tokenize.TokenError, IndentationError, SyntaxError):
                pass
        commands = []
        stack = []
        for index, (kind, value) in enumerate(tokens):
            if kind != tokenize.OP:
                continue
            if value in {"[", "(", "{"}:
                stack.append((value, index))
            elif value in {"]", ")", "}"} and stack:
                opening, start = stack.pop()
                if opening + value not in {"[]", "()", "{}"}:
                    stack.clear()
                    continue
                recovered_source = tokenize.untokenize(tokens[start : index + 1])
                try:
                    ast.parse(recovered_source)
                except (SyntaxError, ValueError):
                    continue
                recovered, _ = _python_dependency_commands(
                    recovered_source, conservative=True
                )
                commands.extend(recovered)
        return commands, text
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    # Resolve only imports whose name is never rebound in this source.
    # Looking across all scopes is deliberately conservative: a shadowed
    # helper must not grant protected-path precision (V2-R4-CONF-3).
    imports: dict[str, set[str]] = {"subprocess": {"subprocess"}}
    rebound: set[str] = set()
    wildcard_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                imports.setdefault(name, set()).add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    wildcard_import = True
                    continue
                name = alias.asname or alias.name
                module = node.module if not node.level else ""
                imports.setdefault(name, set()).add(f"{module}.{alias.name}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            rebound.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                rebound.add(root.id)
        elif isinstance(node, ast.arg):
            rebound.add(node.arg)
        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.ExceptHandler,
                ast.MatchAs,
                ast.MatchStar,
            ),
        ):
            if node.name:
                rebound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            rebound.add(node.rest)

    def imported(node: ast.AST) -> str:
        if wildcard_import:
            return ""
        if isinstance(node, ast.Name):
            values = imports.get(node.id, set())
            return (
                next(iter(values))
                if len(values) == 1 and node.id not in rebound
                else ""
            )
        if isinstance(node, ast.Attribute):
            value = imported(node.value)
            return f"{value}.{node.attr}" if value else ""
        return ""

    def mapping_fragment(node: ast.AST, key: str) -> str:
        if isinstance(node, ast.Dict):
            for name, value in reversed(list(zip(node.keys, node.values, strict=True))):
                if name is None:
                    return "$UNKNOWN"
                if isinstance(name, ast.Constant) and name.value == key:
                    return fragment(value)
        return "$UNKNOWN"

    def fragment(node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return fragment(node.left) + fragment(node.right)
        if isinstance(node, ast.JoinedStr):
            return "".join(fragment(value) for value in node.values)
        # Preserve the template's literal pieces, including attached option
        # prefixes and basenames supplied in substitutions. Recover syntax,
        # never evaluate expressions or field access; the caller structurally
        # withholds directory precision from every expression. No installer
        # flag or dependency filename is special to formatting recovery.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            arguments = (
                node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
            )
            argument_index = 0

            def percent_field(match):
                nonlocal argument_index
                if match.group() == "%%":
                    return "%"
                key = match.group("key")
                if key is not None:
                    value = mapping_fragment(node.right, key)
                else:
                    argument_index += int(match.group("width") == "*") + int(
                        match.group("precision") == "*"
                    )
                    value = (
                        fragment(arguments[argument_index])
                        if argument_index < len(arguments)
                        else "$UNKNOWN"
                    )
                    argument_index += 1
                return value if match.group("conversion") == "s" else "$UNKNOWN"

            return re.sub(
                r"%(?:\((?P<key>[^)]+)\))?[#0\- +]*(?P<width>\*|\d+)?"
                r"(?:\.(?P<precision>\*|\d+))?[hlL]?(?P<conversion>[diouxXeEfFgGcrsa%])",
                percent_field,
                fragment(node.left),
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"format", "format_map"}
        ):
            with contextlib.suppress(ValueError):
                pieces = []
                automatic_index = 0
                for literal, field, _, conversion in string.Formatter().parse(
                    fragment(node.func.value)
                ):
                    pieces.append(literal)
                    if field is None:
                        continue
                    value = "$UNKNOWN"
                    # Plain positional/named fields can carry static path
                    # fragments. Attributes and subscripts stay opaque.
                    if node.func.attr == "format_map":
                        if node.args:
                            value = mapping_fragment(node.args[0], field)
                    elif field == "" or field.isdecimal():
                        index = int(field) if field else automatic_index
                        automatic_index += int(field == "")
                        if index < len(node.args) and not any(
                            isinstance(argument, ast.Starred) for argument in node.args
                        ):
                            value = fragment(node.args[index])
                    elif field.isidentifier():
                        for keyword in node.keywords:
                            if keyword.arg == field:
                                value = fragment(keyword.value)
                    pieces.append(value if conversion in {None, "s"} else "$UNKNOWN")
                return "".join(pieces)
        fragments = [
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and child.value
        ]
        return (
            "$UNKNOWN/" + fragments[-1].rsplit("/", 1)[-1] if fragments else "$UNKNOWN"
        )

    commands = []
    literal_spans = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        words = []
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                words.append(element.value)
            elif imported(element) == "sys.executable":
                words.append("python3")
            else:
                words.append(fragment(element))
        # A missing binding (including a recovered fragment without imports)
        # cannot exclude a possible module installer or grant path precision.
        unknown_executable = (
            bool(node.elts)
            and not isinstance(node.elts[0], ast.Constant)
            and imported(node.elts[0]) != "sys.executable"
        )
        # A literal in an argv that is later transformed is still a path
        # fragment. Precision applies only to the complete argv supplied
        # directly to subprocess, never a list in a join/comprehension/helper.
        parent = parents.get(node)
        if isinstance(parent, ast.keyword):
            parent = parents.get(parent)
        complete = (
            not conservative
            and not unknown_executable
            and isinstance(parent, ast.Call)
            and (
                (bool(parent.args) and parent.args[0] is node)
                or any(
                    keyword.arg == "args" and keyword.value is node
                    for keyword in parent.keywords
                )
            )
            and imported(parent.func)
            in {
                "subprocess.run",
                "subprocess.call",
                "subprocess.check_call",
                "subprocess.check_output",
                "subprocess.Popen",
            }
        )
        inputs = dependency_inputs(
            words,
            unknown_python=unknown_executable,
            precise=[
                bool(complete)
                and isinstance(element, ast.Constant)
                and isinstance(element.value, str)
                for element in node.elts
            ],
        )
        if inputs is None:
            continue
        # Canonical argv transports semantic operands to the shell caller;
        # no original option spelling or expression fragment is reparsed.
        commands.extend(
            shlex.join(["pip", "install", "--requirement", value]) for value in inputs
        )
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                literal_spans.append(
                    (
                        element.lineno,
                        element.col_offset,
                        element.end_lineno,
                        element.end_col_offset,
                    )
                )
    # Remove the handled literals before the fallback scan, or that scan
    # would add their basenames again and defeat precise absolute matching.
    lines = text.splitlines(keepends=True)
    for first, start, last, end in sorted(literal_spans, reverse=True):
        if last is None or end is None:
            continue
        for index in range(first - 1, last):
            data = lines[index].encode("utf-8")
            left = start if index == first - 1 else 0
            right = end if index == last - 1 else len(data)
            lines[index] = (data[:left] + b" " * (right - left) + data[right:]).decode(
                "utf-8"
            )
    return commands, "".join(lines)


def _shell_dependency_commands(
    text: str, *, conservative: bool = False
) -> tuple[list[str], str]:
    """Parse shell argv through the same option grammar as Python argv.

    Remove handled commands before conventional-name fallback, so an attached
    option cannot also contribute a spurious basename beside its precise path.
    Shell expansion stays textual and can never grant a literal directory.
    """
    stream = io.StringIO(text)
    lexer = shlex.shlex(stream, posix=True, punctuation_chars=";&|()\n<>")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lookahead = getattr(lexer, "_pushback_chars", None)
    if lookahead is None:
        return [], text
    values = []
    spans = []
    words = []
    start = 0

    def command_argv(tokens):
        argv = []
        index = 0
        while index < len(tokens):
            word, left, right = tokens[index]
            raw = text[left:right].strip()
            if word in {"<", ">", ">>", "<>", "<&", ">&", "<<<"} and raw == word:
                # A descriptor prefix is syntax only when attached to the
                # operator; the following word is its path/fd operand.
                if (
                    argv
                    and argv[-1].isdecimal()
                    and index
                    and tokens[index - 1][2] == left
                ):
                    argv.pop()
                index += 2
                continue
            if not argv and raw == word and word in _SHELL_COMMAND_PREFIXES:
                index += 1
                continue
            with contextlib.suppress(ValueError):
                word = _shell_quote_remove(raw, literal=True)
            argv.append(word)
            index += 1
        return argv

    try:
        while True:
            left = stream.tell() - len(lookahead)
            word = lexer.get_token()
            right = stream.tell() - len(lookahead)
            if word is None or (word and all(c in ";&|()\n" for c in word)):
                inputs = dependency_inputs(command_argv(words))
                if inputs is not None:
                    values.extend(inputs)
                    spans.append((start, left))
                words = []
                start = right
                if word is None:
                    break
            else:
                words.append((word, left, right))
    except ValueError:
        if conservative:
            return [value.rstrip("/").rsplit("/", 1)[-1] for value in values], text
        # Unknown boundaries must use Python's conservative recovery, not
        # elevate any string fragment to an absolute shell path.
        commands, residual = _python_dependency_commands(text, conservative=True)
        return [
            value
            for command in commands
            for value in (dependency_inputs(shlex.split(command)) or [])
        ], residual
    for left, right in reversed(spans):
        text = text[:left] + " " * (right - left) + text[right:]
    if conservative:
        values = [value.rstrip("/").rsplit("/", 1)[-1] for value in values]
    return values, text


def _verifier_dependency_files(task: "Task") -> list[str] | None:
    """Prove install inputs using a closed grammar, else taint every agent file.

    This is the provenance boundary. The older name/fragment recovery below
    supplies diagnostic hints only: recovering a plausible path cannot prove
    that it is the path the verifier reads. Never execute verifier expressions.
    """
    task_env = getattr(
        getattr(getattr(task, "config", None), "verifier", None), "env", None
    )
    # Even HOME/XDG or interpreter settings can introduce installer inputs;
    # no task environment overlay belongs to the finite literal-only model.
    if isinstance(task_env, dict) and task_env:
        return None
    tests_dir = getattr(getattr(task, "paths", None), "tests_dir", None)
    if not isinstance(tests_dir, Path) or not tests_dir.is_dir():
        return None
    # Execution selects verifier.md before test.sh. No strategy execution
    # environment belongs to this finite script-only model; never certify an
    # unselected script in its place, including for non-script strategies.
    if os.path.lexists(tests_dir / "verifier.md"):
        return None
    sources = []

    def unreadable(error):
        raise error

    try:
        for root, directories, files in os.walk(tests_dir, onerror=unreadable):
            if any(Path(root, name).is_symlink() for name in directories):
                return None
            for name in sorted(files):
                path = Path(root, name)
                if path.suffix not in _VERIFIER_SCRIPT_SUFFIXES:
                    continue
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size > _VERIFIER_SCRIPT_MAX_BYTES
                ):
                    return None
                data = path.read_bytes()
                if b"\0" in data:
                    return None
                sources.append((data.decode("utf-8"), path.suffix))
    except (OSError, UnicodeError):
        return None
    if not sources:
        return None
    names: set[str] = set()

    def python_commands(text):
        tree = ast.parse(text)
        commands = []
        for node in tree.body:
            if isinstance(node, ast.Import) and all(
                alias.name in {"subprocess", "sys"} and alias.asname is None
                for alias in node.names
            ):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "subprocess"
                and node.value.func.attr
                in {"run", "call", "check_call", "check_output", "Popen"}
                and len(node.value.args) == 1
                and isinstance(node.value.args[0], (ast.List, ast.Tuple))
                and all(
                    keyword.arg in {"check", "text", "capture_output"}
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, bool)
                    for keyword in node.value.keywords
                )
            ):
                raise UnknownDependencyInputs("nonliteral Python command")
            argv = []
            for element in node.value.args[0].elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    argv.append(element.value)
                elif (
                    isinstance(element, ast.Attribute)
                    and isinstance(element.value, ast.Name)
                    and element.value.id == "sys"
                    and element.attr == "executable"
                ):
                    argv.append("python3")
                else:
                    raise UnknownDependencyInputs("computed Python argument")
            commands.append(argv)
        return commands

    def shell_commands(text):
        # Only simple commands separated by ;, newline, && or ||. Every word
        # undergoes literal quote removal; grouping, expansions, redirections,
        # heredocs and nested languages are unknown, rather than recovered.
        commands = []
        argv = []
        index = 0
        text = _shell_continuations(text)
        while index < len(text):
            if text[index] in " \t\r":
                index += 1
                continue
            if text[index] == "#":
                end = text.find("\n", index)
                index = len(text) if end < 0 else end
                continue
            if text[index] in ";\n" or text.startswith(("&&", "||"), index):
                if argv:
                    commands.append(argv)
                argv = []
                index += 2 if text.startswith(("&&", "||"), index) else 1
                continue
            word = _HEREDOC_WORD_RE.match(text, index)
            if word is None:
                raise UnknownDependencyInputs("shell structure")
            raw = word.group()
            # Reject even possibly active glob, brace and tilde expansion.
            # Quoted spellings can be supported later only with a proof of
            # their exact value; no recovered spelling grants authority.
            if any(char in raw for char in "*?[]{}~$`"):
                raise UnknownDependencyInputs("shell expansion")
            argv.append(_shell_quote_remove(raw, literal=True))
            index = word.end()
        if argv:
            commands.append(argv)
        return commands

    def protected_requirement(path):
        # A requirement file may itself include arbitrary local files, URLs,
        # build sources or options. Only a regular file shipped under /tests
        # or /verifier containing registry requirements has a finite model.
        parts = path.split("/")
        if len(parts) < 3 or parts[0] or parts[1] not in {"tests", "verifier"}:
            raise UnknownDependencyInputs("dependency outside verifier")
        local = tests_dir
        for part in parts[2:]:
            if part in {"", ".", ".."}:
                raise UnknownDependencyInputs("noncanonical dependency path")
            local = local / part
            if local.is_symlink():
                raise UnknownDependencyInputs("linked dependency")
        if not local.is_file() or local.stat().st_size > _VERIFIER_SCRIPT_MAX_BYTES:
            raise UnknownDependencyInputs("large dependency")
        for line in local.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not registry_requirement(line):
                raise UnknownDependencyInputs("transitive dependency input")

    try:
        for text, suffix in sources:
            python = suffix == ".py" or (
                text.startswith("#!") and "python" in text.splitlines()[0]
            )
            commands = python_commands(text) if python else shell_commands(text)
            for argv in commands:
                if not argv:
                    continue
                executable = argv[0].rsplit("/", 1)[-1]
                if executable in {"echo", "printf", "true", "false", "exit"}:
                    continue
                if (
                    executable == "cd"
                    and len(argv) == 2
                    and argv[1].startswith("/")
                    and posixpath.normpath(argv[1]) == argv[1]
                ):
                    # Carry this selected cwd to the runtime config check;
                    # it may be outside the agent-writable workspace trees.
                    names.update(
                        posixpath.join(argv[1], name)
                        for name in ("uv.toml", "pyproject.toml")
                    )
                    continue
                if (
                    executable == "uv"
                    and argv[1:2] == ["venv"]
                    and all(
                        re.fullmatch(r"[A-Za-z0-9_./-]+", value)
                        and not value.startswith("-")
                        for value in argv[2:]
                    )
                ):
                    continue
                if re.fullmatch(r"python[\d.]*", executable):
                    invocation = python_invocation(argv[1:])
                    if (
                        invocation
                        and invocation.kind == "m"
                        and invocation.value == "pytest"
                    ):
                        continue
                if executable in {"pytest", "py.test"}:
                    continue
                if executable not in {
                    "uv",
                    "uv.__main__",
                    "pip-sync",
                    "pip-compile",
                } and not re.fullmatch(r"(?:python|pip)[\d.]*", executable):
                    raise UnknownDependencyInputs("unmodeled executable")
                inputs = dependency_inputs(argv, strict=True)
                if inputs is None:
                    raise UnknownDependencyInputs("unmodeled command")
                for path in inputs:
                    protected_requirement(path)
                    names.add(path)
        return sorted(names)
    except (OSError, UnicodeError, ValueError, SyntaxError):
        return None


def _verifier_named_dependency_files(task: "Task") -> list[str] | None:
    """Best-effort dependency hints; never evidence for install provenance.

    An absolute name whose directory is known (``/tests/requirements.txt``)
    is kept whole, so only that file counts; any other (relative, or behind
    a variable or ``~``) by its base name, so a file of that name in any
    directory above an environment counts (V2-R1-A-4). None when the
    scripts cannot be read, or a dependency option's file name itself is a
    variable (``-r "$REQS"``): the guard then counts every requirements and
    pylock file beside an environment as an input.
    """
    sources = _verifier_script_sources(task)
    if sources is None:
        return None
    names: set[str] = set()
    bodies = [
        body
        for text, suffix in sources
        for body in _dependency_script_bodies(text, suffix)
    ]
    for text, python_source in bodies:
        if python_source:
            commands, text = _python_dependency_commands(text)
            bodies.extend((command, False) for command in commands)
        values = []
        if not python_source:
            values, text = _shell_dependency_commands(text)
        else:
            # Recovery may contain shell text after an unknown heredoc
            # boundary. Parse each line with the same option grammar, with
            # absolute precision structurally disabled.
            for line in text.splitlines():
                recovered, _ = _shell_dependency_commands(line, conservative=True)
                values.extend(recovered)
        for value in values:
            value = value.rstrip("/")
            name = value.rsplit("/", 1)[-1]
            if "$" in name:
                return None
            if name:
                known = (
                    value.startswith("/")
                    and not any(c in value for c in "$~{}")
                    and ".." not in value.split("/")
                )
                names.add(os.path.normpath(value) if known else name)
        text = _SCRIPT_COMMENT_RE.sub("", text)
        for match in _NAMED_DEPENDENCY_FILE_RE.finditer(text):
            value = match.group("name")
            if not value:
                continue
            value = value.rstrip("/")
            name = value.rsplit("/", 1)[-1]
            if "$" in name:
                # A file named through a variable could be any of them.
                return None
            if not name:
                continue
            start = match.start("name")
            known = (
                not python_source
                and value.startswith("/")
                and not any(c in value for c in "$~{}")
                and ".." not in value.split("/")
                and (start < 0 or _opens_path(text, start))
            )
            names.add(os.path.normpath(value) if known else name)
    return sorted(names)


def _verifier_dependency_hints(task: "Task") -> list[str] | None:
    """The absolute dependency files either parser finds in the verifier's scripts.

    Hints only: the guard checks these first when it
    freezes its proof that the agent left its files alone, so a changed one
    fails the proof without the full walk. Nothing here can make the proof
    hold, and no parser decides what the guard trusts.
    """
    names = {
        name
        for found in (
            _verifier_dependency_files(task),
            _verifier_named_dependency_files(task),
        )
        for name in found or ()
        if name.startswith("/")
    }
    return sorted(names) or None


def _verifier_may_run_uncopied_python(task: "Task", verifier_path: str) -> bool:
    """Whether the verifier may run pytest in a Python without a guard copy.

    True when a verifier script creates or installs a Python (uvx, ``uv run``,
    a venv, conda, a distro package), names one by a path outside the verifier
    PATH or puts such a directory on PATH, and when the scripts cannot be read.
    """
    texts = _verifier_script_texts(task)
    if texts is None:
        return True
    on_path = {entry.rstrip("/") for entry in verifier_path.split(":") if entry}

    def off_path(directory: str) -> bool:
        directory = directory.rstrip("/")
        return not directory.startswith("/") or directory not in on_path

    for text in texts:
        text = _SCRIPT_COMMENT_RE.sub("", text)
        if _UNCOPIED_PYTHON_RE.search(text):
            return True
        if any(
            off_path(match.group("directory"))
            for match in _PYTHON_BY_PATH_RE.finditer(text)
        ):
            return True
        for match in _PATH_ASSIGNMENT_RE.finditer(text):
            entries = match.group("value").strip("\"'").split(":")
            if any(
                entry and entry not in ("$PATH", "${PATH}") and off_path(entry)
                for entry in entries
            ):
                return True
    return False


def pytest_plugin_guard_name(pytest_addopts: str | None) -> str | None:
    """Return the guard module that hardening put in ``PYTEST_ADDOPTS``, if any."""
    try:
        args = shlex.split(pytest_addopts or "")
    except ValueError:
        return None
    for flag, name in itertools.pairwise(args):
        if flag == "-p" and name.startswith(PYTEST_PLUGIN_GUARD_PREFIX):
            return name
    return None


@dataclass(frozen=True)
class GuardMarker:
    """A marker the plugin guard signed: its file name, kind and detail.

    Kinds are named in ``_pytest_plugin_guard.py``: ``loading`` means pytest
    imported the guard but never registered it; ``registered`` that a pytest
    registered it; ``crashed`` holds the traceback of a guard hook that failed
    other than by refusing a plugin; ``installed`` lists the files (and why
    each is untrusted) of a refused plugin the verifier installed itself after
    the agent stopped; ``unrecorded`` says, on its first line, why there was
    no record to tell who put a refused plugin's files in the agent's trees,
    then lists them as ``installed`` does (V2-R2-A-1); ``unproven`` lists, as
    ``installed`` does, the files of a refused plugin test.sh installed after
    the agent stopped from inputs the agent changed, and ``unlisted`` those
    of a vouched release the published table lacks (V2-R1-B-1, V2-R1-B-2).
    """

    name: str
    kind: str
    detail: str


def pytest_plugin_guard_markers_cmd(guard: str) -> str:
    """The sandbox command that prints the guard's markers, one per line.

    Each line is the marker's file name, a tab, and the file's content (one
    line when the guard wrote it). Only regular files are read, at most
    ``_GUARD_MARKER_MAX_BYTES`` of each, so what else lands there cannot hang
    or flood the verifier. The verdict kinds come first, at most
    ``_GUARD_MARKER_MAX_COUNT`` of them. Then come at most
    ``_GUARD_REGISTERED_READ`` ``registered`` markers, of which one is enough,
    so a test.sh that runs pytest hundreds of times (or ``pytest -n auto`` on
    a big host) is still read whole (V2-R1-A-1). The last line is ``!end``,
    after a ``!truncated``, ``!oversize`` or ``!missing`` line when the read
    is incomplete (:func:`pytest_plugin_guard_markers_problem`).
    """
    directory = shlex.quote(pytest_plugin_guard_markers_dir(guard))
    verdicts = " ".join(
        f"./*.{kind}"
        for kind in PYTEST_PLUGIN_GUARD_MARKER_KINDS
        if kind != "registered"
    )
    regular = '[ -f "$f" ] && [ ! -h "$f" ] || continue; '
    oversize = f'[ "$(wc -c < "$f")" -gt {_GUARD_MARKER_MAX_BYTES} ]'
    show = (
        'printf "%s\t" "${f#./}"; '
        f'dd if="$f" bs={_GUARD_MARKER_MAX_BYTES} count=1 2>/dev/null; echo; '
    )
    return (
        f"cd {directory} 2>/dev/null || {{ echo '!missing'; echo '!end'; exit 0; }}; "
        f"n=0; for f in {verdicts}; do {regular}"
        f"n=$((n + 1)); if [ $n -gt {_GUARD_MARKER_MAX_COUNT} ]; then "
        "echo '!truncated'; break; fi; "
        f"if {oversize}; then echo '!oversize'; continue; fi; "
        f"{show}done; "
        f"n=0; for f in ./*.registered; do {regular}"
        f"if {oversize}; then continue; fi; "
        f"n=$((n + 1)); if [ $n -gt {_GUARD_REGISTERED_READ} ]; then break; fi; "
        f"{show}done; echo '!end'"
    )


def pytest_plugin_guard_markers_problem(output: str) -> str | None:
    """Why *output* of the marker reader is not the whole set of markers, or None."""
    lines = output.splitlines()
    if not lines or lines[-1] != _GUARD_MARKERS_END:
        return "the read did not finish"
    for line in lines:
        if line in _GUARD_MARKERS_PROBLEMS:
            return _GUARD_MARKERS_PROBLEMS[line]
    return None


def parse_pytest_plugin_guard_markers(output: str, guard: str) -> list[GuardMarker]:
    """The markers in *output* (see above) that the guard *guard* really signed.

    A line whose name, signature or body does not check out is dropped: a
    solution that writes a marker file cannot make its run unscored.
    """
    key = pytest_plugin_guard_key(guard).encode()
    name_re = re.compile(
        re.escape(guard)
        + r"\.[0-9]+-[0-9a-f]{8}\.("
        + "|".join(PYTEST_PLUGIN_GUARD_MARKER_KINDS)
        + ")"
    )
    markers = []
    for line in output.splitlines():
        name, _, rest = line.partition("\t")
        mac, _, body = rest.partition("\t")
        kind = name_re.fullmatch(name)
        if kind is None or not mac:
            continue
        expected = hmac.new(key, f"{name}\n{body}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, mac):
            logger.warning(
                "Ignoring a plugin guard marker with a bad signature: %s", name
            )
            continue
        try:
            detail = _json.loads(body)
        except ValueError:
            continue
        if isinstance(detail, str):
            markers.append(GuardMarker(name, kind.group(1), detail))
    return markers


def describe_installed_marker(text: str) -> str:
    """Summarise an ``installed`` marker: its first refused file, why, and the count."""
    entries = [line.split("\t", 1) for line in text.splitlines() if line.strip()]
    if not entries:
        return "files the guard could not trust"
    path, *why = entries[0]
    first = f"{path} ({why[0]})" if why and why[0] else path
    more = len(entries) - 1
    return first + (f" and {more} more file{'s' if more > 1 else ''}" if more else "")


_DIST_INFO_RE = re.compile(
    r"/([A-Za-z0-9][A-Za-z0-9_.]*?)-([0-9][^/-]*)\.dist-info(?:/|$)"
)
_SITE_PACKAGES_RE = re.compile(r"^(.*?)/lib(?:64)?/python[0-9.]+/site-packages(?:/|$)")


def installed_marker_plugins(text: str) -> tuple[list[str], list[str]]:
    """What an ``installed`` marker says in words a task author acts on.

    Returns the distributions the refused files belong to (``pytest-json-ctrf
    0.3.5``, from their ``.dist-info``) and the Python environments they were
    installed into (``/root/.venv``).
    """
    plugins: list[str] = []
    environments: list[str] = []
    for line in text.splitlines():
        path = line.partition("\t")[0]
        if (dist := _DIST_INFO_RE.search(path)) is not None:
            name = dist.group(1).replace("_", "-").lower()
            plugins.append(f"{name} {dist.group(2)}")
        if (site := _SITE_PACKAGES_RE.match(path)) is not None:
            environments.append(site.group(1) or "/")
    return list(dict.fromkeys(plugins)), list(dict.fromkeys(environments))


def _infer_pytest_plugins_from_test_script(task: "Task") -> list[str]:
    """Infer safe pytest plugins required by common task test.sh patterns."""
    task_dir = getattr(task, "task_dir", None)
    if not task_dir:
        return []
    verifier_dir = "verifier" if _uses_native_verifier_dir(task) else "tests"
    test_sh = Path(task_dir) / verifier_dir / "test.sh"
    try:
        text = test_sh.read_text()
    except OSError:
        return []

    uncommented = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    plugins: list[str] = []
    if re.search(r"(^|[^\w-])--ctrf([=\s]|$)", uncommented):
        plugins.append("ctrf")
    return plugins


_FEDORA_LIKE = ("fedora", "rhel", "centos", "rocky", "alma")


async def _distro_pip_env(env) -> dict[str, str]:
    """Distro-conditional pip env to neutralize Fedora's user-install fallback.

    Fedora's downstream pip patch routes root pip-installs to ~/.local/lib
    even with PIP_USER=0 + PIP_BREAK_SYSTEM_PACKAGES=1. PYTHONNOUSERSITE=1 then
    hides those installs from python3 at import time. Pinning PIP_PREFIX on
    Fedora-likes only writes them to /usr/local where python3 can find them.

    Setting PIP_PREFIX on Debian/Ubuntu would double-prefix (their downstream
    pip already injects --prefix=/usr/local for root), creating
    /usr/local/usr/local/bin/pytest. So this is conditional on the image distro.
    """
    answered = _verifier_read(env, "os_release")
    if answered is not None:
        text = answered.output.lower()
    else:
        try:
            result = await _read_exec(
                env,
                _on_path(_OS_RELEASE_CMD),
                user="root",
                timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
            )
        except Exception as e:
            logger.warning("distro detection failed (%s); skipping pip env tweaks", e)
            return {}
        text = (result.stdout or "").lower()
    ids: list[str] = []
    for line in text.splitlines():
        if line.startswith("id=") or line.startswith("id_like="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            ids.extend(value.split())
    if any(d in ids for d in _FEDORA_LIKE):
        return {"PIP_PREFIX": "/usr/local"}
    return {}


# Where the verifier's uv and pip state moves to, see
# ``_verifier_tool_state.py``: a new root-owned directory named with this prefix
# and a random suffix, created after the agent stopped, which the plugin guard
# trusts by path. It sits at ``/`` for the reason the guard does: every other
# place may be inside some task's workspace.
_VERIFIER_TOOL_STATE_SCRIPT = Path(_verifier_tool_state.__file__).read_text()
_VERIFIER_TOOL_STATE_PARENT = "/"
VERIFIER_TOOL_STATE_PREFIX = "_benchflow_verifier_"


# The verifier's environment reads (PY8 U12): each reader's own command, run
# as root with the verifier setup budget. _build_verifier_env runs all three in
# one exec first (the last only says whether python3 exists, which
# _python3_absent reads only after the tool-state probe exited 127). Root's
# plumbing pins PATH, so the image's PATH is the one it keeps as data, read
# with a builtin (C8). The image's PYTHONPATH is not among them: plumbing
# never has it, and _trusted_verifier_pythonpath reads it on the task route.
# Read with builtins and byte-encoded (V3: decode_path_field), so the trusted
# PATH program sees the entries exactly.
_IMAGE_PATH_CMD = PATH_QUERY_CMD
_PRINTENV_PATH_CMD = _IMAGE_PATH_CMD
# The image's PYTHONPATH, read with builtins on the task route (not batched:
# plumbing never has it), encoded the same way.
_PRINTENV_PYTHONPATH_CMD = PYTHONPATH_QUERY_CMD
_OS_RELEASE_CMD = "cat /etc/os-release 2>/dev/null || true"
_PYTHON3_PRESENT_CMD = (
    "if command -v python3 >/dev/null 2>&1; then exit 0; else exit 1; fi"
)
_VERIFIER_READS = (
    ("path", _IMAGE_PATH_CMD),
    ("os_release", _OS_RELEASE_CMD),
    ("python3", _PYTHON3_PRESENT_CMD),
)
# (sandbox, answers by name) while _build_verifier_env runs.
_verifier_reads: contextvars.ContextVar[tuple[Any, dict[str, StepAnswer]] | None] = (
    contextvars.ContextVar("benchflow_verifier_reads", default=None)
)


async def _probe_verifier_reads(env: Any) -> dict[str, StepAnswer]:
    """The four reads in one exec, by name; empty when it failed or its
    answer is malformed (each reader then runs its own exec, which raises or
    falls back as it always did). All four are read-only, so reading again
    after a failed probe changes nothing."""
    tag = new_tag()
    try:
        result = await _read_exec(
            env,
            merged_script(
                [
                    command if name in ("path", "pythonpath") else _on_path(command)
                    for name, command in _VERIFIER_READS
                ],
                tag,
            ),
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
    except Exception as exc:
        logger.debug("Verifier environment probe failed: %s", exc)
        return {}
    answers = parse_merged(getattr(result, "stdout", None), tag, len(_VERIFIER_READS))
    return {
        name: answers[index]
        for index, (name, _) in enumerate(_VERIFIER_READS)
        if index in answers
    }


def _verifier_read(env: Any, name: str) -> StepAnswer | None:
    probe = _verifier_reads.get()
    if probe is None or probe[0] is not env:
        return None
    return probe[1].get(name)


async def _verifier_uid(env: Any, user: str | int | None) -> str | None:
    """The uid test.sh runs as (``None`` means the image's user), if it can be told.

    For the image's user, the start probe's ``id -u`` answers when it ran in
    this sandbox generation as the same default user with the same
    environment (PY8 U12). A user given by name or number (root and 0
    included) runs its own ``id -u`` as itself.
    """
    if user is None:
        kept = kept_default_identity(env)
        if kept is not None and kept.uid is not None:
            uid = kept.uid.strip()
            return uid if uid.isdigit() else None
    try:
        result = await _read_exec(
            env,
            _on_path(UID_COMMAND, user=user),
            user=user,
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
    except Exception as exc:
        logger.debug("Cannot tell the verifier's uid: %s", exc)
        return None
    uid = getattr(result, "stdout", None)
    if _exec_return_code(result) != 0 or not isinstance(uid, str):
        return None
    uid = uid.strip()
    return uid if uid.isdigit() else None


async def _verifier_runtime_manifest(
    env: Any,
    task: "Task",
    sandbox_user: str | None,
    *,
    build: bool,
    service: str = "main",
) -> dict[str, Any] | None:
    """The protected verifier runtime's manifest, checked after the agent's turn.

    The tree (C8's ``VERIFIER_DIR`` in *service*) must still be root's alone
    and hold exactly the files BenchFlow put there, and the manifest is read
    from there, not from anything the host kept. A separate verifier sandbox
    (*build*) never ran the agent, so it builds the tree now, from its own
    image's Python (V2-R4-6). A task that needs the tree and has none stops
    hardening: its plugins would otherwise be refused, or loaded from where
    the agent could write.
    """
    from benchflow.sandbox._verifier_runtime import (
        runner_requested,
        runtime_check_command,
        runtime_requested,
    )

    pins, ctrf = runtime_requested(task)
    if not pins and not ctrf and not runner_requested(task):
        return None
    result = await _read_exec(
        env,
        runtime_check_command(),
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        service=service,
    )
    code = _exec_return_code(result)
    if code == 4 and build:
        return await prepare_verifier_runtime(
            env, task, sandbox_user, service=service, built_after_agent=True
        )
    if code != 0:
        raise RuntimeError(
            "Verifier setup failed: BenchFlow's protected verifier runtime "
            + (
                "was not built before the agent's turn"
                if code == 4
                else "is no longer root's alone or no longer holds the files "
                "BenchFlow put there"
            )
        )
    stdout = getattr(result, "stdout", "") or ""
    try:
        manifest = _json.loads(stdout[stdout.find("{") :] if "{" in stdout else "")
    except ValueError:
        manifest = None
    if not isinstance(manifest, dict):
        raise RuntimeError(
            "Verifier setup failed: BenchFlow's protected verifier runtime "
            "has no readable manifest"
        )
    return manifest


def verifier_runtime_wanted(task: "Task") -> bool:
    """Whether the sandbox the agent works in builds the verifier runtime
    before the agent's turn: not for a separate verifier sandbox, which
    builds its own from its own image (V2-R4-6)."""
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    return not separate_verifier_requested(getattr(task, "config", None))


async def prepare_verifier_runtime(
    env: Any,
    task: "Task",
    sandbox_user: str | None,
    *,
    service: str | None = None,
    built_after_agent: bool = False,
) -> dict[str, Any] | None:
    """Build the protected verifier runtime before the agent's turn.

    In the service whose test.sh runs the verifier (its own C8 kit, round 9:
    a target-service verifier's Python decides, never main's; V2-R4-6), and
    not here at all for a separate verifier sandbox. Its manifest is kept on
    *env* for hardening, which checks the tree again after the agent's turn;
    None when the task needs no plugin from it.
    """
    from benchflow.sandbox._verifier_runtime import (
        prepare_verifier_runtime as _prepare,
    )
    from benchflow.sandbox._verifier_runtime import runner_requested, runtime_requested

    pins, ctrf = runtime_requested(task)
    if not pins and not ctrf and not runner_requested(task):
        # Nothing to build, so no exec at all (U12's counted budget).
        env._benchflow_verifier_runtime = None
        return None
    if not built_after_agent and not verifier_runtime_wanted(task):
        env._benchflow_verifier_runtime = None
        return None
    if service is None:
        service = getattr(task.config.verifier, "service", None) or "main"
    if built_after_agent:
        # This sandbox never ran the agent. Prepare its own C8 kit before
        # probing Python; the solver's kit belongs to another sandbox.
        await prepare_runtime_tree(env, service=service)
    manifest = await _prepare(
        env, task, tests_dir=_verifier_confcutdir(task), service=service
    )
    env._benchflow_verifier_runtime = manifest
    return manifest


async def prepare_separate_verifier_runtime(
    env: Any, task: "Task", *, workspace: str
) -> None:
    """Prepare from the fresh verifier image before transferring agent bytes.

    In particular a workspace can hold Python tools. Copying them into C8's
    kit after transfer would bless the submitted tools as image state.
    Keep image helper provenance from this sandbox as well (V2-R5-REV-5).
    """
    from benchflow.sandbox._verifier_runtime import runner_requested, runtime_requested

    if getattr(env, "supports_runtime_kit", False) is not True:
        return
    # The kit, and with it the record of the image's interpreters
    # (IMAGE_PYTHON_GUARD), whether or not the task pins plugins: the guard
    # installer then starts none of the interpreters the transfer brings,
    # which the record does not hold (PT-C8-CONF-2b round 3). The transfer
    # unpacks as root, so its files would pass for the image's: an image
    # LD_PRELOAD or LD_LIBRARY_PATH naming the workspace is not admitted.
    await prepare_runtime_tree(env, transfer_roots=(workspace, "/logs/artifacts"))
    pins, ctrf = runtime_requested(task)
    if not pins and not ctrf and not runner_requested(task):
        return
    await _seed_verifier_workspace(env, workspace=workspace)
    await prepare_verifier_runtime(env, task, None, built_after_agent=True)


async def _agent_gids(env: Any, sandbox_user: str | None) -> tuple[int, ...] | None:
    """The groups of the agent's user, or None when they cannot be told (#1176).

    The plugin guard counts a group-writable path as the agent's only when
    its group is one of these. None keeps every group-writable path
    untrusted: no sandbox user (the agent ran as root), or no answer.
    """
    if not sandbox_user:
        return None
    try:
        result = await _read_exec(
            env,
            f"id -G {shlex.quote(sandbox_user)}",
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
    except Exception as exc:
        logger.debug("Cannot tell the sandbox user's groups: %s", exc)
        return None
    out = getattr(result, "stdout", None)
    if _exec_return_code(result) != 0 or not isinstance(out, str):
        return None
    words = out.split()
    if not words or not all(word.isdigit() for word in words):
        return None
    return tuple(sorted({int(word) for word in words}))


def _verifier_runs_as_root(user: str | int | None) -> bool:
    """Whether *user*, as a task spells it, is root: uid 0 whatever the group.

     ``None`` runs test.sh as the sandbox's default user, root in task images.
     Docker takes ``user[:group]``, so ``0:0`` and ``root:root`` are root too
    . Hardening asks ``id -u`` first (:func:`_verifier_is_root`);
     this is its answer when that cannot be told.
    """
    if user is None:
        return True
    return str(user).split(":", 1)[0].strip() in ("root", "0")


def _verifier_is_root(uid: str | None, user: str | int | None) -> bool:
    """The one answer to whether test.sh runs as root.

    *uid* is ``id -u`` run as the verifier's user (:func:`_verifier_uid`),
    which reads every spelling the sandbox accepts; when it cannot be told,
    the spelling decides. Both the uv and pip state isolation and the plugin
    guard's policy take this answer, so they never disagree.
    """
    if uid is not None:
        return uid == "0"
    return _verifier_runs_as_root(user)


def _warn_image_pip_config(task: "Task", files: Any) -> None:
    """Tell a task which image pip files its verifier's pip does not read."""
    if isinstance(files, list) and files:
        logger.warning(
            "Task %s: the image has pip configuration (%s); the verifier's pip "
            "reads none. %s",
            getattr(task, "name", "?"),
            ", ".join(str(f) for f in files[:5]),
            PIP_CONFIG_FIX,
        )


async def _isolate_verifier_tool_state(
    env: Any,
    task: "Task",
    verifier_env: dict[str, str],
    sandbox_user: str | None,
    workspace: str | None,
    *,
    blocked: tuple[str, ...] | None = None,
    verifier_root: bool | None = None,
    path: str | None = None,
) -> dict[str, str]:
    """Move the verifier's uv and pip state to a fresh directory.

    Returns the variables to add to the verifier environment; the directory
    is the parent of their ``UV_CACHE_DIR``. The plugin guard refused code
    test.sh installed with ``uvx`` wherever uv's default cache was not root's
    alone: in the workspace under a ``WORKDIR /root`` image (some SkillsBench
    tasks scored 0), and in ``/root/.cache/uv`` itself on a runtime whose exec
    mask is 0000 (every Terminal-Bench 2 ``--ctrf`` task unscored on
    Docker-in-Docker). Every location now moves into one directory created
    after the agent stopped, which the guard trusts by path, and uv and pip
    stop reading configuration the agent could have written.

    Only a root verifier in ``main`` gets this: hardening runs there alone
    (#248), and a non-root verifier could not write a root-owned directory. A
    probe that fails leaves the state where it was; the guard then judges it
    by ownership and mode, so this can cost a verifier error but never trust.
    *verifier_root* is hardening's one answer to whether test.sh runs as root
    (:func:`_verifier_is_root`); without it the task's spelling decides.
    *path* is the verifier's PATH, on which a missing ``python3`` is judged
    (V3).
    """
    if verifier_root is None:
        verifier_root = _verifier_runs_as_root(task.config.verifier.user)
    if task.config.verifier.service != "main" or not verifier_root:
        return {}
    if blocked is None:
        blocked = _blocked_verifier_path_prefixes(sandbox_user, workspace)
    # A task that names the verifier's HOME (tests that read the agent's
    # output through ~) keeps it, and what the agent left there.
    keep_home = "HOME" in (task.config.verifier.env or {})
    found = await _move_verifier_tool_state(
        env, task, verifier_env, blocked, path, keep_home=keep_home
    )
    if keep_home:
        for key in ("HOME", *(key for key, _ in _verifier_tool_state.XDG_HOMES)):
            found.pop(key, None)
    return found


async def _move_verifier_tool_state(
    env: Any,
    task: "Task",
    verifier_env: dict[str, str],
    blocked: tuple[str, ...],
    path: str | None,
    *,
    keep_home: bool,
) -> dict[str, str]:
    """The probe of :func:`_isolate_verifier_tool_state` and its fallbacks."""
    directory = os.path.join(
        _VERIFIER_TOOL_STATE_PARENT, VERIFIER_TOOL_STATE_PREFIX + uuid.uuid4().hex
    )
    overlay = {
        key: verifier_env[key]
        for key in _verifier_tool_state.INPUT_KEYS
        if key in verifier_env
    }
    result = await _read_exec(
        env,
        # The protected Python, isolated (C8, PY8-PT-R8-5).
        await root_python_command(
            env,
            _VERIFIER_TOOL_STATE_SCRIPT,
            _json.dumps(overlay),
            _json.dumps(blocked),
            directory,
        ),
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    return_code = _exec_return_code(result)
    if return_code == 127 and await _python3_absent(env, path):
        # A shell-only image: test.sh brings its own Python (uv downloads
        # one), so decide from the paths alone and create the directory in sh.
        found = _verifier_tool_state.overrides(
            overlay, blocked, directory, lexical=True
        )
        # Without Python the probe cannot check the image's inherited config
        # paths or their ownership. Make file-based configuration impossible,
        # including pip's global/XDG files (an empty file only skips user config).
        found["PIP_CONFIG_FILE"] = os.devnull
        found["UV_CONFIG_FILE"] = os.path.join(
            directory, _verifier_tool_state.UV_CONFIG
        )
        quoted = shlex.quote(directory)
        files = " ".join(
            shlex.quote(os.path.join(directory, name))
            for name in (
                _verifier_tool_state.UV_CONFIG,
                _verifier_tool_state.PIP_CONFIG,
            )
        )
        await _checked_exec(
            env,
            _on_path(
                f"mkdir -m 755 {quoted} && for f in {files}; do "
                ': > "$f" && chmod 644 "$f" || exit 1; done'
                + _verifier_home_sh(found.get("HOME")),
                path,
            ),
            "Verifier hardening failed: creating the verifier's uv and pip state",
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
        return found
    try:
        if return_code != 0:
            raise ValueError(
                f"probe exited with rc={return_code}{_exec_failure_detail(result)}"
            )
        stdout = (getattr(result, "stdout", "") or "").strip()
        found = _json.loads(stdout) if stdout else {}
        if isinstance(found, dict):
            _warn_image_pip_config(
                task, found.pop(_verifier_tool_state.PIP_FILES_KEY, None)
            )
        if (
            not isinstance(found, dict)
            or any(
                key not in _verifier_tool_state.OUTPUT_KEYS
                or not isinstance(value, str)
                or not value.startswith("/")
                for key, value in found.items()
            )
            # The guard trusts this directory by path, so every location must
            # be the one asked for.
            or any(
                found.get(key) != os.path.join(directory, entry)
                for key, entry in _verifier_tool_state.MOVED
                if key != "UV_PYTHON_INSTALL_DIR" or key in found
            )
            or any(
                found[key] != expected
                for key, expected in _verifier_home_paths(directory).items()
                if key in found
            )
        ):
            raise ValueError(f"invalid probe response {stdout[:200]!r}")
    except (ValueError, _json.JSONDecodeError) as exc:
        # Nothing a user runs the task with changes this (dogfood p1-5).
        logger.debug(
            "Verifier uv/pip state stays where it was; the plugin guard refuses "
            "plugins installed there: %s",
            exc,
        )
        if keep_home:
            return {}
        return await _verifier_home_without_probe(env, overlay, blocked, path)
    return found


def _verifier_home_paths(directory: str) -> dict[str, str]:
    """Where the probe may put the verifier's HOME and XDG bases."""
    home = os.path.join(directory, _verifier_tool_state.HOME_ENTRY)
    return {
        "HOME": home,
        **{
            key: os.path.join(home, entry)
            for key, entry in _verifier_tool_state.XDG_HOMES
        },
    }


def _verifier_home_sh(home: str | None) -> str:
    """sh that makes *home* root's, mode 700, with its empty rc files."""
    if not home:
        return ""
    files = " ".join(
        shlex.quote(os.path.join(home, name))
        for name in _verifier_tool_state.HOME_FILES
    )
    return (
        f" && mkdir -m 700 {shlex.quote(home)} && for f in {files}; do "
        ': > "$f" && chmod 600 "$f" || exit 1; done'
    )


async def _verifier_home_without_probe(
    env: Any, overlay: dict[str, str], blocked: tuple[str, ...], path: str | None
) -> dict[str, str]:
    """The verifier's new HOME when the probe failed.

    uv and pip may stay where they were (the plugin guard then judges by
    ownership), but a HOME in an agent-writable folder is a way to run code as
    root while the verifier grades, so it moves whenever the paths alone show
    it is in one. A sandbox that cannot make the folder fails hardening."""
    directory = os.path.join(
        _VERIFIER_TOOL_STATE_PARENT, VERIFIER_TOOL_STATE_PREFIX + uuid.uuid4().hex
    )
    found = _verifier_tool_state.home_overrides(
        overlay, blocked, directory, lexical=True
    )
    if not found:
        return {}
    await _checked_exec(
        env,
        _on_path(
            f"mkdir -m 755 {shlex.quote(directory)}" + _verifier_home_sh(found["HOME"]),
            path,
        ),
        "Verifier hardening failed: creating the verifier's home",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    return found


async def _python3_absent(env: Any, path: str | None = None) -> bool:
    """Whether ``python3`` is provably missing on *path*, not merely broken."""
    from benchflow.sandbox._shell import SYSTEM_PATH

    # The batched answer describes only the shared system route. A judged
    # verifier PATH can select a different Python (or none) in the same sandbox.
    answered = _verifier_read(env, "python3") if path in (None, SYSTEM_PATH) else None
    if answered is not None:
        return answered.return_code == 1
    available = await exec_raw(
        env.exec,
        _on_path(
            _PYTHON3_PRESENT_CMD
            if path is None
            else (_select_verifier_python(path) + '[ -n "$bf_python" ]'),
        ),
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    return _exec_return_code(available) == 1


async def _clear_system_route(env, blocked: PathExclusions) -> PathExclusions:
    """*blocked*, marked clear when the kit's Python proves the whole system
    route passes the identity walk (PERF-SBX); unchanged otherwise, and the
    sh walks then run and decide as before."""
    from benchflow._root_python import _sandbox_kit
    from benchflow.sandbox import _system_route_walk

    kit = await _sandbox_kit(env)
    if kit is None or not kit.python or not blocked.inspector:
        return blocked
    spec = _system_route_walk.walk_spec(
        blocked.predicate_groups(),
        _SAFE_VERIFIER_PATH_PARTS,
        index=blocked.inode_index if blocked.inodes else "",
        inspector=blocked.inspector,
    )
    if spec is None:
        return blocked
    try:
        result = await _read_exec(
            env,
            await root_python_command(env, _system_route_walk.SYSTEM_ROUTE_WALK, spec),
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
    except Exception as exc:
        logger.debug("The system route walk did not run; sh decides: %s", exc)
        return blocked
    if (
        _exec_return_code(result) == 0
        and (getattr(result, "stdout", "") or "") == _system_route_walk.CLEAR + "\n"
    ):
        return dataclasses.replace(blocked, system_route_clear=True)
    return blocked


async def _judge_trusted_entries(
    env, raw: str, blocked: tuple[str, ...] | PathExclusions, agent: str | None
) -> TrustedPathReport | None:
    """Run the trusted-PATH program on *raw*; ``None`` when it could not run."""
    entries = _path_entries_to_judge(raw)
    if isinstance(blocked, PathExclusions) and not blocked.inspector:
        raise RuntimeError(
            "Verifier hardening failed: missing pre-agent PATH inspector; recreate this sandbox with a fresh baseline"
        )
    if not entries and not isinstance(blocked, PathExclusions):
        return TrustedPathReport(path=_SAFE_VERIFIER_PATH)
    if isinstance(blocked, PathExclusions):
        blocked = await _clear_system_route(env, blocked)
    result = await _read_exec(
        env,
        _trusted_path_extras_cmd(raw, blocked, agent),
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    if _exec_return_code(result) != 0:
        if isinstance(blocked, PathExclusions):
            raise RuntimeError(
                "Verifier hardening failed: captured PATH exclusions could not "
                "be enforced" + _exec_failure_detail(result)
            )
        logger.warning(
            "The verifier's PATH check could not run in the sandbox, so its "
            "PATH keeps only the system folders, without the image's %s.%s",
            ", ".join(entries),
            _exec_failure_detail(result),
        )
        return None
    output = result.stdout or ""
    kept, other, dropped = _parse_trusted_path_output(output, entries)
    replacements: dict[str, str] = {}
    omitted: list[tuple[str, str]] = []
    pending = None
    for line in output.split("\n"):
        tag, _, encoded = line.partition(" ")
        value = decode_path_field(encoded)
        if tag == "F" and value in kept:
            pending = value
        elif tag == "P" and pending is not None:
            replacements[pending] = value
            pending = None
        elif tag == "T":
            omitted.append(
                (
                    value,
                    "route contains a control or PATH separator; install the tool at a representable image path",
                )
            )
    return TrustedPathReport(
        path=_merge_trusted_verifier_path([replacements.get(p, p) for p in kept]),
        kept=tuple(kept),
        kept_other=tuple(other),
        dropped=tuple(dropped),
        omitted_tools=tuple(omitted),
    )


async def _judge_verifier_path(
    env,
    sandbox_user: str | None,
    workspace: str | None,
    *,
    blocked: tuple[str, ...] | PathExclusions | None = None,
) -> TrustedPathReport:
    """Judge the image's PATH for the verifier, without reporting it.

    The image's PATH folders are judged in the sandbox by root, in /bin/sh
    (``_TRUSTED_PATH_EXTRAS_SCRIPT``): root's folders that group and others
    cannot write are kept, and, for a non-root sandbox user, another user's
    folder that user could never have changed. Runtime locations, the
    sandbox user's own locations and relative entries stay out, each with
    its reason.
    """
    answered = _verifier_read(env, "path")
    if answered is None:
        path_result = await _checked_read(
            env,
            PATH_QUERY_CMD,
            "Verifier hardening failed: reading image PATH",
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
        encoded_path = path_result.stdout or ""
    else:
        if answered.return_code != 0:
            raise RuntimeError("Verifier hardening failed: reading image PATH")
        encoded_path = answered.output
    raw_path = decode_path_field(encoded_path.removesuffix("\n"))
    if not raw_path:
        if isinstance(blocked, PathExclusions):
            await _judge_trusted_entries(env, "", blocked, sandbox_user)
        # Set and empty, the PATH is the working folder (V3-CONFIRM-2). Where
        # root's PATH is pinned, _BENCHFLOW_IMAGE_PATH holds the image's.
        empty = await _checked_read(
            env,
            '{ [ "${_BENCHFLOW_IMAGE_PATH+set}" = set ] || [ "${PATH+set}" = set ]; }'
            " && printf set; exit 0",
            "Verifier hardening failed: reading empty image PATH",
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
        if (getattr(empty, "stdout", "") or "").strip() != "set":
            return TrustedPathReport(path=_SAFE_VERIFIER_PATH)
        return TrustedPathReport(
            path=_SAFE_VERIFIER_PATH,
            dropped=(
                (".", "an empty PATH, a relative entry naming the working folder"),
            ),
        )
    relative = tuple(
        (entry, "a relative PATH entry, which names another folder in every folder")
        for entry in _relative_path_entries(raw_path)
    )
    judged = await _judge_trusted_entries(
        env,
        raw_path,
        blocked
        if blocked is not None
        else _blocked_verifier_path_prefixes(sandbox_user, workspace),
        sandbox_user,
    )
    if judged is None:
        report = TrustedPathReport(
            path=_SAFE_VERIFIER_PATH,
            dropped=tuple(
                (entry, "the PATH check could not run in the sandbox")
                for entry in _path_entries_to_judge(raw_path)
            )
            + relative,
        )
    else:
        kept, dropped = judged.kept, list(judged.dropped)
        # A directory absent now can be created by the verifier upload or a
        # later command. Omitting it is still a PATH change needing a contract.
        accounted = set(kept) | {folder for folder, _ in dropped}
        dropped.extend(
            (entry, "not present when the verifier PATH was checked")
            for entry in _path_entries_to_judge(raw_path)
            if entry not in accounted
        )
        report = dataclasses.replace(judged, dropped=tuple(dropped) + relative)
    if report.dropped:
        logger.warning(
            "The verifier's PATH leaves out %d folder(s) of the image's PATH: %s",
            len(report.dropped),
            report.dropped_text(),
        )
    return report


async def _trusted_verifier_path(
    env,
    sandbox_user: str | None,
    workspace: str | None,
    *,
    blocked: tuple[str, ...] | None = None,
) -> str:
    """Return verifier PATH with trusted image extras preserved.

    Judged by ``_judge_verifier_path`` and reported to
    ``recording_trusted_path()``; ``_build_verifier_env`` adds the refusal
    stubs (``_install_refusal_stubs``) after it.
    """
    report = await _judge_verifier_path(env, sandbox_user, workspace, blocked=blocked)
    report_trusted_path(report)
    return report.path


# Refusal stubs. A program the image has only in a folder
# the verifier's PATH leaves out used to be merely missing: a test.sh that
# caught the failure, sent it to its own log or checked ``command -v`` first
# scored 0, and guessing from "command not found" in test-stdout.txt also
# threw away honest scores that printed those words. Instead each such name
# gets a stub at the end of the verifier's PATH, reached only when no kept
# folder has the name. The stub updates a short numeric marker and exits 127;
# a hit after the verifier ran is a verifier error naming the folder, and only
# a hit is. The stubs folder is new and root's, made after the agent stopped,
# at ``/`` for the reason the plugin guard is. Its marker folders (0733) take
# a non-root verifier's hit too; missing or invalid markers fail verification.
# Code the tests ran could change a marker,
# which turns a score into an error or an error into the score test.sh wrote,
# never into a score test.sh did not write.
#
# Every name a file can have is stubbed, hidden ones and ones starting with
# ``-`` too, except a name with a line break, which no line of the record can
# hold: that refuses hardening (V3-CONFIRM-3). Relative entries have no stubs:
# no global inventory can preserve cwd-dependent lookups or see late helpers.
# They require the task's complete required_tools contract (CONFIRM7-3/5/6).
_REFUSAL_STUBS_PARENT = "/"
REFUSAL_STUBS_PREFIX = "_benchflow_refused_"
# More names than this fails hardening (a verifier error) instead of leaving
# the rest merely missing; each name needs a script and a short marker.
_REFUSAL_STUBS_MAX = 4096
_REFUSAL_STUB = """#!/bin/sh
n=${0##*/}
# A failed or partial append invalidates the marker too. Readback validates
# every pre-created marker, so lost storage cannot masquerade as no use.
if ! printf 'used\\n' >> __HIT__ 2>/dev/null; then
  /bin/rm -f __HIT__ 2>/dev/null
  exit 125
fi
printf '%s: refused: it is only in a folder the verifier'"'"'s PATH leaves out (verifier/trusted-path.json)\\n' "$n" >&2
exit 127
"""
# $1 the stubs folder, $2 the stub, $3 the verifier's PATH, $4 the most names, then each
# folder left out. Prints NUL-framed ``S <folder index> <name>`` records.
# A name is left unstubbed only when a folder of the verifier's PATH has an
# executable file of that name, tested with ``[`` alone: ``command -v`` takes
# a name starting with ``-`` as an option.
_REFUSAL_STUBS_SCRIPT = r"""
nl='
'
kept() {
  rest=$trusted
  while :; do
    t=${rest%%:*}
    [ -n "$t" ] && [ -f "$t/$1" ] && [ -x "$t/$1" ] && return 0
    [ "$rest" = "$t" ] && return 1
    rest=${rest#*:}
  done
}
add() {
  index=$1 p=$2
  [ -f "$p" ] && [ -x "$p" ] || return 0
  b=${p##*/}
  case $b in *"$nl"*) printf '4\n' > "$d/error"; return 0 ;; esac
  kept "$b" && return 0
  if [ ! -e "$d/bin/$b" ] && [ ! -L "$d/bin/$b" ]; then
    n=$((n + 1))
    if [ "$n" -gt "$most" ]; then printf '3\n' > "$d/error"; return 0; fi
    # Fixed-length IDs keep a legal NAME_MAX basename out of evidence paths.
    # Each marker has its own writable folder: removing a failed marker is
    # permitted for non-root verifiers without exposing other marker folders.
    mkdir -m 733 "$d/hits/$n" || return 1
    printf 'ready\n' > "$d/hits/$n/state" || return 1
    chmod 666 "$d/hits/$n/state" || return 1
    printf '#!/bin/sh\n' > "$d/bin/$b" || return 1
    printf 'hit=%s/hits/%s/state\n' "$d" "$n" >> "$d/bin/$b" || return 1
    printf '%s' "$stub" >> "$d/bin/$b" || return 1
    chmod 555 "$d/bin/$b" || return 1
  fi
  printf 'S %s ' "$index"; bf_encode "$b"; printf '\000'
}
d=$1 stub=$2 trusted=$3 most=$4
shift 4
mkdir -m 755 "$d" "$d/bin" "$d/hits" || exit 1
i=0 n=0
for e in "$@"; do
  case $e in
    /*) for p in "$e"/* "$e"/.[!.]* "$e"/..?*; do add "$i" "$p" || exit 1; done ;;
    *) : ;; # The required_tools contract handles relative entries.
  esac
  i=$((i + 1))
done
if [ -e "$d/error" ]; then read -r code < "$d/error"; exit "$code"; fi
""".strip()


class _InService:
    """*env* whose every exec runs in the Compose service *service*."""

    def __init__(self, env, service: str) -> None:
        self._env = env
        self._service = service

    async def exec(self, command: str, **kwargs):
        return await exec_raw(self._env.exec, command, service=self._service, **kwargs)


async def _install_refusal_stubs(env, report: TrustedPathReport) -> TrustedPathReport:
    """Stub each program found only in *report*'s folders left out.

    Returns *report* with the stubs folder and the stubs; a failure is a
    hardening failure, never a PATH that leaves those programs merely missing.
    """
    if not report.dropped:
        return report
    folders = [folder for folder, _ in report.dropped]
    if not any(folder.startswith("/") for folder in folders):
        return report
    directory = os.path.join(
        _REFUSAL_STUBS_PARENT, REFUSAL_STUBS_PREFIX + uuid.uuid4().hex
    )
    stub = _REFUSAL_STUB.replace("__HIT__", '"$hit"')
    script = PATH_FIELD_ENCODE_SH + "\n" + _REFUSAL_STUBS_SCRIPT
    result = await _read_exec(
        env,
        _system_probe(
            script,
            "bf-refusal-stubs",
            directory,
            stub,
            report.path,
            str(_REFUSAL_STUBS_MAX),
            *folders,
        ),
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    return_code = _exec_return_code(result)
    if return_code == 3:
        raise VerifierPathEvidenceError(
            "Verifier hardening failed: the folders the verifier's PATH leaves "
            f"out hold more than {_REFUSAL_STUBS_MAX} programs: "
            + report.dropped_text()
        )
    if return_code == 4:
        raise VerifierPathEvidenceError(
            "Verifier hardening failed: a folder the verifier's PATH leaves out "
            "holds a program whose name has a newline, which the refusal record "
            "cannot name: " + report.dropped_text()
        )
    if return_code != 0:
        raise RuntimeError(
            "Verifier hardening failed: refusing the programs of the folders the "
            f"verifier's PATH leaves out, rc={return_code}"
            + _exec_failure_detail(result)
        )
    stubs: list[tuple[str, str]] = []
    # NUL frames records; CR and other whitespace are filename data.
    for line in (getattr(result, "stdout", "") or "").split("\x00"):
        tag, index, name = [*line.split(" ", 2), "", ""][:3]
        name = decode_path_field(name)
        if tag == "S" and index.isdigit() and int(index) < len(folders) and name:
            stubs.append((name, folders[int(index)]))
    return dataclasses.replace(
        report, refusals=directory, stubs=tuple(dict.fromkeys(stubs))
    )


async def _check_required_path_tools(env, report: TrustedPathReport, verifier) -> None:
    """Check a complete tool contract without advertising invented commands.

    The trusted task author declares every external PATH dependency, including
    lookups and subprocesses. Required tools must exist before verification in
    a kept directory; uploaded/generated tools must be invoked by explicit path.
    No inventory can prove future absence is harmless, even for absolute entries.
    """
    required = getattr(verifier, "required_tools", None)
    if report.dropped and not isinstance(required, list):
        raise RuntimeError(
            "Verifier hardening failed: omitted PATH entries (including relative "
            "entries) require an explicit "
            "[verifier] required_tools contract (complete PATH dependencies; [] "
            "for none). Use absolute paths for uploaded/generated helpers. "
            "Entries left out: " + report.dropped_text()
        )
    if not isinstance(required, list) or not required:
        return
    # D876a: representation omissions are per-tool notes, never a blanket
    # refusal, even when that tool was declared. Other missing tools retain
    # the contract's refusal. The task may use an absolute image path instead.
    omitted = {posixpath.basename(tool) for tool, _ in report.omitted_tools}
    required = [name for name in required if name not in omitted]
    if not required:
        return
    # File tests, not command -v: '-' names cannot become flags, builtins are
    # not external dependencies, and no refused stub counts as an available tool.
    script = r"""
trusted=$1
shift
for name do
  rest=$trusted found=no
  while :; do
    folder=${rest%%:*}
    if [ -f "$folder/$name" ] && [ -x "$folder/$name" ]; then found=yes; break; fi
    [ "$rest" = "$folder" ] && break
    rest=${rest#*:}
  done
  if [ "$found" = no ]; then bf_encode "$name"; printf '\000'; fi
done
"""
    result = await _read_exec(
        env,
        _system_probe(
            PATH_FIELD_ENCODE_SH + "\n" + script,
            "bf-required-tools",
            report.path,
            *required,
            user=verifier.user,
        ),
        user=verifier.user,
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    if _exec_return_code(result) != 0:
        raise RuntimeError(
            "Verifier hardening failed: required_tools check could not run"
        )
    missing = [
        decode_path_field(name) for name in (result.stdout or "").split("\x00") if name
    ]
    if missing:
        raise RuntimeError(
            "Verifier hardening failed: required_tools absent from the trusted PATH: "
            + repr(missing)
            + "; install them in a trusted image directory or invoke uploaded/generated "
            "helpers by absolute path. Entries left out: " + report.dropped_text()
        )


class _EarlyRead:
    """A read's outcome, kept for the step that uses it: its result, or the
    exception it raised, raised again only where that step would have read
    (PERF-G)."""

    def __init__(self, result: Any = None, error: BaseException | None = None):
        self.result = result
        self.error = error

    def get(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.result


async def _outcome(step: Callable[[], Awaitable[Any]]) -> _EarlyRead:
    """*step*'s result, or the exception it raised, kept for the place one
    after another would have raised it (PERF-G)."""
    try:
        return _EarlyRead(await step())
    except Exception as exc:
        return _EarlyRead(error=exc)


async def _read_image_pythonpath(env) -> str:
    """The image's PYTHONPATH, byte-encoded (V3), as the checked read gives it.

    Read on the task's route with builtins (C8; see _image_bash_env):
    root's plumbing never has it, and a printenv as root would start with
    the image's LD_PRELOAD.
    """
    with as_task_command(env):
        pp_result = await _checked_read(
            env,
            _PRINTENV_PYTHONPATH_CMD,
            "Verifier hardening failed: reading image PYTHONPATH",
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
    return pp_result.stdout or ""


async def _early_image_pythonpath(env) -> _EarlyRead:
    """The image's PYTHONPATH, read beside the other reads before the
    preparation: the value comes from the image's environment, which
    nothing in the sandbox changes, so it is the one the reader would get
    later. A failure is kept, not raised (PERF-G)."""
    try:
        return _EarlyRead(await _read_image_pythonpath(env))
    except Exception as exc:
        return _EarlyRead(error=exc)


async def _trusted_verifier_pythonpath(
    env,
    sandbox_user: str | None,
    *,
    image_pythonpath: _EarlyRead | None = None,
) -> str:
    """Return filtered PYTHONPATH preserving only trusted image entries.

    Root's folders that group and others cannot write, judged by the same
    program as PATH, but the workspace is not blocked: it is already
    importable via CWD/pytest and is chowned to root before verification
    (a regular file with another name keeps its owner, so such an executable
    or a link to one drops the entry).

    *image_pythonpath*: the image's value as hardening read it before the
    preparation (:func:`_early_image_pythonpath`, PERF-G); it is judged
    here, after the preparation, as a value read here would be.
    """
    answered = _verifier_read(env, "pythonpath")
    if answered is not None:
        if answered.return_code != 0:
            raise RuntimeError("Verifier hardening failed: reading image PYTHONPATH")
        encoded_pp = answered.output
    elif image_pythonpath is not None:
        encoded_pp = image_pythonpath.get()
    else:
        encoded_pp = await _read_image_pythonpath(env)
    raw_pp = decode_path_field(encoded_pp.removesuffix("\n"))
    if not raw_pp:
        return ""
    judged = await _judge_trusted_entries(
        env, raw_pp, _blocked_verifier_pythonpath_prefixes(sandbox_user), None
    )
    if judged is None:
        return ""
    return ":".join(judged.kept)


# Every non-interactive bash reads the file BASH_ENV names before it reads its
# script: the bash a ``#!/bin/bash`` test.sh starts and the bash the verifier
# starts a test.sh without a shebang under. An image or task whose BASH_ENV
# named a file the agent could write (``/app/startup.sh``, a file in the
# agent's home) let that file run as the verifier user, write a reward and
# exit before test.sh ran: a false pass, as root where the verifier is root.
# BASH_ENV is always named here, so it replaces the image's: the task's
# ``[verifier.env]`` value, or the image's own when the task names none
# (read on the task's route, see _image_bash_env), kept only when it names a root-owned
# regular file, in folders only root can write, outside every agent-writable
# tree (a conda activation script), and emptied otherwise.
# The verifier gets the path the check resolved and checked, never the value
# as written: bash resolves a symlink again from its own
# working folder, so ``/dev/shm/x -> /proc/self/cwd/etc/a.sh`` was a root
# file when checked from ``/`` and the agent's ``/app/etc/a.sh`` when test.sh
# ran in ``/app``. The resolved path has no link left in it, and every folder
# on it is root's alone, so nothing the agent can write changes what it names
# (a path under ``/proc``, ``/sys`` or ``/dev`` is refused: those name
# whatever process or device reads them). Copying the file instead would
# break an activation script that finds its neighbours by its own path.
# The check is POSIX sh with ``readlink -f`` and ``stat -c`` (coreutils or
# BusyBox), so a shell-only image keeps its activation file
# ; ``python3 -I`` answers only where those tools are
# missing, and with neither the value is emptied. It runs from ``/`` and
# fails closed: an unreadable answer, or a path bash would expand (``$``, a
# backquote), empties it.
# The inspector is copied into a fresh root-only directory before the agent
# runs. Cross-checking live utilities was too late: the first
# query already ran a candidate as root, and it could repair its permissions.
# Only the protected copies ever inspect files; live candidates never run.
# As with the verifier shell, the image's system libraries and /etc remain
# the trust anchor. Without a pre-agent inspector, activation is refused.
# The check runs in the service test.sh runs in: a path
# names a different file in another container (a shared volume mounted at
# another place), so main's answer is never carried to it.
# ``ENV`` is emptied too: only an interactive shell reads it, but a test that
# drives one (pexpect, ``sh -i``) would read it. The shells the bash-backed
# sandboxes wrap each command in (Daytona, Modal, AgentCore) never run the
# image's BASH_ENV file themselves (``sandbox._base.HIDE_BASH_ENV``), nor do
# the shells Daytona's toolbox starts for each exec
# (``DaytonaSandbox._create_sandbox_without_toolbox_bash_env``).
# $1 is the value, $2 the Python source, $3 the protected inspector directory.
# It prints the resolved
# path and exits 0 when it is trusted, exits 1 when it is not, 2 when the
# image has no tool that can tell, and 3 when its tools are not root's alone.
_TRUSTED_BASH_ENV_SCRIPT = r"""
inspector=$3
[ -n "$inspector" ] && [ -d "$inspector" ] && [ ! -L "$inspector" ] || exit 2
nl='
'
safe() {
  case $1 in
    /*) ;;
    *) return 1 ;;
  esac
  case $1 in
    *'$'* | *'`'* | *'\'* | *"$nl"*) return 1 ;;
  esac
  return 0
}
by_stat() {
  s=$("$inspector/stat" -L -c '%u %a' "$1" 2>/dev/null) || return 2
  case $s in
    "0 "*) m=${s#0 } ;;
    [0-9]*" "*) return 1 ;;
    *) return 2 ;;
  esac
  case $m in "" | *[!0-7]*) return 2 ;; esac
  [ $((0$m & 022)) -eq 0 ]
}
tool() {
  t=$(command -v "$1" 2>/dev/null) || return 1
  case $t in /*) ;; *) return 1 ;; esac
  by_stat "$t" || return 1
  r=$("$inspector/readlink" -f "$t" 2>/dev/null) && [ -n "$r" ] || return 1
  owned "$r"
}
untrusted() {
  echo "bash-env-check: $1 is not a root-owned file only root can write" >&2
  exit 3
}
owned() {
  p=$1
  [ -f "$p" ] || return 1
  while :; do
    [ ! -h "$p" ] || return 1
    s=$("$inspector/stat" -c '%u %a' "$p" 2>/dev/null) || return 2
    case $s in
      "0 "*) m=${s#0 } ;;
      [0-9]*" "*) return 1 ;;
      *) return 2 ;;
    esac
    case $m in "" | *[!0-7]*) return 2 ;; esac
    [ $((0$m & 022)) -eq 0 ] || return 1
    [ "$p" = / ] && return 0
    p=${p%/*}
    [ -n "$p" ] || p=/
  done
}
have() { command -v "$1" >/dev/null 2>&1; }
safe "$1" || exit 1
if [ -x "$inspector/readlink" ] && [ -x "$inspector/stat" ]; then
  for t in stat ls readlink; do
    tool "$t" || untrusted "$t"
  done
  real=$("$inspector/readlink" -f "$1" 2>/dev/null)
  if [ -n "$real" ]; then
    safe "$real" || exit 1
    owned "$real"
    r=$?
    if [ "$r" -eq 0 ]; then
      printf '%s\n' "$real"
      exit 0
    fi
    [ "$r" -eq 1 ] && exit 1
  elif [ ! -e "$1" ]; then
    exit 1
  fi
fi
if [ ! -f __KIT_RECORD__ ]; then [ -x "$inspector/python3" ] || exit 2; fi
have python3 || exit 2
py=$(command -v python3)
__ROOT_PYTHON__
""".strip()

# The same check where the image has no ``readlink -f``, ``stat -c`` or
# ``ls``. The protected interpreter checks the live candidate's ownership
# (exit 3); the candidate itself never executes.
_TRUSTED_BASH_ENV_PYTHON = r"""
import os
import stat
import sys


def safe(path):
    return path.startswith("/") and not any(c in path for c in "$`\\\n\x00")


def root_only(path):
    while True:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o022:
            return False
        if path == "/":
            return True
        path = os.path.dirname(path)


try:
    if not root_only(os.path.realpath(sys.argv[2])):
        sys.exit(3)
except OSError:
    sys.exit(3)
real = os.path.realpath(sys.argv[1])
try:
    if not safe(real) or not stat.S_ISREG(os.lstat(real).st_mode):
        sys.exit(1)
    if not root_only(real):
        sys.exit(1)
except OSError:
    sys.exit(1)
print(real)
""".strip()

# The check above runs that source on the protected interpreter: the kit's, or
# without a kit the inspector's pre-agent copy (never the live python3 it
# judges), through root Python's one entry point (C8, PY8-PT-R8-5).
_TRUSTED_BASH_ENV_SCRIPT = _TRUSTED_BASH_ENV_SCRIPT.replace(
    "__KIT_RECORD__", KIT_RECORD
).replace(
    "__ROOT_PYTHON__",
    root_python_sh(
        _TRUSTED_BASH_ENV_PYTHON,
        raw_args='"$1" "$py"',
        fallback='"$inspector/python3"',
    ),
)

# The only folders the check's tools are taken from.
_BASH_ENV_CHECK_PATH = _SAFE_VERIFIER_PATH

# /etc is outside the writable runtime trees, unlike /tmp, which can itself
# be the agent's workspace. The random path is held by the host, never read
# from an agent-writable marker. Restored sandboxes without that host record
# fail closed rather than capturing a potentially modified toolchain.
_BASH_ENV_INSPECTOR_ROOT = "/etc"
_BASH_ENV_INSPECTOR_PARENTS = """
import os
import stat
import sys

p = sys.argv[1]
while True:
    st = os.lstat(p)
    if stat.S_ISLNK(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o022:
        sys.exit(1)
    if p == "/":
        break
    p = os.path.dirname(p)
""".strip()


_PATH_BEFORE_LINKS = r"""
inspector=$1
shift
for p do
  identity=$(stat -c '%d:%i' -- "$p") || exit 1
  printf '%s\n' "$identity"
  saved=$inspector/before-links/$identity
  [ -e "$saved" ] || [ -L "$saved" ] || ln -P -- "$p" "$saved" 2>/dev/null || :
done
""".strip()


# The names each PATH folder holds before the agent's turn (ACC-RUN2), in
# the root-only inspector: the image PATH's folders and the system folders,
# each with a slot of empty files, one per name, and an ``index`` line
# ``<slot> <folder>``; ``complete`` last. A folder that is absent gets an
# empty slot (every name in it later is new); one with a line break, or that
# root cannot list, gets none, and the verifier PATH's workspace-tool rule
# then keeps nothing of it. Builtins and the inspector's mkdir only.
# $1: the inspector; $2: the image PATH; then the system folders.
_PATH_NAMES_DIR = "path-names"
_PATH_NAMES_RECORD = r"""
rec=$1/path-names
rest=$2
shift 2
nl='
'
mkdir -m 700 -- "$rec" || exit 1
: > "$rec/index" || exit 1
n=0
bf_one() {
  case "$1" in /*) ;; *) return 0 ;; esac
  case "$1" in *"$nl"*) return 0 ;; esac
  if [ -d "$1" ] && { [ ! -r "$1" ] || [ ! -x "$1" ]; }; then return 0; fi
  n=$((n + 1))
  mkdir -- "$rec/$n" || exit 1
  if [ -d "$1" ]; then
    for e in "$1"/* "$1"/.[!.]* "$1"/..?*; do
      [ -e "$e" ] || [ -L "$e" ] || continue
      : > "$rec/$n/${e##*/}" || exit 1
    done
  fi
  printf '%s %s\n' "$n" "$1" >> "$rec/index" || exit 1
}
while :; do
  case "$rest" in
    *:*) bf_one "${rest%%:*}"; rest=${rest#*:} ;;
    *) bf_one "$rest"; break ;;
  esac
done
for d do bf_one "$d"; done
: > "$rec/complete"
""".strip()


async def _prepare_bash_env_inspector(
    env,
    *,
    prefix_command: str = "",
    service: str = "main",
    blocked: tuple[str, ...] = (),
) -> Any:
    """Save image inspectors before agent launch, alongside the baseline."""
    if service == "main":
        env._benchflow_bash_env_inspector = None
    if getattr(env, "_benchflow_bash_env_capture_allowed", True) is False:
        if prefix_command:
            return await exec_raw(env.exec, prefix_command, user="root")
        return None
    saved = getattr(env, "_benchflow_path_inspectors", None)
    if isinstance(saved, dict):
        env._benchflow_path_inspectors = {
            name: folder for name, folder in saved.items() if name != service
        }
    folder = f"{_BASH_ENV_INSPECTOR_ROOT}/.benchflow-bash-env-{uuid.uuid4().hex}"
    quoted = shlex.quote(folder)
    command = (
        f"umask 077 && mkdir {quoted} && chown 0:0 {quoted} && "
        f"chmod 711 {quoted} && "
        "for t in sh stat readlink ls id find mkdir ln chmod cat od tr cksum getfacl python3; do "
        'p=$(command -v "$t") || continue; '
        f'cp -L "$p" {quoted}/"$t" && '
        f'chown 0:0 {quoted}/"$t" && chmod 700 {quoted}/"$t" || exit 1; '
        "done && "
        # Only immutable receipt encoders are public. Search permission lets
        # any verifier UID reach them, but neither list nor read provenance.
        f"chmod 555 {quoted}/od {quoted}/tr {quoted}/cksum && "
        # Check ancestors now, before the agent could rename our directory.
        # /etc can itself be an agent-owned workspace or a shared mount.
        f"if [ -x {quoted}/stat ]; then p={quoted}; while :; do "
        '[ ! -L "$p" ] || exit 1; '
        f"s=$({quoted}/stat -c '%u %a' \"$p\") || exit 1; "
        'case $s in "0 "*) m=${s#0 } ;; *) exit 1 ;; esac; '
        'case $m in "" | *[!0-7]*) exit 1 ;; esac; '
        "[ $((0$m & 022)) -eq 0 ] || exit 1; "
        '[ "$p" = / ] && break; p=${p%/*}; [ -n "$p" ] || p=/; done; '
        # Without stat: the protected interpreter (the kit's, or before a
        # kit this folder's pre-agent copy), through root Python's one
        # entry point (C8, PY8-PT-R8-5).
        f"elif [ -f {KIT_RECORD} ] || [ -x {quoted}/python3 ]; then "
        + root_python_sh(
            _BASH_ENV_INSPECTOR_PARENTS, folder, fallback=f"{quoted}/python3"
        )
        + " || exit 1; else exit 1; fi && "
        # What each PATH folder holds now (_PATH_NAMES_RECORD); the image
        # PATH is read before PATH names the inspector. A record that fails
        # has no `complete`, which only turns its rule off.
        f"{{ (bf_ip=${{_BENCHFLOW_IMAGE_PATH-$PATH}}; PATH={quoted}; export PATH; "
        f"sh -c {shlex.quote(_PATH_NAMES_RECORD)} bf-path-names {quoted} "
        f'"$bf_ip" {shlex.join(_SAFE_VERIFIER_PATH_PARTS)}) >/dev/null 2>&1 || :; }} && '
        # Preserve pre-existing hard-link identities even when the agent
        # removes its writable name after changing the shared inode.
        # A protected hard-link reference also survives inode renumbering in
        # a container snapshot. Cross-mount links can fail (EXDEV); those
        # mounted files retain their original device/inode across restore.
        f"(PATH={quoted}; export PATH; "
        f"mkdir {quoted}/before-links || exit 1; "
        f"for p in {shlex.join(blocked)}; do "
        'if [ -e "$p" ] || [ -L "$p" ]; then '
        'find -H "$p" ! -type d -links +1 -exec sh -c '
        f"{shlex.quote(_PATH_BEFORE_LINKS)} bf-before-links {quoted} "
        "{} + || exit 1; "
        f"fi; done) > {quoted}/before && printf 'ready\\n'"
    )
    if prefix_command:
        command = f"{prefix_command}; ({command})"
    result = await _read_exec(
        env,
        command,
        # PATH, loader and start-up values: the exec route's alone (C8).
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        **_in_service(service),
    )
    if _exec_return_code(result) == 0 and "ready" in (result.stdout or "").splitlines():
        if service == "main":
            env._benchflow_bash_env_inspector = folder
        saved = getattr(env, "_benchflow_path_inspectors", None)
        env._benchflow_path_inspectors = {
            **(saved if isinstance(saved, dict) else {}),
            service: folder,
        }
    return result


# Where a path names what the process reading it is, not a file on disk.
_BASH_ENV_PSEUDO_PREFIXES = ("/proc", "/sys", "/dev")


def _trusted_bash_env_cmd(value: str, inspector: str = "") -> str:
    """The container-side command that checks a BASH_ENV *value*."""
    return (
        f"cd / && sh -c {shlex.quote(_TRUSTED_BASH_ENV_SCRIPT)} bash-env-check "
        f"{shlex.quote(value)} {shlex.quote(_TRUSTED_BASH_ENV_PYTHON)} "
        f"{shlex.quote(inspector)}"
    )


def _bash_env_refused_on_host(
    value: str, blocked: tuple[str, ...], *, resolved: bool = False
) -> bool:
    """The part of the BASH_ENV trust check the host can do itself.

    A value under an agent-writable prefix is refused here whatever the
    container check answers. A *resolved* path (the check's answer) must also be
    in its normal form and outside the pseudo filesystems.
    """
    if not value.startswith("/") or any(c in value for c in "$`\\\n\x00"):
        return True
    path = posixpath.normpath(value)
    prefixes = blocked + _BASH_ENV_PSEUDO_PREFIXES if resolved else blocked
    for prefix in prefixes:
        prefix = prefix.rstrip("/")
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return resolved and path != value


def _in_service(service: str) -> dict[str, str]:
    """The exec keyword that runs a command in *service* (main by default)."""
    return {} if service == "main" else {"service": service}


async def _image_bash_env(env, service: str = "main") -> str:
    """The image's own BASH_ENV, as an exec without one names it, or empty.

    Read on the task's route (C8): root's plumbing never has the image's
    start-up values; a task command gets them back before it runs, and the
    builtin ``printf`` starts no program that could load what they name.
    """
    try:
        with as_task_command(env):
            result = await _read_exec(
                env,
                "command printf '%s\\n' \"${BASH_ENV-}\"",
                user="root",
                timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
                **_in_service(service),
            )
    except Exception as exc:
        logger.warning(
            "The verifier runs with BASH_ENV empty: the image's BASH_ENV in "
            "service %r could not be read: %s",
            service,
            exc,
        )
        return ""
    if _exec_return_code(result) != 0:
        logger.warning(
            "The verifier runs with BASH_ENV empty: the image's BASH_ENV in "
            "service %r could not be read (status %s).%s",
            service,
            _exec_return_code(result),
            _exec_failure_detail(result),
        )
        return ""
    return (result.stdout or "").rstrip("\n")


async def _trusted_verifier_bash_env(
    env, task: "Task", blocked: tuple[str, ...]
) -> str:
    """The resolved file the verifier's BASH_ENV names, if trusted, else empty.

    The task's ``[verifier.env]`` value; when the task names none, the
    image's own (an empty value the task names stays empty). The image value
    is read in the verifier's service. Main uses protected pre-agent inspectors;
    other services are refused without a map of shared writable paths.
    """
    service = task.config.verifier.service or "main"
    task_env = task.config.verifier.env or {}
    if "BASH_ENV" in task_env:
        # A ``${NAME:-default}`` template is resolved as exec resolves every
        # [verifier.env] value (resolve_env_vars) before the check, never
        # checked in its unexpanded form (b156 PY8-MAIN-34); a plain value is
        # expanded as before.
        from benchflow.task.env import resolve_env_vars

        source = "the task's"
        try:
            resolved = resolve_env_vars({"BASH_ENV": task_env["BASH_ENV"]})
            value = os.path.expandvars(resolved["BASH_ENV"])
        except ValueError as exc:
            logger.warning(
                "The verifier runs with BASH_ENV empty: %s %r cannot be resolved: %s",
                source,
                task_env["BASH_ENV"],
                exc,
            )
            return ""
    elif (
        isinstance(persistent := getattr(env, "_persistent_env", None), dict)
        and "BASH_ENV" in persistent
    ):
        # The sandbox's own env (a task's [environment] env), which exec adds
        # to every command: read here, since soft verification runs with the
        # persistent env held back from exec and the probe would not see it
        # (b162 PY8-MAIN-37). It goes through the same check.
        value = str(persistent["BASH_ENV"])
        source = "the sandbox's"
    else:
        value = await _image_bash_env(env, service)
        source = "the image's"
    if not value:
        return ""
    refused = (
        "The verifier runs with BASH_ENV empty: %s %r is not a root-owned "
        "file outside the agent's reach"
    )
    if _bash_env_refused_on_host(value, blocked):
        logger.warning(refused, source, value)
        return ""
    # Ownership freezing changes shared-volume aliases to root ownership, so
    # a post-freeze stat cannot establish that the agent never wrote them.
    # Backends expose no common mount-provenance map. Until one is available,
    # refuse activation in another service (including protected-looking paths)
    # rather than carry main's blocked prefixes into a different namespace.
    if service != "main":
        logger.warning(
            "The verifier runs with BASH_ENV empty: %s %r in service %r "
            "cannot be checked without shared-mount provenance",
            source,
            value,
            service,
        )
        return ""
    inspector = getattr(env, "_benchflow_bash_env_inspector", None)
    if not isinstance(inspector, str) or not inspector:
        logger.warning(
            "The verifier runs with BASH_ENV empty: %s %r has no protected "
            "pre-agent inspector",
            source,
            value,
        )
        return ""
    # Empty for the check itself, whatever the image names.
    try:
        result = await _read_exec(
            env,
            _trusted_bash_env_cmd(value, inspector),
            # PATH, loader and start-up values: the exec route's alone (C8).
            env={"LC_ALL": "C"},
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
            **_in_service(service),
        )
    except Exception as exc:
        # Emptying it can change the score, so it is said where it is seen
        # .
        logger.warning(
            "The verifier runs with BASH_ENV empty: %s %r could not be checked: %s: %s",
            source,
            value,
            type(exc).__name__,
            exc,
        )
        return ""
    code = _exec_return_code(result)
    if code == 1:
        logger.warning(refused, source, value)
        return ""
    lines = (result.stdout or "").splitlines()
    if code != 0 or len(lines) != 1:
        logger.warning(
            "The verifier runs with BASH_ENV empty: %s %r could not be checked.%s",
            source,
            value,
            _exec_failure_detail(result),
        )
        return ""
    kept = lines[0]
    if _bash_env_refused_on_host(kept, blocked, resolved=True):
        logger.warning(refused, source, value)
        return ""
    return kept


# Wipe /logs/verifier/ contents before the verifier runs. Do not remove the
# directory itself: Daytona DinD bind-mounts it from the remote VM, so deleting
# the mountpoint fails with "Device or resource busy". ``find ... -exec ... +``
# avoids glob ARG_MAX failures if an agent floods the world-writable log dir.
# When ``find`` is unavailable (minimal service images), fall back to
# ``rm -rf /logs/verifier/* ...`` which handles the common case but may miss
# dot-files and can hit ARG_MAX on extreme floods.
_CLEAR_VERIFIER_DIR_CMD = (
    "if [ -L /logs/verifier ]; then rm -f /logs/verifier; fi && "
    "mkdir -p /logs/verifier && "
    "if command -v find >/dev/null 2>&1; then "
    "find /logs/verifier -mindepth 1 -exec rm -rf -- {} +; "
    "else "
    "rm -rf /logs/verifier/* /logs/verifier/.[!.]* /logs/verifier/..?* 2>/dev/null; true; "
    "fi && "
    "chmod 777 /logs/verifier"
)
# The same clear while the trial keeps services (x-keep-services): the folder
# is left root's 0755 instead of 0777. The verifier runs as root then
# (trial_keep), so nothing needs the write for others. On a verifier retried
# in place the kept services are already running: a folder they can write,
# even for the moment between this clear and the release, lets one that holds
# it through a descriptor in flight on a socket create a file of its own
# there (reward.txt) and keep it open, and test.sh would then write into the
# agent's file (K2-R4-A-1).
_CLEAR_VERIFIER_DIR_CLOSED_CMD = (
    _CLEAR_VERIFIER_DIR_CMD.removesuffix("chmod 777 /logs/verifier")
    + "chown root:root /logs/verifier && chmod 755 /logs/verifier"
)

# Legacy verifier rootdir fallback for main-container verification paths.
_ENSURE_APP_DIR_CMD = "mkdir -p /app"


def _exec_return_code(result: Any) -> int:
    if result is None:
        return 0
    for name in ("return_code", "exit_code"):
        value = getattr(result, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _exec_failure_detail(result: Any) -> str:
    result = diagnostic_result(result)
    parts = []
    for name in ("stdout", "stderr"):
        value = getattr(result, name).render()
        if value:
            text = str(value).strip()
            if text:
                parts.append(f"{name}: {text[:2000]}")
    if not parts:
        return ""
    return "\n" + "\n".join(parts)


async def _checked_exec(env: Any, command: str, label: str, /, **kwargs: Any) -> Any:
    result = await exec_raw(env.exec, command, **kwargs)
    return_code = _exec_return_code(result)
    if return_code != 0:
        raise RuntimeError(
            f"{label} exited with rc={return_code}{_exec_failure_detail(result)}"
        )
    return result


# Every exec whose output this module decides on goes through _read_exec:
# the output then arrives as the sandbox printed it, or as a failed read,
# never rewritten by the backend's redaction or merged stderr
# (_exec_receipt.py, CONFIRM-lockdown-7). Execs judged by status alone use
# env.exec. tests/test_lockdown_c40_receipt.py checks that every stdout read
# here comes from a receipt.
_read_exec = receipt_exec


async def _checked_read(env: Any, command: str, label: str, /, **kwargs: Any) -> Any:
    """:func:`_checked_exec` whose output is read from a receipt."""
    result = await _read_exec(env, command, **kwargs)
    if result.return_code != 0:
        raise RuntimeError(
            f"{label} exited with rc={result.return_code}{_exec_failure_detail(result)}"
        )
    return result


# Verifier-setup commands that walk the full rootfs (notably the conftest purge)
# run on the verifier sandbox, whose filesystem can be slow/network-backed
# (Daytona). The find is pruned (see _build_cleanup_cmd) so it completes well
# within this budget; the larger-than-default ceiling keeps a slow-but-valid
# setup from false-erroring the verifier. Owned here so every call site — the
# scoring path (harden_before_verify) and the soft-verify path (rollout, via
# cleanup_verifier_python_hooks) — shares one value rather than scattering it.
VERIFIER_SETUP_TIMEOUT_SEC = 180
# What the guard's install exec keeps of that budget for its steps after the
# input proof (the key, the copies into every Python), so a proof that
# reads a workspace of many gigabytes ends tainted instead of timing the
# exec out (V2-R1-A-2).
_INPUT_PROOF_RESERVE_SEC = 60


async def clear_verifier_output_dir(
    env: Any,
    label: str = "Verifier setup failed: clearing verifier output directory",
    *,
    closed: bool = False,
    **kwargs: Any,
) -> Any:
    """Clear verifier outputs while preserving bind mounts.

    This helper is intentionally service-aware: final anti-tamper hardening
    stays on the agent container, but a verifier running in another service may
    still need its own writable ``/logs/verifier`` output directory.

    ``closed=True`` (the trial keeps services) leaves the folder root's 0755
    instead of 0777 (``_CLEAR_VERIFIER_DIR_CLOSED_CMD``).
    """
    command = _CLEAR_VERIFIER_DIR_CLOSED_CMD if closed else _CLEAR_VERIFIER_DIR_CMD
    return await _checked_exec(env, _on_path(command), label, **kwargs)


async def ensure_legacy_app_dir(
    env: Any,
    label: str = "Verifier setup failed: preparing /app",
    **kwargs: Any,
) -> Any:
    """Prepare the legacy verifier ``/app`` rootdir fallback."""
    return await _checked_exec(env, _on_path(_ENSURE_APP_DIR_CMD), label, **kwargs)


async def soft_verifier_bash_env(
    env: Any, task: "Task", sandbox_user: str | None, workspace: str | None
) -> str:
    """The BASH_ENV a soft verification's test.sh may read: hardening's policy.

    The same check the scored path makes before the verifier
    (:func:`_trusted_verifier_bash_env`): a root-owned file outside the
    agent's reach that the protected pre-agent inspector vouches for, else
    empty. Soft verification never hardens, so without this test.sh's bash
    ran the image's BASH_ENV, which the agent could rewrite to undo the pip
    pin (b127 PY8-MAIN-26). Any failure of the check is empty.
    """
    try:
        return await _trusted_verifier_bash_env(
            env, task, _blocked_verifier_path_prefixes(sandbox_user, workspace)
        )
    except Exception as exc:
        logger.warning("Soft verification runs with BASH_ENV empty: %s", exc)
        return ""


async def cleanup_verifier_python_hooks(
    env: Any,
    task_dir: "Path | str | None",
    label: str = "Verifier setup failed: purging Python injection hooks",
    *,
    timeout_sec: int = VERIFIER_SETUP_TIMEOUT_SEC,
    workspace: str | None = None,
    **kwargs: Any,
) -> Any:
    """Purge agent-injected Python hook files using task hardening settings.

    The conftest purge walks the full rootfs, so the timeout defaults to
    VERIFIER_SETUP_TIMEOUT_SEC (the same budget the scoring path uses in
    harden_before_verify) rather than the caller guessing a value.
    """
    hardening = _read_hardening_config(task_dir)
    workspace_aliases = (
        await _blocked_before_freeze(env, (workspace,)) if workspace else ()
    )
    if _hook_cleanup_on_bfx(env):
        # No interpreter is judged or needed: the host decides, bfx changes.
        await _cleanup_hooks_on_bfx(
            env, hardening, workspace, workspace_aliases, label, timeout_sec=timeout_sec
        )
        return None
    # The purge's interpreter is judged first, as on the scoring path
    # present candidates that all fail refuse; none, no purge.
    root_python = await _judge_root_python(
        env, _blocked_verifier_path_prefixes(None, workspace), None
    )
    if root_python.path is None and root_python.rejected:
        raise RuntimeError(f"{label}: {root_python.missing_text('of this task')}")
    token = _root_python.set(root_python.path)
    try:
        return await _checked_exec(
            env,
            _prep_on_path(_build_cleanup_cmd(hardening, workspace, workspace_aliases)),
            label,
            timeout_sec=timeout_sec,
            **kwargs,
        )
    finally:
        _root_python.reset(token)


# What the start-up hook clean-up says when the sandbox has no protected
# Python to run it on (PT-R16-R2-2; deferred to 0.8.4).
NO_PROTECTED_PYTHON_SKIP = (
    "benchflow: no protected Python in this sandbox, so the Python start-up "
    "hook clean-up was skipped"
)


# Per-task hardening opt-outs. Tasks declare these in task config under
# [verifier.hardening] when their legitimate test setup conflicts with the
# default cleanup (e.g. qutebrowser ships a real conftest.py that the cleanup
# would otherwise delete, breaking pytest collection).
#
# Defaults are secure (all True). Tasks opt out individually:
#
#   [verifier.hardening]
#   cleanup_conftests = false   # don't delete conftest.py before verify
HARDENING_DEFAULTS: dict[str, bool] = {
    "cleanup_conftests": True,
    "cleanup_git": True,
}


def _read_hardening_config(task_dir: "Path | str | None") -> dict[str, bool]:
    """Read [verifier.hardening] from task.toml or task.md frontmatter."""
    import tomllib

    result = dict(HARDENING_DEFAULTS)
    if task_dir is None:
        return result
    root = Path(task_dir)
    toml_path = root / "task.toml"
    document_path = root / "task.md"
    if document_path.exists():
        try:
            from benchflow.task.document import TaskDocument

            data = TaskDocument.from_path(document_path).frontmatter
        except Exception as e:
            logger.warning(f"task.md parse error in {task_dir}: {e}")
            return result
    elif toml_path.exists():
        try:
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            logger.warning(f"task.toml parse error in {task_dir}: {e}")
            return result
    else:
        return result
    overrides = data.get("verifier", {}).get("hardening", {})
    for k, v in overrides.items():
        if k in result and isinstance(v, bool):
            result[k] = v
        else:
            logger.warning(f"task [verifier.hardening] unknown/invalid: {k}={v!r}")
    return result


def _build_cleanup_cmd(
    hardening: dict[str, bool] | None = None,
    workspace: str | None = None,
    workspace_aliases: tuple[str, ...] = (),
    *,
    keep: tuple[str, ...] = (),
    agent_user: bool | None = None,
) -> str:
    """Build the cleanup shell command, honoring per-task hardening opt-outs.

    Steps:
      - conftest.py removal outside /tests (skippable via cleanup_conftests=false)
      - *.py purge from /tmp /var/tmp outside the submission workspace and
        its mount aliases
      - start-up hooks new or changed since the pre-agent record removed,
        kept ones rewritten as fresh root-owned files (HOOK_RULE), in the
        folders test.sh's python3 would search (rebuilt as text on PT's
        protected Python, U5a-CONF-5 included), sparing the .pth files of
        an unrecorded site folder inside *workspace*, which the freeze's
        hooks purge judges (V8)

    sitecustomize/usercustomize/.pth always run — opt-outs there would broaden
    the attack surface beyond what real-world tasks need.

    ``keep`` lists absolute folders the conftest walk prunes
    (there are no verifier-cache exemptions).
    """
    h = hardening or HARDENING_DEFAULTS
    parts: list[str] = []
    if h.get("cleanup_conftests", True):
        parts.append(
            # Prune virtual filesystems so the full-rootfs walk stays bounded and
            # fast — an unpruned `find /` over a slow network-backed FS (Daytona)
            # routinely exceeds the setup timeout and false-errors a valid verifier.
            # `-exec rm`, not `-delete`: GNU find refuses -delete next to -prune
            # (exit 1, nothing removed), which made this sweep a silent no-op.
            # V2's protected verifier runtime holds a copy of the image's site
            # folders, a package's own tests/conftest.py included: it is root's
            # alone and checked file by file before test.sh, so a conftest.py
            # planted or changed there refuses grading anyway; removing one
            # made that check refuse an honest trial.
            "find / -path /proc -prune -o -path /sys -prune -o -path /dev -prune -o "
            + f"-path {shlex.quote(VERIFIER_DIR)} -prune -o "
            + "".join(f"-path {shlex.quote(path)} -prune -o " for path in keep)
            + "-name conftest.py "
            "-not -path '/verifier/*' -not -path '/tests/*' "
            "-exec rm -f -- {} + 2>/dev/null"
        )
    if workspace:
        # pwd -P covers a workspace reached through a symlink. Never use
        # -delete with -prune: its implicit depth traversal defeats the prune.
        wsq = shlex.quote(os.path.normpath(workspace))
        # find -path consumes a glob even when the shell argument is quoted.
        # Escape the complete glob alphabet in lexical and physical paths.
        literal = shlex.quote(
            re.sub(r"([\\*?\[\]])", r"\\\1", os.path.normpath(workspace))
        )
        # A sentinel preserves trailing newlines through command substitution,
        # whether sed preserves an unterminated final line or adds a newline.
        parts.append(
            f"bf_cleanup_workspace=$(cd -P {wsq} && printf '%s.' \"$PWD\" | "
            r"sed 's/[][\\*?]/\\&/g') || exit 1"
        )
        parts.append("bf_cleanup_workspace=${bf_cleanup_workspace%.}")
        aliases = "".join(
            " -o -path "
            + shlex.quote(re.sub(r"([\\*?\[\]])", r"\\\1", os.path.normpath(path)))
            for path in workspace_aliases
            if os.path.normpath(path) != os.path.normpath(workspace)
        )
        parts.append(
            f'find /tmp /var/tmp \\( -path {literal} -o -path "$bf_cleanup_workspace"{aliases} \\) '
            "-prune -o -name '*.py' -exec rm -f -- {} + 2>/dev/null"
        )
    else:
        parts.append("find /tmp /var/tmp -name '*.py' -delete 2>/dev/null")
    # The site-hook clean-up runs only on PT's protected Python (C8,
    # PY8-PT-R8-5): no image python3 starts as root here, so U5a-CONF-2 is not
    # reachable through this helper. PT-R16's HOOK_RULE judges every folder
    # test.sh's python3 would search, rebuilt as text from its inputs
    # (PT-R16-B-1), which keeps U5a's
    # CONF-5 coverage: a relative, empty or "." PYTHONPATH entry names the
    # original working folder. The inputs reach it in its environment, never
    # in argv. 127 is no protected Python (the kit holds
    # none): the clean-up is skipped and says so (PT-R16-R2-2). Any other
    # failure, a hook that could not be removed included, fails the step
    # (PT-R16-B-4).
    parts.append(
        _cleanup_site_cmd(
            workspace,
            agent_user=_hook_agent_user.get() if agent_user is None else agent_user,
            workspace_aliases=workspace_aliases,
        )
    )
    return "; ".join(parts)


# The folders test.sh's python3 would import start-up hooks from, found
# without starting it, judged by PT-R16's HOOK_RULE: the
# pre-agent records' folders, root's user site, the site folders of the
# first python3 on root's route PATH and on the image PATH (test.sh's; B-4),
# each by realpath and by Python's own venv rule (a pyvenv.cfg beside the
# executable or one folder up makes that folder a prefix, D913 call 1), and
# the search inputs (_bf_search_inputs). The inputs arrive in the
# environment as _BF_HOOK_* data, never in argv.
# argv: the original working folder, the workspace.
_CLEANUP_SITE_SCRIPT = (
    READ_IMAGE_PYTHON_PATHS
    + "\n"
    + IMAGE_PYTHON_HOOK_RULE
    + "\n"
    + (
        "import glob, os\n"
        "cwd, workspace = (sys.argv[1:] + [''] * 2)[:2]\n"
        "values = " + IMAGE_PYTHON_SEARCH_VALUES + "\n"
        "dirs = set(_bf_paths['path'] + _bf_paths['user_site'])\n"
        "globbed = set(glob.glob('/root/.local/lib/python3*/site-packages'))\n"
        "prefixes = []\n"
        "for search in (os.environ.get('PATH', ''), os.environ.get('_BF_HOOK_IMAGE_PATH', '')):\n"
        "    for p in search.split(':'):\n"
        "        exe = os.path.join(p, 'python3')\n"
        "        if (not os.path.isabs(p) or p.startswith('/opt/benchflow/')\n"
        "                or not os.path.isfile(exe)):\n"
        "            continue\n"
        "        prefixes.append(os.path.dirname(os.path.dirname(os.path.realpath(exe))))\n"
        "        bindir = os.path.dirname(os.path.abspath(exe))\n"
        "        for cfg in (bindir, os.path.dirname(bindir)):\n"
        "            if os.path.isfile(os.path.join(cfg, 'pyvenv.cfg')):\n"
        "                prefixes.append(cfg)\n"
        "                break\n"
        "        break\n"
        # The prefix site folders are the interpreter's getsitepackages(): the
        # only ones V8's workspace-venv rule spares (never a user site).
        "prefix_sites = set()\n"
        "for prefix in prefixes:\n"
        "    for pat in ('lib/python3*/site-packages', 'lib/python3*/dist-packages'):\n"
        "        prefix_sites.update(glob.glob(os.path.join(glob.escape(prefix), pat)))\n"
        "prefix_sites = set(os.path.normpath(p) for p in prefix_sites)\n"
        "globbed |= prefix_sites\n"
        "dirs |= globbed\n"
        "w = os.path.realpath(workspace) + '/' if workspace else ''\n"
        # A folder the pre-agent records list (every interpreter's search
        # path, root's python3's, a venv's on PATH, a conda env's; the
        # folders root-owned .pth files name are on them) keeps exactly the
        # hooks it held then, unchanged, by the rule site itself follows
        # (HOOK_RULE, shared with the record): sitecustomize and
        # usercustomize in any form in any folder (PT-R15-B1), *.pth only in
        # a site folder, so /app/model.pth is data (PT-R15-B3). An honest
        # editable install's .pth stays wherever its venv lives (PT-R14-B2),
        # and a hook the agent wrote or edited goes, root's own folders
        # included (no fixed /usr skip, PT-R14-A-1). A kept hook is read once
        # without following a link, checked against its digest, and written
        # again as a new root-owned file renamed over the name, so no
        # descriptor a kept service holds reaches what the verifier runs
        # (PT-R15-B2), as for /logs/agent. A link or folder is removed
        # without following it. A folder no record lists (an older kit,
        # root's route-PATH folders) loses every hook, outside /usr/lib and
        # /usr/local/lib, as before.
        "import errno\n"
        "recorded = _bf_recorded_hooks()\n"
        "site_folders = _bf_site_folders() | set(_bf_paths['user_site']) | globbed\n"
        # test.sh's Python also searches the folders its PYTHONPATH,
        # PYTHONHOME, user site and site folders' .pth files add, which no
        # -I record shows (PT-R16-B-1): the search inputs, the
        # arguments, read as text. A folder they add is checked for every
        # sitecustomize/usercustomize form, *.pth only where it is a site
        # folder.
        # The .pth files' folders are read later, from the .pth files this
        # pass keeps: a new one that goes names nothing (PT-R16-R2-3).
        "more, more_sites = _bf_search_inputs(\n"
        "    values, cwd, site_folders,\n"
        "    False)\n"
        "site_folders |= more_sites\n"
        "dirs |= more | more_sites\n"
        # A hook that must go and cannot is a failed clean-up, never a
        # silent keep (PT-R16-B-4).
        "failed = []\n"
        "_BF_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _BF_NB\n"
        "def remove(dfd, name):\n"
        "    if stat.S_ISDIR(os.stat(name, dir_fd=dfd, follow_symlinks=False).st_mode):\n"
        "        fd = os.open(name, _BF_DIR, dir_fd=dfd)\n"
        "        try:\n"
        "            for n in os.listdir(fd):\n"
        "                remove(fd, n)\n"
        "        finally:\n"
        "            os.close(fd)\n"
        "        os.rmdir(name, dir_fd=dfd)\n"
        "    else:\n"
        "        os.unlink(name, dir_fd=dfd)\n"
        "def write(dfd, name, node):\n"
        "    if node[0] == 'd':\n"
        "        os.mkdir(name, 0o700, dir_fd=dfd)\n"
        "        fd = os.open(name, _BF_DIR, dir_fd=dfd)\n"
        "        try:\n"
        # A link inside a kept package is written as the file it pointed
        # to; one that pointed to no regular file is left out.
        "            for n, k in node[2]:\n"
        "                if k[0] != 'l' or k[2] is not None:\n"
        "                    write(fd, n, k)\n"
        "            if os.geteuid() == 0:\n"
        "                os.fchown(fd, 0, 0)\n"
        "            os.fchmod(fd, stat.S_IMODE(node[1]) & 0o755)\n"
        "        finally:\n"
        "            os.close(fd)\n"
        "        return\n"
        "    data, mode = (node[1], node[2]) if node[0] == 'f' else (node[2], 0o644)\n"
        "    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW"
        " | _BF_NB, 0o600, dir_fd=dfd)\n"
        "    try:\n"
        "        while data:\n"
        "            data = data[os.write(fd, data):]\n"
        "        if os.geteuid() == 0:\n"
        "            os.fchown(fd, 0, 0)\n"
        "        os.fchmod(fd, stat.S_IMODE(mode) & 0o755)\n"
        "    finally:\n"
        "        os.close(fd)\n"
        "def fresh(dfd, name, node):\n"
        "    tag = binascii.hexlify(os.urandom(8)).decode()\n"
        "    new, old = '.bf-fresh-' + tag + '.tmp', '.bf-stale-' + tag + '.tmp'\n"
        "    write(dfd, new, node)\n"
        "    if node[0] == 'd':\n"
        "        os.rename(name, old, src_dir_fd=dfd, dst_dir_fd=dfd)\n"
        "        os.rename(new, name, src_dir_fd=dfd, dst_dir_fd=dfd)\n"
        "        remove(dfd, old)\n"
        "    else:\n"
        # rename replaces a link at the name; it never writes through it.
        "        os.rename(new, name, src_dir_fd=dfd, dst_dir_fd=dfd)\n"
        # ACC-RUN2: in a trial with a sandbox user (argv 3 is '1'), a new or
        # changed hook only root could have written stays, written again
        # fresh like a recorded one: a file with one name or a folder of
        # such files, no link,
        # root's with no group or other write, in a folder that is root's
        # alone up to /, outside the workspace, its aliases (argv 4 on) and
        # the trees the agent user may write. The agent user cannot add or
        # change such an entry; root did, during the turn (the task's
        # solution, a service of the task, a package it installed: a cmeel
        # package's `import cmeel_pth`).
        "agent_user = sys.argv[3:4] == ['1']\n"
        "excluded = [os.path.realpath(p).rstrip('/') + '/' for p in\n"
        "            [workspace] + sys.argv[4:] + "
        + repr(list(ROOT_MADE_EXCLUDED))
        + " if p]\n"
        "root_made = []\n"
        "def _bf_root_alone(st):\n"
        "    return st.st_uid == 0 and not st.st_mode & 0o022\n"
        "def _bf_root_tree(dfd, name, depth=0):\n"
        "    st = os.stat(name, dir_fd=dfd, follow_symlinks=False)\n"
        "    if not _bf_root_alone(st):\n"
        "        return False\n"
        # A file with another name (nlink > 1) may have one the agent user
        # could reach (ACC-RUN2 r2). The freeze leaves such a file's owner
        # as it is, so one the agent user owned is refused above even when
        # the clean-up has since removed its other name (ACC-L1).
        "    if stat.S_ISREG(st.st_mode):\n"
        "        return st.st_nlink == 1\n"
        "    if not stat.S_ISDIR(st.st_mode) or depth >= 16:\n"
        "        return False\n"
        "    fd = os.open(name, _BF_DIR, dir_fd=dfd)\n"
        "    try:\n"
        "        names = os.listdir(fd)\n"
        "        return len(names) <= 4096 and all(_bf_root_tree(fd, n, depth + 1) for n in names)\n"
        "    finally:\n"
        "        os.close(fd)\n"
        "def _bf_root_made(dfd, name):\n"
        "    if not agent_user:\n"
        "        return False\n"
        "    try:\n"
        "        here = os.path.realpath('/proc/self/fd/%d' % dfd)\n"
        "        if not here.startswith('/') or any((here.rstrip('/') + '/').startswith(e) for e in excluded):\n"
        "            return False\n"
        "        folder = here\n"
        "        while True:\n"
        "            st = os.stat(folder, follow_symlinks=False)\n"
        "            if not (stat.S_ISDIR(st.st_mode) and _bf_root_alone(st)):\n"
        "                return False\n"
        "            if folder == '/':\n"
        "                break\n"
        "            folder = os.path.dirname(folder)\n"
        "        a, b = os.stat(here), os.fstat(dfd)\n"
        "        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):\n"
        "            return False\n"
        "        return _bf_root_tree(dfd, name)\n"
        "    except (OSError, ValueError, RecursionError):\n"
        "        return False\n"
        "def keep_or_remove(dfd, name, digest):\n"
        "    node = None\n"
        "    if digest is not None:\n"
        "        node = _bf_node(name, dfd)\n"
        "        if not (node is not None and _bf_digest(node) == digest\n"
        "                and not (node[0] == 'l' and node[2] is None)):\n"
        "            node = None\n"
        "    if node is None and _bf_root_made(dfd, name):\n"
        "        node = _bf_node(name, dfd)\n"
        "        if node is not None and node[0] != 'l':\n"
        "            root_made.append(name)\n"
        "        else:\n"
        "            node = None\n"
        "    if node is not None:\n"
        "        try:\n"
        "            fresh(dfd, name, node)\n"
        "            return\n"
        "        except OSError as error:\n"
        # A read-only mount: nobody writes there, so the hook stays as is.
        "            if error.errno == errno.EROFS:\n"
        "                return\n"
        "    remove(dfd, name)\n"
        "def clean(dfd, known, site_folder):\n"
        "    for f in os.listdir(dfd):\n"
        "        try:\n"
        # Bytecode caches of a hook source are only caches: always dropped,
        # a __pycache__ link too.
        "            if f == '__pycache__':\n"
        "                if stat.S_ISLNK(os.stat(f, dir_fd=dfd, follow_symlinks=False).st_mode):\n"
        "                    os.unlink(f, dir_fd=dfd)\n"
        "                    continue\n"
        "                fd = os.open(f, _BF_DIR, dir_fd=dfd)\n"
        "                try:\n"
        "                    for n in os.listdir(fd):\n"
        "                        if _bf_is_hook(n, False):\n"
        "                            remove(fd, n)\n"
        "                finally:\n"
        "                    os.close(fd)\n"
        "            elif _bf_is_hook(f, site_folder):\n"
        "                keep_or_remove(dfd, f, None if known is None else known.get(f))\n"
        "        except OSError as error:\n"
        "            if error.errno != errno.ENOENT:\n"
        "                failed.append(f)\n"
        "def visit(d):\n"
        "    known = recorded.get(d)\n"
        "    if known is None and d.startswith(('/usr/lib', '/usr/local/lib')):\n"
        "        return\n"
        # The search path entry itself, when it is not a plain folder (a
        # zip archive, a link), must be as recorded.
        "    if known is not None and (os.path.islink(d) or (os.path.lexists(d) and not os.path.isdir(d))):\n"
        "        parent = _bf_open_dir(os.path.dirname(d))\n"
        "        if parent is None:\n"
        "            return\n"
        "        try:\n"
        "            node = _bf_node(os.path.basename(d), parent)\n"
        "            if node is None or node[0] != 'l' or _bf_digest(node) != known.get(''):\n"
        "                keep_or_remove(parent, os.path.basename(d), known.get(''))\n"
        "                return\n"
        "        except OSError as error:\n"
        "            if error.errno != errno.ENOENT:\n"
        "                failed.append(d)\n"
        "            return\n"
        "        finally:\n"
        "            os.close(parent)\n"
        "    dfd = _bf_open_dir(d)\n"
        "    if dfd is None:\n"
        "        return\n"
        "    try:\n"
        # V8-R1-B-2: an unrecorded prefix site folder inside the workspace (a
        # workspace venv the freeze's walk judged) keeps its .pth files; a
        # recorded one is the record's to judge (D913 call 6).
        "        spare = (bool(w) and known is None and d in prefix_sites\n"
        "                 and (os.path.realpath(d) + '/').startswith(w))\n"
        "        clean(dfd, known, d in site_folders and not spare)\n"
        "    finally:\n"
        "        os.close(dfd)\n"
        "visited = {}\n"
        "for d in sorted(set(recorded) | dirs):\n"
        "    visited[d] = d in site_folders\n"
        "    visit(d)\n"
        # Then the folders the kept .pth files name, until none is new: a
        # .pth that just went names no folder of test.sh's (PT-R16-R2-3). A
        # literal site.addsitedir argument is a site folder, checked again
        # if it was first visited as a plain one.
        "for _ in range(64):\n"
        "    named, named_sites = set(), set()\n"
        "    for d in sorted(site_folders):\n"
        "        _bf_pth_folders(d, named, named_sites)\n"
        "    site_folders |= named_sites\n"
        "    todo = sorted(d for d in named | named_sites\n"
        "                  if d not in visited or (d in site_folders and not visited[d]))\n"
        "    if not todo:\n"
        "        break\n"
        "    for d in todo:\n"
        "        visited[d] = d in site_folders\n"
        "        visit(d)\n"
        "if root_made:\n"
        "    sys.stderr.write('benchflow: kept %d new or changed Python start-up hook(s) '\n"
        "                     'only root could have written, e.g. %r\\n' % (len(root_made), root_made[0]))\n"
        "if failed:\n"
        "    sys.stderr.write('benchflow: could not remove %d Python start-up hook(s), '\n"
        "                     'e.g. %r\\n' % (len(failed), failed[0]))\n"
        "    sys.exit(3)\n"
    )
)


def _cleanup_site_cmd(
    workspace: str | None = None,
    *,
    agent_user: bool = False,
    workspace_aliases: tuple[str, ...] = (),
) -> str:
    """The site-hook clean-up on the protected Python (_CLEANUP_SITE_SCRIPT).

    test.sh's search-path inputs go in the environment as ``_BF_HOOK_*``
    data (never argv): the route's copies
    (:data:`~benchflow.sandbox._base.PYTHON_SEARCH_DATA`, kept before root's
    route unsets the originals), else the preparation script's
    ``__bf_hook_*`` copies, else the shell's own values; with the image PATH
    (the route's ``_BENCHFLOW_IMAGE_PATH``). argv: the working folder and the
    workspace. 127 (no protected Python, or on the preparation route no
    interpreter V3 judged) is skipped with a line saying so;
    any other failure fails (PT-R16-B-4, PT-R16-R2-2).
    """
    return (
        "( _BF_HOOK_PYTHONPATH="
        '"${_BENCHFLOW_IMAGE_PYTHONPATH-${__bf_hook_pythonpath-${PYTHONPATH-}}}"; '
        "_BF_HOOK_PYTHONHOME="
        '"${_BENCHFLOW_IMAGE_PYTHONHOME-${__bf_hook_pythonhome-${PYTHONHOME-}}}"; '
        "_BF_HOOK_PYTHONUSERBASE="
        '"${_BENCHFLOW_IMAGE_PYTHONUSERBASE-${PYTHONUSERBASE-}}"; '
        '_BF_HOOK_IMAGE_PATH="${_BENCHFLOW_IMAGE_PATH-}"; '
        "export _BF_HOOK_PYTHONPATH _BF_HOOK_PYTHONHOME _BF_HOOK_PYTHONUSERBASE "
        "_BF_HOOK_IMAGE_PATH; "
        # Without a kit, the interpreter V3 judged, which the preparation
        # route names ($__bf_root_python); none: 127, skipped.
        + root_python_sh(
            _CLEANUP_SITE_SCRIPT,
            raw_args='"$PWD" '
            + shlex.quote(workspace or "")
            + (
                " 1"
                + "".join(
                    " " + shlex.quote(a)
                    for a in workspace_aliases
                    if os.path.normpath(a) != os.path.normpath(workspace or "")
                )
                if agent_user
                else ""
            ),
            fallback='"${__bf_root_python:-/nonexistent/benchflow-judged-python3}"',
        )
        + ' ); bf_rc=$?; [ "$bf_rc" = 0 ] || { [ "$bf_rc" = 127 ] && '
        + f'echo "{NO_PROTECTED_PYTHON_SKIP}" >&2; }} || '
        + '{ echo "benchflow: removing Python start-up hooks failed (rc=$bf_rc)" >&2; false; }'
    )


# Backward-compat: the all-defaults cleanup command.
CLEANUP_CMD = _build_cleanup_cmd()


# Verifier-hardening payloads
#
# Each agent-tampering neutralization step renders a shell/python payload. The
# static program bodies live as module-level ``_*_SCRIPT`` constants (matching
# _DISCOVER_PYTEST_PLUGINS_SCRIPT / _TRUSTED_PATH_EXTRAS_SCRIPT); the few that
# splice a workspace path into the shell are rendered by ``_*_cmd`` builders
# (matching _legacy_root_tool_link_cmd / _trusted_path_extras_cmd). The emitted
# strings are byte-identical to the historical inline f-strings — the rendered
# payloads, not the call sites, are the security contract under test.

# Full workspace restore from /testbed_verify (the pre-agent canonical copy) so
# ALL source files — not just build-config files — are reset to their pre-agent
# state before freezing. Closes the editable-install source modification vector
# (F2): agent rewrites /testbed/src/pkg/x.py during its phase; this step
# restores x.py to canonical before verify. rsync is tried first; cp -a via
# copytree is the fallback, on the interpreter the freeze uses, never a
# workspace venv's: the agent's .pth files are still there (V8-R1-B-1).
_RESTORE_WORKSPACE_PY = (
    "import shutil,sys; "
    'shutil.copytree("/testbed_verify",sys.argv[1],dirs_exist_ok=True)'
)

# Purge agent-planted external symlinks, not the image's own venv interpreters.
# /testbed_verify is the pre-agent cp -a copy, locked before agent launch. Read
# its link text without following links (including directory links): the same
# relative path AND target must be present and still resolve as recorded. A missing or unlocked baseline
# grants no exceptions. Any other link, including a recorded one whose text the
# agent changed, goes through the realpath check: in-tree links remain allowed
# as before (OTP fixtures, retargeted version links, etc.); workspace ownership
# (a file with another name keeps its owner) + CLEANUP_CMD are their remaining
# defenses.
# real() raises on a loop and otherwise is the non-strict realpath, so a
# dangling in-tree link (a broken-link fixture, a link to a file the verifier's
# build creates) is kept as before (V8-R1-A-2); realpath(strict=) needs 3.10
# and an old image python3 runs this. os.readlink takes a str: 3.5's rejects
# a Path, and a failed purge now fails hardening (V8-R1-B-1).
# An interpreter started through a kept link reads startup files beside it:
# pyvenv.cfg there and one directory up (its home key picks the standard
# library) and <name>._pth. The exemption holds only while each of these is
# absent in both trees or the same regular file in both (V8-R1-A-1).
# Unchanged text and target say nothing about who can write the target: a
# venv built on a Python under /home/<sandbox_user> (setup_sandbox_user
# chowns it) would keep a link to an interpreter whose standard library the
# agent wrote (V8-R1-A-1, round 1 at b449a20b). So a kept link that resolves
# outside the workspace, and the `home` of each pyvenv.cfg above, must be
# outside argv[2:] (the runtime prefixes /tmp, /var/tmp, /logs, /testbed) and
# pass this interpreter rule (the verifier starts the kept link's target as
# root; which Python root itself runs is PT's): the target, its directory
# and that directory's parent (for `home`: the directory and its parent)
# owned by root (or the user running this) and not group- or
# world-writable, and every directory above them owned the same way and not
# group- or world-writable unless sticky (a sticky directory's entries
# cannot be replaced by others). A home directory that is not ours fails the
# ownership half.
_PURGE_EXTERNAL_SYMLINKS_SCRIPT = """\
import errno, os, sys, pathlib, stat
root = pathlib.Path(sys.argv[1]).resolve()
root_str = str(root) + os.sep
baseline = pathlib.Path('/testbed_verify')
blocked = set()
for prefix in sys.argv[2:]:
    blocked.update({prefix.rstrip('/') or '/', os.path.realpath(prefix)})
owners = (0, os.geteuid())
def in_workspace(path):
    return path == str(root) or (path + os.sep).startswith(root_str)
def trusted(path, strict):
    if not path.startswith('/'):
        return False
    path = os.path.realpath(path)
    if in_workspace(path):
        return True
    if any(path == b or path.startswith(b.rstrip('/') + '/') for b in blocked):
        return False
    chain = [path]
    while chain[-1] != os.path.dirname(chain[-1]):
        chain.append(os.path.dirname(chain[-1]))
    for index, entry in enumerate(chain):
        st = os.lstat(entry)
        if st.st_uid not in owners:
            return False
        if st.st_mode & 0o022 and (index < strict or not st.st_mode & stat.S_ISVTX):
            return False
    return True
def homes_trusted(rel):
    names = [rel.parent / 'pyvenv.cfg']
    if rel.parent.parts:
        names.append(rel.parent.parent / 'pyvenv.cfg')
    for name in names:
        data = content(root, name.parts)
        for line in (data or b'').decode('utf-8', 'surrogateescape').splitlines():
            key, sep, value = line.partition('=')
            if sep and key.strip().lower() == 'home' and not trusted(value.strip(), 2):
                return False
    return True
def real(path):
    try:
        os.stat(path)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise
    return os.path.realpath(path)
def content(top, parts):
    fd = os.open(str(top), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        try:
            child = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        except FileNotFoundError:
            return None
        try:
            if not stat.S_ISREG(os.fstat(child).st_mode):
                raise ValueError(parts)
            data = []
            chunk = os.read(child, 65536)
            while chunk:
                data.append(chunk)
                chunk = os.read(child, 65536)
            return b''.join(data)
        finally:
            os.close(child)
    finally:
        os.close(fd)
def startup_files_recorded(rel):
    names = [rel.parent / 'pyvenv.cfg', rel.parent / (rel.name + '._pth')]
    if rel.parent.parts:
        names.append(rel.parent.parent / 'pyvenv.cfg')
    return all(content(root, n.parts) == content(baseline, n.parts) for n in names)
original = {}
try:
    st = baseline.lstat()
    if stat.S_ISDIR(st.st_mode) and st.st_uid == os.geteuid() and not st.st_mode & 0o077:
        original = {p.relative_to(baseline): os.readlink(str(p))
                    for p in baseline.rglob('*') if p.is_symlink()}
except OSError:
    original = {}
for p in root.rglob('*'):
    if not p.is_symlink():
        continue
    previous = original.get(p.relative_to(root))
    if previous is not None and previous == os.readlink(str(p)):
        try:
            recorded = pathlib.Path(real(str(baseline / p.relative_to(root))))
            if recorded == baseline or baseline in recorded.parents:
                recorded = root / recorded.relative_to(baseline)
            resolved = real(str(p))
            if (pathlib.Path(resolved) == recorded
                    and startup_files_recorded(p.relative_to(root))
                    and trusted(resolved, 3)
                    and homes_trusted(p.relative_to(root))):
                continue
        except (OSError, RuntimeError, ValueError):
            pass
    try:
        target = real(str(p))
    except (OSError, RuntimeError):
        p.unlink()
        continue
    if target != str(root) and not (target + os.sep).startswith(root_str):
        p.unlink()
"""

# Workspace venv startup hooks run before the verifier's first Python command
# (V8-RC1-B-1). A site dir is a workspace directory named
# site-packages or dist-packages, or the in-tree target of a link with that
# name. Only a site dir's own *.pth files run (site.addsitedir), so a .pth
# deeper down, such as a package's model weights, is data and stays; a
# sitecustomize or usercustomize module (source, bytecode, extension or
# package) is removed anywhere in a site tree, since an image .pth can put a
# subdirectory on sys.path.
#
# A .pth runs code only through its ``import`` lines (site.addpackage execs a
# line that starts with ``import`` and treats every other line as a path to add
# to sys.path). So:
#   - A .pth with an import line is kept only when the .pth itself is recorded
#     AND every module its import lines may load from this site dir is recorded
#     too (an agent that edits the imported _virtualenv.py gets nothing)
#     (V8-R1-B-1). "May load" is every word on an import line, so
#     __import__('_distutils_hack') counts, and an entry of that name present in
#     the record but deleted by the agent fails too: the import would otherwise
#     fall through to a directory a path-only .pth added. Otherwise it is
#     deleted. A package tree is compared without its __pycache__ only when
#     that is a real directory holding only the image's own bytecode (the
#     cache purge runs first and removes cache links; this checks again, so
#     a skipped cache purge grants nothing, V8-CONF-1).
#   - A path-only .pth (no import line) runs no code of its own, so a recorded
#     one is kept, and so is a new one whose every path, computed as site
#     computes it, stays inside the workspace, which leaves an editable install
#     a correct answer created working (V8-R1-B-4). A new or changed one
#     that adds a path outside the workspace is deleted, and so is a
#     recorded one in a venv whose kept interpreter link leaves the
#     workspace: that folder (the agent's home, or an image folder holding a
#     link into it) is not checked here, and integ closes that route by
#     removing the link (V8-MERGE-3). A recorded one beside a real
#     interpreter file in the workspace (a --copies venv, a conda install)
#     stays, as on integ (V8-NARROW-1). Which of the two a venv is comes
#     from the pre-agent record, never the live tree, so an agent cannot
#     make a linked venv look like a copied one (V8-NARROW-R2-note1). A new one that adds a relative path with
#     '..', or an unrecorded file (a zip) is deleted (V8-R2-A-1), and so is
#     a new one that is not UTF-8 or starts
#     with a BOM (V8-R3-B-1). Non-ASCII UTF-8 is kept, so an editable install
#     under /app/café still imports (V8-CONF-3).
#   - Each .pth is judged under every spelling an interpreter may read: UTF-8
#     with the BOM dropped (3.12+), UTF-8 with it kept and the locale's
#     encoding (3.11 and older, and 3.12+'s fallback), each split as
#     str.splitlines and at newlines only. Every path any spelling adds gets
#     the checks above, the import lines of every spelling get the import
#     rule, and the sitecustomize/usercustomize modules in the in-workspace
#     directories a kept .pth adds to sys.path are purged, so no startup hook
#     runs through the added path. A recorded .pth whose BOM-kept first line
#     names an unrecorded file is deleted (V8-CONF-2).
# A .pth that is a symlink or not a regular file is deleted without being read.
# A sitecustomize/usercustomize module and a recorded .pth tree are kept only
# when the root-only pre-agent record holds the same regular file, or the same
# tree of them, at the same path; nothing is read through a link. The rest of
# the venv's library code remains agent code, like workspace source, and is
# neither restored nor trusted. Python 3.6+: an old image interpreter may run
# this.
_PURGE_WORKSPACE_PYTHON_HOOKS_SCRIPT = """\
import hashlib, json, locale, os, pathlib, re, shutil, stat, sys
root = pathlib.Path(sys.argv[1]).resolve()
baseline = pathlib.Path('/testbed_verify')
SITES = ('site-packages', 'dist-packages')
MODULES = ('sitecustomize', 'usercustomize')
ENCODINGS = ['utf-8-sig', 'utf-8']
for get_encoding in (lambda: locale.getpreferredencoding(False),
                     lambda: locale.nl_langinfo(locale.CODESET)):
    try:
        ENCODINGS.append(get_encoding())
    except Exception:
        pass
def summary(top, parts):
    fd = os.open(str(top), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        child = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            mode = os.fstat(child).st_mode
            if stat.S_ISDIR(mode):
                return 'dir', tuple(sorted(os.listdir(child)))
            if not stat.S_ISREG(mode):
                return None
            digest = hashlib.sha256()
            chunk = os.read(child, 65536)
            while chunk:
                digest.update(chunk)
                chunk = os.read(child, 65536)
            return 'file', digest.digest()
        finally:
            os.close(child)
    finally:
        os.close(fd)
trusted = False
try:
    st = baseline.lstat()
    trusted = stat.S_ISDIR(st.st_mode) and st.st_uid == os.geteuid() and not st.st_mode & 0o077
except OSError:
    pass
def same_file(parts):
    mine = summary(root, parts)
    return mine is not None and mine[0] == 'file' and mine == summary(baseline, parts)
def without_caches(top, parts):
    found = summary(top, parts)
    if found is None or found[0] != 'dir':
        return found
    if '__pycache__' in found[1]:
        cache_parts = parts + ('__pycache__',)
        cache = summary(top, cache_parts)
        # Only a real cache containing image files may be ignored. Check
        # again here so a failed/omitted cache purge grants no exemption.
        if cache is not None and cache[0] == 'dir' and (
                top == baseline or all(same_file(cache_parts + (name,)) for name in cache[1])):
            return 'dir', tuple(name for name in found[1] if name != '__pycache__')
    return found
def recorded(parts):
    if not trusted:
        return False
    try:
        mine = without_caches(root, parts)
        if mine is None or mine != without_caches(baseline, parts):
            return False
    except OSError:
        return False
    return mine[0] == 'file' or all(recorded(parts + (name,)) for name in mine[1])
def inside(path):
    return path == root or root in path.parents
def read_nofollow(path):
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        data = []
        chunk = os.read(fd, 65536)
        while chunk:
            data.append(chunk)
            chunk = os.read(fd, 65536)
    finally:
        os.close(fd)
    raw = b''.join(data)
    # Python 3.12+ decodes a .pth as UTF-8 with a leading BOM dropped and
    # falls back to the locale's encoding; 3.11 and older decode by the
    # locale (UTF-8 in UTF-8 mode) and keep a BOM in the first line. Every
    # spelling that decodes is checked, so a path one interpreter adds is
    # never one the purge did not look at (V8-R3-B-1, V8-CONF-2). A decoding
    # that fails adds no path in that interpreter. This process runs in the
    # image's locale, as the verifier does.
    texts = set()
    for encoding in ENCODINGS:
        try:
            texts.add(raw.decode(encoding))
        except (UnicodeDecodeError, LookupError):
            pass
    # A new .pth must be UTF-8 without a BOM: no installer writes another
    # form, and the BOM and locale spellings then all name the same paths.
    plain_utf8 = not raw.startswith(b'\\xef\\xbb\\xbf')
    try:
        raw.decode('utf-8')
    except UnicodeDecodeError:
        plain_utf8 = False
    return texts, plain_utf8
def is_import(line):
    return line.startswith('import ') or line.startswith('import\\t')
def listing(top, parts):
    try:
        found = summary(top, parts)
    except OSError:
        return None
    if found is None or found[0] != 'dir':
        return None
    return found[1]
def imports_recorded(site_dir, lines):
    names = set()
    for line in lines:
        if is_import(line):
            names.update(re.findall(r'\\w+', line))
    rel = site_dir.relative_to(root).parts
    mine = listing(root, rel)
    if mine is None:
        return False
    entries = set(mine) | set(listing(baseline, rel) or ())
    for entry in entries:
        if entry.split('.', 1)[0] in names and not recorded(rel + (entry,)):
            return False
    return True
def purge_hooks_in(directory):
    try:
        entries = os.listdir(str(directory))
    except OSError:
        return
    for entry in entries:
        if entry.split('.', 1)[0] not in MODULES:
            continue
        if not (entry in MODULES or entry.endswith(('.py', '.pyc', '.so'))):
            continue
        hook = directory / entry
        if recorded(hook.relative_to(root).parts):
            continue
        try:
            if hook.is_dir() and not hook.is_symlink():
                shutil.rmtree(str(hook))
            else:
                hook.unlink()
        except OSError:
            pass
def path_lines(text):
    # Python 3.13's site splits a .pth as str.splitlines does, 3.12 only at
    # \\n and \\r, so a path line is checked under both splits.
    for line in text.splitlines() + re.split('\\r\\n|\\r|\\n', text):
        if not line.startswith('#') and line.strip() and not is_import(line):
            yield line.rstrip()
def recorded_realpath(path):
    # The file path resolves to in the pre-agent image: every part inside the
    # workspace is read from the root-only baseline copy, never the live
    # tree, so a link the agent replaced, removed or added there changes
    # nothing. Parts outside the workspace resolve as they are. Lexical
    # parts are taken in order (a resolved prefix has no link in it).
    todo = list(path.parts[1:])
    cur = pathlib.Path('/')
    hops = 0
    while todo:
        name = todo.pop(0)
        if name in ('', '.'):
            continue
        if name == '..':
            cur = cur.parent
            continue
        nxt = cur / name
        probe = baseline.joinpath(*nxt.relative_to(root).parts) if inside(nxt) else nxt
        if os.path.islink(str(probe)):
            hops += 1
            if hops > 40:
                raise OSError('link loop')
            target = pathlib.PurePosixPath(os.readlink(str(probe)))
            if target.is_absolute():
                cur = pathlib.Path('/')
                todo = list(target.parts[1:]) + todo
            else:
                todo = list(target.parts) + todo
            continue
        cur = nxt
    return cur
def recorded_entries(folder):
    # The entries of folder as the pre-agent image has it, each as (name,
    # the path to test for a link, its path in the image). The folder is
    # resolved through the record first, so a venv whose bin/ is itself a
    # link is listed where that link leads, as integ's listdir does
    # (V8-NOTE1 review A L-1). Inside the workspace the baseline copy is
    # read; outside it, the folder as it is.
    try:
        where = recorded_realpath(folder)
    except OSError:
        return ()
    if not inside(where):
        try:
            names = os.listdir(str(where))
        except OSError:
            return ()
        return [(name, where / name, where / name) for name in names]
    parts = where.relative_to(root).parts
    if not parts:
        # A bin/ linked to the workspace root lists the copy's own root
        # (V8-NOTE1 r2 review R2-L1); the copy is a trusted real folder.
        try:
            names = os.listdir(str(baseline))
        except OSError:
            return ()
    else:
        names = listing(baseline, parts) or ()
    return [(name, baseline.joinpath(*(parts + (name,))), where / name)
            for name in names]
def kept_link_leaves(site_dir):
    # Whether site_dir belongs to a venv whose interpreter link the freeze
    # kept and which leaves the workspace: the venv's pyvenv.cfg in a folder
    # above it, and a bin/python* there that is a link resolving outside.
    # That venv is the route integ closes by removing the link. A venv made
    # with --copies, or a conda install, has a real interpreter file in the
    # workspace that integ runs with its .pth files intact (V8-NARROW-1).
    # Only a recorded .pth asks, so the baseline is trusted, and the answer
    # comes from it, never from the live tree: an agent that plants a
    # pyvenv.cfg between the site folder and the venv, or replaces the
    # venv's link with a copy of the interpreter, makes no linked venv look
    # like a copied one (V8-NARROW-R2-note1).
    for env in site_dir.parents:
        if env == root or not inside(env):
            return False
        rel = env.relative_to(root).parts
        if os.path.lexists(str(baseline.joinpath(*(rel + ('pyvenv.cfg',))))):
            for folder in ('bin', 'Scripts'):
                for name, probe, path in recorded_entries(env / folder):
                    if name.startswith('python') and os.path.islink(str(probe)) and not inside(
                            recorded_realpath(path)):
                        return True
            return False
    return False
def path_only_kept(site_dir, text, in_record):
    # Each line names the path site.addpackage adds, computed as it does: the
    # line without trailing whitespace, joined to the site dir and made
    # absolute lexically (os.path.abspath, no link resolved); Python then
    # reads that path's realpath. The caller passes each decoded spelling
    # of the .pth. A .pth not in the record is kept only when every path is
    # inside the workspace both lexically and resolved, is absolute or has
    # no '..' part (site joins a relative line to the site dir as the
    # interpreter spells it, which may run through a link), and is a
    # directory, a missing path or a recorded file (a zip on sys.path can hold
    # a sitecustomize; this holds for a recorded .pth too, V8-CONF-2). A kept
    # .pth has the hook modules removed from the in-workspace directories it
    # adds.
    keep = True
    for line in path_lines(text):
        try:
            lexical = pathlib.Path(os.path.abspath(os.path.join(str(site_dir), line)))
            resolved = pathlib.Path(os.path.realpath(str(lexical)))
            if not (inside(lexical) and inside(resolved)):
                # Outside the workspace: a new or changed .pth goes. A
                # recorded one stays, except in a venv whose kept interpreter
                # link leaves the workspace: that interpreter would put a
                # folder on sys.path that nothing here checks (a sitecustomize
                # the agent planted there, or one linked to its home), a route
                # integ closes by removing that link (V8-MERGE-3, V8-b172 sol
                # FIX-3 / astra FIX-4). A real interpreter file in the
                # workspace is integ's case and keeps its recorded .pth
                # (V8-NARROW-1).
                keep = keep and in_record and not kept_link_leaves(site_dir)
                continue
            if not in_record and not line.startswith('/') and '..' in line.split('/'):
                keep = False
                continue
            if resolved.is_dir():
                purge_hooks_in(resolved)
            elif os.path.lexists(str(resolved)) and not recorded(
                    resolved.relative_to(root).parts):
                keep = False
        except (OSError, ValueError):
            keep = False
    return keep
walked = list(os.walk(str(root), followlinks=False))
sites = set()
for directory, dirs, files in walked:
    here = pathlib.Path(directory)
    if here != root and here.name in SITES:
        sites.add(here)
    for name in dirs + files:
        link = here / name
        if name in SITES and link.is_symlink():
            target = pathlib.Path(os.path.realpath(str(link)))
            if inside(target) and target.is_dir():
                sites.add(target)
# A site folder PT's pre-agent record covers (a folder a recorded
# interpreter searches, or one whose hook files it recorded) is the
# record-based clean-up's: this walk keeps only workspace venvs with no
# record (PT-R16 review B). Only a real record counts
# (the files _image_python_paths.READ loads); with none, every site stays here.
def pre_agent_folders():
    entries = []
    try:
        with open('/opt/benchflow/rt/IMAGE_PYTHON_PATHS.json') as handle:
            entries.append(json.load(handle))
    except (OSError, ValueError):
        return set()
    try:
        with open('/opt/benchflow/rt/IMAGE_PYTHON_PATHS.jsonl') as handle:
            entries += [json.loads(line) for line in handle if line.strip()]
    except (OSError, ValueError):
        pass
    out = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in ('path', 'user_site'):
            for folder in entry.get(key) or ():
                if isinstance(folder, str):
                    out.add(os.path.realpath(folder))
        for folder in entry.get('hooks') or {}:
            if isinstance(folder, str):
                out.add(os.path.realpath(folder))
    return out
covered = pre_agent_folders()
sites = {s for s in sites if os.path.realpath(str(s)) not in covered}
for directory, dirs, files in walked:
    here = pathlib.Path(directory)
    if not any(p in sites for p in (here,) + tuple(here.parents)):
        continue
    for name in dirs + files:
        module = name.split('.', 1)[0] in MODULES and (
            name in MODULES or name.endswith(('.py', '.pyc', '.so')))
        is_pth = here in sites and name in files and name.endswith('.pth')
        if not module and not is_pth:
            continue
        path = here / name
        if is_pth:
            found = None if path.is_symlink() else read_nofollow(path)
            if found is not None:
                texts, plain_utf8 = found
                in_record = recorded(path.relative_to(root).parts)
                kept = in_record or plain_utf8
                for text in texts:
                    lines = text.splitlines() + re.split('\\r\\n|\\r|\\n', text)
                    if any(is_import(line) for line in lines):
                        kept = kept and in_record and imports_recorded(here, lines)
                    # A recorded import .pth may also add paths. Apply the
                    # same archive and hook rules to every decoded spelling.
                    kept = path_only_kept(here, text, in_record) and kept
                if kept:
                    continue
        elif recorded(path.relative_to(root).parts):
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(str(path))
            else:
                path.unlink()
        except OSError:
            pass
# The test runner's own modules, added in the workspace: test.sh runs there,
# and pytest started as a module puts it first on sys.path, so a pytest.py (or a
# _pytest, pluggy, iniconfig or py module or package) the agent added would run
# as root in place of the grader and could report a pass. Each top-level one
# the root-only pre-agent copy has no entry for goes; the repository's own
# (an entry the copy has, changed or not) stays.
RUNNER = ('pytest', '_pytest', 'pluggy', 'iniconfig', 'py')
def had_before(name):
    if not trusted:
        return False
    try:
        os.lstat(str(baseline / name))
    except OSError:
        return False
    return True
try:
    top_names = os.listdir(str(root))
except OSError:
    top_names = []
for name in top_names:
    stem = name.split('.', 1)[0]
    if stem not in RUNNER or had_before(name):
        continue
    if not (name == stem or name.endswith(('.py', '.pyc', '.pyw', '.so'))):
        continue
    path = root / name
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(str(path))
        else:
            path.unlink()
    except OSError:
        pass
"""

# Purge __pycache__ trees that did not exist in the pre-agent baseline, so
# agent-planted .pyc bytecode cannot execute even if PYTHONPYCACHEPREFIX is
# bypassed. Baseline-present caches are kept so tasks whose tests diff workspace
# against /testbed_verify don't break, but only the files in them that are the
# same regular file as the baseline's: a .pyc the agent phase added or rewrote
# goes (V8-R3-A-1). Cache symlinks are unlinked without following their targets
# (V8-CONF-1). The hooks purge independently checks the remaining cache before
# excluding it from a recorded package comparison.
# ``__WSQ__`` is the shell-quoted workspace.
_PURGE_PYCACHE_CMD_TEMPLATE = (
    # Pass paths as argv, not lines: dash has no read -d, and paths can contain
    # newlines. Depth-first visits nested caches before removing their parent.
    "find __WSQ__ -depth \\( -type d -o -type l \\) -name __pycache__ -exec sh -c "
    "'workspace=${1%/}; shift; for d do "
    'if [ -h "$d" ]; then rm -f -- "$d"; continue; fi; '
    'rel=${d#"$workspace"/}; '
    'if [ -d "/testbed_verify/$rel" ] && [ ! -h "/testbed_verify/$rel" ]; then '
    'for f in "$d"/* "$d"/.[!.]* "$d"/..?*; do '
    '[ -e "$f" ] || [ -h "$f" ] || continue; '
    'b="/testbed_verify/$rel/${f##*/}"; '
    '[ -f "$f" ] && [ ! -h "$f" ] && [ -f "$b" ] && [ ! -h "$b" ] '
    '&& cmp -s "$f" "$b" || rm -rf -- "$f"; '
    "done; "
    'else rm -rf -- "$d"; fi; '
    "done' sh __WSQ__ {} + 2>/dev/null; true"
)


def _restore_workspace_cmd(workspace: str) -> str:
    """Render the full /testbed_verify → workspace restore command."""
    quoted = shlex.quote(workspace)
    return (
        "if [ -d /testbed_verify ]; then "
        f"rsync -a --delete /testbed_verify/ {quoted}/ 2>/dev/null || {{ "
        + hardening_python_cmd(_RESTORE_WORKSPACE_PY, quoted, workspace)
        + "; }; fi"
    )


def _purge_external_symlinks_cmd(workspace: str) -> str:
    """Render the external-symlink purge command for the workspace.

    It runs on integ's interpreter (_hardening_python) with ``-I -S``, and
    fails when the snippet itself fails: a purge stopped partway leaves the
    rest of the agent's links for the verifier (V8-R1-B-1, round 1 at
    b449a20b). The runtime prefixes follow the workspace: a kept image link
    may not resolve under them.
    """
    prefixes = " ".join(
        shlex.quote(prefix) for prefix in _hardening_python.RUNTIME_PATH_PREFIXES
    )
    return hardening_python_cmd(
        _PURGE_EXTERNAL_SYMLINKS_SCRIPT,
        f"{shlex.quote(workspace)} {prefixes}".rstrip(),
        workspace,
    )


def _purge_pycache_cmd(workspace: str) -> str:
    """Render the baseline-aware __pycache__ purge command for the workspace."""
    return _PURGE_PYCACHE_CMD_TEMPLATE.replace("__WSQ__", shlex.quote(workspace))


# The freeze makes the workspace's repositories root's, so root's git in test.sh
# would trust what the agent wrote there: a hook, an fsmonitor command, a
# filter, diff or merge driver the attributes name, an include, a shell alias
# for a subcommand the image lacks. In every git
# directory of the workspace, each hook and each such config entry that is not
# in root's pre-agent copy (/testbed_verify, trusted as the other purges trust
# it) goes; the repository's own stay. The config is read and written with the
# image's root-owned git, includes off; without one, a changed config file is
# put back from the copy or removed (git runs on its defaults). A commondir
# file sends git to another folder's config and hooks: one in the workspace is
# purged too; one outside it, like a .git file naming a folder outside it, goes
# where the agent ran in this sandbox, unless the copy holds the same pointer.
_PURGE_GIT_CONTROLS_SCRIPT = """\
import os, re, shutil, stat, subprocess, sys
root = sys.argv[1]
baseline = sys.argv[2] if len(sys.argv) > 2 else ''
RUNS = re.compile(
    r'(filter\\..*|diff\\.[^.]+\\..+|merge\\.[^.]+\\..+|include\\..*|includeif\\..*'
    r'|core\\.(fsmonitor|hookspath|sshcommand|gitproxy|askpass|editor|pager'
    r'|alternaterefscommand)|alias\\..*|(diff|merge)tool\\..*'
    r'|diff\\.external|sequence\\.editor|gpg\\..*program|credential\\..*|pager\\..*'
    r'|uploadpack\\..*|remote\\..+\\.(uploadpack|receivepack)|interactive\\.difffilter'
    r'|submodule\\..+\\.update)$', re.I)
DEVNULL = subprocess.DEVNULL
ENV = {'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent', 'LC_ALL': 'C',
       'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'}
trusted = False
if baseline:
    try:
        st = os.lstat(baseline)
        trusted = (stat.S_ISDIR(st.st_mode) and st.st_uid == os.geteuid()
                   and not st.st_mode & 0o077)
    except OSError:
        trusted = False
def base_of(path):
    if not trusted:
        return None
    candidate = os.path.join(baseline, os.path.relpath(path, root))
    try:
        if stat.S_ISREG(os.lstat(candidate).st_mode):
            return candidate
    except OSError:
        pass
    return None
def read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                return b''.join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)
git = None
for candidate in ('/usr/bin/git', '/bin/git', '/usr/local/bin/git'):
    try:
        st = os.stat(candidate)
    except OSError:
        continue
    if stat.S_ISREG(st.st_mode) and st.st_uid == 0 and not st.st_mode & 0o022:
        git = candidate
        break
def entries(path):
    out = subprocess.check_output(
        [git, 'config', '--file', path, '--no-includes', '--null', '--list'],
        env=ENV, stderr=DEVNULL)
    pairs = []
    for item in out.split(b'\\0'):
        if item:
            key, _, value = item.partition(b'\\n')
            pairs.append((key, value))
    return pairs
def put_back(path, base):
    os.unlink(path)
    if base is not None:
        with open(base, 'rb') as source, open(path, 'xb') as target:
            shutil.copyfileobj(source, target)
def config(path):
    try:
        st = os.lstat(path)
    except OSError:
        return
    if stat.S_ISLNK(st.st_mode):
        put_back(path, base_of(path))
        return
    if not stat.S_ISREG(st.st_mode):
        return
    base = base_of(path)
    if base is not None and read(base) == read(path):
        return
    if git is None:
        put_back(path, base)
        return
    try:
        now = entries(path)
        before = entries(base) if base is not None else []
    except (OSError, subprocess.CalledProcessError):
        put_back(path, base)
        return
    keys = []
    for key, _ in now:
        if RUNS.match(key.decode('utf-8', 'surrogateescape')) and key not in keys:
            keys.append(key)
    for key in keys:
        kept = [v for k, v in before if k == key]
        if [v for k, v in now if k == key] == kept:
            continue
        code = subprocess.call([git, 'config', '--file', path, '--unset-all', key],
                               env=ENV, stdout=DEVNULL, stderr=DEVNULL)
        if code not in (0, 5):
            raise SystemExit('git config --unset-all failed: %d' % code)
        for value in kept:
            subprocess.check_call([git, 'config', '--file', path, '--add', key, value],
                                  env=ENV, stdout=DEVNULL, stderr=DEVNULL)
def hooks(gitdir):
    folder = os.path.join(gitdir, 'hooks')
    try:
        st = os.lstat(folder)
    except OSError:
        return
    if not stat.S_ISDIR(st.st_mode):
        os.unlink(folder)
        return
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        st = os.lstat(path)
        if stat.S_ISREG(st.st_mode):
            if name.endswith('.sample'):
                continue
            base = base_of(path)
            if (base is not None and read(base) == read(path)
                    and os.lstat(base).st_mode & 0o111 == st.st_mode & 0o111):
                continue
        if stat.S_ISDIR(st.st_mode):
            shutil.rmtree(path)
        else:
            os.unlink(path)
real_root = os.path.realpath(root)
def follow(path, prefix):
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
        data = read(path)
    except OSError:
        return None
    if not data.startswith(prefix):
        return None
    target = os.fsdecode(data[len(prefix):].rstrip(b'\\r\\n'))
    target = os.path.realpath(os.path.join(os.path.dirname(path), target))
    if target == real_root or target.startswith(real_root.rstrip('/') + '/'):
        return os.path.join(root, os.path.relpath(target, real_root))
    base = base_of(path)
    if baseline and (base is None or read(base) != read(path)):
        os.unlink(path)
    return None
def repository(gitdir):
    config(os.path.join(gitdir, 'config'))
    config(os.path.join(gitdir, 'config.worktree'))
    hooks(gitdir)
for folder, dirs, files in os.walk(root):
    names = set(dirs) | set(files)
    if '.git' in files:
        follow(os.path.join(folder, '.git'), b'gitdir: ')
    if 'HEAD' in names and ('objects' in names or 'commondir' in names):
        repository(folder)
        common = follow(os.path.join(folder, 'commondir'), b'')
        if common is not None:
            repository(common)
        dirs[:] = [d for d in dirs if d not in ('objects', 'hooks', 'refs', 'logs')]
"""


def _purge_git_controls_cmd(workspace: str, *, agent_user: bool = False) -> str:
    """Render the git hook and config purge for the workspace's repositories.

    The pre-agent copy is the baseline only where the agent ran as a user
    other than root in this sandbox (*agent_user*, the conftest sweep's rule);
    elsewhere, as in a separate verifier, every such entry goes.
    """
    baseline = _VERIFIER_WORKSPACE_COPY if agent_user else ""
    return hardening_python_cmd(
        _PURGE_GIT_CONTROLS_SCRIPT,
        f"{shlex.quote(workspace)} {shlex.quote(baseline)}",
        workspace,
    )


def _purge_workspace_python_hooks_cmd(workspace: str) -> str:
    """Remove workspace venv startup hooks unless identical to the image copy.

    Fails when the snippet itself fails: an uncaught error (Python 3.11's
    recursive os.walk on an agent-made 1,100-deep tree) stopped it before it
    purged anything, and the step still passed (V8-R1-B-1, round 1 at
    b449a20b).
    """
    return hardening_python_cmd(
        _PURGE_WORKSPACE_PYTHON_HOOKS_SCRIPT,
        shlex.quote(workspace),
        workspace,
    )


# The kill before grading (K1, after Rust P1.1 and its decision P1-D8).
#
# Probe failures stay distinct from an empty observation (PR #1088, GH #1078):
# an unreadable existing /proc entry is unknown, not proof of quiescence, and
# no readable /proc at all is never quiescence.
#
# A plain walk that signals each pid it listed can be outrun: a process that
# keeps forking makes new pids after the list is taken, and one that restarts
# what was just killed brings it back. So the kill first freezes: it sends
# SIGSTOP to every live process of the sandbox user, pass after pass, until a
# pass finds none it has not stopped (a stopped process cannot fork or
# restart anything), then sends SIGKILL to every live one and looks again, in
# up to 8 rounds. SIGKILL reaches a stopped process.
#
# Before the kill rounds the frozen processes get a grace, as before K1 (which
# sent SIGTERM and waited 1 s): SIGTERM, then SIGCONT, and up to 1 s to exit,
# ending early once none is left; with none frozen there is no wait. A
# handler that saves state on SIGTERM (a trap that writes a result, a
# database that saves on shutdown) still runs, so a verifier that reads it
# scores as before (K1-R1-A-2). What is left is frozen and killed after.
#
# A walk cannot even list a pid hopper: a process that starts its next
# generation and exits at once is gone before the walk reads its status, and
# its child was born after the walk listed /proc (K1-R1-A-1). So each freeze
# and each kill is first sent with the kernel's kill(-1) as the sandbox user
# (root drops to it with setpriv, or su where there is no setpriv): the
# kernel signals every process that user may signal, other than the caller
# and PID 1, in one pass under its task-list lock, and a fork that races it
# fails while the signal is pending, so no child escapes. The walk then
# freezes and kills what kill(-1) cannot reach and is the check that nothing
# is left. A broadcast counts only when the helper proved, inside the drop,
# that it runs as the user's UID and GID, and kill(-1) itself succeeded;
# it then prints ``BF as-user``. Even ESRCH can be fabricated by seccomp,
# so the helper starts a temporary child to receive the signal when there
# are no agent processes. It kills and reaps that child before returning.
# Every failed kill(-1) is unconfirmed delivery.
# A helper that is missing, fails, does not drop or cannot
# broadcast is not trusted: the other one is tried, and when neither works the
# kill fails (exit 2) instead of falling back to the walk a hopper outruns
# . The kill must run as root; only the tests of the walk itself
# render the script with __NEED_ROOT__ 0, as a user that is not root, and then
# send no kill(-1): from any other user it would hit that user's own
# processes.
#
# Per-pid signals go only to the frozen set, before the broadcast of the same
# signal: a frozen process cannot exit, so its pid cannot be reused, but once
# kill(-1) has killed or resumed it, its pid can be freed and given to a root
# process. Killing one frozen process can still free another's
# pid before its turn (a child that set a parent-death signal dies with its
# parent, and PID 1 reaps it), so the user's own processes get their per-pid
# signals as the user, through the same helper: a pid that root's process
# took meanwhile answers EPERM. Only a process whose
# effective UID alone is the user's (real and saved UID root's, which the
# user cannot signal) gets them from root, with the walk's list-then-signal
# window.
#
# A process is the sandbox user's when its real, effective or saved UID is the
# user's. That covers everything kill(-1) as the user reaches
# (the kernel's rule for who may signal whom: a setuid program the agent
# started is the agent's) and everything ``pgrep -u`` would list, so the
# freeze, the grace, the count and the last check all see the same set.
# Root's processes (all UIDs 0) and every other user's are never signalled. A
# zombie (State Z) has no live
# writer, except a thread group whose leader exited while its other threads
# run: its pid shows State Z with Threads above 1, and it is live (Rust
# P1.1-r5-B-1). Signals sent to that pid reach the whole group.
#
# Reads /proc with shell builtins (plus ``id`` and ``sleep``) so it runs on
# busybox and debian-slim images that ship neither procps nor python3. An
# entry that disappears before its status is read has exited. Prints
# ``BF killed <n> end`` (the processes it froze and ended) on success; exits 1
# with ``Sandbox-user writers remain: <pid> (<command>) ...`` on stderr when a
# process survives every round, 2 when the state cannot be observed, the
# script is not root or no helper can signal as the user. The host trusts the
# count only with exit status 0 and exactly one such line, whole.
_KILL_PRELUDE = r"""P=__PROC__
uid=$(id -u __USER__ 2>/dev/null) || { echo 'Invalid sandbox UID' >&2; exit 2; }
case "$uid" in ''|*[!0-9]*|0) echo 'Invalid sandbox UID' >&2; exit 2 ;; esac
if [ ! -r "$P/self/status" ]; then
    echo 'Process table is unavailable' >&2
    exit 2
fi
gid=$(id -g __USER__ 2>/dev/null)
case "$gid" in ''|*[!0-9]*) gid=$uid ;; esac
me=
{ while read -r key first second rest || [ -n "$key" ]; do
    [ "$key" = Uid: ] && me=$second
    key=
done <"$P/self/status"; } 2>/dev/null
if [ "$me" != 0 ] && [ __NEED_ROOT__ != 0 ]; then
    echo 'The kill before grading is not running as root' >&2
    exit 2
fi
# The descriptor inspections in the keeping/release paths need a helper
# that can read as the user too. Signals still check each actual invocation.
as_user=
can_be() (
    check="[ \"\$(id -u)\" = $uid ] && [ \"\$(id -g)\" = $gid ]"
    case "$1" in
        setpriv) setpriv --reuid="$uid" --regid="$gid" --clear-groups -- sh -c "$check" </dev/null >/dev/null 2>&1 ;;
        su) su -s /bin/sh __USER__ -c "$check" </dev/null >/dev/null 2>&1 ;;
    esac
)
if [ "$me" = 0 ]; then
    for helper in setpriv su; do
        if command -v "$helper" >/dev/null 2>&1 && can_be "$helper"; then
            as_user=$helper
            break
        fi
    done
fi
asuser() {
    if [ "$me" != 0 ]; then
        eval "$1" >/dev/null
        return 0
    fi
    body="[ \"\$(id -u)\" = $uid ] && [ \"\$(id -g)\" = $gid ] || exit 3; $1"
    for helper in setpriv su; do
        command -v "$helper" >/dev/null 2>&1 || continue
        case "$helper" in
            setpriv) out=$(setpriv --reuid="$uid" --regid="$gid" --clear-groups -- sh -c "$body" </dev/null 2>/dev/null) ;;
            su) out=$(su -s /bin/sh __USER__ -c "$body" </dev/null 2>/dev/null) ;;
        esac
        if [ "$out" = 'BF as-user' ]; then
            as_user=$helper
            return 0
        fi
    done
    return 1
}
helpless() {
    echo "Cannot send SIG$1 as the sandbox user: neither setpriv nor su ran it as UID $uid GID $gid (install setpriv or su in the image and let root switch to that UID and GID)" >&2
    exit 2
}
everyone() {
    [ "$me" = 0 ] || return 0
    # kill(-1) runs in the helper's own shell, which it skips; in a ``$( )``
    # subshell it would stop that shell, still waiting for it. Its error, if
    # any, goes to the output root reads. A child gives even an otherwise
    # empty UID a recipient: ESRCH never has to count as success. Keep its
    # pid as our unreaped child through cleanup, and signal it as the user.
    asuser "sleep 60 </dev/null >/dev/null 2>&1 & probe=\$!; delivered=; if LC_ALL=C kill -$1 -1 2>&1; then delivered=1; fi; kill -KILL \"\$probe\" 2>/dev/null; wait \"\$probe\" 2>/dev/null; [ \"\$delivered\" = 1 ] && echo BF as-user" || helpless "$1"
}
each() {
    sig=$1
    shift
    mine=
    for pid in "$@"; do
        case "$pid" in
            r*) kill -s "$sig" "${pid#r}" 2>/dev/null ;;
            *) mine="$mine $pid" ;;
        esac
    done
    [ -n "$mine" ] || return 0
    asuser "for p in$mine; do kill -s $sig \$p 2>/dev/null; done; echo BF as-user" || helpless "$sig"
}
live() {
    for entry in "$P"/[0-9]*; do
        pid=${entry##*/}
        case "$pid" in ''|*[!0-9]*) continue ;; esac
        state=
        ruid=
        euid=
        suid=
        threads=
        if { while read -r key first second third rest || [ -n "$key" ]; do
                case "$key" in
                    State:) state=$first ;;
                    Uid:) ruid=$first euid=$second suid=$third ;;
                    Threads:) threads=$first ;;
                esac
                key=
                # Stop once the fields the test below reads are in (Threads
                # only for a zombie): the shell reads the file a byte at a
                # time, and the ~45 lines after Uid made each walk cost
                # ~15 ms in a trial's container (PERF-G).
                if [ -n "$state" ] && [ -n "$euid" ] &&
                    { [ "$state" != Z ] || [ -n "$threads" ]; }; then
                    break
                fi
            done <"$entry/status"; } 2>/dev/null &&
            [ -n "$state" ] && [ -n "$euid" ]; then
            tag=
            if [ "$ruid" != "$uid" ] && [ "$suid" != "$uid" ]; then
                [ "$euid" = "$uid" ] || continue
                tag=r
            fi
            if [ "$state" != Z ]; then
                echo "$tag$pid"
            else
                case "$threads" in ''|*[!0-9]*) ;; *) [ "$threads" -gt 1 ] && echo "$tag$pid" ;; esac
            fi
        elif [ -d "$entry" ]; then
            echo "Cannot inspect process $pid" >&2
            return 2
        fi
    done
    return 0
}
# Keeping uses numeric pids for socket and lineage lookups. Recheck their
# signal role here, then use the same UID-constrained delivery as the plain
# kill. Root's all-zero replacement is skipped; only the effective-UID-only
# population needs root's residual list/signal window.
signal_one() {
    sr= se= ssaved=
    { while read -r key a b c rest || [ -n "$key" ]; do
        [ "$key" = Uid: ] && { sr=$a; se=$b; ssaved=$c; }
        key=
    done <"$P/$2/status"; } 2>/dev/null || return 0
    if [ "$sr" = "$uid" ] || [ "$ssaved" = "$uid" ]; then
        each "$1" "$2"
    elif [ "$se" = "$uid" ]; then
        each "$1" "r$2"
    fi
    return 0
}
"""
# The kill itself, after the prelude (no service kept).
_KILL_SANDBOX_USER_PROCS_BODY = r"""freeze() {
    everyone STOP
    stopped=' '
    pass=0
    while [ "$pass" -lt 100 ]; do
        pass=$((pass + 1))
        found=$(live) || exit 2
        new=
        for pid in $found; do
            case "$stopped" in *" $pid "*) continue ;; esac
            new="$new $pid"
            stopped="$stopped$pid "
        done
        [ -z "$new" ] && break
        each STOP $new
    done
    found=$(live) || exit 2
    for pid in $found; do
        # The root/user signaling role can change during the grace without
        # changing the process. Count its numeric pid only.
        pid=${pid#r}
        case "$killed" in *" $pid "*) ;; *) killed="$killed$pid "; n=$((n + 1)) ;; esac
    done
}
killed=' '
n=0
freeze
each TERM $found
each CONT $found
everyone CONT
if [ -n "$found" ]; then
    if sleep 0.1 2>/dev/null; then tick=0.1 ticks=10; else tick=1 ticks=1; fi
    t=0
    while [ "$t" -lt "$ticks" ]; do
        found=$(live) || exit 2
        [ -z "$found" ] && break
        sleep "$tick"
        t=$((t + 1))
    done
fi
round=0
while [ "$round" -lt 8 ]; do
    round=$((round + 1))
    freeze
    each KILL $found
    everyone KILL
    found=$(live) || exit 2
    if [ -z "$found" ]; then
        echo "BF killed $n end"
        exit 0
    fi
    sleep 1
done
found=$(live) || exit 2
if [ -z "$found" ]; then
    echo "BF killed $n end"
    exit 0
fi
names=
for pid in $found; do
    pid=${pid#r}
    cmd=$(tr '\000\n' ' ?' <"$P/$pid/cmdline" 2>/dev/null | cut -c1-200)
    names="$names $pid ($cmd)"
done
echo "Sandbox-user writers remain:$names" >&2
exit 1"""
_KILL_SANDBOX_USER_PROCS_CMD_TEMPLATE = _KILL_PRELUDE + _KILL_SANDBOX_USER_PROCS_BODY

# The kill that keeps the services a task names (K2, ``x-keep-services``,
# after Rust P1.1 and its decision P1-D8), after the same prelude.
#
# kill(-1) cannot spare a process, so this kill sends no SIGKILL to
# everyone: it freezes as above (kill(-1) with SIGSTOP as the user, then the
# walk, until a pass finds none new; a frozen process cannot fork or hop),
# then kills each frozen process one by one, except those that hold a
# listening TCP socket (IPv4 or IPv6, State 0A in net/tcp and net/tcp6) on a
# kept port and none of whose uids is root's (a setuid-root holder is
# killed). Its walk matches the real, effective or saved UID, the processes
# kill(-1) as the user reaches, so a setuid program the agent started is
# killed here too rather than left frozen. Root inside a container may not
# read another user's fd links (Docker gives it no CAP_SYS_PTRACE), so the
# sandbox user looks as well, while every one of its processes is frozen;
# a pid either look finds is kept. Each look reads every thread's descriptor
# table (``task/<tid>/fd``) as well as the process's: a thread that unshared
# its table (``close_range`` with ``CLOSE_RANGE_UNSHARE``, or ``unshare``)
# holds its listener only there. With no readlink nothing is kept
# and every process is killed, as without the key.
#
# Before the kill rounds, the processes that are not kept get the SIGTERM
# grace as above (each of them, not kill(-1), gets SIGCONT); a process the
# walk did not list stays frozen. A kept process's parent, its parent's
# parent and so on, as long as each is to be ended (``lineage``), gets
# SIGKILL there instead: a launcher such as ``npm start``, ``uv run`` or pm2
# passes the SIGTERM it gets on to the service it started, which would then
# die of it when it is let go on (K2p-R3-A-1).
#
# A kept process is not left stopped while the kill ends the process that
# holds its process group to the session (K2p-R1-A-1). A service started as
# a job of a job-control shell (``cmd &`` in an interactive or ``set -m``
# bash, as a terminal tool's tmux pane runs it) is in a process group of its
# own whose parent, the shell, is in another group of the same session; when
# the last such parent dies while a member of the group is stopped, the
# kernel sends the whole group SIGHUP and then SIGCONT (an orphaned process
# group with a stopped member), so a plain ``&`` service dies and a nohup'd
# one runs on, unfrozen. And an interactive shell that is hung up (its tmux
# server ends in the grace and the terminal closes) passes SIGHUP on to its
# jobs before it exits (K2p-R2-A-1). So when a process still to be ended is
# such a parent (``anchored``: it is in another group of the same session as
# a member of a kept process's group whose parent it is), the kept processes
# and every process in a kept one's process group go on (SIGCONT, ``thaw``),
# and every other process of a kept one's session gets SIGKILL at once, with
# no SIGTERM (an interactive bash ignores SIGTERM and lives on until its
# terminal hangs it up); this happens before the SIGTERM grace and before
# each round's SIGKILL. The kill returns only from a round whose freeze finds
# nothing left but the kept processes, so they are frozen again by then, and
# their groups were orphaned while none of them was stopped. Otherwise the
# kept processes stay frozen throughout (K2p-R2-B-1: a kept master left
# running would start new workers, which do not hold its port, as fast as
# the rounds kill them). When nothing else is left (the kill before grading,
# after the turn-end kill took the rest) a kept process gets no SIGCONT at
# all, so a kept service changes nothing after the evidence capture and
# before the workspace is closed (K2-R2-A-3). A service that is a terminal's
# own command or its foreground job still dies when its terminal or shell
# goes, which the kernel hangs up. Each kept process prints
# ``BF kept <port> <pid> <command>`` (the command cut to 200 characters,
# line breaks shown as ``?``) and is left frozen, so nothing the agent left
# moves while root prepares the verifier; release_kept_services lets it go
# on just before the verifier's test command. With __WAIT__ set (the first
# kill of a trial) it first waits up to 30 s for every kept port to have a
# listener, so a service still loading is not killed for being late, and
# prints ``BF waited <tenths of a second>``.
#
# What holds a kept process is the kernel's cgroup v2 freezer, not SIGSTOP
# (K1-CONFIRM-1). A stopped process goes on at any SIGCONT, and the kernel
# makes one with no sender running: a POSIX timer the listener armed with
# SIGCONT, an F_SETSIG descriptor, a parent-death signal, a child's exit
# signal, an orphaned process group's SIGHUP and SIGCONT. No signal but
# SIGKILL takes a process out of a frozen cgroup, and a child forked there
# is born frozen. So the kill moves each kept process, as soon as it finds
# it (``freezer_hold`` after every ``keepnow``), into the leaf
# /sys/fs/cgroup/benchflow-kept, frozen before anything enters it; the
# rounds then kill the rest, and the receipt is printed only after the leaf
# reports frozen and holds every kept process that is still alive. The
# release checks the same (``freezer_assert``: frozen, and exactly the
# processes it is to let go on) before it reads anything and again right
# before it thaws the leaf, the last thing it does. SIGSTOP remains the
# kill's way of holding everything still while it looks.
#
# Root needs a cgroup v2 hierarchy it may write: on Docker, a trial whose
# task keeps services gets one (``docker-compose-kept-freezer.yaml``: a
# private cgroup namespace with writable-cgroups). ``freezer_check`` makes
# sure /sys/fs/cgroup is cgroup v2 (the mount's type in
# /proc/self/mountinfo, K1-R8-1), that this shell is at the root of its
# own cgroup namespace (so the leaf is the container's, never the host's),
# and that no folder or control file on the way is a link, is not root's
# or has write for group or others: the agent user could otherwise thaw the
# leaf, or move a process out of it (which takes write on the common
# parent's cgroup.procs). When any of that fails (another backend, a Docker
# without writable-cgroups) a kill that would keep a process fails instead
# (exit 2), and the trial is not graded; a kill that keeps nothing needs no
# freezer.
_KEPT_FREEZER = r"""C=/sys/fs/cgroup/benchflow-kept
freezer_error() {
    echo "Cannot hold the kept services in a frozen cgroup: $1" >&2
    exit 2
}
freezer_write() { printf '%s\n' "$2" >"$C/$1"; }
freezer_members() { cat "$C/cgroup.procs"; }
freezer_owned() {
    [ -e "$1" ] && [ ! -h "$1" ] || return 1
    m=$(stat -c '%u %a' "$1" 2>/dev/null) || return 1
    case "$m" in '0 '*) ;; *) return 1 ;; esac
    case "${m#* }" in *[2367]?|*[2367]) return 1 ;; esac
    return 0
}
freezer_fstype() {
    # The type of the last mount on /sys/fs/cgroup, read with the shell
    # alone: an image's stat may not know cgroup2 (BusyBox prints UNKNOWN).
    t=
    { while read -r id parent dev base mp rest; do
        [ "$mp" = /sys/fs/cgroup ] || continue
        rest=${rest#* - }
        t=${rest%% *}
    done <"$P/self/mountinfo"; } 2>/dev/null
    echo "$t"
}
freezer_check() {
    [ "$(freezer_fstype)" = cgroup2 ] || freezer_error "/sys/fs/cgroup is not cgroup v2"
    [ "$(cat "$P/self/cgroup" 2>/dev/null)" = 0::/ ] || freezer_error "not at the root of a cgroup namespace of its own"
    for f in /sys /sys/fs /sys/fs/cgroup /sys/fs/cgroup/cgroup.procs "$C" "$C/cgroup.procs" "$C/cgroup.freeze" "$C/cgroup.events"; do
        freezer_owned "$f" || freezer_error "$f is missing, a link, not root's or writable by others"
    done
}
freezer_frozen() {
    i=0
    while [ "$i" -lt 50 ]; do
        f=
        { while read -r k v; do [ "$k" = frozen ] && f=$v; done <"$C/cgroup.events"; } 2>/dev/null
        [ "$f" = 1 ] && return 0
        sleep 0.1 2>/dev/null || sleep 1
        i=$((i + 1))
    done
    freezer_error "$C did not report frozen"
}
freezer_gone() {
    st=
    th=
    { while read -r key first rest || [ -n "$key" ]; do
        case "$key" in
            State:) st=$first ;;
            Threads:) th=$first ;;
        esac
        key=
    done <"$P/$1/status"; } 2>/dev/null || return 0
    case "$st" in
        '') return 0 ;;
        Z|X) case "$th" in ''|0|1) return 0 ;; esac ;;
    esac
    return 1
}
freezer_hold() {
    [ "$kept" = ' ' ] && return 0
    mkdir "$C" 2>/dev/null || [ -d "$C" ] || freezer_error "cannot make $C"
    freezer_check
    freezer_write cgroup.freeze 1 || freezer_error "cannot freeze $C"
    for p in $kept; do
        freezer_write cgroup.procs "$p" 2>/dev/null && continue
        freezer_gone "$p" || freezer_error "cannot move $p into $C"
    done
    freezer_frozen
    members=$(freezer_members) || freezer_error "cannot read $C/cgroup.procs"
    members=" $(echo $members) "
    for p in $kept; do
        case "$members" in *" $p "*) continue ;; esac
        freezer_gone "$p" || freezer_error "$p is not in $C"
    done
    return 0
}
freezer_assert() {
    freezer_check
    [ "$(cat "$C/cgroup.freeze" 2>/dev/null)" = 1 ] || freezer_error "$C is not frozen"
    freezer_frozen
    members=$(freezer_members) || freezer_error "cannot read $C/cgroup.procs"
    for p in $mine; do
        case " $(echo $members) " in *" $p "*) ;; *) freezer_error "$p is not in $C" ;; esac
    done
    for p in $members; do
        case " $mine " in *" $p "*) ;; *) freezer_error "$p in $C is not one to release" ;; esac
    done
    return 0
}
"""

_KILL_KEEPING_BODY = r"""kp=__KEEP_PORTS__
mine() {
    population=$(live) || return 2
    for member in $population; do echo "${member#r}"; done
}
listen() {
    ki=' '
    for t in "$P/net/tcp" "$P/net/tcp6"; do
        [ -r "$t" ] || continue
        while read -r sl la ra st rest; do
            [ "$st" = 0A ] || continue
            h=${la##*:}
            case "$kp" in *" $h:"*) ;; *) continue ;; esac
            d=${kp#*" $h:"}
            d=${d%% *}
            set -- $rest
            case "$6" in ''|0|*[!0-9]*) ;; *) ki="$ki$6:$d " ;; esac
        done <"$t"
    done
    return 0
}
look='for p in "$@"; do for f in "$P/$p"/fd/* "$P/$p"/task/*/fd/*; do l=$(readlink "$f" 2>/dev/null) || continue; case "$l" in "socket:["*"]") i=${l#socket:?}; i=${i%?} ;; *) continue ;; esac; case "$ki" in *" $i:"*) d=${ki#*" $i:"}; echo "$p ${d%% *}"; break ;; esac; done; done'
holds() {
    set -- "$1"
    eval "$look"
}
look_as_user() {
    [ $# -gt 0 ] || return 0
    case "$as_user" in
        setpriv) setpriv --reuid="$uid" --regid="$gid" --clear-groups -- sh -c "P='$P' ki='$ki'; set -- $*; $look" </dev/null 2>/dev/null ;;
        su) su -s /bin/sh __USER__ -c "P='$P' ki='$ki'; set -- $*; $look" </dev/null 2>/dev/null ;;
    esac
    return 0
}
rootish() {
    { while read -r key a b c d rest || [ -n "$key" ]; do
        if [ "$key" = Uid: ]; then
            [ "$a" = 0 ] || [ "$b" = 0 ] || [ "$c" = 0 ] || [ "$d" = 0 ]
            return
        fi
        key=
    done <"$P/$1/status"; } 2>/dev/null
    return 0
}
kept=' '
keepnow() {
    listen
    [ "$ki" = ' ' ] && return 0
    seen=$(look_as_user $found)
    for pid in $found; do
        case "$kept" in *" $pid "*) continue ;; esac
        port=$(holds "$pid" | while read -r sp sport; do echo "$sport"; break; done)
        if [ -z "$port" ]; then
            port=$(printf '%s\n' "$seen" | while read -r sp sport; do [ "$sp" = "$pid" ] && { echo "$sport"; break; }; done)
        fi
        [ -n "$port" ] || continue
        rootish "$pid" && continue
        kept="$kept$pid "
        cmd=$(tr '\000\n' ' ?' <"$P/$pid/cmdline" 2>/dev/null | cut -c1-200)
        printf 'BF kept %s %s %s\n' "$port" "$pid" "$cmd"
    done
    return 0
}
others() {
    left=
    all=$(mine) || exit 2
    for pid in $all; do
        case "$kept" in *" $pid "*) ;; *) left="$left $pid" ;; esac
    done
}
freeze() {
    everyone STOP
    stopped=' '
    pass=0
    while [ "$pass" -lt 100 ]; do
        pass=$((pass + 1))
        found=$(mine) || exit 2
        new=0
        for pid in $found; do
            case "$stopped" in *" $pid "*) continue ;; esac
            new=1
            signal_one STOP "$pid"
            stopped="$stopped$pid "
        done
        [ "$new" = 0 ] && break
    done
    found=$(mine) || exit 2
}
if [ __WAIT__ = 1 ]; then
    w=0
    while :; do
        listen
        all=1
        for e in $kp; do case "$ki" in *":${e#*:} "*) ;; *) all=0 ;; esac; done
        [ "$all" = 1 ] && break
        [ "$w" -ge 300 ] && break
        if sleep 0.5 2>/dev/null; then w=$((w + 5)); else sleep 1; w=$((w + 10)); fi
    done
    echo "BF waited $w"
fi
seen_pids=' '
note() {
    for pid in $found; do
        case "$seen_pids" in *" $pid "*) ;; *) seen_pids="$seen_pids$pid " ;; esac
    done
}
statf() {
    s=
    sp=
    sg=
    ss=
    { read -r s <"$P/$1/stat" || [ -n "$s" ]; } 2>/dev/null || return 1
    s=${s##*") "}
    set -- $s
    case "$2" in ''|*[!0-9]*) return 1 ;; esac
    case "$3" in ''|*[!0-9]*|0) return 1 ;; esac
    case "$4" in ''|*[!0-9]*|0) return 1 ;; esac
    sp=$2
    sg=$3
    ss=$4
    return 0
}
anchored() {
    [ "$kept" = ' ' ] && return 1
    kg=' '
    ks=' '
    for pid in $kept; do
        statf "$pid" || continue
        kg="$kg$sg "
        ks="$ks$ss "
    done
    [ "$kg" = ' ' ] && return 1
    for pid in $found; do
        statf "$pid" || continue
        case "$kg" in *" $sg "*) ;; *) continue ;; esac
        mg=$sg
        ms=$ss
        pp=$sp
        case "$left " in *" $pp "*) ;; *) continue ;; esac
        statf "$pp" || continue
        [ "$sg" != "$mg" ] && [ "$ss" = "$ms" ] && return 0
    done
    return 1
}
thaw() {
    for pid in $found; do
        case "$kept" in *" $pid "*) signal_one CONT "$pid"; continue ;; esac
        statf "$pid" || continue
        case "$kg" in *" $sg "*) signal_one CONT "$pid" ;; esac
    done
    for pid in $left; do
        statf "$pid" || continue
        case "$ks" in *" $ss "*) signal_one KILL "$pid" ;; esac
    done
    return 0
}
ticks=
settle() {
    if [ -z "$ticks" ]; then
        if sleep 0.1 2>/dev/null; then tick=0.1 ticks=10; else tick=1 ticks=1; fi
    fi
    t=0
    while [ "$t" -lt "$ticks" ]; do
        others
        [ -z "$left" ] && return 0
        sleep "$tick"
        t=$((t + 1))
    done
    others
}
lineage() {
    anc=' '
    for pid in $kept; do
        p=$pid
        n=0
        while [ "$n" -lt 64 ]; do
            n=$((n + 1))
            statf "$p" || break
            p=$sp
            case "$left " in *" $p "*) anc="$anc$p " ;; *) break ;; esac
        done
    done
    return 0
}
freeze
note
keepnow
freezer_hold
others
if [ -n "$left" ]; then
    anchored && thaw
    lineage
    for pid in $left; do
        case "$anc" in
            *" $pid "*) signal_one KILL "$pid" ;;
            *) signal_one TERM "$pid" ;;
        esac
    done
    for pid in $left; do signal_one CONT "$pid"; done
    settle
fi
done_ok() {
    # The receipt only with the leaf frozen and holding every kept
    # process, and nothing else of the user's alive.
    freezer_hold
    others
    [ -z "$left" ] || return 1
    n=0
    for pid in $seen_pids; do
        case "$kept" in *" $pid "*) ;; *) n=$((n + 1)) ;; esac
    done
    k=0
    for pid in $kept; do k=$((k + 1)); done
    echo "BF kept count $k"
    echo "BF killed $n end"
    exit 0
}
round=0
while [ "$round" -lt 8 ]; do
    round=$((round + 1))
    freeze
    note
    keepnow
    freezer_hold
    others
    [ -z "$left" ] && done_ok
    anchored && thaw
    for pid in $found; do
        case "$kept" in *" $pid "*) continue ;; esac
        signal_one KILL "$pid"
    done
    settle
done
freeze
note
keepnow
freezer_hold
others
[ -z "$left" ] && done_ok
names=
for pid in $left; do
    cmd=$(tr '\000\n' ' ?' <"$P/$pid/cmdline" 2>/dev/null | cut -c1-200)
    names="$names $pid ($cmd)"
done
echo "Sandbox-user writers remain:$names" >&2
exit 1"""

# Let the processes the keeping kill left frozen go on (SIGCONT), each only
# while it is still a process of the sandbox user. It runs after the prelude
# (the user's ids, how root drops to the user), right before the verifier's
# first command in the sandbox, after every step of the verifier's own setup,
# and first closes again what a kept service must not change:
#
# - /logs is made root's 0700 and /logs/verifier 0755 again: on a sandbox
#   whose logs are not mounted (Daytona, remote Docker) the verifier clears
#   its output folder after the hardening, and that clear leaves it 0777
#   (K2-R1-A-1).
# - The verifier's code folders become root's alone, so no test is read.
# - A kept process that still holds a way into /logs (its working folder, an
#   open file or folder there, a file there mapped into its memory, a root
#   folder other than ``/``, or a mount or user namespace of its own) is
#   killed rather than continued, and so is one whose links neither root nor
#   the user can read: on Docker /logs/verifier is the host's folder and
#   keeps the host user's uid, which can be the agent user's, and a folder
#   held open is reached without the path through /logs (K2-R1-B-1);
#   /logs/agent holds the trajectory published for the verifier and
#   /logs/artifacts the answers it may read (K2-R2-B-2). Each prints
#   ``BF dropped <pid> <why>``.
# - Every verdict comes first, then every dropped process is killed and the
#   release waits until each has exited (no /proc entry left, or a zombie
#   with no thread left), and only then are the others continued: a
#   continued process could otherwise enter a dropped one's working folder
#   through /proc/<pid>/cwd while that one is still alive (K2-R2-B-1). A
#   dropped process that does not exit within 10 s fails the release, and
#   nothing is continued.
# - A kept pid that has exited (or is no longer the user's) prints
#   ``BF lost <pid>``. Its State line is not read otherwise: the frozen leaf,
#   not SIGSTOP, holds a kept process, and one the kill or the kernel sent
#   SIGCONT is no longer stopped (State S) while still frozen, so
#   ``BF unfrozen`` is no longer printed (K1-CONFIRM-1).
# - The leaf that holds the kept processes (``_KEPT_FREEZER``) must be
#   frozen and hold exactly the processes to release, before anything is
#   read and again right before it is thawed; otherwise the release fails
#   (exit 2) and nothing goes on. Thawing it is the last step, after the
#   SIGCONTs, which only end a job-control stop. Only root writes the leaf's
#   cgroup.freeze and only a release writes 0, so a leaf already thawed
#   means an earlier release of the trial let these processes go on (a
#   verifier retried in place): they are checked and continued again, and
#   the leaf is left as it is.
# - The holder check cannot see a descriptor in flight on a Unix socket: a
#   kept service can pass itself a descriptor of /logs/verifier or
#   /logs/agent during the turn and use it after the release, past /logs at
#   0700 (K2-R3-A-1). Permission is checked at every use of such a
#   descriptor, so /logs/agent and /logs/artifacts lose group and other
#   write, and /logs/verifier, when its owner is the agent user's uid (on
#   Docker it keeps the host user's, which can be the agent's, K2-R1-B-1),
#   becomes root's. Each prints ``BF opened <folder> <uid>:<gid> <mode>``
#   first: the folders are mounted from the host on Docker and cannot be
#   replaced, and the host writes into agent/ after grading, so
#   end_kept_services puts them back once every process of the user is
#   gone after the verifier.
# - A kept process whose standard input, output or error is a terminal
#   prints ``BF onterm <pid> <terminal>``: when the terminal tool's server
#   ended, the terminal was hung up, and every write to it fails (EIO), so
#   such a service may answer nothing (K2p-R3-A-2).
# - 0.2 s after the SIGCONT, a kept process that has exited prints
#   ``BF lost <pid>``: a signal left pending while it was frozen ended it
#   (K2p-R3-A-1).
#
# Prints ``BF released <n>``.
#
# A link the release reads as empty is one it could not read: Linux 7.0
# answers a reader without ptrace access to another user's /proc/<pid>/ns/*
# links (root in a Docker container, which has no CAP_SYS_PTRACE) with an
# empty name and status 0, not EACCES as 6.x did. Read as a namespace of
# its own, it killed every kept service on such hosts; the user's
# own look decides there, as for the links that refuse root outright.
_RELEASE_KEPT_BODY = r"""L=__LOGS__
{ __SHUT_LOGS__; } || { echo "Cannot close $L" >&2; exit 1; }
for d in __CLOSE__; do
    [ -e "$d" ] || continue
    { chown -R root:root "$d" && chmod -R go-rwx "$d"; } || { echo "Cannot close $d" >&2; exit 1; }
done
for d in "$L/agent" "$L/artifacts" "$L/verifier"; do
    [ -d "$d" ] && [ ! -h "$d" ] || continue
    m=$(stat -c '%u:%g %a' "$d") || { echo "Cannot close $d" >&2; exit 1; }
    case "$d:$m" in
        "$L/verifier:$uid:"*) ;;
        "$L/verifier:"*) continue ;;
        *) case "${m#* }" in *[2367]?|*[2367]) ;; *) continue ;; esac ;;
    esac
    printf 'BF opened %s %s\n' "$d" "$m"
    case "$d" in
        "$L/verifier") chown 0 "$d" ;;
        *) chmod go-w "$d" ;;
    esac || { echo "Cannot close $d" >&2; exit 1; }
done
# A thread may have an inspectable empty fd table (close_range with
# CLOSE_RANGE_UNSHARE). Confirm emptiness by reading the directory: an
# unmatched glob alone also occurs when procfs denies access.
hl='for p in "$@"; do
    b=; u=; nt=0
    for t in "$P/$p"/task/[0-9]*; do
        [ -d "$t" ] || continue
        nt=$((nt + 1))
        for x in "$t/cwd" "$t/root" "$t"/fd/*; do
            if [ "$x" = "$t/fd/*" ] && [ ! -L "$x" ]; then
                contents=$(ls -A "$t/fd" 2>/dev/null) && [ -z "$contents" ] || u=1
                continue
            fi
            l=$(readlink "$x" 2>/dev/null) && [ -n "$l" ] || { u=1; continue; }
            case "$x" in
                */root) [ "$l" = / ] || b=view ;;
                *) case "$l" in "$L"|"$L"/*|"$L "*) b=logs ;; esac ;;
            esac
        done
        for x in mnt user; do
            l=$(readlink "$t/ns/$x" 2>/dev/null) && [ -n "$l" ] || { u=1; continue; }
            [ "$l" = "$(readlink "$P/self/ns/$x" 2>/dev/null)" ] || b=view
        done
    done
    if [ -e "$P/$p/maps" ]; then
        { while read -r m1 m2 m3 m4 m5 mp || [ -n "$m1" ]; do
            case "$mp" in "$L"|"$L"/*) b=logs ;; esac
            m1=
        done <"$P/$p/maps"; } 2>/dev/null || u=1
    fi
    [ "$nt" -gt 0 ] || u=1
    if [ -n "$b" ]; then echo "$p $b"
    elif [ -n "$u" ]; then echo "$p unknown"
    else echo "$p ok"; fi
done'
mine=
for pid in __PIDS__; do
    eu=
    st=
    th=
    { while read -r key first second rest || [ -n "$key" ]; do
        case "$key" in
            State:) st=$first ;;
            Uid:) eu=$second ;;
            Threads:) th=$first ;;
        esac
        key=
    done <"$P/$pid/status"; } 2>/dev/null
    if [ -z "$st" ] || [ "$eu" != "$uid" ]; then
        printf 'BF lost %s\n' "$pid"
        continue
    fi
    case "$st" in
        Z|X) case "$th" in ''|0|1) printf 'BF lost %s\n' "$pid"; continue ;; esac ;;
    esac
    mine="$mine $pid"
done
held=
if [ -n "$mine" ]; then
    freezer_check
    case "$(cat "$C/cgroup.freeze" 2>/dev/null)" in
        1) held=1; freezer_assert ;;
        0) ;;
        *) freezer_error "cannot read $C/cgroup.freeze" ;;
    esac
fi
seen=
if [ -n "$mine" ]; then
    case "$as_user" in
        setpriv) seen=$(setpriv --reuid="$uid" --regid="$gid" --clear-groups -- sh -c "P='$P' L='$L'; set -- $mine; $hl" </dev/null 2>/dev/null) ;;
        su) seen=$(su -s /bin/sh __USER__ -c "P='$P' L='$L'; set -- $mine; $hl" </dev/null 2>/dev/null) ;;
    esac
fi
drop=
go=
for pid in $mine; do
    a=$(set -- "$pid"; eval "$hl")
    b=$(printf '%s\n' "$seen" | while read -r sp sr; do [ "$sp" = "$pid" ] && { echo "$sr"; break; }; done)
    case " ${a#* } $b " in
        *" logs "*) why="holds a folder or file in $L" ;;
        *" view "*) why="has a root folder or namespace of its own" ;;
        *" ok "*) why= ;;
        *) why="its working folder and open files cannot be read" ;;
    esac
    if [ -n "$why" ]; then
        drop="$drop $pid"
        printf 'BF dropped %s %s\n' "$pid" "$why"
    else
        go="$go $pid"
    fi
done
gone() {
    st=
    th=
    gu=
    { while read -r key first second rest || [ -n "$key" ]; do
        case "$key" in
            State:) st=$first ;;
            Uid:) gu=$second ;;
            Threads:) th=$first ;;
        esac
        key=
    done <"$P/$1/status"; } 2>/dev/null || return 0
    [ -n "$st" ] || return 0
    [ "$gu" = "$uid" ] || return 0
    case "$st" in Z|X) [ -z "$th" ] || [ "$th" = 1 ] || [ "$th" = 0 ] && return 0 ;; esac
    return 1
}
if [ -n "$drop" ]; then
    for pid in $drop; do signal_one KILL "$pid"; done
    if sleep 0.1 2>/dev/null; then tick=0.1 ticks=100; else tick=1 ticks=10; fi
    t=0
    while :; do
        left=
        for pid in $drop; do gone "$pid" || left="$left $pid"; done
        [ -z "$left" ] && break
        if [ "$t" -ge "$ticks" ]; then
            echo "Kept processes killed for holding /logs did not exit:$left" >&2
            exit 1
        fi
        sleep "$tick"
        t=$((t + 1))
    done
fi
tl='for p in "$@"; do for x in 0 1 2; do l=$(readlink "$P/$p/fd/$x" 2>/dev/null) || continue; case "$l" in /dev/pts/*|/dev/tty*|/dev/console) echo "$p $l"; break ;; esac; done; done'
if [ -n "$go" ]; then
    tty=$(set -- $go; eval "$tl")
    case "$as_user" in
        setpriv) tty="$tty
$(setpriv --reuid="$uid" --regid="$gid" --clear-groups -- sh -c "P='$P'; set -- $go; $tl" </dev/null 2>/dev/null)" ;;
        su) tty="$tty
$(su -s /bin/sh __USER__ -c "P='$P'; set -- $go; $tl" </dev/null 2>/dev/null)" ;;
    esac
    for pid in $go; do
        t=$(printf '%s\n' "$tty" | while read -r tp tt; do [ "$tp" = "$pid" ] && { echo "$tt"; break; }; done)
        [ -n "$t" ] && printf 'BF onterm %s %s\n' "$pid" "$t"
    done
fi
n=0
for pid in $go; do
    signal_one CONT "$pid"
    n=$((n + 1))
done
if [ -n "$held" ]; then
    mine=
    for pid in $go; do gone "$pid" || mine="$mine $pid"; done
    freezer_assert
    freezer_write cgroup.freeze 0 || freezer_error "cannot thaw $C"
fi
if [ -n "$go" ]; then
    sleep 0.2 2>/dev/null || sleep 1
    for pid in $go; do gone "$pid" && printf 'BF lost %s\n' "$pid"; done
fi
echo "BF released $n"
"""

# The count line ends in a word of its own, so a line cut short ("BF killed
# 1" of "BF killed 12 end") is never read as a count.
_KILLED_RE = re.compile(r"BF killed (\d+) end")
_KEPT_RE = re.compile(r"^BF kept (\d+) (\d+) ?(.*)$", re.MULTILINE)
_WAITED_RE = re.compile(r"^BF waited (\d+)$", re.MULTILINE)
_KEPT_COUNT_RE = re.compile(r"^BF kept count (\d+)$", re.MULTILINE)


@dataclass(frozen=True)
class KeptProcess:
    """A process the keeping kill spared: its port, pid and command line."""

    port: int
    pid: int
    command: str


@dataclass(frozen=True)
class KeepingKill:
    """What the keeping kill did."""

    killed: int
    kept: tuple[KeptProcess, ...]
    waited_sec: float


def _confirmed_kill_count(result, stdout: object) -> int:
    """Both kill paths require explicit success and one whole count receipt.

    *stdout* is the kill's output, which the caller read from a receipt."""
    # Missing evidence is not an empty process set: only an
    # integer exit status 0 with exactly one count line confirms the kill;
    # ``_exec_return_code`` would read a missing status as 0.
    return_code = next(
        (
            value
            for value in (
                getattr(result, name, None) for name in ("return_code", "exit_code")
            )
            if isinstance(value, int) and not isinstance(value, bool)
        ),
        None,
    )
    stdout = stdout if isinstance(stdout, str) else ""
    if return_code == 2 and "Invalid sandbox UID" in (
        getattr(result, "stderr", "") or ""
    ):
        raise RuntimeError(
            "Verifier hardening failed: sandbox-user process identity check "
            f"exited with rc={return_code}{_exec_failure_detail(result)}"
        )
    if return_code != 0:
        raise RuntimeError(
            "Verifier hardening failed: sandbox-user process quiescence check "
            f"exited with rc={return_code}{_exec_failure_detail(result)}"
        )
    # Every line that starts like the count must be one whole count line,
    # and there must be exactly one.
    lines = [line for line in stdout.splitlines() if line.startswith("BF killed")]
    counts = [m[1] for line in lines if (m := _KILLED_RE.fullmatch(line))]
    if len(lines) != 1 or len(counts) != 1:
        raise RuntimeError(
            "Verifier hardening failed: sandbox-user process quiescence check "
            f"printed {len(lines)} 'BF killed' lines, not 1 whole receipt "
            f"'BF killed <n> end'{_exec_failure_detail(result)}"
        )
    killed = int(counts[0])
    return killed


async def _kill_sandbox_user_procs(
    env, sandbox_user: str, *, proc_root: str = "/proc", record: bool = True
) -> int:
    """Stop the sandbox user's processes before grading and prove none is left.

    Freezes, gives a SIGTERM grace of up to 1 s, then freezes and kills (see
    the script's comment); raises :class:`RuntimeError`
    when the user is root or unknown, when ``/proc`` cannot be read, or when
    a process survives (the message names each survivor by pid and command).
    Returns how many processes were killed (0 when none was running).
    *proc_root* is for tests: a stand-in ``/proc`` folder.
    """
    from benchflow.sandbox._base import BaseSandbox

    user = shlex.quote(sandbox_user)
    # Root's PATH is the exec route's (the protected kit's once the sandbox
    # has one); this command never sets one of its own (C8, PT-C8-R3-3).
    script = (
        _KILL_SANDBOX_USER_PROCS_CMD_TEMPLATE.replace("__USER__", user)
        .replace("__PROC__", shlex.quote(proc_root))
        .replace("__NEED_ROOT__", "1")
    )
    # Agent code is still running when exec prepares this first root command.
    # No agent-writable persistent-env wrapper may run before the freeze,
    # including turn-end capture and finalization.
    with BaseSandbox.services_kept(env):
        receipt = (
            await _bfx_kill_receipt(env, sandbox_user) if proc_root == "/proc" else None
        )
        if receipt is None:
            result = await _read_exec(
                env,
                kill_prelude(env, script, sandbox_user),
                user="root",
                timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
            )
            stdout = result.stdout
        else:
            result, stdout = _BFX_ANSWERED, receipt
    killed = _confirmed_kill_count(result, stdout)
    if record:
        _kill_record.record_kill_before_grading(sandbox_user, killed)
    return killed


# bfx answered: its receipt carries the count, as the shell's stdout did.
_BFX_ANSWERED = SimpleNamespace(return_code=0, stdout="", stderr="")


async def _bfx_kill_receipt(env: Any, sandbox_user: str) -> str | None:
    """The K1 kill as ``bfx kill`` (TRUSTED-RUNTIME v2 §4): its receipt.

    None where bfx is not delivered, the sandbox has no ``bfx_call``, the
    pre-agent identity table is absent, or bfx did not start the request:
    the shell kill then runs, as before. The account's numbers are the ones
    measured before the agent started (``_identities.snapshot``). Every
    other bfx failure raises, with the shell kill's wording; a kill that may
    have run is never sent again.
    """
    if getattr(type(env), "bfx_call", None) is None:
        return None
    if getattr(env, "bfx_delivery", None) is None:
        return None
    table = vars(env).get("_benchflow_identities")
    if not isinstance(table, Identities):
        return None
    identity = resolve_identity(table, sandbox_user)
    if identity.uid == 0:
        raise RuntimeError(
            "Verifier hardening failed: sandbox-user process identity check "
            "exited with rc=2 (Invalid sandbox UID: root's processes are never "
            "killed)"
        )
    try:
        answer = await env.bfx_call(
            "kill",
            {"accounts": [{"uid": identity.uid, "gid": identity.gid}]},
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
    except (BfxUnavailable, BfxNotStarted) as exc:
        logger.debug("bfx kill did not start (%s); the shell kill runs", exc)
        return None
    except BfxError as exc:
        what = "identity" if exc.kind == "invalid-account" else "quiescence"
        raise RuntimeError(
            f"Verifier hardening failed: sandbox-user process {what} check "
            f"failed in bfx: {exc.message}"
        ) from exc
    except BfxProtocolError as exc:
        raise RuntimeError(
            "Verifier hardening failed: sandbox-user process quiescence check "
            f"gave no bfx answer: {exc}"
        ) from exc
    receipt = answer.get("receipt")
    if not isinstance(receipt, str):
        raise RuntimeError(
            "Verifier hardening failed: sandbox-user process quiescence check "
            "gave a bfx answer without a receipt"
        )
    return receipt


async def kill_switched_accounts(env) -> int:
    """Stop the processes of the accounts the task lets its agent switch to
    (``env.agent_identity_switch``, x-agent-identity-switch), as the agent
    user's are stopped; return how many it killed (K1-R11-1). Raises as
    :func:`_kill_sandbox_user_procs` does. The kill record keeps the agent
    user's counts; these are logged."""
    total = 0
    for account in getattr(env, "agent_identity_switch", ()) or ():
        killed = await _kill_sandbox_user_procs(env, account, record=False)
        if killed:
            logger.info(
                "Killed %d process(es) of %s (x-agent-identity-switch)", killed, account
            )
        total += killed
    return total


def _keeping_kill_cmd(
    sandbox_user: str, ports: "tuple[int, ...]", *, wait: bool, proc_root: str
) -> str:
    """The keeping kill's script (see ``_KILL_KEEPING_BODY``)."""
    kp = " " + " ".join(f"{p:04X}:{p}" for p in sorted(set(ports))) + " "
    return (
        (_KILL_PRELUDE + _KEPT_FREEZER + _KILL_KEEPING_BODY)
        .replace("__USER__", shlex.quote(sandbox_user))
        .replace("__PROC__", shlex.quote(proc_root))
        .replace("__NEED_ROOT__", "1")
        .replace("__KEEP_PORTS__", shlex.quote(kp))
        .replace("__WAIT__", "1" if wait else "0")
    )


async def _kill_sandbox_user_procs_keeping(
    env,
    sandbox_user: str,
    ports: "tuple[int, ...]",
    *,
    wait: bool,
    proc_root: str = "/proc",
) -> KeepingKill:
    """:func:`_kill_sandbox_user_procs`, sparing the sandbox user's listeners
    on ``ports`` (``x-keep-services``), which are left frozen.

    With no ports it is :func:`_kill_sandbox_user_procs` itself. ``wait``
    (the first kill of a trial) first waits up to 30 s for each port to have a
    listener. Raises as :func:`_kill_sandbox_user_procs` does; a kept process
    is no survivor. The frozen processes go on only through
    :func:`release_kept_services`.
    """
    from benchflow.sandbox._base import BaseSandbox

    if not ports:
        killed = await _kill_sandbox_user_procs(env, sandbox_user, proc_root=proc_root)
        return KeepingKill(killed=killed, kept=(), waited_sec=0.0)
    # PATH: the exec route's alone (C8, PT-C8-R3-3).
    with BaseSandbox.services_kept(env):
        result = await _read_exec(
            env,
            kill_prelude(
                env,
                _keeping_kill_cmd(sandbox_user, ports, wait=wait, proc_root=proc_root),
                sandbox_user,
            ),
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    killed = _confirmed_kill_count(result, result.stdout)
    count = _KEPT_COUNT_RE.search(stdout)
    waited = _WAITED_RE.search(stdout)
    kept = tuple(
        KeptProcess(port=int(m.group(1)), pid=int(m.group(2)), command=m.group(3))
        for m in _KEPT_RE.finditer(stdout)
    )
    if count is None or int(count.group(1)) != len(kept) or (wait and waited is None):
        # Lost or cut output is not a kill that kept nothing: the services
        # would stay frozen through grading while the record said no
        # listener was there.
        raise RuntimeError(
            "Verifier hardening failed: the keeping kill exited 0 without a "
            f"complete receipt{_exec_failure_detail(result)}"
        )
    _kill_record.record_kill_before_grading(sandbox_user, killed)
    return KeepingKill(
        killed=killed,
        kept=kept,
        waited_sec=int(waited.group(1)) / 10 if waited else 0.0,
    )


# While services are kept: /logs is root's alone, so no kept service reaches
# /logs/verifier, /logs/agent or /logs/artifacts by path, and /logs/verifier
# loses the write for others it was given at start. test.sh runs as root (a
# trial whose verifier runs as another user keeps no service: trial_keep).
# The host reads its mounted copy of /logs/verifier directly, so its owner
# stays, and that owner can be the agent user's uid: a kept process that
# holds the folder open could still write there, so the release kills any
# that does (release_kept_services), and closes /logs again after the
# verifier's own clear.
#
# What is in /logs/agent (the trajectory published for the verifier) and
# /logs/artifacts (answers a verifier may read) is replaced, so nothing a
# kept service holds or points at is what the verifier reads there:
#
# - Every regular file becomes a fresh copy, root's, with its content, mode
#   (without group or other write, setuid or setgid) and mtime: a new inode.
#   A descriptor a kept service opened before the kill, through a hard link
#   in its home or on a name since deleted, still writes to the old inode,
#   which /logs no longer names (K2-R3-B-2); a chown or chmod would not
#   revoke it.
# - Symbolic links, FIFOs and sockets are removed, never followed: a link
#   into the agent's home would let a kept service change what the verifier
#   reads through it (K2-R3-B-1), and copying what it points at would have
#   root read, as root, any path the agent names (a FIFO would hang).
# - Folders become root's and lose group and other write.
#
# The two folders themselves keep their mode: on Docker they are the host
# trial folder's, which BenchFlow on the host still writes into after
# grading (Claude Code's session log), and /logs at 0700 keeps them out of a
# kept service's reach by path. This runs only while services are kept; with
# nothing kept no agent process is alive after the kill to use a link or a
# descriptor.
_FRESH_COPY = (
    "for f; do "
    '[ -f "$f" ] && [ ! -h "$f" ] || continue; '
    't=$(mktemp "${f%/*}/.bf-fresh.XXXXXX") || exit 1; '
    '{ cat "$f" >"$t" && chmod "$(stat -c %a "$f")" "$t" && '
    'chmod go-w,ug-s "$t" && touch -r "$f" "$t" && mv -f "$t" "$f"; } || '
    '{ rm -f "$t"; exit 1; }; '
    "done"
)


def _close_logs_cmd(logs: str = "/logs") -> str:
    """Close /logs (``logs``) to everything but root (see above)."""
    q = shlex.quote
    return (
        f"chown root:root {q(logs)} && chmod 700 {q(logs)} && "
        f"chmod 755 {q(logs + '/verifier')} && "
        f"for d in {q(logs + '/agent')} {q(logs + '/artifacts')}; do "
        '[ -d "$d" ] || continue; '
        'find "$d" -xdev -mindepth 1 ! -type d ! -type f -exec rm -f {} + && '
        'find "$d" -xdev -type d -exec chown root:root {} + && '
        r'find "$d" -xdev -mindepth 1 -type d \( -perm -0020 -o -perm -0002 \) '
        "-exec chmod go-w {} + && "
        f'find "$d" -xdev -type f -exec sh -c {q(_FRESH_COPY)} sh {{}} + || '
        '{ echo "Cannot close $d" >&2; exit 1; }; '
        "done"
    )


_CLOSE_LOGS_CMD = _close_logs_cmd()

_WORKSPACE_ANCESTRY_SCRIPT = (
    Path(__file__).with_name("_workspace_ancestry.py").read_text()
)
_WORKSPACE_OWNER_SCRIPT = Path(__file__).with_name("_workspace_owner.py").read_text()


def _close_workspace_cmd(workspace: str) -> str:
    """Close the workspace to the kept services, after the freeze.

    The freeze leaves a regular file with another name as it is (I5), so one
    the agent user owned is still theirs; a kept service could write it. Each
    regular file not root's is replaced at its workspace name by a root-owned
    copy (the folders are root's now), and its other names keep the old file.
    Group and other write go from everything else, and anything else not
    root's fails the step (_workspace_owner.py "close"). It needs root's
    Python, which kept services already require.
    """
    return hardening_python_cmd(
        _WORKSPACE_OWNER_SCRIPT,
        "close " + shlex.quote(workspace),
        workspace,
        missing="echo 'closing the workspace needs root Python' >&2; exit 1",
    )


def _at_physical_workspace(workspace: str, command: str) -> str:
    """Resolve the starting root, while recursive tools retain no-follow semantics."""
    # Keep trailing newlines in paths through command substitution. Resolve
    # only the root: find -L or chown -RH would follow nested submission links.
    return (
        f"bf_workspace=$(cd -P {shlex.quote(workspace)} && "
        "printf '%s.' \"$PWD\") || exit 1; "
        "bf_workspace=${bf_workspace%.}; " + command
    )


def _record_kept(
    keep: "KeptServices",
    outcome: KeepingKill,
    *,
    turn_end: "dict[str, Any] | None" = None,
) -> None:
    """Write the kept_services row of the kill's record (and log it).

    ``turn_end`` is what the trial's turn-end kill did, when it ran: it goes
    in the record as ``kept_services_turn_end``, and a pid it kept that this
    kill does not keep is listed under ``kept_services_lost_before_grading``,
    so a service lost between the two kills stays named."""
    rows = [
        {
            "port": k.port,
            "pid": k.pid,
            "command": k.command[:200],
            "source": keep.describe_source(),
        }
        for k in outcome.kept
    ]
    missing = sorted({*keep.ports} - {k.port for k in outcome.kept})
    _kill_record.record_kept_services(
        rows,
        ports=list(keep.ports),
        source=keep.describe_source(),
        no_listener=missing,
        waited_sec=outcome.waited_sec,
    )
    if turn_end is not None:
        now = {k.pid for k in outcome.kept}
        _kill_record.record_kept_services_turn_end(
            turn_end,
            lost=[
                row["pid"] for row in turn_end.get("kept", ()) if row["pid"] not in now
            ],
        )
    for row in rows:
        logger.info(
            "Kept service: port %s pid %s `%s`; names from %s",
            row["port"],
            row["pid"],
            row["command"],
            row["source"],
        )
    if missing:
        logger.info(
            "No listener of the agent user's on kept port %s",
            ", ".join(str(p) for p in missing),
        )


@dataclass(frozen=True)
class KeptRelease:
    """What the release of the kept services did: how many went on, the
    pids it killed instead, each with why, and the kept pids that had exited
    (``lost``). ``unfrozen`` stays empty since the frozen cgroup holds kept
    processes (K1-CONFIRM-1); it was the kept pids no longer stopped."""

    released: int
    dropped: tuple[tuple[int, str], ...] = ()
    lost: tuple[int, ...] = ()
    unfrozen: tuple[tuple[int, str], ...] = ()
    # Folders the release closed further, each with its owner and mode
    # before (``uid:gid``, octal), for end_kept_services to put back.
    opened: tuple[tuple[str, str, str], ...] = ()
    # Kept pids whose standard streams are a terminal, with the terminal.
    on_terminal: tuple[tuple[int, str], ...] = ()


_DROPPED_RE = re.compile(r"^BF dropped (\d+) (.*)$", re.MULTILINE)
_LOST_RE = re.compile(r"^BF lost (\d+)$", re.MULTILINE)
_UNFROZEN_RE = re.compile(r"^BF unfrozen (\d+) (\S+)$", re.MULTILINE)
_OPENED_RE = re.compile(r"^BF opened (\S+) (\d+:\d+) ([0-7]+)$", re.MULTILINE)
_ONTERM_RE = re.compile(r"^BF onterm (\d+) (.+)$", re.MULTILINE)


# The kept-service freezer, tried before any agent code runs (K1-R9-1): a
# trial that keeps services runs this in ``main`` right after the sandbox
# starts, so a sandbox that cannot hold them (a cgroup v1 host, an Engine
# without writable-cgroups, a read-only cgroup) is refused before the
# agent's turn instead of failing the kill after it. It makes the leaf and
# runs the checks the kill and the release run (cgroup v2 by mountinfo, a
# cgroup namespace of its own, every folder and control file root's with no
# write for others), then moves a throwaway child of its own into the leaf,
# freezes it, waits for the kernel to report it frozen, checks the child is
# a member, kills it (a frozen process still dies of SIGKILL) and thaws the
# leaf, which is left empty for the kill. The child is killed on every way
# out. Prints ``BF freezer ok``.
_FREEZER_PROBE_BODY = r"""probe=
trap '[ -z "$probe" ] || kill -KILL "$probe" 2>/dev/null' EXIT
mkdir "$C" 2>/dev/null || [ -d "$C" ] || freezer_error "cannot make $C"
freezer_check
sleep 60 </dev/null >/dev/null 2>&1 &
probe=$!
freezer_write cgroup.procs "$probe" || freezer_error "cannot move a process into $C"
freezer_write cgroup.freeze 1 || freezer_error "cannot freeze $C"
freezer_frozen
members=$(freezer_members) || freezer_error "cannot read $C/cgroup.procs"
case " $(echo $members) " in *" $probe "*) ;; *) freezer_error "a process moved into $C is not listed there" ;; esac
kill -KILL "$probe" 2>/dev/null
wait "$probe" 2>/dev/null
probe=
freezer_write cgroup.freeze 0 || freezer_error "cannot thaw $C"
echo "BF freezer ok"
"""


async def probe_kept_service_freezer(env, *, proc_root: str = "/proc") -> None:
    """Prove the sandbox can hold kept services frozen, before the agent runs.

    Runs ``_FREEZER_PROBE_BODY`` as root; raises
    :class:`KeptServiceFreezerUnavailable` with the sandbox's reason and what
    to do when it fails or prints no receipt. *proc_root* is for tests.
    """
    script = f"P={shlex.quote(proc_root)}\n" + _KEPT_FREEZER + _FREEZER_PROBE_BODY
    # Root's PATH is the exec route's (the protected kit's once the sandbox
    # has one); this command never sets one of its own (C8, PT-C8-R3-3).
    result = await _read_exec(
        env, script, user="root", timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC
    )
    stdout = getattr(result, "stdout", "")
    stdout = stdout if isinstance(stdout, str) else ""
    if _exec_return_code(result) == 0 and re.search(
        r"^BF freezer ok$", stdout, re.MULTILINE
    ):
        return
    raise KeptServiceFreezerUnavailable(
        "This task keeps services (x-keep-services), which BenchFlow holds in a "
        "frozen cgroup v2 of root's from the kill after the agent's turn until "
        "the verifier starts, and this sandbox cannot (the check's reason is "
        "below). No agent code ran. Run it where Docker uses cgroup v2 and knows "
        "writable-cgroups (`docker info` shows Cgroup Version: 2), or run the "
        "task without x-keep-services."
        f"{_exec_failure_detail(result)}"
    )


async def release_kept_services(
    env,
    sandbox_user: str,
    pids: "tuple[int, ...]",
    close: "tuple[str, ...]" = (),
    *,
    proc_root: str = "/proc",
    logs_dir: str = "/logs",
) -> KeptRelease:
    """Close /logs and ``close`` (the verifier's code folders) to everyone
    but root, then let the kept processes ``pids`` go on (see the script's
    comment); a kept process that still holds a way into /logs is killed
    instead and listed in ``dropped``, and the others go on only once every
    such process has exited.

    Runs just before the verifier's first command in the sandbox, so a kept
    service reads no uploaded test. A pid that has exited or is no longer
    the sandbox user's is left alone and listed in ``lost``.
    Raises :class:`RuntimeError` when a folder cannot be closed, a dropped
    process does not exit, or the frozen cgroup that holds the kept
    processes is not as the kill left it: the services stay frozen and the
    trial is not graded with its tests open to them. *proc_root* and
    *logs_dir* are for tests.
    """
    script = (
        (_KILL_PRELUDE + _KEPT_FREEZER + _RELEASE_KEPT_BODY)
        .replace("__USER__", shlex.quote(sandbox_user))
        .replace("__PROC__", shlex.quote(proc_root))
        .replace("__NEED_ROOT__", "1")
        .replace("__LOGS__", shlex.quote(logs_dir))
        .replace("__SHUT_LOGS__", _close_logs_cmd(logs_dir))
        .replace("__CLOSE__", " ".join(shlex.quote(c) for c in close) or "''")
        .replace("__PIDS__", " ".join(str(int(p)) for p in pids) or "''")
    )
    # PATH: the exec route's alone (C8, PT-C8-R3-3).
    result = await _read_exec(
        env,
        kill_prelude(env, script, sandbox_user),
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    if _exec_return_code(result) != 0:
        raise RuntimeError(
            "Verifier setup failed: releasing the kept services"
            f"{_exec_failure_detail(result)}"
        )
    stdout = getattr(result, "stdout", "")
    stdout = stdout if isinstance(stdout, str) else ""
    match = re.search(r"^BF released (\d+)$", stdout, re.MULTILINE)
    if match is None:
        # Without its receipt the release is not known to have closed the
        # folders or let anything go on.
        raise RuntimeError(
            "Verifier setup failed: releasing the kept services exited 0 "
            f"without its receipt{_exec_failure_detail(result)}"
        )
    dropped = tuple((int(m.group(1)), m.group(2)) for m in _DROPPED_RE.finditer(stdout))
    for pid, why in dropped:
        logger.warning("Kept service pid %s killed before grading: %s", pid, why)
    lost = tuple(dict.fromkeys(int(m.group(1)) for m in _LOST_RE.finditer(stdout)))
    for pid in lost:
        logger.warning("Kept service pid %s did not live to be graded", pid)
    on_terminal = tuple(
        (int(m.group(1)), m.group(2)) for m in _ONTERM_RE.finditer(stdout)
    )
    for pid, terminal in on_terminal:
        logger.warning(
            "Kept service pid %s writes to the terminal %s, which ended with the "
            "agent's terminal tool: its writes there fail, so it may not answer; "
            "start it with its output in a file",
            pid,
            terminal,
        )
    opened = tuple(
        (m.group(1), m.group(2), m.group(3)) for m in _OPENED_RE.finditer(stdout)
    )
    unfrozen = tuple(
        (int(m.group(1)), m.group(2)) for m in _UNFROZEN_RE.finditer(stdout)
    )
    for pid, state in unfrozen:
        logger.warning(
            "Kept service pid %s was not frozen before grading (state %s)", pid, state
        )
    return KeptRelease(
        released=int(match.group(1)),
        dropped=dropped,
        lost=lost,
        unfrozen=unfrozen,
        opened=opened,
        on_terminal=on_terminal,
    )


async def end_kept_services(
    env,
    sandbox_user: str,
    opened: "tuple[tuple[str, str, str], ...]" = (),
    *,
    proc_root: str = "/proc",
) -> int:
    """After the verifier: kill every process of the sandbox user (the kept
    services and anything they started), then give the folders the release
    closed further (``opened``: folder, ``uid:gid``, mode) their owner and
    mode back. Returns how many processes were killed.

    Nothing kept lives past grading, so a descriptor a kept service held is
    gone (with any in flight on its sockets) before the folders open again
    (K2-R3-A-1). Raises :class:`RuntimeError` when a process survives the
    kill; then the folders stay closed.
    """
    killed = await _kill_sandbox_user_procs(env, sandbox_user, proc_root=proc_root)
    if opened:
        q = shlex.quote
        cmd = " && ".join(
            f"chown {q(owner)} {q(path)} && chmod {q(mode)} {q(path)}"
            for path, owner, mode in opened
        )
        await _checked_exec(
            env,
            cmd,
            "Restoring the log folders after the kept services",
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
    return killed


# Verifier preparation steps
#
# Each step harden_before_verify runs before the verifier is one shell command
# run as root under its own time budget. On a remote sandbox every exec is a
# round trip (Daytona polls each one), so a backend that sets
# ``verifier_prep_script = True`` runs the steps as one script in one exec
# (_run_prep_script). The script prints a status line per step (its exit
# status and its start and end in /proc/uptime seconds), and the host turns
# those lines back into what the step-by-step path gives: the same steps run,
# in the same order, and the same error, word for word, for the step that
# failed. Other backends (Modal, Apple Container, AgentCore) and test fakes run
# the steps one exec each (_run_prep_step), exactly as before.


@dataclass(frozen=True)
class _PrepStep:
    """One verifier-preparation command and how its outcome is judged.

    With ``label``, a non-zero exit fails hardening with
    ``_checked_exec``'s error. Without it, the exit status is ignored.
    ``best_effort`` also ignores a timeout or a failed exec (the cache
    reclaim). A timeout of any other step fails hardening, as the backend's
    exec timeout always did. ``name`` is the step's own (its status line);
    ``span`` is the ``harden_before_verify`` timing sub-key its seconds add
    to, one per numbered step there (the kill and the workspace freeze are
    more than one command).
    """

    name: str
    span: str
    command: str
    timeout_sec: int
    label: str | None = None
    best_effort: bool = False
    #: Statuses a labelled step accepts (the start-up hook clean-up's 127 is
    #: no protected Python, a skip that says so: PT-R16-R2-2).
    ok_codes: tuple[int, ...] = (0,)


def _clear_verifier_dir_cmd(*, bfx: bool = False) -> str:
    """Empty /logs/verifier, keeping the folder (a remote bind mount), and
    open it to the verifier.

    With *bfx*, ``bfx clear`` empties it beneath a descriptor it holds: no
    link is followed and no mount entered. Where /logs/verifier is a link or
    missing, or bfx is not at ``/.bfx/bfx``, the command is today's, chosen
    before bfx runs; once bfx has run, any answer but success fails the step
    (today's form never runs after a bfx refusal).
    """
    if not bfx:
        return _CLEAR_VERIFIER_DIR_CMD
    from benchflow._bfx.locate import SANDBOX_PATH

    args = _json.dumps({"root": "/logs/verifier", "path": ""}, separators=(",", ":"))
    bfx_path = shlex.quote(SANDBOX_PATH)
    return (
        f"if [ ! -x {bfx_path} ] || [ -L /logs/verifier ] || "
        f"[ ! -d /logs/verifier ]; then {_CLEAR_VERIFIER_DIR_CMD}; else "
        f"__bf_o=$({bfx_path} clear --json {shlex.quote(args)}); "
        'case "$__bf_o" in '
        "*'\"ok\":true'*) chmod 777 /logs/verifier;; "
        "*) printf '%s\\n' \"$__bf_o\" >&2; false;; "
        "esac; fi"
    )


def _verifier_output_steps(*, bfx: bool = False) -> list[_PrepStep]:
    """Wipe /logs/verifier/ contents (keeping remote bind mounts), then
    prepare the legacy /app rootdir fallback."""
    return [
        _PrepStep(
            "clear_verifier_output",
            "clear_verifier_output",
            _clear_verifier_dir_cmd(bfx=bfx),
            VERIFIER_SETUP_TIMEOUT_SEC,
            "Verifier hardening failed: clearing verifier output directory",
        ),
        _PrepStep(
            "prepare_app_dir",
            "prepare_app_dir",
            _ENSURE_APP_DIR_CMD,
            VERIFIER_SETUP_TIMEOUT_SEC,
            "Verifier hardening failed: preparing /app",
        ),
    ]


# The reclaim's own budget: it is best-effort, so a slow one is cut short.
_RECLAIM_TIMEOUT_SEC = 30


def _reclaim_step(workspace: str | None) -> _PrepStep:
    """Reclaim re-downloadable cache space before the verifier installs deps.

    Heavy SkillsBench tasks (playwright + marker-pdf + HF model snapshots) can
    saturate disk-constrained sandboxes — notably Daytona's hard 10GB/sandbox
    cap — so the verifier's own ``uv``/``pip`` install then fails with ENOSPC
    ("No space left on device") and the run is lost to infra instead of a real
    score. Best-effort and result-neutral: only re-downloadable download caches
    (uv/pip/apt) are cleared — never the workspace, agent outputs, installed
    tools, or task assets — and a failure here never blocks the verifier. Runs
    on ``main`` only, consistent with the hardening policy.

    Workspace-aware AND symlink-safe (#601): a task can legitimately use /root,
    /home/<user>, or /tmp/uv-* as its workspace, and an agent can plant
    ``~/.cache -> /app`` so a naive ``rm -rf "$u/.cache/uv"`` would traverse into
    workspace/output state. ``build_reclaim_caches_cmd`` rejects symlinked
    candidates and realpath-guards every deletion against the workspace and
    /logs — this matters because restore_workspace defaults to False.
    """
    return _PrepStep(
        "reclaim_disk",
        "reclaim_disk",
        build_reclaim_caches_cmd(workspace),
        _RECLAIM_TIMEOUT_SEC,
        best_effort=True,
    )


def _bfx_in_scripts(env: Any) -> bool:
    """Whether root's scripts in this sandbox may start bfx from ``/.bfx``:
    delivered read-only at creation on a backend that runs it (Docker)."""
    return (
        getattr(env, "bfx_delivery", None) is not None
        and getattr(type(env), "bfx_call", None) is not None
    )


def _chown_workspace_cmd(workspace: str, *, bfx: bool = False) -> str:
    """Give the workspace tree to root before the verifier.

    A regular file with another name (nlink > 1) is left as it is, so a file
    the agent user owned and linked elsewhere (a start-up hook) stays theirs
    rather than becoming root's at its other name (TRUSTED-RUNTIME v2 I5).

    With *bfx*, ``bfx chown`` does it beneath descriptors it holds: no link
    is followed, no mount entered, and such a file is listed. Where the
    workspace path holds a link, the tree holds a mount, or bfx is not at
    ``/.bfx/bfx``, the command is the plain one; any other bfx failure fails
    the step.

    The plain one is root's Python walking the tree with descriptors
    (_workspace_owner.py "freeze"): every other entry, a link itself (never
    its target) and what a mount holds, is chowned, as ``chown -R`` does,
    each by its own checked syscall. Without root Python, ``find`` chowns
    and a second ``find`` fails the step if any entry it should have given
    to root is not root's (BusyBox ``find`` drops the status of every
    ``-exec ... +`` batch but the last).
    """
    shell = _at_physical_workspace(
        workspace,
        'find "$bf_workspace" -type f -links +1 -prune -o '
        "-exec chown -h root:root {} + || exit 1; "
        'bf_left=$(find "$bf_workspace" ! -user 0 '
        r"\( ! -type f -o -links 1 \) -print -quit) || exit 1; "
        '[ -z "$bf_left" ] || { echo "workspace freeze: entries are left '
        "that are not root's\" >&2; exit 1; }",
    )
    plain = hardening_python_cmd(
        _WORKSPACE_OWNER_SCRIPT,
        "freeze " + shlex.quote(workspace),
        workspace,
        missing=shell,
    )
    if not bfx:
        return plain
    from benchflow._bfx.locate import SANDBOX_PATH

    args = _json.dumps(
        {"root": workspace, "path": "", "uid": 0, "gid": 0, "recursive": True},
        separators=(",", ":"),
    )
    bfx_path = shlex.quote(SANDBOX_PATH)
    return (
        f"if [ ! -x {bfx_path} ]; then {plain}; else "
        f"__bf_o=$({bfx_path} chown --json {shlex.quote(args)}); "
        'case "$__bf_o" in '
        "*'\"ok\":true'*'\"skipped_mounts\":[]'*) ;; "
        '*\'"ok":true\'*|*\'"kind":"link"\'*|*\'"kind":"mount"\'*) '
        f"{plain};; "
        "*) printf '%s\\n' \"$__bf_o\" >&2; false;; "
        "esac; fi"
    )


def _freeze_steps(
    workspace: str, *, bfx: bool = False, git: bool = True
) -> list[_PrepStep]:
    """Purge external symlinks, stale caches and startup hooks, then chown.

    The cache purge runs before the hooks purge and leaves only the image's
    own .pyc files, and the hooks purge compares a recorded import .pth's
    package without its __pycache__, so a package that gained or changed
    bytecode in an agent-phase run (V8-R2-A-3, V8-R3-A-1) still matches the
    record. The symlink and hooks purges fail hardening when their Python
    fails (V8-R1-B-1, round 1 at b449a20b).

    chown workspace to root is belt-and-suspenders against any zombie
    sandbox-user process that survived the pkill above; a regular file with
    another name keeps its owner (see _chown_workspace_cmd).
    """
    return [
        _PrepStep(
            "purge_external_symlinks",
            "freeze_workspace",
            _purge_external_symlinks_cmd(workspace),
            VERIFIER_SETUP_TIMEOUT_SEC,
            "Verifier hardening failed: purging workspace symlinks",
        ),
        _PrepStep(
            "purge_pycache",
            "freeze_workspace",
            _purge_pycache_cmd(workspace),
            VERIFIER_SETUP_TIMEOUT_SEC,
        ),
        _PrepStep(
            "purge_workspace_python_hooks",
            "freeze_workspace",
            _purge_workspace_python_hooks_cmd(workspace),
            VERIFIER_SETUP_TIMEOUT_SEC,
            "Verifier hardening failed: purging workspace Python startup hooks",
        ),
        *(
            [
                _PrepStep(
                    "purge_git_controls",
                    "freeze_workspace",
                    _purge_git_controls_cmd(
                        workspace, agent_user=_hook_agent_user.get()
                    ),
                    VERIFIER_SETUP_TIMEOUT_SEC,
                    "Verifier hardening failed: purging workspace git hooks and config",
                )
            ]
            if git
            else []
        ),
        _PrepStep(
            "chown_workspace",
            "freeze_workspace",
            _chown_workspace_cmd(workspace, bfx=bfx),
            VERIFIER_SETUP_TIMEOUT_SEC,
            "Verifier hardening failed: freezing workspace ownership",
        ),
    ]


def _agent_not_root(sandbox_user: str | None) -> bool:
    """Whether the agent ran as a user other than root, who cannot write a
    root-only folder (the start-up hook clean-up's root-made rule, ACC-RUN2)."""
    return bool(sandbox_user) and sandbox_user not in ("root", "0")


def _cleanup_step(
    hardening: dict[str, bool],
    workspace: str | None = None,
    workspace_aliases: tuple[str, ...] = (),
) -> _PrepStep:
    """Remove injected conftest.py, sitecustomize.py and .pth files. The rootfs
    conftest walk can be slow on network-backed FS, so it gets the shared
    verifier-setup budget, not the default 10s.

    Its status is the start-up hook clean-up's (the command's last part): a
    hook that could not be removed fails hardening (PT-R16-B-4); no
    protected Python (127) exits 0 with :data:`NO_PROTECTED_PYTHON_SKIP`
    on stderr (PT-R16-R2-2)."""
    return _PrepStep(
        "cleanup_conftests",
        "cleanup_conftests",
        _build_cleanup_cmd(hardening, workspace, workspace_aliases),
        VERIFIER_SETUP_TIMEOUT_SEC,
        "Verifier hardening failed: removing Python start-up hooks",
        ok_codes=(0, 127),
    )


def _hook_cleanup_on_bfx(env: Any) -> bool:
    """Whether the hook clean-up runs on bfx in this sandbox: bfx was
    delivered at creation and the backend opens a ``bfx serve`` session. The
    shell form runs only where it was not (never after a bfx answer)."""
    return _bfx_in_scripts(env) and callable(getattr(env, "bfx_session", None))


async def _cleanup_hooks_on_bfx(
    env: Any,
    hardening: dict[str, bool],
    workspace: str | None,
    workspace_aliases: tuple[str, ...],
    label: str,
    *,
    timeout_sec: int = VERIFIER_SETUP_TIMEOUT_SEC,
    agent_user: bool | None = None,
) -> None:
    """Today's clean-up policy (:func:`_build_cleanup_cmd`) decided on the
    host and done through bfx (:mod:`benchflow.sandbox._hook_cleanup`): no
    interpreter in the sandbox is needed. One deadline, *timeout_sec*, bounds
    the whole step: the inputs exec, the session's start, every request and
    the host's decisions. Raises ``RuntimeError(label: ...)`` when it did not
    complete."""
    import asyncio

    from benchflow.sandbox import _hook_cleanup

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec

    def left() -> float:
        return max(0.001, deadline - loop.time())

    try:
        async with asyncio.timeout(timeout_sec):
            # test.sh's search-path inputs, as root's preparation route holds
            # them (the same expansions the shell form reads), from a receipt.
            result = await _read_exec(
                _route_env(env),
                _hook_cleanup.INPUTS_CMD,
                user="root",
                timeout_sec=max(1, int(left())),
            )
            if _exec_return_code(result) != 0:
                raise RuntimeError(
                    f"{label}: the search-path inputs could not be read"
                    f"{_exec_failure_detail(result)}"
                )
            inputs = _hook_cleanup.parse_inputs(getattr(result, "stdout", "") or "")
            try:
                session = env.bfx_session(timeout_sec=left())
                serve = await session.__aenter__()
            except BfxUnavailable as exc:
                raise _hook_cleanup.HookCleanupError(
                    f"bfx serve did not start: {exc}"
                ) from exc
            try:

                async def call(
                    cmd: str, args: Mapping[str, Any], timeout: float
                ) -> dict[str, Any]:
                    return await serve.call(cmd, args, timeout=min(timeout, left()))

                report = await _hook_cleanup.run(
                    call,
                    inputs,
                    cleanup_conftests=hardening.get("cleanup_conftests", True),
                    workspace=workspace,
                    workspace_aliases=workspace_aliases,
                    verifier_dir=VERIFIER_DIR,
                    deadline=deadline,
                    agent_user=(
                        _hook_agent_user.get() if agent_user is None else agent_user
                    ),
                )
            finally:
                await session.__aexit__(None, None, None)
    except TimeoutError as exc:
        raise RuntimeError(
            f"{label}: the clean-up did not finish within {timeout_sec} s"
        ) from exc
    except (_hook_cleanup.HookCleanupError, BfxError, BfxProtocolError) as exc:
        raise RuntimeError(f"{label}: {exc}") from exc
    if report.root_made:
        logger.info(
            "hook clean-up: kept %d new or changed start-up hook(s) only root "
            "could have written, e.g. %s",
            len(report.root_made),
            ", ".join(repr(p) for p in report.root_made[:5]),
        )
    if report.vanished:
        # Skipped one by one, as find skips them; the rest was walked.
        logger.warning(
            "hook clean-up: %d entries vanished while they were walked and were "
            "skipped, e.g. %s",
            len(report.vanished),
            ", ".join(repr(p) for p in report.vanished[:5]),
        )


def _add_seconds(seconds: dict[str, float] | None, span: str, took: float) -> None:
    if seconds is not None:
        seconds[span] = round(seconds.get(span, 0.0) + took, 3)


async def _run_prep_step(
    env: Any, step: _PrepStep, seconds: dict[str, float] | None = None
) -> None:
    """Run one step in its own exec, timing it on the host."""
    start = _timing_spans.clock()
    # Eligible backends also use this path when scratch allocation fails.
    # Keep the same protected launch as the batch, including after env sourcing;
    # every other backend takes the same root route (D911-V3).
    env = _route_env(env)
    try:
        if step.label is not None:
            result = await exec_raw(
                env.exec,
                step.command,
                user="root",
                timeout_sec=step.timeout_sec,
            )
            return_code = _exec_return_code(result)
            if return_code not in step.ok_codes:
                raise RuntimeError(
                    f"{step.label} exited with rc={return_code}"
                    f"{_exec_failure_detail(result)}"
                )
            if return_code == 127 or NO_PROTECTED_PYTHON_SKIP in (
                getattr(result, "stderr", "") or ""
            ):
                logger.warning(NO_PROTECTED_PYTHON_SKIP)
        elif step.best_effort:
            try:
                await exec_raw(
                    env.exec,
                    step.command,
                    user="root",
                    timeout_sec=step.timeout_sec,
                )
            except Exception:
                logger.debug("pre-verifier step %s skipped", step.name, exc_info=True)
        else:
            await exec_raw(
                env.exec,
                step.command,
                user="root",
                timeout_sec=step.timeout_sec,
            )
    finally:
        _add_seconds(seconds, step.span, _timing_spans.clock() - start)


# The script's own lines. Every step's output goes to a file in a root-only
# folder in a sticky, root-owned /tmp, never to the exec's stdout, so these lines are the only thing on
# stdout up to a FAIL line, and nothing the agent left behind can forge one.
_PREP_MARK = "BENCHFLOW-PREP"
_PREP_DIR_PREFIX = "/tmp/.benchflow_prep_"
# Added to the sum of the step budgets for the exec's own timeout, for the
# script's few lines around the steps.
_PREP_EXEC_SLACK_SEC = 30
# Daytona's Docker-in-Docker exec prints this to stderr while a command with a
# timeout of 600 s or more runs (daytona_dind._with_long_exec_heartbeat). The
# script's budget is the sum of its steps', so it can cross that line where no
# single step did; the lines are dropped so a step's error reads as before.
_DIND_HEARTBEAT_LINE = "[benchflow] Daytona DinD command still running"


def _route_env(env: Any) -> Any:
    """*env* whose execs run on the preparation route (:func:`_prep_on_path`).

    Script-capable backends launch it in their system shell (:func:`_prep_env`);
    the others (Modal, Apple Container, AgentCore, kept services) get the same
    route as text, as V3's step helpers always had.
    """
    if getattr(env, "verifier_prep_script", False) is True:
        return _prep_env(env)

    async def execute(command: str, **kwargs: Any) -> Any:
        return await exec_raw(env.exec, _prep_on_path(command), **kwargs)

    return SimpleNamespace(exec=execute)


def _prep_env(env: Any) -> Any:
    """Protected preparation exec, including the receipt and fallback.

     The backend starts its system shell before sourcing persistent env. The
     root route (:func:`_prep_on_path`, shell builtins only) follows that
     source: no env file is added when an empty-env sandbox needs low-disk
     recovery. It is the same route the step-by-step helpers take, so a
     script-capable backend finds root's tools where every other backend does
    . Wrap the complete receipt, not just its payload, so its
     encoder and the backend's EXIT cleanup also inherit the clean environment.
    """

    async def execute(command: str, **kwargs: Any) -> Any:
        return await exec_raw(
            env.exec, _prep_on_path(command), system_shell=True, **kwargs
        )

    return SimpleNamespace(exec=execute)


def _prep_script(steps: list[_PrepStep], work_dir: str) -> str:
    """Render *steps* as one POSIX sh script (see _run_prep_script).

    The whole script is a subshell: the env-file wrapper every backend puts
    before a command has its own EXIT trap, which the script's trap must not
    replace. Each step runs in a subshell of the exec's own shell (``sh`` on
    Docker and Daytona's protected route), in the
    background, with stdin from /dev/null and stdout and stderr in files,
    while a watchdog kills it and every process under it once its budget is
    spent, as Daytona's per-exec ``timeout`` did by killing the command's
    process group. The steps share the script's process group (a shell without
    a terminal cannot give a job its own), so the watchdog finds the step's
    descendants in /proc: it stops each process before listing its children,
    so none can fork past it, then kills them all. The script's own stderr goes to
    /dev/null (a shell reports a killed job there); fd 3 keeps the exec's
    stderr for the failed step's output. The few utilities the script itself
    calls and all steps run on the route's PATH (the protected kit's once the
    sandbox has one, C8), kept as the script starts, never the image PATH.
    """

    q = shlex.quote
    lines = [
        "(",
        "exec 3>&2 2>/dev/null",
        # The route pinned PATH before this text (C8); a step
        # cannot change it for the script's own tools.
        "__bf_path=$PATH",
        'PATH="$__bf_path"; export PATH',
        f"__bf_d={q(work_dir)}",
        # Mode 700 does not stop a rename by another user: only a
        # sticky parent that root owns keeps others from swapping it for a
        # link to a folder of theirs, whose output-file links root would then
        # truncate and write. Any other parent (a Dockerfile's chmod 777
        # /tmp, a /tmp the sandbox user owns or a link) means no script: the
        # host runs the steps one exec each.
        '__bf_up="${__bf_d%/*}"',
        '[ -k "$__bf_up" ] && [ -O "$__bf_up" ] && [ ! -L "$__bf_up" ] '
        f'&& PATH="$__bf_path" mkdir -m 700 "$__bf_d" '
        f"|| {{ echo {_PREP_MARK}-UNAVAILABLE; exit 0; }}",
        'trap \'PATH="$__bf_path" rm -rf "$__bf_d"\' EXIT',
        "__bf_now() { __bf_t=-; read -r __bf_t __bf_x </proc/uptime || __bf_t=-; }",
        # Kill process $1 and its descendants: field 4 of /proc/<pid>/stat
        # is the parent pid (field 2, the name, can hold spaces and ")").
        "__bf_kill_tree() {",
        "IFS=' '; __bf_q=$1; __bf_all=",
        'while [ -n "$__bf_q" ]; do',
        "set -- $__bf_q; __bf_c=$1; shift; __bf_q=$*",
        'kill -s STOP "$__bf_c" || continue',
        '__bf_all="$__bf_all $__bf_c"',
        "for __bf_f in /proc/[0-9]*/stat; do",
        'read -r __bf_l <"$__bf_f" || continue',
        "__bf_n=${__bf_l%% *}; __bf_l=${__bf_l##*') '}",
        "set -- $__bf_l",
        '[ "${2-}" = "$__bf_c" ] && __bf_q="$__bf_q $__bf_n"',
        "done",
        "done",
        '[ -z "$__bf_all" ] || kill -9 $__bf_all',
        "}",
        f"echo {_PREP_MARK}-START",
    ]
    for index, step in enumerate(steps):
        out, err, ready = (f'"$__bf_d/{kind}{index}"' for kind in "oew")
        lines += [
            # The output and watchdog-readiness files are made before the
            # step starts: a
            # redirection that fails in the step's subshell (no inode or
            # quota left in /tmp, which the agent can fill) would skip the
            # command with an exit status the script cannot tell from the
            # command's own, and an unchecked step's status is ignored. If
            # they cannot be made, the step has not started, so the host
            # runs it and the rest one exec each, as it would have without
            # the script. Opening a file that exists needs no new inode.
            f"true >{out} && true >{err} && printf 'waiting\\n' >{ready} || "
            f'{{ echo "{_PREP_MARK}-UNAVAILABLE {step.name}"; exit 0; }}',
            "__bf_now; __bf_t0=$__bf_t",
            "(",
            "trap - EXIT",
            step.command,
            f") </dev/null >{out} 2>{err} 3>&- &",
            "__bf_p=$!",
            # The watchdog: a TERM from the script once the step is done
            # stops its sleep; a sleep that runs out kills the step with
            # everything it started, ignoring the TERM from then on so that a
            # step ending mid-way cannot leave its children stopped, and
            # exits 3, which marks the step timed out (an exit status, not a
            # file, so a full /tmp cannot hide a timeout). A TERM that comes
            # before the sleep's pid is known only sets a flag, checked once
            # it is, and the sleep gets KILL: a TERM can reach the forked
            # child before it execs sleep, while it still has this trap, and
            # be lost. So a step that ends at once does not leave its sleep
            # running for the budget.
            "(",
            "trap - EXIT",
            "__bf_s=; __bf_e=",
            'trap \'__bf_e=1; [ -z "$__bf_s" ] || { kill -9 "$__bf_s"; exit 0; }\' TERM',
            f'PATH="$__bf_path" sleep {int(step.timeout_sec)} &',
            "__bf_s=$!",
            f"printf 'ready\\n' >{ready}",
            '[ -z "$__bf_e" ] || { kill -9 "$__bf_s"; exit 0; }',
            'wait "$__bf_s" || exit 0',
            'trap "" TERM; __bf_kill_tree "$__bf_p"; exit 3',
            ") </dev/null >/dev/null 2>&1 3>&- &",
            "__bf_w=$!",
            # Never cancel a watcher before its TERM handler and sleep pid
            # exist: inherited TERM dispositions can otherwise lose the
            # cancellation, or run a shell's inherited EXIT cleanup early.
            "while :; do",
            "__bf_ready=",
            f"read -r __bf_ready <{ready} || true",
            '[ "$__bf_ready" != ready ] || break',
            'kill -0 "$__bf_w" 2>/dev/null || { '
            '__bf_kill_tree "$__bf_p"; wait "$__bf_p"; exit 125; }',
            "done",
            'wait "$__bf_p"; __bf_rc=$?',
            'kill "$__bf_w"; wait "$__bf_w"; [ "$?" != 3 ] || __bf_rc=timeout',
            "__bf_now",
            f'echo "{_PREP_MARK}-STEP {step.name} $__bf_rc $__bf_t0 $__bf_t"',
        ]
        if step.best_effort:
            continue
        failed = (
            " && ".join(f'[ "$__bf_rc" != {code} ]' for code in step.ok_codes)
            if step.label is not None
            else '[ "$__bf_rc" = timeout ]'
        )
        lines += [
            f"if {failed}; then",
            f'echo "{_PREP_MARK}-FAIL {step.name}"',
            f'PATH="$__bf_path" cat {out}',
            f'PATH="$__bf_path" cat {err} >&3',
            "exit 0",
            "fi",
        ]
    lines += [f"echo {_PREP_MARK}-END", ")"]
    return "\n".join(lines)


def _uptime_seconds(start: str, end: str) -> float | None:
    try:
        return round(max(0.0, float(end) - float(start)), 2)
    except ValueError:
        return None


async def _run_prep_script(
    env: Any, steps: list[_PrepStep], seconds: dict[str, float] | None = None
) -> int:
    """Run *steps* in one exec; return how many of them it ran. Fewer than
    all, with no error raised, means the script could not start the next
    step (0: the sandbox could not start the script at all; otherwise the
    step's output files could not be made), so that step never ran and the
    caller runs it and the rest one exec each.

    The status lines say which steps ran and how each ended. A step that
    fails is reported with the error its own exec gave: for a checked step
    ``<label> exited with rc=<rc>`` and its own stdout and stderr (the rest
    of the exec's stdout after the FAIL line, and the exec's stderr, which
    only the failed step's output reaches); for a timeout, the backend's
    ``Command timed out after <budget> seconds``. On Docker, whose exec
    merges stderr into stdout, a step that wrote to both streams now shows
    its stdout before its stderr instead of interleaved.
    """
    work_dir = f"{_PREP_DIR_PREFIX}{secrets.token_hex(8)}"
    budget = sum(step.timeout_sec for step in steps) + _PREP_EXEC_SLACK_SEC
    result = await _checked_read(
        _prep_env(env),
        _prep_script(steps, work_dir),
        "Verifier hardening failed: reading preparation receipt",
        user="root",
        timeout_sec=budget,
    )
    lines = (getattr(result, "stdout", None) or "").split("\n")
    try:
        start = lines.index(f"{_PREP_MARK}-START")
    except ValueError:
        # Nothing ran: /tmp is not sticky and root's, the folder could not be
        # made, or the shell could not parse the script (it is one subshell,
        # parsed before it runs).
        logger.debug(
            "verifier preparation script did not start (rc=%s); "
            "running the steps one by one",
            _exec_return_code(result),
        )
        return 0
    ran = 0
    status = ""
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line == f"{_PREP_MARK}-END" and ran == len(steps):
            return ran
        if ran < len(steps) and line == f"{_PREP_MARK}-UNAVAILABLE {steps[ran].name}":
            logger.debug(
                "verifier preparation script could not start step %s; "
                "running it and the rest one by one",
                steps[ran].name,
            )
            return ran
        if ran and line == f"{_PREP_MARK}-FAIL {steps[ran - 1].name}":
            step = steps[ran - 1]
            if status == "timeout":
                raise RuntimeError(
                    f"Command timed out after {step.timeout_sec} seconds"
                )
            stderr = (getattr(result, "stderr", None) or "").split("\n")
            while stderr and stderr[0] == _DIND_HEARTBEAT_LINE:
                stderr.pop(0)
            failed = SimpleNamespace(
                stdout="\n".join(lines[index + 1 :]), stderr="\n".join(stderr)
            )
            raise RuntimeError(
                f"{step.label} exited with rc={status}{_exec_failure_detail(failed)}"
            )
        fields = line.split(" ")
        if (
            len(fields) == 5
            and fields[0] == f"{_PREP_MARK}-STEP"
            and ran < len(steps)
            and fields[1] == steps[ran].name
            and (fields[2] == "timeout" or fields[2].isdigit())
        ):
            status = fields[2]
            took = _uptime_seconds(fields[3], fields[4])
            if took is not None:
                _add_seconds(seconds, steps[ran].span, took)
            ran += 1
    last = steps[ran - 1].name if ran else "none"
    raise RuntimeError(
        "Verifier hardening failed: the preparation script stopped after step "
        f"{last} (rc={_exec_return_code(result)}){_exec_failure_detail(result)}"
    )


async def _run_prep_steps(
    env: Any, steps: list[_PrepStep], seconds: dict[str, float] | None = None
) -> None:
    """Run *steps* in order: in one exec where the backend allows it,
    otherwise one exec each. A step the script could not start runs, with
    the steps after it, one exec each."""
    if not steps:
        return
    done = 0
    if getattr(env, "verifier_prep_script", False) is True:
        done = await _run_prep_script(env, steps, seconds)
    for step in steps[done:]:
        await _run_prep_step(env, step, seconds)


async def _reclaim_disk(env, workspace: str | None) -> None:
    """Reclaim re-downloadable cache space before the verifier installs deps.

    Heavy SkillsBench tasks (playwright + marker-pdf + HF model snapshots) can
    saturate disk-constrained sandboxes — notably Daytona's hard 10GB/sandbox
    cap — so the verifier's own ``uv``/``pip`` install then fails with ENOSPC
    ("No space left on device") and the run is lost to infra instead of a real
    score. Best-effort and result-neutral: only re-downloadable download caches
    (uv/pip/apt) are cleared — never the workspace, agent outputs, installed
    tools, or task assets — and a failure here never blocks the verifier. Runs
    on ``main`` only, consistent with the hardening policy.

    Workspace-aware AND symlink-safe (#601): a task can legitimately use /root,
    /home/<user>, or /tmp/uv-* as its workspace, and an agent can plant
    ``~/.cache -> /app`` so a naive ``rm -rf "$u/.cache/uv"`` would traverse into
    workspace/output state. ``build_reclaim_caches_cmd`` rejects symlinked
    candidates and realpath-guards every deletion against the workspace and
    /logs — this matters because restore_workspace defaults to False.
    """
    try:
        await exec_raw(
            env.exec,
            _prep_on_path(build_reclaim_caches_cmd(workspace)),
            user="root",
            timeout_sec=30,
        )
    except Exception:
        logger.debug("pre-verifier disk reclaim skipped", exc_info=True)


async def _restore_workspace_state(env, workspace: str) -> None:
    """Restore workspace + verifier copy from the pre-agent snapshot."""
    await _restore_build_config(env, workspace)
    await _refresh_verifier_workspace(env, workspace)
    await exec_raw(
        env.exec, _prep_on_path(_restore_workspace_cmd(workspace)), user="root"
    )


async def _freeze_workspace(env, workspace: str, *, git: bool = True) -> None:
    """Purge external symlinks, stale caches and startup hooks, then chown
    (see _freeze_steps), one exec each."""
    for step in _freeze_steps(workspace, bfx=_bfx_in_scripts(env), git=git):
        await _run_prep_step(env, step)


async def _build_verifier_env(
    env,
    task: "Task",
    sandbox_user: str | None,
    workspace: str | None,
    *,
    agent_paths: "tuple[str, ...] | None" = None,
    bash_env: str | None = None,
    blocked: tuple[str, ...] | PathExclusions | None = None,
    target_blocked: PathExclusions | None = None,
    image_pythonpath: _EarlyRead | None = None,
) -> dict[str, str]:
    """Assemble the hardened verifier env, re-pinning security invariants.

    Task-level verifier env vars merge over the defaults, then the hard
    invariants are re-pinned so a task cannot replace PATH, strip
    -c /dev/null / --confcutdir, re-enable entry-point plugin loading, or
    inject code via breakpoint()/coverage/Django/Celery startup hooks or a
    bash startup file (BASH_ENV) the agent could write.

    *agent_paths* marks a separate verifier sandbox and lists what its
    transfer wrote: the plugin guard distrusts those (and /logs) alone there,
    instead of every agent-writable tree and anything not root's alone.

    One root exec first reads PATH, /etc/os-release and whether
    python3 exists (PY8 U12); the functions below use its answers while this
    runs and read for themselves when it gave none.
    """
    token = _verifier_reads.set((env, await _probe_verifier_reads(env)))
    try:
        return await _assemble_verifier_env(
            env,
            task,
            sandbox_user,
            workspace,
            agent_paths=agent_paths,
            bash_env=bash_env,
            blocked=blocked,
            target_blocked=target_blocked,
            image_pythonpath=image_pythonpath,
        )
    finally:
        _verifier_reads.reset(token)


async def _assemble_verifier_env(
    env,
    task: "Task",
    sandbox_user: str | None,
    workspace: str | None,
    *,
    agent_paths: "tuple[str, ...] | None" = None,
    bash_env: str | None = None,
    blocked: tuple[str, ...] | PathExclusions | None = None,
    target_blocked: PathExclusions | None = None,
    image_pythonpath: _EarlyRead | None = None,
) -> dict[str, str]:
    """The verifier env of :func:`_build_verifier_env`."""
    service = task.config.verifier.service
    local_seeds = (
        _blocked_verifier_path_prefixes(sandbox_user, workspace)
        if agent_paths is None
        else _separate_verifier_blocked_prefixes(agent_paths)
    )
    if service != "main" and target_blocked is None:
        # Direct callers also cross the namespace boundary by coordinates.
        # A captured main object may not be reinterpreted as target aliases.
        if isinstance(blocked, PathExclusions):
            in_target = _InService(env, service)
            local = await _path_exclusions_before_freeze(in_target, local_seeds)
            target_blocked = await _index_exclusions(
                in_target,
                dataclasses.replace(
                    local,
                    coordinates=blocked.coordinates | local.coordinates,
                    inodes=blocked.inodes | local.inodes,
                ),
            )
        else:
            blocked, target_blocked = await _service_exclusions_before_freeze(
                env,
                blocked if blocked is not None else local_seeds,
                service,
                target_seeds=local_seeds,
            )
    path_blocked = blocked if blocked is not None else local_seeds
    if isinstance(blocked, PathExclusions):
        blocked = blocked.paths()
    if agent_paths is None:
        guard_blocked = (
            blocked
            if blocked is not None
            else _blocked_verifier_path_prefixes(sandbox_user, workspace)
        )
        ownership = True
        agent_gids = await _agent_gids(env, sandbox_user)
        # The pre-agent copy of the workspace, root's alone.
        snapshot = _VERIFIER_WORKSPACE_COPY if workspace else None
    else:
        guard_blocked = _separate_verifier_blocked_prefixes(agent_paths)
        ownership = False
        agent_gids = None
        snapshot = None
        if blocked is not None:
            guard_blocked = blocked
    # Hardening's own probes and the guard use the judged PATH; test.sh gets
    # the refusal stubs after it (``_install_refusal_stubs``). A test.sh in
    # another service gets that service's PATH, judged and stubbed there,
    # with no agent user (V3-CONFIRM-5); main's PATH still runs main's probes.
    # Main's PATH judge and the PYTHONPATH judge change nothing and need
    # nothing from each other: at once where the sandbox allows it
    # (sandbox._in_order, PERF-G). The PYTHONPATH outcome is kept and raised
    # where it was read before, after the PATH contract and its stubs.
    main_report, pythonpath_read = await in_order(
        env,
        lambda: _judge_verifier_path(
            env, sandbox_user, workspace, blocked=path_blocked
        ),
        lambda: _outcome(
            lambda: _trusted_verifier_pythonpath(
                env, sandbox_user, image_pythonpath=image_pythonpath
            )
        ),
    )
    if service == "main":
        in_service = env
        path_report = main_report
    else:
        in_service = _InService(env, service)
        path_report = dataclasses.replace(
            await _judge_verifier_path(
                in_service,
                None,
                workspace,
                blocked=target_blocked,
            ),
            service=service,
        )
    # Preserve the judged folders even when the contract itself refuses grading.
    report_trusted_path(path_report)
    await _check_required_path_tools(in_service, path_report, task.config.verifier)
    path_report = await _install_refusal_stubs(in_service, path_report)
    report_trusted_path(path_report)
    hardened_path = main_report.path
    hardened_pythonpath = pythonpath_read.get()
    distro_env = await _distro_pip_env(env)

    verifier_env = dict(VERIFIER_ENV)
    verifier_env.update(distro_env)
    if task.config.verifier.env:
        from benchflow.task.env import resolve_env_vars

        # Filled like every other task env table: ``${VAR}`` only from a
        # variable the operator named (--allow-task-env). os.path.expandvars
        # here once filled any $VAR of the operator's environment, inside a
        # longer value too.
        verifier_env.update(resolve_env_vars(task.config.verifier.env))
    verifier_env["PATH"] = path_report.run_path
    verifier_env["PYTHONPATH"] = hardened_pythonpath
    # Every pip the verifier runs reads no configuration, whatever the task's
    # env says; the same pin runs again at every verifier
    # exec (verifier_core), so a path that skips hardening keeps it too.
    if verifier_env.get("PIP_CONFIG_FILE") != os.devnull:
        logger.warning(
            "Task %s sets PIP_CONFIG_FILE in its verifier env; the verifier's pip "
            "reads no configuration file. %s",
            getattr(task, "name", "?"),
            PIP_CONFIG_FIX,
        )
    verifier_env = pin_verifier_env(verifier_env)
    verifier_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    verifier_env["PYTHONBREAKPOINT"] = "0"
    verifier_env["COVERAGE_PROCESS_START"] = ""
    verifier_env["DJANGO_SETTINGS_MODULE"] = ""
    verifier_env["CELERY_CONFIG_MODULE"] = ""
    # Always named, so they replace the image's own: BASH_ENV the resolved
    # path of a trusted file the task or the image named, or empty; ENV empty
    # (_TRUSTED_BASH_ENV_SCRIPT).
    if agent_paths is not None:
        # A separate sandbox has no pre-agent capture for the transferred
        # bytes, even if a caller supplies a trusted path from another phase.
        logger.warning(
            "The separate verifier sandbox runs with BASH_ENV empty: "
            "transferred activation has no pre-agent provenance"
        )
        verifier_env["BASH_ENV"] = ""
    else:
        verifier_env["BASH_ENV"] = (
            bash_env
            if bash_env is not None
            else await _trusted_verifier_bash_env(env, task, guard_blocked)
        )
    verifier_env["ENV"] = ""
    # What the verifier's own command exports (_verifier_env.pinned_shell_prefix).
    with contextlib.suppress(AttributeError, ValueError):  # a test double's config
        task.config.verifier._trusted_bash_env = verifier_env["BASH_ENV"]
    # The verifier's uv and pip state moves to a fresh root-owned directory,
    # away from what the agent could have written (a WORKDIR /root image's
    # ~/.cache/uv, ~/.config/uv/uv.toml, ...); the guard trusts it by path, so
    # what test.sh installs there loads whatever the runtime's mask. Whether
    # test.sh runs as root is asked once, and that answer decides both this
    # move and the guard's policy (`0:0` once got the guard's
    # root rule without the move).
    in_main = task.config.verifier.service == "main"
    verifier_uid = (
        await _verifier_uid(env, task.config.verifier.user) if in_main else None
    )
    verifier_root = _verifier_is_root(verifier_uid, task.config.verifier.user)
    # Auto-discover pytest plugins that resolve to root-owned system code, plus
    # task config declarations. Appends -p flags to the hardened base.
    #
    # The move and the discovery need nothing from each other: the move reads
    # the uv and pip settings and makes a new folder at /, the discovery reads
    # the trusted PYTHONPATH and PATH's packages, and neither reads what the
    # other makes. They run side by side where the sandbox allows it; a
    # failure of the move is raised first, as before (sandbox._in_order).
    moved, (required, optional) = await in_order(
        env,
        lambda: _isolate_verifier_tool_state(
            env,
            task,
            verifier_env,
            sandbox_user,
            workspace,
            blocked=guard_blocked,
            verifier_root=verifier_root,
            path=hardened_path,
        ),
        lambda: _discover_pytest_plugins(
            env,
            task,
            sandbox_user,
            workspace,
            pythonpath=hardened_pythonpath,
            blocked=guard_blocked,
            ownership=ownership,
            verifier_path=hardened_path,
            agent_gids=agent_gids,
        ),
    )
    verifier_env.update(moved)
    # This policy does not depend on the probe's interpreter, success, user
    # or service. A verifier may select/create another Python at any time.
    verifier_env["PIP_CONFIG_FILE"] = os.devnull
    # The verifier's plugins from BenchFlow's protected runtime.
    # In a target service (#248) no guard runs; its own tree (its own kit's
    # Python, V2-R4-6) still gives test.sh the protected runner there.
    runtime = await _verifier_runtime_manifest(
        env,
        task,
        sandbox_user,
        build=not ownership,
        service=task.config.verifier.service,
    )
    env._benchflow_verifier_runtime = runtime
    # The interpreters the guard reaches through PYTHONPATH only, which a
    # guard-load verifier error names (PT-C8-CONF-2b round 3).
    env._benchflow_guard_unrecorded = ()
    env._benchflow_guard_why = {}
    if runtime and runtime.get("runner"):
        verifier_env["BENCHFLOW_PYTEST"] = str(runtime["runner"])
    # A task whose tests folder and [verifier] env never name the runner gets
    # none; a file in its image that calls it runs the stub (REL r4 H1).
    from benchflow.sandbox._verifier_runtime import runner_requested

    runner_stub = not (runtime and runtime.get("runner")) and not runner_requested(task)
    # Without the guard nothing else could load the optional ones.
    flags = _plugin_flags([*optional, *required])
    # Hardening, and so the guard, lives in ``main`` only (#248). A test.sh in
    # another service could not import a ``-p`` guard and would score 0.
    ini: ImagePytestIni | None = None
    if in_main:
        # The guard loads the optional plugins only in a pytest that has
        # them (#1175); the declared ones stay -p flags for every pytest.
        (
            guard_directory,
            flags,
            guarded,
            unrecorded,
        ) = await _install_pytest_plugin_guard(
            env,
            sandbox_user,
            workspace,
            _plugin_flags(required),
            verifier_path=hardened_path,
            runtime=runtime,
            tests_dir=_verifier_confcutdir(task),
            blocked=guard_blocked,
            ownership=ownership,
            verifier_uid=verifier_uid,
            optional=optional,
            agent_gids=agent_gids,
            snapshot=snapshot,
            verifier_root=verifier_root,
            # A file test.sh installs in the agent's trees can be its own only
            # when uv and pip read configuration hardening protected (#1170).
            record=verifier_root and bool(moved),
            named_inputs=_verifier_dependency_hints(task),
            runner_stub=runner_stub,
        )
        env._benchflow_guard_unrecorded = tuple(unrecorded)
        guard_name = os.path.basename(guard_directory or "")
        if runner_stub and guard_name.startswith(PYTEST_PLUGIN_GUARD_PREFIX):
            verifier_env["BENCHFLOW_PYTEST"] = pytest_runner_stub_file(guard_name)
        if (
            runtime
            and runtime.get("runner")
            and guard_name.startswith(PYTEST_PLUGIN_GUARD_PREFIX)
        ):
            # The protected runner signs its verdicts like the guard: where
            # and with which key (names only; the key never leaves its file).
            verifier_env["BENCHFLOW_PYTEST_MARKERS"] = os.path.join(
                pytest_plugin_guard_markers_dir(guard_name), guard_name
            )
            verifier_env["BENCHFLOW_PYTEST_KEY"] = pytest_plugin_guard_key_file(
                guard_name
            )
        # Only a Python without a copy needs the guard on PYTHONPATH; anywhere
        # else a task's preflight may read the entry as injected startup state.
        # A Python the pre-agent record does not vouch for counts as one
        # without a copy: where root's run of it looks is not known
        # (PT-C8-CONF-2b round 2).
        if (
            not guarded
            or unrecorded
            or _verifier_may_run_uncopied_python(task, hardened_path)
        ):
            verifier_env["PYTHONPATH"] = ":".join(
                filter(None, (guard_directory, hardened_pythonpath))
            )
        # The image's own pytest configuration, where pytest itself would read
        # it and while the agent left what pytest reads of it as it was; its
        # -p plugins come after the guard, which judges them.
        if workspace and _pytest_may_read_the_workspace_config(task, workspace):
            ini = await _image_pytest_ini(env, workspace)
    elif runner_stub:
        # Target services have no plugin guard, but still need an explicit
        # runner value so an image-side call cannot fall back to plain pytest.
        stub = pytest_runner_stub_file(PYTEST_PLUGIN_GUARD_PREFIX + uuid.uuid4().hex)
        await _checked_exec(
            env,
            f"umask 022; mkdir {shlex.quote(os.path.dirname(stub))} && "
            f"printf %s {shlex.quote(_pytest_runner_stub_source())} > {shlex.quote(stub)} && "
            f"chmod 755 {shlex.quote(stub)}",
            "Verifier setup failed: preparing the runner stub",
            user="root",
            service=task.config.verifier.service,
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
        verifier_env["BENCHFLOW_PYTEST"] = stub
    verifier_env["PYTEST_ADDOPTS"] = _build_pytest_addopts(
        workspace,
        flags,
        verifier_confcutdir=_verifier_confcutdir(task),
        ini=ini,
    )
    return verifier_env


async def harden_before_verify(
    env,
    task: "Task",
    sandbox_user: str | None,
    workspace: str | None = None,
    # Default false because SkillsBench/TB2-style answers often are workspace
    # edits. Going forward, enforce true only via an explicit task/benchmark
    # contract, e.g. task config [verifier] restore_workspace = true after an
    # oracle/diff audit proves the answer is not stored in the workspace.
    restore_workspace: bool = False,
    *,
    agent_paths: "tuple[str, ...] | None" = None,
) -> dict[str, float]:
    """Neutralize agent tampering before running the verifier.

    1. Kill sandbox-user processes (prevent concurrent writes during teardown).
    2. Clear /logs/verifier/ contents while preserving remote bind mounts, and
       prepare the legacy /app rootdir.
    3. Reclaim re-downloadable cache space before the verifier installs deps.
    4. Optionally restore the workspace from the pre-agent snapshot. This is
       destructive to legitimate workspace-edit answers, so it is opt-in.
    5. Purge symlinks and __pycache__ trees from the workspace, then chown it
       to root but for regular files with another name (belt-and-suspenders
       against zombie sandbox writes).
    6. Remove injected conftest.py, sitecustomize.py, .pth files.
    7. Merge trusted env vars into task.config.verifier.env.

    When the trial keeps services (``x-keep-services``, see
    :func:`benchflow.task.keep_services.trial_keep`), step 1 spares the sandbox
    user's listeners on the named ports and leaves them frozen, step 2 closes
    /logs to everyone but root, and step 5 takes group and other write off the
    workspace and gives each file with another name that is not root's a fresh
    root-owned copy. Existing open writers to a root-owned workspace inode
    remain usable; this does not make the live workspace immutable. Use a
    separate verifier when grading needs immutable inputs. The verifier
    lets the services go on just before its test command
    (:func:`release_kept_services`). The kill's record names each one.

    On script-capable backends, steps 2-3 and 5-6 share one exec (two
    around an opt-in restore). The kill, pre-freeze trust decisions and
    verifier-env construction keep their host boundaries. Kept services
    retain the separate path. Status lines fill the existing timing keys.

    *agent_paths* marks a separate verifier sandbox, where no agent process
    ever ran, and lists the paths its transfer wrote (see
    ``_build_verifier_env``).

    Cross-container hardening policy (#248): every step here runs against the
    ``main`` (agent) container only — ``env.exec`` is never passed a
    ``service``. This is deliberate. In multi-container (vulhub-style) tasks
    the agent has a shell only in ``main``; the target/database containers are
    intentionally vulnerable and the agent cannot tamper with them, so they
    need no anti-tamper hardening. ``[verifier].service`` chooses where
    ``test.sh`` *runs*; it does not relocate hardening off ``main``. The one
    exception is test.sh's own PATH: it is judged, and its refusal stubs
    made, in the service test.sh runs in (V3-CONFIRM-5).
    # The image BASH_ENV is also read in that service; non-main activation
    # remains refused without shared-mount provenance.
    """
    seconds: dict[str, float] = {}
    # A separate verifier sandbox (*agent_paths*) got the agent's files from
    # BenchFlow's transfer, written as root: there root's files prove nothing
    # about who chose them, so both rules stay as before.
    same_sandbox = agent_paths is None
    agent_token = _hook_agent_user.set(same_sandbox and _agent_not_root(sandbox_user))
    path_token = _path_workspace.set(workspace if same_sandbox else None)
    try:
        await _harden_steps(
            env, task, sandbox_user, workspace, restore_workspace, agent_paths, seconds
        )
    finally:
        _path_workspace.reset(path_token)
        _hook_agent_user.reset(agent_token)
        for name, took in seconds.items():
            _timing_spans.add(_timing_spans.HARDEN_BEFORE_VERIFY, name, took)
    return seconds


@contextmanager
def _prep_timing(seconds: dict[str, float], name: str):
    start = _timing_spans.clock()
    try:
        yield
    finally:
        _add_seconds(seconds, name, _timing_spans.clock() - start)


async def _harden_steps(
    env: Any,
    task: "Task",
    sandbox_user: str | None,
    workspace: str | None,
    restore_workspace: bool,
    agent_paths: "tuple[str, ...] | None",
    seconds: dict[str, float],
) -> None:
    # Each step is a sub-key of timing.json's harden_before_verify span, by
    # the names in timing_spans.HARDEN_STEPS (a no-op outside a trial).
    from benchflow.task import keep_services as _keep_services

    keep = (
        _keep_services.trial_keep(task, sandbox_user) if agent_paths is None else None
    )
    # 1. Kill sandbox-user processes (prevent concurrent writes during teardown).
    if sandbox_user:
        with _prep_timing(seconds, "kill_sandbox_user_procs"):
            if keep is None:
                await _kill_sandbox_user_procs(env, sandbox_user)
            else:
                outcome = await _kill_sandbox_user_procs_keeping(
                    env,
                    sandbox_user,
                    keep.ports,
                    wait=not _keep_services.waited(task),
                )
                _keep_services.mark_waited(task)
                _record_kept(keep, outcome, turn_end=_keep_services.turn_end(task))
            # The accounts the task lets the agent switch to (K1-R11-1).
            await kill_switched_accounts(env)
            from benchflow.agents.credentials import (
                muse_login_present,
                remove_muse_credentials,
            )

            # Only a sandbox with a Muse login gets Muse's strict clean-up;
            # another agent's home may link .config anywhere (AG-R2-4).
            if await muse_login_present(
                env, sandbox_user, timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC
            ):
                await remove_muse_credentials(
                    env, sandbox_user, timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC
                )
    # Decide before clear/reclaim/restore/freeze changes any ownership. Even
    # main can mount its workspace a second time under an unblocked alias.
    # A file the agent could write must not become trusted merely because
    # our own chown makes that alias look root-owned.
    bash_blocked = (
        _blocked_verifier_path_prefixes(sandbox_user, workspace)
        if agent_paths is None
        else _separate_verifier_blocked_prefixes(agent_paths)
    )

    async def bash_env() -> str:
        if agent_paths is not None:
            return ""
        return await _trusted_verifier_bash_env(env, task, bash_blocked)

    async def guard() -> tuple[tuple[str, ...] | PathExclusions, PathExclusions | None]:
        if not (sandbox_user or agent_paths is not None):
            return bash_blocked, None
        if task.config.verifier.service == "main":
            return await _path_exclusions_before_freeze(env, bash_blocked), None
        return await _service_exclusions_before_freeze(
            env, bash_blocked, task.config.verifier.service
        )

    async def aliases() -> tuple[str, ...]:
        # Cleanup must preserve only the workspace's aliases, not all blocked
        # runtime prefixes (/tmp itself is blocked for verifier tools).
        return await _blocked_before_freeze(env, (workspace,)) if workspace else ()

    async def early_pythonpath() -> _EarlyRead | None:
        # Read on the same task route as BASH_ENV, beside it; only on a
        # sandbox that runs steps side by side, where it costs no time.
        return await _early_image_pythonpath(env) if runs_steps_at_once(env) else None

    # Reads with no host-side decision between them: at once where the
    # sandbox allows it, after the kill and before the first change below
    # (sandbox._in_order, PERF-G). The exclusions also write their inode
    # index, a new private folder in the root-only inspector that none of
    # the others reads, so they go first: a step that changes something goes
    # before the steps that may fail beside it (merge wave batch 35).
    (
        (guard_blocked, target_blocked),
        trusted_bash_env,
        workspace_aliases,
        image_pythonpath,
    ) = await in_order(env, guard, bash_env, aliases, early_pythonpath)
    # One judged interpreter for every root Python step below, before the
    # first of them runs: the freeze's purges, the hook
    # clean-up, the cache reclaim, the restore fallback and the kept-service
    # ancestry check. Candidates that are all rejected refuse grading; an
    # image with no python3 at all skips those steps' Python, as before.
    root_python = await _judge_root_python(env, guard_blocked, sandbox_user)
    if root_python.path is None:
        if root_python.rejected:
            raise RuntimeError(
                "Verifier hardening failed: "
                + root_python.missing_text(_task_image(task))
            )
        # Nothing a user runs the task with can change this, so it is no
        # console line (dogfood p1-5): the trial's result.json lists it for
        # the task's author, and DEBUG names the image.
        logger.debug(
            "Verifier hardening: %s; the Python parts of the workspace freeze, "
            "the hook clean-up and the cache reclaim are skipped",
            root_python.missing_text(_task_image(task)),
        )
        from benchflow._utils.trial_warnings import report_trial_warning

        report_trial_warning(
            "the task's image has no root-owned python3 in /usr/bin, /bin or "
            "/usr/local/bin, so the Python steps of the verifier hardening "
            "(workspace freeze, hook clean-up, cache reclaim) were skipped; "
            "install python3 there to have them run"
        )
    token = _root_python.set(root_python.path)
    try:
        if getattr(env, "verifier_prep_script", False) is True and keep is None:
            # These commands have no host-side decisions between them. The kill
            # and pre-freeze mount/activation proofs above must finish first;
            # moving either into this batch would lose their checked records.
            hardening = _read_hardening_config(getattr(task, "task_dir", None))
            # The hook clean-up is the last step either way: on bfx it runs
            # from the host right after the script, in the same order.
            on_bfx = _hook_cleanup_on_bfx(env)
            before = [
                *_verifier_output_steps(bfx=_bfx_in_scripts(env)),
                _reclaim_step(workspace),
            ]
            after = [
                *(
                    _freeze_steps(
                        workspace,
                        bfx=_bfx_in_scripts(env),
                        git=hardening.get("cleanup_git", True),
                    )
                    if workspace
                    else []
                ),
                *(
                    []
                    if on_bfx
                    else [_cleanup_step(hardening, workspace, workspace_aliases)]
                ),
            ]
            if workspace and restore_workspace:
                await _run_prep_steps(env, before, seconds)
                with _prep_timing(seconds, "restore_workspace"):
                    await _restore_workspace_state(env, workspace)
                await _run_prep_steps(env, after, seconds)
            else:
                await _run_prep_steps(env, before + after, seconds)
            if on_bfx:
                with _prep_timing(seconds, "cleanup_conftests"):
                    await _cleanup_hooks_on_bfx(
                        env,
                        hardening,
                        workspace,
                        workspace_aliases,
                        "Verifier hardening failed: removing Python start-up hooks",
                    )
        else:
            # 2. Wipe /logs/verifier/ contents while preserving remote bind mounts, then
            #    prepare the legacy /app rootdir fallback.
            with _prep_timing(seconds, "clear_verifier_output"):
                await clear_verifier_output_dir(
                    env,
                    "Verifier hardening failed: clearing verifier output directory",
                    user="root",
                    timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
                )
                if keep is not None and sandbox_user:
                    await _checked_exec(
                        env,
                        _CLOSE_LOGS_CMD,
                        "Verifier hardening failed: closing /logs to the kept services",
                        user="root",
                        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
                    )
            with _prep_timing(seconds, "prepare_app_dir"):
                await ensure_legacy_app_dir(
                    env,
                    "Verifier hardening failed: preparing /app",
                    user="root",
                    timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
                )
            # 3. Reclaim re-downloadable cache space before the verifier installs deps.
            with _prep_timing(seconds, "reclaim_disk"):
                await _reclaim_disk(env, workspace)
            # 4. Optionally restore the workspace from the pre-agent snapshot (opt-in;
            #    destructive to legitimate workspace-edit answers).
            if workspace and restore_workspace:
                with _prep_timing(seconds, "restore_workspace"):
                    await _restore_workspace_state(env, workspace)
            # 5. Purge symlinks + __pycache__, then chown the workspace to root
            #    (a regular file with another name keeps its owner).
            if workspace:
                with _prep_timing(seconds, "freeze_workspace"):
                    await _freeze_workspace(
                        env,
                        workspace,
                        git=_read_hardening_config(getattr(task, "task_dir", None)).get(
                            "cleanup_git", True
                        ),
                    )
                    if keep is not None and sandbox_user:
                        # Closing the workspace and checking its ancestors both
                        # run root's judged Python; without one,
                        # kept services are refused.
                        if root_python.path is None:
                            raise RuntimeError(
                                "Verifier hardening failed: protecting workspace "
                                "ancestors: "
                                + root_python.missing_text(_task_image(task))
                            )
                        # A docker exec umask of 0000 leaves what the agent made
                        # writable by others, and a file with another name keeps
                        # the agent user as owner: a kept service could still
                        # change either.
                        await _checked_exec(
                            env,
                            _prep_on_path(_close_workspace_cmd(workspace)),
                            "Verifier hardening failed: closing the workspace to the "
                            "kept services",
                            user="root",
                            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
                        )
                        # Frozen contents are insufficient if a kept process can rename
                        # an ancestor or replace a symlink in the path used by test.sh.
                        # Inspect every resolution step while all writers remain stopped;
                        # incomplete inspection or mutable ancestry refuses.
                        # On the root route, with the judged interpreter.
                        # The kit's Python through PT's one entry point when
                        # the kit has one (the kit first).
                        ancestry = (
                            await root_python_command(
                                env, _WORKSPACE_ANCESTRY_SCRIPT, workspace
                            )
                            if root_python.path == KIT_PYTHON
                            else _on_path(
                                "cd / && "
                                + shlex.quote(root_python.path)
                                + " -I -S -c "
                                + shlex.quote(_WORKSPACE_ANCESTRY_SCRIPT)
                                + " "
                                + shlex.quote(workspace)
                            )
                        )
                        await _checked_exec(
                            env,
                            ancestry,
                            "Verifier hardening failed: protecting workspace ancestors",
                            user="root",
                            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
                        )
            # 6. Remove injected conftest.py, sitecustomize.py, .pth files. The rootfs
            #    conftest walk can be slow on network-backed FS, so use the shared
            #    verifier-setup budget (not the default 10s) — same rationale as the
            #    soft-verify path, on the path that actually scores.
            hardening = _read_hardening_config(getattr(task, "task_dir", None))
            with _prep_timing(seconds, "cleanup_conftests"):
                if _hook_cleanup_on_bfx(env):
                    await _cleanup_hooks_on_bfx(
                        env,
                        hardening,
                        workspace,
                        workspace_aliases,
                        "Verifier hardening failed: removing Python start-up hooks",
                    )
                else:
                    await _run_prep_step(
                        env, _cleanup_step(hardening, workspace, workspace_aliases)
                    )
        # 7. Merge trusted env vars into task.config.verifier.env.
        with _prep_timing(seconds, "build_verifier_env"):
            task.config.verifier.env = await _build_verifier_env(
                env,
                task,
                sandbox_user,
                workspace,
                agent_paths=agent_paths,
                bash_env=trusted_bash_env,
                blocked=guard_blocked,
                target_blocked=target_blocked,
                image_pythonpath=image_pythonpath,
            )
    finally:
        _root_python.reset(token)
