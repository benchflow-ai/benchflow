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

import ipaddress
import itertools
import json as _json
import logging
import os
import re
import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from benchflow.agents.registry import get_sandbox_home_dirs
from benchflow.sandbox import _verifier_tool_state
from benchflow.sandbox._cache_reclaim import build_reclaim_caches_cmd

if TYPE_CHECKING:
    from benchflow.task import Task

logger = logging.getLogger(__name__)


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
_SAFE_PATH_RE = re.compile(r"^/[a-zA-Z0-9_./*?\-]+(/[a-zA-Z0-9_./*?\-]+)*$")


def _validate_locked_path(p: str) -> None:
    """Reject injection and traversal in a locked path."""
    p_norm = os.path.normpath(p)
    if p_norm != p:
        raise ValueError(
            f"Invalid locked path {p!r}: normalizes to {p_norm!r} — "
            f"use the normalized form directly"
        )
    if any(c == ".." for c in p.split("/")):
        raise ValueError(f"Invalid locked path {p!r}: '..' component not allowed")
    if not _SAFE_PATH_RE.match(p):
        raise ValueError(
            f"Invalid locked path {p!r}: must be absolute, "
            f"alphanumeric with /-_.*? only"
        )
    if p.endswith("/") and p != "/":
        raise ValueError(
            f"Invalid locked path {p!r}: trailing slash not allowed "
            f"(chown on '/dir/' may have unintended scope)"
        )


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
    DNS open (the Harbor #2527 failure mode). Docker's embedded resolver
    (127.0.0.11, reached on a rewritten port) is refused outright. Listed IP
    and CIDR entries are reachable directly, for any protocol, as in Harbor.
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
    sandbox_user: str, *, allow_networks: tuple[str, ...] | None = None
) -> str:
    user = shlex.quote(sandbox_user)
    allow = (
        "; " + _allowlist_firewall_cmd(allow_networks)
        if allow_networks is not None
        else ""
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
        f"agent_uid=$(id -u {user}) || exit 86; "
        'iptables -C OUTPUT -o lo -m owner --uid-owner "$agent_uid" '
        "-j ACCEPT 2>/dev/null || "
        'iptables -I OUTPUT 1 -o lo -m owner --uid-owner "$agent_uid" '
        "-j ACCEPT; "
        'iptables -C OUTPUT -m owner --uid-owner "$agent_uid" '
        "-j REJECT 2>/dev/null || "
        'iptables -A OUTPUT -m owner --uid-owner "$agent_uid" -j REJECT; '
        # procfs reports zero stat size even when this interface has content.
        # Install IPv6 rules whenever the kernel exposes the stack, including
        # before a non-loopback interface acquires an address.
        "if [ -e /proc/net/if_inet6 ]; then "
        "command -v ip6tables >/dev/null 2>&1 || "
        "{ echo 'IPv6 enabled but ip6tables unavailable' >&2; exit 86; }; "
        'ip6tables -C OUTPUT -o lo -m owner --uid-owner "$agent_uid" '
        "-j ACCEPT 2>/dev/null || "
        'ip6tables -I OUTPUT 1 -o lo -m owner --uid-owner "$agent_uid" '
        "-j ACCEPT; "
        'ip6tables -C OUTPUT -m owner --uid-owner "$agent_uid" '
        "-j REJECT 2>/dev/null || "
        'ip6tables -A OUTPUT -m owner --uid-owner "$agent_uid" -j REJECT; '
        "fi"
        f"{allow}"
    )


def build_priv_drop_cmd(agent_launch: str, sandbox_user: str) -> str:
    """Build a shell command that drops to sandbox_user via setpriv or su.

    setpriv (util-linux) execs directly; su -l is the fallback for Alpine/BusyBox.
    No outer sh -c wrapper — DockerProcess wraps in bash -c already.
    """
    inner = f"export HOME=/home/{sandbox_user} && {agent_launch}"
    quoted = shlex.quote(inner)
    return (
        f"if setpriv --help 2>&1 | grep -q reuid; then"
        f" exec setpriv --reuid={sandbox_user} --regid={sandbox_user}"
        f" --init-groups -- bash -c {quoted};"
        f" else exec su -l {sandbox_user} -c {quoted};"
        f" fi"
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
    if not (no_web or denylist):
        return
    if not sandbox_user:
        if denylist:
            raise RuntimeError("network_mode='denylist' requires a sandbox_user")
        return

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

    allow_networks = None
    if denylist and agent_env.get(EGRESS_ALLOWLIST_ENV) == "1":
        allow_networks = tuple(
            n for n in agent_env.get(EGRESS_ALLOW_NETWORKS_ENV, "").split(",") if n
        )
    await enforce_sandbox_uid_egress(env, sandbox_user, allow_networks=allow_networks)


async def enforce_sandbox_uid_egress(
    env: Any, sandbox_user: str, *, allow_networks: tuple[str, ...] | None = None
) -> None:
    """Install the shared loopback-only UID firewall without inference setup.

    ``allow_networks`` (allowlist mode) adds the DNS redirect and the direct
    rules for listed IP ranges; see ``_allowlist_firewall_cmd``.
    """
    result = await env.exec(
        _agent_egress_firewall_cmd(sandbox_user, allow_networks=allow_networks),
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


async def setup_sandbox_user(
    env, sandbox_user: str, workspace: str, *, timeout_sec: int = 120
) -> str:
    """Create non-root sandbox user, grant workspace access. Return agent_cwd."""
    if not re.match(r"^[a-z_][a-z0-9_-]*$", sandbox_user):
        raise ValueError(
            f"Invalid sandbox_user: {sandbox_user!r} (must be alphanumeric)"
        )
    logger.info(f"Setting up sandbox user: {sandbox_user}")
    home = f"/home/{sandbox_user}"
    home_dirs = sorted(d for d in get_sandbox_home_dirs() if d != ".local")
    # busybox images (Alpine) ship adduser but not useradd.
    result = await env.exec(
        f"id -u {sandbox_user} >/dev/null 2>&1 || "
        f"useradd -m -s /bin/bash {sandbox_user} || "
        f"adduser -D -s /bin/bash {sandbox_user} && "
        f"{_legacy_root_tool_link_cmd('/root/.local/bin', f'{home}/.local/bin')} && "
        f"{_legacy_root_tool_link_cmd('/root/.nvm', f'{home}/.nvm')} && "
        f"for d in {' '.join(home_dirs)}; do "
        f"mkdir -p {home}/$d && "
        f"if [ -d /root/$d ]; then "
        f"cp -a /root/$d/. {home}/$d/ 2>/dev/null || true; fi; done && "
        f"chown -R {sandbox_user}:{sandbox_user} {home} && "
        f"chown -R {sandbox_user}:{sandbox_user} {shlex.quote(workspace)} && "
        f"for d in /output /outputs; do "
        f'if [ -d "$d" ] && [ ! -L "$d" ]; then '
        f'chown -R {sandbox_user}:{sandbox_user} "$d"; fi; done',
        timeout_sec=timeout_sec,
    )
    if _exec_return_code(result) != 0:
        # A later step (a chown on a read-only mount) may fail once the user
        # exists, as before; a missing user would only surface much later.
        exists = await env.exec(f"id -u {sandbox_user} >/dev/null 2>&1", timeout_sec=30)
        detail = _exec_failure_detail(result)
        if _exec_return_code(exists) != 0:
            raise RuntimeError(
                f"could not create sandbox user {sandbox_user!r}: the image "
                "has neither a working useradd nor busybox adduser; run as "
                f"root with --sandbox-user none or add one to the image.{detail}"
            )
        logger.warning(f"Sandbox user {sandbox_user} setup step failed.{detail}")
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

    # Build shell command: reject symlinks, chown before chmod
    parts = []
    for p in paths:
        parts.append(
            f"for d in {p}; do "
            f'  [ -L "$d" ] && echo "WARN: skipping symlink $d" >&2 && continue; '
            f'  [ -e "$d" ] || continue; '
            f'  chown root:root "$d" && chmod 700 "$d"; '
            f"done"
        )
    cmd = " && ".join(parts)
    await env.exec(cmd, timeout_sec=30)


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
_SNAPSHOT_DIR = "/tmp/.benchflow_build_snapshot"
_SNAPSHOT_MANIFEST = f"{_SNAPSHOT_DIR}/manifest.json"


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
    """
    probes = [
        f"if [ -f {workspace}/{fname} ]; then "
        f"cp --preserve=all {workspace}/{fname} {_SNAPSHOT_DIR}/{fname} "
        f"&& echo {fname}=present; "
        f"else echo {fname}=absent; fi"
        for fname in _BUILD_CONFIG_FILES
    ]
    result = await env.exec(
        f"mkdir -p {_SNAPSHOT_DIR} && chmod 700 {_SNAPSHOT_DIR} && "
        f"{{ {'; '.join(probes)}; }}",
        user="root",
    )
    reported = set((result.stdout or "").split())
    manifest = {fname: f"{fname}=present" in reported for fname in _BUILD_CONFIG_FILES}
    manifest_json = _json.dumps(manifest)
    await env.exec(
        f"echo {shlex.quote(manifest_json)} > {_SNAPSHOT_MANIFEST}",
        user="root",
    )


async def _restore_build_config(env, workspace: str) -> None:
    """Restore build-config files to their pre-agent state.

    Files that existed pre-agent are restored from the snapshot; files that
    didn't are removed if the agent created them.
    """
    result = await env.exec(f"cat {_SNAPSHOT_MANIFEST}", user="root")
    manifest: dict[str, bool] = _json.loads(result.stdout)
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
        await env.exec(cmd, user="root")


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
    await env.exec(" && ".join(_log_dir_cmds(sandbox_user)), user="root")


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
    for cmd in cmds:
        await env.exec(cmd, user="root")


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
        await env.exec(cmd, user="root")


# Verifier hardening

# Trusted env vars for verifier execution — override any agent pollution.
# Intentionally omitted (negative guard in test_verify.py explains why):
# PYTHONHOME.
VERIFIER_ENV: dict[str, str] = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTEST_ADDOPTS": (
        "-c /dev/null "  # block pyproject.toml/pytest.ini/tox.ini/setup.cfg discovery
        "--confcutdir=/tests "  # block conftest.py walk-up beyond /tests
        "-p no:cacheprovider"
        # --rootdir is injected dynamically by _build_pytest_addopts() based on
        # the task's actual workspace, so it works for both /app (Harbor/SWE-bench)
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
    "PIP_NO_USER_CONFIG": "1",
    # PEP-668 base images (Fedora, recent Debian) refuse pip installs into
    # system-site without this flag. Verifier runs as root and system-site is
    # root-owned, so allowing it is safe; without it, tasks that pip-install
    # pytest in test.sh either fail outright or fall back to a user-site path
    # that PYTHONNOUSERSITE=1 hides at import time.
    "PIP_BREAK_SYSTEM_PACKAGES": "1",
    # /root is root-owned; sandbox_user cannot pre-stage caches there. Pip
    # config is already blocked by the PIP_* / PYTHONNOUSERSITE vars above.
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
_RUNTIME_PATH_PREFIXES = ("/tmp", "/var/tmp", "/logs", "/testbed")

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


def _build_pytest_addopts(
    workspace: str | None = None,
    plugin_flags: str = "",
    *,
    verifier_confcutdir: str = _LEGACY_VERIFIER_CONFCUTDIR,
) -> str:
    """Build PYTEST_ADDOPTS with a dynamic --rootdir based on the workspace.

    Without an explicit --rootdir, -c /dev/null causes pytest to fall back to
    /dev as rootdir, producing broken test node IDs (../dev/::test_foo).
    The rootdir must point to a directory that actually exists in the container.
    """
    rootdir = workspace or _DEFAULT_ROOTDIR
    base_addopts = VERIFIER_ENV["PYTEST_ADDOPTS"].replace(
        f"--confcutdir={_LEGACY_VERIFIER_CONFCUTDIR}",
        f"--confcutdir={shlex.quote(verifier_confcutdir)}",
    )
    addopts = f"{base_addopts} --rootdir={shlex.quote(rootdir)}"
    if plugin_flags:
        addopts += f" {plugin_flags}"
    return addopts


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


_TRUSTED_PATH_EXTRAS_SCRIPT = r"""
import json
import os
import stat
import sys

raw_path = json.loads(sys.argv[1])
safe_parts = set(json.loads(sys.argv[2]))
blocked_prefixes = tuple(json.loads(sys.argv[3]))


def under_path(path, prefix):
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


trusted = []
seen = set(safe_parts)
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
    try:
        real = os.path.realpath(entry)
        st = os.stat(real)
    except OSError:
        continue
    if not stat.S_ISDIR(st.st_mode):
        continue
    if any(under_path(real, prefix) for prefix in blocked_prefixes):
        continue
    if st.st_uid != 0:
        continue
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        continue
    trusted.append(entry)
print(json.dumps(trusted))
""".strip()


def _trusted_path_extras_cmd(raw_path: str, blocked_prefixes: tuple[str, ...]) -> str:
    """Build the container-side command that validates verifier PATH extras."""
    return (
        f"python3 -c {shlex.quote(_TRUSTED_PATH_EXTRAS_SCRIPT)} "
        f"{shlex.quote(_json.dumps(raw_path))} "
        f"{shlex.quote(_json.dumps(_SAFE_VERIFIER_PATH_PARTS))} "
        f"{shlex.quote(_json.dumps(blocked_prefixes))}"
    )


def _discover_pytest_plugins_cmd(
    blocked_prefixes: tuple[str, ...], pythonpath: str | None = None
) -> str:
    """Build the container-side pytest plugin discovery command.

    Runs from ``/`` so the image WORKDIR (usually the agent-writable workspace)
    is not on ``sys.path``, without user site-packages, and, when given, with
    the verifier's trusted PYTHONPATH instead of the image's raw one: the same
    locations the verifier's own interpreter starts from.
    """
    env = "PYTHONNOUSERSITE=1 "
    if pythonpath is not None:
        env = f"PYTHONPATH={shlex.quote(pythonpath)} " + env
    return (
        f"cd / && {env}python3 -c {shlex.quote(_DISCOVER_PYTEST_PLUGINS_SCRIPT)} "
        f"{shlex.quote(_json.dumps(blocked_prefixes))}"
    )


async def _discover_pytest_plugin_flags(
    env,
    task: "Task",
    sandbox_user: str | None = None,
    workspace: str | None = None,
    pythonpath: str | None = None,
) -> str:
    """Only enable plugins whose current registration and code are trusted.

    Missing aliases are deferred to the protected guard in the final pytest
    interpreter, allowing trusted verifier scripts to install plugins with uvx.
    """
    requested = list(
        dict.fromkeys(
            [
                name.strip()
                for name in [
                    *task.config.verifier.pytest_plugins,
                    *_infer_pytest_plugins_from_test_script(task),
                ]
                if name.strip()
            ]
        )
    )
    try:
        result = await env.exec(
            _discover_pytest_plugins_cmd(
                _blocked_verifier_path_prefixes(sandbox_user, workspace), pythonpath
            ),
            user="root",
            timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
        )
        if _exec_return_code(result) == 127:
            # Minimal shell-only images need no Python. A trusted verifier may
            # also install its Python/uvx runtime later; the protected guard is
            # still installed below and validates plugins in that interpreter.
            # Do not confuse a broken interpreter/discovery with proven absence.
            available = await env.exec(
                "if command -v python3 >/dev/null 2>&1; then exit 0; else exit 1; fi",
                user="root",
                timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
            )
            if _exec_return_code(available) == 1:
                return " ".join(f"-p {shlex.quote(name)}" for name in requested)
        if _exec_return_code(result) != 0:
            raise ValueError(
                "sandbox discovery command failed" + _exec_failure_detail(result)
            )
        discovered = _json.loads(result.stdout)
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
    return " ".join(
        f"-p {shlex.quote(p)}" for p in dict.fromkeys([*plugins, *requested])
    )


# pytest imports the guard by module name (``-p <guard>``), so any interpreter
# that runs pytest without being able to import it aborts, exits 1, and test.sh
# records reward 0 for a correct solution. A copy therefore goes into the first
# site directory that each existing Python keeps on its isolated sys.path: every
# python/python3/python3.N on the verifier PATH plus each pytest script's shebang
# interpreter. That reaches ``python3 -I`` and a test.sh that replaces
# PYTHONPATH (``PYTHONPATH=/app pytest``). Anything that does not run as Python 3
# with ``-I`` is skipped; a Python that cannot take or resolve the copy fails
# hardening, a verifier error instead of a silent zero. The copy is written by
# the candidate itself, from ``/``, as root, after solver quiescence.
#
# Only PYTHONPATH reaches a Python that did not exist on the verifier PATH at
# hardening (uvx, ``uv run``, a fresh venv, a distro package, a venv run by a
# path off PATH), so the guard directory goes there too, but only when the
# verifier may need it: when no Python took a copy, or when a verifier script
# names such a Python (``_verifier_may_run_uncopied_python``). Tasks read a set
# PYTHONPATH as a startup-injection hook (``os.environ.get("PYTHONPATH")`` in an
# anti-tamper preflight), so a verifier whose Pythons all hold a copy gets the
# image's trusted PYTHONPATH, as before the guard existed. A verifier that
# creates a Python in a way the scan does not recognise gets ``Error importing
# plugin "<guard>"``, which is a verifier error below, not a score.
#
# After the run the verifier turns a guard pytest could not load into a verifier
# error instead of scoring test.sh's 0. A guard pytest imported but never ran
# (pluggy refused its hooks) leaves its ``loading`` marker in the verifier
# output directory, which hardening wiped after quiescence; that check needs no
# output. A guard pytest could not import at all is found only by pytest's
# message in a top-level verifier log (``verifier_scan._has_guard_load_failure``),
# so a test.sh that discards pytest's output still scores that run 0. No sound
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
# Where the guard leaves ``<guard>.<token>.<kind>`` markers: the verifier output
# directory, which the verifier downloads and hardening empties after quiescence.
_PYTEST_PLUGIN_GUARD_MARKERS_DIR = "/logs/verifier"

_INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT = r"""
import importlib, importlib.util, os, site, sys, sysconfig
source_path, name = sys.argv[1], sys.argv[2]
with open(source_path, 'rb') as handle:
    source = handle.read()
on_path = set(os.path.realpath(entry) for entry in sys.path if entry)
candidates = [sysconfig.get_path('purelib'), sysconfig.get_path('platlib')]
candidates += getattr(site, 'getsitepackages', list)()
candidates.append(sysconfig.get_path('stdlib'))
for directory in candidates:
    if directory and os.path.isdir(directory) and os.path.realpath(directory) in on_path:
        break
else:
    sys.exit('no site directory on the isolated sys.path of ' + sys.executable)
target = os.path.join(directory, name + '.py')
try:
    with open(target, 'xb') as handle:
        handle.write(source)
    os.chmod(target, 0o444)
except FileExistsError:
    with open(target, 'rb') as handle:
        if handle.read() != source:
            sys.exit('a different module occupies ' + target)
importlib.invalidate_caches()
spec = importlib.util.find_spec(name)
if spec is None or os.path.realpath(spec.origin or '') != os.path.realpath(target):
    sys.exit('the guard does not resolve to ' + target)
""".strip()

_INSTALL_PYTEST_PLUGIN_GUARD_CMD_TEMPLATE = r"""
install_guard() {
    "$1" -I -c 'import sys; sys.exit(sys.version_info[0] != 3)' \
        </dev/null >/dev/null 2>&1 || return 0
    if "$1" -I -c __SCRIPT__ __GUARD__ __NAME__ </dev/null >/dev/null; then
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


def _pytest_plugin_guard_source(
    name: str,
    blocked: tuple[str, ...],
    requested: list[str],
    markers_dir: str | None = None,
    *,
    trusted: tuple[str, ...] = (),
) -> str:
    """Return the armed guard module *name*: its policy, then its load marker.

    *trusted* names directories whose contents the guard trusts by path: the
    verifier's uv and pip state, created after the agent stopped.
    """
    markers_dir = markers_dir or _PYTEST_PLUGIN_GUARD_MARKERS_DIR
    return (
        _DISCOVER_PYTEST_PLUGINS_SCRIPT
        + "\n_BENCHFLOW_BLOCKED = "
        + repr(blocked)
        + "\n_BENCHFLOW_REQUESTED = "
        + repr(requested)
        + "\n_BENCHFLOW_TRUSTED = "
        + repr(tuple(trusted))
        + "\n_BENCHFLOW_MARKERS = "
        + repr(os.path.join(markers_dir, name))
        + "\n_mark('loading')\n"
    )


async def _install_pytest_plugin_guard(
    env,
    sandbox_user,
    workspace,
    plugin_flags,
    verifier_path=_SAFE_VERIFIER_PATH,
    *,
    trusted: tuple[str, ...] = (),
):
    """Create an unguessable protected bootstrap after solver quiescence.

    Returns the guard directory, the ``-p`` flags with the guard first, and the
    Pythons that took a copy.
    """
    name = PYTEST_PLUGIN_GUARD_PREFIX + uuid.uuid4().hex
    directory = os.path.join(_PYTEST_PLUGIN_GUARD_PARENT, name)
    guard = os.path.join(directory, name + ".py")
    source = _pytest_plugin_guard_source(
        name,
        _blocked_verifier_path_prefixes(sandbox_user, workspace),
        shlex.split(plugin_flags)[1::2],
        trusted=trusted,
    )
    install_into_interpreters = (
        _INSTALL_PYTEST_PLUGIN_GUARD_CMD_TEMPLATE.replace(
            "__GUARD__", shlex.quote(guard)
        )
        .replace("__NAME__", shlex.quote(name))
        .replace("__PATH__", shlex.quote(verifier_path))
        .replace("__SCRIPT__", shlex.quote(_INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT))
    )
    result = await _checked_exec(
        env,
        f"mkdir -m 755 {shlex.quote(directory)} && "
        f"printf %s {shlex.quote(source)} > {shlex.quote(guard)} && "
        f"chmod 444 {shlex.quote(guard)} && {{\n{install_into_interpreters}\n}}",
        "Verifier hardening failed: installing protected pytest plugin guard",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    stdout = getattr(result, "stdout", "")
    guarded = tuple(
        line.removeprefix("guarded ")
        for line in (stdout if isinstance(stdout, str) else "").splitlines()
        if line.startswith("guarded ")
    )
    return directory, f"-p {name} {plugin_flags}".strip(), guarded


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


def _verifier_script_texts(task: "Task") -> list[str] | None:
    """Return the text of every script in the task's verifier dir, if readable."""
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
                    texts.append(data.decode("utf-8", "replace"))
    except OSError:
        return None
    return texts or None


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


def pytest_plugin_guard_markers(
    verifier_dir: Path, guard: str, kind: str
) -> list[Path]:
    """Return the ``kind`` markers the guard *guard* left in *verifier_dir*.

    Kinds are named in ``_pytest_plugin_guard.py``: ``loading`` means pytest
    imported the guard but never registered it; ``crashed`` holds the
    traceback of a guard hook that failed other than by refusing a plugin;
    ``installed`` lists the files of a refused plugin the verifier installed
    itself after the agent stopped.
    """
    try:
        return sorted(verifier_dir.glob(f"{guard}.*.{kind}"))
    except OSError:
        return []


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
    try:
        result = await env.exec(
            "cat /etc/os-release 2>/dev/null || true",
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


def _verifier_runs_as_root(user: str | int | None) -> bool:
    # ``None`` runs test.sh as the sandbox's default user, root in task images.
    return user is None or str(user) in ("root", "0")


async def _isolate_verifier_tool_state(
    env: Any,
    task: "Task",
    verifier_env: dict[str, str],
    sandbox_user: str | None,
    workspace: str | None,
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
    """
    if task.config.verifier.service != "main" or not _verifier_runs_as_root(
        task.config.verifier.user
    ):
        return {}
    blocked = _blocked_verifier_path_prefixes(sandbox_user, workspace)
    directory = os.path.join(
        _VERIFIER_TOOL_STATE_PARENT, VERIFIER_TOOL_STATE_PREFIX + uuid.uuid4().hex
    )
    overlay = {
        key: verifier_env[key]
        for key in _verifier_tool_state.INPUT_KEYS
        if key in verifier_env
    }
    result = await env.exec(
        f"cd / && python3 -c {shlex.quote(_VERIFIER_TOOL_STATE_SCRIPT)} "
        f"{shlex.quote(_json.dumps(overlay))} {shlex.quote(_json.dumps(blocked))} "
        f"{shlex.quote(directory)}",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    return_code = _exec_return_code(result)
    if return_code == 127 and await _python3_absent(env):
        # A shell-only image: test.sh brings its own Python (uv downloads
        # one), so decide from the paths alone and create the directory in sh.
        found = _verifier_tool_state.overrides(
            overlay, blocked, directory, lexical=True
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
            f"mkdir -m 755 {quoted} && for f in {files}; do "
            ': > "$f" && chmod 644 "$f" || exit 1; done',
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
            )
        ):
            raise ValueError(f"invalid probe response {stdout[:200]!r}")
    except (ValueError, _json.JSONDecodeError) as exc:
        logger.warning(
            "Verifier uv/pip state stays where it was; the plugin guard refuses "
            "plugins installed there: %s",
            exc,
        )
        return {}
    return found


async def _python3_absent(env: Any) -> bool:
    """Whether ``python3`` is provably missing, not merely broken."""
    available = await env.exec(
        "if command -v python3 >/dev/null 2>&1; then exit 0; else exit 1; fi",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    return _exec_return_code(available) == 1


async def _trusted_verifier_path(
    env, sandbox_user: str | None, workspace: str | None
) -> str:
    """Return verifier PATH with trusted image extras preserved.

    Dockerfile PATH additions are accepted only after container-side stat
    checks prove they are root-owned directories and not group/world writable.
    Runtime locations and sandbox-user writable locations stay excluded.
    """
    path_result = await env.exec(
        "printenv PATH", user="root", timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC
    )
    raw_path = path_result.stdout or ""
    if not raw_path.strip():
        return _SAFE_VERIFIER_PATH
    cmd = _trusted_path_extras_cmd(
        raw_path, _blocked_verifier_path_prefixes(sandbox_user, workspace)
    )
    result = await env.exec(cmd, user="root", timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC)
    if _exec_return_code(result) != 0:
        logger.debug(
            "Trusted verifier PATH extras unavailable; using safe PATH.%s",
            _exec_failure_detail(result),
        )
        extras = []
    else:
        try:
            extras = _json.loads(result.stdout or "[]")
        except _json.JSONDecodeError:
            logger.warning(
                "Could not parse trusted verifier PATH extras; using safe PATH"
            )
            extras = []
        if not isinstance(extras, list):
            logger.warning("Invalid trusted verifier PATH extras; using safe PATH")
            extras = []
    return _merge_trusted_verifier_path([e for e in extras if isinstance(e, str)])


async def _trusted_verifier_pythonpath(
    env,
    sandbox_user: str | None,
) -> str:
    """Return filtered PYTHONPATH preserving only trusted image entries.

    Same root-owned, non-world-writable validation as PATH, but does not
    block the workspace — it is already importable via CWD/pytest and
    is chowned to root before verification.
    """
    pp_result = await env.exec(
        "printenv PYTHONPATH 2>/dev/null || true",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    raw_pp = (pp_result.stdout or "").strip()
    if not raw_pp:
        return ""
    blocked = _blocked_verifier_pythonpath_prefixes(sandbox_user)
    cmd = _trusted_path_extras_cmd(raw_pp, blocked)
    result = await env.exec(cmd, user="root", timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC)
    try:
        extras = _json.loads(result.stdout or "[]")
    except _json.JSONDecodeError:
        return ""
    if not isinstance(extras, list):
        return ""
    return ":".join(e for e in extras if isinstance(e, str))


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
    parts = []
    for name in ("stdout", "stderr"):
        value = getattr(result, name, None)
        if value:
            text = str(value).strip()
            if text:
                parts.append(f"{name}: {text[:2000]}")
    if not parts:
        return ""
    return "\n" + "\n".join(parts)


async def _checked_exec(env: Any, command: str, label: str, **kwargs: Any) -> Any:
    result = await env.exec(command, **kwargs)
    return_code = _exec_return_code(result)
    if return_code != 0:
        raise RuntimeError(
            f"{label} exited with rc={return_code}{_exec_failure_detail(result)}"
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


async def clear_verifier_output_dir(
    env: Any,
    label: str = "Verifier setup failed: clearing verifier output directory",
    **kwargs: Any,
) -> Any:
    """Clear verifier outputs while preserving bind mounts.

    This helper is intentionally service-aware: final anti-tamper hardening
    stays on the agent container, but a verifier running in another service may
    still need its own writable ``/logs/verifier`` output directory.
    """
    return await _checked_exec(env, _CLEAR_VERIFIER_DIR_CMD, label, **kwargs)


async def ensure_legacy_app_dir(
    env: Any,
    label: str = "Verifier setup failed: preparing /app",
    **kwargs: Any,
) -> Any:
    """Prepare the legacy verifier ``/app`` rootdir fallback."""
    return await _checked_exec(env, _ENSURE_APP_DIR_CMD, label, **kwargs)


async def cleanup_verifier_python_hooks(
    env: Any,
    task_dir: "Path | str | None",
    label: str = "Verifier setup failed: purging Python injection hooks",
    *,
    timeout_sec: int = VERIFIER_SETUP_TIMEOUT_SEC,
    **kwargs: Any,
) -> Any:
    """Purge agent-injected Python hook files using task hardening settings.

    The conftest purge walks the full rootfs, so the timeout defaults to
    VERIFIER_SETUP_TIMEOUT_SEC (the same budget the scoring path uses in
    harden_before_verify) rather than the caller guessing a value.
    """
    hardening = _read_hardening_config(task_dir)
    return await _checked_exec(
        env, _build_cleanup_cmd(hardening), label, timeout_sec=timeout_sec, **kwargs
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


def _build_cleanup_cmd(hardening: dict[str, bool] | None = None) -> str:
    """Build the cleanup shell command, honoring per-task hardening opt-outs.

    Steps:
      - conftest.py removal outside /tests (skippable via cleanup_conftests=false)
      - *.py purge from /tmp /var/tmp (always — covers module-shadow via cwd)
      - sitecustomize.py/usercustomize.py removal from writable sys.path
      - .pth removal from writable sys.path

    sitecustomize/usercustomize/.pth always run — opt-outs there would broaden
    the attack surface beyond what real-world tasks need.
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
            "find / -path /proc -prune -o -path /sys -prune -o -path /dev -prune -o "
            "-name conftest.py "
            "-not -path '/verifier/*' -not -path '/tests/*' "
            "-exec rm -f -- {} + 2>/dev/null"
        )
    parts.append("find /tmp /var/tmp -name '*.py' -delete 2>/dev/null")
    parts.append(
        'python3 -c "'
        "import sys,os;"
        "[os.remove(os.path.join(d,f)) "
        " for d in sys.path "
        " for f in ('sitecustomize.py','usercustomize.py') "
        " if d and not d.startswith('/usr/lib') and not d.startswith('/usr/local/lib') "
        " and os.path.isfile(os.path.join(d,f))];"
        "[os.remove(os.path.join(d,f)) "
        " for d in sys.path if d and os.path.isdir(d) "
        " for f in os.listdir(d) if f.endswith('.pth') "
        " and not d.startswith('/usr/lib') and not d.startswith('/usr/local/lib') "
        " and os.path.isfile(os.path.join(d,f))]"
        '" 2>/dev/null || true'
    )
    return "; ".join(parts)


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
# copytree is the fallback. ``__WSQ__`` is the shell-quoted workspace.
_RESTORE_WORKSPACE_CMD_TEMPLATE = (
    "if [ -d /testbed_verify ]; then "
    "  rsync -a --delete /testbed_verify/ __WSQ__/ 2>/dev/null || "
    "  python3 -c 'import shutil,sys; "
    'shutil.copytree("/testbed_verify",sys.argv[1],dirs_exist_ok=True)\''
    "  __WSQ__; "
    "fi"
)

# Purge symlinks whose realpath escapes the workspace (e.g. agent-planted
# utils.py -> /tmp/evil.py). In-tree symlinks are preserved — some source trees
# (OTP, lake-packages, etc.) ship them as fixtures. Residual risk: agent plants
# utils.py -> ./evil.py inside workspace. Mitigated by the workspace chown +
# CLEANUP_CMD path scrubbing. The workspace is passed as argv, never spliced.
_PURGE_EXTERNAL_SYMLINKS_SCRIPT = (
    "import os, sys, pathlib;"
    "root = pathlib.Path(sys.argv[1]).resolve();"
    "root_str = str(root) + os.sep;"
    "[p.unlink() for p in root.rglob('*')"
    " if p.is_symlink()"
    " and not (str(p.resolve()) + os.sep).startswith(root_str)"
    " and str(p.resolve()) != str(root)]"
)

# Purge __pycache__ trees that did not exist in the pre-agent baseline, so
# agent-planted .pyc bytecode cannot execute even if PYTHONPYCACHEPREFIX is
# bypassed. Baseline-present caches are kept so tasks whose tests diff workspace
# against /testbed_verify don't break. ``__WSQ__`` is the shell-quoted workspace.
_PURGE_PYCACHE_CMD_TEMPLATE = (
    "if [ -d /testbed_verify ]; then "
    "  find __WSQ__ -type d -name __pycache__ -print0 "
    "  | while IFS= read -r -d '' d; do "
    "      rel=${d#__WSQ__/}; "
    '      [ -d "/testbed_verify/$rel" ] || rm -rf "$d"; '
    "  done; "
    "else "
    "  find __WSQ__ -type d -name '__pycache__'"
    " -exec rm -rf {} + 2>/dev/null; "
    "fi; true"
)


def _restore_workspace_cmd(workspace: str) -> str:
    """Render the full /testbed_verify → workspace restore command."""
    return _RESTORE_WORKSPACE_CMD_TEMPLATE.replace("__WSQ__", shlex.quote(workspace))


def _purge_external_symlinks_cmd(workspace: str) -> str:
    """Render the external-symlink purge command for the workspace."""
    return (
        f"python3 -c {shlex.quote(_PURGE_EXTERNAL_SYMLINKS_SCRIPT)} "
        f"{shlex.quote(workspace)} 2>/dev/null; true"
    )


def _purge_pycache_cmd(workspace: str) -> str:
    """Render the baseline-aware __pycache__ purge command for the workspace."""
    return _PURGE_PYCACHE_CMD_TEMPLATE.replace("__WSQ__", shlex.quote(workspace))


# Selectively adapted from PR #1088, with probe failures kept
# distinct from an empty observation. A zombie has no live writer; an
# unreadable existing /proc entry is unknown, not proof of quiescence.
#
# Reads /proc directly with shell builtins (plus ``id`` and ``sleep``) so it
# runs on busybox and debian-slim images that ship neither procps nor python3;
# pgrep reads the same files. A process matches on its effective UID, like
# ``pgrep -u``. An entry that disappears before its status is read has exited.
_ASSERT_SANDBOX_USER_QUIESCENT_CMD_TEMPLATE = r"""
uid=$(id -u __USER__ 2>/dev/null) || exit 2
case "$uid" in ''|*[!0-9]*|0) echo 'Invalid sandbox UID' >&2; exit 2 ;; esac
if [ ! -r /proc/self/status ]; then
    echo 'Process table is unavailable' >&2
    exit 2
fi
live() {
    for entry in /proc/[0-9]*; do
        pid=${entry##*/}
        case "$pid" in ''|*[!0-9]*) continue ;; esac
        state=
        euid=
        if { while read -r key first second rest || [ -n "$key" ]; do
                case "$key" in
                    State:) state=$first ;;
                    Uid:) euid=$second ;;
                esac
                key=
            done <"$entry/status"; } 2>/dev/null &&
            [ -n "$state" ] && [ -n "$euid" ]; then
            if [ "$euid" = "$uid" ] && [ "$state" != Z ]; then echo "$pid"; fi
        elif [ -d "$entry" ]; then
            echo "Cannot inspect process $pid" >&2
            return 2
        fi
    done
    return 0
}
for sig in TERM KILL; do
    survivors=$(live) || exit 2
    [ -n "$survivors" ] || exit 0
    for pid in $survivors; do kill -s "$sig" "$pid" 2>/dev/null; done
    sleep 1
done
survivors=$(live) || exit 2
[ -n "$survivors" ] || exit 0
echo 'Sandbox-user writers remain:' $survivors >&2
exit 1
""".strip()


async def _kill_sandbox_user_procs(env, sandbox_user: str) -> None:
    """Stop writers and require an affirmative final quiescence observation."""
    user = shlex.quote(sandbox_user)
    safe_path = f"export PATH={shlex.quote(_SAFE_VERIFIER_PATH)}; "
    await _checked_exec(
        env,
        safe_path
        + f"uid=$(id -u {user} 2>/dev/null) || exit 2; "
        + 'case "$uid" in ""|*[!0-9]*|0) echo "Invalid sandbox UID" >&2; exit 2;; esac; '
        + f"pkill -u {user} 2>/dev/null; sleep 1; pkill -9 -u {user} 2>/dev/null || true",
        "Verifier hardening failed: sandbox-user process identity check",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    await _checked_exec(
        env,
        safe_path
        + _ASSERT_SANDBOX_USER_QUIESCENT_CMD_TEMPLATE.replace("__USER__", user),
        "Verifier hardening failed: sandbox-user process quiescence check",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )


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
        await env.exec(
            build_reclaim_caches_cmd(workspace),
            user="root",
            timeout_sec=30,
        )
    except Exception:
        logger.debug("pre-verifier disk reclaim skipped", exc_info=True)


async def _restore_workspace_state(env, workspace: str) -> None:
    """Restore workspace + verifier copy from the pre-agent snapshot."""
    await _restore_build_config(env, workspace)
    await _refresh_verifier_workspace(env, workspace)
    await env.exec(_restore_workspace_cmd(workspace), user="root")


async def _freeze_workspace(env, workspace: str) -> None:
    """Purge external symlinks + stale __pycache__, then chown to root.

    chown workspace to root is belt-and-suspenders against any zombie
    sandbox-user process that survived the pkill above.
    """
    await env.exec(
        _purge_external_symlinks_cmd(workspace),
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    await env.exec(
        _purge_pycache_cmd(workspace),
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    await _checked_exec(
        env,
        f"chown -R root:root {shlex.quote(workspace)}",
        "Verifier hardening failed: freezing workspace ownership",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )


async def _build_verifier_env(
    env, task: "Task", sandbox_user: str | None, workspace: str | None
) -> dict[str, str]:
    """Assemble the hardened verifier env, re-pinning security invariants.

    Task-level verifier env vars merge over the defaults, then the hard
    invariants are re-pinned so a task cannot replace PATH, strip
    -c /dev/null / --confcutdir, re-enable entry-point plugin loading, or
    inject code via breakpoint()/coverage/Django/Celery startup hooks.
    """
    hardened_path = await _trusted_verifier_path(env, sandbox_user, workspace)
    hardened_pythonpath = await _trusted_verifier_pythonpath(env, sandbox_user)
    distro_env = await _distro_pip_env(env)

    verifier_env = dict(VERIFIER_ENV)
    verifier_env.update(distro_env)
    if task.config.verifier.env:
        verifier_env.update(
            {k: os.path.expandvars(v) for k, v in task.config.verifier.env.items()}
        )
    verifier_env["PATH"] = hardened_path
    verifier_env["PYTHONPATH"] = hardened_pythonpath
    verifier_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    verifier_env["PYTHONBREAKPOINT"] = "0"
    verifier_env["COVERAGE_PROCESS_START"] = ""
    verifier_env["DJANGO_SETTINGS_MODULE"] = ""
    verifier_env["CELERY_CONFIG_MODULE"] = ""
    # The verifier's uv and pip state moves to a fresh root-owned directory,
    # away from what the agent could have written (a WORKDIR /root image's
    # ~/.cache/uv, ~/.config/uv/uv.toml, ...); the guard trusts it by path, so
    # what test.sh installs there loads whatever the runtime's mask.
    moved = await _isolate_verifier_tool_state(
        env, task, verifier_env, sandbox_user, workspace
    )
    verifier_env.update(moved)
    tool_state = (
        (os.path.dirname(moved["UV_CACHE_DIR"]),) if "UV_CACHE_DIR" in moved else ()
    )
    # Auto-discover pytest plugins that resolve to root-owned system code, plus
    # task config declarations. Appends -p flags to the hardened base.
    flags = await _discover_pytest_plugin_flags(
        env, task, sandbox_user, workspace, pythonpath=hardened_pythonpath
    )
    # Hardening, and so the guard, lives in ``main`` only (#248). A test.sh in
    # another service could not import a ``-p`` guard and would score 0.
    if task.config.verifier.service == "main":
        guard_directory, flags, guarded = await _install_pytest_plugin_guard(
            env,
            sandbox_user,
            workspace,
            flags,
            verifier_path=hardened_path,
            trusted=tool_state,
        )
        # Only a Python without a copy needs the guard on PYTHONPATH; anywhere
        # else a task's preflight may read the entry as injected startup state.
        if not guarded or _verifier_may_run_uncopied_python(task, hardened_path):
            verifier_env["PYTHONPATH"] = ":".join(
                filter(None, (guard_directory, hardened_pythonpath))
            )
    verifier_env["PYTEST_ADDOPTS"] = _build_pytest_addopts(
        workspace,
        flags,
        verifier_confcutdir=_verifier_confcutdir(task),
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
) -> None:
    """Neutralize agent tampering before running the verifier.

    1. Kill sandbox-user processes (prevent concurrent writes during teardown).
    2. Clear /logs/verifier/ contents while preserving remote bind mounts, and
       prepare the legacy /app rootdir.
    3. Reclaim re-downloadable cache space before the verifier installs deps.
    4. Optionally restore the workspace from the pre-agent snapshot. This is
       destructive to legitimate workspace-edit answers, so it is opt-in.
    5. Purge symlinks and __pycache__ trees from the workspace, then chown it
       to root (belt-and-suspenders against zombie sandbox writes).
    6. Remove injected conftest.py, sitecustomize.py, .pth files.
    7. Merge trusted env vars into task.config.verifier.env.

    Cross-container hardening policy (#248): every step here runs against the
    ``main`` (agent) container only — ``env.exec`` is never passed a
    ``service``. This is deliberate. In multi-container (vulhub-style) tasks
    the agent has a shell only in ``main``; the target/database containers are
    intentionally vulnerable and the agent cannot tamper with them, so they
    need no anti-tamper hardening. ``[verifier].service`` chooses where
    ``test.sh`` *runs*; it does not relocate hardening off ``main``.
    """
    # 1. Kill sandbox-user processes (prevent concurrent writes during teardown).
    if sandbox_user:
        await _kill_sandbox_user_procs(env, sandbox_user)
    # 2. Wipe /logs/verifier/ contents while preserving remote bind mounts, then
    #    prepare the legacy /app rootdir fallback.
    await clear_verifier_output_dir(
        env,
        "Verifier hardening failed: clearing verifier output directory",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    await ensure_legacy_app_dir(
        env,
        "Verifier hardening failed: preparing /app",
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    # 3. Reclaim re-downloadable cache space before the verifier installs deps.
    await _reclaim_disk(env, workspace)
    # 4. Optionally restore the workspace from the pre-agent snapshot (opt-in;
    #    destructive to legitimate workspace-edit answers).
    if workspace and restore_workspace:
        await _restore_workspace_state(env, workspace)
    # 5. Purge symlinks + __pycache__, then chown the workspace to root.
    if workspace:
        await _freeze_workspace(env, workspace)
    # 6. Remove injected conftest.py, sitecustomize.py, .pth files. The rootfs
    #    conftest walk can be slow on network-backed FS, so use the shared
    #    verifier-setup budget (not the default 10s) — same rationale as the
    #    soft-verify path, on the path that actually scores.
    hardening = _read_hardening_config(getattr(task, "task_dir", None))
    await env.exec(
        _build_cleanup_cmd(hardening),
        user="root",
        timeout_sec=VERIFIER_SETUP_TIMEOUT_SEC,
    )
    # 7. Merge trusted env vars into task.config.verifier.env.
    task.config.verifier.env = await _build_verifier_env(
        env, task, sandbox_user, workspace
    )
