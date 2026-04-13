"""Sandbox user setup, path lockdown, and verifier hardening.

Owns the "agent runs as non-root" lifecycle:
    - Creating the sandbox user and copying root's tooling into its home
    - Building the privilege-drop wrapper (setpriv / su) for agent launch
    - Locking down solution/test paths so the sandbox user cannot read them
    - Hardening the environment before the verifier runs

Does not own:
    - Spawning the agent process — see _acp_run.py
    - Running the verifier itself — see SDK._verify
"""

import json as _json
import logging
import re
import shlex
from typing import TYPE_CHECKING

from benchflow.agents.registry import get_sandbox_home_dirs

if TYPE_CHECKING:
    from harbor.models.task.task import Task

logger = logging.getLogger(__name__)


# ── Path lockdown defaults and validation ─────────────────────────────────────

_DEFAULT_LOCKED = ["/solution", "/tests"]
_SAFE_PATH_RE = re.compile(r"^/[a-zA-Z0-9_./*?\-]+(/[a-zA-Z0-9_./*?\-]+)*$")


def _validate_locked_path(p: str) -> None:
    """Reject injection and traversal in a locked path."""
    import os

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
    - sandbox_user set, paths=None → defaults (/solution, /tests)
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


# ── Sandbox user + privilege drop ─────────────────────────────────────────────


def build_priv_drop_cmd(agent_launch: str, sandbox_user: str) -> str:
    """Build a shell command that drops to sandbox_user via setpriv or su.

    setpriv (util-linux) execs directly; su -l is the fallback for Alpine/BusyBox.
    No outer sh -c wrapper — DockerProcess wraps in bash -c already.
    """
    inner = (
        f"export HOME=/home/{sandbox_user} && cd /home/{sandbox_user} && {agent_launch}"
    )
    quoted = shlex.quote(inner)
    return (
        f"if setpriv --help 2>&1 | grep -q reuid; then"
        f" exec setpriv --reuid={sandbox_user} --regid={sandbox_user}"
        f" --init-groups -- bash -c {quoted};"
        f" else exec su -l {sandbox_user} -c {quoted};"
        f" fi"
    )


async def setup_sandbox_user(env, sandbox_user: str, workspace: str) -> str:
    """Create non-root sandbox user, grant workspace access. Return agent_cwd."""
    if not re.match(r"^[a-z_][a-z0-9_-]*$", sandbox_user):
        raise ValueError(
            f"Invalid sandbox_user: {sandbox_user!r} (must be alphanumeric)"
        )
    logger.info(f"Setting up sandbox user: {sandbox_user}")
    await env.exec(
        f"id -u {sandbox_user} >/dev/null 2>&1 || "
        f"useradd -m -s /bin/bash {sandbox_user} && "
        f"mkdir -p /home/{sandbox_user}/.local/bin && "
        "if [ -d /root/.local/bin ]; then "
        f"cp -aL /root/.local/bin/. /home/{sandbox_user}/.local/bin/ 2>/dev/null || true; fi && "
        "if [ -d /root/.nvm ]; then "
        f"cp -a /root/.nvm/. /home/{sandbox_user}/.nvm/ 2>/dev/null || true; fi && "
        f"for d in {' '.join(sorted(get_sandbox_home_dirs()))}; do "
        f"if [ -d /root/$d ]; then mkdir -p /home/{sandbox_user}/$d && "
        f"cp -a /root/$d/. /home/{sandbox_user}/$d/ 2>/dev/null || true; fi; done && "
        f"chown -R {sandbox_user}:{sandbox_user} /home/{sandbox_user} && "
        f"chown -R {sandbox_user}:{sandbox_user} {shlex.quote(workspace)}",
        timeout_sec=30,
    )
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


# ── Build-config snapshot / restore (Tier 2) ─────────────────────────────────

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
    """
    await env.exec(
        f"mkdir -p {_SNAPSHOT_DIR} && chmod 700 {_SNAPSHOT_DIR}",
        user="root",
    )
    manifest: dict[str, bool] = {}
    for fname in _BUILD_CONFIG_FILES:
        src = f"{workspace}/{fname}"
        dst = f"{_SNAPSHOT_DIR}/{fname}"
        result = await env.exec(
            f"if [ -f {src} ]; then "
            f"  cp --preserve=all {src} {dst} && echo present; "
            f"else "
            f"  echo absent; "
            f"fi",
            user="root",
        )
        manifest[fname] = result.stdout.strip() == "present"
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


# ── Verifier user setup (Tier 3) ─────────────────────────────────────────────

# Dedicated non-root OS user for the verifier phase.
# Tasks that need root (e.g. apt-get in test.sh) can opt out via
# `[verifier] user = "root"` in task.toml — that will log a warning.
_VERIFIER_USER = "verifier"


async def _setup_verifier_user(env, workspace: str = "/testbed") -> None:
    """Create a dedicated non-root OS user to run the verifier phase.

    Called once after setup_sandbox_user, before agent launch.
    - Locks /logs/ parent so sandbox_user cannot rename /logs/verifier/ out.
    - Seeds /testbed_verify as a root-owned read-only copy of the workspace
      so harden_before_verify can restore the full workspace to pre-agent
      canonical state before freezing.
    """
    cmds = [
        # Lock /logs/ parent: sandbox_user cannot rename /logs/verifier/ out.
        "chown root:root /logs && chmod 755 /logs",
        # Create group first — useradd --gid requires it to pre-exist.
        f"getent group {_VERIFIER_USER} &>/dev/null || groupadd -r {_VERIFIER_USER}",
        # System user: no login shell, no supplementary groups.
        f"id {_VERIFIER_USER} &>/dev/null || "
        f"useradd -r -s /bin/false "
        f"--gid {_VERIFIER_USER} --groups '' {_VERIFIER_USER}",
        # Wipe any pre-staged home dir the agent may have planted before useradd.
        f"rm -rf /home/{_VERIFIER_USER} && "
        f"mkdir -p /home/{_VERIFIER_USER} && "
        f"chown {_VERIFIER_USER}:{_VERIFIER_USER} /home/{_VERIFIER_USER} && "
        f"chmod 700 /home/{_VERIFIER_USER}",
        # /logs/verifier/ must be writable so test.sh can write reward.txt.
        # Tier 1 resets this to 777 at hardentime; ownership is set here as
        # a structural guard before the agent starts.
        f"mkdir -p /logs/verifier && "
        f"chown {_VERIFIER_USER}:{_VERIFIER_USER} /logs/verifier && "
        f"chmod 700 /logs/verifier",
        # Seed root-owned read-only workspace copy from the actual workspace
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


# ── Verifier hardening ────────────────────────────────────────────────────────

# Trusted env vars for verifier execution — override any agent pollution.
# Intentionally omitted (negative guard in test_verify.py explains why):
# PYTHONHOME.
VERIFIER_ENV: dict[str, str] = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTEST_ADDOPTS": (
        "-c /dev/null "  # block pyproject.toml/pytest.ini/tox.ini/setup.cfg discovery
        "--confcutdir=/tests "  # block conftest.py walk-up beyond /tests
        "--rootdir=/tests "
        "-p no:cacheprovider"
    ),
    # Block pytest11 entry-point plugins. An agent can modify a pre-installed
    # package's plugin source to forge a reward; -c /dev/null does not block
    # entry-point registration. Tasks that need specific plugins declare them
    # in task.toml [verifier] pytest_plugins = [...].
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    # Redirect .pyc cache reads/writes to a non-existent directory so
    # agent-pre-compiled __pycache__ bytecode is never found or executed.
    "PYTHONPYCACHEPREFIX": "/nonexistent",
    "PYTHONPATH": "",
    "PYTHONSTARTUP": "",
    "PYTHONSAFEPATH": "1",  # drop implicit '' (cwd) from sys.path
    "LD_PRELOAD": "",
    "LD_LIBRARY_PATH": "",
    # Prevent pip from writing to user site-packages during pip install -e .
    "PYTHONNOUSERSITE": "1",
    "PIP_USER": "0",
    "PIP_NO_USER_CONFIG": "1",
    # Force HOME to a non-existent path so pip cannot read any pre-staged pip.conf.
    "HOME": "/nonexistent",
    # Disable breakpoint() — any other value imports an arbitrary callable.
    "PYTHONBREAKPOINT": "0",
    # Prevent coverage.py from importing a config file as Python on startup.
    "COVERAGE_PROCESS_START": "",
    # Prevent Django/Celery from importing an agent-controlled module at startup.
    "DJANGO_SETTINGS_MODULE": "",
    "CELERY_CONFIG_MODULE": "",
}

# Wipe and recreate /logs/verifier/ before the verifier runs.
# rm -rf severs hardlinks, removes symlink replacements, and eliminates
# variant filenames/subdirs the agent may have pre-staged.
_CLEAR_VERIFIER_DIR_CMD = (
    "rm -rf /logs/verifier && mkdir -p /logs/verifier && chmod 777 /logs/verifier"
)

# Remove injected conftest.py, sitecustomize.py/usercustomize.py, and .pth
# files from writable sys.path entries (preserves /usr/lib, /usr/local/lib).
CLEANUP_CMD = (
    "find / -name conftest.py -not -path '/tests/*' -delete 2>/dev/null; "
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


async def harden_before_verify(
    env, task: "Task", sandbox_user: str | None, workspace: str | None = None
) -> None:
    """Neutralize agent tampering before running the verifier.

    1. Kill sandbox-user processes (prevent concurrent writes during teardown).
    2. Assert all sandbox-user processes are dead, then wipe/recreate
       /logs/verifier/ with a clean root-owned directory.
    3. Restore build-config files to pre-agent state (if workspace provided).
    4. Sync restored build-config files into /testbed_verify (if workspace provided).
    4b.Full workspace restore from /testbed_verify — resets ALL source files to
       pre-agent canonical state, not just the build-config subset.
    4c.Purge symlinks and __pycache__ trees from workspace.
    5. Freeze workspace read-only (chown root + chmod a-w) so editable-install
       source files cannot execute agent code during the verify phase.
    6. Remove injected conftest.py, sitecustomize.py, .pth files.
    7. Merge trusted env vars into task.config.verifier.env.
    8. Set verifier user to the dedicated non-root account unless opted out.
    """
    # Resolve verifier user before any mutation so hooks cannot override it.
    _resolved_verifier_user = (
        task.config.verifier.user
        if task.config.verifier.user is not None
        else _VERIFIER_USER
    )
    if _resolved_verifier_user == "root":
        logger.warning("verifier running as root (opt-out set in task.toml)")

    if sandbox_user:
        await env.exec(
            f"pkill -u {sandbox_user} 2>/dev/null; "
            f"sleep 1; pkill -9 -u {sandbox_user} 2>/dev/null || true",
            timeout_sec=10,
        )
        # Second pass: catch any processes that slipped through (e.g. cron/at jobs).
        await env.exec(
            f"! pgrep -u {sandbox_user} > /dev/null 2>&1 || "
            f"(sleep 1 && pkill -9 -u {sandbox_user}; sleep 1)",
            user="root",
        )
    # Wipe and recreate /logs/verifier/ with a clean root-owned directory.
    await env.exec(_CLEAR_VERIFIER_DIR_CMD, user="root")
    if workspace:
        await _restore_build_config(env, workspace)
        await _refresh_verifier_workspace(env, workspace)
        # Full workspace restore from /testbed_verify (the pre-agent canonical
        # copy) so ALL source files — not just build-config files — are reset
        # to their pre-agent state before freezing.  Closes the editable-install
        # source modification vector (F2): agent rewrites /testbed/src/pkg/x.py
        # during its phase; this step restores x.py to canonical before verify.
        await env.exec(
            f"if [ -d /testbed_verify ]; then "
            f"  rsync -a --delete /testbed_verify/ {shlex.quote(workspace)}/ 2>/dev/null || "
            f"  (rm -rf {shlex.quote(workspace)} && cp -a /testbed_verify/. {shlex.quote(workspace)}/); "
            f"fi",
            user="root",
        )
        # Purge symlinks so an agent-planted symlink pointing to a writable
        # external target (e.g. utils.py -> /tmp/evil.py) cannot be followed
        # during import after the workspace is frozen.
        await env.exec(
            f"find {shlex.quote(workspace)} -type l -delete 2>/dev/null; true",
            user="root",
        )
        # Purge __pycache__ trees so pre-compiled .pyc bytecode cannot execute
        # even if PYTHONPYCACHEPREFIX is bypassed (defense-in-depth).
        await env.exec(
            f"find {shlex.quote(workspace)} -type d -name '__pycache__'"
            f" -exec rm -rf {{}} + 2>/dev/null; true",
            user="root",
        )
        # Freeze the workspace read-only so agent-modified source files in
        # editable installs (e.g. /testbed/src/pkg/utils.py) cannot execute
        # during the verifier phase. Agent is already dead at this point.
        await env.exec(
            f"chown -R root:root {shlex.quote(workspace)} && "
            f"chmod -R a-w {shlex.quote(workspace)}",
            user="root",
        )
    await env.exec(CLEANUP_CMD, user="root", timeout_sec=10)

    verifier_env = dict(VERIFIER_ENV)
    if task.config.verifier.env:
        verifier_env.update(task.config.verifier.env)
    # Hard security invariants — re-pin after task-env merge so a task cannot
    # strip -c /dev/null / --confcutdir, re-enable entry-point plugin loading,
    # or inject code via breakpoint()/coverage/Django/Celery startup hooks.
    verifier_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    verifier_env["PYTHONBREAKPOINT"] = "0"
    verifier_env["COVERAGE_PROCESS_START"] = ""
    verifier_env["DJANGO_SETTINGS_MODULE"] = ""
    verifier_env["CELERY_CONFIG_MODULE"] = ""
    # Re-enable explicitly declared plugins by appending -p flags to the
    # hardened base — never to a task-supplied PYTEST_ADDOPTS.
    allowed_plugins = task.config.verifier.pytest_plugins or []
    base_addopts = VERIFIER_ENV["PYTEST_ADDOPTS"]
    if allowed_plugins:
        flags = " ".join(f"-p {shlex.quote(p)}" for p in allowed_plugins)
        verifier_env["PYTEST_ADDOPTS"] = base_addopts + f" {flags}"
    else:
        verifier_env["PYTEST_ADDOPTS"] = base_addopts
    task.config.verifier.env = verifier_env
    task.config.verifier.user = _resolved_verifier_user
