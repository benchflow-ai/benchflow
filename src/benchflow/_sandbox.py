"""Sandbox user setup, path lockdown, and verifier hardening.

Owns the "agent runs as non-root" lifecycle:
    - Creating the sandbox user and copying root's tooling into its home
    - Building the privilege-drop wrapper (setpriv / su) for agent launch
    - Locking down solution/test paths so the sandbox user cannot read them
    - Hardening the environment before the verifier runs (kill sandbox
      processes, scrub injected conftest/.pth files, install trusted env)

Plus the supporting cast that only this module touches:
    - Path validation against shell-injection (_validate_locked_path)
    - Effective lockdown path resolution (_resolve_locked_paths)
    - VERIFIER_ENV / CLEANUP_CMD module constants (consumed only by
      harden_before_verify; never read directly from sdk.py)

Does not own:
    - Spawning the agent process — see _acp_run.py (which imports
      build_priv_drop_cmd from here as the one allowed horizontal hop)
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
    """Validate a locked path — reject injection and traversal."""
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

    setpriv (util-linux, Debian/Ubuntu) execs directly with no parent process.
    su -l is the universal fallback (works on Alpine/BusyBox too).
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

    Runs after all root-level setup but before agent launch.
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

# Files whose pre-agent state is snapshotted and restored before verification.
# Covers all common build backends to prevent setup.py / pyproject.toml hijacks.
_BUILD_CONFIG_FILES = (
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "noxfile.py",
    "hatch.toml",
    "flit.ini",
    "MANIFEST.in",
)
# chmod 700: root-only so the sandbox_user cannot read or overwrite the snapshot.
_SNAPSHOT_DIR = "/tmp/.benchflow_build_snapshot"
_SNAPSHOT_MANIFEST = f"{_SNAPSHOT_DIR}/manifest.json"


async def _snapshot_build_config(env, workspace: str) -> None:
    """Snapshot build-config files before the agent runs.

    Absence/presence is recorded in a separate manifest.json rather than
    embedding a sentinel string inside the captured files — that way an agent
    cannot forge "this file was absent" by planting a magic string in setup.py.

    ORDERING INVARIANT: must be called before the agent launches. The agent
    owns the workspace files (chown'd by setup_sandbox_user) and could modify
    them the instant it starts. Calling this after agent launch would capture
    a potentially-tainted snapshot.
    """
    # chmod 700: root-only so the sandbox_user cannot tamper with the snapshot.
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
    # Write manifest as JSON — only booleans, never file content.
    manifest_json = _json.dumps(manifest)
    await env.exec(
        f"echo {shlex.quote(manifest_json)} > {_SNAPSHOT_MANIFEST}",
        user="root",
    )


async def _restore_build_config(env, workspace: str) -> None:
    """Restore build-config files to their pre-agent state.

    Reads the manifest written by _snapshot_build_config to determine which
    files existed before the agent ran. Files that existed are restored from
    the snapshot; files that didn't are removed if the agent created them.
    """
    result = await env.exec(f"cat {_SNAPSHOT_MANIFEST}", user="root")
    manifest: dict[str, bool] = _json.loads(result.stdout)
    for fname in _BUILD_CONFIG_FILES:
        src = f"{_SNAPSHOT_DIR}/{fname}"
        dst = f"{workspace}/{fname}"
        if manifest.get(fname):
            # File existed pre-agent: restore the snapshot copy.
            cmd = (
                f"cp --preserve=timestamps {src} {dst} "
                f"&& chown root:root {dst} && chmod 644 {dst}"
            )
        else:
            # File did not exist pre-agent: remove anything the agent created.
            cmd = f"rm -f {dst}"
        await env.exec(cmd, user="root")


# ── Verifier hardening ────────────────────────────────────────────────────────

# Trusted env vars for verifier execution — override any agent pollution.
# Intentionally omitted (negative guards in test_verify.py explain why):
# PYTEST_DISABLE_PLUGIN_AUTOLOAD, PYTHONNOUSERSITE, PYTHONHOME.
VERIFIER_ENV: dict[str, str] = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTEST_ADDOPTS": (
        "-c /dev/null "  # block pyproject.toml/pytest.ini/tox.ini/setup.cfg discovery
        "--confcutdir=/tests "  # block conftest.py walk-up beyond /tests
        "--rootdir=/tests "
        "-p no:cacheprovider"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": "",
    "PYTHONSTARTUP": "",
    "PYTHONSAFEPATH": "1",  # drop implicit '' (cwd) from sys.path
    "LD_PRELOAD": "",
    "LD_LIBRARY_PATH": "",
}

# Wipe and recreate /logs/verifier/ before the verifier runs.
# rm -rf severs hardlinks (G1), replaces symlink replacements (G2), and
# removes all variant filenames and subdirectories (G4).
# chmod 777 restores Harbor's expectation for the verifier write.
_CLEAR_VERIFIER_DIR_CMD = (
    "rm -rf /logs/verifier && mkdir -p /logs/verifier && chmod 777 /logs/verifier"
)

# Cleanup command for pytest hook / Python startup injection.
# Removes conftest.py outside /tests, sitecustomize.py/usercustomize.py
# and .pth files from writable sys.path entries (preserves /usr/lib,
# /usr/local/lib).
CLEANUP_CMD = (
    "find / -maxdepth 5 -name conftest.py -not -path '/tests/*' -delete 2>/dev/null; "
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
    2. Assert all sandbox-user processes are dead, then wipe and recreate
       /logs/verifier/ so the verifier always writes into a clean directory
       owned by root — severs hardlinks, removes symlink replacements, and
       eliminates variant filenames the agent may have pre-staged.
    3. Restore build-config files (setup.py, pyproject.toml, etc.) to their
       pre-agent state so that pip/test runners see the originals, not agent-
       modified versions. Only runs when workspace is provided.
    4. Remove injected conftest.py, sitecustomize.py, .pth files.
    5. Merge trusted env vars into task.config.verifier.env.
    """
    if sandbox_user:
        await env.exec(
            f"pkill -u {sandbox_user} 2>/dev/null; "
            f"sleep 1; pkill -9 -u {sandbox_user} 2>/dev/null || true",
            timeout_sec=10,
        )
        # Second pass: catch any processes that slipped through (e.g. cron/at
        # jobs started during the agent phase).
        await env.exec(
            f"! pgrep -u {sandbox_user} > /dev/null 2>&1 || "
            f"(sleep 1 && pkill -9 -u {sandbox_user}; sleep 1)",
            user="root",
        )
    # Wipe and recreate /logs/verifier/ with a clean root-owned directory.
    await env.exec(_CLEAR_VERIFIER_DIR_CMD, user="root")
    if workspace:
        await _restore_build_config(env, workspace)
    await env.exec(CLEANUP_CMD, timeout_sec=10)

    verifier_env = dict(VERIFIER_ENV)
    if task.config.verifier.env:
        verifier_env.update(task.config.verifier.env)
    task.config.verifier.env = verifier_env
