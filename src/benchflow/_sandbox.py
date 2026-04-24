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
import os
import re
import shlex
from pathlib import Path
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
        f"export HOME=/home/{sandbox_user} && {agent_launch}"
    )
    quoted = shlex.quote(inner)
    return (
        f"if setpriv --help 2>&1 | grep -q reuid; then"
        f" exec setpriv --reuid={sandbox_user} --regid={sandbox_user}"
        f" --init-groups -- bash -c {quoted};"
        f" else exec su -l {sandbox_user} -c {quoted};"
        f" fi"
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
        timeout_sec=timeout_sec,
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


# ── Verifier hardening ────────────────────────────────────────────────────────

# Trusted env vars for verifier execution — override any agent pollution.
# Intentionally omitted (negative guard in test_verify.py explains why):
# PYTHONHOME.
VERIFIER_ENV: dict[str, str] = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTEST_ADDOPTS": (
        "-c /dev/null "  # block pyproject.toml/pytest.ini/tox.ini/setup.cfg discovery
        "--confcutdir=/tests "  # block conftest.py walk-up beyond /tests
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

# Container-side script to enumerate pre-installed pytest11 entry points.
# Runs after sandbox_user processes are killed, so the agent cannot install
# new packages between enumeration and verification. The sandbox_user cannot
# pip install (not root), so all discovered plugins are image-authored.
_DISCOVER_PYTEST_PLUGINS_SCRIPT = r"""
import json, sys
try:
    from importlib.metadata import entry_points
    try:
        eps = list(entry_points(group='pytest11'))
    except TypeError:
        eps = list(entry_points().get('pytest11', []))
    names = sorted(set(ep.name for ep in eps))
    print(json.dumps(names))
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    print("[]")
""".strip()


def _under_path(path: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(f"{prefix}/")


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


async def _discover_pytest_plugin_flags(env, task: "Task") -> str:
    """Auto-discover pytest plugins from root-owned system packages.

    Runs a container-side script that enumerates pytest11 entry points and
    filters to only those whose dist-info is in a root-owned directory.
    Falls back to task.toml pytest_plugins declarations if discovery fails.
    Replaces the previous hand-curated whitelist mechanism.
    """
    plugins: list[str] = []

    # Container-side auto-discovery
    try:
        result = await env.exec(
            f"python3 -c {shlex.quote(_DISCOVER_PYTEST_PLUGINS_SCRIPT)}",
            user="root", timeout_sec=15,
        )
        if result.stderr:
            logger.debug(f"Plugin discovery stderr: {result.stderr.strip()}")
        discovered = _json.loads(result.stdout or "[]")
        if isinstance(discovered, list):
            plugins.extend(p for p in discovered if isinstance(p, str) and p.strip())
        logger.info(f"Discovered {len(plugins)} pytest plugins from container")
    except Exception as e:
        logger.warning(f"Pytest plugin discovery failed, using task.toml fallback: {e}")

    # Merge task.toml declarations as fallback
    declared = getattr(task.config.verifier, "pytest_plugins", None)
    if declared:
        for name in declared:
            if isinstance(name, str) and name.strip() and name.strip() not in plugins:
                plugins.append(name.strip())

    return " ".join(f"-p {shlex.quote(p)}" for p in plugins)


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
            "cat /etc/os-release 2>/dev/null || true", user="root", timeout_sec=5
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


async def _trusted_verifier_path(
    env, sandbox_user: str | None, workspace: str | None
) -> str:
    """Return verifier PATH with trusted image extras preserved.

    Dockerfile PATH additions are accepted only after container-side stat
    checks prove they are root-owned directories and not group/world writable.
    Runtime locations and sandbox-user writable locations stay excluded.
    """
    path_result = await env.exec("printenv PATH", user="root", timeout_sec=10)
    raw_path = path_result.stdout or ""
    if not raw_path.strip():
        return _SAFE_VERIFIER_PATH
    cmd = _trusted_path_extras_cmd(
        raw_path, _blocked_verifier_path_prefixes(sandbox_user, workspace)
    )
    result = await env.exec(cmd, user="root", timeout_sec=10)
    try:
        extras = _json.loads(result.stdout or "[]")
    except _json.JSONDecodeError:
        logger.warning("Could not parse trusted verifier PATH extras; using safe PATH")
        extras = []
    if not isinstance(extras, list):
        logger.warning("Invalid trusted verifier PATH extras; using safe PATH")
        extras = []
    return _merge_trusted_verifier_path([e for e in extras if isinstance(e, str)])


async def _trusted_verifier_pythonpath(
    env, sandbox_user: str | None,
) -> str:
    """Return filtered PYTHONPATH preserving only trusted image entries.

    Same root-owned, non-world-writable validation as PATH, but does not
    block the workspace — it is already importable via CWD/pytest and
    is chowned to root before verification.
    """
    pp_result = await env.exec("printenv PYTHONPATH 2>/dev/null || true", user="root", timeout_sec=10)
    raw_pp = (pp_result.stdout or "").strip()
    if not raw_pp:
        return ""
    blocked = _blocked_verifier_pythonpath_prefixes(sandbox_user)
    cmd = _trusted_path_extras_cmd(raw_pp, blocked)
    result = await env.exec(cmd, user="root", timeout_sec=10)
    try:
        extras = _json.loads(result.stdout or "[]")
    except _json.JSONDecodeError:
        return ""
    if not isinstance(extras, list):
        return ""
    return ":".join(e for e in extras if isinstance(e, str))


# Wipe and recreate /logs/verifier/ before the verifier runs.
# rm -rf severs hardlinks, removes symlink replacements, and eliminates
# variant filenames/subdirs the agent may have pre-staged.
_CLEAR_VERIFIER_DIR_CMD = (
    "rm -rf /logs/verifier && mkdir -p /logs/verifier && chmod 777 /logs/verifier"
)

# Remove injected conftest.py, sitecustomize.py/usercustomize.py, and .pth
# files from writable sys.path entries (preserves /usr/lib, /usr/local/lib).
# Also purge *.py from temp dirs: covers module-shadow via non-workspace cwd.
CLEANUP_CMD = (
    "find / -name conftest.py -not -path '/tests/*' -delete 2>/dev/null; "
    "find /tmp /var/tmp -name '*.py' -delete 2>/dev/null; "
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
    env,
    task: "Task",
    sandbox_user: str | None,
    workspace: str | None = None,
    # Default false because SkillsBench/TB2-style answers often are workspace
    # edits. Going forward, enforce true only via an explicit task/benchmark
    # contract, e.g. task.toml [verifier] restore_workspace = true after an
    # oracle/diff audit proves the answer is not stored in the workspace.
    restore_workspace: bool = False,
) -> None:
    """Neutralize agent tampering before running the verifier.

    1. Kill sandbox-user processes (prevent concurrent writes during teardown).
    2. Assert all sandbox-user processes are dead, then wipe/recreate
       /logs/verifier/ with a clean root-owned directory.
    3. Optionally restore the workspace from the pre-agent snapshot. This is
       destructive to legitimate workspace-edit answers, so it is opt-in.
    4. Purge symlinks and __pycache__ trees from workspace.
    5. chown workspace to root (belt-and-suspenders against zombie sandbox writes).
    6. Remove injected conftest.py, sitecustomize.py, .pth files.
    7. Merge trusted env vars into task.config.verifier.env.
    """

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
    if workspace and restore_workspace:
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
            f"  python3 -c 'import shutil,sys; shutil.copytree(\"/testbed_verify\",sys.argv[1],dirs_exist_ok=True)'"
            f"  {shlex.quote(workspace)}; "
            f"fi",
            user="root",
        )
    if workspace:
        # Purge symlinks whose realpath escapes the workspace (e.g. agent-planted
        # utils.py -> /tmp/evil.py). In-tree symlinks are preserved — some
        # source trees (OTP, lake-packages, etc.) ship them as fixtures.
        # Residual risk: agent plants utils.py -> ./evil.py inside workspace.
        # Mitigated by the workspace chown below + CLEANUP_CMD path scrubbing.
        purge_external_symlinks = (
            "import os, sys, pathlib;"
            "root = pathlib.Path(sys.argv[1]).resolve();"
            "root_str = str(root) + os.sep;"
            "[p.unlink() for p in root.rglob('*')"
            " if p.is_symlink()"
            " and not (str(p.resolve()) + os.sep).startswith(root_str)"
            " and str(p.resolve()) != str(root)]"
        )
        await env.exec(
            f"python3 -c {shlex.quote(purge_external_symlinks)} "
            f"{shlex.quote(workspace)} 2>/dev/null; true",
            user="root",
        )
        # Purge __pycache__ trees that did not exist in the pre-agent baseline,
        # so agent-planted .pyc bytecode cannot execute even if
        # PYTHONPYCACHEPREFIX is bypassed. Baseline-present caches are kept so
        # tasks whose tests diff workspace against /testbed_verify don't break.
        await env.exec(
            f"if [ -d /testbed_verify ]; then "
            f"  find {shlex.quote(workspace)} -type d -name __pycache__ -print0 "
            f"  | while IFS= read -r -d '' d; do "
            f"      rel=${{d#{shlex.quote(workspace)}/}}; "
            f'      [ -d "/testbed_verify/$rel" ] || rm -rf "$d"; '
            f"  done; "
            f"else "
            f"  find {shlex.quote(workspace)} -type d -name '__pycache__'"
            f" -exec rm -rf {{}} + 2>/dev/null; "
            f"fi; true",
            user="root",
        )
        # chown workspace to root: belt-and-suspenders against any zombie
        # sandbox-user process that survived the pkill above.
        await env.exec(
            f"chown -R root:root {shlex.quote(workspace)}",
            user="root",
        )
    await env.exec(CLEANUP_CMD, user="root", timeout_sec=10)

    hardened_path = await _trusted_verifier_path(env, sandbox_user, workspace)
    hardened_pythonpath = await _trusted_verifier_pythonpath(env, sandbox_user)
    distro_env = await _distro_pip_env(env)

    verifier_env = dict(VERIFIER_ENV)
    verifier_env.update(distro_env)
    if task.config.verifier.env:
        verifier_env.update(task.config.verifier.env)
    # Hard security invariants — re-pin after task-env merge so a task cannot
    # replace PATH, strip -c /dev/null / --confcutdir, re-enable entry-point
    # plugin loading, or inject code via breakpoint()/coverage/Django/Celery
    # startup hooks.
    verifier_env["PATH"] = hardened_path
    verifier_env["PYTHONPATH"] = hardened_pythonpath
    verifier_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    verifier_env["PYTHONBREAKPOINT"] = "0"
    verifier_env["COVERAGE_PROCESS_START"] = ""
    verifier_env["DJANGO_SETTINGS_MODULE"] = ""
    verifier_env["CELERY_CONFIG_MODULE"] = ""
    # Auto-discover pytest plugins from root-owned system packages and
    # task.toml declarations. Appends -p flags to the hardened base.
    base_addopts = VERIFIER_ENV["PYTEST_ADDOPTS"]
    flags = await _discover_pytest_plugin_flags(env, task)
    if flags:
        verifier_env["PYTEST_ADDOPTS"] = base_addopts + f" {flags}"
    else:
        verifier_env["PYTEST_ADDOPTS"] = base_addopts
    task.config.verifier.env = verifier_env
