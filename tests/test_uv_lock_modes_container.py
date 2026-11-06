"""After the in-sandbox proxy install nothing under /opt/runtime is another
user's to write (DET-R1), in a real container with the real install.

uv 0.13.1 makes every lock file it creates writable by everyone (mode 666,
past the umask): one per cached wheel, the cache's, the managed Python's and
each venv's. Since C8 put uv's cache in /opt/runtime, the in-sandbox proxy
install left them there, and the agent install's root-only check of
/opt/runtime refused every later agent install ("not root-only:
.../granian/....lock"): codex-acp and claude-agent-acp under an allowlist or
denylist. The install is BenchFlow's real one (pinned uv, managed CPython,
the locked proxy set), so it needs the network; skipped without a Docker
daemon.
"""

from __future__ import annotations

import subprocess
import uuid

import pytest

from tests.test_c8_protected_runtime_container import (
    LABEL,
    _docker_ready,
    _dx,
    _sandbox,
    pytestmark,  # noqa: F401  (skipped without docker)
)

IMAGE = "ubuntu:24.04"
WRITABLE = "find /opt/runtime ! -type l \\( -perm -0020 -o -perm -0002 \\) -print"


@pytest.fixture
def box():
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "det-r1-uv-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        [
            "docker", "run", "-d", "--name", name, "--label", LABEL,
            IMAGE, "sleep", "1800",
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {IMAGE}: {run.stderr.strip()}")
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


async def _install_proxy(sandbox) -> None:
    """One role's in-trial proxy install, as ``_start_sandbox_litellm`` runs it."""
    from benchflow.providers import litellm_runtime as runtime
    from benchflow.providers.proxy_requirements import proxy_install

    install = proxy_install(None)
    runtime_dir = f"{runtime.LITELLM_SANDBOX_ROOT}/det-r1-{uuid.uuid4().hex[:8]}"
    paths = await runtime._upload_runtime_files_to_sandbox(
        sandbox, runtime_dir=runtime_dir, config={}
    )
    requirements = f"{runtime_dir}/{install.file_name}"
    await runtime._upload_text(sandbox, install.text, requirements, ".txt")
    python = await runtime._ensure_sandbox_litellm(
        sandbox,
        venv_dir=paths["venv"],
        requirements_path=requirements,
        extra_requirements=install.extra_requirements,
        install_timeout_sec=900,
    )
    assert python == f"{paths['venv']}/bin/python"


async def _assert_root_only(box: str, sandbox) -> None:
    from benchflow._root_only import root_only_tree_check

    writable = _dx(box, WRITABLE)
    assert writable.returncode == 0, writable.stderr
    assert writable.stdout == "", f"writable by another user:\n{writable.stdout}"
    # The agent install's own check of the whole tree (registry's JS agents).
    check = await sandbox.exec(
        root_only_tree_check("/opt/runtime", "BenchFlow's agent runtime folder"),
        user="root",
    )
    assert check.return_code == 0, check.stderr


@pytest.mark.asyncio
async def test_nothing_under_opt_benchflow_is_writable_after_the_proxy_install(box):
    sandbox = _sandbox(box)

    await _install_proxy(sandbox)

    # uv did make its lock files (one per cached wheel among them), so the
    # check below is not vacuous.
    locks = _dx(box, "find /opt/runtime/cache/uv -name '*.lock' -type f | wc -l")
    assert int(locks.stdout.strip() or 0) > 1, locks
    await _assert_root_only(box, sandbox)
    cache = _dx(box, "stat -c %a /opt/runtime/cache")
    assert cache.stdout.strip() == "700", cache


@pytest.mark.asyncio
async def test_a_second_roles_proxy_install_reuses_the_cache_and_stays_root_only(
    box,
):
    """A second role's install in the same sandbox: the cache from the first,
    opened by an agent install's ``chmod -R a+rX /opt/runtime``, passes the
    install's own check, is private again while uv runs, and is sealed after."""
    sandbox = _sandbox(box)
    await _install_proxy(sandbox)
    opened = _dx(box, "chmod -R a+rX /opt/runtime")
    assert opened.returncode == 0, opened.stderr

    await _install_proxy(sandbox)

    cache = _dx(box, "stat -c %a /opt/runtime/cache")
    assert cache.stdout.strip() == "700", cache
    await _assert_root_only(box, sandbox)


@pytest.mark.asyncio
async def test_the_seal_takes_write_off_uv_git_locks_and_nothing_else(box):
    """From the 0.9.0 journeys: openhands is installed from git with uv, which
    leaves ``cache/uv/git-v1/locks/<hash>`` writable by everyone (mode 666,
    no suffix). The seal took only ``*.lock`` names, so the in-sandbox model
    proxy refused the cache ("is not root's alone") and openhands could not
    run a model on Daytona. The layout is the one a real openhands install
    leaves; a file of the cache that is not a lock keeps its mode, so the
    root-only check still names it."""
    from benchflow._root_only import root_only_tree_check
    from benchflow._uv_pin import UV_CACHE_DIR, uv_lock_seal_command

    git_lock = f"{UV_CACHE_DIR}/git-v1/locks/2ed127526e4df323"
    wheel_lock = f"{UV_CACHE_DIR}/wheels-v7/.lock"
    stray = f"{UV_CACHE_DIR}/git-v1/checkouts/stray.txt"
    made = _dx(
        box,
        f"umask 022 && mkdir -p {UV_CACHE_DIR}/git-v1/locks {UV_CACHE_DIR}/wheels-v7 "
        f"{UV_CACHE_DIR}/git-v1/checkouts && chmod 700 /opt/runtime/cache && "
        f"umask 0 && : > {git_lock} && : > {wheel_lock} && printf x > {stray}",
    )
    assert made.returncode == 0, made.stderr
    sandbox = _sandbox(box)

    sealed = await sandbox.exec(uv_lock_seal_command(), user="root")
    assert sealed.return_code == 0, sealed.stderr

    modes = _dx(box, f"stat -c '%a %n' {git_lock} {wheel_lock} {stray}")
    assert modes.stdout.split("\n")[:3] == [
        f"644 {git_lock}",
        f"644 {wheel_lock}",
        f"666 {stray}",
    ]
    check = root_only_tree_check(UV_CACHE_DIR, "BenchFlow's uv cache")
    refused = await sandbox.exec(check, user="root")
    assert refused.return_code != 0 and "stray.txt" in refused.stderr
    _dx(box, f"chmod go-w {stray}")
    passed = await sandbox.exec(check, user="root")
    assert passed.return_code == 0, passed.stderr
