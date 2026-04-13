"""Tests for _setup_verifier_user (Tier 3 sandbox hardening)."""

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_env():
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
    return env


def _cmds(env):
    """Return list of (command_string, call_object) pairs from env.exec calls."""
    return [(c.args[0], c) for c in env.exec.call_args_list]


@pytest.mark.asyncio
async def test_setup_verifier_user_locks_logs_parent():
    """/logs/ is chowned to root:root and chmod 755; the lock call runs as root."""
    from benchflow._sandbox import _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env)

    pairs = _cmds(env)
    match = next(
        (
            call
            for cmd, call in pairs
            if "chown root:root /logs" in cmd and "chmod 755 /logs" in cmd
        ),
        None,
    )
    assert match is not None, (
        "expected a call with 'chown root:root /logs' and 'chmod 755 /logs'"
    )
    # Must not accidentally match /logs/verifier — verify the bare /logs lock is present.
    cmd = match.args[0]
    assert (
        "chown root:root /logs " in cmd
        or "chown root:root /logs &&" in cmd
        or cmd.startswith("chown root:root /logs")
    ), f"lock cmd appears to target a subdirectory: {cmd!r}"
    assert match.kwargs.get("user") == "root"


@pytest.mark.asyncio
async def test_setup_verifier_user_creates_user_no_extra_groups():
    """useradd includes --groups '' to strip supplementary groups; runs as root."""
    from benchflow._sandbox import _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env)

    pairs = _cmds(env)
    match = next(
        (call for cmd, call in pairs if "useradd" in cmd and "--groups ''" in cmd), None
    )
    assert match is not None, "expected useradd call with --groups ''"
    assert match.kwargs.get("user") == "root"


@pytest.mark.asyncio
async def test_setup_verifier_user_wipes_home_dir():
    """Pre-staged /home/verifier is wiped, recreated, and locked 700; runs as root."""
    from benchflow._sandbox import _VERIFIER_USER, _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env)

    pairs = _cmds(env)
    wipe_match = next(
        (call for cmd, call in pairs if f"rm -rf /home/{_VERIFIER_USER}" in cmd),
        None,
    )
    assert wipe_match is not None, f"expected rm -rf /home/{_VERIFIER_USER}"
    assert f"mkdir -p /home/{_VERIFIER_USER}" in wipe_match.args[0]
    assert f"chmod 700 /home/{_VERIFIER_USER}" in wipe_match.args[0]
    assert wipe_match.kwargs.get("user") == "root"


@pytest.mark.asyncio
async def test_setup_verifier_user_chowns_logs_verifier():
    """/logs/verifier/ is owned by the verifier user with chmod 700; runs as root."""
    from benchflow._sandbox import _VERIFIER_USER, _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env)

    pairs = _cmds(env)
    match = next(
        (
            call
            for cmd, call in pairs
            if f"chown {_VERIFIER_USER}:{_VERIFIER_USER} /logs/verifier" in cmd
            and "chmod 700 /logs/verifier" in cmd
        ),
        None,
    )
    assert match is not None
    assert match.kwargs.get("user") == "root"


@pytest.mark.asyncio
async def test_setup_verifier_user_creates_testbed_verify():
    """/testbed_verify is wiped, seeded from workspace, chowned root, made world-readable; runs as root."""
    from benchflow._sandbox import _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env, workspace="/testbed")

    pairs = _cmds(env)

    # All three operations in one atomic command.
    match = next(
        (
            call
            for cmd, call in pairs
            if "cp -a /testbed /testbed_verify" in cmd
            and "chown -R root:root /testbed_verify" in cmd
            and "chmod -R o+rX /testbed_verify" in cmd
        ),
        None,
    )
    assert match is not None, (
        "expected a single call containing cp, chown, and chmod for /testbed_verify"
    )
    assert match.kwargs.get("user") == "root"

    # rm -rf /testbed_verify must precede cp -a (clean-slate guarantee).
    rm_idx = next(
        (i for i, c in enumerate(pairs) if "rm -rf /testbed_verify" in c[0]), None
    )
    cp_idx = next(
        (i for i, c in enumerate(pairs) if "cp -a /testbed /testbed_verify" in c[0]),
        None,
    )
    assert rm_idx is not None, "rm -rf /testbed_verify not found"
    assert cp_idx is not None, "cp -a /testbed /testbed_verify not found"
    assert rm_idx <= cp_idx, "rm -rf /testbed_verify must precede cp -a"


@pytest.mark.asyncio
async def test_setup_verifier_user_seeds_from_workspace_param():
    """workspace param controls which directory is copied to /testbed_verify."""
    from benchflow._sandbox import _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env, workspace="/app")

    cmds_list = [c.args[0] for c in env.exec.call_args_list]
    seed_cmd = next(
        (c for c in cmds_list if "cp -a" in c and "/testbed_verify" in c), None
    )
    assert seed_cmd is not None, "seed command not found"
    assert "/app" in seed_cmd, "workspace=/app must be seeded into /testbed_verify"
    assert "/testbed " not in seed_cmd, (
        "default /testbed must not appear when workspace=/app"
    )


@pytest.mark.asyncio
async def test_setup_verifier_user_creates_group_before_user():
    """groupadd (or getent guard) must run before useradd --gid to avoid exit-6 failure."""
    from benchflow._sandbox import _VERIFIER_USER, _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env)

    cmds_list = [c.args[0] for c in env.exec.call_args_list]
    groupadd_idx = next(
        (i for i, c in enumerate(cmds_list) if "groupadd" in c and _VERIFIER_USER in c),
        None,
    )
    # Match the specific verifier useradd call (--gid + user name) to avoid
    # false matches against any other useradd calls in a different function.
    useradd_idx = next(
        (
            i
            for i, c in enumerate(cmds_list)
            if "useradd" in c and "--gid" in c and _VERIFIER_USER in c
        ),
        None,
    )
    assert groupadd_idx is not None, (
        f"expected a groupadd call for {_VERIFIER_USER!r} — "
        "useradd --gid requires the group to pre-exist"
    )
    assert useradd_idx is not None, (
        f"expected useradd --gid {_VERIFIER_USER} call not found"
    )
    assert groupadd_idx < useradd_idx, (
        f"groupadd (idx={groupadd_idx}) must precede useradd (idx={useradd_idx})"
    )


def test_oracle_branch_setup_calls():
    """Regression guard: oracle mode must wire up all pre-verify setup calls.

    Checks the code structure because the full run() mock surface is too
    expensive for a unit test. Four specific bugs are guarded:

    1. _setup_verifier_user missing → verifier OS user never created → verifier_error
    2. _snapshot_build_config missing → workspace not frozen; oracle can tamper setup.py
    3. agent_cwd not set → _verify(workspace=None) → freeze + restore skipped entirely
    4. agent_cwd hardcoded to "/app" → breaks tasks whose WORKDIR is /testbed or other;
       must be detected via `pwd` like the non-oracle path
    """
    import inspect

    from benchflow import sdk as sdk_mod

    source = inspect.getsource(sdk_mod.SDK.run)
    oracle_pos = source.find('agent == "oracle"')
    else_pos = source.find("\n            else:", oracle_pos)
    assert oracle_pos != -1, "oracle branch not found in SDK.run"
    oracle_block = source[oracle_pos:else_pos]

    assert "_setup_verifier_user" in oracle_block, (
        "_setup_verifier_user not in oracle branch — verifier OS user never created"
    )
    assert "_snapshot_build_config" in oracle_block, (
        "_snapshot_build_config not in oracle branch — build-config tampering not mitigated"
    )
    assert "agent_cwd" in oracle_block, (
        "agent_cwd not assigned in oracle branch — _verify(workspace=None) skips freeze+restore"
    )
    assert '"pwd"' in oracle_block or "'pwd'" in oracle_block, (
        "agent_cwd must be detected via pwd (not hardcoded) — "
        "different Harbor tasks use different WORKDIR values (/testbed, /app, etc.)"
    )


@pytest.mark.asyncio
async def test_setup_verifier_user_useradd_precedes_home_wipe():
    """useradd must run before the home-dir wipe to prevent agent pre-staging /home/verifier."""
    from benchflow._sandbox import _VERIFIER_USER, _setup_verifier_user

    env = _make_env()
    await _setup_verifier_user(env)

    cmds_list = [c.args[0] for c in env.exec.call_args_list]
    useradd_idx = next(
        (
            i
            for i, c in enumerate(cmds_list)
            if "useradd" in c and "--gid" in c and _VERIFIER_USER in c
        ),
        None,
    )
    wipe_idx = next(
        (i for i, c in enumerate(cmds_list) if f"rm -rf /home/{_VERIFIER_USER}" in c),
        None,
    )
    assert useradd_idx is not None, "useradd call not found"
    assert wipe_idx is not None, "home-dir wipe call not found"
    assert useradd_idx < wipe_idx, (
        f"useradd (idx={useradd_idx}) must precede home-dir wipe (idx={wipe_idx}); "
        "otherwise an agent can pre-stage /home/verifier before the user exists"
    )
