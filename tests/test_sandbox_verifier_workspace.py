"""Tests for _seed_verifier_workspace (pre-agent workspace snapshot)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_env():
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
    return env


def _cmds(env):
    """Return list of (command_string, call_object) pairs from env.exec calls."""
    return [(c.args[0], c) for c in env.exec.call_args_list]


@pytest.mark.asyncio
async def test_seed_verifier_workspace_locks_logs_parent():
    """/logs/ is chowned to root:root and chmod 755; the lock call runs as root."""
    from benchflow._sandbox import _seed_verifier_workspace

    env = _make_env()
    await _seed_verifier_workspace(env)

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
async def test_seed_verifier_workspace_creates_testbed_verify():
    """/testbed_verify is wiped, seeded from workspace, chowned root, made world-readable; runs as root."""
    from benchflow._sandbox import _seed_verifier_workspace

    env = _make_env()
    await _seed_verifier_workspace(env, workspace="/testbed")

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
async def test_seed_verifier_workspace_seeds_from_workspace_param():
    """workspace param controls which directory is copied to /testbed_verify."""
    from benchflow._sandbox import _seed_verifier_workspace

    env = _make_env()
    await _seed_verifier_workspace(env, workspace="/app")

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
async def test_harden_restore_fallback_uses_shutil():
    """Fallback must use shutil not rm -rf — rm -rf crashes with EOVERFLOW on old LFS images."""
    from benchflow._sandbox import harden_before_verify

    env = _make_env()
    task = MagicMock()
    task.config.verifier.env = {}
    with (
        patch("benchflow._sandbox._restore_build_config", AsyncMock()),
        patch("benchflow._sandbox._refresh_verifier_workspace", AsyncMock()),
    ):
        await harden_before_verify(
            env, task, sandbox_user=None, workspace="/testbed", restore_workspace=True
        )

    restore = next(
        (c.args[0] for c in env.exec.call_args_list if "rsync" in c.args[0]), None
    )
    if restore is None:
        pytest.skip("restore_workspace path not exercised in current _sandbox.py")
    fallback = restore.split("||", 1)[1]
    assert (
        "shutil" in fallback
        and "rm -rf" not in fallback
        and "rmtree" not in fallback
        and "dirs_exist_ok=True" in fallback
    )


def test_oracle_branch_setup_calls():
    """Regression guard: oracle mode must wire up all pre-verify setup calls.

    Checks the code structure because the full run() mock surface is too
    expensive for a unit test. Four specific bugs are guarded:

    1. _seed_verifier_workspace missing → /testbed_verify never seeded →
       full workspace restore in harden_before_verify has nothing to rsync from
    2. _snapshot_build_config missing → workspace not snapshotted; oracle can tamper setup.py
    3. agent_cwd not set → _verify(workspace=None) → restore skipped entirely
    4. agent_cwd hardcoded to "/app" → breaks tasks whose WORKDIR is /testbed or other;
       must be detected via `pwd` like the non-oracle path
    """
    import inspect

    from benchflow import trial as trial_mod

    source = inspect.getsource(trial_mod.Trial.install_agent)
    oracle_pos = source.find('agent == "oracle"')
    assert oracle_pos != -1, "oracle branch not found in Trial.install_agent"
    oracle_block = source[oracle_pos:]

    assert "_seed_verifier_workspace" in oracle_block, (
        "_seed_verifier_workspace not in oracle branch — "
        "/testbed_verify never seeded, full workspace restore has nothing to rsync from"
    )
    assert "_snapshot_build_config" in oracle_block, (
        "_snapshot_build_config not in oracle branch — build-config tampering not mitigated"
    )
    assert "agent_cwd" in oracle_block, (
        "agent_cwd not assigned in oracle branch — _verify(workspace=None) skips restore"
    )
    # pwd detection happens before the oracle branch in install_agent
    assert '"pwd"' in source or "'pwd'" in source, (
        "agent_cwd must be detected via pwd (not hardcoded) — "
        "different Harbor tasks use different WORKDIR values (/testbed, /app, etc.)"
    )
