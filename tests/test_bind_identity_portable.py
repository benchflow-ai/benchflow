"""A kept Docker checkpoint restores where BenchFlow's bfx mount is another inode.

Guards 0.10.0rc6's fix: a checkpoint's label records the identity (device and
inode) of every bind its first start saw, and a restore proved the
replacement against it, except BenchFlow's log folders and the private task
copy. BenchFlow's read-only bfx mount at ``/.bfx`` binds a folder in the
machine's (and user's) cache, so on a second host, or with another cache
folder on the same host, it is another inode, and every restore refused with
"Container bind identity changed since the first start". It is now expected
at the restoring sandbox's own identity: what proves it is the bfx mount
check every start and restore takes (one read-only mount of this machine's
verified folder), not the inode. Any other bind still keeps the
checkpoint's identity and refuses when swapped
(``test_docker_restore_child_binds.test_a_childs_non_log_bind_keeps_the_parents_identity``).
"""

from __future__ import annotations

import pytest

from benchflow.sandbox import _bind_targets
from tests.integration.deterministic import harness as h
from tests.test_docker_restore_child_binds import _no_leftovers, _sandbox

SANDBOX, SKIP_REASON = h.select_sandbox()


def test_the_bfx_mount_takes_the_sandboxs_own_identity():
    """P1: /.bfx and the log folders take the restoring sandbox's own
    identity; an operator bind keeps the checkpoint's."""
    carried = {"main": {"/.bfx": "1:9", "/opt/op": "1:8", "/logs/agent": "1:1"}}
    own = {"main": {"/.bfx": "2:9", "/opt/op": "2:8", "/logs/agent": "2:1"}}
    assert _bind_targets.with_own_log_binds(carried, own) == {
        "main": {"/.bfx": "2:9", "/opt/op": "1:8", "/logs/agent": "2:1"}
    }
    assert {"/.bfx"} == _bind_targets.BFX_TARGETS


def test_only_main_s_bfx_mount_takes_the_sandboxs_own_identity():
    """Review of #384: the exemption is proved only for main
    (``_check_bfx_mounts``), so a sidecar's /.bfx keeps the carried identity."""
    carried = {"main": {"/.bfx": "1:9"}, "side": {"/.bfx": "1:7"}}
    own = {"main": {"/.bfx": "2:9"}, "side": {"/.bfx": "2:7"}}
    assert _bind_targets.with_own_log_binds(carried, own) == {
        "main": {"/.bfx": "2:9"},
        "side": {"/.bfx": "1:7"},
    }


def test_a_sandbox_that_started_without_bfx_keeps_the_carried_identity():
    """P1: nothing of its own at /.bfx, so the checkpoint's stands and a
    replacement with another bfx folder refuses."""
    carried = {"main": {"/.bfx": "1:9"}}
    assert _bind_targets.with_own_log_binds(carried, {"main": {}}) == carried


def test_the_refusal_names_the_targets_that_changed():
    with pytest.raises(RuntimeError, match=r"\(main /opt/op\)"):
        _bind_targets.verify(
            {"main": {"/opt/op": "2:8", "/.bfx": "2:9"}},
            {"main": {"/opt/op": "1:8", "/.bfx": "2:9"}},
        )


@pytest.mark.deterministic
async def test_a_checkpoint_restores_with_another_bfx_cache_folder(
    tmp_path, monkeypatch
):
    """P2: the checkpoint is taken where bfx's cache is one folder and
    restored into a sandbox whose cache is another, as on a second host."""
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else f"Docker-only ({SANDBOX})")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache-a"))
    parent = _sandbox(tmp_path, "parent")
    child = _sandbox(tmp_path, "child")
    image = None
    try:
        await parent.start(force_build=False)
        if getattr(parent, "bfx_delivery", None) != "bind":
            pytest.skip("bfx is not delivered by bind here (no pinned bfx built)")
        await parent.exec("echo checkpoint > /app/state", user="root")
        image = await parent.snapshot()
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache-b"))
        await child.start(force_build=False)
        await child.restore(image)
        state = await child.exec("cat /app/state", user="root")
        assert (state.return_code, state.stdout.strip()) == (0, "checkpoint")
        # The replacement mounts this sandbox's own verified folder.
        (folder,) = (tmp_path / "cache-b" / "benchflow" / "bfx").glob("*-x86_64")
        seen = await child.exec("stat -L -c %d:%i /.bfx", user="root")
        host = folder.stat()
        assert seen.stdout.strip() == f"{host.st_dev}:{host.st_ino}", seen
    finally:
        await child.stop(delete=True)
        await parent.stop(delete=True)
        if image is not None:
            await parent.delete_snapshot(image)
    await _no_leftovers(parent, child)
