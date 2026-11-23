"""the read grant before the copy at test.sh's exit.

A test.sh that writes its reward through ``mkstemp`` and a rename, or under ``umask 077``, leaves root's 0600 file
on the mounted outputs; the host reads them as its own user. Root lets the host read them, and must never reach a
file outside the output folder doing it (REV2 R2-1), nor lose the reward to a cap or one long argv (R2-4):

- top-level names only, the reward names first (``reward.txt``, ``reward.json``, ``reward-details.json``,
  ``criteria-runner-manifest.json``, ``ctrf.json``), the rest after them, capped;
- through bfx's own ``chmod`` (it acts on the descriptor it opened, never follows a link, skips a hardlinked file),
  only after bfx's ``stat`` shows the folder root's (or the verifier user's) with no group or other write;
- the reward names on that structural check alone (R-810); the rest only while the freeze is proven held in the
  sandbox, or where no agent code can run.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.task.paths import RolloutPaths, SandboxPaths
from benchflow.task.verifier import Verifier

pytestmark = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads a mode-000 file, so an unreadable output cannot be staged",
)

FOLDER = "/logs/verifier"


class FakeBfx:
    """bfx as the grant sees it: ``stat`` of the folder, ``chmod`` of one name, ``scan`` of the freeze's state."""

    def __init__(
        self, host_dir, *, folder_mode=0o755, folder_uid=0, links=(), state=None
    ):
        self.host_dir = host_dir
        self.folder_mode = folder_mode
        self.folder_uid = folder_uid
        self.links = set(links)
        self.state = state
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        self.calls.append((cmd, args))
        if cmd == "stat":
            return {
                "entries": [
                    {
                        "path": p,
                        "type": "dir",
                        "uid": self.folder_uid,
                        "gid": 0,
                        "mode": self.folder_mode,
                        "exists": True,
                    }
                    for p in args.get("paths") or ()
                ]
            }
        if cmd == "chmod":
            name = args.get("path")
            if name in self.links:
                return {
                    "changed": 0,
                    "skipped_links": 1,
                    "skipped_mounts": [],
                    "skipped_hardlinks": [],
                }
            target = self.host_dir / name
            if target.exists():
                # The host here is the file's owner, not "other" as on a real mount: read for all.
                target.chmod(target.stat().st_mode | 0o444)
            return {
                "changed": 1,
                "skipped_links": 0,
                "skipped_mounts": [],
                "skipped_hardlinks": [],
            }
        if cmd == "scan":
            state = self.state
            if state is None:
                return {"entries": [], "contents": {}, "errors": []}
            path, text = state
            import base64

            return {
                "entries": [
                    {
                        "path": path,
                        "type": "file",
                        "uid": 0,
                        "mode": 0o644,
                        "exists": True,
                    }
                ],
                "contents": {
                    path: {
                        "b64": base64.b64encode(text.encode()).decode(),
                        "size": len(text),
                    }
                },
                "errors": [],
            }
        raise AssertionError(f"unexpected bfx command {cmd}")

    def chmods(self) -> list[str]:
        return [a.get("path") for c, a in self.calls if c == "chmod"]


def _verifier(tmp_path, names, *, bfx=None, mode=0o000):
    (tmp_path / "verifier").mkdir()
    for name in names:
        path = tmp_path / "verifier" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1")
        path.chmod(mode)
    task = MagicMock()
    task.config.verifier.user = "root"
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=0, stdout="", stderr="")
        )
    )
    if bfx is not None:
        sandbox.bfx_call = bfx
        sandbox.bfx_delivery = "bind"
    verifier = Verifier(
        task=task, rollout_paths=RolloutPaths(tmp_path), sandbox=sandbox
    )
    return verifier, sandbox


def _shell_chmods(sandbox) -> list[str]:
    said = []
    for call in sandbox.exec.await_args_list:
        command = call.kwargs.get("command") or (call.args[0] if call.args else "")
        if "chmod" in str(command):
            said.append(str(command))
    return said


def _readable(tmp_path, name) -> bool:
    try:
        with open(tmp_path / "verifier" / name, "rb"):
            return True
    except PermissionError:
        return False


@pytest.mark.asyncio
async def test_a_name_swapped_for_a_link_after_the_host_check_is_never_followed(
    tmp_path,
):
    """810-U1 (R2-1). The host sees ``reward.txt`` as a regular file it can't open; in the sandbox the name is a
    link by the time root acts. The grant goes through bfx's ``chmod`` (which skips a link) and never through a
    shell ``chmod``, which follows it to wherever it points.

    Mutant: the shell's ``chmod go+r`` put back."""
    bfx = FakeBfx(tmp_path / "verifier", links={"reward.txt"})
    verifier, sandbox = _verifier(tmp_path, ["reward.txt"], bfx=bfx)
    await verifier._let_host_read_outputs(SandboxPaths(), "main")
    assert not _shell_chmods(sandbox), _shell_chmods(sandbox)
    assert (
        "chmod",
        {"root": FOLDER, "path": "reward.txt", "set": "044"},
    ) in bfx.calls, bfx.calls


@pytest.mark.asyncio
async def test_only_top_level_names_are_granted(tmp_path):
    """810-U2 (R2-1). A subfolder of the outputs may be another user's to change (a task's own ``chmod 777``);
    nothing under one is granted, by any route.

    Mutant: the recursive listing (``rglob``) put back."""
    bfx = FakeBfx(tmp_path / "verifier")
    verifier, sandbox = _verifier(tmp_path, ["reward.txt", "sub/x"], bfx=bfx)
    await verifier._let_host_read_outputs(SandboxPaths(), "main")
    granted = bfx.chmods() + _shell_chmods(sandbox)
    assert granted, "precondition: the reward was granted"
    assert not any("sub" in str(g) for g in granted), granted


@pytest.mark.asyncio
async def test_reward_names_come_first_and_the_rest_is_capped_without_one_argv(
    tmp_path, monkeypatch
):
    """810-U3 (R2-4). 600 unreadable names sort before ``reward.txt``. The reward names are granted first, the
    rest after them up to the cap, with the count left out recorded; nothing is one long argv.

    Mutant: the names sorted by path and cut at the cap (the reward cut off)."""
    names = [f"aaa-{i:03d}" for i in range(600)] + ["reward.txt", "ctrf.json"]
    bfx = FakeBfx(tmp_path / "verifier")
    verifier, sandbox = _verifier(tmp_path, names, bfx=bfx)
    verifier.freeze_at_exit = True
    await verifier._let_host_read_outputs(SandboxPaths(), "main")
    chmods = bfx.chmods()
    assert chmods[:2] == ["reward.txt", "ctrf.json"], chmods[:5]
    assert _readable(tmp_path, "reward.txt"), (
        "the reward is still unreadable to the host"
    )
    grant = getattr(verifier, "read_grant", None) or {}
    assert len(chmods) <= 512 and grant.get("not_granted") == len(names) - len(
        chmods
    ), grant
    assert not _shell_chmods(sandbox), "a shell chmod (one argv) was used"


@pytest.mark.asyncio
@pytest.mark.parametrize("freeze", ["none", "not-held", "held"])
async def test_the_rest_waits_for_a_held_freeze(tmp_path, freeze):
    """810-U4. Agent code may run (off/audit). With no freeze, or one not proven held in the
    sandbox (its state file without ``frozen_at``), only the reward names are granted; ``other.txt`` waits. Held:
    both.

    Mutant: the stillness gate on the rest removed."""
    state = "/run/bf-freeze-test"
    text = {
        "none": None,
        "not-held": "unsettled\n",
        "held": "ours 12 3 1\nfrozen_at 1.0\n",
    }[freeze]
    bfx = FakeBfx(tmp_path / "verifier", state=None if text is None else (state, text))
    verifier, sandbox = _verifier(tmp_path, ["reward.json", "other.txt"], bfx=bfx)
    verifier.agent_processes_alive = True
    if text is not None:
        verifier._freeze_state = (state, "main")
    await verifier._let_host_read_outputs(SandboxPaths(), "main")
    assert not _shell_chmods(sandbox), _shell_chmods(sandbox)
    chmods = bfx.chmods()
    assert "reward.json" in chmods, chmods
    if freeze == "held":
        assert "other.txt" in chmods, chmods
    else:
        assert "other.txt" not in chmods, chmods


@pytest.mark.asyncio
@pytest.mark.parametrize("folder", [(0o777, 0), (0o775, 0), (0o755, 1000)])
async def test_an_open_output_folder_gets_no_grant(tmp_path, folder):
    """810-U5. bfx's ``stat`` shows the output folder writable by another user (a task's ``chmod 777``, or
    the folder given away): another user's process could swap a name, so nothing is granted, and the reason is
    recorded.

    Mutant: the folder check removed."""
    mode, uid = folder
    bfx = FakeBfx(tmp_path / "verifier", folder_mode=mode, folder_uid=uid)
    verifier, sandbox = _verifier(tmp_path, ["reward.txt"], bfx=bfx)
    await verifier._let_host_read_outputs(SandboxPaths(), "main")
    assert not bfx.chmods() and not _shell_chmods(sandbox), (
        bfx.calls,
        _shell_chmods(sandbox),
    )
    grant = getattr(verifier, "read_grant", None) or {}
    assert grant.get("why"), grant
