"""a file mount root under gVisor (Modal) is a file mount root.

The kernel's mount table names ``/etc/resolv.conf`` as a mount on Docker and on Modal. On Linux bfx's open of a root
refuses a file (``O_DIRECTORY``), so its scan reports ``{"op": "walk", "error": "not-directory"}``. gVisor lets the
open through and refuses the directory read instead: ``{"op": "list", "error": "reading a directory: ENOTDIR"}``.
Either is a mount root with nothing beneath it, skipped before and after the turn. The same error on a path that is
not a mount root, or any other error on a root, still leaves the record incomplete. Without this, every non-root
Modal trial was refused before its turn (``integrity_unavailable``) in audit and off (dogfood C3 part 5).

And the Modal preflight finds the token where Modal's own client does: ``MODAL_CONFIG_PATH``, else ``~/.modal.toml``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchflow.sandbox import _verifier_links

# The walks' own bfx answers are given here: no stand-in (tests/_fake_bfx.py).
pytestmark = [pytest.mark.bfx_refusal, pytest.mark.timeout(30)]

RESOLV = "/etc/resolv.conf"
GVISOR = {"op": "list", "error": "reading a directory: ENOTDIR"}
LINUX = {"op": "walk", "error": "not-directory"}
MOUNTS = ["/", "/dev", "/dev/pts", "/dev/shm", RESOLV, "/__modal/mounts/x"]


def _env(errors, *, link=None):
    """A sandbox whose kernel mount table is Modal's and whose bfx ``scan`` reports *errors* (each with its
    path)."""

    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        assert cmd == "scan", cmd
        if args.get("mounts"):
            return {
                "entries": [],
                "mounts": [{"id": i, "mount_point": m} for i, m in enumerate(MOUNTS)],
            }
        roots = [w["root"] for w in args.get("walks") or ()]
        assert RESOLV in roots, roots
        entries = [{"path": r, "type": "file" if r == RESOLV else "dir"} for r in roots]
        if link:
            entries.append(link)
        return {"entries": entries, "errors": [dict(e) for e in errors], "mounts": []}

    return SimpleNamespace(bfx_call=bfx_call, bfx_delivery="volume")


async def _record(env):
    return await _verifier_links.record_before_turn(env, timeout_sec=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["gvisor", "linux"])
async def test_a_file_mount_root_is_skipped_before_the_turn_in_either_form(shape):
    """(C3 part 5) The scan before the turn on Modal reports /etc/resolv.conf's mount root as gVisor
    does: the record is still whole, with its links, so the trial reaches its turn. Linux's form is the control.

    Mutant: only Linux's form skipped (the dogfood's refusal)."""
    form = GVISOR if shape == "gvisor" else LINUX
    link = {
        "path": "/etc/alt",
        "type": "link",
        "link": "/usr/bin/x",
        "dev": 1,
        "ino": 2,
    }
    walked = await _record(_env([{**form, "path": RESOLV}], link=link))
    assert walked.complete is True and walked.why is None, walked
    assert RESOLV in walked.roots and walked.links == {"/etc/alt": "/usr/bin/x"}, walked


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("shape", "path"),
    [
        ("gvisor", "/app/data"),
        ("gvisor", "/etc/resolv.conf/x"),
        ("gvisor", "/"),
        ("linux", "/app/data"),
        ("linux", "/etc/resolv.conf/x"),
    ],
)
async def test_the_same_error_beneath_a_root_or_on_slash_still_fails_closed(
    shape, path
):
    """Only a mount root the kernel's table names (other than /) may be a file: the same error anywhere
    else is a folder the walk could not read, and the record is incomplete. (Linux's form on / is the earlier
    rule, unchanged.)

    Mutant: the error skipped wherever it is."""
    form = GVISOR if shape == "gvisor" else LINUX
    walked = await _record(_env([{**form, "path": path}]))
    assert walked.complete is False and "could not read 1 path" in str(walked.why), (
        walked
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        {"op": "list", "error": "reading a directory: EACCES"},
        {"op": "list", "error": "reading a directory: EIO"},
        {"op": "lstat", "error": "not-directory"},
        {"op": "path", "error": "not-directory"},
    ],
)
async def test_any_other_error_on_a_mount_root_still_fails_closed(error):
    """A mount root whose listing failed for any other reason (or at another step) is not taken as a
    file.

    Mutant: any error on a mount root skipped."""
    walked = await _record(_env([{**error, "path": RESOLV}]))
    assert walked.complete is False, walked


@pytest.mark.asyncio
async def test_after_the_turn_bfx_walk_s_not_dir_on_a_file_mount_root_is_skipped():
    """After the turn each root is walked by bfx ``walk``, which spells both forms as kind ``not-dir``
    (gVisor's message: "reading a directory: ENOTDIR"): the root is skipped and the walk is whole. Beneath a
    root the same kind still fails the walk."""
    from benchflow._bfx.client import BfxError

    calls = []

    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        if args.get("mounts"):
            return {
                "entries": [],
                "mounts": [{"id": i, "mount_point": m} for i, m in enumerate(MOUNTS)],
            }
        assert cmd == "walk", cmd
        calls.append(args.get("root"))
        if args.get("root") == RESOLV:
            raise BfxError("not-dir", "reading a directory: ENOTDIR")
        return {"entries": [], "vanished": [], "deeper": []}

    env = SimpleNamespace(bfx_call=bfx_call, bfx_delivery="volume")
    _verifier_links.inherit_record(env, _verifier_links.Walked("bfx"))
    found = await _verifier_links.links_after_grading(env, timeout_sec=5)
    assert found["ran"] is True, found
    assert RESOLV in calls, calls


# -- the Modal preflight -------------------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["config-path", "home"])
def test_the_modal_preflight_finds_the_token_where_modal_does(
    tmp_path, monkeypatch, where
):
    """(C3 part 5, try 1) Modal's client reads ``MODAL_CONFIG_PATH``, else ``~/.modal.toml``; the
    preflight said "Modal requires authentication" for a token Modal itself would use.

    Mutant: MODAL_CONFIG_PATH ignored."""
    from benchflow.sandbox.modal_impl import ModalSandbox

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    if where == "config-path":
        config = tmp_path / "secrets" / "modal.toml"
        config.parent.mkdir()
        config.write_text("[benchflow]\ntoken_id = 'x'\ntoken_secret = 'y'\n")
        monkeypatch.setenv("MODAL_CONFIG_PATH", str(config))
    else:
        (home / ".modal.toml").write_text("[benchflow]\ntoken_id = 'x'\n")
        monkeypatch.delenv("MODAL_CONFIG_PATH", raising=False)
    ModalSandbox.preflight()


def test_the_modal_preflight_still_refuses_with_no_token_anywhere(
    tmp_path, monkeypatch
):
    """A MODAL_CONFIG_PATH that names no file, no ~/.modal.toml and no token variables: refused, with
    the remedy (control)."""
    from benchflow.sandbox.modal_impl import ModalSandbox

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(tmp_path / "missing.toml"))
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    with pytest.raises(SystemExit, match="Modal requires authentication"):
        ModalSandbox.preflight()
