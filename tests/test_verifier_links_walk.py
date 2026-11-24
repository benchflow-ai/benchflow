"""The link walk's own rules (and).

- every link found after grading counts, root's too (REVIEW-FINAL F1: test.sh can lock the agent's link
  down to root).
- the walk's roots come from the kernel's mount table; a removal names the entry the walk saw
  (``expect {dev, ino}``); no image ``rm`` ever removes a link; a folder that vanishes mid-walk is no error.
- H2 Z5 (B1's half): the walker record names the guarantee bfx's launch gives.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _verifier_links

pytestmark = pytest.mark.bfx_refusal

R = "/logs/verifier/reward.txt"


# -- the walk's roots ---------------------------------------------------------------------------------------

_DEPTH = 10
_NESTED = [
    "/" + "/".join(f"m{j}" for j in range(1, i + 1)) for i in range(1, _DEPTH + 1)
]
_DEEP_LINK = _NESTED[-1] + "/x"


def _mount_table_bfx(extra_mounts=()):
    """Ten mounts, each inside the last; a link to the reward in the deepest; and /etc/hosts, a file bind mount."""
    calls = []
    mounts = [{"id": 1, "mount_point": "/"}]
    mounts += [{"id": 10 + i, "mount_point": p} for i, p in enumerate(_NESTED)]
    mounts += [
        {"id": 90, "mount_point": "/etc/hosts"},
        {"id": 91, "mount_point": "/proc"},
        {"id": 92, "mount_point": "/proc/sys"},
        {"id": 93, "mount_point": "/logs/verifier"},
    ]
    mounts += [dict(m) for m in extra_mounts]

    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        calls.append((cmd, args))
        assert cmd == "scan", cmd
        if args.get("walks"):
            entries, errors = [], []
            for walk in args["walks"]:
                root = walk.get("root")
                if root == "/":
                    entries.append({"path": "/m1", "type": "dir", "mount": True})
                    entries.append(
                        {"path": "/etc/hosts", "type": "file", "mount": True}
                    )
                elif root in _NESTED:
                    i = _NESTED.index(root)
                    if i + 1 < len(_NESTED):
                        entries.append(
                            {"path": _NESTED[i + 1], "type": "dir", "mount": True}
                        )
                    else:
                        entries.append(
                            {
                                "path": _DEEP_LINK,
                                "type": "link",
                                "link": R,
                                "dev": 3,
                                "ino": 4,
                            }
                        )
                elif root == "/etc/hosts":
                    errors.append(
                        {"path": "/etc/hosts", "op": "walk", "error": "not-directory"}
                    )
            return {"entries": entries, "errors": errors, "mounts": []}
        if args.get("mounts"):
            return {"entries": [], "mounts": mounts, "errors": []}
        return {"entries": [], "errors": [], "mounts": []}

    return bfx_call, calls


@pytest.mark.asyncio
async def test_walk_roots_come_from_the_mount_table():
    """(mount discovery). The walk took its roots from its own listing, eight rounds deep, and past the
    eighth it said nothing: a mount nine deep was never walked, and the walk read complete. The roots now come
    from mountinfo: every mount is walked (a file bind mount like /etc/hosts is no error; /proc, /sys and the output
    folder are skipped), so the link in the tenth finds its way into the record.

    Mutant: the eight-round listing put back."""
    bfx_call, _ = _mount_table_bfx()
    env = SimpleNamespace(bfx_call=bfx_call, bfx_delivery="bind", exec=AsyncMock())
    walked = await _verifier_links.walk(env, roots=None, timeout_sec=5)
    assert walked.complete is True, walked.why
    assert _DEEP_LINK in walked.links, sorted(walked.links)
    assert _NESTED[-1] in walked.roots and "/proc/sys" not in walked.roots, walked.roots


@pytest.mark.asyncio
async def test_a_mount_point_the_walk_cannot_name_leaves_it_incomplete():
    """. A mount point whose name the mount table's text decode replaced (not UTF-8) names no folder the walk
    can open: the walk is incomplete, never complete without it."""
    bfx_call, _ = _mount_table_bfx(
        extra_mounts=[{"id": 99, "mount_point": "/odd�name"}]
    )
    env = SimpleNamespace(bfx_call=bfx_call, bfx_delivery="bind", exec=AsyncMock())
    walked = await _verifier_links.walk(env, roots=None, timeout_sec=5)
    assert walked.complete is False, (walked.complete, walked.why)


# -- a removal names the entry the walk saw; never the image's rm -------------------------------------------


def _removal_env(remove):
    """A sandbox with a complete record from before the turn and, now, the agent's link /app/x to the reward."""
    calls = []

    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        calls.append((cmd, args))
        if cmd == "remove":
            return await remove(args)
        if cmd == "walk":
            # After the turn the walk is bfx's ``walk`` from each mount (paths relative to it).
            link = {"path": "app/x", "type": "link", "link": R, "dev": 7, "ino": 42}
            return {
                "entries": [link] if args.get("root") == "/" else [],
                "vanished": [],
                "deeper": [],
            }
        if args.get("mounts"):
            return {
                "entries": [],
                "mounts": [{"id": 1, "mount_point": "/"}],
                "errors": [],
            }
        return {"entries": [], "errors": [], "mounts": []}  # the confirming look: gone

    async def exec_(command, *args, **kwargs):
        if "readlink" in str(command) or "realpath" in str(command):
            return SimpleNamespace(return_code=0, stdout=f"/app/x\0{R}\0", stderr="")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    env = SimpleNamespace(
        bfx_call=bfx_call, bfx_delivery="bind", exec=AsyncMock(side_effect=exec_)
    )
    _verifier_links.inherit_record(env, _verifier_links.Walked("bfx"))
    return env, calls


@pytest.mark.asyncio
async def test_a_removal_names_the_entry_the_walk_saw():
    """(bfx ``remove`` without ``expect``). The removal carries the (dev, ino) the walk saw, so bfx refuses
    a name swapped in between; a refused removal is a link not removed (Q6 for a non-root agent).

    Mutant: ``expect`` dropped."""

    async def removed(args):
        return {"removed": 1}

    env, calls = _removal_env(removed)
    found = await _verifier_links.remove_turn_links(env, timeout_sec=5)
    removes = [a for c, a in calls if c == "remove"]
    assert removes and removes[0].get("expect") == {"dev": 7, "ino": 42}, removes
    assert [r.get("removed") for r in found["links"]] == [True], found

    from benchflow._bfx.client import BfxError

    async def changed(args):
        raise BfxError("changed", "app/x is no longer the entry that was scanned")

    env, calls = _removal_env(changed)
    found = await _verifier_links.remove_turn_links(env, timeout_sec=5)
    (row,) = found["links"]
    assert row["removed"] is False and "changed" in str(row.get("error")), row


@pytest.mark.asyncio
async def test_a_removal_never_uses_the_image_s_rm():
    """(F27). bfx becomes unavailable for the removal: the link is not removed (the check's outcome
    applies), and no ``rm`` of the image runs in its place.

    Mutant: the image-``rm`` fallback put back."""
    from benchflow._bfx.locate import BfxUnavailable

    async def gone(args):
        raise BfxUnavailable("the main container is not running")

    env, _ = _removal_env(gone)
    found = await _verifier_links.remove_turn_links(env, timeout_sec=5)
    commands = [
        str(c.args[0] if c.args else c.kwargs.get("command"))
        for c in env.exec.await_args_list
    ]
    assert not any("rm -f" in c for c in commands), commands
    (row,) = found["links"]
    assert row["removed"] is False, row


# -- a folder that vanishes mid-walk ------------------------------------------------------------------------


def _walk_errors_bfx(errors):
    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        if args.get("walks"):
            return {"entries": [], "errors": [dict(e) for e in errors], "mounts": []}
        if args.get("mounts"):
            return {
                "entries": [],
                "mounts": [{"id": 1, "mount_point": "/"}],
                "errors": [],
            }
        return {"entries": [], "errors": [], "mounts": []}

    return bfx_call


@pytest.mark.asyncio
async def test_a_folder_that_vanishes_mid_scan_is_not_an_error():
    """(R-816: Python now). bfx reports a folder removed between its listing and its reading (``reading a
    directory: ENOENT``, or ``missing`` on the reopen): an honest live process does that (a build tool, a watcher),
    and the design says a vanished name is no error. The walk is complete and lists it as vanished; any other error
    still makes it incomplete.

    Mutant: the vanished mapping removed (an honest churning process gives a non-root agent 0)."""
    vanished = [
        {"path": "/tmp/churn/d3", "op": "list", "error": "reading a directory: ENOENT"},
        {"path": "/tmp/churn/d4", "op": "list", "error": "missing"},
    ]
    env = SimpleNamespace(
        bfx_call=_walk_errors_bfx(vanished), bfx_delivery="bind", exec=AsyncMock()
    )
    await _verifier_links.record_before_turn(env, timeout_sec=5)
    walked = getattr(env, _verifier_links.INVENTORY)
    assert walked.complete is True, walked.why
    assert set(getattr(walked, "vanished", ())) == {"/tmp/churn/d3", "/tmp/churn/d4"}, (
        walked
    )
    other = [*vanished, {"path": "/srv/x", "op": "lstat", "error": "permission"}]
    env = SimpleNamespace(
        bfx_call=_walk_errors_bfx(other), bfx_delivery="bind", exec=AsyncMock()
    )
    await _verifier_links.record_before_turn(env, timeout_sec=5)
    walked = getattr(env, _verifier_links.INVENTORY)
    assert walked.complete is False, walked


# -- H2 Z5 (B1's half): the walker record names the guarantee ------------------------------------------------------


def test_the_walker_record_names_its_guarantee():
    """map-H2 §1.2a, Z5: the trial's ``provenance.walker`` says whom bfx's launch holds against: the sandbox's own
    ``bfx_guarantee`` where it has one (B2's launch-aware property), else the delivery's."""

    async def bfx_call(*_a, **_k):
        return {}

    argv = SimpleNamespace(
        bfx_call=bfx_call,
        bfx_delivery="bind",
        _bfx_arch="x86_64",
        bfx_guarantee="any-agent",
    )
    shell = SimpleNamespace(
        bfx_call=bfx_call,
        bfx_delivery="per-call",
        _bfx_arch="x86_64",
        bfx_guarantee="non-root-agent",
    )
    plain = SimpleNamespace(bfx_call=bfx_call, bfx_delivery="bind", _bfx_arch="x86_64")
    assert _verifier_links.walker_record(argv).get("guarantee") == "any-agent"
    assert _verifier_links.walker_record(shell).get("guarantee") == "non-root-agent"
    assert _verifier_links.walker_record(plain).get("guarantee") == "any-agent"
    off = SimpleNamespace(bfx_delivery=None)
    assert _verifier_links.walker_record(off).get("guarantee") is None


def test_dead_walkers_are_gone():
    """(dead shell/find walkers): the link module runs no program of the image's but ``kernel_resolve``
    (names it). The shell and ``find`` walkers, and every shell fallback, are gone.

    Mutant: a shell fallback put back."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(_verifier_links.__file__).read_text())
    runners = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", None))
                    == "exec_raw"
                ):
                    runners.add(fn.name)
    assert runners == {"kernel_resolve"}, runners
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not names & {"_SHELL_WALK", "_FIND_WALK", "_walk_shell", "_walk_find"}, (
        names
        & {
            "_SHELL_WALK",
            "_FIND_WALK",
            "_walk_shell",
            "_walk_find",
        }
    )


@pytest.mark.parametrize("bfx_call", [True, False], ids=["route", "no-route"])
def test_the_sandbox_s_own_reason_bfx_cannot_run_is_the_refusal_s(bfx_call):
    """B2 (11-18): a sandbox that knows why bfx can't run there (AgentCore's opt-in route, an arch, a failed
    upload) says so in ``bfx_unavailable_why``; the refusal before the turn names that reason, under BenchFlow's own
    head, never "BENCHFLOW_BFX_DELIVERY=off" for a route that was never asked for.

    Mutant: the sandbox's reason ignored."""

    async def call(*_a, **_k):
        return {}

    why = "AgentCore runs bfx only with BENCHFLOW_BFX_PROVIDERS=on (opt-in route)"
    env = SimpleNamespace(bfx_delivery=None, bfx_unavailable_why=why)
    if bfx_call:
        env.bfx_call = call
    said = _verifier_links.walker_unavailable(env)
    assert said is not None and said.startswith(_verifier_links.BFX_UNAVAILABLE), said
    assert why in said and "BENCHFLOW_BFX_DELIVERY=off" not in said, said
    shell = SimpleNamespace(
        bfx_call=call,
        bfx_delivery="volume",
        _bfx_arch="x86_64",
        bfx_guarantee="non-root-agent",
    )
    # Modal's volume is reached by argv and still holds only against non-root (B2's live run): the property decides.
    assert _verifier_links.walker_record(shell).get("guarantee") == "non-root-agent"


# -- REVIEW-FINAL F1: every link found after grading counts (the exemption reverted) ------------------------


@pytest.mark.asyncio
async def test_a_root_owned_link_found_after_grading_still_counts():
    """REVIEW-FINAL F1 (blocker). test.sh's own ``chown -R root:root /app`` (or a root ``cp -r``, ``tar x``) makes
    the agent's link root's, in folders the agent can no longer write. Ownership after grading says nothing about
    who made the link, so a link into the output folder found after grading counts against a non-root agent (Q1),
    whoever owns it now; rc6's rule. The wine link (jq-windows) is the named honest cost again.

    Mutant: the root-made exemption put back (the forged 1.0 scores `verified`)."""
    calls = []

    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        calls.append((cmd, args))
        link = {
            "path": "/app/out/grade.log",
            "type": "link",
            "link": R,
            "uid": 0,
            "dev": 1,
            "ino": 9,
        }
        if cmd == "walk":
            return {
                "entries": [{**link, "path": "app/out/grade.log"}]
                if args.get("root") == "/"
                else [],
                "vanished": [],
                "deeper": [],
            }
        if args.get("walks"):
            return {"entries": [link], "errors": [], "mounts": []}
        if args.get("paths"):
            # The facts after test.sh's ``chown -R root:root /app``: every folder root's 0755, the link root's.
            folders = [
                {
                    "path": p,
                    "type": "dir",
                    "uid": 0,
                    "mode": 0o755,
                    "exists": True,
                    "mount_id": 1,
                    "dev": 1,
                    "ino": i,
                }
                for i, p in enumerate(("/", "/app", "/app/out"))
            ]
            return {
                "entries": [*folders, {**link, "exists": True, "mount_id": 1}],
                "mounts": [{"id": 1, "mount_point": "/", "fstype": "overlay"}],
                "errors": [],
            }
        if args.get("mounts"):
            return {
                "entries": [],
                "mounts": [{"id": 1, "mount_point": "/", "fstype": "overlay"}],
                "errors": [],
            }
        return {"entries": [], "errors": [], "mounts": []}

    async def exec_(command, *args, **kwargs):
        return SimpleNamespace(
            return_code=0, stdout=f"/app/out/grade.log\0{R}\0", stderr=""
        )

    env = SimpleNamespace(
        bfx_call=bfx_call, bfx_delivery="bind", exec=AsyncMock(side_effect=exec_)
    )
    _verifier_links.inherit_record(env, _verifier_links.Walked("bfx"))
    found = await _verifier_links.links_after_grading(env, timeout_sec=5)
    assert [r["path"] for r in found["links"]] == ["/app/out/grade.log"], found
    assert not found.get("links_root_made"), found


# -- REVIEW-FINAL F2: after the turn, a name that vanishes mid-walk makes the walk incomplete ------------------------


def _hiding_bfx():
    """A live process renames a folder on the link's path while the walk reads it: bfx's ``scan`` skips the name it
    can no longer open without a word (``scan.rs:362``); its ``walk`` names it as vanished."""
    calls = []

    async def bfx_call(cmd, args=None, *, timeout_sec=None):
        args = dict(args or {})
        calls.append((cmd, args))
        if cmd == "walk":
            vanished = [{"path": "app/bfm/n3"}] if args.get("root") == "/" else []
            return {"entries": [], "vanished": vanished, "deeper": []}
        if args.get("walks"):
            return {
                "entries": [],
                "errors": [],
                "mounts": [],
            }  # the link's folder skipped, silently
        if args.get("mounts"):
            return {
                "entries": [],
                "mounts": [{"id": 1, "mount_point": "/"}],
                "errors": [],
            }
        return {"entries": [], "errors": [], "mounts": []}

    return bfx_call, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("check", ["remove_turn_links", "links_after_grading"])
async def test_after_the_turn_a_name_that_vanishes_mid_walk_leaves_it_incomplete(check):
    """REVIEW-FINAL F2 (high). After the turn the agent's processes are alive and are the adversary: one that keeps
    renaming a folder on a link's path makes a scan skip it and still read complete (the reviewer: the link missing
    from 67 of 200 scans, a forged 1.0 `verified` in 1 of 6 trials). After the turn the walk names every name that
    vanished while it read (bfx's ``walk``), and any one makes it incomplete: for a non-root agent, the Q6 0.

    Mutants: the after-turn walk by ``scan`` (silent skip); a vanished name tolerated after the turn."""
    bfx_call, _ = _hiding_bfx()
    env = SimpleNamespace(bfx_call=bfx_call, bfx_delivery="bind", exec=AsyncMock())
    _verifier_links.inherit_record(env, _verifier_links.Walked("bfx"))
    found = await getattr(_verifier_links, check)(env, timeout_sec=5)
    assert found["ran"] is False, found
    assert "vanished" in str(found["why"]), found


@pytest.mark.asyncio
async def test_before_the_turn_a_vanished_folder_is_still_no_error():
    """REVIEW-FINAL F2. Before the turn no agent code exists yet: a folder BenchFlow's own concurrent steps make and
    remove while the record is taken stays no error, so an honest trial isn't refused."""
    vanished = [
        {
            "path": "/tmp/bf-step/d3",
            "op": "list",
            "error": "reading a directory: ENOENT",
        }
    ]
    env = SimpleNamespace(
        bfx_call=_walk_errors_bfx(vanished), bfx_delivery="bind", exec=AsyncMock()
    )
    await _verifier_links.record_before_turn(env, timeout_sec=5)
    record = getattr(env, _verifier_links.INVENTORY)
    assert record.complete is True, record
