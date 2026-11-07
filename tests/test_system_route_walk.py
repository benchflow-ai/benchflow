"""PERF-SBX: the system route's fast sh walk clears only what the full walk clears.

The verifier's always-kept system route is proven after the agent's turn by
``_trusted_path_extras_cmd``'s system check (``bf_judge`` per folder with
``bf_identity_only=1``). ``bf_fast_system`` walks the same routes without a
subshell per component and with one ``stat`` per folder. Every fixture runs
the full walk alone (the specification), the fast walk alone, and the
command as shipped, on the same real files: the fast walk must never clear a
route the full walk refuses, and the shipped command must answer exactly as
the full walk does, message included.
"""

import ast
import dataclasses
import os
import shutil
import subprocess
import sys

import pytest

from benchflow._runtime_tree import RuntimeKit
from benchflow.sandbox import _system_route_walk, lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._mount_aliases import PathExclusions
from tests.test_verifier_target_aliases import mount

_FAST_DECISION = "if bf_fast_system; then bf_fast=1; else bf_fast=; fi"
_FAST_ONLY = (
    "if bf_fast_system; then echo FAST-CLEAR >&2; else echo FAST-NO >&2; fi; exit 0"
)


@pytest.fixture
def inspector(tmp_path):
    folder = tmp_path / "inspector"
    folder.mkdir(mode=0o700)
    for name in ("sh", "stat", "readlink", "ls", "id", "find", "cat"):
        source = shutil.which(name)
        if source:
            shutil.copyfile(source, folder / name)
            (folder / name).chmod(0o700)
    return str(folder)


def executable(path):
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)


def exclusions(main, target, blocked, inspector, inodes=(), index=""):
    captured = PathExclusions.capture(main, blocked).for_target(target, ())
    return dataclasses.replace(
        captured, inodes=frozenset(inodes), inspector=inspector, inode_index=index
    )


def command(monkeypatch, folders, blocked):
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", tuple(folders))
    text = lockdown._trusted_path_extras_cmd("", blocked)
    assert text.count(_FAST_DECISION) == 1
    return text


def run(text):
    return subprocess.run(
        ["/bin/sh", "-c", text], capture_output=True, text=True, timeout=60
    )


def fast_cleared(text):
    fast = run(text.replace(_FAST_DECISION, _FAST_ONLY))
    assert fast.returncode == 0, fast.stderr
    assert fast.stderr.strip() in ("FAST-CLEAR", "FAST-NO"), fast.stderr
    return fast.stderr.strip() == "FAST-CLEAR"


def py_walk(folders, blocked):
    """(rc, stdout) of the kit-Python walk on this interpreter, kit flags."""
    spec = _system_route_walk.walk_spec(
        blocked.predicate_groups(),
        tuple(folders),
        index=blocked.inode_index if blocked.inodes else "",
        inspector=blocked.inspector,
    )
    assert spec is not None
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _system_route_walk.SYSTEM_ROUTE_WALK,
            spec,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd="/",
    )
    return result.returncode, result.stdout


def walks(monkeypatch, folders, blocked):
    """(full walk rc, fast walk cleared) after checking the shipped command
    and the kit-Python walk on the same files."""
    text = command(monkeypatch, folders, blocked)
    full = run(text.replace(_FAST_DECISION, "bf_fast="))
    cleared = fast_cleared(text)
    py_rc, py_out = py_walk(folders, blocked)
    py_cleared = py_rc == 0 and py_out == _system_route_walk.CLEAR + "\n"
    if full.returncode != 0:
        assert not py_cleared, f"python cleared a refused route: {full.stderr}"
    else:
        assert py_cleared, (py_rc, py_out)
    shipped = run(text)
    assert (shipped.returncode, shipped.stdout, shipped.stderr) == (
        full.returncode,
        full.stdout,
        full.stderr,
    )
    if full.returncode != 0:
        # The property that keeps the check: never clear what it refuses.
        assert not cleared, f"fast walk cleared a refused route: {full.stderr}"
        assert "excluded system PATH" in full.stderr
    return full.returncode, cleared


@pytest.mark.parametrize(
    "route",
    [
        "file-mount",
        "absolute",
        "relative",
        "chain-in",
        "directory-hop",
        "folder-link",
        "dot-dot",
    ],
)
def test_agent_reachable_tool_is_refused(tmp_path, monkeypatch, inspector, route):
    tools = tmp_path / "tools"
    excluded = tmp_path / "excluded"
    safe = tmp_path / "safe"
    for path in (tools, excluded, safe):
        path.mkdir()
    executable(safe / "honest")
    executable(excluded / "planted")
    folders = [str(tools)]
    if route == "file-mount":
        executable(tools / "cmp")
        alias, source = tools / "cmp", "/volume/work/bin/cmp"
    else:
        alias, source = excluded, "/volume/work"
        if route == "absolute":
            (tools / "cmp").symlink_to(excluded / "planted")
        elif route == "relative":
            (tools / "cmp").symlink_to("../excluded/planted")
        elif route == "chain-in":
            (tools / "cmp").symlink_to("../safe/hop")
            (safe / "hop").symlink_to("../excluded/planted")
        elif route == "directory-hop":
            (tools / "cmp").symlink_to("../excluded/hop/honest")
            (excluded / "hop").symlink_to(safe, target_is_directory=True)
        elif route == "dot-dot":
            # The lexical '..' after a link: safe/../excluded is excluded.
            (tools / "cmp").symlink_to(str(safe) + "/../excluded/planted")
        else:
            # The system folder itself is a link into the excluded tree.
            folder = tmp_path / "folder"
            folder.symlink_to(excluded, target_is_directory=True)
            folders = [str(folder)]
    main = mount("/", "/", "0:1", 1) + mount("/volume/work", "/app")
    target = mount("/", "/", "0:2", 1) + mount(source, str(alias))
    blocked = exclusions(main, target, ("/app",), inspector)
    rc, cleared = walks(monkeypatch, folders, blocked)
    assert rc != 0 and not cleared


@pytest.mark.parametrize(
    "link", ["honest-chain", "directory", "dangling", "loop", "folder-link", "none"]
)
def test_honest_system_route_is_cleared(tmp_path, monkeypatch, inspector, link):
    tools = tmp_path / "tools"
    safe = tmp_path / "safe"
    excluded = tmp_path / "excluded"
    for path in (tools, safe, excluded):
        path.mkdir()
    executable(tools / "cmp")
    executable(tools / ".hidden")
    executable(tools / "..dots")
    executable(safe / "honest")
    (tools / "data").write_text("not a tool")
    folders = [str(tools)]
    if link == "honest-chain":
        (tools / "hop").symlink_to("../safe/hop")
        (safe / "hop").symlink_to("honest")
    elif link == "directory":
        (tools / "dir").symlink_to(excluded, target_is_directory=True)
    elif link == "dangling":
        (tools / "gone").symlink_to(excluded / "absent")
    elif link == "loop":
        (tools / "loop").symlink_to(tools / "loop")
    elif link == "folder-link":
        # A merged-/usr image: /bin is a link to usr/bin.
        folder = tmp_path / "bin"
        folder.symlink_to("tools", target_is_directory=True)
        folders = [str(tools), str(folder)]
    main = mount("/", "/", "0:1", 1) + mount("/volume/work", "/app")
    target = mount("/", "/", "0:2", 1) + mount("/volume/work", str(excluded))
    blocked = exclusions(main, target, ("/app",), inspector)
    rc, cleared = walks(monkeypatch, folders, blocked)
    assert rc == 0 and cleared


@pytest.mark.parametrize("hops", [40, 41])
def test_link_chains_up_to_the_limit(tmp_path, monkeypatch, inspector, hops):
    """Forty hops pass both walks; past that the kernel's [ -f ] drops it."""
    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "end")
    for i in range(hops):
        (tools / f"l{i}").symlink_to(f"l{i + 1}" if i < hops - 1 else "end")
    blocked = exclusions(
        mount("/", "/", "0:1", 1), mount("/", "/", "0:2", 1), (), inspector
    )
    rc, cleared = walks(monkeypatch, [str(tools)], blocked)
    assert rc == 0 and cleared


def _indexed(tmp_path, st, name="inodes-x"):
    index = tmp_path / "inspector" / name
    index.mkdir()
    (index / f"{st.st_dev}:{st.st_ino}").touch()
    return str(index)


@pytest.mark.parametrize("kind", ["hardlink", "link-itself", "hop-target"])
def test_captured_inode_is_refused(tmp_path, monkeypatch, inspector, kind):
    """A tool that is, links through, or hard-links a captured agent inode."""
    tools = tmp_path / "tools"
    agent = tmp_path / "agent"
    tools.mkdir()
    agent.mkdir()
    executable(agent / "planted")
    executable(tools / "honest")
    if kind == "hardlink":
        os.link(agent / "planted", tools / "cmp")
        st = os.lstat(agent / "planted")
    elif kind == "link-itself":
        (tools / "cmp").symlink_to("real")
        executable(tools / "real")
        st = os.lstat(tools / "cmp")
    else:
        # Only a hop after the leaf is captured: the per-path stat decides.
        (tools / "cmp").symlink_to(agent / "planted")
        st = os.lstat(agent / "planted")
    blocked = exclusions(
        mount("/", "/", "0:1", 1),
        mount("/", "/", "0:2", 1),
        (),
        inspector,
        inodes={(st.st_dev, st.st_ino)},
        index=_indexed(tmp_path, st),
    )
    rc, cleared = walks(monkeypatch, [str(tools)], blocked)
    assert rc != 0 and not cleared


def test_uncaptured_inodes_still_clear(tmp_path, monkeypatch, inspector):
    tools = tmp_path / "tools"
    other = tmp_path / "other"
    tools.mkdir()
    other.mkdir()
    executable(tools / "cmp")
    (tools / "hop").symlink_to(other / "elsewhere")
    executable(other / "elsewhere")
    executable(other / "unrelated")
    st = os.lstat(other / "unrelated")
    blocked = exclusions(
        mount("/", "/", "0:1", 1),
        mount("/", "/", "0:2", 1),
        (),
        inspector,
        inodes={(st.st_dev, st.st_ino)},
        index=_indexed(tmp_path, st),
    )
    rc, cleared = walks(monkeypatch, [str(tools)], blocked)
    assert rc == 0 and cleared


def test_batched_identity_count_mismatch_never_clears(tmp_path, monkeypatch, inspector):
    """A stat that answers for fewer files than it was asked about defers."""
    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "a")
    executable(tools / "b")
    st = os.lstat(tools / "a")
    blocked = exclusions(
        mount("/", "/", "0:1", 1),
        mount("/", "/", "0:2", 1),
        (),
        inspector,
        inodes={(1, 1)},
        index=_indexed(tmp_path, os.lstat(tools)),
    )
    stat = tmp_path / "inspector" / "stat"
    stat.write_text(f"#!/bin/sh\necho {st.st_dev}:{st.st_ino}\n")
    stat.chmod(0o700)
    assert not fast_cleared(command(monkeypatch, [str(tools)], blocked))


def test_known_identity_predicate_follows_the_shell_predicate():
    blocked = dataclasses.replace(
        PathExclusions.capture(mount("/", "/", "0:1", 1), ("/app",)),
        inodes=frozenset({(1, 2)}),
        inode_index="/i/x",
    )
    known = lockdown._known_identity_predicate(blocked.shell_predicate(), True)
    assert known.startswith("bf_inodes=1\nbf_under_known() {")
    assert "stat" not in known and "bf_inode=$2" in known
    assert lockdown._known_identity_predicate("bf_under() { return 1; }", False) == (
        'bf_inodes=\nbf_under_known() { bf_under "$1"; }'
    )
    with pytest.raises(RuntimeError, match="unexpected PATH predicate"):
        lockdown._known_identity_predicate("bf_under() { return 1; }", True)


def test_inspector_stat_disagreement_defers(tmp_path, inspector):
    """When the inspector's stat spells identities otherwise, never clear."""
    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "cmp")
    index = _indexed(tmp_path, os.lstat(tools))
    fake = tmp_path / "inspector" / "stat"
    fake.write_text("#!/bin/sh\necho 9:9\n")
    fake.chmod(0o700)
    blocked = exclusions(
        mount("/", "/", "0:1", 1),
        mount("/", "/", "0:2", 1),
        (),
        inspector,
        inodes={(1, 1)},
        index=index,
    )
    assert py_walk([str(tools)], blocked) == (4, "")


def test_walk_spec_refuses_text_it_cannot_carry():
    groups = [("/", ["/a\udcff"])]
    assert (
        _system_route_walk.walk_spec(groups, ("/bin",), index="", inspector="/i")
        is None
    )
    assert (
        _system_route_walk.walk_spec([("/", [])], ("/bin",), index="/i/x", inspector="")
        is None
    )


def test_the_walk_parses_as_python_3_6():
    ast.parse(_system_route_walk.SYSTEM_ROUTE_WALK, feature_version=(3, 6))


class _Env:
    """A sandbox that runs each command here, with a kit (Python or not)."""

    def __init__(self, python):
        self.commands = []
        self.runtime_kit = RuntimeKit(trusted=("/usr/bin",), python=python, bash=False)

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        result = subprocess.run(
            ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=60
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


async def _local_root_python_command(env, script, *args):
    import shlex

    from benchflow._root_python import _script

    return shlex.join([sys.executable, "-I", "-S", "-B", "-c", _script(script), *args])


@pytest.mark.parametrize("planted", [False, True])
async def test_judge_uses_the_python_walk_and_sh_decides_refusals(
    tmp_path, monkeypatch, inspector, planted
):
    tools = tmp_path / "tools"
    excluded = tmp_path / "excluded"
    tools.mkdir()
    excluded.mkdir()
    executable(tools / "honest")
    executable(excluded / "planted")
    if planted:
        (tools / "cmp").symlink_to(excluded / "planted")
    main = mount("/", "/", "0:1", 1) + mount("/volume/work", "/app")
    target = mount("/", "/", "0:2", 1) + mount("/volume/work", str(excluded))
    blocked = exclusions(main, target, ("/app",), inspector)
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(tools),))
    monkeypatch.setattr(lockdown, "root_python_command", _local_root_python_command)
    env = _Env(python=True)
    if planted:
        with pytest.raises(RuntimeError, match="excluded system PATH"):
            await lockdown._judge_trusted_entries(env, "", blocked, None)
        # The walk refused, so the sh walks ran and wrote the message.
        assert _FAST_DECISION in env.commands[-1]
    else:
        assert await lockdown._judge_trusted_entries(env, "", blocked, None)
        assert _FAST_DECISION not in env.commands[-1]
    assert "def route(rest):" in env.commands[0]


async def test_judge_without_kit_python_runs_the_sh_walks(
    tmp_path, monkeypatch, inspector
):
    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "honest")
    blocked = exclusions(
        mount("/", "/", "0:1", 1), mount("/", "/", "0:2", 1), (), inspector
    )
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(tools),))
    env = _Env(python=False)
    await lockdown._judge_trusted_entries(env, "", blocked, None)
    assert len(env.commands) == 1 and _FAST_DECISION in env.commands[0]


def test_predicate_groups_drive_the_shell_predicate():
    import shlex

    blocked = PathExclusions.capture(
        mount("/", "/", "0:1", 1)
        + mount("/volume/work", "/app")
        + mount("/other", "/app/sub", "8:2", 3),
        ("/app",),
    )
    groups = blocked.predicate_groups()
    assert [t for t, _ in groups] == ["/app/sub", "/app", "/"]
    predicate = blocked.shell_predicate()
    for target, paths in groups:
        assert shlex.quote(target) in predicate
        for path in paths:
            assert shlex.quote(path) in predicate
