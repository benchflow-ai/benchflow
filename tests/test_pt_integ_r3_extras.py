"""The root-only image PATH/PYTHONPATH check without a kit
Python.

On a conda or pyenv image the protected kit holds no Python, so the Python
check exited 127 and the verifier lost every image PATH entry (a conda task's
``python`` was not found; integ scored it 1). The same rule now runs on facts
the kit's own ``realpath`` and ``stat`` report, decided on the host: an entry
is kept only when it, and every parent, is root-owned and not group- or
world-writable; anything unreadable or unknown is dropped. One test per rule
branch, then the real probe text against this host's filesystem.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._runtime_tree import KIT_BIN, KIT_PYTHON, KIT_RECORD
from benchflow.sandbox import lockdown

DIR = "directory"
LINK = "symbolic link"


def _chain(kind: str, path: str, facts: dict[str, str]) -> list[str]:
    """``kind path stat`` triples for *path* and each parent, from *facts*
    (path -> "uid mode type"); a missing path reads as unreadable."""
    out = []
    p = path
    while True:
        out += [kind, p, facts.get(p, "?")]
        if p == "/":
            return out
        p = os.path.dirname(p.rstrip("/")) or "/"


def _probe(entries: list[tuple[str, str | None]], facts: dict[str, str]) -> str:
    fields: list[str] = []
    for entry, real in entries:
        fields += ["E", entry]
        if real is None:
            fields.append("X")
            continue
        fields += ["R", real, *_chain("R", real, facts), *_chain("L", entry, facts)]
    return "\0".join([*fields, "END", ""])


ROOT_DIRS = {
    "/": f"0 755 {DIR}",
    "/opt": f"0 755 {DIR}",
    "/opt/conda": f"0 755 {DIR}",
    "/opt/conda/bin": f"0 755 {DIR}",
}


def _decide(entries, facts, blocked=()):
    return lockdown._decide_root_only_extras(
        [e for e, _ in entries], tuple(blocked), _probe(entries, facts)
    )


def test_a_root_owned_folder_with_root_owned_parents_is_kept():
    assert _decide([("/opt/conda/bin", "/opt/conda/bin")], ROOT_DIRS) == [
        "/opt/conda/bin"
    ]


@pytest.mark.parametrize(
    "path, facts",
    [
        ("/opt/conda/bin", f"0 775 {DIR}"),
        ("/opt/conda/bin", f"0 757 {DIR}"),
        ("/opt/conda", f"0 777 {DIR}"),
        ("/opt", f"0 1777 {DIR}"),
        ("/opt/conda", f"1000 755 {DIR}"),
    ],
    ids=[
        "group-writable",
        "world-writable",
        "writable-parent",
        "sticky-parent",
        "agent-owned-parent",
    ],
)
def test_a_writable_or_foreign_folder_or_parent_drops_the_entry(path, facts):
    assert (
        _decide([("/opt/conda/bin", "/opt/conda/bin")], {**ROOT_DIRS, path: facts})
        == []
    )


def test_a_root_owned_link_in_the_path_is_followed_and_kept():
    facts = {**ROOT_DIRS, "/usr": f"0 755 {DIR}", "/usr/conda": f"0 777 {LINK}"}
    # /usr/conda/bin -> /opt/conda/bin through a root-owned link (a link's own
    # mode means nothing); the target and its parents are root's alone.
    assert _decide(
        [("/usr/conda/bin", "/opt/conda/bin")],
        {**facts, "/usr/conda/bin": f"0 755 {DIR}"},
    ) == ["/usr/conda/bin"]


@pytest.mark.parametrize(
    "facts",
    [
        {
            "/usr": f"0 755 {DIR}",
            "/usr/conda": f"1000 777 {LINK}",
            "/usr/conda/bin": f"0 755 {DIR}",
        },
        {
            "/usr": f"0 777 {DIR}",
            "/usr/conda": f"0 777 {LINK}",
            "/usr/conda/bin": f"0 755 {DIR}",
        },
    ],
    ids=["agent-owned-link", "link-in-a-writable-folder"],
)
def test_a_link_the_agent_could_change_drops_the_entry(facts):
    assert _decide([("/usr/conda/bin", "/opt/conda/bin")], {**ROOT_DIRS, **facts}) == []


def test_a_link_to_a_writable_target_drops_the_entry():
    facts = {**ROOT_DIRS, "/srv": f"0 777 {DIR}", "/srv/py": f"0 755 {DIR}"}
    assert _decide([("/opt/conda/bin", "/srv/py")], facts) == []


def test_an_unreadable_component_drops_the_entry():
    facts = dict(ROOT_DIRS)
    del facts["/opt/conda"]  # stat failed: "?"
    assert _decide([("/opt/conda/bin", "/opt/conda/bin")], facts) == []


def test_an_entry_that_does_not_resolve_is_dropped_and_the_rest_kept():
    entries = [("/opt/missing", None), ("/opt/conda/bin", "/opt/conda/bin")]
    assert _decide(entries, ROOT_DIRS) == ["/opt/conda/bin"]


def test_a_blocked_prefix_drops_the_entry_either_way():
    assert (
        _decide([("/opt/conda/bin", "/opt/conda/bin")], ROOT_DIRS, ["/opt/conda"]) == []
    )
    facts = {**ROOT_DIRS, "/app": f"0 755 {DIR}", "/app/bin": f"0 777 {LINK}"}
    assert _decide([("/app/bin", "/opt/conda/bin")], facts, ["/app"]) == []


@pytest.mark.parametrize("cut", ["no-end", "half-entry", "garbage"])
def test_a_cut_or_malformed_probe_keeps_nothing(cut):
    full = _probe([("/opt/conda/bin", "/opt/conda/bin")], ROOT_DIRS)
    stdout = {
        "no-end": full[: -len("END\0")],
        "half-entry": "E\0/opt/conda/bin\0R\0",
        "garbage": "nonsense",
    }[cut]
    assert lockdown._decide_root_only_extras(["/opt/conda/bin"], (), stdout) == []


@pytest.mark.asyncio
async def test_no_kit_or_a_kit_python_leaves_the_python_check(monkeypatch):
    async def read_exec(env, command, **kwargs):
        return SimpleNamespace(return_code=3, stdout="", stderr="")

    monkeypatch.setattr(lockdown, "_read_exec", read_exec)
    assert (
        await lockdown._extras_without_kit_python(object(), "/opt/conda/bin", ())
        is None
    )


def test_the_probe_text_reports_real_facts(tmp_path):
    """The probe text itself, the kit's paths mapped to this host's tools:
    root's /usr/lib is kept; the test's own folder (not root's) is dropped."""
    record = tmp_path / "RECORD"
    record.write_text("kit\n")
    mine = tmp_path / "mine"
    mine.mkdir()
    entries = ["/usr/lib", str(mine), str(tmp_path / "missing")]
    command = (
        lockdown._extras_probe_cmd(entries)
        .replace(KIT_RECORD, str(record))
        .replace(KIT_PYTHON, str(tmp_path / "no-python"))
        .replace(f"{KIT_BIN}/", "/usr/bin/")
    )
    done = subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    kept = lockdown._decide_root_only_extras(entries, (), done.stdout)
    usr_lib = Path("/usr/lib")
    expected = (
        ["/usr/lib"]
        if usr_lib.stat().st_uid == 0
        and not usr_lib.stat().st_mode & 0o022
        and Path("/usr").stat().st_uid == 0
        else []
    )
    assert kept == expected
    assert str(mine) not in kept or os.getuid() == 0
    assert shlex.quote(str(mine)) in command


def test_without_a_kit_the_probe_uses_the_system_tools(tmp_path):
    """No kit (no sandbox user) and no python3 on root's route: the same
    probe runs with the system folders' realpath/stat (PATH pinned)."""
    entries = ["/usr/lib", str(tmp_path)]
    command = lockdown._extras_probe_cmd(entries).replace(
        KIT_RECORD, str(tmp_path / "no-record")
    )
    done = subprocess.run(
        ["/bin/sh", "-c", command],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/nonexistent"},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.endswith("END\0")
    kept = lockdown._decide_root_only_extras(entries, (), done.stdout)
    assert str(tmp_path) not in kept or os.getuid() == 0


def test_a_kit_with_a_python_leaves_the_python_check(tmp_path):
    record = tmp_path / "RECORD"
    record.write_text("kit\n")
    python = tmp_path / "python3"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    command = (
        lockdown._extras_probe_cmd(["/usr/lib"])
        .replace(KIT_RECORD, str(record))
        .replace(KIT_PYTHON, str(python))
    )
    done = subprocess.run(["/bin/sh", "-c", command], capture_output=True, timeout=60)
    assert done.returncode == 3


def test_a_busybox_realpath_without_options_still_resolves(tmp_path):
    """PT-INTEG-R3B-1: busybox's realpath accepts no options, so ``-e --``
    made every entry unresolved (dropped) on a busybox kit. A stand-in that
    refuses any option still resolves the probe's entries."""
    kit_bin = tmp_path / "kit-bin"
    kit_bin.mkdir()
    (kit_bin / "realpath").write_text(
        '#!/bin/sh\ncase "$1" in -*) echo "realpath: unrecognized option" >&2; exit 1;; esac\n'
        'exec /usr/bin/realpath "$@"\n'
    )
    (kit_bin / "realpath").chmod(0o755)
    for tool in ("stat", "dirname"):
        (kit_bin / tool).symlink_to(f"/usr/bin/{tool}")
    record = tmp_path / "RECORD"
    record.write_text("kit\n")
    command = (
        lockdown._extras_probe_cmd(["/usr/lib"])
        .replace(KIT_RECORD, str(record))
        .replace(KIT_PYTHON, str(tmp_path / "no-python"))
        .replace(f"{KIT_BIN}/", f"{kit_bin}/")
    )
    done = subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    fields = done.stdout.split("\0")
    assert fields[:3] == ["E", "/usr/lib", "R"], fields[:4]
