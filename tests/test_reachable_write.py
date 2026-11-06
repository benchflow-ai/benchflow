"""benchflow._utils.reachable_write: writes into sandbox-reachable folders (R4-1).

Every writer is tried against every entry a sandbox can leave at the name or
on the way to it: absolute and relative links, a FIFO, a hard link, a folder,
a folder on the way replaced by a link, and a folder swapped between the
check and the rename. A dummy controller marker outside the root must keep
its bytes in every case.
"""

import os
from pathlib import Path

import pytest

from benchflow._utils import reachable_write as rw

MARKER = "DUMMY_CONTROLLER_MARKER"


@pytest.fixture
def layout(tmp_path):
    rollout = tmp_path / "rollout"
    root = rollout / "verifier"
    root.mkdir(parents=True)
    controller = tmp_path / "controller-only"
    controller.mkdir()
    victim = controller / "marker.json"
    victim.write_text(MARKER)
    return root, controller, victim


def _plant(name: Path, kind: str, victim: Path, root: Path) -> Path | None:
    """Leave ``kind`` at ``name``; return a file that must keep its bytes, if any."""
    if kind == "absolute-link":
        name.symlink_to(victim)
    elif kind == "relative-link":
        name.symlink_to(os.path.relpath(victim, start=name.parent))
    elif kind == "fifo":
        os.mkfifo(name)
    elif kind == "hard-link":
        shared = root / "shared.txt"
        shared.write_text(MARKER)
        os.link(shared, name)
        return shared
    elif kind == "folder":
        name.mkdir()
        (name / "inner").symlink_to(victim)
    return None


KINDS = ["absolute-link", "relative-link", "fifo", "hard-link", "folder"]


def _write(writer: str, path: Path, root: Path, private: Path | None) -> bytes:
    if writer == "write":
        rw.write_reachable(path, "new", root=root, parents=True)
        return b"new"
    assert private is not None
    staged = private / "staged"
    staged.write_text("placed")
    rw.place_reachable(path, staged, root=root, parents=True)
    return b"placed"


@pytest.mark.parametrize("writer", ["write", "place"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("nested", [False, True])
def test_every_planted_entry_is_replaced_never_followed(layout, writer, kind, nested):
    root, _, victim = layout
    folder = root / "judge" / "deep" if nested else root
    folder.mkdir(parents=True, exist_ok=True)
    name = folder / "reward.json"
    kept = _plant(name, kind, victim, root)
    with rw.private_folder(root) as private:
        expected = _write(writer, name, root, private)
    assert victim.read_text() == MARKER
    if kept is not None:
        assert kept.read_text() == MARKER
    assert name.is_file() and not name.is_symlink()
    assert name.read_bytes() == expected
    assert oct(name.stat().st_mode & 0o777) == oct(0o644) or writer == "place"


@pytest.mark.parametrize("writer", ["write", "place", "append", "rename"])
@pytest.mark.parametrize("form", ["absolute", "relative"])
def test_a_folder_on_the_way_replaced_by_a_link_refuses(layout, writer, form):
    root, controller, victim = layout
    target = controller if form == "absolute" else os.path.relpath(controller, root)
    (root / "judge").symlink_to(target, target_is_directory=True)
    path = root / "judge" / "marker.json"
    with pytest.raises(rw.ReachableWriteError), rw.private_folder(root) as private:
        if writer == "write":
            rw.write_reachable(path, "new", root=root)
        elif writer == "place":
            (private / "f").write_text("new")
            rw.place_reachable(path, private / "f", root=root)
        elif writer == "append":
            rw.append_reachable(path, "new", root=root)
        else:
            rw.rename_reachable(path, "other.json", root=root)
    assert victim.read_text() == MARKER
    assert sorted(p.name for p in controller.iterdir()) == ["marker.json"]


@pytest.mark.parametrize("writer", ["write", "place", "rename"])
@pytest.mark.parametrize("depth", [1, 2])
def test_a_folder_swapped_between_check_and_rename_refuses(
    layout, monkeypatch, writer, depth
):
    """The held handle keeps the write in the original folder; the swap is refused."""
    root, controller, victim = layout
    parts = ["judge", "deep"][:depth]
    folder = root.joinpath(*parts)
    folder.mkdir(parents=True)
    (folder / "old.json").write_text("old")
    swapped = folder.with_name(folder.name + "-moved")
    real_check = rw._check_in_place
    calls = []

    def check_then_swap(*args, **kwargs):
        real_check(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            folder.rename(swapped)
            folder.symlink_to(controller, target_is_directory=True)

    monkeypatch.setattr(rw, "_check_in_place", check_then_swap)
    with pytest.raises(rw.ReachableWriteError), rw.private_folder(root) as private:
        if writer == "write":
            rw.write_reachable(folder / "marker.json", "new", root=root)
        elif writer == "place":
            (private / "f").write_text("new")
            rw.place_reachable(folder / "marker.json", private / "f", root=root)
        else:
            rw.rename_reachable(folder / "old.json", "marker.json", root=root)
    assert victim.read_text() == MARKER
    assert sorted(p.name for p in controller.iterdir()) == ["marker.json"]


def test_a_root_that_is_a_link_refuses(layout):
    root, controller, victim = layout
    linked = root.parent / "linked"
    linked.symlink_to(controller, target_is_directory=True)
    with pytest.raises(rw.ReachableWriteError):
        rw.write_reachable(linked / "marker.json", "new", root=linked)
    assert victim.read_text() == MARKER


def test_paths_outside_the_root_are_refused(layout):
    root, controller, victim = layout
    with pytest.raises(ValueError):
        rw.write_reachable(
            root / ".." / ".." / "controller-only" / "marker.json", "x", root=root
        )
    with pytest.raises(ValueError):
        rw.write_reachable(controller / "marker.json", "x", root=root)
    with pytest.raises(ValueError):
        rw.write_reachable(root, "x", root=root)
    assert victim.read_text() == MARKER


def test_private_custody_is_outside_the_root_and_released(layout, monkeypatch):
    """R13 replaces named staging with an anonymous descriptor (7c24f4d88 R12-2)."""
    root, _, _ = layout
    seen = []
    real_publish = rw._publish_stream

    def publish(source, target, **kwargs):
        seen.append(source)
        assert os.fstat(source.fileno()).st_nlink == 0
        return real_publish(source, target, **kwargs)

    monkeypatch.setattr(rw, "_publish_stream", publish)
    rw.write_reachable(root / "reward.json", "{}", root=root)
    assert seen and seen[0].closed
    assert [p.name for p in root.parent.iterdir()] == ["verifier"]


def test_placed_trees_keep_links_as_links_and_retire_old_entries(layout):
    root, _, victim = layout
    old = root / "taskmd-judge"
    old.mkdir()
    (old / "trap").symlink_to(victim)
    with rw.private_folder(root) as private:
        tree = private / "tree"
        tree.mkdir()
        (tree / "record.json").write_text("{}")
        (tree / "inert").symlink_to("/nonexistent")
        rw.place_reachable(old, tree, root=root)
    assert sorted(p.name for p in old.iterdir()) == ["inert", "record.json"]
    assert (old / "inert").is_symlink()
    assert victim.read_text() == MARKER
    assert [p.name for p in root.parent.iterdir()] == ["verifier"]


def test_entries_of_a_download_replace_their_own_names(layout):
    root, controller, victim = layout
    (root / "reward.json").symlink_to(victim)
    (root / "judge").symlink_to(controller, target_is_directory=True)
    (root / "kept.txt").write_text("kept")
    with rw.private_folder(root) as private:
        landing = private / "verifier"
        (landing / "judge").mkdir(parents=True)
        (landing / "reward.json").write_text('{"reward": 1}')
        (landing / "judge" / "notes.txt").write_text("notes")
        rw.place_reachable_entries(root, landing, root=root)
    assert victim.read_text() == MARKER
    assert sorted(p.name for p in controller.iterdir()) == ["marker.json"]
    assert (root / "reward.json").read_text() == '{"reward": 1}'
    assert (root / "judge" / "notes.txt").read_text() == "notes"
    assert (root / "kept.txt").read_text() == "kept"


@pytest.mark.parametrize("kind", ["absolute-link", "relative-link", "fifo", "folder"])
def test_append_refuses_a_planted_entry_and_appends_to_a_file(layout, kind):
    root, _, victim = layout
    name = root / "judgments.jsonl"
    _plant(name, kind, victim, root)
    with pytest.raises(rw.ReachableWriteError):
        rw.append_reachable(name, "x\n", root=root)
    assert victim.read_text() == MARKER


def test_append_never_writes_through_a_hard_link(layout):
    root, _, _ = layout
    shared = root / "shared.txt"
    shared.write_text("a\n")
    os.link(shared, root / "judgments.jsonl")
    rw.append_reachable(root / "judgments.jsonl", "b\n", root=root)
    assert shared.read_text() == "a\n"
    assert (root / "judgments.jsonl").read_text() == "a\nb\n"


def test_rename_within_the_folder_replaces_a_link_at_the_new_name(layout):
    root, _, victim = layout
    (root / "reward.txt").write_text("1\n")
    (root / "script-reward.txt").symlink_to(victim)
    rw.rename_reachable(root / "reward.txt", "script-reward.txt", root=root)
    assert victim.read_text() == MARKER
    assert (root / "script-reward.txt").read_text() == "1\n"
    assert not (root / "reward.txt").exists()
    with pytest.raises(ValueError):
        rw.rename_reachable(root / "script-reward.txt", "../x", root=root)


@pytest.mark.parametrize("kind", KINDS)
def test_refuse_special_refuses_without_writing(layout, kind):
    root, _, victim = layout
    name = root / "review.json"
    kept = _plant(name, kind, victim, root)
    if kind == "hard-link":
        rw.write_reachable(name, "new", root=root, refuse_special=True)
        assert kept is not None and kept.read_text() == MARKER
        assert name.read_text() == "new"
        return
    with pytest.raises(rw.ReachableWriteError):
        rw.write_reachable(name, "new", root=root, refuse_special=True)
    assert victim.read_text() == MARKER
    assert os.path.lexists(name) and not (name.is_file() and not name.is_symlink())
