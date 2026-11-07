"""Guard CONFIRM-lockdown-6 at bd23e190 across directory and symlink ancestry."""

import os
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _workspace_ancestry as ancestry


@pytest.fixture(autouse=True)
def image_umask():
    # Image trees are made under umask 022. A gate whose umask is 002 (a
    # login on a shared host) would make every directory the test creates
    # group-writable, which these checks rightly refuse.
    old = os.umask(0o022)
    yield
    os.umask(old)


@pytest.fixture
def ownership(monkeypatch):
    # Exercise real descriptor-relative lookups even in an unprivileged gate.
    # The ownership view alone models root-created image trees and one agent uid.
    actual_stat, actual_fstat = os.stat, os.fstat
    agents = set()

    def view(st):
        return SimpleNamespace(
            st_mode=st.st_mode, st_uid=1500 if st.st_ino in agents else 0
        )

    system = SimpleNamespace(**vars(os))
    system.stat = lambda *a, **kw: view(actual_stat(*a, **kw))
    system.fstat = lambda fd: view(actual_fstat(fd))
    monkeypatch.setattr(ancestry, "os", system)
    return agents, actual_stat


@pytest.mark.parametrize("component", ["job", "parent", "leaf"])
@pytest.mark.parametrize("mode", [0o777, 0o775, 0o757, 0o700])
def test_mutability_at_every_ancestor_is_refused(tmp_path, ownership, component, mode):
    agents, actual_stat = ownership
    workspace = tmp_path / "job" / "parent" / "leaf"
    workspace.mkdir(parents=True)
    selected = {
        "job": workspace.parent.parent,
        "parent": workspace.parent,
        "leaf": workspace,
    }[component]
    selected.chmod(mode)
    if mode == 0o700:
        agents.add(actual_stat(selected).st_ino)
    with pytest.raises(ValueError, match="workspace"):
        ancestry.check(str(workspace))


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("agent_child", [False, True])
def test_sticky_parents_protect_only_root_owned_entries(
    tmp_path, ownership, sticky, agent_child
):
    agents, actual_stat = ownership
    parent = tmp_path / "job"
    workspace = parent / "submission"
    workspace.mkdir(parents=True)
    parent.chmod(0o1777 if sticky else 0o777)
    if agent_child:
        agents.add(actual_stat(workspace).st_ino)
    if sticky and not agent_child:
        ancestry.check(str(workspace))
    else:
        with pytest.raises(ValueError, match="workspace"):
            ancestry.check(str(workspace))


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("mutable", ["none", "logical", "target", "intermediate"])
def test_every_symlink_resolution_component_is_checked(
    tmp_path, ownership, relative, mutable
):
    logical, middle, physical = [
        tmp_path / name for name in ("logical", "middle", "physical")
    ]
    for directory in (logical, middle, physical):
        directory.mkdir()
    workspace = physical / "submission ' [*?]\\\n"
    workspace.mkdir()
    (middle / "jump").symlink_to("../physical" if relative else physical)
    (logical / "workspace").symlink_to(
        "../middle/jump/" + workspace.name
        if relative
        else str(middle / "jump" / workspace.name)
    )
    if mutable != "none":
        {"logical": logical, "target": physical, "intermediate": middle}[mutable].chmod(
            0o777
        )
        with pytest.raises(ValueError, match="workspace"):
            ancestry.check(str(logical / "workspace"))
    else:
        ancestry.check(str(logical / "workspace"))


def test_nested_submission_links_are_not_traversed(tmp_path, ownership):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "external").symlink_to("/missing/dummy-target")
    ancestry.check(str(workspace))


@pytest.mark.parametrize("agent_link", [False, True])
def test_sticky_symlink_entry_must_also_be_root_owned(tmp_path, ownership, agent_link):
    agents, actual_stat = ownership
    parent = tmp_path / "sticky"
    parent.mkdir(mode=0o1777)
    parent.chmod(0o1777)
    target = tmp_path / "physical"
    target.mkdir()
    workspace = parent / "link"
    workspace.symlink_to(target)
    if agent_link:
        agents.add(actual_stat(workspace, follow_symlinks=False).st_ino)
        with pytest.raises(ValueError, match="workspace"):
            ancestry.check(str(workspace))
    else:
        ancestry.check(str(workspace))


def test_dotdot_does_not_erase_a_mutable_resolution_component(tmp_path, ownership):
    (tmp_path / "mutable").mkdir(mode=0o777)
    (tmp_path / "mutable").chmod(0o777)
    (tmp_path / "safe").mkdir()
    (tmp_path / "mutable" / "jump").symlink_to("../safe")
    with pytest.raises(ValueError, match="workspace"):
        ancestry.check(str(tmp_path) + "/mutable/jump/../safe")


@pytest.mark.parametrize("kind", ["relative", "missing", "file", "loop"])
def test_unresolvable_workspace_fails_closed(tmp_path, ownership, kind):
    workspace = tmp_path / "workspace"
    if kind == "file":
        workspace.write_text("dummy")
    elif kind == "loop":
        workspace.symlink_to("workspace")
    with pytest.raises((OSError, ValueError)):
        ancestry.check("relative" if kind == "relative" else str(workspace))
