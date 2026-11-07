"""Guards the fix at 363d815b: keep protected installed runtimes."""

import os
from pathlib import Path

import pytest

from benchflow.sandbox import _verifier_tool_state as state


@pytest.fixture(autouse=True)
def image_umask():
    # Image trees are made under umask 022. A gate whose umask is 002 (a
    # login on a shared host) would make every directory the test creates
    # group-writable, which these checks rightly refuse.
    old = os.umask(0o022)
    yield
    os.umask(old)


def _image_ownership(monkeypatch, trees):
    ancestors = {
        (st.st_dev, st.st_ino)
        for tree in trees
        for parent in Path(tree).parents
        for st in [parent.stat()]
    }
    monkeypatch.setattr(
        state,
        "owned_safely",
        lambda st: (st.st_dev, st.st_ino) in ancestors or not st.st_mode & 0o022,
    )


@pytest.mark.parametrize("explicit", [True, False])
def test_root_protected_managed_runtime_remains_discoverable(
    tmp_path, monkeypatch, explicit
):
    monkeypatch.setattr(state, "owned_safely", lambda st: True)
    install = tmp_path / "root/.local/share/uv/python"
    executable = install / "cpython-3.11/bin/python3.11"
    executable.parent.mkdir(parents=True)
    executable.write_text("trusted installed interpreter\n")
    env = {"HOME": str(tmp_path / "root")}
    if explicit:
        env["UV_PYTHON_INSTALL_DIR"] = str(install)
    found = state.overrides(env, ("/app",), str(tmp_path / "fresh"))
    assert found.get("UV_PYTHON_INSTALL_DIR", str(install)) == str(install)
    assert os.path.isfile(executable)


@pytest.mark.parametrize(
    "unsafe", ["workspace", "writable-descendant", "symlink", "linked-descendant"]
)
def test_agent_writable_managed_runtime_is_replaced(tmp_path, monkeypatch, unsafe):
    install = tmp_path / "install"
    install.mkdir()
    blocked = tmp_path / "agent"
    blocked.mkdir()
    _image_ownership(monkeypatch, [install, tmp_path / "linked"])
    if unsafe == "workspace":
        prefixes = (str(install),)
    else:
        prefixes = (str(blocked),)
        if unsafe == "symlink":
            (install / "unsafe").symlink_to(blocked)
        elif unsafe == "linked-descendant":
            linked = tmp_path / "linked"
            linked.mkdir()
            (linked / "unsafe").mkdir(mode=0o777)
            (linked / "unsafe").chmod(0o777)
            (install / "link").symlink_to(linked)
        else:
            (install / "unsafe").mkdir(mode=0o777)
            (install / "unsafe").chmod(0o777)
    found = state.overrides(
        {"UV_PYTHON_INSTALL_DIR": str(install)}, prefixes, str(tmp_path / "fresh")
    )
    assert found["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / "fresh/uv-python")


def test_trusted_runtime_directory_cycle_does_not_hang(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "owned_safely", lambda st: True)
    install = tmp_path / "install"
    install.mkdir()
    (install / "loop").symlink_to(install)
    assert state.protected_python_installation(str(install), ("/app",))


def test_managed_runtime_is_seeded_without_shared_locks_or_writable_trees(
    tmp_path, monkeypatch
):
    source = tmp_path / "installed"
    python = source / "cpython-3.11/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("trusted image runtime")
    (source / ".lock").write_text("untrusted lock contents")
    (source / ".lock").chmod(0o666)
    unsafe = source / "cpython-3.10"
    unsafe.mkdir()
    unsafe.chmod(0o777)
    _image_ownership(monkeypatch, [source])
    destination = tmp_path / "fresh"
    state.seed_managed_python(
        {"UV_PYTHON_INSTALL_DIR": str(source)}, (), str(destination)
    )
    assert (
        destination / "cpython-3.11/bin/python"
    ).read_text() == "trusted image runtime"
    assert not (destination / ".lock").exists()
    assert not (destination / "cpython-3.10").exists()
