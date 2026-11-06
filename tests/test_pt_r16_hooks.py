"""PT round 16: one rule for Python start-up hooks, in the record and the clean-up.

Guards the fix of PT-R15-B1, PT-R15-B2 and PT-R15-B3 (reviews of daec118b):

- B1: ``sitecustomize`` and ``usercustomize`` in any form the import system
  finds (a package folder, a sourceless ``.pyc``, an extension module) are
  hooks in every search folder;
- B3: a ``.pth`` is a hook only in a site folder; elsewhere on the search
  path (``/app/model.pth``) it is data and stays;
- B2: a kept hook is a fresh copy, so a descriptor held on the old file
  never reaches what the verifier runs; a link at a hook name is never
  followed.

Each test takes the real record (CAPTURE, run as the kit runs it, ``-I``, by
a throwaway venv's Python, with test.sh's search-path inputs as arguments:
the project folder on PYTHONPATH) and runs the real clean-up script
(``lockdown._CLEANUP_SITE_SCRIPT``) on temporary folders, with dummy markers.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile

import pytest

from benchflow._image_python_paths import (
    CAPTURE,
    IMAGE_PYTHON_PATHS,
    IMAGE_PYTHON_PATHS_EACH,
)
from benchflow.sandbox import lockdown

MARK = "# dummy marker\n"


@pytest.fixture
def layout(tmp_path):
    """A venv (its site-packages is a site folder) and a project folder on
    its search path through PYTHONPATH (not a site folder)."""
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True
    )
    py = venv / "bin" / "python"
    site = subprocess.run(
        [str(py), "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    app = tmp_path / "app"
    app.mkdir()
    (app / "solution.py").write_text("answer = 42\n")
    return {"py": py, "site": site, "app": app, "tmp": tmp_path}


def _inputs(layout) -> dict[str, str]:
    """test.sh's search-path inputs as root's route hands them over, in the
    environment, never argv: PYTHONPATH (the project
    folder first), PYTHONHOME, PYTHONUSERBASE."""
    pythonpath = os.pathsep.join(map(str, layout.get("pythonpath", [layout["app"]])))
    return {
        "_BF_HOOK_PYTHONPATH": pythonpath,
        "_BF_HOOK_PYTHONHOME": layout.get("home", ""),
        "_BF_HOOK_PYTHONUSERBASE": layout.get("userbase", ""),
    }


def _cwd(layout) -> str:
    """The working folder, the one search-path input in argv."""
    return str(layout.get("cwd", "/"))


def _capture(layout, *extra) -> None:
    """The record as the kit takes it (PT-R16-B-2): ``-I -B``, so the
    project folder reaches it only through the inputs, never sys.path."""
    layout["pythonpath"] = [layout["app"], *extra]
    subprocess.run(
        [str(layout["py"]), "-I", "-B", "-c", CAPTURE,
         str(layout["tmp"] / "paths.json"), _cwd(layout)],
        env={"PATH": "/usr/bin:/bin", "HOME": str(layout["tmp"]), **_inputs(layout)},
        cwd="/",
        check=True,
    )  # fmt: skip


def _run_clean(layout) -> subprocess.CompletedProcess:
    script = lockdown._CLEANUP_SITE_SCRIPT.replace(
        IMAGE_PYTHON_PATHS, str(layout["tmp"] / "paths.json")
    ).replace(IMAGE_PYTHON_PATHS_EACH, str(layout["tmp"] / "none.jsonl"))
    # An empty PATH: the clean-up adds the site folders of the first
    # python3 on it, which must not be the test host's.
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, _cwd(layout)],
        capture_output=True,
        text=True,
        env={"PATH": "", **_inputs(layout)},
    )


def _clean(layout) -> None:
    done = _run_clean(layout)
    assert done.returncode == 0, done.stderr


def _hook_forms(folder, stem):
    package = folder / stem
    package.mkdir()
    (package / "__init__.py").write_text(MARK)
    return {
        "package": package,
        "pyc": folder / f"{stem}.pyc",
        "so": folder / f"{stem}.cpython-312-x86_64-linux-gnu.so",
    }


@pytest.mark.parametrize("where", ["app", "site"])
@pytest.mark.parametrize("stem", ["sitecustomize", "usercustomize"])
def test_every_form_of_a_new_customize_hook_is_removed(layout, where, stem):
    """PT-R15-B1, fails on daec118b: a package, a sourceless .pyc and an
    extension module named sitecustomize or usercustomize were not hooks to
    the clean-up, so ``import sitecustomize`` ran the agent's one as root."""
    _capture(layout)
    folder = layout["app"] if where == "app" else layout["tmp"] / layout["site"]
    forms = _hook_forms(folder, stem)
    forms["pyc"].write_bytes(b"\x00dummy")
    forms["so"].write_bytes(b"\x7fELFdummy")
    _clean(layout)
    left = {name for name, path in forms.items() if os.path.lexists(path)}
    assert left == set()
    assert (layout["app"] / "solution.py").read_text() == "answer = 42\n"


def test_a_pth_outside_a_site_folder_is_data_and_stays(layout):
    """PT-R15-B3, fails on daec118b: /app/model.pth (PyTorch weights, which
    site never reads in a non-site folder) was removed and the honest
    solution scored 0. A new .pth in the venv's site-packages still goes."""
    _capture(layout)
    model = layout["app"] / "model.pth"
    model.write_bytes(b"PK\x03\x04 dummy weights")
    planted = layout["tmp"] / layout["site"] / "zz-agent.pth"
    planted.write_text("import os  # dummy marker\n")
    _clean(layout)
    assert model.read_bytes() == b"PK\x03\x04 dummy weights"
    assert not planted.exists()


def test_a_kept_pth_is_a_fresh_file_a_held_descriptor_cannot_reach(layout):
    """PT-R15-B2, fails on daec118b: the honest .pth matched its record and
    was kept as the same inode, so a descriptor a kept service held wrote
    into what the verifier's Python reads."""
    honest = layout["tmp"] / layout["site"] / "_dummy_proj.pth"
    honest.write_text(str(layout["app"]) + "\n")
    _capture(layout)
    with open(honest, "r+") as held:
        _clean(layout)
        held.seek(0)
        held.write("import os  # dummy marker written late\n")
    assert honest.read_text() == str(layout["app"]) + "\n"


def test_a_link_at_a_hook_name_is_never_followed(layout):
    """PT-R15-B2, fails on daec118b: sha() followed a link, so a recorded
    sitecustomize.py swapped for a link to an agent-writable file with the
    same bytes was kept, and the file behind it could change later. The
    link is not the recorded file, so it goes; the file it pointed to is
    not touched."""
    hook = layout["app"] / "sitecustomize.py"
    hook.write_text(MARK)
    _capture(layout)
    victim = layout["tmp"] / "victim-a.py"
    victim.write_text(MARK)
    hook.unlink()
    hook.symlink_to(victim)
    _clean(layout)
    assert not os.path.lexists(hook)
    assert victim.read_text() == MARK


def test_an_image_link_at_a_hook_name_becomes_a_copy_of_its_file(layout):
    """Debian links /usr/lib/python3.X/sitecustomize.py to /etc: kept, as a
    fresh regular file with the same bytes, the file in /etc untouched."""
    etc = layout["tmp"] / "etc-sitecustomize.py"
    etc.write_text(MARK)
    hook = layout["app"] / "sitecustomize.py"
    hook.symlink_to(etc)
    _capture(layout)
    _clean(layout)
    assert not hook.is_symlink()
    assert hook.read_text() == MARK
    assert etc.read_text() == MARK


def test_a_package_hook_is_kept_whole_or_removed_whole(layout):
    """PT-R15-B1: a sitecustomize package is recorded by its whole tree. An
    unchanged one is kept (a fresh copy); one with a file added goes."""
    kept = layout["tmp"] / layout["site"] / "sitecustomize"
    kept.mkdir()
    (kept / "__init__.py").write_text(MARK)
    changed = layout["app"] / "usercustomize"
    changed.mkdir()
    (changed / "__init__.py").write_text(MARK)
    _capture(layout)
    before = os.stat(kept / "__init__.py").st_ino
    (changed / "extra.py").write_text(MARK)
    _clean(layout)
    assert (kept / "__init__.py").read_text() == MARK
    assert os.stat(kept / "__init__.py").st_ino != before
    assert not os.path.lexists(changed)


def test_a_hook_bytecode_cache_is_dropped(layout):
    """An unchecked-hash .pyc in __pycache__ would run instead of a kept
    sitecustomize.py: caches of a hook are always dropped."""
    (layout["app"] / "sitecustomize.py").write_text(MARK)
    _capture(layout)
    cache = layout["app"] / "__pycache__"
    cache.mkdir()
    (cache / "sitecustomize.cpython-312.pyc").write_bytes(b"\x00dummy")
    (cache / "solution.cpython-312.pyc").write_bytes(b"\x00dummy")
    _clean(layout)
    assert sorted(os.listdir(cache)) == ["solution.cpython-312.pyc"]
    assert (layout["app"] / "sitecustomize.py").read_text() == MARK


def test_a_new_zip_archive_on_the_search_path_is_removed(layout):
    """A search path entry that was absent before the agent and is now a
    zip archive would let ``import sitecustomize`` load from it."""
    archive = layout["tmp"] / "extra.zip"
    _capture(layout, archive)
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("sitecustomize.py", MARK)
    _clean(layout)
    assert not archive.exists()


# --- V8's narrowed outside-.pth cases (c911a719), as controls ---


@pytest.mark.parametrize(
    "link", [True, False], ids=["hook-linked-home", "agent-folder"]
)
def test_a_recorded_outside_pth_in_a_linked_venv_runs_no_agent_hook(layout, link):
    """V8's tests/test_workspace_freeze.py::test_a_recorded_pth_adding_a_folder_
    outside_the_workspace_is_removed (V8-MERGE-3), on the shared rule: the
    linked venv's recorded path-only .pth names a folder outside, either an
    image folder whose sitecustomize.py links into the agent's home or the
    agent's own folder. The folder is on the recorded interpreter's search
    path, so it is checked like any other: the changed link or the new
    sitecustomize.py goes, and no agent hook runs. Decided differently from
    V8: the recorded .pth itself stays (V8 removes it, because its walk does
    not check the folder)."""
    home = layout["tmp"] / "home" / "agent"
    home.mkdir(parents=True)
    (home / "startup.py").write_text("# benign at record time\n")
    if link:
        outside = layout["tmp"] / "opt" / "sitehooks"
        outside.mkdir(parents=True)
        (outside / "sitecustomize.py").symlink_to(home / "startup.py")
    else:
        outside = home
    site = layout["tmp"] / layout["site"]
    external = site / "sitehooks.pth"
    external.write_text(f"{outside}\n")
    (layout["app"] / "src").mkdir()
    editable = site / "__editable__.proj.pth"
    editable.write_text(f"{layout['app'] / 'src'}\n")
    assert os.path.islink(layout["py"])  # a linked venv
    _capture(layout)
    marker = layout["tmp"] / "hook-ran"
    hook = "startup.py" if link else "sitecustomize.py"
    (home / hook).write_text(f"open({str(marker)!r}, 'w').write('ran')\n")
    # Control: the venv's Python runs the agent's hook through that .pth.
    subprocess.run([str(layout["py"]), "-c", "pass"], check=True, cwd="/")
    assert marker.exists()
    marker.unlink()

    _clean(layout)

    assert editable.read_text() == f"{layout['app'] / 'src'}\n"
    assert external.read_text() == f"{outside}\n"
    subprocess.run([str(layout["py"]), "-c", "pass"], check=True, cwd="/")
    assert not marker.exists()


def test_a_copies_venv_keeps_its_recorded_outside_pth(tmp_path):
    """V8's tests/test_workspace_freeze.py::test_a_real_workspace_interpreter_
    keeps_its_recorded_outside_pth (V8-NARROW-1), on the shared rule: a
    ``--copies`` venv's recorded editable .pth naming /opt/<lib>/src stays
    and the project imports; a new outside .pth goes. Same as V8."""
    venv = tmp_path / "app" / ".venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", "--copies", str(venv)],
        check=True,
    )
    py = venv / "bin" / "python"
    assert not py.is_symlink()
    site = next((venv / "lib").glob("python*/site-packages"))
    project = tmp_path / "opt" / "honestlib" / "src"
    project.mkdir(parents=True)
    (project / "honestlib.py").write_text("ANSWER = 42\n")
    editable = site / "__editable__.honestlib-1.0.pth"
    editable.write_text(f"{project}\n")
    app = tmp_path / "app"
    layout = {"py": py, "site": str(site), "app": app, "tmp": tmp_path}
    _capture(layout)
    planted = site / "zz_agent.pth"
    planted.write_text(f"{tmp_path / 'home' / 'agent'}\n")

    _clean(layout)

    assert editable.read_text() == f"{project}\n"
    assert not planted.exists()
    done = subprocess.run(
        [str(py), "-c", "import honestlib; print(honestlib.ANSWER)"],
        capture_output=True,
        text=True,
        cwd="/",
    )
    assert done.stdout.strip() == "42", done.stderr


# --- round 2: folders only test.sh's search-path inputs add (PT-R16-B-1) ---


def test_a_pythonpath_folder_loses_the_agents_hook_and_keeps_its_data(layout):
    """PT-R16-B-1, fails on 91c603c6: the kit's record runs ``-I``, so the
    project folder on test.sh's PYTHONPATH (``ENV PYTHONPATH=/app``) was in
    no record and no clean-up folder, and a sitecustomize there ran as
    root. It is recorded from the inputs now: the agent's hook goes, the
    image's own sitecustomize.py and the honest model.pth stay."""
    (layout["app"] / "sitecustomize.py").write_text(MARK)
    _capture(layout)
    package = layout["app"] / "usercustomize"
    package.mkdir()
    (package / "__init__.py").write_text(MARK)
    (layout["app"] / "model.pth").write_bytes(b"PK\x03\x04 dummy weights")
    _clean(layout)
    assert not package.exists()
    assert (layout["app"] / "sitecustomize.py").read_text() == MARK
    assert (layout["app"] / "model.pth").read_bytes() == b"PK\x03\x04 dummy weights"


def test_a_relative_pythonpath_entry_is_read_against_the_working_folder(layout):
    """PT-R16-B-1: an empty or relative PYTHONPATH entry names a folder under
    the original working folder, which the clean-up visits too (unrecorded
    here: every hook form goes, a .pth there is data)."""
    work = layout["tmp"] / "work"
    (work / "rel").mkdir(parents=True)
    _capture(layout)
    layout["pythonpath"] = [layout["app"], "", "rel"]
    layout["cwd"] = str(work)
    for folder in (work, work / "rel"):
        (folder / "sitecustomize.py").write_text(MARK)
        (folder / "data.pth").write_text("dummy data\n")
    _clean(layout)
    for folder in (work, work / "rel"):
        assert not (folder / "sitecustomize.py").exists()
        assert (folder / "data.pth").read_text() == "dummy data\n"


def test_the_user_base_site_folder_is_a_site_folder(layout):
    """PT-R16-B-1: PYTHONUSERBASE names a user site whose .pth files run, so
    a new .pth there goes and the one recorded before the agent stays."""
    user_site = layout["tmp"] / "ub" / "lib" / "python3.12" / "site-packages"
    user_site.mkdir(parents=True)
    (user_site / "honest.pth").write_text("/opt/dummy\n")
    layout["userbase"] = str(layout["tmp"] / "ub")
    _capture(layout)
    (user_site / "zz-agent.pth").write_text("import os  # dummy marker\n")
    _clean(layout)
    assert (user_site / "honest.pth").read_text() == "/opt/dummy\n"
    assert not (user_site / "zz-agent.pth").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the folder's mode")
def test_a_hook_that_cannot_be_removed_fails_the_clean_up(layout):
    """PT-R16-B-4, fails on 91c603c6: a removal that failed (EACCES here,
    as EBUSY or EPERM would) kept the hook and the step still succeeded.
    Now the clean-up exits non-zero, and the step's shell text fails on it
    (127, no Python at all, still passes)."""
    _capture(layout)
    hook = layout["app"] / "sitecustomize.py"
    hook.write_text(MARK)
    layout["app"].chmod(0o555)
    try:
        done = _run_clean(layout)
    finally:
        layout["app"].chmod(0o755)
    assert done.returncode == 3, (done.returncode, done.stderr)
    assert "could not remove 1 Python start-up hook" in done.stderr
    cmd = lockdown._build_cleanup_cmd({"cleanup_conftests": False})
    assert cmd.endswith(
        '; bf_rc=$?; [ "$bf_rc" = 0 ] || { [ "$bf_rc" = 127 ] && '
        f'echo "{lockdown.NO_PROTECTED_PYTHON_SKIP}" >&2; }} || '
        '{ echo "benchflow: removing Python start-up hooks failed (rc=$bf_rc)" '
        ">&2; false; }"
    )


# --- round 3: review of round 2 (PT-R16-R2-1, R2-3, R2-4) ---------------------


def test_a_binary_pth_in_a_site_folder_is_read_past(layout):
    """PT-R16-R2-1, fails on 16c906df: a model checkpoint saved as
    ``weights.pth`` beside the packages holds NUL bytes; its "path lines"
    reached os.open, whose ValueError stopped the kit's capture (so the task
    never ran) and the clean-up. site itself passes over such a line. Both
    now finish, and the file stays."""
    weights = layout["tmp"] / layout["site"] / "weights.pth"
    weights.write_bytes(b"PK\x03\x04\x00\x00dummy\n/opt/\x00weights\nmore\x00\n")
    _capture(layout)
    _clean(layout)
    assert (
        weights.read_bytes() == b"PK\x03\x04\x00\x00dummy\n/opt/\x00weights\nmore\x00\n"
    )


def test_a_new_pth_that_goes_widens_nothing(layout):
    """PT-R16-R2-3, fails on 16c906df: a new .pth (an editable install of
    the agent's project) named /app/src, so the clean-up removed the
    project's own sitecustomize.py there although the .pth itself goes and
    test.sh's Python never searches /app/src. Folders are now read only
    from the .pth files the clean-up keeps: the image's recorded one still
    brings its folder in, and the agent's hook there goes."""
    project = layout["tmp"] / "proj" / "src"
    project.mkdir(parents=True)
    image_lib = layout["tmp"] / "opt-lib"
    image_lib.mkdir()
    site = layout["tmp"] / layout["site"]
    (site / "image-lib.pth").write_text(f"{image_lib}\n")
    _capture(layout)
    (project / "sitecustomize.py").write_text(MARK)
    (site / "__editable__.proj.pth").write_text(f"{project}\n")
    (image_lib / "sitecustomize.py").write_text(MARK)
    _clean(layout)
    assert not (site / "__editable__.proj.pth").exists()
    assert (project / "sitecustomize.py").read_text() == MARK
    assert (site / "image-lib.pth").read_text() == f"{image_lib}\n"
    assert not (image_lib / "sitecustomize.py").exists()


def test_pythonhomes_exec_prefix_site_folder_is_a_site_folder(layout):
    """PT-R16-R2-4, fails on 16c906df: PYTHONHOME is prefix:exec_prefix
    and site runs the .pth files under both (site.PREFIXES); only the
    prefix was read, so a new .pth under exec_prefix stayed."""
    prefix = layout["tmp"] / "py"
    exec_site = layout["tmp"] / "py-x86_64" / "lib" / "python3.12" / "site-packages"
    exec_site.mkdir(parents=True)
    (exec_site / "honest.pth").write_text("/opt/dummy\n")
    layout["home"] = f"{prefix}:{layout['tmp'] / 'py-x86_64'}"
    _capture(layout)
    (exec_site / "zz-agent.pth").write_text("import os  # dummy marker\n")
    _clean(layout)
    assert (exec_site / "honest.pth").read_text() == "/opt/dummy\n"
    assert not (exec_site / "zz-agent.pth").exists()
