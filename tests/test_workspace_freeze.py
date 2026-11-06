"""Behavioral regressions for the fix on py8/freeze-fixes (base 49ae4837).

Run the actual purge payloads, including U5a's one-script route, on disposable
trees. No sandbox paths are touched: only the baseline literal is relocated.
"""

from __future__ import annotations

import functools
import marshal
import os
import py_compile
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import SYSTEM_PATH as _ROUTE_PATH
from benchflow.sandbox._hardening_python import TRUSTED_SYSTEM_PATH as _SYSTEM_PATH


def _as_judged(command: str, env: dict[str, str]) -> str:
    """*command* as the root route runs it, naming the interpreter the host's
    judge accepted: the python3 the step's PATH finds, as is
    outside the workspace (a venv elsewhere stays that venv), and by its real
    file inside it (the judge rejects a workspace venv link and takes its
    target). These tests name every workspace folder "workspace..."."""
    found = shutil.which("python3", path=env["PATH"]) or ""
    inside = any(part.startswith("workspace") for part in Path(found).parts)
    judged = os.path.realpath(found) if found and inside else found
    return f"__bf_root_python={shlex.quote(judged)}; " + command


@pytest.fixture(params=["dash", "bash"])
def shell(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[str, dict[str, str]]:
    executable = shutil.which(request.param)
    if executable is None:
        pytest.skip(f"{request.param} is not installed")
    # The cache command itself invokes sh. Test its interpreter as well as the
    # outer shell, without changing the machine's /bin/sh.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sh").symlink_to(executable)
    return executable, {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


@pytest.fixture(params=[False, True], ids=["one-exec-per-step", "one-script"])
def run_purge(tmp_path: Path, shell, request: pytest.FixtureRequest):
    baseline = tmp_path / "baseline"
    baseline.mkdir(mode=0o700)
    parent = tmp_path / "sticky"
    parent.mkdir(mode=0o700)
    parent.chmod(0o1700)
    executable, env = shell

    def run(command: str) -> None:
        command = command.replace("/testbed_verify", str(baseline))
        if request.param:
            step = lockdown._PrepStep("purge", "freeze_workspace", command, 10)
            command = lockdown._prep_script([step], str(parent / "prep"))
            # The runner resets PATH for cleanup tools. Point only its nested
            # sh at the chosen interpreter too, so both sh variants are real.
            command = command.replace(
                f"__bf_path={shlex.quote(_ROUTE_PATH)}",
                f"__bf_path={shlex.quote(env['PATH'])}",
            )
        result = subprocess.run(
            [executable, "-c", _as_judged(command, env)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        if request.param:
            assert "BENCHFLOW-PREP-END" in result.stdout, result.stdout

    return baseline, run


@pytest.fixture
def image_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Make a stand-in for an image file outside the workspace that a kept
    link may resolve to (V8-R1-A-1, round 1 at b449a20b): the runtime
    prefixes are unblocked, so the test's folder under /tmp stands for
    /usr/local/bin, and the modes are set, since the umask may make new
    files and folders group-writable."""
    from benchflow.sandbox import _hardening_python

    monkeypatch.setattr(_hardening_python, "RUNTIME_PATH_PREFIXES", ())

    def make(path: Path, text: str = "trusted interpreter") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o755)
        for parent in path.parents:
            if parent == tmp_path or tmp_path not in parent.parents:
                break
            parent.chmod(0o755)
        return path

    return make


@pytest.mark.parametrize(
    ("case", "kept"),
    [
        ("trusted", True),
        ("sticky-above", True),
        ("runtime-prefix", False),
        ("writable-file", False),
        ("writable-directory", False),
        ("writable-prefix", False),
        ("writable-above", False),
        ("writable-home", False),
    ],
)
def test_recorded_link_to_an_interpreter_the_agent_can_write_is_removed(
    tmp_path: Path,
    run_purge,
    image_python,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    kept: bool,
):
    """Guards V8-R1-A-1 (round 1 at b449a20b): unchanged is not trusted.

    The image's venv links to an interpreter in a tree the agent can write (a
    Python under /home/<sandbox_user>, which setup_sandbox_user chowns to the
    agent, or one in a 0777 folder). Its text, its target and pyvenv.cfg
    match the record, but a sitecustomize the agent writes into that
    interpreter's standard library would run in every verifier venv Python,
    so the link goes, as on the base. Another user's tree is the container
    repro's case; here the test's own user stands for root.
    """
    from benchflow.sandbox import _hardening_python

    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    prefix = tmp_path / "home-agent" / "py"
    interpreter = image_python(prefix / "bin" / "python3.12")
    home = interpreter.parent
    if case == "runtime-prefix":
        blocked = (str(tmp_path / "home-agent"),)
        monkeypatch.setattr(_hardening_python, "RUNTIME_PATH_PREFIXES", blocked)
    elif case == "writable-file":
        interpreter.chmod(0o775)
    elif case == "writable-directory":
        interpreter.parent.chmod(0o777)
    elif case == "writable-prefix":
        prefix.chmod(0o775)
    elif case in ("writable-above", "sticky-above"):
        prefix.parent.chmod(0o777 if case == "writable-above" else 0o1777)
    elif case == "writable-home":
        # The link's target passes; the standard library pyvenv.cfg picks
        # does not.
        home = tmp_path / "agent-home" / "bin"
        home.mkdir(parents=True)
        home.chmod(0o777)
    (venv_bin / "python3.12").symlink_to(interpreter)
    (venv_bin / "python").symlink_to("python3.12")
    (venv / "pyvenv.cfg").write_text(f"home = {home}\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)

    run(lockdown._purge_external_symlinks_cmd(str(workspace)))

    assert (venv_bin / "python3.12").is_symlink() == kept
    # The in-tree link is judged as before, by where it leads when the purge
    # reaches it: through a kept python3.12 to the trusted interpreter, kept;
    # once python3.12 is gone, nowhere outside, kept dangling (V8-R1-A-2);
    # while python3.12 is still there, to the interpreter, removed. Which of
    # the last two happens is the order the filesystem lists the folder in
    # (ext4 and tmpfs list python first), and neither leaves a way to
    # the interpreter the agent can write.
    link = venv_bin / "python"
    if kept:
        assert link.is_symlink()
    else:
        assert not link.exists()
        assert not link.is_symlink() or os.readlink(link) == "python3.12"


@pytest.mark.parametrize("retarget", ["external", "internal", "loop"])
def test_image_venv_links_survive_and_changed_links_get_the_realpath_check(
    tmp_path: Path, run_purge, image_python, retarget: str
):
    """Guards the fix against deleting the image venv's external interpreter.

    A recorded link the agent retargets is judged like a new link: kept when it
    stays in the workspace (V8-R1-B-1), removed when it escapes or loops.
    The loop case guards V8-RC1-A-2 at 1c28867b on Python 3.12 and 3.13.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace with 'quotes'\nand [glob]"
    venv_bin = workspace / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    interpreter = image_python(tmp_path / "image-python")
    (venv_bin / "python").symlink_to(interpreter)
    (venv_bin / "python3").symlink_to("python")
    (workspace / "retargeted").symlink_to(interpreter)
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)

    evil = tmp_path / "agent-module"
    evil.write_text("untrusted")
    (workspace / "planted").symlink_to(evil)
    # Same target as the image is not enough: the path must match too.
    (workspace / "copied-python").symlink_to(interpreter)
    (workspace / "retargeted").unlink()
    target = {"external": evil, "internal": "source", "loop": "retargeted"}[retarget]
    (workspace / "retargeted").symlink_to(target)
    (workspace / "source").write_text("answer")
    (workspace / "in-tree").symlink_to("source")

    run(lockdown._purge_external_symlinks_cmd(str(workspace)))

    assert (venv_bin / "python").is_symlink()
    assert os.readlink(venv_bin / "python") == str(interpreter)
    assert (venv_bin / "python3").resolve() == interpreter
    for name in ("planted", "copied-python"):
        assert not (workspace / name).is_symlink(), name
    assert (workspace / "in-tree").read_text() == "answer"
    if retarget == "internal":
        assert os.readlink(workspace / "retargeted") == "source"
        assert (workspace / "retargeted").read_text() == "answer"
    else:
        assert not (workspace / "retargeted").is_symlink()


@pytest.mark.parametrize("tamper", ["changed_up", "planted_beside"])
def test_kept_interpreter_link_drops_when_pyvenv_cfg_is_tampered(
    tmp_path: Path, run_purge, image_python, tamper: str
):
    """Guards V8-R1-A-1: pyvenv.cfg's home can move the kept interpreter.

    The agent rewrites pyvenv.cfg (or plants one beside the interpreter) so the
    venv python reads an in-workspace standard library. The exemption must then
    drop, so the escaping interpreter link is removed as the base did.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace with 'quotes'\nand [glob]"
    venv = workspace / ".venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    interpreter = image_python(tmp_path / "image-python")
    (venv_bin / "python").symlink_to(interpreter)
    (venv / "pyvenv.cfg").write_text(f"home = {interpreter.parent}\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)

    fake_bin = workspace / "fake" / "bin"
    fake_bin.mkdir(parents=True)
    if tamper == "changed_up":
        (venv / "pyvenv.cfg").write_text(f"home = {fake_bin}\n")
    else:
        (venv_bin / "pyvenv.cfg").write_text(f"home = {fake_bin}\n")

    run(lockdown._purge_external_symlinks_cmd(str(workspace)))

    assert not (venv_bin / "python").is_symlink()


def test_kept_interpreter_link_survives_an_unchanged_pyvenv_cfg(
    tmp_path: Path, run_purge, image_python
):
    """Guards V8-R1-A-1's fix against removing an honest venv interpreter."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    interpreter = image_python(tmp_path / "image-python")
    (venv_bin / "python").symlink_to(interpreter)
    (venv / "pyvenv.cfg").write_text(f"home = {interpreter.parent}\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)

    run(lockdown._purge_external_symlinks_cmd(str(workspace)))

    assert (venv_bin / "python").is_symlink()
    assert os.readlink(venv_bin / "python") == str(interpreter)


@pytest.mark.parametrize("recorded_case", [True, False])
def test_dangling_in_tree_links_are_kept(
    tmp_path: Path, run_purge, recorded_case: bool
):
    """Guards V8-R1-A-2: a link to a missing in-tree target is kept.

    A broken-link fixture, or a link to a file the verifier's build creates,
    can only resolve inside the workspace, so removing it adds no safety and
    breaks a correct answer. A dangling link that escapes is still removed.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    (workspace / "fixtures").mkdir(parents=True)
    (workspace / "fixtures" / "broken_link").symlink_to("missing-target.txt")
    (workspace / "compile_commands.json").symlink_to("build/compile_commands.json")
    if recorded_case:
        shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
        baseline.chmod(0o700)
    (workspace / "escape").symlink_to(tmp_path / "missing-outside")

    run(lockdown._purge_external_symlinks_cmd(str(workspace)))

    assert (workspace / "fixtures" / "broken_link").is_symlink()
    assert (workspace / "compile_commands.json").is_symlink()
    assert not (workspace / "escape").is_symlink()


def test_recorded_pth_goes_when_the_module_it_imports_is_edited(
    tmp_path: Path, run_purge
):
    """Guards V8-R1-B-1: a recorded import .pth is only as trusted as its module.

    uv, virtualenv and poetry seed _virtualenv.pth plus _virtualenv.py. An agent
    that only appends to _virtualenv.py would get root code at venv startup, so
    the recorded .pth is removed unless the module it imports is recorded too.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    site = workspace / ".venv" / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (site / "_virtualenv.pth").write_text("import _virtualenv\n")
    (site / "_virtualenv.py").write_text("# image shim\n")
    (site / "ok.pth").write_text("import okmod\n")
    (site / "okmod.py").write_text("# image module\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    with (site / "_virtualenv.py").open("a") as handle:
        handle.write("PAYLOAD = 1\n")

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert not (site / "_virtualenv.pth").exists()
    # The library module stays (it is agent code, like workspace source); only
    # the .pth that would import it at startup goes.
    assert (site / "_virtualenv.py").exists()
    # A recorded .pth whose imported module is unchanged is still kept.
    assert (site / "ok.pth").read_text() == "import okmod\n"
    assert (site / "okmod.py").read_text() == "# image module\n"


@pytest.mark.parametrize(
    "attack", ["untouched", "deleted_and_shadowed", "dunder_import"]
)
def test_recorded_import_pth_cannot_reach_agent_code(
    tmp_path: Path, run_purge, attack: str
):
    """Guards V8-R1-B-1 beyond a plain `import name` line.

    A recorded .pth may load its module through `__import__('name')` (as
    setuptools' distutils-precedence.pth does), and an agent may delete the
    recorded module and put its own copy on a directory an earlier path-only
    .pth adds. Either way the venv's python would run agent code at startup, so
    the recorded .pth must go; untouched, it stays.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    (site / "_virtualenv.pth").write_text("import _virtualenv\n")
    (site / "_virtualenv.py").write_text("# image shim\n")
    hack = site / "_hookmod"
    hack.mkdir()
    (hack / "__init__.py").write_text("def go():\n    pass\n")
    (site / "precedence.pth").write_text("import os; __import__('_hookmod').go()\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    marker = workspace / "agent-ran"
    payload = f"import pathlib; pathlib.Path({str(marker)!r}).write_text('ran')\n"
    if attack == "deleted_and_shadowed":
        (site / "_virtualenv.py").unlink()
        shadow = workspace / "shadow"
        shadow.mkdir()
        (shadow / "_virtualenv.py").write_text(payload)
        (site / "00_agent.pth").write_text(f"{shadow}\n")
    elif attack == "dunder_import":
        with (hack / "__init__.py").open("a") as handle:
            handle.write(payload)

    _run_freeze_purges(workspace, run, hooks_only=True)

    verifier = subprocess.run(
        [str(venv / "bin" / "python"), "-c", "pass"],
        env={**os.environ, **lockdown.VERIFIER_ENV},
        capture_output=True,
        text=True,
    )
    assert verifier.returncode == 0, verifier.stderr
    assert not marker.exists()
    if attack == "untouched":
        assert (site / "_virtualenv.pth").read_text() == "import _virtualenv\n"
        assert (site / "precedence.pth").exists()
    elif attack == "deleted_and_shadowed":
        assert not (site / "_virtualenv.pth").exists()
        assert (site / "precedence.pth").exists()
    else:
        assert not (site / "precedence.pth").exists()
        assert (site / "_virtualenv.pth").exists()


def test_recorded_import_pth_survives_an_agent_phase_pycache(tmp_path: Path, run_purge):
    """Guards V8-R2-A-3: a fresh __pycache__ does not cost a recorded .pth.

    An image built with uv sync (no bytecode) records setuptools'
    distutils-precedence.pth and _distutils_hack/ without __pycache__. Any agent
    run of the venv python writes _distutils_hack/__pycache__; the freeze removes
    that cache, and must do so before judging the package tree, or the shim's
    .pth goes and `import distutils` fails in the verifier on Python 3.12+.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    pth = site / "distutils-precedence.pth"
    pth.write_text(
        "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; "
        "enabled = os.environ.get(var, 'local') == 'local'; "
        "enabled and __import__('_distutils_hack').add_shim(); \n"
    )
    (site / "_distutils_hack").mkdir()
    (site / "_distutils_hack" / "__init__.py").write_text("def add_shim():\n    pass\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    subprocess.run([str(venv / "bin" / "python"), "-c", "pass"], env=env, check=True)
    assert (site / "_distutils_hack" / "__pycache__").is_dir()

    _run_freeze_purges(workspace, run)

    assert pth.exists()
    assert not (site / "_distutils_hack" / "__pycache__").exists()
    verifier = subprocess.run(
        [
            str(venv / "bin" / "python"),
            "-c",
            "import sys; print('_distutils_hack' in sys.modules)",
        ],
        env={**os.environ, **lockdown.VERIFIER_ENV},
        capture_output=True,
        text=True,
    )
    assert verifier.returncode == 0, verifier.stderr
    assert verifier.stdout.strip() == "True"


def test_recorded_pth_with_a_bom_has_the_hooks_purged_where_old_pythons_look(
    tmp_path: Path, run_purge
):
    """Guards V8-R3-B-1 for a recorded .pth: Python 3.11 and older read its
    BOM into the first line, so site adds <site>/\ufeff<line>. The agent made
    that an in-tree link to a folder holding a sitecustomize, which goes."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    site = workspace / ".venv" / "lib" / "python3.11" / "site-packages"
    (site / "pkg").mkdir(parents=True)
    (site / "image.pth").write_bytes(b"\xef\xbb\xbfpkg\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    hooks = workspace / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text("raise RuntimeError('agent')\n")
    (site / "\ufeffpkg").symlink_to(hooks, target_is_directory=True)

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert (site / "image.pth").read_bytes() == b"\xef\xbb\xbfpkg\n"
    assert not (hooks / "sitecustomize.py").exists()


@pytest.mark.parametrize("linked", [False, True], ids=["file", "link"])
def test_recorded_bom_pth_cannot_add_an_unrecorded_startup_archive(
    tmp_path: Path, run_purge, linked: bool
):
    """Guards V8-CONF-2 at 40618b7b: the BOM-retaining path needs file custody.

    Python 3.11 and older add the BOM-prefixed filename to sys.path. A ZIP
    there can load sitecustomize just as a directory can, including via a link.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    site = workspace / ".venv" / "lib" / "python3.11" / "site-packages"
    (site / "pkg").mkdir(parents=True)
    pth = site / "image.pth"
    pth.write_bytes(b"\xef\xbb\xbfpkg\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    added = site / "\ufeffpkg"
    archive = workspace / "hooks.zip" if linked else added
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("sitecustomize.py", "raise RuntimeError('agent')\n")
    if linked:
        added.symlink_to(archive)

    _run_freeze_purges(workspace, run)

    assert not pth.exists()
    assert zipfile.is_zipfile(archive)


@pytest.mark.parametrize("cache_target", ["directory", "file", "missing"])
@pytest.mark.parametrize("hooks_only", [False, True], ids=["all-purges", "hooks-only"])
def test_recorded_import_pth_cannot_trust_a_symlinked_cache(
    tmp_path: Path, run_purge, cache_target: str, hooks_only: bool
):
    """Guards V8-CONF-1 at 40618b7b: ignored cache names cannot hide bytecode.

    The source and .pth stay recorded. An in-tree cache link sends Python -I
    to timestamp-valid agent bytecode despite PYTHONPYCACHEPREFIX. Non-directory
    and dangling cache links must be removed as well.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    pth = site / "image.pth"
    pth.write_text("import _distutils_hack\n")
    package = site / "_distutils_hack"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("IMAGE = True\n")
    pyc = Path(py_compile.compile(str(source), doraise=True))
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    cache = package / "__pycache__"
    marker = tmp_path / "bytecode-ran"
    poison = workspace / "poison"
    if cache_target == "directory":
        poison.mkdir()
        code = compile(
            f"import pathlib; pathlib.Path({str(marker)!r}).write_text('ran')\n",
            str(source),
            "exec",
        )
        (poison / pyc.name).write_bytes(pyc.read_bytes()[:16] + marshal.dumps(code))
    elif cache_target == "file":
        poison.write_bytes(b"agent")
    shutil.rmtree(cache)
    cache.symlink_to(poison, target_is_directory=cache_target == "directory")

    _run_freeze_purges(workspace, run, hooks_only=hooks_only)

    verifier = subprocess.run(
        [str(venv / "bin" / "python"), "-I", "-c", "pass"],
        env={**os.environ, **lockdown.VERIFIER_ENV},
        capture_output=True,
        text=True,
    )
    assert verifier.returncode == 0, verifier.stderr
    assert not marker.exists()
    assert cache.is_symlink() == hooks_only
    assert pth.exists() != hooks_only
    assert source.read_text() == "IMAGE = True\n"


@pytest.mark.parametrize("spelling", ["caf\u00e9", "\u6a21\u5757", "line\u0085break"])
def test_unicode_editable_pth_keeps_the_answer_and_purges_startup_hooks(
    tmp_path: Path, run_purge, spelling: str
):
    """Guards V8-CONF-3 at 40618b7b: Unicode paths are valid editable installs.

    Check both the modern splitlines and older newline-only interpretations,
    and import the answer with the real venv interpreter after hardening.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    package = workspace / spelling / "src"
    package.mkdir(parents=True)
    (package / "answer.py").write_text("VALUE = 7\n")
    marker = tmp_path / "unicode-hook-ran"
    (package / "sitecustomize.py").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('ran')\n"
    )
    pth = site / "editable.pth"
    pth.write_text(f"{package}\n", encoding="utf-8")

    _run_freeze_purges(workspace, run)

    assert pth.read_text(encoding="utf-8") == f"{package}\n"
    assert not (package / "sitecustomize.py").exists()
    # Python 3.13 splits U+0085 as a newline, so this spelling is not a
    # usable editable path there; it must still have its hooks sanitized.
    if spelling != "line\u0085break":
        verifier = subprocess.run(
            [
                str(venv / "bin" / "python"),
                "-I",
                "-c",
                "import answer; print(answer.VALUE)",
            ],
            env={**os.environ, **lockdown.VERIFIER_ENV},
            capture_output=True,
            text=True,
        )
        assert verifier.returncode == 0, verifier.stderr
        assert verifier.stdout.strip() == "7"
    assert not marker.exists()


def test_new_pth_that_is_not_utf8_is_removed(tmp_path: Path, run_purge):
    """Guards V8-CONF-3's limit: only a UTF-8 .pth gets the Unicode path rule.

    No installer writes another encoding, and an older site would read these
    bytes by the locale, so the .pth goes rather than being judged.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    site = workspace / ".venv" / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    (workspace / "src").mkdir()
    (site / "agent.pth").write_bytes(os.fsencode(workspace / "src") + b"-\xe9\n")

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert not (site / "agent.pth").exists()


def test_new_pth_with_a_bom_before_its_import_line_is_removed(
    tmp_path: Path, run_purge
):
    """Guards V8-R1-B-1: Python 3.13's site drops a UTF-8 BOM before the
    import check, so a BOM-prefixed import line runs code and is not path-only.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    site = workspace / ".venv" / "lib" / "python3.13" / "site-packages"
    site.mkdir(parents=True)
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    (site / "agent.pth").write_bytes(b"\xef\xbb\xbfimport os; os.getpid()\n")

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert not (site / "agent.pth").exists()


def test_path_only_editable_install_pth_is_kept(tmp_path: Path, run_purge):
    """Guards V8-R1-B-4: a path-only .pth a correct answer installs is kept.

    pip/uv install -e writes a path-only .pth into the venv. It runs no code, so
    it stays and `import <pkg>` from the venv still works; the sitecustomize the
    directory it adds would load is purged instead.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    site = workspace / ".venv" / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    pkg_src = workspace / "src"
    pkg_src.mkdir()
    (pkg_src / "mylib.py").write_text("VALUE = 7\n")
    (site / "__editable__.mylib.pth").write_text(f"{pkg_src}\n")
    (pkg_src / "sitecustomize.py").write_text("raise RuntimeError('agent')\n")

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert (site / "__editable__.mylib.pth").read_text() == f"{pkg_src}\n"
    assert (pkg_src / "mylib.py").read_text() == "VALUE = 7\n"
    assert not (pkg_src / "sitecustomize.py").exists()


@pytest.mark.parametrize(
    "added", ["outside", "leading_space", "zip", "nul_byte", "bom", "non_ascii"]
)
def test_new_path_only_pth_cannot_add_a_startup_hook(
    tmp_path: Path, run_purge, added: str
):
    """Guards V8-R2-A-1: a new path-only .pth is kept only when what site adds
    stays where the freeze can see it.

    site adds abspath(join(site_dir, line.rstrip())). A directory the agent can
    write outside the workspace, a leading-space line whose '..' climbs out of
    the site tree, or a zip on sys.path would each carry a sitecustomize into
    the verifier's venv python; a line Python skips must not stop the purge.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    marker = tmp_path / "hook-ran"
    hook = f"import pathlib; pathlib.Path({str(marker)!r}).write_text('ran')\n"
    if added == "outside":
        hooks = tmp_path / "agent-home" / "hooks"
        hooks.mkdir(parents=True)
        (hooks / "sitecustomize.py").write_text(hook)
        line = f"{hooks}\n"
    elif added == "leading_space":
        hooks = workspace / "hooks"
        hooks.mkdir()
        (hooks / "sitecustomize.py").write_text(hook)
        line = f" /../{os.path.relpath(hooks, site)}\n"
    elif added == "zip":
        archive = workspace / "hooks.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("sitecustomize.py", hook)
        line = f"{archive}\n"
    elif added == "nul_byte":
        # Python's site skips a path with a NUL byte and goes on to the next
        # line; the purge must not stop there either.
        hooks = workspace / "hooks"
        hooks.mkdir()
        (hooks / "sitecustomize.py").write_text(hook)
        line = f"bad\x00path\n{hooks}\n"
    elif added == "bom":
        # V8-R3-B-1: Python 3.11 and older keep the BOM, so site adds
        # <site>/\ufeffzzz, an in-tree link to a folder outside every site
        # tree; 3.12+ and the purge read <site>/zzz, which is missing. The
        # hook runs only on 3.11 and older, so the test also checks the .pth
        # went, which it can on any version.
        hooks = workspace / "hooks"
        hooks.mkdir()
        (hooks / "sitecustomize.py").write_text(hook)
        (site / "\ufeffzzz").symlink_to(hooks, target_is_directory=True)
        line = "\ufeffzzz\n"
    else:
        # Python 3.12 and older decode a .pth by the locale, so a non-ASCII
        # path can name another folder than the one checked (V8-R3-A-2): the
        # purge checks every spelling, and the hook must not run.
        hooks = workspace / "hooks-\u00e9"
        hooks.mkdir()
        (hooks / "sitecustomize.py").write_text(hook)
        line = f"{hooks}\n"
    (site / "00_agent.pth").write_text(line, encoding="utf-8")

    _run_freeze_purges(workspace, run, hooks_only=True)

    verifier = subprocess.run(
        [str(venv / "bin" / "python"), "-c", "pass"],
        env={**os.environ, **lockdown.VERIFIER_ENV},
        capture_output=True,
        text=True,
    )
    assert verifier.returncode == 0, verifier.stderr
    assert not marker.exists()
    # A non-ASCII UTF-8 path is kept with its hooks purged, so a Unicode
    # editable install still imports (V8-CONF-3); every other one goes.
    assert (site / "00_agent.pth").exists() == (added == "non_ascii")


def test_recorded_link_cannot_escape_through_a_replaced_component(
    tmp_path: Path, run_purge
):
    """Guards V8-RC1-A-1 at 1c28867b: unchanged text is not a target check."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    (workspace / "a" / "b").mkdir(parents=True)
    (workspace / "a" / "c").mkdir()
    (workspace / "x").write_text("image")
    link = workspace / "a" / "b" / "link"
    link.symlink_to("../c/../../x")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    (workspace / "a" / "c").rmdir()
    (workspace / "a" / "c").symlink_to(".")
    (tmp_path / "x").write_text("agent")
    assert link.read_text() == "agent"

    run(lockdown._purge_external_symlinks_cmd(str(workspace)))

    assert not link.is_symlink()
    assert (workspace / "a" / "c").is_symlink()
    assert (tmp_path / "x").read_text() == "agent"


def test_baseline_directory_links_do_not_authorize_unrecorded_paths(
    tmp_path: Path, run_purge
):
    """Guards the fix against following a baseline directory link outside it."""
    baseline, run = run_purge
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "planted").symlink_to(tmp_path / "evil")
    (baseline / "directory").symlink_to(outside, target_is_directory=True)
    workspace = tmp_path / "workspace"
    (workspace / "directory").mkdir(parents=True)
    (workspace / "directory" / "planted").symlink_to(tmp_path / "evil")
    run(lockdown._purge_external_symlinks_cmd(str(workspace)))
    assert not (workspace / "directory" / "planted").is_symlink()


@pytest.mark.parametrize("untrusted", ["missing", "unlocked", "symlink"])
def test_untrusted_baseline_cannot_exempt_an_external_link(
    tmp_path: Path, run_purge, image_python, untrusted: str
):
    """Guards the external-link exception against missing or agent-replaceable records."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image_python(tmp_path / "outside")
    (workspace / "link").symlink_to(tmp_path / "outside")
    (baseline / "link").symlink_to(tmp_path / "outside")
    if untrusted == "missing":
        shutil.rmtree(baseline)
    elif untrusted == "unlocked":
        baseline.chmod(0o777)
    else:
        real = tmp_path / "untrusted"
        baseline.rename(real)
        baseline.symlink_to(real, target_is_directory=True)
    run(lockdown._purge_external_symlinks_cmd(str(workspace)))
    assert not (workspace / "link").is_symlink()


def test_pycache_purge_keeps_image_cache_and_removes_planted_cache(
    tmp_path: Path, run_purge
):
    """Guards the fix against dash rejecting the old read -d cache loop."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace with 'quotes'\nand [glob]"
    image_cache = workspace / "package" / "__pycache__"
    image_cache.mkdir(parents=True)
    (image_cache / "original.pyc").write_bytes(b"image")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    planted = workspace / "odd '\n[path]" / "__pycache__"
    planted.mkdir(parents=True)
    (planted / "evil.pyc").write_bytes(b"agent")
    nested = image_cache / "nested" / "__pycache__"
    nested.mkdir(parents=True)
    (nested / "evil.pyc").write_bytes(b"agent")

    run(lockdown._purge_pycache_cmd(str(workspace)))

    assert not planted.exists()
    assert not nested.exists()
    assert (image_cache / "original.pyc").read_bytes() == b"image"


def test_pycache_purge_keeps_only_the_image_files_in_an_image_cache(
    tmp_path: Path, run_purge
):
    """Guards V8-R3-A-1: an image cache keeps only the image's own .pyc files,
    so what the hooks purge compares without __pycache__ is the image's."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    cache = workspace / "package" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "same.pyc").write_bytes(b"image")
    (cache / "rewritten.pyc").write_bytes(b"image")
    (cache / ".hidden.pyc").write_bytes(b"image")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    (cache / "rewritten.pyc").write_bytes(b"agent")
    (cache / "added.pyc").write_bytes(b"agent")
    (cache / ".added.pyc").write_bytes(b"agent")
    (cache / "linked.pyc").symlink_to(cache / "same.pyc")

    run(lockdown._purge_pycache_cmd(str(workspace)))

    assert sorted(p.name for p in cache.iterdir()) == [".hidden.pyc", "same.pyc"]
    assert (cache / "same.pyc").read_bytes() == b"image"


@pytest.mark.parametrize("agent", ["adds_bytecode", "rewrites_bytecode"])
def test_recorded_import_pth_survives_bytecode_added_to_an_image_cache(
    tmp_path: Path, run_purge, agent: str
):
    """Guards V8-R3-A-1: the image ran its venv python once, so the record
    holds _distutils_hack/__pycache__ with __init__ only. Importing setuptools
    in the agent phase adds override's .pyc there (or a stale .pyc is
    rewritten); the shim's .pth must stay and still load in the verifier.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    pth = site / "distutils-precedence.pth"
    pth.write_text(
        "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; "
        "enabled = os.environ.get(var, 'local') == 'local'; "
        "enabled and __import__('_distutils_hack').add_shim(); \n"
    )
    hack = site / "_distutils_hack"
    hack.mkdir()
    (hack / "__init__.py").write_text(
        "import sys\ndef add_shim():\n    sys.shim_loaded = True\n"
    )
    (hack / "override.py").write_text("X = 1\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    python = str(venv / "bin" / "python")
    subprocess.run([python, "-c", "pass"], env=env, check=True)
    image_cache = sorted(p.name for p in (hack / "__pycache__").iterdir())
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    if agent == "adds_bytecode":
        subprocess.run(
            [python, "-c", "import _distutils_hack.override"], env=env, check=True
        )
        assert len(list((hack / "__pycache__").iterdir())) == 2
    else:
        for pyc in (hack / "__pycache__").iterdir():
            pyc.write_bytes(pyc.read_bytes() + b"\0")

    _run_freeze_purges(workspace, run)

    assert pth.exists()
    left = sorted(p.name for p in (hack / "__pycache__").iterdir())
    assert left == (image_cache if agent == "adds_bytecode" else [])
    verifier = subprocess.run(
        [python, "-c", "import sys; print(getattr(sys, 'shim_loaded', False))"],
        env={**os.environ, **lockdown.VERIFIER_ENV},
        capture_output=True,
        text=True,
    )
    assert verifier.returncode == 0, verifier.stderr
    assert verifier.stdout.strip() == "True"


def test_pycache_purge_without_baseline(tmp_path: Path, run_purge):
    """Guards the POSIX purge fallback when no pre-agent copy exists."""
    baseline, run = run_purge
    baseline.rmdir()
    workspace = tmp_path / "workspace"
    cache = workspace / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "evil.pyc").write_bytes(b"agent")
    run(lockdown._purge_pycache_cmd(str(workspace)))
    assert not cache.exists()


def _run_freeze_purges(workspace: Path, run, *, hooks_only: bool = False) -> None:
    # Leave ownership alone in the portable unit test. The root container repro
    # runs the complete preparation, including chown, before verification.
    run(
        "; ".join(
            step.command
            for step in lockdown._freeze_steps(str(workspace))
            if step.name != "chown_workspace"
            and (not hooks_only or step.name == "purge_workspace_python_hooks")
        )
    )


@pytest.mark.parametrize(
    "link", [True, False], ids=["hook-linked-home", "agent-folder"]
)
def test_a_recorded_pth_adding_a_folder_outside_the_workspace_is_removed(
    tmp_path: Path, run_purge, link: bool
):
    """Guards V8-MERGE-3 (b151 sol) and b172's sol FIX-3 / astra FIX-4 at
    a974b81e: the image venv's unchanged path-only .pth adds a folder outside
    the workspace, and the kept venv interpreter puts it on sys.path in the
    verifier. Either the folder is an image one holding a sitecustomize.py
    that links into the agent's home (the agent changes only the home file),
    or it is the agent's own folder. The recorded .pth was kept; integ
    removes the venv's interpreter link, so no venv starts. Now such a .pth
    goes, recorded or not, and the in-workspace editable install stays."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    home = tmp_path / "home" / "agent"
    home.mkdir(parents=True)
    (home / "startup.py").write_text("# benign at record time\n")
    if link:
        outside = tmp_path / "opt" / "sitehooks"
        outside.mkdir(parents=True)
        (outside / "sitecustomize.py").symlink_to(home / "startup.py")
    else:
        outside = home
    external = site / "sitehooks.pth"
    external.write_text(f"{outside}\n")
    (workspace / "src").mkdir()
    editable = site / "__editable__.proj.pth"
    editable.write_text(f"{workspace / 'src'}\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    marker = tmp_path / "hook-ran"
    hook = "startup.py" if link else "sitecustomize.py"
    (home / hook).write_text(f"open({str(marker)!r}, 'w').write('ran')\n")

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert not external.exists()
    assert editable.read_text() == f"{workspace / 'src'}\n"
    subprocess.run([str(venv / "bin" / "python"), "-c", "pass"], check=True)
    assert not marker.exists()


def test_a_real_workspace_interpreter_keeps_its_recorded_outside_pth(
    tmp_path: Path, run_purge
):
    """Guards V8-NARROW-1 at b7953f45: the image builds its venv with
    ``python -m venv --copies /app/.venv`` and installs a project from
    /opt/<lib> into it in editable mode, so the venv's recorded path-only
    .pth names /opt/<lib>/src. The venv's interpreter is a real file in the
    workspace: no link for integ to remove, and integ runs it with that .pth.
    The outside-path rule removed it, and a correct answer failed on import.
    A recorded .pth stays here; a new outside one still goes."""
    baseline, run = run_purge
    workspace = tmp_path / "app"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", "--copies", str(venv)],
        check=True,
    )
    assert not (venv / "bin" / "python").is_symlink()
    site = next((venv / "lib").glob("python*/site-packages"))
    project = tmp_path / "opt" / "honestlib" / "src"
    project.mkdir(parents=True)
    (project / "honestlib.py").write_text("ANSWER = 42\n")
    editable = site / "__editable__.honestlib-1.0.pth"
    editable.write_text(f"{project}\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    # The agent's own outside .pth, made after the record: still removed.
    planted = site / "zz_agent.pth"
    planted.write_text(f"{tmp_path / 'home' / 'agent'}\n")

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert editable.read_text() == f"{project}\n"
    assert not planted.exists()
    done = subprocess.run(
        [
            str(venv / "bin" / "python"),
            "-c",
            "import honestlib; print(honestlib.ANSWER)",
        ],
        capture_output=True,
        text=True,
    )
    assert done.stdout.strip() == "42", done.stderr


@pytest.mark.parametrize(
    "disguise",
    [
        "planted-pyvenv-cfg",
        "copied-interpreter",
        "copied-under-planted-cfg",
        "copied-link-target-only",
    ],
)
def test_a_linked_venv_made_to_look_copied_still_loses_its_outside_pth(
    tmp_path: Path, run_purge, disguise: str
):
    """Guards V8-NARROW-R2-note1 (the review at
    aa934bb5): whether a venv's interpreter link leaves the workspace was
    read from the live tree, so the agent could make the image's linked venv
    look like a ``--copies`` one, and its recorded .pth adding a folder
    outside the workspace was kept. Two routes: a ``pyvenv.cfg`` planted
    between the site folder and the venv (the walk stopped there, found no
    ``bin/`` and said "not linked"; Python never reads that file), or the
    ``bin/python*`` links replaced by copies of the interpreter (all of them,
    or only the one that leaves, so the others resolve to it inside the
    workspace). The answer now comes from the pre-agent copy, links resolved
    there too, so the .pth goes and no agent hook runs."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    assert (venv / "bin" / "python").is_symlink()
    site = next((venv / "lib").glob("python*/site-packages"))
    home = tmp_path / "home" / "agent"
    home.mkdir(parents=True)
    (home / "startup.py").write_text("# benign at record time\n")
    outside = tmp_path / "opt" / "sitehooks"
    outside.mkdir(parents=True)
    (outside / "sitecustomize.py").symlink_to(home / "startup.py")
    external = site / "sitehooks.pth"
    external.write_text(f"{outside}\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    marker = tmp_path / "hook-ran"
    (home / "startup.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\n")
    if disguise in ("planted-pyvenv-cfg", "copied-under-planted-cfg"):
        (site.parent / "pyvenv.cfg").write_text("home = /nowhere\n")
    if disguise != "planted-pyvenv-cfg":
        for link in sorted((venv / "bin").iterdir()):
            if not (link.name.startswith("python") and link.is_symlink()):
                continue
            if disguise == "copied-link-target-only" and not os.path.isabs(
                os.readlink(link)
            ):
                continue
            real = os.path.realpath(link)
            link.unlink()
            shutil.copy2(real, link)
        links = [p for p in (venv / "bin").glob("python*") if p.is_symlink()]
        # With the target only, its siblings still link to it, and every
        # one resolves inside the workspace in the live tree.
        assert bool(links) == (disguise == "copied-link-target-only")
        bin_dir = os.path.realpath(venv / "bin")
        assert all(os.path.dirname(os.path.realpath(p)) == bin_dir for p in links)

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert not external.exists()
    # The venv's own interpreter file: a link to a copied interpreter makes
    # the copy look for its standard library beside the link.
    runner = os.path.realpath(venv / "bin" / "python")
    if not runner.startswith(os.path.realpath(venv) + os.sep):
        runner = str(venv / "bin" / "python")
    subprocess.run([runner, "-c", "pass"], check=True)
    assert not marker.exists()


@pytest.mark.parametrize("bin_target", ["folder", "workspace-root"])
def test_a_linked_venv_whose_bin_is_a_link_loses_its_outside_pth(
    tmp_path: Path, run_purge, bin_target: str
):
    """Guards V8-NOTE1 review A L-1 (at
    0153928f): the image's linked venv has a ``bin/`` that is itself a link
    to a folder in the workspace. Reading the pre-agent copy without
    following that link found no interpreter link, so the venv looked
    copied and its recorded .pth adding an outside folder was kept, where
    integ (which lists through the link) removes it. The folder is now
    resolved through the record first, so the .pth goes and no agent hook
    runs. The same holds for a ``bin/`` linked to the workspace root itself,
    whose copy is listed at the copy's root (V8-NOTE1 r2 review R2-L1, at
    032b6684)."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    if bin_target == "folder":
        (venv / "bin").rename(workspace / "venvbin")
        (venv / "bin").symlink_to("../venvbin")
    else:
        for entry in list((venv / "bin").iterdir()):
            entry.rename(workspace / entry.name)
        (venv / "bin").rmdir()
        (venv / "bin").symlink_to(workspace)
    assert (venv / "bin" / "python").is_symlink()
    assert not os.path.realpath(venv / "bin" / "python").startswith(
        os.path.realpath(workspace) + os.sep
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    home = tmp_path / "home" / "agent"
    home.mkdir(parents=True)
    (home / "startup.py").write_text("# benign at record time\n")
    outside = tmp_path / "opt" / "sitehooks"
    outside.mkdir(parents=True)
    (outside / "sitecustomize.py").symlink_to(home / "startup.py")
    external = site / "sitehooks.pth"
    external.write_text(f"{outside}\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    marker = tmp_path / "hook-ran"
    (home / "startup.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\n")

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert not external.exists()
    if bin_target == "folder":
        # With bin/ at the workspace root, the interpreter looks for
        # pyvenv.cfg beside the root, not in the venv, so it is not a venv
        # run; the removed .pth is the check there.
        subprocess.run([str(venv / "bin" / "python"), "-c", "pass"], check=True)
        assert not marker.exists()


@pytest.mark.parametrize("chain", ["loop", "absolute-reentry"])
def test_a_recorded_interpreter_link_chain_is_resolved_in_the_copy(
    tmp_path: Path, run_purge, chain: str
):
    """Pins two branches of the pre-agent link resolution behind
    V8-NARROW-R2-note1's fix (V8-NOTE1 review B M-1, at 0153928f). A loop
    among the venv's recorded interpreter links stops after 40 hops and the
    recorded outside .pth goes (fails closed). An absolute link back into
    the workspace is read from the copy again, so a hop there that the
    agent replaces with a copy of the interpreter still leaves, and the .pth
    goes."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    hop = workspace / "pyhome" / "bin" / "python3.12"
    hop.parent.mkdir(parents=True)
    if chain == "loop":
        (venv_bin / "python3.12").symlink_to("python")
        (venv_bin / "python").symlink_to("python3.12")
    else:
        hop.symlink_to(_trusted_interpreter())
        (venv_bin / "python3.12").symlink_to(hop)
        (venv_bin / "python").symlink_to("python3.12")
    (venv / "pyvenv.cfg").write_text(f"home = {hop.parent}\n")
    site = venv / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    outside = tmp_path / "opt" / "sitehooks"
    outside.mkdir(parents=True)
    external = site / "sitehooks.pth"
    external.write_text(f"{outside}\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    if chain == "absolute-reentry":
        real = os.path.realpath(hop)
        hop.unlink()
        shutil.copy2(real, hop)

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert not external.exists()


def test_a_venv_linked_to_a_workspace_interpreter_keeps_its_recorded_outside_pth(
    tmp_path: Path, run_purge
):
    """Guards V8-NARROW-R2-note1's fix: the pre-agent copy resolves the
    venv's interpreter links, relative ones included, to a file inside the
    workspace (an interpreter the image installed under it). The venv is not
    one whose link leaves, so its recorded .pth adding an outside folder
    stays, as on integ."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    venv = workspace / ".venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    real = workspace / "pyhome" / "bin" / "python3.12"
    real.parent.mkdir(parents=True)
    real.write_text("stand-in interpreter\n")
    real.chmod(0o755)
    (venv_bin / "python3.12").symlink_to("../../pyhome/bin/python3.12")
    (venv_bin / "python").symlink_to("python3.12")
    (venv / "pyvenv.cfg").write_text(f"home = {real.parent}\n")
    site = venv / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    outside = tmp_path / "opt" / "honestlib" / "src"
    outside.mkdir(parents=True)
    editable = site / "__editable__.honestlib-1.0.pth"
    editable.write_text(f"{outside}\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert editable.read_text() == f"{outside}\n"


def test_workspace_venv_hooks_do_not_run_in_verifier(tmp_path: Path, run_purge):
    """Guards V8-RC1-B-1 at 1c28867b: venv hooks must not run before tests."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace with 'quotes'\nand [glob]"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    (site / "image.pth").write_text("import sys\n")
    (site / "changed.pth").write_text("import sys\n")
    (site / "sitecustomize.py").write_text("# image hook\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    markers = [workspace / "pth-ran", workspace / "sitecustomize-ran"]
    for name, marker in zip(("zz_agent.pth", "sitecustomize.py"), markers, strict=True):
        (site / name).write_text(
            f"import pathlib, os; pathlib.Path({str(marker)!r}).write_text(str(os.geteuid()))\n"
        )
    (site / "changed.pth").write_text("# changed by agent\n")
    (site / "usercustomize.py").write_text("raise RuntimeError('agent hook')\n")
    (site / "answer_library.py").write_text("ANSWER = 42\n")

    _run_freeze_purges(workspace, run)

    verifier = subprocess.run(
        [str(venv / "bin" / "python"), "-c", "import sys; print(sys.prefix)"],
        env={**os.environ, **lockdown.VERIFIER_ENV},
        capture_output=True,
        text=True,
    )
    assert verifier.returncode == 0, verifier.stderr
    assert verifier.stdout.strip() == str(venv)
    assert not any(marker.exists() for marker in markers)
    assert (site / "image.pth").read_text() == "import sys\n"
    for name in ("zz_agent.pth", "sitecustomize.py", "usercustomize.py"):
        assert not (site / name).exists(), name
    # An agent edit that leaves the recorded .pth path-only runs no code, so it
    # stays (V8-R1-B-4); the venv still starts clean because it imports nothing.
    assert (site / "changed.pth").read_text() == "# changed by agent\n"
    assert (site / "answer_library.py").read_text() == "ANSWER = 42\n"


def _venv_first_workspace(tmp_path: Path, baseline: Path, env: dict[str, str]):
    """A workspace venv first on PATH (the usual uv image) holding the image's
    editable project as a recorded path-only .pth."""
    workspace = tmp_path / "workspace with 'quotes' and [glob]"
    venv = workspace / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    source = workspace / "src"
    source.mkdir()
    (source / "proj.py").write_text("VALUE = 7\n")
    editable = site / "__editable__.proj.pth"
    editable.write_text(f"{source}\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    env["PATH"] = f"{venv / 'bin'}:{env['PATH']}"
    return workspace, venv, site, editable


def _without_tmp_sweep(command: str) -> str:
    """*command* with the cleanup's /tmp and /var/tmp ``*.py`` sweep made a
    no-op: it is not under test and would reach other tests' files. Both
    forms: the plain one, and integ's that prunes the workspace first."""
    command = command.replace("find /tmp /var/tmp -name '*.py' -delete", "true")
    return re.sub(
        r"find /tmp /var/tmp \\\( .*? \\\) -prune -o -name '\*\.py' "
        r"-exec rm -f -- \{\} \+",
        "true",
        command,
    )


def _cleanup_dry_run(
    executable: str, env: dict[str, str], workspace: str | None = None
) -> str:
    """The cleanup's sitecustomize/.pth sweep with every removal printed."""
    command = lockdown._build_cleanup_cmd({"cleanup_conftests": False}, workspace)
    command = _without_tmp_sweep(command)
    # PT-R16's HOOK_RULE removes through remove(dfd, name), by descriptor:
    # print the full path instead (Linux /proc names the folder).
    removal = "def remove(dfd, name):\n"
    assert removal in command
    command = command.replace(
        removal,
        removal + '    print(os.path.join(os.readlink(f"/proc/self/fd/{dfd}"), name))\n'
        "    return\n",
    )
    result = subprocess.run(
        [executable, "-c", _as_judged(command, env)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_restore_fallback_runs_no_workspace_venv_python(tmp_path: Path, shell):
    """Guards V8-R1-B-1 at 40618b7b for the opt-in restore's copytree fallback.

    Without rsync the restore ran a bare ``python3``: with the venv first on
    PATH, its site ran the agent's .pth as root before anything removed it.
    The fallback runs with ``-I -S`` (no site), so even the venv's own
    python3, integ's choice, runs no .pth.
    """
    executable, env = shell
    baseline = tmp_path / "baseline"
    workspace, _, site, _ = _venv_first_workspace(tmp_path, baseline, env)
    marker = tmp_path / "agent-pth-ran"
    (site / "zz_agent.pth").write_text(
        f"import os; open({str(marker)!r}, 'a').write(str(os.geteuid()) + '\\n')\n"
    )
    (workspace / "src" / "proj.py").write_text("VALUE = 0\n")
    no_rsync = tmp_path / "no-rsync"
    no_rsync.mkdir()
    (no_rsync / "rsync").write_text("#!/bin/sh\nexit 1\n")
    (no_rsync / "rsync").chmod(0o755)
    env["PATH"] = f"{no_rsync}:{env['PATH']}"
    command = lockdown._restore_workspace_cmd(str(workspace))
    command = command.replace("/testbed_verify", str(baseline))

    # The copy's own exit status is not under test: copytree follows the
    # venv's interpreter links and reports copying each onto itself.
    subprocess.run(
        [executable, "-c", _as_judged(command, env)], env=env, capture_output=True
    )

    assert not marker.exists(), marker.read_text()
    # The fallback ran (on a trusted interpreter) and restored the source.
    assert (workspace / "src" / "proj.py").read_text() == "VALUE = 7\n"


def _no_python3_path(tmp_path: Path) -> Path:
    """A folder with every command on this PATH except python3, so a test can
    stand for an image whose system path has no Python."""
    tools = tmp_path / "tools-without-python3"
    tools.mkdir()
    for entry in os.environ["PATH"].split(os.pathsep):
        if not os.path.isdir(entry):
            continue
        for name in os.listdir(entry):
            source = Path(entry) / name
            if name == "python3" or (tools / name).exists():
                continue
            if source.is_file() and os.access(source, os.X_OK):
                (tools / name).symlink_to(source)
    return tools


@functools.cache
def _trusted_interpreter() -> str:
    """A real interpreter on this machine that root's lookup and the symlink
    purge's kept-link rule can trust: owned by root or us, with its directory
    and that directory's parent, none of them group- or world-writable, and
    every directory above them owned the same way and not group- or
    world-writable unless sticky (the umask may make ours group-writable).
    Test venvs are built on it, so their recorded interpreter links stay."""
    candidates = [sys.executable]
    for directory in _SYSTEM_PATH.split(":"):
        candidates.append(os.path.join(directory, "python3"))
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if not re.fullmatch(r"python3\.[0-9]+", os.path.basename(real)):
            continue
        chain = [real]
        while chain[-1] != os.path.dirname(chain[-1]):
            chain.append(os.path.dirname(chain[-1]))
        try:
            stats = [os.stat(path) for path in chain]
        except OSError:
            continue
        if all(
            st.st_uid in (0, os.geteuid())
            and not (st.st_mode & 0o022 and (index < 3 or not st.st_mode & 0o1000))
            for index, st in enumerate(stats)
        ):
            return real
    pytest.fail("no python3 here is owned by root or us and not group-writable")


def _stand_in_interpreter(path: Path) -> Path:
    """A root-or-our-own executable that prints the path it ran as."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\necho "$0"\n')
    for item in (path, path.parent, path.parent.parent):
        item.chmod(0o755)
    return path


@pytest.mark.parametrize("route", ["one-exec-per-step", "one-script"])
def test_freeze_purges_fail_when_their_snippet_fails(
    tmp_path: Path, shell, monkeypatch: pytest.MonkeyPatch, route: str
):
    """Guards V8-R1-B-1 (round 1 at b449a20b): a purge that stops partway
    fails hardening.

    The symlink purge and the hooks purge ended in
    ``2>/dev/null || true``, so an error the snippet does not catch stopped it
    partway and the step still passed: on Python 3.11 an agent-made
    1,100-deep folder makes the hooks purge's os.walk raise RecursionError
    before it purges anything, and the agent's import .pth stayed for every
    verifier venv Python. The stand-in interpreter fails as that run does.
    """
    from benchflow.sandbox import _hardening_python

    monkeypatch.setattr(_hardening_python, "RUNTIME_PATH_PREFIXES", ())
    tools = _no_python3_path(tmp_path)
    monkeypatch.setattr(_hardening_python, "TRUSTED_SYSTEM_PATH", str(tools))
    executable, env = shell
    shell_bin = env["PATH"].split(":", 1)[0]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    crashing = _stand_in_interpreter(tmp_path / "image" / "bin" / "python3.11")
    crashing.write_text(
        "#!/bin/sh\necho 'RecursionError: maximum recursion depth exceeded' >&2\n"
        "exit 1\n"
    )
    (crashing.parent / "python3").symlink_to(crashing.name)
    # V8's purges; the injected-hook cleanup is U5a's helper (its own rule).
    names = {"purge_external_symlinks", "purge_workspace_python_hooks"}
    steps = [s for s in lockdown._freeze_steps(str(workspace)) if s.name in names]
    assert len(steps) == 2
    path = f"{shell_bin}:{crashing.parent}:{tools}"

    for step in steps:
        assert step.label is not None, step.name
        # The /tmp sweep is not under test and would reach other tests' files.
        command = _without_tmp_sweep(step.command)
        assert "find /" not in command
        if route == "one-script":
            parent = tmp_path / f"sticky-{step.name}"
            parent.mkdir(mode=0o700)
            parent.chmod(0o1700)
            command = lockdown._prep_script(
                [lockdown._PrepStep(step.name, step.span, command, 10, step.label)],
                str(parent / "prep"),
            ).replace(
                # U5a's runner gives its steps its own tool path: the image's
                # crashing python3 is on it, as a system Python would be.
                f"__bf_path={shlex.quote(_ROUTE_PATH)}",
                f"__bf_path={shlex.quote(f'{crashing.parent}:{tools}')}",
            )
        result = subprocess.run(
            [executable, "-c", _as_judged(command, {"PATH": path})],
            env={**env, "PATH": path},
            capture_output=True,
            text=True,
        )
        if route == "one-script":
            assert f"BENCHFLOW-PREP-FAIL {step.name}" in result.stdout, step.name
        else:
            assert result.returncode == 1, (step.name, result.stderr)
        assert "RecursionError" in result.stderr, step.name


def test_freeze_runs_on_the_python_behind_a_workspace_venv(
    tmp_path: Path, run_purge, shell, monkeypatch: pytest.MonkeyPatch
):
    """Guards V8-R2-A-1 at d599a042 end to end with a real venv.

    The image's only Python is the one the workspace venv links to (outside
    the system path): the purges must still run (on integ's choice, the
    venv's python3, with ``-I -S``) and remove the agent's
    import .pth without running it.
    """
    from benchflow.sandbox import _hardening_python

    interpreter = _trusted_interpreter()
    tools = _no_python3_path(tmp_path)
    monkeypatch.setattr(_hardening_python, "TRUSTED_SYSTEM_PATH", str(tools))
    baseline, run = run_purge
    _, env = shell
    shell_bin = env["PATH"].split(":", 1)[0]
    env["PATH"] = f"{shell_bin}:{tools}"
    workspace, _, site, editable = _venv_first_workspace(tmp_path, baseline, env)
    # The image's only Python, outside the system path the lookup gets: the
    # venv's python3 links to it in the image (the record) too.
    for tree in (workspace, baseline):
        link = tree / ".venv" / "bin" / "python3"
        link.unlink()
        link.symlink_to(interpreter)
    marker = tmp_path / "agent-pth-ran"
    agent = site / "zz_agent.pth"
    agent.write_text(f"import os; open({str(marker)!r}, 'a').write('ran\\n')\n")

    _run_freeze_purges(workspace, run)

    assert not marker.exists(), marker.read_text()
    assert not agent.exists()
    assert editable.read_text() == f"{workspace / 'src'}\n"


@pytest.mark.parametrize(
    "venv_in_workspace", [True, False], ids=["workspace", "elsewhere"]
)
def test_cleanup_leaves_a_workspace_venvs_pth_files_to_the_hooks_purge(
    tmp_path: Path, shell, venv_in_workspace: bool
):
    """Guards V8-R1-B-2 as 5c77d08b brings it back: with
    integ's interpreter, the image PATH's python3, a workspace venv first on
    PATH is the cleanup's Python once the freeze has kept its link, and the
    cleanup swept that venv's .pth files: the image's editable install went
    and a correct answer failed on import. A workspace's site folders are
    the hooks purge's (step 5); a venv elsewhere is still swept."""
    executable, env = shell
    workspace = tmp_path / "workspace"
    venv = (workspace if venv_in_workspace else tmp_path / "opt") / ".venv"
    subprocess.run(
        [_trusted_interpreter(), "-m", "venv", "--without-pip", str(venv)], check=True
    )
    workspace.mkdir(exist_ok=True)
    site = next((venv / "lib").glob("python*/site-packages"))
    editable = site / "__editable__.proj.pth"
    editable.write_text(f"{workspace / 'src'}\n")
    env["PATH"] = f"{venv / 'bin'}:{env['PATH']}"

    removed = _cleanup_dry_run(executable, env, str(workspace)).split("\n")

    assert (str(editable) in removed) is not venv_in_workspace, removed


def test_workspace_hook_purge_matches_what_python_runs(tmp_path: Path, run_purge):
    """Guards V8-RC1-B-1: every importable hook form goes, data .pth files stay.

    Python runs only a site dir's own *.pth files, so a package's model weights
    or a checkpoint named *.pth elsewhere are kept; a sitecustomize module is
    removed in any importable form anywhere in a site tree, also through a
    site-packages link to an in-tree directory.
    """
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    site = workspace / "env" / "lib" / "python3" / "dist-packages"
    site.mkdir(parents=True)
    recorded = site / "usercustomize"
    recorded.mkdir()
    (recorded / "__init__.py").write_text("# image package\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    hooks = [
        site / "sitecustomize.pyc",
        site / "sitecustomize.cpython-312-x86_64-linux-gnu.so",
        site / "pkg" / "sitecustomize.py",
        site / "sitecustomize" / "__init__.py",
    ]
    data = [
        site / "pkg" / "weights.pth",
        workspace / "models" / "best.pth",
        workspace / "sitecustomize.py",
    ]
    for path in hooks + data:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("agent")
    (site / "sitecustomize.txt").write_text("not a module")
    linked = workspace / "libs"
    linked.mkdir()
    (linked / "agent.pth").write_text("import sys\n")
    (linked / "nested.pth").mkdir()
    (workspace / "env" / "site-packages").symlink_to("../libs")

    _run_freeze_purges(workspace, run, hooks_only=True)

    for path in hooks:
        assert not path.exists(), path
    assert not (site / "sitecustomize").exists()
    assert not (linked / "agent.pth").exists()
    assert (linked / "nested.pth").is_dir()
    for path in data:
        assert path.read_text() == "agent", path
    assert (site / "sitecustomize.txt").exists()
    assert (recorded / "__init__.py").read_text() == "# image package\n"
    (recorded / "extra.py").write_text("agent")
    _run_freeze_purges(workspace, run, hooks_only=True)
    assert not recorded.exists()


@pytest.mark.parametrize("site_name", ["site-packages", "dist-packages"])
def test_workspace_hook_purge_does_not_follow_links(
    tmp_path: Path, run_purge, site_name: str
):
    """Guards V8-RC1-B-1 at 1c28867b: neither walk nor comparison follows links."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    site = workspace / "nested" / site_name
    site.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "outside.pth"
    sentinel.write_text("outside unchanged")
    (site / "linked.pth").symlink_to(sentinel)
    (site / "directory").symlink_to(outside, target_is_directory=True)
    (site / "agent.pth").write_text("import sys  # same bytes, unsafe baseline\n")
    (baseline / "nested").symlink_to(workspace / "nested", target_is_directory=True)

    _run_freeze_purges(workspace, run, hooks_only=True)

    assert not (site / "linked.pth").is_symlink()
    assert (site / "directory").is_symlink()
    assert not (site / "agent.pth").exists()
    assert sentinel.read_text() == "outside unchanged"


@pytest.mark.parametrize("untrusted", ["missing", "unlocked", "symlink"])
def test_untrusted_baseline_cannot_exempt_workspace_hooks(
    tmp_path: Path, run_purge, untrusted: str
):
    """Guards V8-RC1-B-1 at 1c28867b: only the locked image copy exempts hooks."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    site = workspace / "dist-packages"
    site.mkdir(parents=True)
    hook = site / "agent.pth"
    hook.write_text("import sys\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    if untrusted == "missing":
        shutil.rmtree(baseline)
    elif untrusted == "unlocked":
        baseline.chmod(0o777)
    else:
        real = tmp_path / "untrusted"
        baseline.rename(real)
        baseline.symlink_to(real, target_is_directory=True)

    _run_freeze_purges(workspace, run)

    assert not hook.exists()


def test_a_test_runner_module_the_agent_added_is_removed(tmp_path: Path, run_purge):
    """Guards the fix against the reward hack found while checking
    benchflow-ai/benchflow#1116: test.sh runs in the workspace, so under
    ``python3 -m pytest`` an agent's /app/pytest.py replaced pytest, ran as
    root and scored an unsolved task 1.0 (0.9.0 and a28f2b29)."""
    _baseline, run = run_purge
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pytest.py").write_text("print('1 passed')\n")
    (workspace / "_pytest").mkdir()
    (workspace / "_pytest" / "__init__.py").write_text("")
    (workspace / "pluggy.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")
    (workspace / "py.pyc").write_bytes(b"")
    (workspace / "solution.py").write_text("ANSWER = 42\n")
    (workspace / "pytest.ini").write_text("[pytest]\n")

    run(lockdown._purge_workspace_python_hooks_cmd(str(workspace)))

    assert sorted(p.name for p in workspace.iterdir()) == ["pytest.ini", "solution.py"]


def test_the_repositorys_own_runner_named_module_stays(tmp_path: Path, run_purge):
    """A repository that ships a top-level py.py or pluggy package keeps it,
    changed or not: the pre-agent copy has an entry by that name."""
    baseline, run = run_purge
    workspace = tmp_path / "workspace"
    (workspace / "pluggy").mkdir(parents=True)
    (workspace / "pluggy" / "__init__.py").write_text("VERSION = 1\n")
    (workspace / "py.py").write_text("# the repository's\n")
    shutil.copytree(workspace, baseline, dirs_exist_ok=True, symlinks=True)
    baseline.chmod(0o700)
    (workspace / "pluggy" / "__init__.py").write_text("VERSION = 2\n")

    run(lockdown._purge_workspace_python_hooks_cmd(str(workspace)))

    assert (workspace / "pluggy" / "__init__.py").read_text() == "VERSION = 2\n"
    assert (workspace / "py.py").read_text() == "# the repository's\n"
