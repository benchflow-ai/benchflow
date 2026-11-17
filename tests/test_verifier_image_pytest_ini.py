"""Strict passes pytest the image's own configuration file, never the agent's
(V1p's finding, review round 1: V3-R1-A-1, V3-R1-A-2;, the lead's
13:57Z ruling on CIR's strict-boundary hole).

The verifier ran pytest with ``-c /dev/null`` so that a configuration file the
agent wrote or edited is never read. That also hid the image's own: pytest-dev's
tox.ini ``[pytest] addopts = -ra -p pytester --strict-markers`` was dropped, so
every test that takes ``testdir`` errored with "fixture 'testdir' not found"
and the pytest-5809 and pytest-5840 oracles failed on 0.8 and on the fixes
alike. Without any ``-c`` (G13a) pytest read the agent's file, whose
``addopts`` load before PYTEST_ADDOPTS: ``-p bf_evil`` ran the agent's module
ahead of the plugin guard. Strict now always passes ``-c``: the file pytest
would read, chosen on the image's copies in the root-only build snapshot and
copied outside the workspace (SWT-bench's ``git add -A`` must not see it),
its ``pythonpath`` passed absolute; else ``/dev/null``. ``--rootdir`` is the
workspace, where pytest itself finds the image's file.
The ini's ``-p`` plugins move after BenchFlow's plugin guard, which judges
them. A task whose tests live in the verifier package (``pytest
/tests/test_outputs.py``) keeps ``-c /dev/null``: pytest never reads the
workspace's file for them.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox.lockdown import harden_before_verify
from tests.test_sandbox_hardening import _make_env, _make_task

TOX_INI = (
    "[tox]\nenvlist = py\n\n[pytest]\naddopts = -ra -p pytester --strict-markers\n"
)
SWE_TEST_SH = "#!/bin/bash\ncd /testbed\npytest -rA testing/test_pastebin.py\n"
GUARD = "_runtime_guard_"
COPY = re.compile(r"[0-9a-f]{16}\.(ini|cfg|toml)")


def _probe_stdout(marker: str, image: dict[str, str]) -> str:
    """What the probe prints for these files in the build snapshot (kind
    "file", or the kind itself when the value is "<link>", "<big>", ...)."""
    out = ""
    for name in lockdown._PYTEST_CONFIG_NAMES:
        if name not in image:
            continue
        text = image[name]
        if text.startswith("<") and text.endswith(">"):
            out += f"{marker} {name} image {text[1:-1]}\n"
        else:
            out += f"{marker} {name} image file\n{text}\n"
    return out


def _ini_answer(image: dict[str, str]):
    """A sandbox whose snapshot holds *image*; every other exec is quiet."""
    probes: list[str] = []
    copies: list[str] = []

    def side_effect(cmd, **kwargs):
        if "bf-pytest-ini" in cmd:
            probes.append(cmd)
            marker = shlex.split(cmd)[-1]
            return SimpleNamespace(
                stdout=_probe_stdout(marker, image), stderr="", exit_code=0
            )
        if lockdown._PYTEST_CONFIG_DIR + "/" in cmd:
            copies.append(cmd)
        return SimpleNamespace(stdout="", stderr="", exit_code=0)

    env = _make_env(side_effect=side_effect)
    env.probes = probes
    env.copies = copies
    return env


def _task_with_package(tmp_path: Path, files: dict[str, str] | None = None):
    """A task whose verifier package (tests/) holds *files* (a SWE-bench-style
    test.sh by default)."""
    task = _make_task()
    package = tmp_path / "task" / "tests"
    package.mkdir(parents=True)
    for name, text in (files or {"test.sh": SWE_TEST_SH}).items():
        (package / name).write_text(text)
    task.paths.tests_dir = package
    return task


def _options(task) -> dict[str, str]:
    args = shlex.split(task.config.verifier.env["PYTEST_ADDOPTS"])
    return {
        "-c": args[args.index("-c") + 1],
        "rootdir": next(a for a in args if a.startswith("--rootdir=")).split("=", 1)[1],
        "args": args,
    }


def _uses_dev_null(task, workspace: str) -> None:
    options = _options(task)
    assert options["-c"] == "/dev/null"
    assert options["rootdir"] == workspace
    args = options["args"]
    assert args[-2] == "-p" and args[-1].startswith(GUARD)


def _uses_the_images_copy(task, env, workspace: str, name: str, suffix: str) -> str:
    """The -c file is BenchFlow's copy of the image's *name*, outside the
    workspace, the rootdir the workspace; the copy comes from the snapshot,
    and no probe looked at the workspace."""
    options = _options(task)
    path = options["-c"]
    folder, base = path.rsplit("/", 1)
    assert folder == lockdown._PYTEST_CONFIG_DIR and COPY.fullmatch(base)
    assert path.endswith(suffix)
    assert options["rootdir"] == workspace
    [probe] = env.probes
    assert workspace not in probe
    [copy] = env.copies
    assert f"cp {lockdown._SNAPSHOT_DIR}/{name} " in copy and path in copy
    return path


@pytest.mark.asyncio
async def test_the_images_ini_replaces_dev_null(tmp_path):
    """pytest-dev's tox.ini reaches pytest, its -p pytester after the guard."""
    task = _task_with_package(tmp_path)
    env = _ini_answer({"tox.ini": TOX_INI})
    await harden_before_verify(
        env,
        task,
        sandbox_user="agent",
        workspace="/testbed",
        integrity="strict",
    )

    _uses_the_images_copy(task, env, "/testbed", "tox.ini", ".ini")
    args = _options(task)["args"]
    assert args[args.index("-o") + 1] == "addopts=-ra --strict-markers"
    guard = next(i for i, a in enumerate(args) if a.startswith(GUARD))
    assert args[guard + 1 :] == ["-p", "pytester"]


@pytest.mark.asyncio
async def test_an_ini_without_plugins_is_passed_as_it_is(tmp_path):
    task = _task_with_package(tmp_path)
    env = _ini_answer({"pytest.ini": "[pytest]\naddopts = -ra\nmarkers = slow\n"})
    await harden_before_verify(
        env,
        task,
        sandbox_user="agent",
        workspace="/app",
        integrity="strict",
    )

    _uses_the_images_copy(task, env, "/app", "pytest.ini", ".ini")
    assert " -o " not in task.config.verifier.env["PYTEST_ADDOPTS"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "text", "suffix"),
    [
        ("tox.ini", TOX_INI, ".ini"),
        (
            "setup.cfg",
            "[metadata]\nname = x\n[tool:pytest]\naddopts = -p pytester\n",
            ".cfg",
        ),
        (
            "pyproject.toml",
            '[project]\nname = "x"\n[tool.pytest.ini_options]\naddopts = "-p pytester"\n',
            ".toml",
        ),
    ],
    ids=["tox-ini", "setup-cfg", "pyproject-toml"],
)
async def test_the_copy_keeps_the_kind_of_file_pytest_reads(
    tmp_path, name, text, suffix
):
    """pytest reads a file given with -c by its suffix: [pytest] in .ini,
    [tool:pytest] in .cfg, [tool.pytest.ini_options] in .toml."""
    task = _task_with_package(tmp_path)
    env = _ini_answer({name: text})
    await harden_before_verify(
        env, task, sandbox_user="agent", workspace="/testbed", integrity="strict"
    )

    _uses_the_images_copy(task, env, "/testbed", name, suffix)
    assert task.config.verifier.env["PYTEST_ADDOPTS"].endswith(" -p pytester")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image",
    [
        {},  # the image has no configuration file
        {"tox.ini": "<link>"},
        {"tox.ini": "<other>"},
        {"tox.ini": "<big>"},
        # A second addopts in [pytest]: pytest's iniconfig refuses the file.
        {"tox.ini": TOX_INI + "addopts = -p evil\n"},
        {"tox.ini": "[pytest]\naddopts = -vp evil\n"},  # a hidden -p
        {"pyproject.toml": "[tool.pytest.ini_options\n"},  # unreadable
    ],
    ids=["none", "link", "other", "big", "duplicate-key", "bundled-p", "bad-toml"],
)
async def test_anything_else_keeps_dev_null(tmp_path, image):
    task = _task_with_package(tmp_path)
    env = _ini_answer(image)
    await harden_before_verify(
        env,
        task,
        sandbox_user="agent",
        workspace="/testbed",
        integrity="strict",
    )

    _uses_dev_null(task, "/testbed")
    assert env.copies == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "package",
    [
        {
            "test.sh": "#!/bin/bash\ncd /app && pytest /tests/test_outputs.py\n",
            "test_outputs.py": "def test_answer():\n    assert True\n",
        },
        {"test.sh": "#!/bin/bash\npytest $TEST_DIR\n", "conftest.py": ""},
        {
            "test.sh": "#!/bin/bash\ncat > /tmp/check.py <<'E'\ndef test_x(): pass\n"
            "E\ncd /app && pytest /tmp/check.py\n"
        },
    ],
    ids=["tests-in-package", "conftest-in-package", "test-written-to-tmp"],
)
async def test_tests_outside_the_workspace_keep_dev_null(tmp_path, package):
    """Guards V3-R1-A-1: the workspace's file was forced on verifier tests in
    /tests, which pytest never reads it for. /app's addopts = --cov failed a
    correct answer with a usage error."""
    task = _task_with_package(tmp_path, package)
    env = _ini_answer(
        {"pyproject.toml": "[tool.pytest.ini_options]\naddopts = '--cov=proj'\n"}
    )

    await harden_before_verify(
        env, task, sandbox_user="agent", workspace="/app", integrity="strict"
    )

    _uses_dev_null(task, "/app")
    assert env.probes == []


@pytest.mark.asyncio
async def test_a_helper_a_python_runs_outside_the_workspace_keeps_the_image_ini(
    tmp_path,
):
    """Guards SWT-bench's test.sh runs helpers by absolute path
    (`python /root/test_command_generator.py`, `python /root/parser.py`) and
    then the repository's own tests in /testbed. Those helpers are no test
    pytest is pointed at, so the image's tox.ini, and with it `-p pytester`,
    is used; pytest-7236 scored 0 under `-c /dev/null`. A test file pytest is
    pointed at outside the workspace still keeps `-c /dev/null`
    (test_tests_outside_the_workspace_keep_dev_null)."""
    package = {
        "test.sh": "#!/bin/bash\n"
        "python /root/test_command_generator.py --test_patch_file /root/model_test.patch\n"
        "rm -rf /testbed/* && cp -r /root/tmp_repo/* /testbed\n"
        "bash /root/run_model_test_base_code.sh\n"
        "python /root/parser.py > /root/parser_output.txt 2>&1\n"
    }
    task = _task_with_package(tmp_path, package)
    env = _ini_answer(
        {"tox.ini": "[pytest]\naddopts = -rfEX -p pytester --strict-markers\n"}
    )

    await harden_before_verify(
        env, task, sandbox_user="agent", workspace="/testbed", integrity="strict"
    )

    _uses_the_images_copy(task, env, "/testbed", "tox.ini", ".ini")
    assert task.config.verifier.env["PYTEST_ADDOPTS"].endswith(" -p pytester")


@pytest.mark.parametrize(
    ("line", "refused"),
    [
        ("pytest /tmp/check.py", True),
        ("python -m pytest -q \\\n    /tmp/check.py", True),
        ("py.test /opt/t.py", True),
        ("python /root/parser.py", False),
        ("python3 /root/trace.py --count", False),
        ("cd /testbed && pytest tests/test_x.py", False),
    ],
)
def test_only_a_test_pytest_is_pointed_at_counts(tmp_path, line, refused):
    from benchflow.sandbox.lockdown import _pytest_may_read_the_workspace_config

    task = _task_with_package(tmp_path, {"test.sh": "#!/bin/bash\n" + line + "\n"})

    assert _pytest_may_read_the_workspace_config(task, "/testbed") is not refused


@pytest.mark.asyncio
async def test_tests_in_the_package_pass_under_a_workspace_cov_addopts(
    tmp_path, monkeypatch
):
    """Guards V3-R1-A-1 with a real pytest: ``cd /app && pytest
    /tests/test_outputs.py`` with an unchanged /app/pyproject.toml that sets
    ``addopts = --cov=proj`` and ``python_functions = check_*``: 1 passed, as
    pytest gives without BenchFlow."""
    app = tmp_path / "app"
    app.mkdir()
    pyproject = (
        "[tool.pytest.ini_options]\n"
        'addopts = "--cov=proj --strict-markers"\n'
        'python_functions = ["check_*"]\n'
    )
    (app / "pyproject.toml").write_text(pyproject)
    task = _task_with_package(
        tmp_path,
        {
            "test.sh": "#!/bin/bash\ncd /app && pytest /tests/test_outputs.py\n",
            "test_outputs.py": "def test_answer():\n    assert 1 + 1 == 2\n",
        },
    )
    tests = task.paths.tests_dir
    monkeypatch.setattr(lockdown, "_verifier_confcutdir", lambda task: str(tests))

    await harden_before_verify(
        _ini_answer({"pyproject.toml": pyproject}),
        task,
        sandbox_user="agent",
        workspace=str(app),
        integrity="strict",
    )
    # The guard itself is not importable here; what comes before it is.
    addopts = task.config.verifier.env["PYTEST_ADDOPTS"]
    addopts = addopts.split(f" -p {GUARD}")[0]
    done = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests / "test_outputs.py")],
        cwd=app,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTEST_ADDOPTS": addopts,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert "1 passed" in done.stdout


def test_the_verifier_package_decides_where_pytest_reads_from(tmp_path):
    from benchflow.sandbox.lockdown import _pytest_may_read_the_workspace_config

    swe = _task_with_package(tmp_path / "a")
    assert _pytest_may_read_the_workspace_config(swe, "/testbed")
    in_workspace = _task_with_package(
        tmp_path / "b", {"test.sh": "#!/bin/bash\npython /testbed/run.py\n"}
    )
    assert _pytest_may_read_the_workspace_config(in_workspace, "/testbed")
    elsewhere = _task_with_package(
        tmp_path / "c", {"test.sh": "#!/bin/bash\npytest /opt/hidden/test_x.py\n"}
    )
    assert not _pytest_may_read_the_workspace_config(elsewhere, "/testbed")
    unreadable = _make_task()
    assert not _pytest_may_read_the_workspace_config(unreadable, "/testbed")


def test_the_probe_output_is_read_back_exactly():
    from benchflow.sandbox.lockdown import (
        _choose_image_pytest_config,
        _parse_pytest_config_probe,
    )

    stdout = _probe_stdout(
        "M1",
        {"pytest.ini": "<link>", "tox.ini": TOX_INI, "setup.cfg": "<other>"},
    )
    records = _parse_pytest_config_probe(stdout, "M1")
    assert records == {
        ("pytest.ini", "image"): ("link", ""),
        ("tox.ini", "image"): ("file", TOX_INI),
        ("setup.cfg", "image"): ("other", ""),
    }
    # The image's pytest.ini is a link: nothing is used.
    assert _choose_image_pytest_config(records) is None
    del records["pytest.ini", "image"]
    assert _choose_image_pytest_config(records) == ("tox.ini", TOX_INI)
    # pytest's order: tox.ini has no [pytest], so setup.cfg's [tool:pytest].
    assert _choose_image_pytest_config(
        {
            ("pyproject.toml", "image"): ("file", "[project]\nname = 'x'\n"),
            ("tox.ini", "image"): ("file", "[tox]\nenvlist = py\n"),
            ("setup.cfg", "image"): ("file", "[tool:pytest]\naddopts = -q\n"),
        }
    ) == ("setup.cfg", "[tool:pytest]\naddopts = -q\n")
    # An empty pytest.ini is still pytest's choice.
    assert _choose_image_pytest_config(
        {("pytest.ini", "image"): ("file", ""), ("tox.ini", "image"): ("file", TOX_INI)}
    ) == ("pytest.ini", "")
    # Output that does not start with the marker is not read.
    assert _parse_pytest_config_probe("x\n" + stdout, "M1") == {}


def test_ini_files_are_read_as_iniconfig_reads_them():
    from benchflow.sandbox.lockdown import _iniconfig_sections

    assert _iniconfig_sections(
        "; c\n[pytest] # c\naddopts =\n  -ra\n  -q\n[x\nmarkers: slow\n"
    ) == {"pytest": {"addopts": "-ra\n-q\n[x", "markers": "slow"}}
    for text in (
        "addopts = -q\n",  # no section
        "[pytest]\n[pytest]\n",  # duplicate section
        "[pytest]\na = 1\na = 2\n",  # duplicate name
        "[pytest]\n  -q\n",  # continuation of nothing
        "[pytest]\njust words\n",  # unreadable line
    ):
        with pytest.raises(ValueError):
            _iniconfig_sections(text)


def test_ini_addopts_are_read_as_pytest_reads_them():
    from benchflow.sandbox.lockdown import _ini_addopts, _split_ini_plugins

    assert _ini_addopts("tox.ini", TOX_INI) == [
        "-ra",
        "-p",
        "pytester",
        "--strict-markers",
    ]
    setup_cfg = "[metadata]\nname = x\n[tool:pytest]\naddopts =\n  -p xdist\n  -q\n"
    assert _ini_addopts("setup.cfg", setup_cfg) == ["-p", "xdist", "-q"]
    pyproject = '[tool.pytest.ini_options]\naddopts = ["-ppytester", "-q"]\n'
    assert _ini_addopts("pyproject.toml", pyproject) == ["-ppytester", "-q"]
    assert _ini_addopts("pytest.ini", "[pytest]\nmarkers = slow\n") == []

    assert _split_ini_plugins(["-ra", "-p", "pytester", "-pno:cacheprovider"]) == (
        ["-ra", "-p", "no:cacheprovider"],
        ["pytester"],
    )
    assert _split_ini_plugins(["-Wignore::DeprecationWarning", "-rap"]) == (
        ["-Wignore::DeprecationWarning", "-rap"],
        [],
    )
    assert _split_ini_plugins(["-vp", "evil"]) is None
    assert _split_ini_plugins(["-p"]) is None


EVIL = "import pathlib\npathlib.Path(__file__).with_name('bf-evil-loaded').write_text('1')\n"


def test_a_real_pytest_reads_the_images_copy_never_the_agents_file(tmp_path):
    """The built PYTEST_ADDOPTS give a real pytest the image's options and
    its plugin from BenchFlow's copy outside the project, with the project
    as rootdir; the agent's tox.ini, whose addopts would load its own module
    ahead of the guard (CIR's bf_evil), is never read, with the copy or with
    ``-c /dev/null``."""
    from benchflow.sandbox.lockdown import ImagePytestIni, _build_pytest_addopts

    project = tmp_path / "testbed"
    (project / "testing").mkdir(parents=True)
    (tmp_path / "cfg").mkdir()
    copy = tmp_path / "cfg" / "0123456789abcdef.ini"
    copy.write_text(TOX_INI)
    (project / "tox.ini").write_text("[pytest]\naddopts = -p bf_evil\n")
    (project / "bf_evil.py").write_text(EVIL)
    (project / "testing" / "test_uses_pytester.py").write_text(
        "def test_it(pytester):\n    assert pytester is not None\n"
    )

    def run(ini):
        addopts = _build_pytest_addopts(
            str(project), "", verifier_confcutdir=str(project), ini=ini
        )
        return subprocess.run(
            [sys.executable, "-m", "pytest", "testing/test_uses_pytester.py"],
            cwd=project,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTEST_ADDOPTS": addopts,
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            },
            capture_output=True,
            text=True,
            timeout=120,
        )

    honoured = run(
        ImagePytestIni(
            path=str(copy), addopts="-ra --strict-markers", plugins=("pytester",)
        )
    )
    assert honoured.returncode == 0, honoured.stdout + honoured.stderr
    assert "1 passed" in honoured.stdout
    assert f"rootdir: {project}" in honoured.stdout
    assert "configfile: ../cfg/0123456789abcdef.ini" in honoured.stdout

    hidden = run(None)
    assert "fixture 'pytester' not found" in hidden.stdout, hidden.stdout
    assert f"rootdir: {project}" in hidden.stdout
    assert not (project / "bf-evil-loaded").exists()


def test_the_copys_pythonpath_resolves_from_the_images_folder(tmp_path):
    """pytest resolves ``pythonpath`` from the folder of the file it is
    given: from BenchFlow's copy, ``pythonpath = src`` would name the copy's
    folder. The -o override gives the image folder's (CIR's ``pythonpath =
    .``, SWE-bench-style ``src`` layouts)."""
    from benchflow.sandbox.lockdown import (
        ImagePytestIni,
        _build_pytest_addopts,
        _ini_pythonpath,
    )

    project = tmp_path / "app"
    (project / "src").mkdir(parents=True)
    (project / "tests").mkdir()
    (project / "src" / "mod.py").write_text("def answer():\n    return 42\n")
    (project / "tests" / "test_mod.py").write_text(
        "import mod\n\n\ndef test_mod():\n    assert mod.answer() == 42\n"
    )
    ini = "[pytest]\npythonpath = src\n"
    (tmp_path / "cfg").mkdir()
    copy = tmp_path / "cfg" / "0123456789abcdef.ini"
    copy.write_text(ini)

    def run(pythonpath):
        addopts = _build_pytest_addopts(
            str(project),
            "",
            verifier_confcutdir=str(project),
            ini=ImagePytestIni(path=str(copy), pythonpath=pythonpath),
        )
        return subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_mod.py"],
            cwd=project,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTEST_ADDOPTS": addopts,
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            },
            capture_output=True,
            text=True,
            timeout=120,
        )

    assert _ini_pythonpath("pytest.ini", ini, str(project)) == str(project / "src")
    honoured = run(_ini_pythonpath("pytest.ini", ini, str(project)))
    assert honoured.returncode == 0, honoured.stdout + honoured.stderr
    assert "1 passed" in honoured.stdout
    unresolved = run(None)
    assert "No module named 'mod'" in unresolved.stdout, unresolved.stdout


# The probe itself, in a container (skipped without Docker): root's build
# snapshot and the workspace as the agent left it.


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


class ContainerEnv:
    def __init__(self, name: str) -> None:
        self.name = name

    async def exec(self, command: str, user: str | None = None, **kwargs):
        done = subprocess.run(
            ["docker", "exec", "-u", user or "root", self.name, "sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return SimpleNamespace(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )


@pytest.fixture
def box():
    if not _docker_ready():
        pytest.skip("docker required")
    name = f"bf-v3-ini-{uuid.uuid4().hex[:8]}"
    run = subprocess.run(
        [
            *("docker", "run", "-d", "--network", "none", "--name", name),
            *("alpine:3.20", "sleep", "600"),
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        pytest.skip(f"could not start the container: {run.stderr[:200]}")
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _write(name: str, path: str, text: str) -> None:
    done = subprocess.run(
        ["docker", "exec", "-i", name, "sh", "-c", f"cat > {shlex.quote(path)}"],
        input=text,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def _sh(name: str, command: str) -> str:
    done = subprocess.run(
        ["docker", "exec", name, "sh", "-c", command], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


IMAGE_FILES = {
    # pytest would read tox.ini: pyproject.toml has no pytest table.
    "pyproject.toml": "[project]\nname = 'x'\n",
    "tox.ini": TOX_INI,
    "setup.cfg": "[tool:pytest]\naddopts = -q\n",
}
SNAPSHOT = lockdown._SNAPSHOT_DIR


def _image(name: str, files=IMAGE_FILES) -> None:
    """The image's workspace, and the snapshot _snapshot_build_config takes
    of it before the agent's turn."""
    _sh(name, f"mkdir -p /testbed {SNAPSHOT} && chmod 700 {SNAPSHOT}")
    for file, text in files.items():
        _write(name, f"/testbed/{file}", text)
        _write(name, f"{SNAPSHOT}/{file}", text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "used"),
    [
        ("true", True),
        # Whatever the agent did to the workspace, the image's tox.ini is used.
        ("printf '[pytest]\\naddopts = -p evil\\n' > /testbed/pytest.ini", True),
        ("printf 'addopts = -p evil\\n' >> /testbed/tox.ini", True),
        ("rm /testbed/tox.ini && ln -s /etc/passwd /testbed/tox.ini", True),
        ("rm /testbed/tox.ini /testbed/setup.cfg", True),
        # No snapshot, or one that is not root's alone: -c /dev/null.
        (f"rm -r {SNAPSHOT}", False),
        (f"chmod 755 {SNAPSHOT}", False),
        (f"chown 1000 {SNAPSHOT}", False),
        (
            f"rm {SNAPSHOT}/tox.ini && ln -s /testbed/setup.cfg {SNAPSHOT}/tox.ini",
            False,
        ),
    ],
    ids=[
        "unchanged",
        "agent-added-pytest-ini",
        "edited",
        "replaced-by-link",
        "removed",
        "no-snapshot",
        "open-snapshot",
        "snapshot-not-roots",
        "snapshot-link",
    ],
)
async def test_the_probe_reads_only_the_images_file(box, agent, used):
    from benchflow.sandbox.lockdown import _image_pytest_ini

    _image(box)
    _sh(box, agent)

    ini = await _image_pytest_ini(ContainerEnv(box), "/testbed")

    if not used:
        assert ini is None
        return
    assert ini is not None
    folder, base = ini.path.rsplit("/", 1)
    assert folder == lockdown._PYTEST_CONFIG_DIR and COPY.fullmatch(base)
    assert ini.plugins == ("pytester",)
    assert ini.addopts == "-ra --strict-markers"
    assert _sh(box, f"cat {ini.path}") == TOX_INI
    assert _sh(box, f"stat -c '%u %a' {ini.path}").split() == ["0", "644"]


@pytest.mark.asyncio
async def test_the_probe_follows_pytests_order(box):
    """With no pytest table in tox.ini either, setup.cfg's [tool:pytest] counts."""
    from benchflow.sandbox.lockdown import ImagePytestIni, _image_pytest_ini

    _image(box, dict(IMAGE_FILES, **{"tox.ini": "[tox]\nenvlist = py\n"}))

    ini = await _image_pytest_ini(ContainerEnv(box), "/testbed")

    assert ini is not None and ini.path.endswith(".cfg")
    assert ini == ImagePytestIni(path=ini.path)
    assert _sh(box, "ls -A /testbed").split() == sorted(IMAGE_FILES)
    assert _sh(box, f"cat {ini.path}") == IMAGE_FILES["setup.cfg"]


# -- a pytest before 3.5 has no --rootdir (astropy-7166, R334) --------


def _with_pytests(pytests):
    from benchflow.sandbox._image_manifest import ImageManifest

    return SimpleNamespace(
        _benchflow_image_manifest=ImageManifest(taken="pre-transfer", pytests=pytests)
    )


@pytest.mark.parametrize(
    ("pytests", "passed"),
    [
        ({}, True),
        ({"/s/pytest-8.3.3.dist-info": "8.3.3"}, True),
        ({"/s/pytest-3.5.0.dist-info": "3.5.0"}, True),
        ({"/s/pytest-3.3.1-py3.6.egg-info": "3.3.1"}, False),
        (
            {
                "/s/pytest-3.3.1-py3.6.egg-info": "3.3.1",
                "/t/pytest-8.3.3.dist-info": "8.3.3",
            },
            True,
        ),
        ({"/s/pytest-dev.dist-info": "dev"}, True),
    ],
    ids=["no-pytest", "new", "3.5", "old", "mixed", "unreadable"],
)
def test_rootdir_is_left_out_only_when_every_image_pytest_predates_3_5(pytests, passed):
    assert lockdown._image_pytests_take_rootdir(_with_pytests(pytests)) is passed


def test_without_a_manifest_rootdir_is_passed():
    assert lockdown._image_pytests_take_rootdir(SimpleNamespace()) is True


def test_the_addopts_leave_rootdir_out_when_asked():
    from benchflow.sandbox.lockdown import ImagePytestIni, _build_pytest_addopts

    for ini in (None, ImagePytestIni(path="/cfg/x.ini")):
        kept = shlex.split(_build_pytest_addopts("/testbed", "", ini=ini))
        left = shlex.split(
            _build_pytest_addopts("/testbed", "", ini=ini, rootdir=False)
        )
        assert "--rootdir=/testbed" in kept
        assert not any(a.startswith("--rootdir") for a in left)
        assert (
            left[0] == "-c" and [a for a in kept if a != "--rootdir=/testbed"] == left
        )


@pytest.mark.asyncio
async def test_strict_leaves_rootdir_out_for_an_old_image_pytest(tmp_path):
    """The wiring: an image manifest that found only pytest 3.3.1."""
    from benchflow.sandbox._image_manifest import ImageManifest

    task = _task_with_package(tmp_path)
    env = _ini_answer({})
    env._benchflow_image_manifest = ImageManifest(
        taken="pre-transfer", pytests={"/s/pytest-3.3.1-py3.6.egg-info": "3.3.1"}
    )
    await harden_before_verify(
        env, task, sandbox_user="agent", workspace="/testbed", integrity="strict"
    )

    args = shlex.split(task.config.verifier.env["PYTEST_ADDOPTS"])
    assert args[:2] == ["-c", "/dev/null"]
    assert not any(a.startswith("--rootdir") for a in args)
    assert args[-2] == "-p" and args[-1].startswith("_runtime_guard_")
