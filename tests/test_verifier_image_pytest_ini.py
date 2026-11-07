"""The verifier honours the image's own pytest configuration where pytest
itself would read it and while the agent left what pytest reads of it
unchanged (V1p's finding, review round 1: V3-R1-A-1, V3-R1-A-2).

The verifier ran pytest with ``-c /dev/null`` so that a configuration file the
agent wrote or edited is never read. That also hid the image's own: pytest-dev's
tox.ini ``[pytest] addopts = -ra -p pytester --strict-markers`` was dropped, so
every test that takes ``testdir`` errored with "fixture 'testdir' not found"
and the pytest-5809 and pytest-5840 oracles failed on 0.8 and on the fixes
alike. Hardening now finds the file pytest would read in the image's copy of
the workspace (``/testbed_verify``) and uses the workspace's one when its
pytest section or table is the image's. Its ``-p`` plugins move after
BenchFlow's plugin guard, which judges them: an ini's ``-p`` would load before
the guard. A task whose tests live in the verifier package (``pytest
/tests/test_outputs.py``) keeps ``-c /dev/null``: pytest never reads the
workspace's file for them.
"""

from __future__ import annotations

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


def _probe_stdout(marker: str, image: dict[str, str], work: dict[str, str]) -> str:
    """What the probe prints for these image and workspace files (kind "file",
    or the kind itself when the value is "<link>", "<other>", ...)."""
    out = ""
    for name in lockdown._PYTEST_CONFIG_NAMES:
        for side, files in (("image", image), ("work", work)):
            if name not in files:
                continue
            text = files[name]
            if text.startswith("<") and text.endswith(">"):
                out += f"{marker} {name} {side} {text[1:-1]}\n"
            else:
                out += f"{marker} {name} {side} file\n{text}\n"
    return out


def _ini_answer(image: dict[str, str], work: dict[str, str] | None = None):
    """A sandbox whose ini probe finds *image* in /testbed_verify and *work*
    (the same files by default) in the workspace; every other probe is quiet."""
    probes: list[str] = []

    def side_effect(cmd, **kwargs):
        if "bf-pytest-ini" in cmd:
            probes.append(cmd)
            marker = shlex.split(cmd)[-1]
            stdout = _probe_stdout(marker, image, image if work is None else work)
            return SimpleNamespace(stdout=stdout, stderr="", exit_code=0)
        return SimpleNamespace(stdout="", stderr="", exit_code=0)

    env = _make_env(side_effect=side_effect)
    env.probes = probes
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


@pytest.mark.asyncio
async def test_the_images_unchanged_ini_replaces_dev_null(tmp_path):
    """pytest-dev's tox.ini reaches pytest, -p pytester last."""
    task = _task_with_package(tmp_path)
    await harden_before_verify(
        _ini_answer({"tox.ini": TOX_INI}),
        task,
        sandbox_user="agent",
        workspace="/testbed",
    )

    args = shlex.split(task.config.verifier.env["PYTEST_ADDOPTS"])
    assert "/dev/null" not in args
    assert args[args.index("-c") + 1] == "/testbed/tox.ini"
    assert args[args.index("-o") + 1] == "addopts=-ra --strict-markers"
    # The ini's plugin comes after the guard, so the guard judges it.
    plugins = [args[i + 1] for i, arg in enumerate(args) if arg == "-p"]
    assert plugins[-1] == "pytester"
    assert plugins.index("pytester") > next(
        i for i, name in enumerate(plugins) if name.startswith("_runtime_guard_")
    )


@pytest.mark.asyncio
async def test_an_ini_without_plugins_is_passed_as_it_is(tmp_path):
    task = _task_with_package(tmp_path)
    await harden_before_verify(
        _ini_answer({"pytest.ini": "[pytest]\naddopts = -ra\nmarkers = slow\n"}),
        task,
        sandbox_user="agent",
        workspace="/app",
    )

    addopts = task.config.verifier.env["PYTEST_ADDOPTS"]
    assert addopts.startswith("-c /app/pytest.ini ")
    assert " -o " not in addopts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("image", "work"),
    [
        ({}, {}),  # the image has no configuration file
        ({"tox.ini": TOX_INI}, {}),  # the agent removed it
        ({"tox.ini": TOX_INI}, {"tox.ini": "<link>"}),  # or made it a link
        ({"tox.ini": "<open>"}, {"tox.ini": TOX_INI}),  # the agent could write the copy
        ({"tox.ini": "<big>"}, {"tox.ini": TOX_INI}),
        (
            {"tox.ini": TOX_INI},
            {"tox.ini": TOX_INI.replace("--strict-markers", "--deselect x")},
        ),
        # A second addopts in [pytest]: pytest's iniconfig refuses the file.
        ({"tox.ini": TOX_INI}, {"tox.ini": TOX_INI + "addopts = -p evil\n"}),
        ({"tox.ini": "[pytest]\naddopts = -vp evil\n"}, None),  # a hidden -p
        # pyproject.toml's table changed; tox.ini is not pytest's choice then.
        (
            {"pyproject.toml": "[tool.pytest.ini_options]\naddopts = '-q'\n"},
            {"pyproject.toml": "[tool.pytest.ini_options]\naddopts = '-p evil'\n"},
        ),
    ],
    ids=[
        "none",
        "removed",
        "link",
        "open-copy",
        "big",
        "section-edited",
        "duplicate-key",
        "bundled-p",
        "pyproject-table-edited",
    ],
)
async def test_anything_else_keeps_dev_null(tmp_path, image, work):
    task = _task_with_package(tmp_path)
    await harden_before_verify(
        _ini_answer(image, work), task, sandbox_user="agent", workspace="/testbed"
    )

    assert task.config.verifier.env["PYTEST_ADDOPTS"].startswith("-c /dev/null ")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "image", "work"),
    [
        (
            "tox.ini",
            TOX_INI,
            TOX_INI + "\n[testenv]\ndeps = hypothesis\ncommands = pytest\n",
        ),
        (
            "setup.cfg",
            "[metadata]\nname = x\n[tool:pytest]\naddopts = -p pytester\n",
            "[metadata]\nname = x\n[options]\ninstall_requires = attrs\n"
            "[tool:pytest]\naddopts = -p pytester\n",
        ),
        (
            "pyproject.toml",
            '[project]\nname = "x"\n[tool.pytest.ini_options]\naddopts = "-p pytester"\n',
            '[project]\nname = "x"\ndependencies = ["attrs"]\n\n'
            "[tool.pytest.ini_options]  # unchanged\naddopts = '-p pytester'\n",
        ),
    ],
    ids=["tox-testenv", "setup-cfg-options", "pyproject-dependencies"],
)
async def test_an_edit_outside_the_pytest_section_keeps_the_images_file(
    tmp_path, name, image, work
):
    """Guards V3-R1-A-2: the whole-file compare dropped the image's file for
    an edit pytest never reads (a gold patch adding a dependency), so two
    correct answers, or the oracle and the nop, ran under different
    configurations."""
    task = _task_with_package(tmp_path)
    await harden_before_verify(
        _ini_answer({name: image}, {name: work}),
        task,
        sandbox_user="agent",
        workspace="/testbed",
    )

    args = shlex.split(task.config.verifier.env["PYTEST_ADDOPTS"])
    assert args[args.index("-c") + 1] == f"/testbed/{name}"
    assert [args[i + 1] for i, arg in enumerate(args) if arg == "-p"][-1] == (
        "pytester"
    )


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

    await harden_before_verify(env, task, sandbox_user="agent", workspace="/app")

    assert task.config.verifier.env["PYTEST_ADDOPTS"].startswith("-c /dev/null ")
    assert env.probes == []


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
    )
    # The guard itself is not importable here; what comes before it is.
    addopts = task.config.verifier.env["PYTEST_ADDOPTS"]
    addopts = addopts.split(" -p _runtime_guard_")[0]
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
        _choose_unchanged_pytest_config,
        _parse_pytest_config_probe,
    )

    stdout = _probe_stdout(
        "M1",
        {"pytest.ini": "<link>", "tox.ini": TOX_INI},
        {"tox.ini": TOX_INI + "\n", "setup.cfg": "<other>"},
    )
    records = _parse_pytest_config_probe(stdout, "M1")
    assert records == {
        ("pytest.ini", "image"): ("link", ""),
        ("tox.ini", "image"): ("file", TOX_INI),
        ("tox.ini", "work"): ("file", TOX_INI + "\n"),
        ("setup.cfg", "work"): ("other", ""),
    }
    # The image's pytest.ini is a link: nothing is used.
    assert _choose_unchanged_pytest_config(records) is None
    del records["pytest.ini", "image"]
    assert _choose_unchanged_pytest_config(records) == ("tox.ini", TOX_INI)
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


def test_pytest_loads_the_inis_plugin_through_the_addopts(tmp_path):
    """The built PYTEST_ADDOPTS give a real pytest the ini's options and its
    plugin; ``-c /dev/null`` gives neither (the 0.8 failure)."""
    from benchflow.sandbox.lockdown import ImagePytestIni, _build_pytest_addopts

    project = tmp_path / "testbed"
    (project / "testing").mkdir(parents=True)
    (project / "tox.ini").write_text(TOX_INI)
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
            path=str(project / "tox.ini"),
            addopts="-ra --strict-markers",
            plugins=("pytester",),
        )
    )
    assert honoured.returncode == 0, honoured.stdout + honoured.stderr
    assert "1 passed" in honoured.stdout

    hidden = run(None)
    assert hidden.returncode != 0
    assert "fixture 'pytester' not found" in hidden.stdout


# The probe itself, in a container (skipped without Docker): root's
# /testbed_verify copy and the workspace as the agent left it.


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


def _sh(name: str, command: str) -> None:
    done = subprocess.run(
        ["docker", "exec", name, "sh", "-c", command], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


IMAGE_FILES = {
    # pytest would read tox.ini: pyproject.toml has no pytest table.
    "pyproject.toml": "[project]\nname = 'x'\n",
    "tox.ini": TOX_INI,
    "setup.cfg": "[tool:pytest]\naddopts = -q\n",
}


def _image(name: str) -> None:
    _sh(name, "mkdir -p /testbed")
    for file, text in IMAGE_FILES.items():
        _write(name, f"/testbed/{file}", text)
    _sh(name, "cp -a /testbed /testbed_verify && chmod 700 /testbed_verify")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        ("true", "/testbed/tox.ini"),
        # A pytest.ini only the agent added is not read; the image's tox.ini is.
        (
            "printf '[pytest]\\naddopts = -p evil\\n' > /testbed/pytest.ini",
            "/testbed/tox.ini",
        ),
        ("printf 'addopts = -p evil\\n' >> /testbed/tox.ini", None),
        ("rm /testbed/tox.ini && ln -s /testbed_verify/tox.ini /testbed/tox.ini", None),
        ("rm -r /testbed_verify", None),
        # V3-R1-A-2: tox's own sections are not what pytest reads.
        (
            "printf '\\n[testenv]\\ndeps = attrs\\n' >> /testbed/tox.ini",
            "/testbed/tox.ini",
        ),
        ("sed -i 's/--strict-markers/--deselect=x/' /testbed/tox.ini", None),
        # V3-R1-B-2: a copy the agent user could have written is not the image's.
        (
            "chmod 755 /testbed_verify && chmod 666 /testbed_verify/tox.ini",
            None,
        ),
        ("chmod 666 /testbed_verify/tox.ini", "/testbed/tox.ini"),
    ],
    ids=[
        "unchanged",
        "agent-added-pytest-ini",
        "edited",
        "replaced-by-link",
        "no-copy",
        "edited-outside-the-pytest-section",
        "pytest-section-edited",
        "copy-writable-by-others",
        "closed-copy",
    ],
)
async def test_the_probe_reads_only_the_images_unchanged_file(box, agent, expected):
    from benchflow.sandbox.lockdown import _image_pytest_ini

    _image(box)
    _sh(box, agent)

    ini = await _image_pytest_ini(ContainerEnv(box), "/testbed")

    assert (ini.path if ini else None) == expected
    if expected:
        assert ini is not None
        assert ini.plugins == ("pytester",)
        assert ini.addopts == "-ra --strict-markers"


@pytest.mark.asyncio
async def test_the_probe_follows_pytests_order(box):
    """With no pytest table in tox.ini either, setup.cfg's [tool:pytest] counts."""
    from benchflow.sandbox.lockdown import ImagePytestIni, _image_pytest_ini

    IMAGE = dict(IMAGE_FILES, **{"tox.ini": "[tox]\nenvlist = py\n"})
    _sh(box, "mkdir -p /testbed")
    for file, text in IMAGE.items():
        _write(box, f"/testbed/{file}", text)
    _sh(box, "cp -a /testbed /testbed_verify")

    ini = await _image_pytest_ini(ContainerEnv(box), "/testbed")

    assert ini == ImagePytestIni(path="/testbed/setup.cfg")
