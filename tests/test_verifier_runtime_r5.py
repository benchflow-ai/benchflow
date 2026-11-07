"""Guards V2 round 4's review (b81, Astra and Sol) for the protected runner.

Each runner test writes the real ``run.py`` (:func:`runner_source`) for a
tree in a folder of the test's (a site folder holding the running pytest,
a pinned plugin and its dependency), and starts it as the kit would:
``python -I -S -B``. The custody shell, the offline migration and the
per-service build are checked on their own.
"""

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _verifier_runtime as runtime
from benchflow.sandbox import lockdown
from benchflow.task.config import TaskConfig
from tests.test_verifier_runtime import wheel

PLUGIN = (
    "def pytest_configure(config):\n"
    "    config.fakeplug = True\n"
    "def pytest_collection_modifyitems(items):\n"
    "    import bf_dep_mod\n"
    "    bf_dep_mod.used()\n"
)
DEPENDENCY = "def used():\n    pass\n"
PYTEST_PACKAGES = (
    "pytest",
    "_pytest",
    "pluggy",
    "iniconfig",
    "packaging",
    "py",
    "pygments",
)


def hostile(proof: Path) -> str:
    """A module that leaves *proof* and turns every test into a pass."""
    return (
        "from pathlib import Path\n"
        f"Path({str(proof)!r}).touch()\n"
        "from _pytest.python import Function\n"
        "Function.runtest = lambda self: None\n"
        "def used():\n    pass\n"
    )


def _copy_package(name: str, site: Path) -> None:
    module = importlib.import_module(name)
    origin = Path(module.__file__)
    if origin.name == "__init__.py":
        shutil.copytree(
            origin.parent, site / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    else:
        shutil.copy2(origin, site / origin.name)


class Tree:
    """A protected tree in *root*: site, tests, the runner, a workspace."""

    def __init__(self, root: Path):
        self.root = root
        self.site = root / "site"
        self.tests = root / "tests"
        self.app = root / "app"
        for folder in (self.site, self.tests, self.app):
            folder.mkdir(parents=True)
        for name in PYTEST_PACKAGES:
            _copy_package(name, self.site)
        (self.site / "fakeplug_mod.py").write_text(PLUGIN)
        (self.site / "bf_dep_mod.py").write_text(DEPENDENCY)
        info = self.site / "fakeplug-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: fakeplug\nVersion: 1.0\n\n"
        )
        (info / "entry_points.txt").write_text("[pytest11]\nfakeplug = fakeplug_mod\n")
        paths = sysconfig.get_paths()
        stdlib = [paths["stdlib"], os.path.join(paths["platstdlib"], "lib-dynload")]
        self.run = root / "run.py"
        self.run.write_text(
            runtime.runner_source(
                site=str(self.site),
                stdlib=[p for p in dict.fromkeys(stdlib) if os.path.isdir(p)],
                tests=str(self.tests),
                plugins={"fakeplug": "fakeplug_mod"},
                allowed_top=["fakeplug_mod"],
                protected_top=[*PYTEST_PACKAGES, "fakeplug_mod", "bf_dep_mod"],
            )
        )

    def pytest(self, *args: str) -> subprocess.CompletedProcess:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
        return subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(self.run), *args],
            cwd=self.tests,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )


@pytest.fixture
def tree(tmp_path):
    return Tree(tmp_path)


def test_the_runner_scores_an_honest_run(tree):
    (tree.tests / "test_outputs.py").write_text(
        "def test_answer(request):\n    assert request.config.fakeplug\n"
    )
    result = tree.pytest("test_outputs.py")
    assert result.returncode == 0, result.stdout + result.stderr
    (tree.tests / "test_outputs.py").write_text(
        "def test_answer():\n    assert False\n"
    )
    assert tree.pytest("test_outputs.py").returncode == 1


def test_a_conftest_plugin_from_the_workspace_is_refused_before_it_runs(tree):
    """V2-R4-1 (Astra): the trusted conftest puts /app on sys.path and
    declares ``verifier_helpers``; the agent replaced it with a module that
    turns every test into a pass. The runner refused nothing and imported
    it."""
    proof = tree.root / "helper-imported"
    (tree.app / "verifier_helpers.py").write_text(hostile(proof))
    (tree.tests / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(tree.app)!r})\n"
        "pytest_plugins = ['verifier_helpers']\n"
    )
    (tree.tests / "test_outputs.py").write_text(
        "def test_answer():\n    assert False\n"
    )

    result = tree.pytest("test_outputs.py")

    assert not proof.exists()
    assert result.returncode != 0
    assert "is not one of the task's pinned plugin releases" in (
        result.stdout + result.stderr
    )


def test_a_plugins_dependency_never_comes_from_the_workspace(tree):
    """V2-R4-2 (Astra): a protected plugin imports a dependency after a
    conftest put /app first on sys.path; the agent's module of that name
    must not be the one imported."""
    proof = tree.root / "dependency-imported"
    (tree.app / "bf_dep_mod.py").write_text(hostile(proof))
    (tree.tests / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(tree.app)!r})\n"
    )
    (tree.tests / "test_outputs.py").write_text(
        "def test_answer():\n    assert False\n"
    )

    result = tree.pytest("test_outputs.py")

    assert not proof.exists()
    assert result.returncode == 1, result.stdout + result.stderr


def test_the_standard_library_never_comes_from_the_workspace(tree):
    """Sol V2-R4-1: a module of the standard library the agent left in /app
    must not replace the protected one, whatever sys.path says."""
    proof = tree.root / "stdlib-imported"
    (tree.app / "colorsys.py").write_text(hostile(proof))
    (tree.tests / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(tree.app)!r})\n"
    )
    (tree.tests / "test_outputs.py").write_text(
        "def test_answer():\n    import colorsys\n    assert colorsys.rgb_to_hsv\n"
        "    assert False\n"
    )

    result = tree.pytest("test_outputs.py")

    assert not proof.exists()
    assert result.returncode == 1, result.stdout + result.stderr


def test_a_pinned_plugin_declared_again_by_its_module_is_loaded_once(tree):
    """V2-R4-4 (Astra), Sol V2-R4-2: ``-p fakeplug`` registers the module as
    ``fakeplug``; a conftest's ``pytest_plugins = ["fakeplug_mod"]`` asked
    for it again and pluggy stopped pytest before any test ran."""
    (tree.tests / "conftest.py").write_text("pytest_plugins = ['fakeplug_mod']\n")
    (tree.tests / "test_outputs.py").write_text(
        "def test_answer(request):\n    assert request.config.fakeplug\n"
    )

    result = tree.pytest("test_outputs.py")

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_runner_runs_from_the_root_with_the_callers_paths(tree, tmp_path):
    (tree.tests / "test_outputs.py").write_text(
        "import os\ndef test_answer():\n    assert os.getcwd() == '/'\n"
    )
    report = tree.tests / "out" / "report.xml"
    report.parent.mkdir()
    result = tree.pytest("test_outputs.py", "--junitxml", "out/report.xml")
    assert result.returncode == 0, result.stdout + result.stderr
    assert report.exists()


def _sums(folder: Path) -> None:
    lines = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        relative = "./" + path.relative_to(folder).as_posix()
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    (folder / "SHA256SUMS").write_text("".join(lines))


@pytest.mark.skipif(shutil.which("sha256sum") is None, reason="needs coreutils")
def test_custody_refuses_a_changed_or_added_file(tmp_path):
    """Sol V2-R4-1: the runner's launch check holds the tree (and C8's kit)
    to its manifest: a file changed after the build, or added beside it (a
    module or a byte-code cache the runner would import), refuses the run."""
    folder = tmp_path / "tree"
    (folder / "lib").mkdir(parents=True)
    (folder / "lib/random.py").write_text("SEED = 1\n")
    _sums(folder)
    check = runtime._same_files_check(str(folder), str(folder / "SHA256SUMS"), "t")

    def run():
        return subprocess.run(["sh", "-c", check], capture_output=True, text=True)

    assert run().returncode == 0
    (folder / "lib/__pycache__").mkdir()
    (folder / "lib/__pycache__/random.cpython-312.pyc").write_bytes(b"x")
    assert run().returncode != 0
    shutil.rmtree(folder / "lib/__pycache__")
    assert run().returncode == 0
    (folder / "lib/random.py").write_text("import os\nos._exit(0)\n")
    assert run().returncode != 0


@pytest.mark.skipif(shutil.which("sha256sum") is None, reason="needs coreutils")
def test_the_kit_check_lets_unread_caches_be_but_no_module(tmp_path):
    """Root's own uses of C8's kit Python write byte code into the kit (its
    wrapper has no -B). The runner never reads a cache (-X pycache_prefix),
    so an added cache is let be; an added or changed module is not."""
    folder = tmp_path / "kit"
    (folder / "lib").mkdir(parents=True)
    (folder / "lib/glob.py").write_text("X = 1\n")
    lines = f"{hashlib.sha256(b'X = 1\n').hexdigest()}  ./lib/glob.py\n"
    (folder / "MANIFEST").write_text(lines)
    check = runtime._same_files_check(
        str(folder), str(folder / "MANIFEST"), "kit", caches_unread=True
    )

    def run():
        return subprocess.run(["sh", "-c", check], capture_output=True, text=True)

    (folder / "lib/__pycache__").mkdir()
    (folder / "lib/__pycache__/glob.cpython-312.pyc").write_bytes(b"x")
    assert run().returncode == 0
    (folder / "lib/evil.py").write_text("")
    assert run().returncode != 0
    (folder / "lib/evil.py").unlink()
    (folder / "lib/glob.py").write_text("X = 2\n")
    assert run().returncode != 0


def test_the_runner_script_checks_custody_before_it_starts(tmp_path):
    script = runtime.runner_script()
    assert script.startswith(f"#!{runtime.KIT_SH}\n")
    first_exec = script.index("exec ")
    assert script.index(f"PATH={runtime.KIT_BIN}") < first_exec
    assert script.index(f"cd {runtime.KIT}") < first_exec
    assert script.index(f"cd {runtime.RUNTIME}") < first_exec
    assert (
        f"exec {runtime.KIT_PYTHON} -B -X pycache_prefix={runtime.NO_CACHE} " in script
    )


def _legacy_cache(home: Path, name: str, version: str, data: bytes) -> None:
    info, files = runtime._wheel_files(data)
    folder = home / ".cache" / "benchflow" / "plugin-releases"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"pinned-{name}-{version}.json").write_text(
        json.dumps({"format": 2, "checked": 1.0, "releases": {info: [files]}})
    )


def test_an_offline_upgrade_keeps_a_cache_the_old_release_checked(
    tmp_path, monkeypatch
):
    """V2-R4-5 (Astra), Sol V2-R4-3: a host whose pre-0.8.4 cache checked
    pytest-timeout 2.3.1 and that keeps its wheel locally ran offline; the
    new cache was empty and the trial stopped before the agent."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("BENCHFLOW_PLUGIN_RELEASES", "offline")
    data = wheel(
        "pytest-timeout",
        "2.3.1",
        entry="timeout = pytest_timeout",
        module="pytest_timeout",
        source="X = 1\n",
    )
    _legacy_cache(home, "pytest-timeout", "2.3.1", data)
    wheels = tmp_path / "wheelhouse"
    wheels.mkdir()
    (wheels / "pytest_timeout-2.3.1-py3-none-any.whl").write_bytes(data)

    resolution = runtime.resolve(
        ["pytest-timeout==2.3.1"],
        ctrf=False,
        interpreter=None,
        wheelhouses=(str(wheels),),
    )

    assert [w.name for w in resolution.wheels] == ["pytest-timeout"]
    assert resolution.wheels[0].data == data
    # Kept in the new cache, so the next offline run needs no wheelhouse.
    again = runtime.resolve(["pytest-timeout==2.3.1"], ctrf=False, interpreter=None)
    assert again.wheels[0].data == data


def test_an_offline_upgrade_never_takes_a_wheel_the_old_cache_did_not_check(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("BENCHFLOW_PLUGIN_RELEASES", "offline")
    genuine = wheel("pytest-timeout", "2.3.1", module="pytest_timeout", source="X=1\n")
    _legacy_cache(home, "pytest-timeout", "2.3.1", genuine)
    other = wheel("pytest-timeout", "2.3.1", module="pytest_timeout", source="X=2\n")
    wheels = tmp_path / "wheelhouse"
    wheels.mkdir()
    (wheels / "pytest_timeout-2.3.1-py3-none-any.whl").write_bytes(other)

    with pytest.raises(runtime.RuntimeUnavailable, match="BENCHFLOW_PLUGIN_WHEELHOUSE"):
        runtime.resolve(
            ["pytest-timeout==2.3.1"],
            ctrf=False,
            interpreter=None,
            wheelhouses=(str(wheels),),
        )


def test_offline_without_benchflows_pytest_builds_the_plugins_without_a_runner(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("BENCHFLOW_PLUGIN_RELEASES", "offline")
    data = wheel("pytest-timeout", "2.3.1", module="pytest_timeout", source="X=1\n")
    _legacy_cache(home, "pytest-timeout", "2.3.1", data)
    wheels = tmp_path / "wheelhouse"
    wheels.mkdir()
    (wheels / "pytest_timeout-2.3.1-py3-none-any.whl").write_bytes(data)
    from tests.test_verifier_runtime import LINUX_312

    resolution = runtime.resolve(
        ["pytest-timeout==2.3.1"],
        ctrf=False,
        interpreter=LINUX_312,
        wheelhouses=(str(wheels),),
    )

    assert not resolution.runner
    assert [w.name for w in resolution.wheels] == ["pytest-timeout"]


def _task(tmp_path: Path, toml: str) -> SimpleNamespace:
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test.sh").write_text("pytest /tests/test_outputs.py\n")
    return SimpleNamespace(
        task_dir=tmp_path,
        paths=SimpleNamespace(tests_dir=tests, uses_native_verifier_dir=False),
        config=TaskConfig.model_validate_toml(toml),
    )


@pytest.mark.verifier_runtime
async def test_a_separate_verifier_builds_nothing_in_the_agents_sandbox(
    tmp_path, monkeypatch
):
    """V2-R4-6 (Astra): the solver's Python 3.9 decided whether the verifier
    image's Python 3.12 could load the pins, before the agent ran."""
    task = _task(
        tmp_path,
        'version = "1.0"\n[verifier]\nsandbox_mode = "separate"\n'
        'pytest_plugin_releases = ["pytest-asyncio==1.3.0"]\n',
    )
    calls = []

    async def prepare(*args, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(runtime, "prepare_verifier_runtime", prepare)
    env = SimpleNamespace()

    assert await lockdown.prepare_verifier_runtime(env, task, "agent") is None
    assert calls == []


@pytest.mark.verifier_runtime
async def test_a_target_service_verifier_builds_in_its_own_service(
    tmp_path, monkeypatch
):
    """Sol V2-R4-4: a ``grader`` service with Python 3.12 runs the verifier,
    but main's Python 3.9 decided the pins."""
    task = _task(
        tmp_path,
        'version = "1.0"\n[verifier]\nservice = "grader"\n'
        'pytest_plugin_releases = ["pytest==9.0.3"]\n',
    )
    probed = []

    class Sandbox:
        async def runtime_kit_for_route(self, service="main"):
            probed.append(service)
            return None

        async def exec(self, command, **kwargs):
            probed.append(("exec", kwargs.get("service")))
            return SimpleNamespace(stdout="", stderr="", return_code=3)

    def stop(*args, **kwargs):
        raise runtime.RuntimePinError("stop here")

    monkeypatch.setattr(runtime, "resolve", stop)
    with pytest.raises(runtime.RuntimePinError):
        await lockdown.prepare_verifier_runtime(Sandbox(), task, "agent")
    assert probed[0] == "grader"
    assert ("exec", "grader") in probed


@pytest.mark.parametrize(
    ("passing", "reward"), [(True, 1.0), (False, 0.0)], ids=["pass", "fail"]
)
async def test_a_pytest_the_guard_never_reached_is_scored_as_test_sh_wrote(
    tmp_path, monkeypatch, passing, reward
):
    """V2-R4-3 (Astra) refused every run in which no pytest signed in, on
    the reasoning that code before pytest (a .pth an install chose) had
    cleared PYTEST_ADDOPTS. A missing marker is no evidence of that: on
    Terminal-Bench 2's qemu tasks test.sh never reaches pytest, and the
    oracle and nop were both refused (GUARD-LOAD). Code that runs before
    pytest could as well keep the guard and set the result, and the task's
    script is trusted (contract B), so the reward is test.sh's."""
    from tests.test_verifier_uv_state import FAILING, PASSING, score

    def leave(layout):
        # test.sh's own environment for its pytest: the guard gone, ctrf kept.
        layout.pytest_env = {"PYTEST_ADDOPTS": "-p ctrf_model"}

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if passing else FAILING,
        before=leave,
        workspace_name="app",
    )

    assert error is None, error
    assert rewards == {"reward": reward}


def test_the_runner_registers_through_the_guards_marker_format():
    source = runtime.runner_source(
        site="/s",
        stdlib=["/l"],
        tests="/t",
        plugins={},
        allowed_top=[],
        protected_top=[],
    )
    assert 'mac + "\\t" + body' in source
    assert '_mark("registered")' in source
    assert "/_runtime_guard_" in source
    compile(source, "run.py", "exec")
