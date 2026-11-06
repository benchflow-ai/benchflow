"""Guards V2-on-PT's reviews (V2PT-1..3 at 15ae58f2, V2PT-R2-1 at e1659a42).

Every plugin, helper and module here is honest; a changed file only leaves a
dummy marker or holds a dummy answer.
"""

import pytest

from benchflow.sandbox import _verifier_runtime as runtime
from tests.test_verifier_runtime_r5 import Tree
from tests.test_verifier_runtime_r7 import HELPERS, _image_workspace


@pytest.fixture
def tree(tmp_path):
    return Tree(tmp_path)


SOLUTION_FIXTURE = (
    "import pytest\n"
    "@pytest.fixture\n"
    "def solution():\n"
    "    import solution\n"
    "    return solution\n"
)


def _tests_imports(tree):
    """Write the runner's tests-import set as BenchFlow builds it: a static
    scan of the trusted tests folder (V2PT-R2-1)."""
    names = runtime.scan_tests_imports(tree.tests)
    tree.run.write_text(
        tree.run.read_text().replace(
            "PATH_OPTIONS = (",
            f"TESTS_IMPORTS = frozenset({names!r})\nPATH_OPTIONS = (",
        )
    )


@pytest.mark.parametrize("module", ["baseline", "new"])
def test_a_helper_fixture_hands_tests_the_submitted_code(tree, module):
    """V2PT-1: an unchanged image helper's fixture imports the submission for
    a test that also imports it itself; the test gets the submitted module,
    whether the agent changed a baseline file or wrote a new one."""
    if module == "baseline":
        (tree.app / "solution.py").write_text("ANSWER = 0\n")  # in the baseline
    _image_workspace(tree, SOLUTION_FIXTURE)
    (tree.app / "solution.py").write_text("ANSWER = 42\n")
    (tree.tests / "test_ok.py").write_text(
        "def test_ok(solution):\n"
        "    import solution as direct  # after the fixture's import\n"
        "    assert solution.ANSWER == 42 and direct is solution\n"
    )
    _tests_imports(tree)
    result = tree.pytest("test_ok.py")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["fixture", "hook"])
def test_a_changed_module_the_tests_never_import_is_refused_for_a_helper(tree, kind):
    """V2PT-R2-1: hooks and fixtures alike. A helper importing a workspace
    module the agent changed, which the trusted tests do not import
    themselves, is refused before that code runs, with the reason; the
    known limitation of a helper handing tests a submission they never
    import (V2-r4.md)."""
    helper = SOLUTION_FIXTURE if kind == "fixture" else HELPERS["hook"]
    module = "solution" if kind == "fixture" else "image_dep"
    (tree.app / "solution.py").write_text("ANSWER = 0\n")
    _image_workspace(tree, helper)
    marker = tree.root / "changed-module-ran"
    (tree.app / f"{module}.py").write_text(
        f"open({str(marker)!r}, 'w').close()\nANSWER = 42\nVALUE = 1\n"
    )
    (tree.tests / "test_ok.py").write_text(
        "def test_ok(solution):\n    assert solution.ANSWER == 42\n"
        if kind == "fixture"
        else "def test_ok(request):\n    assert request.config.image_dep == 1\n"
    )
    _tests_imports(tree)
    result = tree.pytest("test_ok.py")
    assert not marker.exists(), result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr
    assert "the task's tests do not import it themselves" in (
        result.stdout + result.stderr
    )


def test_the_tests_import_scan_reads_names_without_running_them(tmp_path):
    """V2PT-R2-1: ``import``/``from`` names of the tests, parents included;
    relative imports and a file that does not parse add nothing."""
    (tmp_path / "test_a.py").write_text(
        "import os.path\nfrom pkg.sub import solution\nfrom . import helper\n"
        "def f():\n    import late\n"
    )
    (tmp_path / "test_b.py").write_text("def broken(:\n")
    names = runtime.scan_tests_imports(tmp_path)
    assert {"os", "os.path", "pkg", "pkg.sub", "pkg.sub.solution", "late"} <= set(names)
    assert "helper" not in names
    assert runtime.scan_tests_imports(tmp_path / "absent") == []


def test_the_runner_carries_the_guards_entry_point_code():
    """V2PT-2/3: one loader. The runner's text holds the guard's block verbatim."""
    from pathlib import Path

    from benchflow.sandbox import _pytest_plugin_guard as guard

    block = runtime.entry_points_source()
    assert block in Path(guard.__file__).read_text()
    source = runtime.runner_source(
        site="/s", stdlib=[], tests="/t", plugins={}, allowed_top=[], protected_top=[]
    )
    assert block in source
    assert "def _record_distribution" not in source


# --- The guard path: test.sh runs its own pytest (V2PT-2, V2PT-3) ---------

GRADING = (
    "class Checks:\n"
    "    def pytest_runtest_call(self, item):\n"
    "        with open(__ANSWER__) as handle:\n"
    "            assert handle.read().strip() == 'correct', 'dummy grading hook'\n"
    "checks = Checks()\n"
)


def _pinning(monkeypatch, pin):
    from benchflow.task.config import TaskConfig
    from tests import test_verifier_uv_state as uv_state

    original = uv_state.make_task

    def make_task(layout, *args, env=None, **kwargs):
        task = original(layout, *args, env=env, **kwargs)
        task.config = TaskConfig.model_validate_toml(
            f'version = "1.0"\n[verifier]\npytest_plugin_releases = ["{pin}"]\n'
        )
        task.config.verifier.env = {"HOME": str(layout.home), **(env or {})}
        return task

    monkeypatch.setattr(uv_state, "make_task", make_task)


@pytest.mark.parametrize("agent", ["oracle", "nop"])
async def test_a_pinned_object_entry_point_grades_on_the_guard_path(
    tmp_path, monkeypatch, agent
):
    """V2PT-2: ``grade = grading:checks`` registers ``checks`` when test.sh
    runs its own pytest; its grading hook passes the oracle and fails the nop."""
    from tests.test_verifier_uv_state import PASSING, score

    answer = tmp_path / "answer"
    answer.write_text("correct\n" if agent == "oracle" else "absent\n")
    _pinning(monkeypatch, "grade-plugin==1.0")
    grading = GRADING.replace("__ANSWER__", repr(str(answer)))
    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        pinned={"grade-plugin==1.0": {"grade": ("grading", grading, "grading:checks")}},
    )
    assert error is None, error
    assert rewards == {"reward": 1.0 if agent == "oracle" else 0.0}


@pytest.mark.parametrize(
    "requirement, reward", [("ctrf_model==0.3.5", 1.0), ("ctrf_model>=9", 0.0)]
)
async def test_required_plugins_finds_the_pinned_release_on_the_guard_path(
    tmp_path, monkeypatch, requirement, reward
):
    """V2PT-3: pytest's ``required_plugins`` sees the pinned release the guard
    registered, and still refuses a version it is not."""
    from tests.test_verifier_uv_state import CTRF_PLUGIN, PASSING, score

    _pinning(monkeypatch, "ctrf-model==0.3.5")

    def leave(layout):
        layout.pytest_args = ("-o", f"required_plugins={requirement}")

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        before=leave,
        published=False,
        pinned={
            "ctrf-model==0.3.5": {
                "ctrf": ("ctrf_model", CTRF_PLUGIN, "ctrf_model", "0.3.5")
            }
        },
    )
    assert error is None, error
    assert rewards == {"reward": reward}


# --- Round 4: the lows of the r3 review (L1, L3, L5) -----------------------


def test_string_imports_the_tests_make_themselves_count(tmp_path):
    """L1: ``pytest.importorskip``, ``importlib.import_module`` and
    ``__import__`` with a literal name are the tests' own imports; a computed
    or relative name is not."""
    (tmp_path / "test_a.py").write_text(
        "import importlib, pytest\n"
        "solution = pytest.importorskip('solution')\n"
        "helper = importlib.import_module('pkg.helper')\n"
        "plain = __import__('plain_mod')\n"
        "name = 'computed'\n"
        "dynamic = importlib.import_module(name)\n"
        "rel = importlib.import_module('.sibling', 'pkg')\n"
    )
    names = set(runtime.scan_tests_imports(tmp_path))
    assert {"solution", "pkg", "pkg.helper", "plain_mod"} <= names
    assert "computed" not in names and ".sibling" not in names


def test_a_helper_fixture_serves_a_submission_the_tests_importorskip(tree):
    """L1 through the runner: the tests take the submission with
    ``pytest.importorskip``; the helper fixture's import of the changed
    module is the tests' own, so the honest run is graded."""
    (tree.app / "solution.py").write_text("ANSWER = 0\n")  # in the baseline
    _image_workspace(tree, SOLUTION_FIXTURE)
    (tree.app / "solution.py").write_text("ANSWER = 42\n")
    (tree.tests / "test_ok.py").write_text(
        "import pytest\n"
        "def test_ok(solution):\n"
        "    direct = pytest.importorskip('solution')\n"
        "    assert solution.ANSWER == 42 and direct is solution\n"
    )
    _tests_imports(tree)
    result = tree.pytest("test_ok.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_an_unreadable_tests_file_does_not_stop_preparation(tmp_path, monkeypatch):
    """L5: a tests file too deeply nested for ``ast.parse`` adds nothing and
    is listed as unread, instead of preparation stopping on the error."""
    import ast

    (tmp_path / "test_ok.py").write_text("import solution\n")
    (tmp_path / "test_deep.py").write_text("DEEP = 1  # dummy: parse fails\n")
    real = ast.parse

    def parse(source, *args, **kwargs):
        if isinstance(source, bytes) and b"DEEP" in source:
            raise RecursionError("dummy: maximum recursion depth exceeded")
        return real(source, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", parse)
    assert "solution" in runtime.scan_tests_imports(tmp_path)
    assert runtime.scan_tests(tmp_path) == (["solution"], ["test_deep.py"])


def test_a_refusal_names_the_tests_files_benchflow_could_not_read(tree):
    """L5: when the scan could not read a tests file, the refusal says so."""
    (tree.app / "solution.py").write_text("ANSWER = 0\n")
    _image_workspace(tree, SOLUTION_FIXTURE)
    (tree.app / "solution.py").write_text("ANSWER = 42\n")
    (tree.tests / "test_ok.py").write_text(
        "def test_ok(solution):\n    assert solution.ANSWER == 42\n"
    )
    tree.run.write_text(
        tree.run.read_text().replace(
            "PATH_OPTIONS = (", "TESTS_UNREAD = ['test_deep.py']\nPATH_OPTIONS = ("
        )
    )
    result = tree.pytest("test_ok.py")
    assert result.returncode != 0, result.stdout + result.stderr
    assert "could not read these tests files" in result.stdout + result.stderr
    assert "test_deep.py" in result.stdout + result.stderr


async def test_prepare_writes_the_tests_own_imports_into_the_runner(
    tmp_path, monkeypatch
):
    """L3: the real wiring. ``prepare_verifier_runtime`` scans the task's
    own tests folder and the runner it uploads holds what they import."""
    from pathlib import Path
    from types import SimpleNamespace

    from tests._receipt_fakes import receipt_aware
    from tests.test_verifier_runtime_r5 import _task

    task = _task(
        tmp_path,
        'version = "1.0"\n[verifier]\npytest_plugin_releases = ["fakeplug==1.0"]\n',
    )
    (task.paths.tests_dir / "test_outputs.py").write_text(
        "import pytest\nimport answer_mod\nsolution = pytest.importorskip('solution')\n"
    )
    # The runner is built for a task whose tests start it (ACC-RUN2).
    (task.paths.tests_dir / "test.sh").write_text(
        '"$BENCHFLOW_PYTEST" /tests/test_outputs.py\n'
    )
    interpreter = runtime.Interpreter(
        exe="/opt/runtime/rt/py/python3",
        version=(3, 12),
        machine="x86_64",
        glibc=(2, 39),
        musl=None,
        markers={},
    )

    async def probe(env, service="main"):
        return interpreter

    monkeypatch.setattr(runtime, "probe_interpreter", probe)
    monkeypatch.setattr(
        runtime,
        "resolve",
        lambda *args, **kwargs: runtime.Resolution(
            wheels=[], plugins={}, plugin_top=[], runner=True
        ),
    )
    uploaded = {}

    class Uploaded(Exception):
        pass

    class Sandbox:
        @receipt_aware
        async def exec(self, command, **kwargs):
            return SimpleNamespace(stdout="", stderr="", return_code=0)

        async def upload_dir(self, tree, staging, service="main", **_kwargs):
            uploaded["run"] = (Path(tree) / "run.py").read_text()
            raise Uploaded

    with pytest.raises(Uploaded):
        await runtime.prepare_verifier_runtime(Sandbox(), task, tests_dir="/tests")
    line = next(
        ln for ln in uploaded["run"].splitlines() if ln.startswith("TESTS_IMPORTS =")
    )
    assert "'answer_mod'" in line and "'solution'" in line and "'pytest'" in line
