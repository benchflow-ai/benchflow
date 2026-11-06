"""Guards V2 round 7 against the b139 review's findings at 41c0fe95.

Every plugin, image helper and distribution here is honest; a changed file
only leaves a dummy marker, so a test proves what ran, never an attack.
"""

import hashlib
import os
import shutil
import sysconfig

import pytest

from benchflow.sandbox import _verifier_runtime as runtime
from tests.test_verifier_runtime import wheel
from tests.test_verifier_runtime_r5 import PYTEST_PACKAGES, Tree


@pytest.fixture
def tree(tmp_path):
    return Tree(tmp_path)


def _image_workspace(tree, helper: str, dependency: str = "VALUE = 1\n"):
    """An unchanged image helper in the workspace with its baseline copy."""
    (tree.app / "image_helper.py").write_text(helper)
    (tree.app / "image_dep.py").write_text(dependency)
    snapshot = tree.root / "baseline"
    shutil.copytree(tree.app, snapshot)
    files = {
        p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in snapshot.rglob("*.py")
    }
    tree.run.write_text(
        tree.run.read_text().replace(
            "PATH_OPTIONS = (",
            f"WORKSPACE = {str(tree.app)!r}\nSNAPSHOT = {str(snapshot)!r}\n"
            f"IMAGE_FILES = {files!r}\nPATH_OPTIONS = (",
        )
    )
    (tree.tests / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(tree.app)!r})\n"
        "pytest_plugins = ['image_helper']\n"
    )


HELPERS = {
    "hook": (
        "def pytest_configure(config):\n"
        "    import image_dep\n"
        "    config.image_dep = image_dep.VALUE\n"
    ),
    "import_module": (
        "import importlib\n"
        "def pytest_configure(config):\n"
        "    config.image_dep = importlib.import_module('image_dep').VALUE\n"
    ),
    "fixture": (
        "import pytest\n"
        "@pytest.fixture\n"
        "def image_value():\n"
        "    import image_dep\n"
        "    return image_dep.VALUE\n"
    ),
}


@pytest.mark.parametrize("deferred", sorted(HELPERS))
@pytest.mark.parametrize("changed", [False, True])
def test_image_helper_deferred_imports_are_checked(tree, deferred, changed):
    """V2-R6-C1 (41c0fe95): a dependency an unchanged helper imports from a
    hook or fixture is checked like one its module body imports."""
    _image_workspace(tree, HELPERS[deferred])
    marker = tree.root / "changed-dependency-ran"
    if changed:
        (tree.app / "image_dep.py").write_text(
            f"open({str(marker)!r}, 'w').close()\nVALUE = 1\n"
        )
    (tree.tests / "test_ok.py").write_text(
        "def test_ok(image_value):\n    assert image_value == 1\n"
        if deferred == "fixture"
        else "def test_ok(request):\n    assert request.config.image_dep == 1\n"
    )
    result = tree.pytest("test_ok.py")
    assert not marker.exists(), result.stdout + result.stderr
    assert (result.returncode != 0) == changed, result.stdout + result.stderr


def test_tests_still_import_the_submitted_code_beside_a_helper(tree):
    """V2-R6-C1's control: only helper code is held to the baseline; a test
    importing the submitted module gets the submitted module."""
    (tree.app / "solution.py").write_text("ANSWER = 0\n")  # in the baseline
    _image_workspace(tree, HELPERS["hook"])
    (tree.app / "solution.py").write_text("ANSWER = 42\n")
    (tree.tests / "test_ok.py").write_text(
        "def test_ok(request):\n"
        "    import solution\n"
        "    assert solution.ANSWER == 42\n"
        "    assert request.config.image_dep == 1\n"
    )
    result = tree.pytest("test_ok.py")
    assert result.returncode == 0, result.stdout + result.stderr


GRADING = (
    "class Checks:\n"
    "    def __init__(self, answer):\n"
    "        self.answer = answer\n"
    "    def pytest_runtest_call(self, item):\n"
    "        with open(self.answer) as handle:\n"
    "            assert handle.read().strip() == 'correct', 'dummy grading hook'\n"
    "checks = Checks(__ANSWER__)\n"
)


def _rewrite_runner(tree, plugins, top):
    paths = sysconfig.get_paths()
    stdlib = [paths["stdlib"], os.path.join(paths["platstdlib"], "lib-dynload")]
    tree.run.write_text(
        runtime.runner_source(
            site=str(tree.site),
            stdlib=[p for p in dict.fromkeys(stdlib) if os.path.isdir(p)],
            tests=str(tree.tests),
            plugins=plugins,
            allowed_top=[top],
            protected_top=[*PYTEST_PACKAGES, top, "bf_dep_mod"],
        )
    )


def test_pinned_object_entry_point_keeps_its_attribute():
    """V2-R6-C2 (41c0fe95): ``grade = grading:checks`` names an object."""
    data = wheel("grade-plugin", "1.0", entry="grade = grading:checks [extra]")
    assert runtime.wheel_metadata(data)[1] == {"grade": "grading:checks"}
    data = wheel("grade-plugin", "1.0", entry="grade = grading")
    assert runtime.wheel_metadata(data)[1] == {"grade": "grading"}


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_pinned_object_entry_point_grades(tree, agent):
    """V2-R6-C2 (41c0fe95): the runner registers the entry point's object,
    so the grading hook it carries runs; the nop does not pass."""
    answer = tree.root / "answer"
    answer.write_text("correct\n" if agent == "oracle" else "absent\n")
    source = GRADING.replace("__ANSWER__", repr(str(answer)))
    data = wheel(
        "grade-plugin", "1.0", entry="grade = grading:checks", module="grading"
    )
    (tree.site / "grading.py").write_text(source)
    info = tree.site / "grade_plugin-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: grade-plugin\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text("[pytest11]\ngrade = grading:checks\n")
    _rewrite_runner(tree, runtime.wheel_metadata(data)[1], "grading")
    (tree.tests / "test_delegated.py").write_text("def test_delegated():\n    pass\n")
    result = tree.pytest("test_delegated.py")
    expected = 0 if agent == "oracle" else 1
    assert result.returncode == expected, result.stdout + result.stderr


@pytest.mark.parametrize(
    "requirement, passes", [("fakeplug==1.0", True), ("fakeplug>=2", False)]
)
def test_required_plugins_sees_the_pinned_release(tree, requirement, passes):
    """V2-R6-C4 (41c0fe95): pytest's ``required_plugins`` finds the pinned
    release the runner loaded, and still refuses a version it is not."""
    (tree.tests / "test_ok.py").write_text(
        "def test_ok(request):\n    assert request.config.fakeplug\n"
    )
    result = tree.pytest("-o", f"required_plugins={requirement}", "test_ok.py")
    assert (result.returncode == 0) == passes, result.stdout + result.stderr
    if not passes:
        assert "Missing required plugins" in result.stdout + result.stderr


def test_runtime_metadata_wins_over_the_image_copy(tree):
    """V2-R6-C5 (41c0fe95): the image has its own pytest release; metadata
    for pytest is the protected runtime's, the code that runs."""
    import pytest as running

    info = tree.site / f"pytest-{running.__version__}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: pytest\nVersion: {running.__version__}\n")
    image = tree.root / "copied-image-site"
    (image / "pytest-7.0.0.dist-info").mkdir(parents=True)
    (image / "pytest-7.0.0.dist-info/METADATA").write_text(
        "Name: pytest\nVersion: 7.0.0\n"
    )
    (image / "image_only-1.0.dist-info").mkdir()
    (image / "image_only-1.0.dist-info/METADATA").write_text(
        "Name: image-only\nVersion: 1.0\n"
    )
    tree.run.write_text(
        tree.run.read_text().replace(
            "PATH_OPTIONS = (",
            f"IMAGE_SITES = [{str(image)!r}]\nPATH_OPTIONS = (",
        )
    )
    (tree.tests / "test_ok.py").write_text(
        "import importlib.metadata as metadata\n"
        "import pytest\n"
        "def test_ok():\n"
        "    assert metadata.version('pytest') == pytest.__version__\n"
        "    first = next(d for d in metadata.distributions()\n"
        "                 if d.metadata['Name'] == 'pytest')\n"
        "    assert first.version == pytest.__version__\n"
        "    assert metadata.version('image-only') == '1.0'\n"
    )
    result = tree.pytest("test_ok.py")
    assert result.returncode == 0, result.stdout + result.stderr
