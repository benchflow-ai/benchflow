"""A pytest before 4.4 gets the guard's optional plugins as it loads entry points, and the guard's own failure there is recorded (R334).

astropy-7166 (Python 3.6, pytest 3.3.1) scored a silent 0 under
``--integrity strict``. Discovery reads entry_points.txt on Python 3.6, so the
armed guard had the image's plugins as optional ones, and it loaded each with
``import_plugin(<entry point name>)``. Before pytest 4.4 that imports a module
of that name, and hypothesis's ``hypothesispytest`` is
``hypothesis.extra.pytestplugin``: pytest stopped before it read its
arguments, the guard took the ImportError for the plugin's own and left no
record, and the verifier scored test.sh's 0. The guard now loads such a plugin
as that pytest loads an entry point (its module, the object it names,
registered under its name). A failure of the guard's own there (Python finding
no module the guard asked for, or any exception outside the plugin's own
import) leaves the ``guard_error`` verdict and the ``crashed`` marker. Only the
plugin's own code failing as it imports stays that plugin's failure, as with
``-p``.

The container tests run the armed guard in a real pytest 3.3.1 on Python 3.6,
an image built from python:3.6-slim, with the plugins installed as the
image's.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox import lockdown
from tests.exec_diet_fakes import docker_available


def _raised_as_pytest_331_does(missing: str) -> ImportError:
    """pytest 3.3.1's import_plugin: the ImportError it raises while handling
    Python's ModuleNotFoundError for *missing*."""
    try:
        try:
            raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
        except ImportError as error:
            # As pytest does: raised while handling, never from it.
            raise ImportError(  # noqa: B904
                f'Error importing plugin "{missing}": {error.args[0]}'
            )
    except ImportError as raised:
        return raised


def test_asked_is_the_module_and_every_package_above_it():
    assert guard._asked("hypothesis.extra.pytestplugin", "hypothesispytest") == {
        "hypothesis",
        "hypothesis.extra",
        "hypothesis.extra.pytestplugin",
        "hypothesispytest",
    }


def test_a_module_the_guard_asked_for_and_python_lacks_is_a_guard_error(
    monkeypatch,
):
    """R334's failure: pytest 3.3.1 imported the entry point's name. That is
    the guard's own failure, never the plugin's."""
    monkeypatch.setattr(guard, "_OPTIONAL_FAILURE", None)
    exc = _raised_as_pytest_331_does("hypothesispytest")
    path = "/sp/hypothesis-3.44.2.dist-info/entry_points.txt"
    with pytest.raises(guard.GuardError) as raised:
        guard._optional_failure(exc, guard._asked("hypothesispytest"), path)
    assert raised.value.path == path
    assert raised.value.cause == (
        'ImportError: Error importing plugin "hypothesispytest": No module '
        "named 'hypothesispytest'"
    )
    assert raised.value.__cause__ is exc
    assert guard._OPTIONAL_FAILURE is None


def test_the_plugins_own_failure_stays_the_plugins(monkeypatch):
    """The plugin's code ran and imported a module Python lacks: the
    plugin's failure, as with ``-p``. Nothing is raised here; the caller
    re-raises it, and _stopping leaves no record of it."""
    monkeypatch.setattr(guard, "_OPTIONAL_FAILURE", None)
    exc = _raised_as_pytest_331_does("nosuchdep_bf")
    guard._optional_failure(exc, guard._asked("fakeplug.extra.pytestplugin"), "p")
    assert guard._OPTIONAL_FAILURE is exc
    marks = []
    monkeypatch.setattr(guard, "_mark", lambda kind, detail="": marks.append(kind))
    monkeypatch.setattr(guard, "_unmark", lambda kind: None)
    guard._stopping(exc)
    assert marks == []


def test_a_refusal_while_a_plugin_loads_stays_a_refusal(monkeypatch):
    monkeypatch.setattr(guard, "_OPTIONAL_FAILURE", None)
    refusal = guard.Rejected("Verifier plugin trust rejected: x (y)")
    guard._optional_failure(refusal, {"x"}, "p")
    assert guard._OPTIONAL_FAILURE is None


def test_any_failure_of_the_guards_own_code_leaves_a_guard_error_and_crashed(
    monkeypatch,
):
    """Not only a GuardError: an exception from anywhere in the guard's own
    code (preparse, a hook, an optional plugin's loading) leaves the
    ``guard_error`` verdict with the ``crashed`` marker, so the verifier's
    record says the guard failed, never a silent 0."""
    marks = []
    monkeypatch.setattr(
        guard, "_mark", lambda kind, detail="": marks.append((kind, detail))
    )
    monkeypatch.setattr(guard, "_unmark", lambda kind: None)
    guard._stopping(TypeError("forced by the test"))
    assert [kind for kind, _ in marks] == ["verdicts", "crashed"]
    (entry,) = json.loads(marks[0][1])
    assert entry["kind"] == entry["verdict"] == "guard_error"
    assert entry["path"] == ""
    assert entry["why"] == "TypeError: forced by the test"
    assert "TypeError: forced by the test" in marks[1][1]


@pytest.mark.parametrize(
    ("folder", "name", "version"),
    [
        ("/sp/pytest_cov-2.5.1.dist-info", "pytest-cov", "2.5.1"),
        ("/sp/hypothesis-3.44.2-py3.6.egg-info", "hypothesis", "3.44.2"),
    ],
)
def test_a_distribution_is_named_as_pkg_resources_names_it(folder, name, version):
    dist = guard._Distribution(folder)
    assert (dist.project_name, dist.version, dist.location) == (name, version, "/sp")


OLD_PYTEST_IMAGE = "benchflow-test-py36-pytest331:1"
_DOCKERFILE = """\
FROM python:3.6-slim
RUN pip install --no-cache-dir --disable-pip-version-check pytest==3.3.1 \\
    pluggy==0.6.0 attrs==17.3.0 py==1.11.0 six==1.16.0 \\
 && python -c "import pytest; assert pytest.__version__ == '3.3.1'"
"""
NAME = "_bf_guard_oldpytest"


@pytest.fixture(scope="module")
def old_pytest_image():
    if not docker_available():
        pytest.skip("needs a Docker daemon")
    present = subprocess.run(
        ["docker", "image", "inspect", OLD_PYTEST_IMAGE], capture_output=True
    )
    if present.returncode != 0:
        built = subprocess.run(
            ["docker", "build", "-q", "-t", OLD_PYTEST_IMAGE, "-"],
            input=_DOCKERFILE,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if built.returncode != 0:
            pytest.skip(f"cannot build {OLD_PYTEST_IMAGE}: {built.stderr[-300:]}")
    return OLD_PYTEST_IMAGE


# The image's plugin, as hypothesis's: an entry point whose name is no module.
_IMAGE_PLUGIN = r"""
s=/usr/local/lib/python3.6/site-packages
mkdir -p $s/fakeplug/extra $s/fakeplug-1.0.dist-info
: > $s/fakeplug/__init__.py
: > $s/fakeplug/extra/__init__.py
printf 'import pytest\n\n\n@pytest.fixture\ndef fakeplug_answer():\n    return 42\n' > $s/fakeplug/extra/pytestplugin.py
printf '[pytest11]\nfakeplugpytest = fakeplug.extra.pytestplugin\n' > $s/fakeplug-1.0.dist-info/entry_points.txt
printf 'Metadata-Version: 2.1\nName: fakeplug\nVersion: 1.0\n' > $s/fakeplug-1.0.dist-info/METADATA
"""
# A second image plugin whose own code fails as it imports.
_BROKEN_PLUGIN = r"""
mkdir -p $s/brokenplug-1.0.dist-info
printf 'import nosuchdep_bf\n' > $s/brokenplug.py
printf '[pytest11]\nbrokenpytest = brokenplug\n' > $s/brokenplug-1.0.dist-info/entry_points.txt
printf 'Metadata-Version: 2.1\nName: brokenplug\nVersion: 1.0\n' > $s/brokenplug-1.0.dist-info/METADATA
"""
_RUN = rf"""
mkdir -p /tests /testbed /g/.markers
printf 'def test_answer(fakeplug_answer):\n    assert fakeplug_answer == 42\n' > /tests/test_answer.py
cp /src/{NAME}.py /g/{NAME}.py
sleep 1
printf '%s' "$GUARD_KEY" > /g/.key
cd /testbed
PYTHONPATH=/g PYTEST_ADDOPTS="-p {NAME}" python -m pytest -p no:cacheprovider /tests/test_answer.py 2>&1
echo "exit=$?"
for f in /g/.markers/*; do
  [ -e "$f" ] || continue
  echo "marker ${{f##*.}}"
  cut -f2- "$f"
done
"""


def _run(
    tmp_path: Path, image: str, setup: str, mutant: str = ""
) -> tuple[int, str, dict]:
    """The armed guard in pytest 3.3.1, as test.sh's pytest starts it; the
    exit status, the output, and each marker's body by kind."""
    source = lockdown._pytest_plugin_guard_source(
        NAME,
        ("/testbed",),
        [],
        markers_dir="/g/.markers",
        key_file="/g/.key",
        tests_dir="/tests",
        agent_gids=(1000,),
        optional=["fakeplugpytest"],
    )
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / f"{NAME}.py").write_text(source + mutant)
    done = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "0:0",
            "-e",
            "GUARD_KEY=" + "ab" * 32,
            "-v",
            f"{folder}:/src:ro",
            image,
            "sh",
            "-c",
            "set -e\n" + setup + "set +e\n" + _RUN,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = done.stdout
    assert "exit=" in out, (out, done.stderr)
    status = int(out.split("exit=", 1)[1].split()[0])
    markers = {}
    lines = out.split("exit=", 1)[1].splitlines()[1:]
    for index, line in enumerate(lines):
        if line.startswith("marker "):
            body = lines[index + 1] if index + 1 < len(lines) else '""'
            markers[line.split(" ", 1)[1]] = json.loads(body)
    return status, out, markers


def test_an_entry_point_plugin_loads_on_pytest_3_and_the_honest_run_passes(
    tmp_path, old_pytest_image
):
    """R334's astropy-7166 shape: the plugin's fixture is there, the test
    passes (test.sh's 1.0), the header lists the plugin's distribution as
    pytest 3 lists one it autoloaded, and the guard records no stop."""
    status, out, markers = _run(tmp_path, old_pytest_image, _IMAGE_PLUGIN)
    assert status == 0, out
    assert "1 passed" in out, out
    assert "plugins: fakeplug-1.0" in out, out
    assert "Error importing plugin" not in out, out
    # pytest 3 ran to its own cleanup: the guard's end-of-session record.
    assert set(markers) == {"finished"}, markers


@pytest.mark.parametrize(
    ("mutant", "why"),
    [
        # R334's bug put back: the entry point's name to import_plugin.
        (
            "\n\ndef _load_entry_point(manager, name):\n"
            "    manager.import_plugin(name)\n",
            'ImportError: Error importing plugin "fakeplugpytest": No module '
            "named 'fakeplugpytest'",
        ),
        # Any other failure of the guard's own code while the plugin loads.
        (
            "\n\ndef _bf_entry_object(module, attributes):\n"
            "    raise TypeError('forced by the test')\n",
            "TypeError: forced by the test",
        ),
    ],
    ids=["by-name", "guard-code"],
)
def test_the_guards_own_failure_on_pytest_3_is_recorded_not_a_silent_0(
    tmp_path, old_pytest_image, mutant, why
):
    """pytest stops before any test, and the guard leaves the guard_error
    verdict and the crashed marker, so the verifier records the guard's
    failure instead of scoring test.sh's 0."""
    status, out, markers = _run(tmp_path, old_pytest_image, _IMAGE_PLUGIN, mutant)
    assert status != 0, out
    assert {"verdicts", "crashed"} <= set(markers), (markers, out)
    assert not {"refused", "loading", "checking"} & set(markers), markers
    (entry,) = json.loads(markers["verdicts"])
    assert entry["kind"] == entry["verdict"] == "guard_error", entry
    assert entry["guard"] == NAME
    assert entry["python"].startswith("3.6."), entry
    assert entry["why"] == why, entry


def test_a_plugins_own_import_failure_on_pytest_3_stays_the_plugins(
    tmp_path, old_pytest_image
):
    """A second image plugin whose code imports a module Python lacks: pytest
    stops with that plugin's ImportError, as it would autoloading it, and the
    guard records no failure of its own (test.sh scores the run)."""
    status, out, markers = _run(
        tmp_path, old_pytest_image, _IMAGE_PLUGIN + _BROKEN_PLUGIN
    )
    assert status != 0, out
    assert "No module named 'nosuchdep_bf'" in out, out
    assert set(markers) == {"registered"}, markers
