"""Pytest plugins a task's test.sh installs are decided before the run.

The plugin guard never loads what test.sh installs, so a pinned install is
served from BenchFlow's protected runtime, and an unpinned one refuses the task
before any sandbox, unscored, with its reason; pytest-timeout alone is left
unloaded while nothing in the task's files uses it. The install lines below
are the registry tasks' own (skillsbench@1.0, terminal-bench-pro@1.0,
quixbugs@1.0).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.sandbox._verifier_runtime import runtime_requested
from benchflow.task import verifier_plugin_installs as vpi
from benchflow.task.task import Task
from tests import test_dry_run_capabilities as dry_run
from tests import test_verifier_runtime as vr

_task = dry_run._task
_same_in_dry_run_and_run = dry_run._same_in_dry_run_and_run
# The dry run tests' fixtures: `bench eval run` on tmp_path/tasks, and the run
# stopped where it would create its sandbox.
bench = dry_run.bench
sandbox_reached = dry_run.sandbox_reached
# A fake PyPI for the plugin releases' metadata (the runtime tests').
pypi = vr.pypi

# skillsbench@1.0 react-performance-debugging, tests/test.sh
REACT = """#!/bin/bash
if ! python3 -c 'import pytest_playwright, pytest_asyncio' 2>/dev/null; then
  python3 -m pip install --break-system-packages pytest==8.3.4 pytest-asyncio==0.24.0 \\
    httpx==0.28.1 pytest-playwright==0.6.2 playwright==1.49.1
  python3 -m playwright install chromium 2>/dev/null || true
fi
pytest /tests/test_outputs.py -v
"""
# skillsbench@1.0 crystallographic-wyckoff-position-analysis, tests/test.sh
CRYSTAL = """#!/bin/bash
# Install pytest for testing
pip3 install --break-system-packages pytest==8.3.4 pytest-json-report==1.5.0 --quiet || true
pytest /tests/test_outputs.py -v
"""
# terminal-bench-pro@1.0 fix-web-app-security-vulnerability, tests/test.sh
DJANGO = """#!/bin/bash
uv init
uv add pytest==8.4.1 django==4.2 pytest-django

# Set Django settings module
export DJANGO_SETTINGS_MODULE=config.settings
uv run pytest test_outputs.py -rA
"""
# quixbugs@1.0 quixbugs-python-*, tests/test.sh
QUIXBUGS = """#!/bin/bash
set -Eeuo pipefail
# Initialize uv and add pytest
uv init
uv add pytest pytest-timeout
if uv run pytest test_outputs.py -v -rA; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


def _make(root: Path, test_sh: str, *, verifier: str = "", extra: dict | None = None):
    task = _task(root, "t", verifier=verifier)
    (task / "tests" / "test.sh").write_text(test_sh)
    for name, text in (extra or {}).items():
        (task / "tests" / name).write_text(text)
    return Task(task)


def test_pinned_asyncio_and_playwright_are_served_from_the_runtime(tmp_path):
    task = _make(tmp_path, REACT)

    assert vpi.implied_pins(task) == [
        "pytest-asyncio==0.24.0",
        "pytest-playwright==0.6.2",
    ]
    assert vpi.refusal(task) is None
    pins, _ = runtime_requested(task)
    assert {"pytest-asyncio==0.24.0", "pytest-playwright==0.6.2"} <= set(pins)


def test_pinned_json_report_is_served_from_the_runtime(tmp_path):
    task = _make(tmp_path, CRYSTAL)

    assert vpi.implied_pins(task) == ["pytest-json-report==1.5.0"]
    assert vpi.refusal(task) is None
    assert "pytest-json-report==1.5.0" in runtime_requested(task)[0]


def test_unpinned_pytest_django_refuses_the_task(tmp_path):
    task = _make(tmp_path, DJANGO)

    why = vpi.refusal(task)

    assert why is not None and "tests/test.sh installs pytest-django unpinned" in why
    assert "pytest_plugin_releases" in why
    assert vpi.implied_pins(task) == []


# Plugin blocking is strict's alone (the lead's 09:58Z ruling): in the default,
# integrity audit, test.sh runs as its authors wrote it.
STRICT = ("--integrity", "strict")


def test_in_the_default_mode_an_unpinned_plugin_install_is_not_refused(
    tmp_path, bench, sandbox_reached
):
    task = _task(tmp_path / "tasks", "django")
    (task / "tests" / "test.sh").write_text(DJANGO)

    run = bench("docker", jobs="run")

    recorded = dry_run._recorded(tmp_path / "run")
    assert "verifier.pytest_plugin_releases" not in recorded, recorded
    assert "SandboxCreationReached" in recorded, (run.output, recorded)


def test_unpinned_django_is_refused_the_same_in_the_dry_run_and_the_run(
    tmp_path, bench, sandbox_reached
):
    task = _task(tmp_path / "tasks", "django")
    (task / "tests" / "test.sh").write_text(DJANGO)

    recorded = _same_in_dry_run_and_run(tmp_path, bench, "docker", "django", *STRICT)

    assert "verifier.pytest_plugin_releases" in recorded
    assert "pytest-django unpinned" in recorded


def test_inert_pytest_timeout_is_left_unloaded_and_named(tmp_path):
    """quixbugs: `uv add pytest pytest-timeout`, and nothing uses it."""
    task = _make(tmp_path, QUIXBUGS)

    assert vpi.refusal(task) is None
    assert vpi.inert_installs(task) == ["tests/test.sh: pytest-timeout"]
    assert vpi.INERT_WARNING == "pytest-timeout installed unpinned, not loaded (inert)"


def test_inert_quixbugs_like_task_passes_the_plan_gate(
    tmp_path, bench, sandbox_reached
):
    """It runs as before: the run goes on to its sandbox, nothing refuses it."""
    task = _task(tmp_path / "tasks", "quix")
    (task / "tests" / "test.sh").write_text(QUIXBUGS)

    dry = bench("docker", "--dry-run", jobs="dry")

    assert dry.exit_code == 0, dry.output
    assert "pytest_plugin_releases" not in dry.output


@pytest.mark.parametrize(
    ("change", "use"),
    [
        (
            {"test.sh": QUIXBUGS.replace("-v -rA", "-v -rA --timeout=60")},
            "passes --timeout",
        ),
        (
            {
                "test.sh": QUIXBUGS.replace(
                    "set -Eeuo", "export PYTEST_TIMEOUT=60\nset -Eeuo"
                )
            },
            "sets PYTEST_TIMEOUT",
        ),
        ({"pytest.ini": "[pytest]\ntimeout = 30\n"}, "sets a timeout ini key"),
        (
            {
                "test_outputs.py": "import pytest\n\n@pytest.mark.timeout(5)\ndef test_x():\n    pass\n",
                "pytest.ini": "[pytest]\naddopts = --strict-markers\n",
            },
            "@pytest.mark.timeout under strict markers",
        ),
    ],
    ids=["--timeout", "PYTEST_TIMEOUT", "ini-key", "mark+strict"],
)
def test_pytest_timeout_a_task_uses_is_refused(tmp_path, change, use):
    script = change.pop("test.sh", QUIXBUGS)
    task = _make(tmp_path, script, extra=change)

    why = vpi.refusal(task)

    assert why is not None and "pytest-timeout unpinned" in why and use in why, why


def test_a_timeout_mark_without_strict_markers_stays_inert(tmp_path):
    task = _make(
        tmp_path,
        QUIXBUGS,
        extra={
            "test_outputs.py": "import pytest\n\n@pytest.mark.timeout(5)\ndef test_x():\n    pass\n"
        },
    )

    assert vpi.refusal(task) is None
    assert vpi.inert_installs(task)


def test_ctrf_alone_needs_nothing(tmp_path):
    task = _make(
        tmp_path,
        "#!/bin/bash\nuv init\nuv add pytest pytest-json-ctrf\nuv run pytest --ctrf /logs/verifier/ctrf.json\n",
    )

    assert vpi.refusal(task) is None
    assert vpi.implied_pins(task) == []
    assert vpi.inert_installs(task) == []


def test_a_declared_pin_covers_an_unpinned_install(tmp_path):
    task = _make(
        tmp_path, DJANGO, verifier='pytest_plugin_releases = ["pytest-django==4.9.0"]\n'
    )

    assert vpi.refusal(task) is None
    assert vpi.implied_pins(task) == []


# -- the lead's (3): the dependency closure, before any sandbox ---------------

# pytest-json-report 1.5.0's Requires-Dist, as PyPI serves it.
JSON_REPORT_REQUIRES = ("pytest>=3.8.0", "pytest-metadata")


def _json_report(fake):
    fake.add(
        "pytest-json-report",
        "1.5.0",
        vr.wheel(
            "pytest-json-report",
            "1.5.0",
            requires=JSON_REPORT_REQUIRES,
            entry="json-report = pytest_jsonreport.plugin",
            module="pytest_jsonreport",
        ),
    )


def test_crystallographic_is_refused_the_same_in_the_dry_run_and_the_run(
    tmp_path, bench, sandbox_reached, pypi
):
    """skillsbench@1.0 crystallographic-wyckoff-position-analysis pins
    pytest-json-report==1.5.0 in test.sh but not pytest-metadata, which it
    needs: refused before any sandbox, with the reason the sandbox's own
    preparation gave, in the dry run as in the run."""
    _json_report(pypi)
    task = _task(tmp_path / "tasks", "crystal")
    (task / "tests" / "test.sh").write_text(CRYSTAL)

    recorded = _same_in_dry_run_and_run(tmp_path, bench, "docker", "crystal", *STRICT)

    assert "verifier.pytest_plugin_releases" in recorded
    assert (
        "pytest-json-report==1.5.0 needs pytest-metadata, which the task does not pin"
    ) in recorded
    assert "SandboxCreationReached" not in recorded


def test_the_plan_gate_gives_the_reason_the_sandbox_would(tmp_path, pypi):
    from benchflow.sandbox import _verifier_runtime as runtime

    _json_report(pypi)
    task = _make(tmp_path, CRYSTAL)

    why = vpi.closure_refusal(task)
    with pytest.raises(runtime.RuntimePinError) as raised:
        runtime.resolve(["pytest-json-report==1.5.0"], ctrf=False, interpreter=None)

    assert why == str(raised.value)


def test_a_fully_pinned_closure_passes_the_plan_gate(tmp_path, bench, pypi):
    _json_report(pypi)
    pypi.add(
        "pytest-metadata",
        "3.1.1",
        vr.wheel(
            "pytest-metadata",
            "3.1.1",
            requires=["pytest>=7.0.0"],
            entry="metadata = pytest_metadata.plugin",
            module="pytest_metadata",
        ),
    )
    task = _task(tmp_path / "tasks", "pinned")
    (task / "tests" / "test.sh").write_text(
        CRYSTAL.replace(
            "pytest-json-report==1.5.0",
            "pytest-json-report==1.5.0 pytest-metadata==3.1.1",
        )
    )

    dry = bench("docker", "--dry-run", jobs="dry")

    assert vpi.closure_refusal(Task(task)) is None
    assert dry.exit_code == 0, dry.output


@pytest.mark.parametrize(
    ("marker", "refused"),
    [('python_version < "3.11"', True), ('sys_platform == "win32"', False)],
    ids=["any-python", "never-linux"],
)
def test_a_dependency_counts_under_any_python_a_verifier_may_have(
    tmp_path, pypi, marker, refused
):
    """The image's Python is not known before the sandbox: what any Linux
    CPython 3.8-3.14 would need must be pinned; what no Linux needs is not."""
    pypi.add(
        "pytest-thing",
        "1.0",
        vr.wheel(
            "pytest-thing",
            "1.0",
            requires=[f"rich>=13; {marker}"],
            entry="thing = thing",
        ),
    )
    task = _make(
        tmp_path, "#!/bin/bash\npip install pytest-thing==1.0\npytest /tests\n"
    )

    why = vpi.closure_refusal(task)

    assert (why is not None) is refused, why
    if refused:
        assert "pytest-thing==1.0 needs rich>=13" in why


def test_without_pypi_or_a_cache_the_sandbox_decides(tmp_path, pypi, monkeypatch):
    """Offline with nothing cached: no refusal here; the sandbox's
    preparation reports the network's failure, as before."""
    monkeypatch.setenv("BENCHFLOW_PLUGIN_RELEASES", "offline")
    task = _make(tmp_path, CRYSTAL)

    assert vpi.closure_refusal(task) is None
