"""REL r4 H1: ``$BENCHFLOW_PYTEST`` names a stub when no runner was prepared.

A task whose tests folder and ``[verifier] env`` never name the runner gets
none (ACC-RUN2). ``$BENCHFLOW_PYTEST`` is still set, so no script falls back
to the image's own pytest: it names a root-owned stub in the guard's
directory that prints why and exits 4. The stub tells the verifier nothing:
the agent's code runs as test.sh's user and can run it too, so no signal of
the stub may make a trial unscored. The trial scores what
test.sh reports. ``bench check`` warns before a run when only the build
context or the solution names the runner. The real trials are in
``test_runner_stub_docker``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from benchflow.sandbox import lockdown

GUARD = lockdown.PYTEST_PLUGIN_GUARD_PREFIX + "0123abcd"


async def _assemble(monkeypatch, tmp_path, test_sh: str, *, service="main"):
    from benchflow._utils.trusted_path import TrustedPathReport
    from tests.test_sandbox_hardening import _make_env, _make_task

    install = AsyncMock(return_value=("/" + GUARD, "-p " + GUARD, (), ()))
    for name, result in (
        ("_probe_verifier_reads", None),
        ("_judge_verifier_path", TrustedPathReport(path="/usr/bin:/bin")),
        ("_isolate_verifier_tool_state", {}),
        ("_discover_pytest_plugins", ([], [])),
        ("_trusted_verifier_pythonpath", ""),
        ("_distro_pip_env", {}),
        ("_verifier_uid", 0),
    ):
        monkeypatch.setattr(lockdown, name, AsyncMock(return_value=result))
    monkeypatch.setattr(lockdown, "_install_pytest_plugin_guard", install)
    task = _make_task()
    task.config.verifier.service = service
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test.sh").write_text(test_sh)
    task.task_dir = tmp_path
    sandbox = _make_env()
    env = await lockdown._build_verifier_env(
        sandbox, task, "agent", "/app", bash_env=""
    )
    return env, sandbox, install


async def test_a_task_that_never_names_the_runner_gets_the_stub(monkeypatch, tmp_path):
    env, _sandbox, install = await _assemble(
        monkeypatch, tmp_path, "#!/bin/bash\n/usr/local/bin/grade\n"
    )
    assert env.get("BENCHFLOW_PYTEST") == f"/{GUARD}/benchflow-pytest"
    assert install.call_args.kwargs["runner_stub"] is True


async def test_a_target_service_also_gets_a_runner_stub(monkeypatch, tmp_path):
    """FIX-49 r4: c278f22a6 left the runner unset in a target service."""
    env, sandbox, install = await _assemble(
        monkeypatch, tmp_path, "#!/bin/sh\n/opt/grade\n", service="target"
    )
    assert env["BENCHFLOW_PYTEST"].endswith("/benchflow-pytest")
    assert "_runtime_guard_" not in env["PYTEST_ADDOPTS"]
    install.assert_not_awaited()
    assert any(
        "benchflow-pytest" in str(call.args) and call.kwargs.get("service") == "target"
        for call in sandbox.exec.call_args_list
    )


async def test_a_test_sh_that_names_the_runner_gets_no_stub(monkeypatch, tmp_path):
    env, _sandbox, install = await _assemble(
        monkeypatch, tmp_path, '#!/bin/bash\n"$BENCHFLOW_PYTEST" /tests\n'
    )
    assert install.call_args.kwargs["runner_stub"] is False
    assert "BENCHFLOW_PYTEST" not in env


def test_the_stub_prints_the_error_exits_4_and_writes_nothing(tmp_path):
    """At 6c7426dcb the stub left ``norunner`` in the guard's marker folder,
    which the verifier user (and so the agent's code) can write."""
    source = lockdown._pytest_runner_stub_source()
    assert lockdown.pytest_plugin_guard_markers_dir(GUARD) not in source
    assert ">" not in source.replace(">&2", "")  # no file is written
    stub = tmp_path / "stub"
    stub.write_text(source)
    work = tmp_path / "work"
    work.mkdir()
    done = subprocess.run(
        ["sh", str(stub), "-q", "/tests"], cwd=work, capture_output=True, text=True
    )
    assert done.returncode == 4
    assert done.stdout == ""
    assert done.stderr.startswith(
        "benchflow-pytest: error: no protected test runner was prepared, because "
        "the task's test.sh (and the rest of its tests folder) and [verifier] "
        "env never name $BENCHFLOW_PYTEST."
    ), done.stderr
    assert "The task must name $BENCHFLOW_PYTEST in test.sh" in done.stderr
    assert list(work.iterdir()) == []


def test_the_marker_read_has_no_stub_line():
    cmd = lockdown.pytest_plugin_guard_markers_cmd(GUARD)
    assert "norunner" not in cmd
    with pytest.raises(TypeError):
        lockdown.pytest_plugin_guard_markers_cmd(GUARD, runner_stub=True)  # type: ignore[call-arg]
    assert not hasattr(lockdown, "pytest_runner_stub_ran")
    assert not hasattr(lockdown, "RUNNER_STUB_MARKER")


async def test_a_norunner_file_is_no_verdict(monkeypatch):
    """Whatever the verifier user leaves in the marker folder under the
    stub's old name, the verifier takes no verdict from it (M2)."""
    from types import SimpleNamespace

    import benchflow.task.verifier_core as core_module

    sandbox = SimpleNamespace(_benchflow_runner_stub=GUARD)
    core = core_module.Verifier.__new__(core_module.Verifier)
    core._sandbox = sandbox
    core._logger = SimpleNamespace(warning=lambda *a, **k: None, debug=lambda *a: None)

    async def answer(_sandbox, cmd, **_kwargs):
        assert "norunner" not in cmd
        return SimpleNamespace(stdout="norunner\t\n!norunner\n!end\n", exit_code=0)

    monkeypatch.setattr(core_module, "receipt_exec", answer)
    markers = await core._plugin_guard_markers(GUARD, "main")
    assert "norunner" not in markers


# --- bench check: the runner named only in the image or the solution ---------


def _task(tmp_path: Path, *, test_sh: str, toml: str = 'version = "1.0"\n') -> Path:
    task = tmp_path / "grader-task"
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "solution").mkdir()
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY grade /usr/local/bin/grade\n"
    )
    (task / "environment" / "grade").write_text("#!/bin/bash\nexit 0\n")
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\ntouch /app/ok\n")
    (task / "instruction.md").write_text("Create /app/ok.\n")
    (task / "task.toml").write_text(toml)
    (task / "tests" / "test.sh").write_text(test_sh)
    return task


def _runner_warnings(task: Path) -> list[str]:
    from benchflow._utils.task_authoring import check_task_warnings

    return [w for w in check_task_warnings(task) if "BENCHFLOW_PYTEST" in w]


IMAGE_GRADER = '#!/bin/bash\n"$BENCHFLOW_PYTEST" -q /tests/test_x.py\n'


def test_check_warns_when_only_the_build_context_names_the_runner(tmp_path):
    from benchflow.cli.main import app

    task = _task(tmp_path, test_sh="#!/bin/bash\n/usr/local/bin/grade\n")
    (task / "environment" / "grade").write_text(IMAGE_GRADER)
    (warning,) = _runner_warnings(task)
    assert warning.startswith("environment/grade names $BENCHFLOW_PYTEST, but the")
    assert "exits 4, and the trial scores what test.sh reports" in warning
    assert "\n    next: name $BENCHFLOW_PYTEST in test.sh" in warning
    # FIX-49 r4: the advice also applies without pins or CTRF.
    assert "even without plugin pins or --ctrf" in warning

    out = CliRunner().invoke(app, ["tasks", "check", str(task)])
    assert "warning:" in out.output and "next: name $BENCHFLOW_PYTEST" in out.output
    assert out.exit_code == 0, out.output


def test_check_warns_when_only_the_solution_names_the_runner(tmp_path):
    task = _task(tmp_path, test_sh="#!/bin/bash\n/usr/local/bin/grade\n")
    (task / "solution" / "solve.sh").write_text(
        "#!/bin/bash\necho '$BENCHFLOW_PYTEST -q' > /app/run-tests\n"
    )
    (warning,) = _runner_warnings(task)
    assert warning.startswith("solution/solve.sh names $BENCHFLOW_PYTEST")


@pytest.mark.parametrize(
    ("test_sh", "toml"),
    [
        ("#!/bin/bash\n# runs $BENCHFLOW_PYTEST\n/usr/local/bin/grade\n", None),
        (
            "#!/bin/bash\n/usr/local/bin/grade\n",
            'version = "1.0"\n\n[verifier.env]\nRUNNER = "${BENCHFLOW_PYTEST}"\n',
        ),
    ],
    ids=["test-sh", "verifier-env"],
)
def test_check_is_quiet_when_the_task_names_the_runner(tmp_path, test_sh, toml):
    task = _task(tmp_path, test_sh=test_sh, **({"toml": toml} if toml else {}))
    (task / "environment" / "grade").write_text(IMAGE_GRADER)
    assert _runner_warnings(task) == []


def test_check_is_quiet_when_nothing_names_the_runner(tmp_path):
    task = _task(tmp_path, test_sh="#!/bin/bash\n/usr/local/bin/grade\n")
    assert _runner_warnings(task) == []
