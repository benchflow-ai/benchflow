"""The ``x-keep-services`` key loads in every format and is checked (K2).

task.md draft 2 ``[sandbox] x-keep-services``, task.toml ``[metadata]
x-keep-services`` and BenchFlow task.md front matter ``metadata:
x-keep-services:``: a list of at most 8 ``{ port, name }`` tables, ports 1 to
65535 (the Rust ``bench``'s key). A wrong value is refused when the
task loads and by ``bench tasks check``, each problem named at its place.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.task import Task
from benchflow.task.config import TaskConfig


def _ks():
    """The module under test, imported in each test, so that each test fails on
    its own where the module is missing (the base)."""
    from benchflow.task import keep_services

    return keep_services


REPO = Path(__file__).resolve().parent.parent
DRAFT2_HELLO = REPO / "tests/fixtures/taskmd/examples/hello-world"
NATIVE_HELLO = REPO / "tests/examples/hello-world-task"


def _split_task(parent: Path, keep_toml: str, name: str = "svc-task") -> Path:
    task = parent / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[metadata]\nauthor_name = "x"\n'
        + keep_toml
        + "\n[verifier]\ntimeout_sec = 60.0\n\n[agent]\ntimeout_sec = 60.0\n\n"
        '[environment]\ndocker_image = "ubuntu:24.04"\n'
    )
    (task / "instruction.md").write_text("Start a web service on port 5000.\n")
    (task / "environment/Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests/test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "solution/solve.sh").write_text("#!/bin/bash\ntrue\n")
    return task


def test_read_names_every_problem_at_its_place():
    services, problems = _ks().read(
        [
            {"port": 0},
            {"port": 65536},
            {"port": "80"},
            {"name": "x"},
            {"port": 22, "nmae": "ssh"},
            {"port": 22, "name": ""},
            7,
            {"port": True},
            {"port": 9000, "x-note": "an extension"},
        ]
    )
    assert [p for p, _ in problems] == [
        "",
        "[0].port",
        "[1].port",
        "[2].port",
        "[3]",
        "[4]",
        "[5].name",
        "[6]",
        "[7].port",
    ]
    # More than eight: nothing is kept.
    assert services == []
    assert _ks().read({"port": 1})[1][0][0] == ""


def test_read_keeps_ports_and_names_and_refuses_a_port_named_twice():
    services, problems = _ks().read([{"port": 5000, "name": "api"}, {"port": 8080}])
    assert problems == []
    assert services == [_ks().KeepService(5000, "api"), _ks().KeepService(8080)]
    _services, problems = _ks().read([{"port": 5000}, {"port": 5000, "name": "again"}])
    assert problems == [("[1].port", "port 5000 is named twice")]


def test_task_toml_metadata_key_loads():
    config = TaskConfig.model_validate_toml(
        '[metadata]\nx-keep-services = [{ port = 5000, name = "api" }, { port = 8080 }]\n'
    )
    assert config.metadata["x-keep-services"][1] == {"port": 8080}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            "[{ port = 0 }]",
            "x-keep-services[0].port must be a whole number from 1 to 65535, not 0",
        ),
        ('"5000"', "x-keep-services must be a list of { port, name } tables"),
        (
            "[" + ", ".join(f"{{ port = {p} }}" for p in range(5000, 5009)) + "]",
            "x-keep-services names 9 services; at most 8 are kept",
        ),
        ("[{ port = 80, host = 1 }]", "x-keep-services[0] has an unknown key 'host'"),
    ],
)
def test_task_toml_bad_values_are_refused_with_their_place(value, message):
    with pytest.raises(ValueError, match=r"metadata\." + _re(message)):
        TaskConfig.model_validate_toml(f"[metadata]\nx-keep-services = {value}\n")


def _re(text: str) -> str:
    import re

    return re.escape(text)


def test_bench_tasks_check_refuses_a_bad_task_toml_value(tmp_path):
    task = _split_task(tmp_path, 'x-keep-services = [{ port = 70000, name = "api" }]\n')
    result = CliRunner().invoke(app, ["tasks", "check", str(task)])
    assert result.exit_code == 1, result.output
    flat = " ".join(result.output.split())
    assert (
        "metadata.x-keep-services[0].port must be a whole number from 1 to 65535"
        in flat
    )


def test_bench_tasks_check_says_what_a_good_task_toml_keeps(tmp_path):
    task = _split_task(tmp_path, 'x-keep-services = [{ port = 5000, name = "api" }]\n')
    result = CliRunner().invoke(app, ["tasks", "check", str(task)])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "keeps the agent's services on port 5000 (api) for grading" in flat
    assert "from the task" in flat
    # K2-R1-A-4, K2-R2-B-2: the check says where a kept service cannot write.
    assert (
        "while graded they cannot write under /logs, nor in the workspace except "
        "to a file held open since before the kill, so they keep their state "
        "under /tmp or the agent user's home"
    ) in flat, result.output
    # K2-R4-B-1: and that they read the verifier's command lines.
    assert (
        "they can read every command line in the sandbox, so test.sh must keep "
        "keys off its commands' arguments"
    ) in flat, result.output
    kept = _ks().for_task(Task(task))
    assert kept is not None and kept.ports == (5000,) and kept.source == "the task"


def _native(tmp_path: Path, keep_yaml: str) -> Path:
    task = tmp_path / "native"
    shutil.copytree(NATIVE_HELLO, task)
    text = (task / "task.md").read_text()
    assert "\nmetadata:\n" in text
    (task / "task.md").write_text(
        text.replace("\nmetadata:\n", "\nmetadata:\n" + keep_yaml, 1)
    )
    return task


def test_native_task_md_front_matter_key_loads_and_bad_ones_are_refused(tmp_path):
    good = _native(tmp_path, "  x-keep-services:\n  - port: 5000\n    name: api\n")
    kept = _ks().for_task(Task(good))
    assert kept is not None and kept.services == (_ks().KeepService(5000, "api"),)

    bad = tmp_path / "bad"
    bad.mkdir()
    bad_task = _native(bad, "  x-keep-services:\n  - port: 5000\n    name: [1]\n")
    result = CliRunner().invoke(app, ["tasks", "check", str(bad_task)])
    assert result.exit_code == 1, result.output
    flat = " ".join(result.output.split())
    assert (
        "metadata.x-keep-services[0].name must be a label of 1 to 64 characters" in flat
    )


def _draft2(tmp_path: Path, keep_toml: str) -> Path:
    task = tmp_path / "draft2"
    shutil.copytree(DRAFT2_HELLO, task)
    text = (task / "task.md").read_text()
    marker = '[agent]\ntimeout = "2m"\n'
    assert marker in text
    (task / "task.md").write_text(
        text.replace(marker, marker + "\n[sandbox]\n" + keep_toml, 1)
    )
    return task


def test_draft2_sandbox_key_is_honored_as_metadata(tmp_path):
    from benchflow.taskmd import load_plan

    task = _draft2(tmp_path, 'x-keep-services = [{ port = 5000, name = "api" }]\n')
    plan = load_plan(task)
    assert plan.frontmatter["metadata"]["x-keep-services"] == [
        {"port": 5000, "name": "api"}
    ]
    assert any(f.field == "[sandbox] x-keep-services" for f in plan.honored)
    result = CliRunner().invoke(app, ["tasks", "check", str(task)])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "keeps the agent's services on port 5000 (api) for grading" in flat


def test_draft2_bad_value_is_refused_by_bench_tasks_check(tmp_path):
    task = _draft2(tmp_path, "x-keep-services = [{ port = 0 }]\n")
    result = CliRunner().invoke(app, ["tasks", "check", str(task)])
    assert result.exit_code == 1, result.output
    flat = " ".join(result.output.split())
    assert "[sandbox] x-keep-services" in flat
    assert "[0].port must be a whole number from 1 to 65535, not 0" in flat
