"""Task refusals say it in BenchFlow's terms, not another runner's.

REPORT B18 of the Oct 13 dogfood (round PY8, target D2b): two messages a
user sees when BenchFlow refuses a task.toml task named a third-party
runner. The multi-step refusal now says "multi-step tasks are not supported
(one prompt per task)", and the collect-hook refusal names task.toml's
``[[verifier.collect]]`` hooks; the unknown-key refusal beside them says "a
newer task.toml key", and the separate verifier's "no verifier image"
refusal names ``[verifier.environment]`` as task.toml's. Each is checked where it is made and where
``bench tasks check`` prints it.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.task import Task
from benchflow.task.runtime_capabilities import validate_task_runtime_support

ONE_STEP_TOML = """\
version = "1.0"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
"""

MULTI_STEP_TOML = """\
version = "1.0"

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048

[[steps]]
name = "create-file"

[steps.agent]
timeout_sec = 30.0

[[steps]]
name = "append-content"

[steps.agent]
timeout_sec = 30.0
"""

COLLECT_HOOK = '\n[[verifier.collect]]\nservice = "main"\ncommand = "cat /tmp/x"\n'


def _one_step_task(root: Path, toml: str) -> Path:
    task = root / "one-step"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\necho hi\n")
    (task / "instruction.md").write_text("Write hello.txt.\n")
    (task / "task.toml").write_text(toml)
    return task


def _multi_step_task(root: Path) -> Path:
    task = root / "multi-step"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "task.toml").write_text(MULTI_STEP_TOML)
    for name in ("create-file", "append-content"):
        step = task / "steps" / name
        (step / "tests").mkdir(parents=True)
        (step / "instruction.md").write_text("Do the step.\n")
        (step / "tests" / "test.sh").write_text(
            "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
        )
    return task


def _check_output(task_dir: Path) -> str:
    """What ``bench tasks check`` prints, its wrapped lines joined."""
    result = CliRunner().invoke(
        app, ["tasks", "check", str(task_dir), "--sandbox", "docker"]
    )
    assert result.exit_code == 1, result.output
    return " ".join(result.output.split())


def test_the_multi_step_refusal_names_no_other_runner(tmp_path: Path) -> None:
    task_dir = _multi_step_task(tmp_path)
    issues = validate_task_runtime_support(
        Task(task_dir).config, sandbox="docker", task_dir=task_dir
    )
    assert [(i.path, i.reason) for i in issues] == [
        ("steps", "multi-step tasks are not supported (one prompt per task)")
    ]


VERIFIER_SANDBOX = "\n[verifier.environment]\ncpus = 1\n"
NO_VERIFIER_IMAGE = (
    "no verifier image: add tests/Dockerfile or a docker_image to "
    "[verifier.sandbox] ([verifier.environment] in task.toml)"
)


def test_the_no_verifier_image_refusal_names_task_toml(tmp_path: Path) -> None:
    """D2b-R1-B-2: a separate verifier with nothing to build it from."""
    task_dir = _one_step_task(tmp_path, ONE_STEP_TOML + VERIFIER_SANDBOX)
    issues = validate_task_runtime_support(
        Task(task_dir).config, sandbox="docker", task_dir=task_dir
    )
    assert [(i.path, i.reason) for i in issues] == [
        ("verifier.sandbox", NO_VERIFIER_IMAGE)
    ]
