"""A converted task's separate verifier is graded as its task.toml source.

Guards the fix against the layout check (abcf2781c) added to
``_format_verifier``. ``bench import --remove-legacy`` moves ``tests/`` to
``verifier/`` and keeps the config, so the verifier image is still built from
the task's own Dockerfile (``plan_verifier_image`` reads ``verifier/`` as the
tests folder). Judged by the folder's name, the converted task lost the
format's transfer: the agent's whole workspace replaced the verifier image's
instead of only the declared artifacts crossing. Terminal-Bench 4.1's
atrx-vep-crispr, whose verifier image puts ``/app/data/ensembl-vep-release-115``
on PATH, then held the agent's files in that folder, hardening dropped it
from PATH, and the trial crashed with "omitted PATH entries ... require an
explicit [verifier] required_tools contract" while its task.toml source
scored 1. Hardening was right about the transfer it was given; the transfer
is fixed. Its solve.sh also runs ``python3 /solution/solve.py``, and the
converted oracle folder is uploaded to ``/oracle``: ``/solution`` now names it
when the sandbox has none, as ``/tests`` names ``/verifier``.

The Docker scenario (deterministic tier: real Docker, the oracle, no model)
runs one small task in both layouts: an image PATH folder under ``/app`` the
verifier never uses is kept and the trial scores 1, and a PATH folder a
declared artifact fills from the agent's sandbox is still refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow._utils.task_authoring.scaffolding import migrate_task_to_task_md
from benchflow.rollout._separate_verifier import _format_verifier, _verifier_task
from benchflow.task import Task
from benchflow.task.verifier_sandbox import plan_verifier_image
from tests.integration.deterministic import harness as h

SANDBOX, SKIP_REASON = h.select_sandbox()
BASE = (
    "FROM python:3.12-slim@sha256:"
    "a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1\n"
)
# Both images put an image tool folder under the workspace on PATH, as
# atrx-vep-crispr's put /app/data/ensembl-vep-release-115.
TOOL = (
    "RUN mkdir -p /app/tool /app/output /app/bin && "
    "printf '#!/bin/sh\\necho image-tool\\n' > /app/tool/greet && "
    "chmod 755 /app/tool/greet\n"
    'ENV PATH="/app/tool:/app/bin:${PATH}"\n'
)
AGENT_DOCKERFILE = BASE + TOOL + "WORKDIR /app\n"
VERIFIER_DOCKERFILE = BASE + TOOL + "COPY . /tests/\nWORKDIR /app\n"
TASK_TOML = """version = "1.0"
artifacts = {artifacts}

[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "fixture"

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 4096
"""
# The agent plants a program in the image's tool folder and one in /app/bin.
# Its helper is read from /solution, as Terminal-Bench 4.1's solve.sh does:
# converted, the folder is uploaded to /oracle.
WRITE_SH = "#!/bin/bash\necho ok > /app/output/answer.txt\n"
SOLVE_SH = """#!/bin/bash
set -e
bash /solution/write.sh
printf '#!/bin/sh\\necho planted\\n' > /app/tool/greet
printf '#!/bin/sh\\necho planted\\n' > /app/bin/helper
chmod 755 /app/bin/helper
"""
# Never runs a program from /app/tool.
UNUSED_TEST_SH = """#!/bin/bash
if [ "$(cat /app/output/answer.txt 2>/dev/null)" = ok ]; then r=1; else r=0; fi
echo "$r" > /logs/verifier/reward.txt
"""
# Runs the agent's program from /app/bin, a folder a declared artifact fills.
PLANTED_TEST_SH = """#!/bin/bash
if [ "$(helper)" = planted ]; then r=1; else r=0; fi
echo "$r" > /logs/verifier/reward.txt
"""
SCENARIOS = {
    "unused": ('["/app/output/answer.txt"]', UNUSED_TEST_SH),
    "planted": ('["/app/output/answer.txt", "/app/bin/"]', PLANTED_TEST_SH),
}


def _task(root: Path, scenario: str, *, converted: bool) -> Path:
    artifacts, test_sh = SCENARIOS[scenario]
    task = root / f"{scenario}-{'converted' if converted else 'toml'}"
    for sub in ("environment", "solution", "tests"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(TASK_TOML.format(artifacts=artifacts))
    (task / "instruction.md").write_text("Write ok to /app/output/answer.txt.\n")
    (task / "environment" / "Dockerfile").write_text(AGENT_DOCKERFILE)
    (task / "tests" / "Dockerfile").write_text(VERIFIER_DOCKERFILE)
    for path, text in (
        (task / "solution" / "solve.sh", SOLVE_SH),
        (task / "solution" / "write.sh", WRITE_SH),
        (task / "tests" / "test.sh", test_sh),
    ):
        path.write_text(text)
        path.chmod(0o755)
    if converted:
        migrate_task_to_task_md(task, remove_legacy=True)
        assert (task / "verifier" / "Dockerfile").is_file()
        assert (task / "oracle" / "write.sh").is_file()
        assert not (task / "tests").exists() and not (task / "task.toml").exists()
    return task


def _format(task_dir: Path) -> tuple[bool, bool]:
    task = Task(task_dir)
    plan = plan_verifier_image(task.config, task_dir)
    return _format_verifier(task, plan), _verifier_task(
        task, plan
    )._benchflow_bundled_tests


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_a_converted_task_keeps_the_task_toml_verifier_format(tmp_path, scenario):
    legacy = _task(tmp_path, scenario, converted=False)
    converted = _task(tmp_path, scenario, converted=True)
    assert Task(converted).paths.uses_native_verifier_dir is True
    assert _format(legacy) == (True, True)
    # Before the converted layout read (False, False).
    assert _format(converted) == _format(legacy)


def test_the_converter_marks_its_output(tmp_path):
    from benchflow.task._document_parse import converted_from_task_toml

    converted = _task(tmp_path, "unused", converted=True)
    assert "converted_from: task.toml" in (converted / "task.md").read_text()
    assert converted_from_task_toml(Task(converted)) is True
    assert (
        converted_from_task_toml(Task(_task(tmp_path, "planted", converted=False)))
        is False
    )


def test_a_task_md_written_as_one_keeps_benchflows_own_rule(tmp_path):
    """review: only a task bench import converted (its
    benchflow.converted_from mark) gets the task.toml format's transfer; a
    task.md written as one, with the same verifier/Dockerfile and artifacts,
    keeps the whole workspace and the upload of its verifier folder."""
    task_dir = _task(tmp_path, "unused", converted=True)
    text = (task_dir / "task.md").read_text()
    (task_dir / "task.md").write_text(
        text.replace("benchflow:\n  converted_from: task.toml\n", "")
    )
    assert "converted_from" not in (task_dir / "task.md").read_text()
    assert Task(task_dir).paths.uses_native_verifier_dir is True
    assert _format(task_dir) == (False, False)


def test_a_verifier_from_the_agent_image_is_still_not_the_format(tmp_path):
    task_dir = _task(tmp_path, "unused", converted=True)
    (task_dir / "verifier" / "Dockerfile").unlink()
    assert _format(task_dir) == (False, False)


@pytest.fixture
def require_docker() -> None:
    if SANDBOX != "docker":
        reason = SKIP_REASON if SANDBOX is None else f"Docker-only scenario ({SANDBOX})"
        pytest.skip(reason)


async def _oracle(task: Path, jobs: Path, integrity: str | None = None):
    if integrity is None:
        from benchflow.sdk import SDK

        result = await SDK().run(
            task_path=task, agent="oracle", jobs_dir=jobs, environment="docker"
        )
    else:
        import benchflow as bf

        result = await bf.run(
            bf.RolloutConfig(
                task_path=task,
                agent="oracle",
                jobs_dir=jobs,
                environment="docker",
                integrity=integrity,
            )
        )
    (record,) = jobs.glob("**/verifier-sandbox/verifier-sandbox.json")
    return result, json.loads(record.read_text())


@pytest.mark.deterministic
@pytest.mark.usefixtures("require_docker")
@pytest.mark.parametrize("converted", [False, True], ids=["task-toml", "converted"])
async def test_an_image_path_folder_the_verifier_never_uses_is_graded(
    tmp_path, converted
):
    task = _task(tmp_path / "tasks", "unused", converted=converted)
    result, record = await _oracle(task, tmp_path / "jobs")
    said = "\n".join(
        p.read_text() for p in (tmp_path / "jobs").glob("**/verifier/test-stdout.txt")
    )
    assert result.error is None, result.error
    assert result.verifier_error is None, result.verifier_error
    assert result.rewards == {"reward": 1.0}, said
    transfer = record["transfer"]
    # Only the declared artifact crossed: /app/tool is the verifier image's.
    assert transfer["artifacts_only"] is True
    assert "/app" not in transfer["agent_paths"]
    # The record is written only when hardening left a folder out.
    for trusted in (tmp_path / "jobs").glob("**/verifier/trusted-path.json"):
        dropped = [d["folder"] for d in json.loads(trusted.read_text())["dropped"]]
        assert "/app/tool" not in dropped


@pytest.mark.deterministic
@pytest.mark.usefixtures("require_docker")
@pytest.mark.parametrize("converted", [False, True], ids=["task-toml", "converted"])
async def test_a_path_folder_the_agent_filled_is_still_refused(tmp_path, converted):
    task = _task(tmp_path / "tasks", "planted", converted=converted)
    # Strict: dropping the agent's PATH folders is strict's hardening since
    # the faithful default (the lead, 09:58Z); audit runs test.sh as
    # written, with the declared artifact's folder on PATH.
    result, record = await _oracle(task, tmp_path / "jobs", integrity="strict")
    assert "/app/bin" in record["transfer"]["agent_paths"]
    assert result.rewards is None
    assert result.verifier_error is not None
    assert "required_tools" in result.verifier_error
    assert "/app/bin" in result.verifier_error
    assert "/app/tool" not in result.verifier_error
