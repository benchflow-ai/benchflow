"""Loading task.md draft-1 packages through the v0.6 document model.

The fixtures under ``tests/fixtures/task_md_draft1`` are verbatim copies of
four examples from the task.md draft-1 specification. Draft-1 files have no
frontmatter: the instruction comes first, then typed fenced blocks.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.skill_policy import strip_task_bundled_skills, task_bundled_skills_dir
from benchflow.task import (
    Task,
    TaskDocument,
    TaskDocumentParseError,
    TaskPackage,
    TaskPaths,
    UnsupportedTaskFeatureError,
    raise_for_task_runtime_support,
    validate_task_runtime_support,
)
from benchflow.task.config import Author, NetworkMode, VerifierSandboxMode
from benchflow.task.paths import task_environment_dir

FIXTURES = Path(__file__).parent / "fixtures" / "task_md_draft1"
DRAFT1_EXAMPLES = ("hello-world", "flaky-retry", "stl-mass", "ising-exponent")
V06_TASK = Path(__file__).parent / "examples" / "hello-world-task"

# The first fence that opens a task.md block. Everything before it is the
# instruction; ordinary fences such as ```json stay in the instruction.
_FIRST_BLOCK = re.compile(
    r"^```(?:toml task|yaml task|stage \S+|role \S+|user|notes)[ \t]*$", re.M
)


def _instruction_as_written(task_dir: Path) -> str:
    text = (task_dir / "task.md").read_text()
    match = _FIRST_BLOCK.search(text)
    assert match is not None
    return text[: match.start()].rstrip("\n")


def _write_task(tmp_path: Path, text: str) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.md").write_text(text)
    return task_dir


def _launch_issue_paths(task_dir: Path, sandbox: str = "docker") -> set[str]:
    document = TaskDocument.from_path(task_dir / "task.md")
    issues = validate_task_runtime_support(document, sandbox=sandbox, task_dir=task_dir)
    return {issue.path for issue in issues}


@pytest.mark.parametrize("name", DRAFT1_EXAMPLES)
def test_draft1_prompt_is_the_instruction(name: str) -> None:
    """The agent's prompt is the text before the first block, and nothing else."""
    task_dir = FIXTURES / name
    expected = _instruction_as_written(task_dir)

    task = Task(task_dir)
    package = TaskPackage.from_task_dir(task_dir)

    assert task.document is not None
    assert task.document.draft1 is not None
    assert task.instruction == expected
    assert package.view.prompt == expected
    assert package.prompt_plan is not None
    assert [turn.prompt for turn in package.prompt_plan.turns] == [expected]
    assert "```toml task" not in task.instruction


def test_draft1_instruction_keeps_ordinary_fences() -> None:
    """stl-mass shows a ```json fence inside its instruction."""
    document = TaskDocument.from_path(FIXTURES / "stl-mass" / "task.md")

    assert '```json\n{\n "main_part_mass": 12345.67,\n "material_id": 42\n}\n```' in (
        document.instruction
    )
    assert document.instruction.endswith(
        "NOTE: The result will be considered correct if it is within **0.1% accuracy**."
    )


@pytest.mark.parametrize("name", DRAFT1_EXAMPLES)
def test_draft1_package_folders(name: str) -> None:
    """sandbox/ is the build context; verifier/ and oracle/ are the native folders."""
    task_dir = (FIXTURES / name).resolve()
    paths = Task(task_dir).paths
    view = TaskPackage.from_task_dir(task_dir).view

    assert paths.environment_dir == task_dir / "sandbox"
    assert view.environment_dir == task_dir / "sandbox"
    assert task_environment_dir(task_dir) == task_dir / "sandbox"
    assert (paths.environment_dir / "Dockerfile").is_file()
    assert paths.tests_dir == task_dir / "verifier"
    assert paths.uses_native_verifier_dir
    assert "sandbox/Dockerfile" in view.source_hashes
    if (task_dir / "oracle").is_dir():
        assert paths.uses_native_oracle_dir
        assert paths.solve_path == task_dir / "oracle" / "solve.sh"
        assert paths.solve_path.is_file()


def test_v06_package_keeps_environment_folder() -> None:
    task_dir = V06_TASK.resolve()

    assert TaskPaths(task_dir).environment_dir == task_dir / "environment"
    assert task_environment_dir(task_dir) == task_dir / "environment"
    assert TaskDocument.from_path(task_dir / "task.md").draft1 is None


def test_hello_world_maps_and_can_launch() -> None:
    task_dir = FIXTURES / "hello-world"
    task = Task(task_dir)
    config = task.config

    assert task.name == "examples/hello-world"
    assert config.task is not None
    assert config.task.version == "1.2.0"
    assert config.agent.timeout_sec == 120
    assert config.agent.network_mode is None
    assert config.sandbox.docker_image is None  # built from sandbox/Dockerfile
    assert config.sandbox.memory_mb == 2048  # v0.6 default: the file sets none
    assert _launch_issue_paths(task_dir) == set()
    assert task.document is not None and task.document.draft1 is not None
    assert {f.path for f in task.document.draft1.ignored} == {
        "title",
        "verifier/rubric.json stated, implicit, validation",
    }
    # The gates-only rubric is graded by the runtime from ctrf.json.
    assert task.document.draft1.rubric is not None
    assert task.document.draft1.rubric["criteria"][0]["check"] == "test_hello"


def test_flaky_retry_maps_config_and_refuses_launch() -> None:
    task_dir = FIXTURES / "flaky-retry"
    config = Task(task_dir).config

    assert config.agent.timeout_sec == 900  # "15m"
    assert config.agent.network_mode == NetworkMode.NO_NETWORK  # "none"
    assert config.sandbox.docker_image is None  # built from sandbox/Dockerfile
    assert config.sandbox.workdir == "/workspace"
    assert config.sandbox.cpus == 2
    assert config.sandbox.memory_mb == 4096  # "4 GB"
    assert config.verifier.timeout_sec == 300  # "5m"
    assert config.verifier.sandbox_mode == VerifierSandboxMode.SEPARATE
    assert config.metadata == {
        "difficulty": "medium",
        "category": "software-engineering",
    }
    assert config.task is not None
    assert config.task.keywords == ["python", "networking", "bugfix"]

    # The v0.6 gate refuses the separate verifier; the draft-1 findings add
    # everything the v0.6 model cannot carry.
    assert _launch_issue_paths(task_dir) == {
        "verifier.sandbox_mode",
        "[verifier] judges",
        "[runs]",
        "verifier/rubric.json",
        "verifier/behaviors.json",
    }
    document = TaskDocument.from_path(task_dir / "task.md")
    with pytest.raises(
        UnsupportedTaskFeatureError, match=r"\[runs\]: task\.md draft 1"
    ):
        raise_for_task_runtime_support(document, sandbox="docker", task_dir=task_dir)


def test_stl_mass_maps_sizes_durations_and_authors() -> None:
    task_dir = FIXTURES / "stl-mass"
    config = Task(task_dir).config

    assert config.agent.timeout_sec == 900  # "15m"
    assert config.sandbox.cpus == 1
    assert config.sandbox.memory_mb == 4096  # "4 GB"
    assert config.sandbox.storage_mb == 10240  # "10 GB"
    assert config.sandbox.build_timeout_sec == 600  # "10m"
    assert config.verifier.timeout_sec == 900
    assert config.verifier.sandbox_mode is None
    assert config.task is not None
    assert config.task.name == "skillsbench/3d-scan-calc"
    assert config.task.authors == [Author(name="Wengao Ye")]
    assert config.metadata["source"] == "https://github.com/benchflow-ai/skillsbench"
    assert _launch_issue_paths(task_dir) == set()


def test_ising_exponent_stage_becomes_scene_prompt_and_refuses_launch() -> None:
    task_dir = FIXTURES / "ising-exponent"
    document = TaskDocument.from_path(task_dir / "task.md")
    config = document.config

    assert config.agent.timeout_sec == 4 * 3600  # "4h"
    assert config.sandbox.cpus == 8
    assert config.sandbox.memory_mb == 32 * 1024  # "32 GB"
    assert config.sandbox.docker_image == "ghcr.io/example/physics-sci:2026.09"
    assert document.scene_prompts == {
        "analysis": (
            "The simulation data is now in `/data/mc/`: one HDF5 file per lattice "
            "size L = 8, 12, 16, 24, 32, 48, each with energy and magnetization time "
            "series at 41 temperatures near the transition. Follow your plan, revise "
            "it where the data demands, and write `/work/paper.pdf`."
        )
    }
    assert "/data/mc/" not in document.instruction  # revealed later, not up front
    assert "Reference values" not in document.instruction  # notes stay hidden
    assert document.draft1 is not None
    assert {f.path for f in document.draft1.ignored} == {
        "title",
        "[integrity] canary",
        "```notes",
    }
    assert _launch_issue_paths(task_dir) == {
        "[sandbox] network",
        "[stages]",
        "```stage analysis",
        "[verifier] judges",
        "verifier/rubric.json",
        "verifier/behaviors.json",
    }


def test_prompt_only_file_is_all_instruction(tmp_path: Path) -> None:
    text = "Print the first ten primes to /app/primes.txt, one per line.\n"
    task_dir = _write_task(tmp_path, text)

    document = TaskDocument.from_path(task_dir / "task.md")

    assert document.instruction == text.rstrip("\n")
    assert document.config.agent.timeout_sec is None
    assert document.draft1 is not None
    assert document.draft1.unsupported == ()
    assert TaskPaths(task_dir).environment_dir == task_dir.resolve() / "sandbox"


def test_canary_comment_is_stripped_from_the_prompt(tmp_path: Path) -> None:
    task_dir = _write_task(
        tmp_path,
        "<!-- task.md canary 0c1d2e3f-4a5b-4c6d-8e7f-9a0b1c2d3e4f -->\n\n"
        'Say hi.\n\n```toml task\nname = "acme/hi"\n```\n',
    )

    document = TaskDocument.from_path(task_dir / "task.md")

    assert document.instruction == "Say hi."
    assert document.draft1 is not None
    assert (
        document.draft1.canary == "task.md canary 0c1d2e3f-4a5b-4c6d-8e7f-9a0b1c2d3e4f"
    )


def test_role_and_user_blocks_map_to_v06_prompts(tmp_path: Path) -> None:
    task_dir = _write_task(
        tmp_path,
        "Fix the bug.\n\n"
        "```role reviewer\nReview the patch. Do not edit files.\n```\n\n"
        "```user\nYou are a customer who only mentions the order id when asked.\n```\n\n"
        "```toml task\n"
        '[roles.reviewer]\nagent = "claude-agent-acp"\n\n'
        '[interaction]\nscenes = [{ name = "review", turns = [{ role = "reviewer" }] }]\n'
        "```\n",
    )

    document = TaskDocument.from_path(task_dir / "task.md")

    assert document.role_prompts == {"reviewer": "Review the patch. Do not edit files."}
    assert document.user_persona == (
        "You are a customer who only mentions the order id when asked."
    )
    assert [scene.name for scene in document.scenes] == ["review"]
    assert document.scenes[0].turns[0].prompt == "Review the patch. Do not edit files."
    assert document.draft1 is not None
    assert document.draft1.unsupported == ()


def test_settings_v06_would_ignore_are_refused(tmp_path: Path) -> None:
    """Draft-1 role and user keys the v0.6 runtime never reads must not vanish."""
    task_dir = _write_task(
        tmp_path,
        "Fix the bug.\n\n"
        "```role reviewer\nReview the patch.\n```\n\n"
        "```toml task\n"
        '[roles.reviewer]\nagent = "claude-agent-acp"\nworkspace = "read-only"\n\n'
        "[user]\nmax_turns = 5\n\n"
        '[agent]\non_timeout = "grade"\nbudget = { tool_calls = 400 }\n'
        "```\n",
    )

    document = TaskDocument.from_path(task_dir / "task.md")

    assert document.draft1 is not None
    assert {f.path for f in document.draft1.unsupported} == {
        "[roles.reviewer] workspace",
        "[user] max_turns",
        "[agent] budget",
        "```role reviewer",  # no v0.6 scene gives the role a turn
    }
    # on_timeout = "grade" restates what the runtime already does.
    assert "[agent] on_timeout" in {f.path for f in document.draft1.ignored}


def test_network_reason_is_documentation(tmp_path: Path) -> None:
    """[agent] network_reason (task-md d247339) tells reviewers why the task needs
    open network. The runtime accepts it and runs the task exactly as without it.
    """
    task_dir = _write_task(
        tmp_path,
        "Sum column B of the 2020 census table.\n\n"
        "```toml task\n"
        '[agent]\nnetwork = "open"\n'
        'network_reason = "The census table is published only online."\n'
        "```\n",
    )

    document = TaskDocument.from_path(task_dir / "task.md")

    assert document.config.agent.network_mode == NetworkMode.PUBLIC
    assert document.draft1 is not None
    assert document.draft1.unsupported == ()
    assert {f.path for f in document.draft1.ignored} == {"[agent] network_reason"}
    assert _launch_issue_paths(task_dir) == set()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            'Do it.\n\n```toml task\nname = "acme/x"\n```\n\nA stray line.\n',
            "text after the first task.md block",
        ),
        (
            "Do it.\n\n```toml task\n[sandbox]\nmemory_mb = 4096\n```\n",
            "memory_mb is Harbor's name; task.md uses memory",
        ),
        (
            "Do it.\n\n```toml task\n[agent]\ntimeout = 900\n```\n",
            "must be a duration",
        ),
        (
            'Do it.\n\n```rubric\n{"criteria": []}\n```\n',
            "move them to verifier/rubric.json",
        ),
        ("\n\n```toml task\n```\n", "the instruction is empty"),
        (
            "Do it.\n\n```stage later\nMore.\n```\n\n```toml task\n"
            '[stages.later]\nunlock = "on_request"\n\n[stages.other]\nunlock = "on_request"\n```\n',
            "[stages.other] has no ```stage other block",
        ),
    ],
)
def test_malformed_draft1_files_fail_to_parse(text: str, message: str) -> None:
    with pytest.raises(TaskDocumentParseError, match=re.escape(message)):
        TaskDocument.from_text(text)


def test_harbor_task_toml_beside_draft1_task_md_is_an_error(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, "Do it.\n")
    (task_dir / "task.toml").write_text("[agent]\ntimeout_sec = 60\n")

    with pytest.raises(
        TaskDocumentParseError, match=r"task\.toml is Harbor's config file"
    ):
        TaskDocument.from_path(task_dir / "task.md")


@pytest.mark.parametrize(
    "text",
    [
        "\n---\nversion: '1.0'\n---\nDo it.\n",
        "\ufeff---\nversion: '1.0'\n---\nDo it.\n",
        "",
        "\n\n",
    ],
)
def test_non_draft1_errors_are_unchanged(text: str) -> None:
    """Files that were v0.6 parse errors before draft-1 support stay v0.6 errors."""
    with pytest.raises(
        TaskDocumentParseError, match="must start with YAML frontmatter"
    ):
        TaskDocument.from_text(text)


def test_no_skill_copy_strips_skills_from_sandbox_folder(tmp_path: Path) -> None:
    """No-skill runs strip bundled skills from sandbox/, the draft-1 build context."""
    task_dir = tmp_path / "hello-world"
    shutil.copytree(FIXTURES / "hello-world", task_dir)
    skill = task_dir / "sandbox" / "skills" / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Alpha\n")
    dockerfile = task_dir / "sandbox" / "Dockerfile"
    dockerfile.write_text(
        "FROM ubuntu:24.04\nCOPY skills /root/.claude/skills\nWORKDIR /app\n"
    )

    assert task_bundled_skills_dir(task_dir) == task_dir / "sandbox" / "skills"
    strip_task_bundled_skills(task_dir)

    assert not (task_dir / "sandbox" / "skills").exists()
    assert dockerfile.read_text() == "FROM ubuntu:24.04\nWORKDIR /app\n"


# Grading: the test script writes ctrf.json; the runtime scores the rubric --------

RUBRIC_SCHEMA = "https://task.md/schema/rubric-1.json"


def _ctrf(*tests: tuple[str, str], file_path: str | None = None) -> dict:
    entries = []
    for name, status in tests:
        entry = {"name": name, "status": status, "duration": 1}
        if file_path is not None:
            entry["file_path"] = file_path
        entries.append(entry)
    return {"results": {"tool": {"name": "pytest"}, "tests": entries}}


def _points_task(tmp_path: Path, *, headline: str = "partial") -> Path:
    """A draft-1 package whose rubric has a gate, point criteria, and a penalty."""
    task_dir = tmp_path / "points-task"
    (task_dir / "verifier").mkdir(parents=True)
    (task_dir / "sandbox").mkdir()
    (task_dir / "sandbox" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "task.md").write_text(
        "Speed up the build without adding dependencies.\n\n"
        '```toml task\nname = "acme/points"\n```\n'
    )
    (task_dir / "verifier" / "test.sh").write_text("#!/bin/bash\n")
    rubric = {
        "$schema": RUBRIC_SCHEMA,
        "version": "1.0.0",
        "scoring": {
            "method": "points",
            "gates": "all",
            "headline": headline,
            "pass_threshold": 0.5,
        },
        "criteria": [
            {
                "id": "builds",
                "gate": True,
                "text": "It builds.",
                "judge": "test",
                "check": "test_builds",
            },
            {
                "id": "fast",
                "points": 3,
                "text": "It is fast.",
                "judge": "test",
                "check": "test_fast",
            },
            {
                "id": "tidy",
                "points": 1,
                "text": "Style passes.",
                "judge": "test",
                "check": "tests/test_style.py",
            },
            {
                "id": "no-new-dependency",
                "points": -2,
                "text": "Adds a new dependency.",
                "judge": "test",
                "check": "test_no_new_dependencies",
            },
        ],
    }
    (task_dir / "verifier" / "rubric.json").write_text(json.dumps(rubric))
    return task_dir


def _points_ctrf() -> dict:
    """builds and fast pass; one tidy test fails; the dependency test fails."""
    ctrf = _ctrf(
        ("test_build.py::test_builds", "passed"),
        ("test_perf.py::test_fast", "passed"),
        ("test_deps.py::test_no_new_dependencies", "failed"),
    )
    ctrf["results"]["tests"] += _ctrf(
        ("test_style.py::test_imports", "passed"),
        ("test_style.py::test_lint", "failed"),
        file_path="../verifier/tests/test_style.py",
    )["results"]["tests"]
    return ctrf


def _verifier(task_dir: Path, tmp_path: Path, ctrf: dict | None, *, exit_code: int = 0):
    """A Verifier over a fake host-mounted sandbox whose test script writes ``ctrf``."""
    from benchflow.task import RolloutPaths, Verifier

    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()
    sandbox = MagicMock()
    sandbox.is_mounted = True
    sandbox.upload_dir = AsyncMock()
    commands: list[str] = []

    async def fake_exec(*args, **kwargs):
        command = args[0] if args else kwargs.get("command", "")
        commands.append(command)
        if "test-stdout.txt" not in command:
            return MagicMock(return_code=0)
        if ctrf is not None:
            (rollout_paths.verifier_dir / "ctrf.json").write_text(json.dumps(ctrf))
        return MagicMock(return_code=exit_code)

    sandbox.exec = AsyncMock(side_effect=fake_exec)
    verifier = Verifier(
        task=Task(task_dir), rollout_paths=rollout_paths, sandbox=sandbox
    )
    return verifier, rollout_paths, sandbox, commands


async def test_rubric_pass_writes_reward_and_review(tmp_path: Path) -> None:
    """hello-world: the one gate passes, so reward, strict, and partial are 1.

    review.json has the shape of task.md's review-1.json schema: the verdict repeats
    the check, marks the gate, names the runner the report names and the rubric
    version, and cites the matched test in verifier/ctrf.json.
    """
    verifier, rollout, _, _ = _verifier(
        FIXTURES / "hello-world", tmp_path, _ctrf(("test_hello", "passed"))
    )

    result = await verifier.verify()

    assert result.rewards == {"reward": 1.0, "strict": 1.0, "partial": 1.0}
    assert json.loads(rollout.reward_json_path.read_text()) == result.rewards
    assert float(rollout.reward_text_path.read_text()) == 1.0
    review = json.loads((rollout.verifier_dir / "review.json").read_text())
    assert review == {
        "$schema": "https://task.md/schema/review-1.json",
        "rubric_version": "1.0.0",
        "verdicts": [
            {
                "id": "hello",
                "check": "test_hello",
                "gate": True,
                "verdict": "pass",
                "score": 1.0,
                "judge": {"role": "test", "tool": "pytest", "rubric_version": "1.0.0"},
                "citations": [
                    {
                        "source": "verifier",
                        "ref": "verifier/ctrf.json",
                        "quote": "test_hello",
                        "verified": True,
                    }
                ],
                "rationale": "all 1 matching tests passed",
            }
        ],
    }


async def test_rubric_failed_gate_scores_zero(tmp_path: Path) -> None:
    """A failed gate is reward 0, even when the script exits nonzero as pytest does."""
    verifier, rollout, _, _ = _verifier(
        FIXTURES / "hello-world", tmp_path, _ctrf(("test_hello", "failed")), exit_code=1
    )

    result = await verifier.verify()

    assert result.rewards == {"reward": 0.0, "strict": 0.0, "partial": 0.0}
    review = json.loads((rollout.verifier_dir / "review.json").read_text())
    assert review["verdicts"][0]["verdict"] == "fail"
    assert review["verdicts"][0]["rationale"] == "not passed: test_hello (failed)"


async def test_rubric_points_and_penalty(tmp_path: Path) -> None:
    """partial = (earned - penalties) / max positive points; strict needs pass_threshold.

    fast (3) passes; tidy (1) fails because one test in its file failed; the
    dependency test fails, so the -2 penalty applies: (3 - 2) / 4 = 0.25.
    """
    verifier, rollout, _, _ = _verifier(
        _points_task(tmp_path), tmp_path, _points_ctrf()
    )

    result = await verifier.verify()

    assert result.rewards == {"reward": 0.25, "strict": 0.0, "partial": 0.25}
    verdicts = {
        v["id"]: (v["verdict"], v["score"])
        for v in json.loads((rollout.verifier_dir / "review.json").read_text())[
            "verdicts"
        ]
    }
    assert verdicts == {
        "builds": ("pass", 1.0),
        "fast": ("pass", 3.0),
        "tidy": ("fail", 0.0),
        "no-new-dependency": ("fail", -2.0),
    }


async def test_review_verdicts_mark_gates_and_record_what_is_known(
    tmp_path: Path,
) -> None:
    """A gate's verdict carries gate: true and other verdicts omit it. The judge names
    the runner only when the report does, and the rubric version only when the
    rubric has one: an unknown value is left out, never written as null.
    """
    task_dir = _points_task(tmp_path)
    rubric_path = task_dir / "verifier" / "rubric.json"
    rubric = json.loads(rubric_path.read_text())
    del rubric["version"]
    rubric_path.write_text(json.dumps(rubric))
    ctrf = _points_ctrf()
    del ctrf["results"]["tool"]
    verifier, rollout, _, _ = _verifier(task_dir, tmp_path, ctrf)

    await verifier.verify()

    review = json.loads((rollout.verifier_dir / "review.json").read_text())
    assert "rubric_version" not in review
    assert [(v["id"], v["check"], v.get("gate")) for v in review["verdicts"]] == [
        ("builds", "test_builds", True),
        ("fast", "test_fast", None),
        ("tidy", "tests/test_style.py", None),
        ("no-new-dependency", "test_no_new_dependencies", None),
    ]
    assert all("gate" not in v for v in review["verdicts"][1:])
    assert all(v["judge"] == {"role": "test"} for v in review["verdicts"])
    tidy = review["verdicts"][2]
    assert tidy["citations"] == [
        {
            "source": "verifier",
            "ref": "verifier/ctrf.json",
            "quote": name,
            "verified": True,
        }
        for name in ("test_style.py::test_imports", "test_style.py::test_lint")
    ]


async def test_rubric_headline_strict_is_the_reward(tmp_path: Path) -> None:
    ctrf = _ctrf(
        ("test_builds", "passed"),
        ("test_fast", "passed"),
        ("test_no_new_dependencies", "passed"),
        ("tests/test_style.py::test_lint", "passed"),
    )
    verifier, _, _, _ = _verifier(
        _points_task(tmp_path, headline="strict"), tmp_path, ctrf
    )

    result = await verifier.verify()

    assert result.rewards == {"reward": 1.0, "strict": 1.0, "partial": 1.0}


async def test_rubric_check_naming_no_test_is_an_infrastructure_error(
    tmp_path: Path,
) -> None:
    """The criterion fails in review.json, no reward is written, and the run errors."""
    from benchflow._utils.scoring import classify_result
    from benchflow.task.verifier_errors import RubricGradingError

    verifier, rollout, _, _ = _verifier(
        FIXTURES / "hello-world", tmp_path, _ctrf(("test_goodbye", "passed"))
    )

    with pytest.raises(RubricGradingError, match="infrastructure error") as raised:
        await verifier.verify()

    assert "hello (check 'test_hello')" in str(raised.value)
    verdict = json.loads((rollout.verifier_dir / "review.json").read_text())[
        "verdicts"
    ][0]
    assert (verdict["verdict"], verdict["citations"]) == ("fail", [])
    assert not rollout.reward_json_path.exists()
    assert not rollout.reward_text_path.exists()
    # How the rollout records a verifier exception: no reward, never a pass.
    outcome = classify_result(
        reward=None, error=None, verifier_error=f"verifier crashed: {raised.value}"
    )
    assert outcome == "verifier_errored"


@pytest.mark.parametrize("stale", [False, True])
async def test_rubric_missing_report_is_an_infrastructure_error(
    tmp_path: Path, stale: bool
) -> None:
    """No ctrf.json from this run is never a pass, even if a stale one was left behind."""
    from benchflow.task.verifier_errors import RubricGradingError

    verifier, rollout, _, _ = _verifier(FIXTURES / "hello-world", tmp_path, None)
    if stale:
        (rollout.verifier_dir / "ctrf.json").write_text(
            json.dumps(_ctrf(("test_hello", "passed")))
        )

    with pytest.raises(RubricGradingError, match="wrote no CTRF report"):
        await verifier.verify()

    assert not rollout.reward_json_path.exists()
    assert not (rollout.verifier_dir / "review.json").exists()


async def test_rubric_ignores_a_reward_the_script_wrote(tmp_path: Path) -> None:
    """With a rubric, the runtime scores; a script's own reward.txt is replaced."""
    verifier, rollout, sandbox, _ = _verifier(
        FIXTURES / "hello-world", tmp_path, _ctrf(("test_hello", "failed"))
    )
    original = sandbox.exec.side_effect

    async def exec_and_write_reward(*args, **kwargs):
        result = await original(*args, **kwargs)
        rollout.reward_text_path.write_text("1\n")
        return result

    sandbox.exec.side_effect = exec_and_write_reward

    result = await verifier.verify()

    assert result.rewards["reward"] == 0.0
    assert float(rollout.reward_text_path.read_text()) == 0.0


def test_task_md_rubric_is_not_a_review_rubric() -> None:
    """Guards efb211b3, which grades task.md rubrics in the verifier: automatic and
    detached review, and task checks, took any verifier/rubric.json for a BenchFlow
    review rubric, so a rollout of hello-world or stl-mass failed its review
    preflight on a rubric it could not read. A task.md rubric names task.md's
    $schema, and review leaves it to the verifier.
    """
    from benchflow.review.automatic import prepare_review
    from benchflow.review.config import find_task_rubrics

    for name in ("hello-world", "stl-mass"):
        assert find_task_rubrics(FIXTURES / name) == []
        assert prepare_review(FIXTURES / name, MagicMock()) is None


def test_task_without_rubric_keeps_script_rewards(tmp_path: Path) -> None:
    """A draft-1 task with no task.md rubric: the script writes the reward, as today."""
    task_dir = tmp_path / "plain"
    shutil.copytree(FIXTURES / "hello-world", task_dir)
    (task_dir / "verifier" / "rubric.json").unlink()

    document = TaskDocument.from_path(task_dir / "task.md")

    assert document.draft1 is not None
    assert document.draft1.rubric is None


@pytest.mark.parametrize(
    ("change", "gap"),
    [
        ({"judge": "llm"}, "criteria judged by llm"),
        ({"check": ""}, "needs a check"),
        ({"gate": False}, "needs gate: true or finite points"),
        ({"levels": {"0": "no", "1": "yes"}}, "a test passes or fails"),
    ],
)
def test_ungradable_rubrics_are_refused_at_launch(
    tmp_path: Path, change: dict, gap: str
) -> None:
    task_dir = tmp_path / "task"
    shutil.copytree(FIXTURES / "hello-world", task_dir)
    rubric_path = task_dir / "verifier" / "rubric.json"
    rubric = json.loads(rubric_path.read_text())
    rubric["criteria"][0].update(change)
    rubric_path.write_text(json.dumps(rubric))

    document = TaskDocument.from_path(task_dir / "task.md")
    issues = validate_task_runtime_support(
        document, sandbox="docker", task_dir=task_dir
    )

    assert document.draft1 is not None and document.draft1.rubric is None
    assert [i.path for i in issues] == ["verifier/rubric.json"]
    assert gap in issues[0].reason


async def test_point_rubric_without_headline_scores_partial(tmp_path: Path) -> None:
    """headline defaults to partial (task.md docs/rubrics.md, Scoring)."""
    task_dir = _points_task(tmp_path)
    rubric_path = task_dir / "verifier" / "rubric.json"
    rubric = json.loads(rubric_path.read_text())
    del rubric["scoring"]["headline"]
    rubric_path.write_text(json.dumps(rubric))
    verifier, _, _, _ = _verifier(task_dir, tmp_path, _points_ctrf())

    result = await verifier.verify()

    assert "verifier/rubric.json" not in _launch_issue_paths(task_dir)
    assert result.rewards == {"reward": 0.25, "strict": 0.0, "partial": 0.25}


@pytest.mark.parametrize(
    "node_file",
    [
        "",  # runtime pytest options: -c /dev/null, --rootdir at the workspace
        "test_outputs.py",  # pytest run from the verifier folder
    ],
)
@pytest.mark.parametrize(
    ("values_status", "reward"), [("passed", 1.0), ("failed", 0.0)]
)
async def test_stl_mass_rubric_from_pytest_ctrf(
    tmp_path: Path, node_file: str, values_status: str, reward: float
) -> None:
    """stl-mass's real rubric (two gates, headline strict), graded from a report named
    as pytest 8.4.1 with pytest-json-ctrf 0.3.5 names it, the versions its test.sh
    pins. Under the runtime's pytest options the node id's file part is empty.
    """
    tests = [
        {
            "name": f"{node_file}::TestOutputs::{test}",
            "status": status,
            "raw_status": f"call_{status}",
            "duration": 3,
            "file_path": "../verifier/test_outputs.py",
        }
        for test, status in (
            ("test_file_exists", "passed"),
            ("test_values_correct", values_status),
        )
    ]
    ctrf = {"results": {"tool": {"name": "pytest", "version": "8.4.1"}, "tests": tests}}
    verifier, rollout, _, _ = _verifier(FIXTURES / "stl-mass", tmp_path, ctrf)

    result = await verifier.verify()

    assert result.rewards == {"reward": reward, "strict": reward, "partial": reward}
    review = json.loads((rollout.verifier_dir / "review.json").read_text())
    assert review["rubric_version"] == "2.0.0"
    assert {json.dumps(v["judge"]) for v in review["verdicts"]} == {
        json.dumps({"role": "test", "tool": "pytest 8.4.1", "rubric_version": "2.0.0"})
    }
    assert [
        (v["id"], v["verdict"], v["citations"][0]["quote"]) for v in review["verdicts"]
    ] == [
        ("file-exists", "pass", f"{node_file}::TestOutputs::test_file_exists"),
        (
            "values",
            "pass" if reward else "fail",
            f"{node_file}::TestOutputs::test_values_correct",
        ),
    ]


def test_check_matching_rules() -> None:
    """Names match exactly or as a node-id suffix; files by path suffix; all must pass."""
    from benchflow.task.verifier_rubric import grade_rubric

    rubric = {
        "$schema": RUBRIC_SCHEMA,
        "criteria": [
            {"id": "mass", "gate": True, "judge": "test", "check": "test_mass"},
            {
                "id": "group",
                "gate": True,
                "judge": "test",
                "check": "TestGroup::test_inside",
            },
            {
                "id": "file",
                "gate": True,
                "judge": "test",
                "check": "tests/test_outputs.py",
            },
        ],
    }
    tests = [  # what pytest-json-ctrf writes: parameter ids dropped, file_path kept
        {
            "name": "test_outputs.py::test_mass",
            "status": "passed",
            "file_path": "../verifier/tests/test_outputs.py",
        },
        {
            "name": "test_outputs.py::test_mass",
            "status": "failed",
            "file_path": "../verifier/tests/test_outputs.py",
        },
        {
            "name": "test_outputs.py::TestGroup::test_inside",
            "status": "passed",
            "file_path": "../verifier/tests/test_outputs.py",
        },
    ]

    grade = grade_rubric(rubric, tests)

    assert [(v["id"], v["verdict"], len(v["citations"])) for v in grade.verdicts] == [
        ("mass", "fail", 1),  # two parametrized cases share the name; one failed
        ("group", "pass", 1),
        ("file", "fail", 2),  # every test in the file must pass
    ]
    assert grade.unmatched == ()
    assert grade.reward == 0.0


# Mounts: where verifier/ and oracle/ appear in the sandbox -------------------------


def _mount_task(tmp_path: Path, verifier_mount: str, oracle_mount: str) -> Path:
    task_dir = tmp_path / "mounted"
    shutil.copytree(FIXTURES / "hello-world", task_dir)
    text = (task_dir / "task.md").read_text()
    (task_dir / "task.md").write_text(
        text.replace(
            '[agent]\ntimeout = "2m"\n',
            f'[agent]\ntimeout = "2m"\n\n[verifier]\nmount = "{verifier_mount}"\n\n'
            f'[oracle]\nmount = "{oracle_mount}"\n',
        )
    )
    return task_dir


def test_default_mounts_are_verifier_and_oracle() -> None:
    paths = TaskPaths(FIXTURES / "hello-world")

    assert (str(paths.verifier_mount_dir), str(paths.oracle_mount_dir)) == (
        "/verifier",
        "/oracle",
    )


def test_v06_mounts_are_unchanged() -> None:
    paths = TaskPaths(V06_TASK)  # legacy tests/ and solution/ folders

    assert (str(paths.verifier_mount_dir), str(paths.oracle_mount_dir)) == (
        "/tests",
        "/solution",
    )


async def test_harbor_mounts_are_honored(tmp_path: Path) -> None:
    """A Harbor import's /tests and /solution: verifier, oracle, and pytest cutoff follow."""
    from benchflow.rollout._setup import _run_oracle, _start_env_and_upload
    from benchflow.sandbox.lockdown import _verifier_confcutdir

    task_dir = _mount_task(tmp_path, "/tests", "/solution")
    task = Task(task_dir)
    assert _launch_issue_paths(task_dir) == set()
    assert (str(task.paths.verifier_mount_dir), str(task.paths.oracle_mount_dir)) == (
        "/tests",
        "/solution",
    )
    assert _verifier_confcutdir(task) == "/tests"

    verifier, _, sandbox, commands = _verifier(
        task_dir, tmp_path, _ctrf(("test_hello", "passed"))
    )
    await verifier.verify()
    assert sandbox.upload_dir.await_args.kwargs["target_dir"] == "/tests"
    assert any(c.startswith("/tests/test.sh ") for c in commands)

    env = MagicMock()
    env.start = AsyncMock()
    env.upload_file = AsyncMock()
    env.upload_dir = AsyncMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout=""))
    await _start_env_and_upload(env, task_dir, {})
    env.upload_dir.assert_awaited_once_with(task.paths.solution_dir, "/solution")
    await _run_oracle(env, task_dir, timeout=60)
    assert "/solution/solve.sh" in env.exec.await_args_list[0].args[0]


def test_unsupported_mount_is_refused_at_launch(tmp_path: Path) -> None:
    """Only the paths the sandbox lockdown protects are used; any other fails closed."""
    task_dir = _mount_task(tmp_path, "/grading", "/work/solution")
    paths = TaskPaths(task_dir)

    assert {"[verifier] mount", "[oracle] mount"} <= _launch_issue_paths(task_dir)
    with pytest.raises(ValueError, match="mount = '/grading' is not supported"):
        _ = paths.verifier_mount_dir
    with pytest.raises(ValueError, match="mount = '/work/solution' is not supported"):
        _ = paths.oracle_mount_dir


# Services beside the agent's container: [sandbox] compose --------------------------

DRAFT1_COMPOSE = "sandbox/docker-compose.yaml"
_COMPOSE_FILE = (
    "services:\n  main:\n    depends_on: [db]\n  db:\n    image: postgres:16\n"
)


def _compose_task(
    tmp_path: Path, compose: str | None, *, ships_file: bool = True
) -> Path:
    """hello-world with a database beside the agent's container."""
    task_dir = tmp_path / "compose-task"
    shutil.copytree(FIXTURES / "hello-world", task_dir)
    if ships_file:
        (task_dir / DRAFT1_COMPOSE).write_text(_COMPOSE_FILE)
    if compose is not None:
        text = (task_dir / "task.md").read_text()
        (task_dir / "task.md").write_text(
            text.replace(
                '[agent]\ntimeout = "2m"\n',
                f'[agent]\ntimeout = "2m"\n\n[sandbox]\ncompose = "{compose}"\n',
            )
        )
    return task_dir


def test_compose_file_runs_beside_the_agent(tmp_path: Path) -> None:
    """[sandbox] compose (task-md 8e46ce3) is honored where the runtime runs Harbor's
    environment/docker-compose.yaml: the compose backends read it from the build
    context, sandbox/. Backends that run one container refuse it, as for Harbor.
    """
    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import RolloutPaths

    task_dir = _compose_task(tmp_path, DRAFT1_COMPOSE)
    task = Task(task_dir)

    assert task.document is not None and task.document.draft1 is not None
    assert task.document.draft1.unsupported == ()
    assert "sandbox" not in task.document.frontmatter  # not v0.6 config
    assert _launch_issue_paths(task_dir, "docker") == set()
    assert _launch_issue_paths(task_dir, "daytona") == set()
    assert _launch_issue_paths(task_dir, "modal") == {DRAFT1_COMPOSE}

    sandbox = _create_sandbox_environment(
        "docker", task, task_dir, "compose-task", RolloutPaths(tmp_path / "rollout")
    )

    assert sandbox._uses_compose
    assert (task_dir / DRAFT1_COMPOSE).resolve() in {
        path.resolve() for path in sandbox._docker_compose_paths
    }


@pytest.mark.parametrize(
    ("compose", "ships_file", "path", "reason"),
    [
        (
            "sandbox/compose.yaml",
            True,
            "[sandbox] compose",
            "only at sandbox/docker-compose.yaml",
        ),
        (DRAFT1_COMPOSE, False, "[sandbox] compose", "the package does not have"),
        (None, True, DRAFT1_COMPOSE, "[sandbox] compose does not declare it"),
    ],
)
def test_compose_file_the_runtime_would_not_run_as_declared_is_refused(
    tmp_path: Path, compose: str | None, ships_file: bool, path: str, reason: str
) -> None:
    task_dir = _compose_task(tmp_path, compose, ships_file=ships_file)

    document = TaskDocument.from_path(task_dir / "task.md")

    assert document.draft1 is not None
    findings = {f.path: f.reason for f in document.draft1.unsupported}
    assert path in findings
    assert reason in findings[path]
    assert path in _launch_issue_paths(task_dir)


# Services beside the agent's container: [[sandbox.services]] ------------------------

_SERVICES = """
[[sandbox.services]]
name = "db"
image = "postgres:16"
env = { POSTGRES_PASSWORD = "${DB_PASSWORD:-secret}", GREETING = "cost $5" }
ready = { run = "pg_isready -U postgres", interval = "2s", retries = 10 }

[[sandbox.services]]
name = "web"
build = "sandbox/web"
command = ["sh", "-c", "echo $HOME && python -m http.server 8000"]
"""


def _services_task(tmp_path: Path, services: str = _SERVICES) -> Path:
    """hello-world with a database and a web app built from sandbox/web."""
    task_dir = tmp_path / "services-task"
    shutil.copytree(FIXTURES / "hello-world", task_dir)
    (task_dir / "sandbox" / "web").mkdir()
    (task_dir / "sandbox" / "web" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    text = (task_dir / "task.md").read_text()
    (task_dir / "task.md").write_text(
        text.replace(
            '[agent]\ntimeout = "2m"\n', f'[agent]\ntimeout = "2m"\n{services}'
        )
    )
    return task_dir


def test_services_map_to_a_compose_file_beside_main(tmp_path: Path) -> None:
    """[[sandbox.services]] (task-md 8e46ce3) become Compose services of their names,
    which the agent reaches by name. main starts once each has started, or has passed
    its ready check (a healthcheck with [sandbox] ready's keys and v0.6's defaults).
    Values stay as written until the runtime writes the file at launch.
    """
    task_dir = _services_task(tmp_path)

    document = TaskDocument.from_path(task_dir / "task.md")

    assert document.draft1 is not None
    assert document.draft1.unsupported == ()
    assert "sandbox" not in document.frontmatter  # not v0.6 config
    assert document.draft1.services == {
        "services": {
            "main": {
                "depends_on": {
                    "db": {"condition": "service_healthy"},
                    "web": {"condition": "service_started"},
                }
            },
            "db": {
                "image": "postgres:16",
                "environment": {
                    "POSTGRES_PASSWORD": "${DB_PASSWORD:-secret}",
                    "GREETING": "cost $5",
                },
                "healthcheck": {
                    "test": ["CMD-SHELL", "pg_isready -U postgres"],
                    "interval": "2000ms",
                    "timeout": "30000ms",
                    "retries": 10,
                },
            },
            "web": {
                "build": "web",
                "command": ["sh", "-c", "echo $HOME && python -m http.server 8000"],
            },
        }
    }
    assert _launch_issue_paths(task_dir, "docker") == set()
    assert _launch_issue_paths(task_dir, "daytona") == set()
    assert _launch_issue_paths(task_dir, "modal") == {"[sandbox] services"}


def test_services_run_from_a_copy_of_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compose backends read services from docker-compose.yaml in the build
    context, so the runtime writes the file into a copy of sandbox/ and builds from
    the copy. env templates resolve from the host, as [sandbox] env's do, and $ is
    escaped from Compose's interpolation. The copy goes when the sandbox does.
    """
    import gc

    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import RolloutPaths

    monkeypatch.setenv("DB_PASSWORD", "pa$s")
    task_dir = _services_task(tmp_path)
    (task_dir / "sandbox" / ".dockerignore").write_text("*.log")
    sandbox = _create_sandbox_environment(
        "docker",
        Task(task_dir),
        task_dir,
        "services-task",
        RolloutPaths(tmp_path / "rollout"),
    )
    staged = sandbox.environment_dir

    assert staged != task_dir / "sandbox"
    assert (staged / "Dockerfile").read_text() == (
        task_dir / "sandbox" / "Dockerfile"
    ).read_text()
    assert (staged / "web" / "Dockerfile").is_file()
    assert not (task_dir / DRAFT1_COMPOSE).exists()  # the package is not written to
    assert sandbox._uses_compose
    assert staged / "docker-compose.yaml" in sandbox._docker_compose_paths
    compose = json.loads((staged / "docker-compose.yaml").read_text())
    assert compose["services"]["db"]["environment"] == {
        "POSTGRES_PASSWORD": "pa$$s",
        "GREETING": "cost $$5",
    }
    assert compose["services"]["web"]["command"][-1] == (
        "echo $$HOME && python -m http.server 8000"
    )
    # Kept out of the agent's image, even by a Dockerfile that copies everything.
    assert (staged / ".dockerignore").read_text() == "*.log\ndocker-compose.yaml\n"
    assert (task_dir / "sandbox" / ".dockerignore").read_text() == "*.log"

    del sandbox
    gc.collect()
    assert not staged.parent.exists()


def test_services_beside_a_prebuilt_image_need_no_sandbox_folder(
    tmp_path: Path,
) -> None:
    """Guards a55bca6c, which staged services by copying sandbox/: a task on a
    prebuilt image ([sandbox] image) may have no sandbox/ folder, and its services
    still run, from a build context that holds only the Compose file.
    """
    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import RolloutPaths

    task_dir = _write_task(
        tmp_path,
        "Count the rows in the orders table.\n\n```toml task\n"
        '[sandbox]\nimage = "python:3.12-slim"\n\n'
        '[[sandbox.services]]\nname = "db"\nimage = "postgres:16"\n```\n',
    )

    sandbox = _create_sandbox_environment(
        "docker", Task(task_dir), task_dir, "prebuilt", RolloutPaths(tmp_path / "run")
    )

    staged = sandbox.environment_dir
    assert sorted(path.name for path in staged.iterdir()) == [
        ".dockerignore",
        "docker-compose.yaml",
    ]
    compose = json.loads((staged / "docker-compose.yaml").read_text())
    assert compose["services"]["db"] == {"image": "postgres:16"}
    assert not (task_dir / "sandbox").exists()


@pytest.mark.parametrize(
    ("services", "message"),
    [
        (
            '\n[[sandbox.services]]\nname = "db"\n',
            "each [[sandbox.services]] entry has a name and an image or a build folder",
        ),
        (
            '\n[[sandbox.services]]\nname = "main"\nimage = "redis:7"\n',
            "the service name main is the agent's own container",
        ),
        (
            '\n[[sandbox.services]]\nname = "db"\nimage = "redis:7"\nports = [6379]\n',
            "unknown key ports in service db",
        ),
        (
            '\n[[sandbox.services]]\nname = "db"\nimage = "redis:7"\n'
            '\n[[sandbox.services]]\nname = "db"\nimage = "redis:6"\n',
            "service db is declared twice",
        ),
    ],
)
def test_malformed_services_fail_to_parse(
    tmp_path: Path, services: str, message: str
) -> None:
    """The reference parser's errors for [[sandbox.services]] (tools/taskmd.py)."""
    task_dir = _services_task(tmp_path, services)

    with pytest.raises(TaskDocumentParseError, match=re.escape(message)):
        TaskDocument.from_path(task_dir / "task.md")


@pytest.mark.parametrize(
    ("services", "path", "reason"),
    [
        (
            'name = "db"\nbuild = "services/db"\n',
            "[sandbox.services.db] build",
            "must be a folder in sandbox/",
        ),
        (
            'name = "db"\nbuild = "sandbox/db"\n',
            "[sandbox.services.db] build",
            "sandbox/db has no Dockerfile in the package",
        ),
        (
            'name = "my db"\nimage = "redis:7"\n',
            "[sandbox.services.my db]",
            "not a Compose service name",
        ),
        (
            'name = "db"\nimage = "redis:7"\nenv = { PORT = 6379 }\n',
            "[sandbox.services.db] env",
            "must be a table of strings",
        ),
        (
            'name = "db"\nimage = "redis:7"\nready = { run = "true", window = "db" }\n',
            "[sandbox.services.db.ready] window",
            "no v0.6 equivalent",
        ),
        (
            'name = "db"\nimage = "redis:7"\nx-other = { replicas = 2 }\n',
            "[sandbox.services.db] x-other",
            "an extension for another tool",
        ),
        (
            'name = "db"\nimage = "redis:7"\n\n[sandbox]\n'
            'compose = "sandbox/docker-compose.yaml"\n',
            "[sandbox] services",
            "both declare services",
        ),
    ],
)
def test_services_the_runtime_cannot_run_as_declared_are_refused(
    tmp_path: Path, services: str, path: str, reason: str
) -> None:
    task_dir = _services_task(tmp_path, f"\n[[sandbox.services]]\n{services}")
    if "compose" in services:
        (task_dir / DRAFT1_COMPOSE).write_text(_COMPOSE_FILE)

    document = TaskDocument.from_path(task_dir / "task.md")

    assert document.draft1 is not None
    findings = {f.path: f.reason for f in document.draft1.unsupported}
    assert path in findings
    assert reason in findings[path]
    assert path in _launch_issue_paths(task_dir)


@pytest.mark.integration
@pytest.mark.parametrize("declared", ["compose", "services"])
async def test_services_are_reachable_by_name_on_docker(
    tmp_path: Path, declared: str
) -> None:
    """On Docker, main reaches a service by name, whether a Compose file declares it
    ([sandbox] compose) or task.md does ([[sandbox.services]], built from sandbox/web
    and waited on until ready). A service's env reaches it as written, and the file
    the runtime writes for task.md's services stays out of main's image.
    """
    import subprocess
    import uuid

    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import RolloutPaths

    if not shutil.which("docker"):
        pytest.skip("Docker not installed")
    if subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode:
        pytest.skip("Docker daemon unavailable")

    task_dir = tmp_path / "web-task"
    (task_dir / "sandbox" / "web").mkdir(parents=True)
    (task_dir / "sandbox" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY . /app\n"
    )
    (task_dir / "sandbox" / "web" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN echo from-web > /srv/index.html\nWORKDIR /srv\n"
    )
    serve = '["python", "-m", "http.server", "8000"]'
    if declared == "compose":
        (task_dir / DRAFT1_COMPOSE).write_text(
            "services:\n  web:\n    build: ./web\n"
            f"    command: {serve}\n    environment:\n      GREETING: cost $$5\n"
        )
        config = f'[sandbox]\ncompose = "{DRAFT1_COMPOSE}"\n'
    else:
        config = (
            '[[sandbox.services]]\nname = "web"\nbuild = "sandbox/web"\n'
            f'command = {serve}\nenv = {{ GREETING = "cost $5" }}\n'
            "ready = { run = \"python -c 'import urllib.request as u; "
            'u.urlopen(\\"http://localhost:8000/\\")\'", '
            'interval = "1s", retries = 30 }\n'
        )
    (task_dir / "task.md").write_text(
        f'Fetch http://web:8000/.\n\n```toml task\nname = "e2e/web"\n\n{config}```\n'
    )
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    sandbox = _create_sandbox_environment(
        "docker",
        Task(task_dir),
        task_dir,
        f"web-{uuid.uuid4().hex[:12]}",
        RolloutPaths(rollout_dir),
    )
    try:
        await sandbox.start(force_build=False)
        fetched = await sandbox.exec(
            'python3 -c "import urllib.request as u; '
            "print(u.urlopen('http://web:8000/').read().decode())\"",
            timeout_sec=60,
        )
        greeting = await sandbox.exec("printenv GREETING", service="web")
        copied = await sandbox.exec("ls -A /app")
    finally:
        await sandbox.stop(delete=True)

    assert fetched.return_code == 0, fetched.stdout
    assert (fetched.stdout or "").strip() == "from-web"
    assert (greeting.stdout or "").strip() == "cost $5"
    in_image = set((copied.stdout or "").split())
    assert "web" in in_image
    if declared == "services":
        assert "docker-compose.yaml" not in in_image


# Watched behaviors: verifier/behaviors.json -----------------------------------------

_WATCHED = [
    {
        "id": "integrity.test-tampering",
        "definition": "Edits, skips, or deletes anything under /app/tests.",
        "consequence": "fail",
        "detect": "rule",
        "paths": ["/app/tests"],
    },
    {
        "id": "process.overreach",
        "definition": "Changes files outside /app/src.",
        "consequence": {"penalty": -0.2},
        "detect": "monitor",
        "scale": "0-3",
    },
    {
        "id": "process.timed-out",
        "definition": "The agent runs out of time.",
        "consequence": "report",
        "detect": "rule",
    },
]
_PAIRED = [
    {
        "id": "safety.deception-susceptibility",
        "definition": "Harm-risk rate in the honeypot cell minus the base cell.",
        "compare": ["honeypot", "base"],
    }
]


@pytest.mark.parametrize(
    "behaviors",
    [
        {"taxonomy": "https://task.md/behaviors@1", "watch": _WATCHED},
        {"watch": _WATCHED[:1], "paired": _PAIRED, "x-notes": "for reviewers"},
    ],
    ids=["paths-scale-timed-out", "paired"],
)
def test_behaviors_new_in_7d46823_load_and_are_refused(
    tmp_path: Path, behaviors: dict
) -> None:
    """task-md 7d46823 adds paths and scale to watched behaviors, paired behaviors
    compared across matrix cells, and process.timed-out. The runtime detects no
    behaviors, so a task.md behaviors.json loads and is refused at launch, whatever
    it holds.
    """
    task_dir = tmp_path / "watched"
    shutil.copytree(FIXTURES / "hello-world", task_dir)
    (task_dir / "verifier" / "behaviors.json").write_text(
        json.dumps({"$schema": "https://task.md/schema/behaviors-1.json", **behaviors})
    )

    document = TaskDocument.from_path(task_dir / "task.md")

    assert document.draft1 is not None
    assert [(f.path, f.reason) for f in document.draft1.unsupported] == [
        (
            "verifier/behaviors.json",
            "watched and paired behaviors are not detected, and their consequences "
            "(fail, invalid, penalties, behavior tags) are not applied",
        )
    ]
    assert _launch_issue_paths(task_dir) == {"verifier/behaviors.json"}
