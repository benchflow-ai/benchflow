"""Loading task.md draft-1 packages through the v0.6 document model.

The fixtures under ``tests/fixtures/task_md_draft1`` are verbatim copies of
four examples from the task.md draft-1 specification. Draft-1 files have no
frontmatter: the instruction comes first, then typed fenced blocks.
"""

from __future__ import annotations

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

    assert '```json\n{"main_part_mass": 12345.67, "material_id": 42}\n```' in (
        document.instruction
    )
    assert document.instruction.endswith(
        "The result counts as correct within **0.1%**."
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
    assert config.task.version == "1.1.0"
    assert config.agent.timeout_sec == 120
    assert config.agent.network_mode is None
    assert config.sandbox.docker_image is None  # built from sandbox/Dockerfile
    assert config.sandbox.memory_mb == 2048  # v0.6 default: the file sets none
    assert _launch_issue_paths(task_dir) == set()
    assert task.document is not None and task.document.draft1 is not None
    assert {f.path for f in task.document.draft1.ignored} == {
        "title",
        "verifier/rubric.json",
    }


def test_flaky_retry_maps_config_and_refuses_launch() -> None:
    task_dir = FIXTURES / "flaky-retry"
    config = Task(task_dir).config

    assert config.agent.timeout_sec == 900  # "15m"
    assert config.agent.network_mode == NetworkMode.NO_NETWORK  # "none"
    assert config.sandbox.docker_image == "python:3.12-slim"
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
    from benchflow.task import RolloutPaths, Verifier

    task_dir = _mount_task(tmp_path, "/tests", "/solution")
    task = Task(task_dir)
    assert _launch_issue_paths(task_dir) == set()
    assert (str(task.paths.verifier_mount_dir), str(task.paths.oracle_mount_dir)) == (
        "/tests",
        "/solution",
    )
    assert _verifier_confcutdir(task) == "/tests"

    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()
    sandbox = MagicMock()
    sandbox.is_mounted = True
    sandbox.upload_dir = AsyncMock()
    commands: list[str] = []

    async def fake_exec(*args, **kwargs):
        command = args[0] if args else kwargs.get("command", "")
        commands.append(command)
        if "test-stdout.txt" in command:
            rollout_paths.reward_text_path.write_text("1\n")
        return MagicMock(return_code=0)

    sandbox.exec = AsyncMock(side_effect=fake_exec)
    verifier = Verifier(task=task, rollout_paths=rollout_paths, sandbox=sandbox)
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
