"""Path models for tasks and rollouts — internalized from Harbor.

Terminology:
    - TaskPaths: filesystem layout of a task specification ($T$)
    - RolloutPaths: output directory of a single rollout (episode)
    - SandboxPaths: static mount points inside the sandbox container
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class TaskPaths:
    """Filesystem layout for a task directory.

    ::

        task_dir/
        ├── task.md              # native unified format, or:
        ├── instruction.md        # legacy split format
        ├── task.toml             # legacy split format
        ├── environment/
        │   ├── Dockerfile
        │   └── ...
        ├── oracle/              # native reference/oracle files, or:
        ├── solution/            # legacy oracle files
        │   ├── solve.sh
        │   └── ...
        └── verifier/            # native verifier files, or:
        └── tests/               # legacy verifier files
            ├── test.sh
            └── ...
    """

    CONFIG_FILENAME = "task.toml"
    DOCUMENT_FILENAME = "task.md"
    NATIVE_ORACLE_DIRNAME = "oracle"
    LEGACY_SOLUTION_DIRNAME = "solution"
    NATIVE_VERIFIER_DIRNAME = "verifier"
    LEGACY_TESTS_DIRNAME = "tests"

    def __init__(self, task_dir: Path | str) -> None:
        self.task_dir = Path(task_dir).resolve()

    @property
    def instruction_path(self) -> Path:
        return self.task_dir / "instruction.md"

    @property
    def task_document_path(self) -> Path:
        return self.task_dir / self.DOCUMENT_FILENAME

    @property
    def readme_path(self) -> Path:
        return self.task_dir / "README.md"

    @property
    def gitignore_path(self) -> Path:
        return self.task_dir / ".gitignore"

    @property
    def config_path(self) -> Path:
        return self.task_dir / self.CONFIG_FILENAME

    @property
    def environment_dir(self) -> Path:
        return self.task_dir / "environment"

    @property
    def oracle_dir(self) -> Path:
        return self.task_dir / self.NATIVE_ORACLE_DIRNAME

    @property
    def legacy_solution_dir(self) -> Path:
        return self.task_dir / self.LEGACY_SOLUTION_DIRNAME

    @property
    def solution_dir(self) -> Path:
        if self.oracle_dir.exists():
            return self.oracle_dir
        return self.legacy_solution_dir

    @property
    def uses_native_oracle_dir(self) -> bool:
        return self.oracle_dir.exists()

    @property
    def solve_path(self) -> Path:
        return self.solution_dir / "solve.sh"

    @property
    def verifier_source_dir(self) -> Path:
        return self.task_dir / self.NATIVE_VERIFIER_DIRNAME

    @property
    def legacy_tests_dir(self) -> Path:
        return self.task_dir / self.LEGACY_TESTS_DIRNAME

    @property
    def tests_dir(self) -> Path:
        if self.verifier_source_dir.exists():
            return self.verifier_source_dir
        return self.legacy_tests_dir

    @property
    def uses_native_verifier_dir(self) -> bool:
        return self.verifier_source_dir.exists()

    @property
    def test_path(self) -> Path:
        return self.tests_dir / "test.sh"

    @property
    def verifier_document_path(self) -> Path:
        return self.tests_dir / "verifier.md"

    def is_valid(self, disable_verification: bool = False) -> bool:
        has_legacy_definition = (
            self.config_path.exists() and self.instruction_path.exists()
        )
        has_document_definition = self.task_document_path.exists()
        return (
            (has_legacy_definition or has_document_definition)
            and self.environment_dir.exists()
            and (disable_verification or self.test_path.exists())
        )


@dataclass(frozen=True)
class RolloutPaths:
    """Output directory for a single rollout (episode).


    ::

        rollout_dir/
        ├── agent/          # Agent logs
        ├── verifier/       # Verifier logs & reward files
        ├── artifacts/      # Collected artifacts
        ├── trajectory/     # ACP trajectory JSONL
        ├── config.json     # Rollout configuration
        ├── result.json     # Rollout result
        └── rewards.jsonl   # Reward events
    """

    rollout_dir: Path

    def mkdir(self) -> None:
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.verifier_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config_path(self) -> Path:
        return self.rollout_dir / "config.json"

    @property
    def agent_dir(self) -> Path:
        return self.rollout_dir / "agent"

    @property
    def artifacts_dir(self) -> Path:
        return self.rollout_dir / "artifacts"

    @property
    def artifacts_manifest_path(self) -> Path:
        return self.artifacts_dir / "manifest.json"

    @property
    def verifier_dir(self) -> Path:
        return self.rollout_dir / "verifier"

    @property
    def test_stdout_path(self) -> Path:
        return self.verifier_dir / "test-stdout.txt"

    @property
    def test_stderr_path(self) -> Path:
        return self.verifier_dir / "test-stderr.txt"

    @property
    def reward_text_path(self) -> Path:
        return self.verifier_dir / "reward.txt"

    @property
    def reward_json_path(self) -> Path:
        return self.verifier_dir / "reward.json"

    @property
    def result_path(self) -> Path:
        return self.rollout_dir / "result.json"

    @property
    def exception_message_path(self) -> Path:
        return self.rollout_dir / "exception.txt"

    @property
    def log_path(self) -> Path:
        return self.rollout_dir / "rollout.log"


@dataclass(frozen=True)
class SandboxPaths:
    """Static mount points inside the sandbox container.

    These are always POSIX paths since sandboxes run Linux.
    """

    logs_dir: PurePosixPath = PurePosixPath("/logs")  # noqa: RUF009
    agent_dir: PurePosixPath = logs_dir / "agent"
    verifier_dir: PurePosixPath = logs_dir / "verifier"
    artifacts_dir: PurePosixPath = logs_dir / "artifacts"
    verifier_code_dir: PurePosixPath = PurePosixPath("/verifier")  # noqa: RUF009
    tests_dir: PurePosixPath = PurePosixPath("/tests")  # noqa: RUF009
    oracle_dir: PurePosixPath = PurePosixPath("/oracle")  # noqa: RUF009
    solution_dir: PurePosixPath = PurePosixPath("/solution")  # noqa: RUF009
    instruction_path: PurePosixPath = PurePosixPath("/instruction.md")  # noqa: RUF009
    task_document_path: PurePosixPath = PurePosixPath("/task.md")  # noqa: RUF009
    reward_text_path: PurePosixPath = verifier_dir / "reward.txt"
    reward_json_path: PurePosixPath = verifier_dir / "reward.json"
