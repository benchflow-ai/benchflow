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
        ├── instruction.md
        ├── task.toml
        ├── environment/
        │   ├── Dockerfile
        │   └── ...
        ├── solution/
        │   ├── solve.sh
        │   └── ...
        └── tests/
            ├── test.sh
            └── ...
    """

    CONFIG_FILENAME = "task.toml"

    def __init__(self, task_dir: Path | str) -> None:
        self.task_dir = Path(task_dir).resolve()

    @property
    def instruction_path(self) -> Path:
        return self.task_dir / "instruction.md"

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
    def solution_dir(self) -> Path:
        return self.task_dir / "solution"

    @property
    def solve_path(self) -> Path:
        return self.solution_dir / "solve.sh"

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    @property
    def test_path(self) -> Path:
        return self.tests_dir / "test.sh"

    def is_valid(self, disable_verification: bool = False) -> bool:
        return (
            self.config_path.exists()
            and self.environment_dir.exists()
            and self.instruction_path.exists()
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

    logs_dir: PurePosixPath = PurePosixPath("/logs")
    agent_dir: PurePosixPath = logs_dir / "agent"
    verifier_dir: PurePosixPath = logs_dir / "verifier"
    artifacts_dir: PurePosixPath = logs_dir / "artifacts"
    tests_dir: PurePosixPath = PurePosixPath("/tests")
    solution_dir: PurePosixPath = PurePosixPath("/solution")
    reward_text_path: PurePosixPath = verifier_dir / "reward.txt"
    reward_json_path: PurePosixPath = verifier_dir / "reward.json"
