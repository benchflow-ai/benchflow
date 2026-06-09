"""Path models for tasks and rollouts — internalized from Harbor.

Terminology:
    - TaskPaths: filesystem layout of a task specification ($T$)
    - RolloutPaths: output directory of a single rollout (episode)
    - SandboxPaths: static mount points inside the sandbox container
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_SCRIPT_ARTIFACT_SUFFIXES = {
    ".js",
    ".mjs",
    ".py",
    ".rb",
    ".sh",
    ".ts",
}


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

    def has_verifier_entrypoint(self) -> bool:
        """Return whether the selected verifier package has something runnable.

        Historical callers treated ``test.sh`` as the only verifier entrypoint.
        Native packages can now select verifier strategies from
        ``verifier/verifier.md``, so validity needs to follow the selected
        strategy rather than the legacy script filename alone.
        """

        if not self.tests_dir.is_dir():
            return False

        try:
            from benchflow.task.verifier_document import load_verifier_document

            document = load_verifier_document(self.tests_dir)
        except Exception:
            return False
        if document is None:
            return self.test_path.exists()

        strategy = document.selected_strategy
        if strategy.type == "script":
            local_scripts = local_script_strategy_files(
                strategy.command,
                verifier_dir=self.tests_dir,
            )
            return bool(local_scripts) and all(path.is_file() for path in local_scripts)
        if strategy.type == "llm-judge":
            if not _strategy_file_exists(
                strategy.rubric_path,
                verifier_dir=self.tests_dir,
            ):
                return False
            return strategy.context_file is None or _strategy_file_exists(
                strategy.context_file,
                verifier_dir=self.tests_dir,
            )
        if strategy.type == "reward-kit":
            return _reward_kit_strategy_files_exist(
                root=strategy.root_path,
                entrypoint=strategy.entrypoint,
                criteria=strategy.criteria_path,
                verifier_dir=self.tests_dir,
            )
        if strategy.type == "ors-episode":
            return bool(strategy.inputs)
        return strategy.type == "agent-judge"

    def is_valid(self, disable_verification: bool = False) -> bool:
        has_legacy_definition = (
            self.config_path.exists() and self.instruction_path.exists()
        )
        has_document_definition = self.task_document_path.exists()
        return (
            (has_legacy_definition or has_document_definition)
            and self.environment_dir.exists()
            and (disable_verification or self.has_verifier_entrypoint())
        )


def local_script_strategy_files(
    command: str | None,
    *,
    verifier_dir: Path,
) -> tuple[Path, ...]:
    """Return local verifier files referenced by a script strategy command.

    Commands often use interpreter wrappers such as ``bash test.sh`` or
    ``uv run python verify.py``. The verifier runner will fail later if those
    local artifacts are absent, so path validation should inspect likely script
    arguments instead of only argv[0].
    """

    if command is None:
        return (Path(),)
    try:
        tokens = shlex.split(command)
    except ValueError:
        return (Path(),)
    if not tokens:
        return (Path(),)
    paths: list[Path] = []
    for token in tokens:
        if not _looks_like_local_script_artifact(token):
            continue
        relative = _safe_relative_path(token)
        if relative is None:
            paths.append(Path())
            continue
        paths.append(verifier_dir / Path(*relative.parts))
    return tuple(paths)


def _looks_like_local_script_artifact(token: str) -> bool:
    if not token or token.startswith("-") or "://" in token or any(c.isspace() for c in token):
        return False
    path = PurePosixPath(token)
    if path.is_absolute():
        return False
    return "/" in token or Path(token).suffix in _SCRIPT_ARTIFACT_SUFFIXES


def _reward_kit_strategy_files_exist(
    *,
    root: str | None,
    entrypoint: str | None,
    criteria: str | None,
    verifier_dir: Path,
) -> bool:
    if root is None:
        return False
    root_path = _safe_relative_path(root)
    entrypoint_path = _safe_relative_path(entrypoint or "reward.py")
    if root_path is None or entrypoint_path is None:
        return False
    if not (verifier_dir / Path(*root_path.parts) / Path(*entrypoint_path.parts)).is_file():
        return False
    return criteria is None or _strategy_file_exists(
        criteria,
        verifier_dir=verifier_dir,
    )


def _strategy_file_exists(value: str | None, *, verifier_dir: Path) -> bool:
    if value is None:
        return False
    path = Path(value)
    if path.is_absolute():
        return path.is_file()
    relative = _safe_relative_path(value)
    if relative is None:
        return False
    return (verifier_dir / Path(*relative.parts)).is_file()


def _safe_relative_path(value: str) -> PurePosixPath | None:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        return None
    return path


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
    def reward_details_json_path(self) -> Path:
        return self.verifier_dir / "reward-details.json"

    @property
    def reward_kit_manifest_path(self) -> Path:
        return self.verifier_dir / "reward-kit-manifest.json"

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
    reward_text_path: PurePosixPath = verifier_dir / "reward.txt"
    reward_json_path: PurePosixPath = verifier_dir / "reward.json"
    reward_details_json_path: PurePosixPath = verifier_dir / "reward-details.json"
    reward_kit_manifest_path: PurePosixPath = verifier_dir / "reward-kit-manifest.json"
