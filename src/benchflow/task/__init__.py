"""Task package — native BenchFlow types with RL-first terminology.

- Task, TaskPaths: problem specification ($T$)
- TaskConfig, SandboxConfig, VerifierConfig, AgentConfig: configuration ($C$)
- RolloutPaths: rollout output paths
- SandboxPaths: container mount points
- Verifier, VerifierResult: reward function ($V$)
- resolve_env_vars: env template resolution
"""

from benchflow.task.config import (
    ORG_NAME_PATTERN,
    Author,
    MCPServerConfig,
    PackageInfo,
    SandboxConfig,
    SolutionConfig,
    TaskConfig,
    VerifierConfig,
)
from benchflow.task.config import (
    AgentConfig as TaskAgentConfig,
)
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import (
    RolloutPaths,
    SandboxPaths,
    TaskPaths,
)
from benchflow.task.task import Task
from benchflow.task.verifier import (
    AddTestsDirError,
    DownloadVerifierDirError,
    RewardFileEmptyError,
    RewardFileNotFoundError,
    Verifier,
    VerifierOutputParseError,
    VerifierResult,
)

__all__ = [
    # Task ($T$)
    "Task",
    "TaskPaths",
    "TaskConfig",
    # Configuration ($C$)
    "SandboxConfig",
    "VerifierConfig",
    "TaskAgentConfig",
    "SolutionConfig",
    "MCPServerConfig",
    "PackageInfo",
    "Author",
    "ORG_NAME_PATTERN",
    # Rollout paths
    "RolloutPaths",
    # Sandbox paths
    "SandboxPaths",
    # Verifier ($V$)
    "Verifier",
    "VerifierResult",
    "RewardFileEmptyError",
    "RewardFileNotFoundError",
    "VerifierOutputParseError",
    "AddTestsDirError",
    "DownloadVerifierDirError",
    # Utilities
    "resolve_env_vars",
]
