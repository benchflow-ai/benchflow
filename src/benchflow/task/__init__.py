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
    ArtifactConfig,
    Author,
    HealthcheckConfig,
    JudgeVerifierConfig,
    MCPServerConfig,
    MultiStepRewardStrategy,
    NetworkMode,
    PackageInfo,
    SandboxConfig,
    SolutionConfig,
    StepConfig,
    TaskConfig,
    TaskOS,
    TpuSpec,
    VerifierConfig,
    VerifierEnvironmentMode,
    VerifierHardeningConfig,
)
from benchflow.task.config import (
    AgentConfig as TaskAgentConfig,
)
from benchflow.task.document import (
    TASK_DOCUMENT_FILENAME,
    TaskDocument,
    TaskDocumentParseError,
    render_task_md_from_legacy,
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
    RubricNotFoundError,
    Verifier,
    VerifierOutputParseError,
    VerifierResult,
)

__all__ = [
    # Task ($T$)
    "Task",
    "TaskPaths",
    "TaskConfig",
    "TASK_DOCUMENT_FILENAME",
    "TaskDocument",
    "TaskDocumentParseError",
    # Configuration ($C$)
    "SandboxConfig",
    "VerifierConfig",
    "JudgeVerifierConfig",
    "VerifierHardeningConfig",
    "HealthcheckConfig",
    "TaskAgentConfig",
    "SolutionConfig",
    "MCPServerConfig",
    "ArtifactConfig",
    "StepConfig",
    "TpuSpec",
    "NetworkMode",
    "TaskOS",
    "VerifierEnvironmentMode",
    "MultiStepRewardStrategy",
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
    "RubricNotFoundError",
    "VerifierOutputParseError",
    "AddTestsDirError",
    "DownloadVerifierDirError",
    # Utilities
    "resolve_env_vars",
    "render_task_md_from_legacy",
]
