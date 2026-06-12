"""Architecture contracts imported by the rollout kernel."""

from benchflow.agents.protocol import (
    Agent,
    AgentCapabilities,
    AgentProtocolError,
    AskUserHandler,
    AskUserRequest,
    Session,
    StopReason,
)
from benchflow.contracts.planes import RolloutPlanes, default_rollout_planes
from benchflow.contracts.user import (
    BaseUser,
    DocumentNudgeUser,
    FunctionUser,
    ModelDocumentNudgeUser,
    PassthroughUser,
    RoundResult,
)
from benchflow.environment.protocol import (
    EnvHandle,
    Environment,
    EnvState,
    ReadinessProbe,
    StateSnapshot,
)
from benchflow.rewards.protocol import Reward, RewardFunc, VerifyResult
from benchflow.sandbox.protocol import (
    ExecResult,
    ImageBuilder,
    ImageConfig,
    ImageRef,
    Sandbox,
    SandboxImage,
    SandboxSnapshotNotSupported,
    SandboxStartupFailure,
)

__all__ = [
    "Agent",
    "AgentCapabilities",
    "AgentProtocolError",
    "AskUserHandler",
    "AskUserRequest",
    "BaseUser",
    "DocumentNudgeUser",
    "Environment",
    "EnvHandle",
    "EnvState",
    "ExecResult",
    "FunctionUser",
    "ImageBuilder",
    "ImageConfig",
    "ImageRef",
    "ModelDocumentNudgeUser",
    "PassthroughUser",
    "ReadinessProbe",
    "Reward",
    "RewardFunc",
    "RoundResult",
    "RolloutPlanes",
    "Sandbox",
    "SandboxImage",
    "SandboxSnapshotNotSupported",
    "SandboxStartupFailure",
    "Session",
    "StateSnapshot",
    "StopReason",
    "VerifyResult",
    "default_rollout_planes",
]
