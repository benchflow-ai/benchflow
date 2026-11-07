"""Architecture contracts imported by the rollout kernel.

The names resolve on first use (PEP 562): ``benchflow.contracts.user``, which
task prompts import, runs this file first, and the agent protocol here pulls
in the ACP package, which a run that starts no ACP agent never needs.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

_LAZY = {
    "RolloutPlanes": "benchflow.contracts.planes",
    "default_rollout_planes": "benchflow.contracts.planes",
    "Reward": "benchflow.rewards.protocol",
    "RewardFunc": "benchflow.rewards.protocol",
    "VerifyResult": "benchflow.rewards.protocol",
    "Agent": "benchflow.agents.protocol",
    "AgentCapabilities": "benchflow.agents.protocol",
    "AgentProtocolError": "benchflow.agents.protocol",
    "AskUserHandler": "benchflow.agents.protocol",
    "AskUserRequest": "benchflow.agents.protocol",
    "Session": "benchflow.agents.protocol",
    "StopReason": "benchflow.agents.protocol",
    "BaseUser": "benchflow.contracts.user",
    "DocumentNudgeUser": "benchflow.contracts.user",
    "FunctionUser": "benchflow.contracts.user",
    "ModelDocumentNudgeUser": "benchflow.contracts.user",
    "PassthroughUser": "benchflow.contracts.user",
    "RoundResult": "benchflow.contracts.user",
    "EnvHandle": "benchflow.environment.protocol",
    "Environment": "benchflow.environment.protocol",
    "EnvState": "benchflow.environment.protocol",
    "ReadinessProbe": "benchflow.environment.protocol",
    "StateSnapshot": "benchflow.environment.protocol",
    "ExecResult": "benchflow.sandbox.protocol",
    "ImageBuilder": "benchflow.sandbox.protocol",
    "ImageConfig": "benchflow.sandbox.protocol",
    "ImageRef": "benchflow.sandbox.protocol",
    "Sandbox": "benchflow.sandbox.protocol",
    "SandboxImage": "benchflow.sandbox.protocol",
    "SandboxSnapshotNotSupported": "benchflow.sandbox.protocol",
    "SandboxStartupFailure": "benchflow.sandbox.protocol",
}

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


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
