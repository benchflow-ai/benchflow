"""Backward-compatible imports for progressive-disclosure user contracts."""

from benchflow.contracts.user import (
    BaseUser,
    DocumentNudgeUser,
    FunctionUser,
    ModelDocumentNudgeUser,
    PassthroughUser,
    RoundResult,
)

__all__ = [
    "BaseUser",
    "DocumentNudgeUser",
    "FunctionUser",
    "ModelDocumentNudgeUser",
    "PassthroughUser",
    "RoundResult",
]
