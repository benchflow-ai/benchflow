"""Backward-compatible imports for progressive-disclosure user contracts."""

from benchflow.contracts.user import (
    BaseUser,
    FunctionUser,
    PassthroughUser,
    RoundResult,
)

__all__ = ["BaseUser", "FunctionUser", "PassthroughUser", "RoundResult"]
