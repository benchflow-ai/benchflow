"""Verifier result model and exception hierarchy.

Extracted from ``benchflow.task.verifier`` as a pure leaf cluster. The base
``VerifierOutputParseError`` is defined before its subclasses
(``UnsupportedVerifierStrategyError``, ``AgentJudgeInputError``,
``ORSEpisodeInputError``) so subclassing resolves at import time.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class VerifierResult(BaseModel):
    """Result from the verifier — reward dict."""

    model_config = {"strict": True}

    rewards: dict[str, Any] | None = None


class RewardFileEmptyError(Exception):
    pass


class RewardFileNotFoundError(Exception):
    pass


class VerifierOutputParseError(Exception):
    pass


class AddTestsDirError(Exception):
    pass


class DownloadVerifierDirError(Exception):
    pass


class RubricNotFoundError(Exception):
    """Raised when an llm-judge verifier cannot locate its rubric file."""


class UnsupportedVerifierStrategyError(VerifierOutputParseError):
    """Raised when ``verifier/verifier.md`` selects a non-executable strategy."""


class AgentJudgeInputError(VerifierOutputParseError):
    """Raised when an agent-judge strategy cannot read declared inputs."""


class ORSEpisodeInputError(VerifierOutputParseError):
    """Raised when an ors-episode strategy cannot read declared reward evidence."""
