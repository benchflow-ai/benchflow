"""Verifier result model and exception hierarchy.

Extracted from ``benchflow.task.verifier`` as a pure leaf cluster. The base
``VerifierOutputParseError`` is defined before its subclasses
(``UnsupportedVerifierStrategyError``, ``AgentJudgeInputError``,
``ORSEpisodeInputError``) so subclassing resolves at import time.
"""

from __future__ import annotations

import errno
from typing import Any

from pydantic import BaseModel

from benchflow._utils.reachable_write import PrivateStorageError
from benchflow._utils.text import describe_exception
from benchflow.rewards.validation import RewardOutOfRange
from benchflow.sandbox._engine_api import EngineError, EngineUnavailable
from benchflow.sandbox.daytona_start import _is_daytona_error


class VerifierResult(BaseModel):
    """Result from the verifier — reward dict."""

    model_config = {"strict": True}

    rewards: dict[str, Any] | None = None


class RewardFileEmptyError(Exception):
    """A reward file the verifier left empty.

    ``file`` is its name under the rollout (``verifier/reward.txt``), which
    the messages name instead of the host path; ``path`` is the file itself.
    """

    def __init__(self, message: str, *, file: str = "", path: Any = None) -> None:
        super().__init__(message)
        self.file = file
        self.path = path


class RewardFileNotFoundError(Exception):
    pass


class VerifierOutputParseError(Exception):
    pass


class VerifierSetupError(VerifierOutputParseError):
    """BenchFlow's own setup step failed before the test command ran."""


class VerifierInfrastructureError(RuntimeError):
    """A controller operation failed independently of submitted evidence."""


class VerifierExecutionError(VerifierInfrastructureError):
    """The sandbox API failed, rather than returning the test command's status."""


class RewardOutOfRangeError(VerifierOutputParseError, RewardOutOfRange):
    """A reward file holds a finite number outside the task's range (#1178).

    Carries :class:`~benchflow.rewards.validation.RewardOutOfRange`'s ``key``,
    ``value`` and ``reward_range``; ``file`` is the reward file's name as the
    trial folder shows it (``verifier/reward.txt``). The rollout scores 0
    with a warning and retains ``reward_out_of_range_info``.
    """

    def __init__(self, error: RewardOutOfRange, *, file: str) -> None:
        RewardOutOfRange.__init__(
            self,
            error.key,
            error.value,
            error.reward_range if error.declared else None,
            source=file,
        )
        self.file = file


class AddTestsDirError(VerifierInfrastructureError):
    """The backend could not upload the protected verifier package."""


class DownloadVerifierDirError(VerifierInfrastructureError):
    """The backend could not capture verifier evidence."""


def verifier_infrastructure_failure(error: BaseException) -> bool:
    """Keep typed controller failures distinct, including through wrappers.

    Earlier verifier code can damage sandbox state, but cannot manufacture a
    transport exception or a controller resource failure. Inspect types and
    errno, never diagnostic text that submitted code could print.
    """
    # Error classification is a failure-path operation. Ordinary local runs
    # must not load HTTP clients or model-judge dependencies just to start.
    import httpx

    from benchflow.rewards.builtins import JudgeScoringError
    from benchflow.rewards.llm import JudgeEnvironmentError
    from benchflow.taskmd.judging import JudgeError

    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        if isinstance(
            error,
            (
                VerifierInfrastructureError,
                PrivateStorageError,
                JudgeError,
                JudgeScoringError,
                JudgeEnvironmentError,
                EngineError,
                EngineUnavailable,
                httpx.HTTPError,
                ConnectionError,
                MemoryError,
            ),
        ) or _is_daytona_error(error):
            return True
        if isinstance(error, OSError) and error.errno in (
            errno.EIO,
            errno.ENOSPC,
            errno.EDQUOT,
            errno.ENOMEM,
            errno.EMFILE,
            errno.ENFILE,
            errno.EROFS,
        ):
            return True
        cause = error.__cause__
        if cause is None:
            break
        error = cause
    return False


def verifier_failure_reason(error: BaseException) -> str:
    """Name the failed step and retain its underlying controller cause."""
    from benchflow.rewards.builtins import JudgeScoringError
    from benchflow.taskmd.judging import JudgeError

    reasons = [describe_exception(error)]
    seen = {id(error)}
    # These judge boundaries withhold raw provider text. Their chained causes
    # can contain credentials or responses and must not be rendered here.
    while not isinstance(error, (JudgeScoringError, JudgeError)):
        cause = error.__cause__
        if cause is None or id(cause) in seen:
            break
        seen.add(id(cause))
        reasons.append(describe_exception(cause))
        error = cause
    return "; caused by ".join(reasons)


class PluginGuardLoadError(Exception):
    """Guard diagnostics prevent accepting a pass; the rollout scores 0.

    Output and markers do not prove their cause: test.sh's user can write both.
    """

    def __init__(self, message: str, *, guard_copy_diagnostic: str = "") -> None:
        super().__init__(message)
        # Keep hardening's interpreter list without repeating older error text
        # that assigns blame or says the trial is unscored (FIX-49 merge).
        self.guard_copy_diagnostic = guard_copy_diagnostic


class PluginGuardMarkersUnreadableError(Exception):
    """Raised when the plugin guard's markers could not all be read back.

    The read failed, timed out or stopped at its limits, so a refusal may be
    among those missed. The verifier user can cause this by changing the
    markers; the rollout scores 0 rather than discarding the trial.
    """


class VerifierStartUnconfirmedError(Exception):
    """The test command's verifier-writable start receipt was not found."""


class RubricNotFoundError(VerifierInfrastructureError):
    """Raised when an llm-judge verifier cannot locate its rubric file."""


class UnsupportedVerifierStrategyError(VerifierOutputParseError):
    """Raised when ``verifier/verifier.md`` selects a non-executable strategy."""


class AgentJudgeInputError(VerifierOutputParseError):
    """Raised when an agent-judge strategy cannot read declared inputs."""


class ORSEpisodeInputError(VerifierOutputParseError):
    """Raised when an ors-episode strategy cannot read declared reward evidence."""
