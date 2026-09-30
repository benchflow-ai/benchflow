"""Agent-plane exception types with no runtime imports.

:class:`UsageLimitError` is shared by every way an agent runs: the ACP client
raises it when claude-agent-acp reports its typed quota failure (or the CLI's
own "You've hit your ... limit" text), and a native harness raises it from a
Messages API answer whose ``anthropic-ratelimit-unified-status`` is
``rejected`` (:meth:`UsageLimitError.from_headers`). A rollout records it as
the unscored ``usage_limit`` category and never retries it; an
``Evaluation`` stops starting trials and raises it once its trials finish, so
a caller can catch it and run the rest on another login.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, ClassVar

from benchflow.agents.usage_limits import (
    format_reset,
    parse_limit_text,
    parse_unified_headers,
)
from benchflow.errors import Fault, UserError


class AgentProtocolError(Exception):
    """Contract-level agent protocol failure."""

    message: str


class UsageLimitError(AgentProtocolError, UserError):
    """The login the agent runs on has used up its subscription usage.

    Nothing the agent or BenchFlow retries on that login can succeed before
    ``resets_at``: switch to another login, or wait.

    Attributes:
        detail: the agent's own words, for example ``You've hit your weekly
            limit · resets Oct 2, 4pm (UTC)``.
        login: which login ran out, as a label (``CLAUDE_CODE_OAUTH_TOKEN
            (environment)``, ``claude login (~/.claude/.credentials.json)``);
            never the token. None until the rollout names it.
        window: ``5-hour``, ``7-day``, ``7-day Opus``, ``7-day Sonnet``, or
            None when the message does not say.
        resets_at: when the window resets (aware, UTC), or None.
        result: set by ``Evaluation.run``: the job's ``EvaluationResult``
            (everything that finished, with its ``job_dir``).
    """

    category: ClassVar[str] = "usage_limit"
    fault: ClassVar[Fault] = "setup"

    def __init__(
        self,
        detail: str,
        *,
        login: str | None = None,
        window: str | None = None,
        resets_at: datetime | None = None,
    ) -> None:
        self.detail = detail.strip()
        self.login = login
        self.window = window
        self.resets_at = resets_at
        self.result: Any = None
        self.message = self.detail
        self.hint = (
            "switch to another login or wait for the reset, then resume the job "
            "(`bench eval resume <job_dir>`); `bench doctor` shows each login's headroom"
        )
        super().__init__(self.describe())

    def describe(self) -> str:
        """``usage limit reached on login L: 7-day window, resets 2026-10-02 16:00 UTC (<detail>)``."""
        where = f" on login {self.login}" if self.login else ""
        window = f"{self.window} window, " if self.window else ""
        return (
            f"usage limit reached{where}: {window}resets "
            f"{format_reset(self.resets_at)} ({self.detail})"
        )

    def __str__(self) -> str:
        return self.describe()

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild_usage_limit,
            (self.detail, self.login, self.window, self.resets_at),
        )

    def with_login(self, login: str | None) -> UsageLimitError:
        """A copy naming ``login`` (the rollout knows it; the ACP client does not)."""
        return UsageLimitError(
            self.detail, login=login, window=self.window, resets_at=self.resets_at
        )

    def to_dict(self) -> dict[str, Any]:
        """The fields a result records (no secret: the login is a label)."""
        return {
            "login": self.login,
            "window": self.window,
            "resets_at": self.resets_at.isoformat() if self.resets_at else None,
            "detail": self.detail,
        }

    @classmethod
    def from_text(
        cls, text: str | None, *, now: datetime | None = None
    ) -> UsageLimitError | None:
        """The error a usage-limit message describes, or None when it is not one."""
        info = parse_limit_text(text, now=now)
        if info is None:
            return None
        return cls(info.detail, window=info.window, resets_at=info.resets_at)

    @classmethod
    def from_headers(
        cls, headers: Mapping[str, Any] | None, *, detail: str | None = None
    ) -> UsageLimitError | None:
        """The error a Messages API answer's unified rate-limit headers describe.

        None unless ``anthropic-ratelimit-unified-status`` is ``rejected``.
        """
        headroom = parse_unified_headers(headers)
        if headroom is None or not headroom.limited:
            return None
        text = detail or (
            "HTTP 429 with anthropic-ratelimit-unified-status: rejected"
            + (f" ({headroom.window} window)" if headroom.window else "")
        )
        return cls(text, window=headroom.window, resets_at=headroom.resets_at)


def _rebuild_usage_limit(
    detail: str, login: str | None, window: str | None, resets_at: datetime | None
) -> UsageLimitError:
    return UsageLimitError(detail, login=login, window=window, resets_at=resets_at)
