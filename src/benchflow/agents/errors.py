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

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from benchflow.agents.usage_limits import (
    format_reset,
    is_usage_limit_text,
    parse_limit_text,
    parse_rate_limit_info,
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
        """``usage limit reached on login L: 7-day window, resets 2026-10-02 16:00 UTC (<detail>)``.

        The window and the reset are left out when the agent did not say them.
        """
        where = f" on login {self.login}" if self.login else ""
        facts = []
        if self.window:
            facts.append(f"{self.window} window")
        if self.resets_at is not None:
            facts.append(f"resets {format_reset(self.resets_at)}")
        told = f": {', '.join(facts)}" if facts else ""
        return f"usage limit reached{where}{told} ({self.detail})"

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

    @classmethod
    def from_rate_limit_info(
        cls, info: Mapping[str, Any] | None, *, detail: str
    ) -> UsageLimitError | None:
        """The error a rejected rate-limit record describes, or None.

        The record is Claude Code's ``rate_limit_event`` (see
        :func:`benchflow.agents.usage_limits.parse_rate_limit_info`).
        """
        limit = parse_rate_limit_info(info, detail=detail)
        if limit is None:
            return None
        return cls(limit.detail, window=limit.window, resets_at=limit.resets_at)

    @classmethod
    def from_result(cls, result: Any) -> UsageLimitError | None:
        """The usage limit a finished trial ended on, or None.

        ``result`` is a ``RolloutResult`` (``bf.run``, ``Evaluation`` results,
        ``bf.load_trial(...).result``) or a ``result.json`` mapping. The
        login, window and reset come from the trial's ``usage_limit_info``.
        """
        if isinstance(result, Mapping):
            data: Mapping[str, Any] = result
            rollout_dir = None
        else:
            data = {
                "error": getattr(result, "error", None),
                "error_category": getattr(result, "error_category", None),
            }
            rollout_dir = getattr(result, "rollout_dir", None)
        error = data.get("error")
        text = error if isinstance(error, str) else ""
        if data.get("error_category") != cls.category and not (
            text.lower().startswith("usage limit reached") or is_usage_limit_text(text)
        ):
            return None
        info = data.get("usage_limit_info")
        if not isinstance(info, Mapping) and rollout_dir is not None:
            try:
                saved = json.loads((Path(rollout_dir) / "result.json").read_text())
            except (OSError, ValueError):
                saved = {}
            info = saved.get("usage_limit_info") if isinstance(saved, dict) else None
        if isinstance(info, Mapping):
            resets = info.get("resets_at")
            try:
                resets_at = datetime.fromisoformat(resets) if resets else None
            except (TypeError, ValueError):
                resets_at = None
            return cls(
                str(info.get("detail") or text or "usage limit reached"),
                login=info.get("login"),
                window=info.get("window"),
                resets_at=resets_at,
            )
        return (
            _from_description(text)
            or cls.from_text(text)
            or cls(text or "usage limit reached")
        )


# What :meth:`UsageLimitError.describe` writes before its ``(<detail>)``.
_DESCRIBED = re.compile(
    r"usage limit reached(?: on login (?P<login>.+?))?"
    r"(?:: (?:(?P<window>[\w -]+) window)?(?:, )?"
    r"(?:resets (?P<resets>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC)?)?"
)


def _opening_paren(text: str) -> int | None:
    """Where the parenthesized group that ends ``text`` opens."""
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        depth += {")": 1, "(": -1}.get(text[i], 0)
        if depth == 0:
            return i
    return None


def _from_description(text: str) -> UsageLimitError | None:
    """The error :meth:`UsageLimitError.describe` wrote ``text`` for, or None.

    A trial records the description as its ``error``; this reads it back
    when the trial's ``usage_limit_info`` is not at hand.
    """
    i = _opening_paren(text) if text.endswith(")") else None
    if i is None or i < 1 or text[i - 1] != " ":
        return None
    described = _DESCRIBED.fullmatch(text[: i - 1])
    if described is None:
        return None
    resets = described["resets"]
    return UsageLimitError(
        text[i + 1 : -1],
        login=described["login"],
        window=described["window"],
        resets_at=(
            datetime.strptime(resets, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            if resets
            else None
        ),
    )


def _rebuild_usage_limit(
    detail: str, login: str | None, window: str | None, resets_at: datetime | None
) -> UsageLimitError:
    return UsageLimitError(detail, login=login, window=window, resets_at=resets_at)
