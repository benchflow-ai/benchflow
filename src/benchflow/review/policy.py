"""Configuration for automatic rubric review of completed rollouts.

The policy lives outside :mod:`benchflow.rollout` so the detached review
runner and every configuration surface share one set of defaults.  Secret
environment values are deliberately omitted from serialized artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from benchflow._utils.config import (
    normalize_agent_name,
    normalize_reasoning_effort,
)

DEFAULT_REVIEWER_AGENT = "codex-acp"
DEFAULT_REVIEWER_MODEL = "openai/gpt-5.6-sol"
DEFAULT_REVIEWER_REASONING_EFFORT = "xhigh"
DEFAULT_REVIEWER_TIMEOUT_SEC = 1800
DEFAULT_REVIEWER_MAX_RETRIES = 1


@dataclass
class RubricReviewConfig:
    """Policy for the automatic reviewer that runs when a task has a rubric.

    ``environment=None`` inherits the source rollout's sandbox backend.  This
    keeps Docker, Daytona, and other providers on the same execution path.
    ``agent_env`` is runtime-only and is never persisted with its values.
    """

    enabled: bool = True
    agent: str = DEFAULT_REVIEWER_AGENT
    model: str = DEFAULT_REVIEWER_MODEL
    reasoning_effort: str = DEFAULT_REVIEWER_REASONING_EFFORT
    environment: str | None = None
    timeout_sec: int = DEFAULT_REVIEWER_TIMEOUT_SEC
    max_retries: int = DEFAULT_REVIEWER_MAX_RETRIES
    allow_open_network: bool = False
    agent_env: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("rubric_review.enabled must be a boolean")
        self.agent = normalize_agent_name(self.agent)
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("rubric_review.model must be a non-empty string")
        self.model = self.model.strip()
        normalized_effort = normalize_reasoning_effort(self.reasoning_effort)
        if normalized_effort is None:
            raise ValueError("rubric_review.reasoning_effort cannot be empty")
        self.reasoning_effort = normalized_effort
        if self.environment is not None:
            if not isinstance(self.environment, str) or not self.environment.strip():
                raise ValueError(
                    "rubric_review.environment must be null or a non-empty string"
                )
            self.environment = self.environment.strip()
        if isinstance(self.timeout_sec, bool) or not isinstance(self.timeout_sec, int):
            raise ValueError("rubric_review.timeout_sec must be an integer")
        if self.timeout_sec <= 0:
            raise ValueError("rubric_review.timeout_sec must be positive")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise ValueError("rubric_review.max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("rubric_review.max_retries must be non-negative")
        if type(self.allow_open_network) is not bool:
            raise ValueError("rubric_review.allow_open_network must be a boolean")
        if not isinstance(self.agent_env, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.agent_env.items()
        ):
            raise ValueError("rubric_review.agent_env must map strings to strings")

    @classmethod
    def coerce(
        cls, value: RubricReviewConfig | dict[str, Any] | None
    ) -> RubricReviewConfig:
        """Normalize a dataclass, YAML/SDK mapping, or omitted policy."""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**cast("dict[str, Any]", value))
        raise ValueError("rubric_review must be a mapping or RubricReviewConfig")

    def to_config_artifact(self) -> dict[str, Any]:
        """Return the reproducibility-safe form written to run artifacts."""

        artifact = self.to_mapping()
        artifact.pop("agent_env")
        artifact["agent_env_keys"] = sorted(self.agent_env)
        return artifact

    def to_mapping(self) -> dict[str, Any]:
        """Return the complete runtime form used by private worker payloads."""

        return {
            "enabled": self.enabled,
            "agent": self.agent,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "environment": self.environment,
            "timeout_sec": self.timeout_sec,
            "max_retries": self.max_retries,
            "allow_open_network": self.allow_open_network,
            "agent_env": dict(self.agent_env),
        }
