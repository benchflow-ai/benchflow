"""Reviewer execution options shared by automatic and detached review."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from benchflow._utils.config import normalize_agent_name, normalize_reasoning_effort
from benchflow._utils.config_redaction import _should_record_env_entry
from benchflow.sandbox.providers import is_known_provider, providers_phrase

# Keep the runtime reproducible; a mutable python:3.13-slim tag can drift.
REVIEWER_IMAGE = (
    "python@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91"
)

REVIEWER_AGENT_TIMEOUT_SEC = 1800
# The reviewer's ACP idle watchdog: 600 s, the rollout default reviewers used
# to inherit with no way to change it (#1143). 0 disables it.
REVIEWER_IDLE_TIMEOUT_SEC = 600


class ReviewerConfig(BaseModel):
    """One reviewer runtime, independent of solver credentials and budgets.

    ``to_dict`` is for private execution payloads. Persisted/public artifacts
    must use ``to_config_artifact`` so reviewer credentials cannot leak.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    agent: str = "opencode"
    model: str | None = None
    reasoning_effort: str | None = None
    environment: str = "docker"
    timeout_sec: int = Field(default=REVIEWER_AGENT_TIMEOUT_SEC, gt=0)
    # Idle seconds before a reviewer prompt is aborted; 0 disables the
    # watchdog and leaves timeout_sec in charge.
    idle_timeout_sec: int = Field(default=REVIEWER_IDLE_TIMEOUT_SEC, ge=0)
    concurrency: int = Field(default=4, gt=0)
    image: str = REVIEWER_IMAGE
    agent_env: dict[str, str] = Field(default_factory=dict, repr=False)
    open_network: bool = False

    @field_validator("agent")
    @classmethod
    def normalize_agent(cls, value: str) -> str:
        """Resolve an agent alias to its registry name."""
        return normalize_agent_name(value)

    @field_validator("reasoning_effort")
    @classmethod
    def normalize_effort(cls, value: str | None) -> str | None:
        """Normalize the reasoning effort label."""
        return normalize_reasoning_effort(value)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """Refuse an unknown sandbox provider."""
        if not is_known_provider(value):
            raise ValueError(f"Reviewer sandbox must be one of: {providers_phrase()}")
        return value

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        """Refuse an empty image reference or one with whitespace."""
        if not value or any(char.isspace() for char in value):
            raise ValueError("Reviewer image must be a non-empty image reference")
        return value

    @classmethod
    def coerce(cls, value: object = None) -> Self:
        """A ReviewerConfig from itself, a mapping, or None (the defaults)."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls.model_validate(value)

    def to_dict(self) -> dict:
        """The config as a JSON-ready mapping (agent_env values included)."""
        return self.model_dump(mode="json")

    def to_config_artifact(self) -> dict:
        """The config as recorded in a trial's config.json: agent_env keys, and only values safe to record."""
        result = self.model_dump(mode="json", exclude={"agent_env"})
        result["agent_env"] = {
            name: value
            for name, value in self.agent_env.items()
            if _should_record_env_entry(name, value)
        }
        result["agent_env_keys"] = sorted(self.agent_env)
        return result
