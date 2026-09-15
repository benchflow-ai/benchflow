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
    concurrency: int = Field(default=4, gt=0)
    image: str = REVIEWER_IMAGE
    agent_env: dict[str, str] = Field(default_factory=dict, repr=False)
    open_network: bool = False

    @field_validator("agent")
    @classmethod
    def normalize_agent(cls, value: str) -> str:
        return normalize_agent_name(value)

    @field_validator("reasoning_effort")
    @classmethod
    def normalize_effort(cls, value: str | None) -> str | None:
        return normalize_reasoning_effort(value)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        if not is_known_provider(value):
            raise ValueError(f"Reviewer sandbox must be one of: {providers_phrase()}")
        return value

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        if not value or any(char.isspace() for char in value):
            raise ValueError("Reviewer image must be a non-empty image reference")
        return value

    @classmethod
    def coerce(cls, value: object = None) -> Self:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls.model_validate(value)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def to_config_artifact(self) -> dict:
        result = self.model_dump(mode="json", exclude={"agent_env"})
        result["agent_env"] = {
            name: value
            for name, value in self.agent_env.items()
            if _should_record_env_entry(name, value)
        }
        result["agent_env_keys"] = sorted(self.agent_env)
        return result
