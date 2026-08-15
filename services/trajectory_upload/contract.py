"""Typed public and storage contracts for trajectory contributions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from benchflow.publish.traj_capture import MAX_FILE_BYTES, validate_source_id

MAX_ARTIFACTS = 8
MAX_ARTIFACT_BYTES = MAX_FILE_BYTES
MAX_CAPTURE_BYTES = 2 * MAX_ARTIFACT_BYTES
MAX_MANIFEST_BYTES = 1024**2
ARTIFACT_NAME = re.compile(r"^trajectory/[A-Za-z0-9._-]{1,128}\.jsonl$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sha256: str
    bytes: int = Field(ge=1, le=MAX_ARTIFACT_BYTES)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not ARTIFACT_NAME.fullmatch(value):
            raise ValueError("artifact name is outside trajectory/*.jsonl")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("artifact sha256 must be 64 lowercase hex characters")
        return value


class UploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"]
    kind: Literal["bronze.trajectory"]
    source_id: str
    traj_digest: str
    uploaded_by: str | None = Field(default=None, max_length=256)
    artifacts: list[Artifact] = Field(min_length=1, max_length=MAX_ARTIFACTS)

    @field_validator("source_id")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return validate_source_id(value)

    @field_validator("traj_digest")
    @classmethod
    def validate_traj_digest(cls, value: str) -> str:
        prefix, separator, digest = value.partition(":")
        if prefix != "sha256" or separator != ":" or not SHA256.fullmatch(digest):
            raise ValueError("traj_digest must be sha256:<64 lowercase hex characters>")
        return value

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique")
        if sum(artifact.bytes for artifact in self.artifacts) > MAX_CAPTURE_BYTES:
            raise ValueError(f"capture exceeds {MAX_CAPTURE_BYTES} bytes")
        if self.traj_digest != f"sha256:{trajectory_digest(self.artifacts)}":
            raise ValueError("traj_digest does not match the artifact hashes")
        return self


class ToolInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["benchflow"]
    version: str = Field(min_length=1, max_length=64)


Scalar = str | int | float | bool | None


class RunInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: Scalar = None
    model: Scalar = None
    harness: Scalar = None
    skill_mode: Scalar = None
    task_id: Scalar = None
    reward: Scalar = None


class RedactionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applied: Literal[True]
    replacements: int = Field(ge=0)


class ContributionManifest(UploadRequest):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    tool: ToolInfo
    run: RunInfo
    redaction: RedactionInfo


@dataclass(frozen=True)
class UploadObject:
    name: str
    put_url: str
    headers: dict[str, str]


@dataclass(frozen=True)
class UploadGrant:
    upload_id: str
    bucket: str
    base_url: str
    prefix: str
    objects: tuple[UploadObject, ...]
    expires_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "bucket": self.bucket,
            "base_url": self.base_url,
            "prefix": self.prefix,
            "objects": [
                {
                    "name": item.name,
                    "put_url": item.put_url,
                    "headers": item.headers,
                }
                for item in self.objects
            ],
            "expires_at": self.expires_at.isoformat().replace("+00:00", "Z"),
        }


def trajectory_digest(artifacts: list[Artifact]) -> str:
    digest_input = "\n".join(
        f"{artifact.name}\t{artifact.sha256}"
        for artifact in sorted(artifacts, key=lambda item: item.name)
    )
    return hashlib.sha256(digest_input.encode()).hexdigest()
