"""Typed public and storage contracts for trajectory contributions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from benchflow.publish.traj_capture import (
    ARTIFACT_NAME_PATTERN as _ARTIFACT_NAME_PATTERN,
)
from benchflow.publish.traj_capture import (
    MAX_ARTIFACTS,
    MAX_CAPTURE_BYTES,
    MAX_EMAIL_LENGTH,
    MAX_FILE_BYTES,
    MAX_GITHUB_ID_LENGTH,
    MAX_UPLOADED_BY_LENGTH,
    validate_artifact_name,
    validate_email,
    validate_github_id,
    validate_source_id,
)
from benchflow.publish.traj_capture import (
    MAX_MANIFEST_BYTES as _MAX_MANIFEST_BYTES,
)

ARTIFACT_NAME = _ARTIFACT_NAME_PATTERN
MAX_ARTIFACT_BYTES = MAX_FILE_BYTES
MAX_MANIFEST_BYTES = _MAX_MANIFEST_BYTES
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sha256: str
    bytes: int = Field(ge=1, le=MAX_ARTIFACT_BYTES)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_artifact_name(value)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("artifact sha256 must be 64 lowercase hex characters")
        return value


class ContributorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    github_id: str = Field(min_length=1, max_length=MAX_GITHUB_ID_LENGTH)
    email: str = Field(min_length=3, max_length=MAX_EMAIL_LENGTH)

    @field_validator("github_id")
    @classmethod
    def validate_github(cls, value: str) -> str:
        return validate_github_id(value)

    @field_validator("email")
    @classmethod
    def validate_contributor_email(cls, value: str) -> str:
        return validate_email(value)


class UploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.1.0", "1.2.0"]
    kind: Literal["bronze.trajectory"]
    source_id: str
    traj_digest: str
    uploaded_by: str | None = Field(default=None, max_length=MAX_UPLOADED_BY_LENGTH)
    contributor: ContributorInfo
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
    version: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$",
    )


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


class TrajectoryPreviewInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1)
    kind: str = Field(min_length=1, max_length=64)
    summary: str = Field(min_length=1, max_length=4001)


class TrajectoryReportInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_file: str
    format: Literal[
        "BenchFlow ACP",
        "Claude Code",
        "Codex",
        "LLM exchanges",
        "OpenTrace",
        "Generic JSONL",
    ]
    file_count: int = Field(ge=1, le=MAX_ARTIFACTS)
    size_bytes: int = Field(ge=1, le=MAX_CAPTURE_BYTES)
    total_steps: int = Field(ge=0)
    thinking_steps: int = Field(ge=0)
    tool_call_steps: int = Field(ge=0)
    human_steps: int = Field(ge=0)
    created_at: datetime
    created_at_source: Literal["trajectory timestamp", "file timestamp"]
    masked_values: int = Field(ge=0)
    preview: list[TrajectoryPreviewInfo] = Field(max_length=20)

    @field_validator("primary_file")
    @classmethod
    def validate_primary_file(cls, value: str) -> str:
        return validate_artifact_name(value)

    @model_validator(mode="after")
    def validate_step_partition(self) -> Self:
        categorized = self.thinking_steps + self.tool_call_steps + self.human_steps
        if self.total_steps != categorized:
            raise ValueError("trajectory report step counts must partition total steps")
        expected_numbers = list(range(1, len(self.preview) + 1))
        if [step.number for step in self.preview] != expected_numbers:
            raise ValueError("trajectory report preview must contain the first steps")
        if len(self.preview) > self.total_steps:
            raise ValueError("trajectory report preview exceeds total steps")
        return self


class ContributionManifest(UploadRequest):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0", "1.1.0", "1.2.0"]
    contributor: ContributorInfo | None = None
    created_at: datetime
    tool: ToolInfo
    run: RunInfo
    redaction: RedactionInfo
    trajectory_report: TrajectoryReportInfo | None = None

    @model_validator(mode="after")
    def validate_manifest_version(self) -> Self:
        if self.schema_version in {"1.1.0", "1.2.0"} and self.contributor is None:
            raise ValueError(
                f"schema {self.schema_version} requires contributor metadata"
            )
        if self.schema_version == "1.2.0" and self.trajectory_report is None:
            raise ValueError("schema 1.2.0 requires trajectory report metadata")
        if self.schema_version != "1.2.0" and self.trajectory_report is not None:
            raise ValueError("trajectory report metadata requires schema 1.2.0")
        if self.trajectory_report is not None:
            report = self.trajectory_report
            artifact_names = {artifact.name for artifact in self.artifacts}
            if report.primary_file not in artifact_names:
                raise ValueError("trajectory report primary file is not an artifact")
            if report.file_count != len(self.artifacts):
                raise ValueError(
                    "trajectory report file count does not match artifacts"
                )
            if report.size_bytes != sum(item.bytes for item in self.artifacts):
                raise ValueError("trajectory report size does not match artifacts")
            if report.masked_values > self.redaction.replacements:
                raise ValueError(
                    "trajectory report masked values exceed redaction count"
                )
        return self


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
