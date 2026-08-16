"""Two-phase trajectory staging tests for interactive inspection."""

from __future__ import annotations

import json
from pathlib import Path

from benchflow.publish.redact import REDACTED
from benchflow.publish.traj_capture import (
    finalize_trajectory_capture,
    stage_trajectory_artifacts,
    stage_trajectory_capture,
)
from services.trajectory_upload.validation import validate_local_capture


def test_artifacts_are_inspectable_before_contributor_manifest_finalization(
    tmp_path: Path,
) -> None:
    """Guards the interactive trajectory-report follow-up to PR #992."""
    source = tmp_path / "capture.jsonl"
    source.write_text(
        json.dumps({"api_key": "opaque-prefixless-value", "type": "message"}) + "\n",
        encoding="utf-8",
    )

    with stage_trajectory_artifacts(source, source_id="two-phase") as artifacts:
        staging_dir = artifacts.files[0].local_path.parents[1]
        assert not (staging_dir / "manifest.json").exists()
        assert artifacts.redaction_replacements == 1
        assert artifacts.files[0].created_at is not None
        assert REDACTED in artifacts.files[0].local_path.read_text()

        staged = finalize_trajectory_capture(
            artifacts,
            github_id="benchflow-user",
            email="user@example.com",
        )

        assert staged.files[-1].relname == "manifest.json"
        assert staged.manifest["contributor"] == {
            "github_id": "benchflow-user",
            "email": "user@example.com",
        }
        assert staged.artifact_redaction_replacements == 1
        assert staged.redaction_replacements == 1


def test_staged_secret_marker_passes_the_independent_server_scan(
    tmp_path: Path,
) -> None:
    """Guards the upload-redaction follow-up to PR #992."""
    source = tmp_path / "capture.jsonl"
    source.write_text(
        json.dumps(
            {
                "api_key": "opaque-prefixless-value",
                "text": "API_KEY=another-prefixless-value",
                "command": ["tool", "--client-secret", "third-prefixless-value"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with stage_trajectory_capture(source, source_id="demo") as staged:
        payload = staged.files[0].local_path.read_text(encoding="utf-8")
        assert payload.count(REDACTED) == 3
        manifest_bytes = staged.files[-1].local_path.read_bytes()
        paths = {item.relname: item.local_path for item in staged.files[:-1]}
        validated = validate_local_capture(manifest_bytes, paths)

    assert validated.manifest.source_id == "demo"
