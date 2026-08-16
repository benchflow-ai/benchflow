"""Two-phase trajectory staging tests for interactive inspection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.publish.redact import REDACTED
from benchflow.publish.traj_capture import (
    finalize_trajectory_capture,
    stage_trajectory_artifacts,
    stage_trajectory_capture,
)
from benchflow.publish.traj_report import build_trajectory_report
from services.trajectory_upload.validation import (
    CaptureRejected,
    validate_local_capture,
)


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


def test_final_manifest_persists_the_complete_trajectory_report(
    tmp_path: Path,
) -> None:
    """Guards PR #1008: uploaded manifests retain every displayed report field."""
    source = tmp_path / "capture.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"type": "user_message", "text": "Please inspect"}),
                json.dumps({"type": "agent_thought", "text": "I should inspect"}),
                json.dumps({"type": "tool_call", "title": "Read", "content": "README"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with stage_trajectory_artifacts(source, source_id="report-manifest") as artifacts:
        report = build_trajectory_report(
            artifacts.files,
            masked_values=artifacts.redaction_replacements,
        )
        report_metadata = report.as_manifest_metadata()
        staged = finalize_trajectory_capture(
            artifacts,
            github_id="benchflow-user",
            email="user@example.com",
            trajectory_report=report_metadata,
        )
        manifest_bytes = staged.files[-1].local_path.read_bytes()
        validated = validate_local_capture(
            manifest_bytes,
            {item.relname: item.local_path for item in staged.files[:-1]},
        )
        semantically_false = json.loads(manifest_bytes)
        semantically_false["trajectory_report"]["thinking_steps"] -= 1
        semantically_false["trajectory_report"]["human_steps"] += 1
        with pytest.raises(CaptureRejected, match="does not match uploaded artifacts"):
            validate_local_capture(
                json.dumps(semantically_false).encode(),
                {item.relname: item.local_path for item in staged.files[:-1]},
            )

    assert staged.manifest["schema_version"] == "1.2.0"
    assert staged.manifest["trajectory_report"] == report_metadata
    persisted = validated.manifest.trajectory_report
    assert persisted is not None
    assert persisted.total_steps == 3
    assert persisted.thinking_steps == 1
    assert persisted.tool_call_steps == 1
    assert persisted.human_steps == 1
    assert [step.summary for step in persisted.preview] == [
        "Please inspect",
        "I should inspect",
        "Read: README",
    ]

    tampered = json.loads(manifest_bytes)
    tampered["trajectory_report"]["thinking_steps"] += 1
    with pytest.raises(CaptureRejected, match="step counts must partition"):
        validate_local_capture(
            json.dumps(tampered).encode(),
            {item.relname: item.local_path for item in staged.files[:-1]},
        )


def test_legacy_safe_marker_keeps_artifact_and_report_in_sync(tmp_path: Path) -> None:
    """Guards PR #1008 against the live report-mismatch rejection.

    A legacy ``***REDACTED***`` marker was normalized only in the manifest
    preview because normalization did not increment the secret count, leaving
    the staged artifact and independently recomputed report inconsistent.
    """
    source = tmp_path / "capture.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": 'Use API_KEY="***REDACTED***" for the request',
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with stage_trajectory_artifacts(source, source_id="legacy-marker") as artifacts:
        report = build_trajectory_report(
            artifacts.files,
            masked_values=artifacts.redaction_replacements,
        )
        staged = finalize_trajectory_capture(
            artifacts,
            github_id="benchflow-user",
            email="user@example.com",
            trajectory_report=report.as_manifest_metadata(),
        )
        artifact_text = staged.files[0].local_path.read_text(encoding="utf-8")
        validated = validate_local_capture(
            staged.files[-1].local_path.read_bytes(),
            {item.relname: item.local_path for item in staged.files[:-1]},
        )

    assert "***REDACTED***" not in artifact_text
    assert REDACTED in artifact_text
    assert staged.artifact_redaction_replacements == 0
    assert validated.manifest.trajectory_report is not None
    assert REDACTED in validated.manifest.trajectory_report.preview[0].summary


def test_report_manifest_requires_contributor_metadata(tmp_path: Path) -> None:
    """Guards PR #1008 against creating an invalid anonymous schema 1.2 manifest."""
    source = tmp_path / "capture.jsonl"
    source.write_text('{"type":"user_message","text":"hello"}\n')

    with stage_trajectory_artifacts(source, source_id="report-manifest") as artifacts:
        report = build_trajectory_report(
            artifacts.files,
            masked_values=artifacts.redaction_replacements,
        )
        with pytest.raises(
            ValueError,
            match="trajectory report metadata requires contributor metadata",
        ):
            finalize_trajectory_capture(
                artifacts,
                trajectory_report=report.as_manifest_metadata(),
            )


def test_query_param_secrets_stage_stably_and_pass_the_server_rescan(
    tmp_path: Path,
) -> None:
    """Guards PR #1008: query-param secret carriers made client redaction
    diverge ("did not converge") and the independent server rescan reject the
    staged marker as an unredacted secret."""
    sas_signature = "A1" * 20
    source = tmp_path / "capture.jsonl"
    source.write_text(
        json.dumps(
            {
                "type": "user_message",
                "text": "GET https://api/v1?access_token=opaque-prefixless&page=2",
                "url": f"https://acct.blob/c/f?sv=2023-01-01&sig={sas_signature}&se=x",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with stage_trajectory_capture(source, source_id="query-demo") as staged:
        payload = staged.files[0].local_path.read_text(encoding="utf-8")
        assert payload.count(REDACTED) == 2
        assert "opaque-prefixless" not in payload
        assert sas_signature not in payload
        assert "&page=2" in payload
        manifest_bytes = staged.files[-1].local_path.read_bytes()
        paths = {item.relname: item.local_path for item in staged.files[:-1]}
        validate_local_capture(manifest_bytes, paths)
