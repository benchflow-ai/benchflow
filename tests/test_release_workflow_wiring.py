"""Regression coverage for the unit -> live integration -> preview chain."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _workflow(name: str) -> dict:
    # BaseLoader preserves GitHub's literal ``on`` key instead of applying
    # YAML 1.1's obsolete on=true coercion.
    loaded = yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def _plan_scope_script(integration: dict) -> str:
    steps = integration["jobs"]["detect-scope"]["steps"]
    return next(
        step["run"] for step in steps if step.get("name") == "Plan scope (trusted)"
    )


def _release_gate_script(integration: dict) -> str:
    return integration["jobs"]["release-gate"]["steps"][0]["run"]


def _provenance_validation_script(preview: dict) -> str:
    steps = preview["jobs"]["publish"]["steps"]
    return next(
        step["run"]
        for step in steps
        if step.get("name") == "Validate tested release provenance"
    )


def _provenance_discovery_script(preview: dict) -> str:
    steps = preview["jobs"]["provenance"]["steps"]
    return next(
        step["run"]
        for step in steps
        if step.get("name") == "Find tested release provenance"
    )


def _run_main_gate(
    tmp_path: Path,
    integration: dict,
    *,
    conclusion: str,
    source_event: str = "push",
    source_branch: str = "main",
) -> tuple[subprocess.CompletedProcess[str], str]:
    output = tmp_path / "github-output"
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(output),
        "EVENT_NAME": "workflow_run",
        "SOURCE_CONCLUSION": conclusion,
        "SOURCE_EVENT": source_event,
        "SOURCE_BRANCH": source_branch,
        "BASE_REF": "tested-sha",
        "HEAD_SHA": "tested-sha",
    }
    result = subprocess.run(
        ["bash", "-c", _plan_scope_script(integration)],
        cwd=WORKFLOWS.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    github_output = output.read_text() if output.exists() else ""
    return result, github_output


def _run_terminal_gate(
    tmp_path: Path,
    integration: dict,
    *,
    source: str = "success",
    detect: str = "success",
    rollout: str = "success",
    fixtures: str = "success",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", _release_gate_script(integration)],
        cwd=tmp_path,
        env={
            **os.environ,
            "SOURCE_CONCLUSION": source,
            "DETECT_SCOPE_RESULT": detect,
            "ROLLOUT_SMOKE_RESULT": rollout,
            "FIXTURE_SCENARIOS_RESULT": fixtures,
            "TESTED_SHA": "b" * 40,
            "SOURCE_RUN_NUMBER": "456",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )


def _run_provenance_validation(
    tmp_path: Path,
    preview: dict,
    *,
    tested_sha: str,
    source_run_number: str,
) -> tuple[subprocess.CompletedProcess[str], str]:
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "tested-sha").write_text(tested_sha + "\n")
    (provenance / "source-run-number").write_text(source_run_number + "\n")
    output = tmp_path / "github-output"
    result = subprocess.run(
        ["bash", "-c", _provenance_validation_script(preview)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PROVENANCE_DIR": str(provenance),
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    github_output = output.read_text() if output.exists() else ""
    return result, github_output


def _run_provenance_discovery(
    tmp_path: Path,
    preview: dict,
    *,
    artifact_count: int,
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    fake_gh = bin_dir / "gh"
    fake_gh.write_text("#!/bin/sh\nprintf '%s\\n' \"$ARTIFACT_COUNT\"\n")
    fake_gh.chmod(0o755)
    output = tmp_path / "github-output"
    result = subprocess.run(
        ["bash", "-c", _provenance_discovery_script(preview)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "ARTIFACT_COUNT": str(artifact_count),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": "benchflow-ai/benchflow",
            "INTEGRATION_RUN_ID": "123",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    github_output = output.read_text() if output.exists() else ""
    return result, github_output


def test_preview_publish_is_downstream_of_tested_main_live_gate() -> None:
    """Guards the 0.6.6 fix for the workflow-chain regression from PR #802."""
    integration = _workflow("integration-light.yml")
    preview = _workflow("internal-preview-release.yml")

    assert integration["on"]["workflow_run"] == {
        "workflows": ["test"],
        "types": ["completed"],
    }
    assert preview["on"]["workflow_run"] == {
        "workflows": ["integration-light"],
        "types": ["completed"],
    }

    integration_gate = integration["jobs"]["rollout-smoke"]["if"]
    for fail_closed_condition in (
        "github.event.workflow_run.conclusion == 'success'",
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_branch == 'main'",
    ):
        assert fail_closed_condition in integration_gate

    terminal_gate = integration["jobs"]["release-gate"]
    # fixture-scenarios grades the coherent PR-head code+fixture pair that
    # rollout-smoke's src-only overlay structurally cannot; it is a gate, not
    # advisory, so the preview chain must depend on it too.
    assert terminal_gate["needs"] == [
        "detect-scope",
        "rollout-smoke",
        "fixture-scenarios",
    ]
    for terminal_condition in (
        "always()",
        "github.event_name == 'workflow_run'",
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_branch == 'main'",
    ):
        assert terminal_condition in terminal_gate["if"]

    detect_steps = integration["jobs"]["detect-scope"]["steps"]
    resolve_step = next(
        step for step in detect_steps if step.get("name") == "Resolve refs"
    )
    assert "github.event.workflow_run.head_sha" in resolve_step["env"]["BASE_REF"]
    assert "github.event.workflow_run.head_sha" in resolve_step["env"]["HEAD_SHA"]

    smoke_steps = integration["jobs"]["rollout-smoke"]["steps"]
    overlay_step = next(
        step
        for step in smoke_steps
        if step.get("name") == "Overlay PR-head src/benchflow (code-under-test)"
    )
    assert (
        overlay_step["env"]["HEAD_SHA"] == "${{ needs.detect-scope.outputs.head_sha }}"
    )

    release_steps = terminal_gate["steps"]
    enforce_step = release_steps[0]
    assert enforce_step["env"]["TESTED_SHA"] == (
        "${{ github.event.workflow_run.head_sha }}"
    )
    assert enforce_step["env"]["SOURCE_RUN_NUMBER"] == (
        "${{ github.event.workflow_run.run_number }}"
    )
    upload_step = next(
        step
        for step in release_steps
        if step.get("name") == "Upload tested release provenance"
    )
    assert "actions/upload-artifact@" in upload_step["uses"]
    assert upload_step["with"]["name"] == "internal-preview-provenance"
    assert upload_step["with"]["path"] == "release-provenance"

    preview_steps = preview["jobs"]["publish"]["steps"]
    preview_gate = preview["jobs"]["provenance"]["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in preview_gate
    assert "github.event.workflow_run.event == 'workflow_run'" in preview_gate
    assert "github.event.workflow_run.head_branch == 'main'" in preview_gate
    assert preview["jobs"]["provenance"]["outputs"]["available"] == (
        "${{ steps.artifact.outputs.available }}"
    )
    assert preview["jobs"]["publish"]["needs"] == "provenance"
    assert preview["jobs"]["publish"]["if"] == (
        "needs.provenance.outputs.available == 'true'"
    )
    download_step = next(
        step
        for step in preview_steps
        if step.get("name") == "Download tested release provenance"
    )
    assert "actions/download-artifact@" in download_step["uses"]
    assert download_step["with"]["run-id"] == "${{ github.event.workflow_run.id }}"
    checkout_step = next(
        step for step in preview_steps if "actions/checkout@" in step.get("uses", "")
    )
    assert checkout_step["with"]["ref"] == (
        "${{ steps.provenance.outputs.tested_sha }}"
    )
    version_step = next(
        step
        for step in preview_steps
        if step.get("name") == "Compute internal preview version"
    )
    assert "${{ steps.provenance.outputs.source_run_number }}" in version_step["run"]


def test_preview_requires_exactly_one_release_provenance_artifact(
    tmp_path: Path,
) -> None:
    """Guards the fix from PR #997 against PR-only preview publication failures."""
    preview = _workflow("internal-preview-release.yml")

    missing, missing_output = _run_provenance_discovery(
        tmp_path / "missing", preview, artifact_count=0
    )
    present, present_output = _run_provenance_discovery(
        tmp_path / "present", preview, artifact_count=1
    )
    duplicate, duplicate_output = _run_provenance_discovery(
        tmp_path / "duplicate", preview, artifact_count=2
    )

    assert missing.returncode == 0, missing.stderr
    assert missing_output == "available=false\n"
    assert present.returncode == 0, present.stderr
    assert present_output == "available=true\n"
    assert duplicate.returncode != 0
    assert duplicate_output == ""


def test_pr_integration_is_not_cancelled_by_test_workflow_completion() -> None:
    """Guards PR #944 against cross-event integration concurrency cancellation."""
    integration = _workflow("integration-light.yml")

    assert integration["concurrency"]["group"] == (
        "integration-light-${{ github.event_name }}-"
        "${{ github.event.pull_request.head.sha || "
        "github.event.workflow_run.head_sha || github.ref }}"
    )


def test_failed_main_unit_run_makes_integration_gate_red(tmp_path: Path) -> None:
    """Guards the 0.6.6 fix against publishing after a failed main unit run."""
    integration = _workflow("integration-light.yml")

    result, github_output = _run_main_gate(
        tmp_path,
        integration,
        conclusion="failure",
    )

    assert result.returncode != 0
    assert "live release gate is red" in result.stdout
    assert github_output == ""


def test_successful_main_unit_run_requires_live_integration(tmp_path: Path) -> None:
    """Guards the 0.6.6 fix against skipping live integration on tested main."""
    integration = _workflow("integration-light.yml")

    result, github_output = _run_main_gate(
        tmp_path,
        integration,
        conclusion="success",
    )

    assert result.returncode == 0, result.stderr
    assert github_output == "should_run=true\n"


def test_non_main_test_run_stays_credential_free_noop(tmp_path: Path) -> None:
    """Guards the 0.6.6 fix against exposing release secrets to PR test runs."""
    integration = _workflow("integration-light.yml")

    result, github_output = _run_main_gate(
        tmp_path,
        integration,
        conclusion="success",
        source_event="pull_request",
        source_branch="contributor-branch",
    )

    assert result.returncode == 0, result.stderr
    assert github_output == "should_run=false\n"


def test_terminal_gate_rejects_skipped_live_integration(tmp_path: Path) -> None:
    """Guards the 0.6.6 fix against a skipped live job publishing a preview."""
    integration = _workflow("integration-light.yml")

    result = _run_terminal_gate(tmp_path, integration, rollout="skipped")

    assert result.returncode != 0
    assert "live integration concluded skipped" in result.stdout


def test_terminal_gate_rejects_failed_fixture_scenarios(tmp_path: Path) -> None:
    """The coherent-tree job must gate the preview, not merely be waited on.

    Listing a job in ``needs`` only makes the gate WAIT for it; without an
    explicit conclusion check a red fixture-scenarios would still publish.
    """
    integration = _workflow("integration-light.yml")

    result = _run_terminal_gate(tmp_path, integration, fixtures="failure")

    assert result.returncode != 0
    assert "fixture scenarios concluded failure" in result.stdout


def test_terminal_gate_accepts_only_fully_green_chain(tmp_path: Path) -> None:
    """Guards the 0.6.6 fix for the fully green preview-publication chain."""
    integration = _workflow("integration-light.yml")

    result = _run_terminal_gate(tmp_path, integration)

    assert result.returncode == 0, result.stderr
    assert "both succeeded" in result.stdout
    assert (tmp_path / "release-provenance" / "tested-sha").read_text() == (
        "b" * 40 + "\n"
    )
    assert (tmp_path / "release-provenance" / "source-run-number").read_text() == (
        "456\n"
    )


def test_preview_accepts_valid_explicit_provenance(tmp_path: Path) -> None:
    """Guards the 0.6.6 fix for exact tested-SHA transport between workflows."""
    preview = _workflow("internal-preview-release.yml")
    tested_sha = "a" * 40

    result, github_output = _run_provenance_validation(
        tmp_path,
        preview,
        tested_sha=tested_sha,
        source_run_number="123",
    )

    assert result.returncode == 0, result.stderr
    assert github_output == f"tested_sha={tested_sha}\nsource_run_number=123\n"


def test_preview_rejects_invalid_explicit_provenance(tmp_path: Path) -> None:
    """Guards the 0.6.6 fix against drift or injection in release provenance."""
    preview = _workflow("internal-preview-release.yml")

    result, github_output = _run_provenance_validation(
        tmp_path,
        preview,
        tested_sha="refs/heads/main",
        source_run_number="123",
    )

    assert result.returncode != 0
    assert "invalid tested SHA" in result.stdout
    assert github_output == ""
