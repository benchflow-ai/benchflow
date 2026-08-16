"""Ownership and deployment-layout guards for the trajectory-upload service."""

from __future__ import annotations

import os
from pathlib import Path

_SERVICE_ROOT = Path("services/trajectory_upload")


def test_trajectory_upload_has_one_service_ownership_boundary() -> None:
    """Guards the service consolidation completed in PR #1008."""
    assert not Path("infra/trajectory-upload").exists()
    required = {
        "Dockerfile",
        "README.md",
        "azure_backend.py",
        "broker_app.py",
        "contract.py",
        "infra/lifecycle.json",
        "infra/main.bicep",
        "infra/production.bicepparam",
        "scripts/deploy.sh",
        "scripts/rollback.sh",
        "scripts/smoke-test.sh",
        "validation.py",
        "validator-entrypoint",
        "validator.py",
    }
    actual = {
        str(path.relative_to(_SERVICE_ROOT))
        for path in _SERVICE_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert required <= actual


def test_trajectory_upload_operational_scripts_are_executable() -> None:
    """Guards the executable deployment entrypoints added in PR #1008."""
    for name in ("deploy.sh", "rollback.sh", "smoke-test.sh"):
        path = _SERVICE_ROOT / "scripts" / name
        assert path.stat().st_mode & os.X_OK


def test_trajectory_upload_workflow_delegates_to_service_scripts() -> None:
    """Guards PR #1008 against moving Azure logic back into GitHub Actions."""
    workflow = Path(".github/workflows/deploy-trajectory-upload.yml").read_text()

    assert "./services/trajectory_upload/scripts/deploy.sh" in workflow
    assert "./services/trajectory_upload/scripts/smoke-test.sh" in workflow
    assert "az " not in workflow


def test_trajectory_upload_deploy_uses_colocated_bicep() -> None:
    """Guards the Bicep deployment boundary established in PR #1008."""
    deploy = (_SERVICE_ROOT / "scripts" / "deploy.sh").read_text()

    assert '"${task_service_root}/infra/main.bicep"' in deploy
    assert '"${task_service_root}/infra/production.bicepparam"' in deploy
    assert "infra/trajectory-upload" not in deploy


def test_trajectory_upload_bicep_does_not_reconfigure_data_services() -> None:
    """Guards the Azure queue-service deployment fix from PR #1008."""
    bicep = (_SERVICE_ROOT / "infra" / "main.bicep").read_text()

    assert "resource queueService" in bicep
    assert "queueServices@2023-05-01' existing" in bicep
    assert "tableServices@2023-05-01' existing" in bicep
