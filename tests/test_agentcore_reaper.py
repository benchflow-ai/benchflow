"""Cleanup and AWS response-shape regressions for AgentCore runtimes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from benchflow.sandbox import agentcore_provisioning as provisioning


class TestReaper:
    def _control(self, runtimes, tags):
        control = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"agentRuntimes": runtimes}]
        control.get_paginator.return_value = paginator
        control.list_tags_for_resource.side_effect = lambda resourceArn: {
            "tags": tags.get(resourceArn, {})
        }
        return control

    def test_only_benchflow_managed_runtimes_are_reaped(self):
        """Never delete something another tool created in the same account."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        old = datetime.now(UTC) - timedelta(days=3)
        runtimes = [
            {"agentRuntimeArn": "arn:mine", "agentRuntimeId": "mine", "createdAt": old},
            {
                "agentRuntimeArn": "arn:other",
                "agentRuntimeId": "other",
                "createdAt": old,
            },
        ]
        expired = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        control = self._control(
            runtimes,
            {
                "arn:mine": {
                    provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE,
                    provisioning.LEASE_TAG: expired,
                }
            },
        )

        report = reap_stale_runtimes(control, max_age_minutes=60)

        assert report.deleted == ["mine"]
        assert report.skipped_unmanaged == 1

    def test_recent_runtimes_are_kept(self):
        """A runtime from a run still in flight must survive cleanup."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        runtimes = [
            {
                "agentRuntimeArn": "arn:mine",
                "agentRuntimeId": "mine",
                "createdAt": datetime.now(UTC) - timedelta(minutes=5),
            }
        ]
        expired = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        control = self._control(
            runtimes,
            {
                "arn:mine": {
                    provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE,
                    provisioning.LEASE_TAG: expired,
                }
            },
        )

        report = reap_stale_runtimes(control, max_age_minutes=1440)

        assert report.deleted == []
        assert report.skipped_recent == 1
        control.delete_agent_runtime.assert_not_called()

    def test_dry_run_reports_without_deleting(self):
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        runtimes = [
            {
                "agentRuntimeArn": "arn:mine",
                "agentRuntimeId": "mine",
                "createdAt": datetime.now(UTC) - timedelta(days=3),
            }
        ]
        expired = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        control = self._control(
            runtimes,
            {
                "arn:mine": {
                    provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE,
                    provisioning.LEASE_TAG: expired,
                }
            },
        )

        report = reap_stale_runtimes(control, max_age_minutes=60, dry_run=True)

        assert report.deleted == ["mine"]
        control.delete_agent_runtime.assert_not_called()

    def test_unreadable_tags_fail_closed(self):
        """If tags can't be read, assume it isn't ours."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        control = self._control(
            [
                {
                    "agentRuntimeArn": "arn:x",
                    "agentRuntimeId": "x",
                    "createdAt": datetime.now(UTC) - timedelta(days=3),
                }
            ],
            {},
        )
        control.list_tags_for_resource.side_effect = RuntimeError("denied")

        report = reap_stale_runtimes(control, max_age_minutes=60)

        assert report.deleted == []
        assert report.skipped_unmanaged == 1


class TestCleanupCommandGating:
    """``bench sandbox cleanup`` must not phone AWS unless AgentCore is set up."""

    def test_cleanup_is_inert_without_agentcore_configuration(self, monkeypatch):
        """A dev machine with AWS credentials must not trigger a live call.

        ``boto3`` ships with several unrelated extras, so importability is not
        evidence that this account is being used for AgentCore runs.
        """
        from benchflow.cli.sandbox import _cleanup_agentcore_runtimes

        monkeypatch.delenv("BENCHFLOW_AGENTCORE_ROLE_ARN", raising=False)
        with patch("boto3.Session") as session:
            assert (
                _cleanup_agentcore_runtimes(dry_run=True, max_age_minutes=60) is False
            )
        session.assert_not_called()

    def test_credential_failure_degrades_instead_of_crashing(self, monkeypatch):
        """Cleanup may still have Daytona work to do; don't abort the command."""
        from benchflow.cli.sandbox import _cleanup_agentcore_runtimes

        monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/x")
        with patch("boto3.Session", side_effect=RuntimeError("no credentials")):
            assert (
                _cleanup_agentcore_runtimes(dry_run=True, max_age_minutes=60) is False
            )

    def test_daytona_failure_does_not_block_agentcore_cleanup(self, monkeypatch):
        """One backend's broken credentials must not strand the other's resources.

        The Daytona path exits when DAYTONA_API_KEY is missing; before this was
        isolated it aborted the whole command, silently leaving AgentCore
        runtimes to accumulate against a 100-per-account quota.
        """
        import typer

        from benchflow.cli import sandbox as sandbox_cli

        calls: list[str] = []
        monkeypatch.setattr(sandbox_cli, "_daytona_sdk_available", lambda: True)
        monkeypatch.setattr(
            sandbox_cli,
            "_cleanup_agentcore_runtimes",
            lambda **kw: calls.append("agentcore") or True,
        )
        fake_main = MagicMock()
        fake_main._cleanup_daytona_sandboxes.side_effect = typer.Exit(1)
        monkeypatch.setitem(__import__("sys").modules, "benchflow.cli.main", fake_main)

        sandbox_cli.sandbox_cleanup(dry_run=True, max_age_minutes=60)

        assert calls == ["agentcore"]


class TestAwsResponseShapeConformance:
    """Pin the mocked shapes to the SDK's real ones.

    Every fixture in this file invents AWS responses. When an invented shape
    drifts from the service model the tests keep passing while production
    breaks — which is exactly how the reaper came to read a ``createdAt`` that
    ``ListAgentRuntimes`` never returns, and so deleted fresh runtimes.
    """

    def _shape(self, operation, member=None):
        boto3 = pytest.importorskip("boto3")
        client = boto3.Session(region_name="us-west-2").client(
            "bedrock-agentcore-control",
            aws_access_key_id="x",
            aws_secret_access_key="y",
        )
        shape = client.meta.service_model.operation_model(operation).output_shape
        return shape.members[member].member if member else shape

    def test_list_agent_runtimes_has_no_created_at(self):
        """The field the reaper originally keyed on does not exist here."""
        members = self._shape("ListAgentRuntimes", "agentRuntimes").members

        assert "lastUpdatedAt" in members
        assert "createdAt" not in members

    def test_get_agent_runtime_carries_the_bound_image(self):
        """The adoption check reads this to refuse a mismatched runtime."""
        members = self._shape("GetAgentRuntime").members

        assert "agentRuntimeArtifact" in members
        container = members["agentRuntimeArtifact"].members["containerConfiguration"]
        assert "containerUri" in container.members


class TestReaperAgeHandling:
    def _control(self, runtimes, tags):
        control = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"agentRuntimes": runtimes}]
        control.get_paginator.return_value = paginator
        control.list_tags_for_resource.side_effect = lambda resourceArn: {
            "tags": tags.get(resourceArn, {})
        }
        return control

    def _managed(self, arn, *, leased_until=None):
        """Managed tags. Default lease is expired, i.e. reapable."""
        expiry = leased_until or (datetime.now(UTC) - timedelta(hours=1))
        return {
            arn: {
                provisioning.MANAGED_TAG: provisioning.MANAGED_VALUE,
                provisioning.LEASE_TAG: expiry.isoformat(),
            }
        }

    def test_fresh_runtime_in_the_real_list_shape_is_kept(self):
        """Reproduces the reported P1 with the shape AWS actually returns.

        ``ListAgentRuntimes`` has only ``lastUpdatedAt``; keying on
        ``createdAt`` yielded None, skipped the age comparison, and selected a
        minutes-old runtime for deletion under a one-day policy.
        """
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        runtimes = [
            {
                "agentRuntimeArn": "arn:mine",
                "agentRuntimeId": "mine",
                "agentRuntimeName": "bf_x",
                "lastUpdatedAt": datetime.now(UTC) - timedelta(minutes=3),
                "status": "READY",
            }
        ]
        control = self._control(runtimes, self._managed("arn:mine"))

        report = reap_stale_runtimes(control, max_age_minutes=1440, dry_run=True)

        assert report.deleted == []
        assert report.skipped_recent == 1

    def test_runtime_with_no_timestamp_is_kept(self):
        """Unknown age must never be read as 'old enough to delete'."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        control = self._control(
            [{"agentRuntimeArn": "arn:mine", "agentRuntimeId": "mine"}],
            self._managed("arn:mine"),
        )

        report = reap_stale_runtimes(control, max_age_minutes=1440)

        assert report.deleted == []
        assert report.skipped_recent == 1
        control.delete_agent_runtime.assert_not_called()

    def test_genuinely_old_runtime_in_the_real_shape_is_reaped(self):
        """The positive control: cleanup must still do its job."""
        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes

        control = self._control(
            [
                {
                    "agentRuntimeArn": "arn:mine",
                    "agentRuntimeId": "mine",
                    "lastUpdatedAt": datetime.now(UTC) - timedelta(days=3),
                }
            ],
            self._managed("arn:mine"),
        )

        report = reap_stale_runtimes(control, max_age_minutes=1440)

        assert report.deleted == ["mine"]
