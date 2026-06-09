"""Tests for SandboxConfig network-policy reconciliation and TaskConfig schema_version.

Covers review bug #4: an explicit network_mode that contradicts the deprecated
allow_internet flag must not be silently overridden into a self-contradictory
state, plus schema_version validation.
"""

import pytest

from benchflow.task.config import (
    NetworkMode,
    SandboxConfig,
    TaskConfig,
    _validate_network_policy_fields,
)


class TestExplicitNetworkModeContradiction:
    def test_allowlist_with_allow_internet_false_raises(self):
        with pytest.raises(ValueError):
            SandboxConfig(
                network_mode="allowlist",
                allowed_hosts=["x.com"],
                allow_internet=False,
            )

    def test_public_with_allow_internet_false_raises(self):
        with pytest.raises(ValueError):
            SandboxConfig(network_mode="public", allow_internet=False)


class TestLegacyAllowInternetBackCompat:
    def test_allow_internet_false_without_explicit_mode_resolves_no_network(self):
        cfg = SandboxConfig(allow_internet=False)
        assert cfg.network_mode == NetworkMode.NO_NETWORK
        assert cfg.allowed_hosts is None
        assert cfg.allow_internet is False


class TestPolicyInvariantAfterReconcile:
    def test_no_network_with_stale_allowed_hosts_is_never_produced(self):
        """The legacy override path must not yield no-network + allowed_hosts.

        Reconciliation re-runs _validate_network_policy_fields at the end, so a
        contradictory combination is rejected rather than silently produced.
        """
        with pytest.raises(ValueError):
            SandboxConfig(
                network_mode="allowlist",
                allowed_hosts=["x.com"],
                allow_internet=False,
            )

    def test_reconciled_object_satisfies_policy_validator(self):
        cfg = SandboxConfig(allow_internet=False)
        # Must not raise: the post-reconcile state is internally consistent.
        _validate_network_policy_fields(cfg.network_mode, cfg.allowed_hosts)


class TestSchemaVersionValidation:
    def test_unknown_major_version_raises(self):
        with pytest.raises(ValueError):
            TaskConfig(schema_version="99.0")

    def test_non_parseable_version_raises(self):
        with pytest.raises(ValueError):
            TaskConfig(schema_version="banana")

    def test_explicit_supported_version_accepted(self):
        cfg = TaskConfig(schema_version="1.0")
        assert cfg.schema_version == "1.0"

    def test_default_version_accepted(self):
        cfg = TaskConfig()
        assert cfg.schema_version == "1.3"
