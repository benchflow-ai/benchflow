"""Guard ports from PR #651 (main): schema_version major gate, agent-judge
empty-inputs backstop, verifier-document empty-list fix, and the explicit
network-mode/allow_internet contradiction hard-error.

Each test mirrors the source-PR test of the same guard, adapted to the v0.6
module layout (``verifier_core`` behind the ``benchflow.task.verifier``
façade).
"""

from __future__ import annotations

import textwrap

import pytest

from benchflow.task import Verifier
from benchflow.task.config import (
    NetworkMode,
    SandboxConfig,
    TaskConfig,
    _validate_network_policy_fields,
)
from benchflow.task.verifier import AgentJudgeInputError
from benchflow.task.verifier_document import (
    VerifierDocument,
    VerifierDocumentParseError,
    VerifierStrategy,
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

    def test_explicit_no_network_with_allow_internet_false_is_consistent(self):
        cfg = SandboxConfig(network_mode="no-network", allow_internet=False)
        assert cfg.network_mode == NetworkMode.NO_NETWORK
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


def _agent_judge_doc(inputs_literal: str) -> str:
    return textwrap.dedent(
        f"""\
        ---
        verifier:
          name: demo
          default_strategy: judge
          strategies:
            judge:
              type: agent-judge
              role: grader
              isolation: verifier-only
              inputs: {inputs_literal}
        ---

        ## role:grader

        Grade the work.
        """
    )


def _ors_episode_doc(inputs_literal: str) -> str:
    return textwrap.dedent(
        f"""\
        ---
        verifier:
          name: demo
          default_strategy: episode
          strategies:
            episode:
              type: ors-episode
              inputs: {inputs_literal}
        ---
        """
    )


def test_agent_judge_empty_inputs_rejected() -> None:
    with pytest.raises(VerifierDocumentParseError, match="non-empty"):
        VerifierDocument.from_text(_agent_judge_doc("[]"))


def test_agent_judge_populated_inputs_accepted() -> None:
    doc = VerifierDocument.from_text(_agent_judge_doc('["report.md"]'))
    assert doc.strategies["judge"].inputs == ("report.md",)


def test_ors_episode_empty_inputs_rejected() -> None:
    with pytest.raises(VerifierDocumentParseError, match="non-empty"):
        VerifierDocument.from_text(_ors_episode_doc("[]"))


def test_ors_episode_populated_inputs_accepted() -> None:
    doc = VerifierDocument.from_text(_ors_episode_doc('["episode.jsonl"]'))
    assert doc.strategies["episode"].inputs == ("episode.jsonl",)


async def test_collect_agent_judge_inputs_empty_raises() -> None:
    """Runtime backstop mirroring _collect_ors_episode_inputs: an evidence-less
    agent-judge strategy must not silently call the judge and grade a reward."""
    strategy = VerifierStrategy(name="judge", type="agent-judge", config={"inputs": []})
    verifier = Verifier(task=None, rollout_paths=None, sandbox=None)
    with pytest.raises(AgentJudgeInputError):
        await verifier._collect_agent_judge_inputs(strategy)
