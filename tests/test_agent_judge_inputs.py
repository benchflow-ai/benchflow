"""Tests for empty-inputs guards on agent-judge / ors-episode verifier strategies.

Review bug #6: the validator used ``not all(isinstance(i, str) and i for i in inputs)``
but ``all([])`` is ``True``, so ``inputs: []`` slipped past the "non-empty list of
strings" check. Separately, ``Verifier._collect_agent_judge_inputs`` lacked the
emptiness backstop that ``_collect_ors_episode_inputs`` already has.
"""

from __future__ import annotations

import textwrap

import pytest

from benchflow.task import Verifier
from benchflow.task.verifier import AgentJudgeInputError
from benchflow.task.verifier_document import (
    VerifierDocument,
    VerifierDocumentParseError,
    VerifierStrategy,
)


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


def test_ors_episode_empty_inputs_rejected() -> None:
    with pytest.raises(VerifierDocumentParseError, match="non-empty"):
        VerifierDocument.from_text(_ors_episode_doc("[]"))


async def test_collect_agent_judge_inputs_empty_raises() -> None:
    """Runtime backstop mirroring _collect_ors_episode_inputs: an evidence-less
    agent-judge strategy must not silently call the judge and grade a reward."""
    strategy = VerifierStrategy(name="judge", type="agent-judge", config={"inputs": []})
    verifier = Verifier(task=None, rollout_paths=None, sandbox=None)
    with pytest.raises(AgentJudgeInputError):
        await verifier._collect_agent_judge_inputs(strategy)
