"""Guards #396: dense LLM-judge reward events declare correct space/granularity.

Per-criterion ``RewardEvent`` instances emitted by ``LLMJudgeRewardFunc`` carry
``step=idx`` and are *not* the terminal outcome of a trial. Before #396 they
serialized as ``space="output"`` / ``granularity="terminal"`` because the
``RewardEvent`` defaults were left untouched — masquerading as the final
verifier signal and corrupting credit assignment for trainers and ORS
exporters. This module pins the contract: dense events are
``granularity="step"``, and the space follows whatever the rubric / inline
criterion declared.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.rewards.builtins import LLMJudgeRewardFunc
from benchflow.rewards.rubric_config import (
    Criterion,
    load_rubric_json,
    load_rubric_toml,
)

_MOCK_PASS = '```json\n{"verdict": "pass", "reasoning": "good"}\n```'


# ---------------------------------------------------------------------------
# Dense event tags
# ---------------------------------------------------------------------------


class TestDenseEventTags:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_dense_event_is_step_granularity(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """A per-criterion event with ``step`` set must be tagged
        ``granularity="step"``. The pre-#396 default of ``"terminal"`` made a
        step-indexed event indistinguishable from the final outcome."""
        mock_judge.return_value = _MOCK_PASS
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(criteria=[{"description": "A", "id": "a"}])
        await func.score(tmp_path)

        ev = func.events[0]
        assert ev.step == 0
        assert ev.granularity == "step"

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_dense_event_defaults_to_output_space(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """Back-compat: an untagged criterion still emits ``space="output"`` —
        but with the correct step granularity."""
        mock_judge.return_value = _MOCK_PASS
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(criteria=[{"description": "A", "id": "a"}])
        await func.score(tmp_path)

        ev = func.events[0]
        assert ev.space == "output"
        assert ev.granularity == "step"

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_inline_criterion_declares_action_space(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """A process-like criterion can opt into a non-output space — the dense
        event carries the declared space."""
        mock_judge.return_value = _MOCK_PASS
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(
            criteria=[
                {"description": "Tool choice quality", "id": "tools", "space": "action"},
            ],
        )
        await func.score(tmp_path)

        ev = func.events[0]
        assert ev.space == "action"
        assert ev.granularity == "step"

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_multiple_criteria_carry_distinct_spaces(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """Per-criterion spaces propagate independently to each dense event."""
        mock_judge.return_value = _MOCK_PASS
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(
            criteria=[
                {"description": "Reasoning quality", "id": "r", "space": "reasoning"},
                {"description": "Memory update", "id": "m", "space": "memory"},
                {"description": "Final answer", "id": "o"},  # defaults to output
            ],
        )
        await func.score(tmp_path)

        assert [e.space for e in func.events] == ["reasoning", "memory", "output"]
        assert all(e.granularity == "step" for e in func.events)
        assert [e.step for e in func.events] == [0, 1, 2]

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_invalid_inline_space_rejected(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """A bogus space value is rejected loudly, not silently downgraded —
        otherwise a misconfigured rubric would re-introduce the original
        mistag bug under a different name. Inline criteria are parsed
        lazily in ``_load_rubric``; the error must surface from ``score()``."""
        mock_judge.return_value = _MOCK_PASS
        func = LLMJudgeRewardFunc(
            criteria=[{"description": "x", "space": "bogus"}],
        )
        with pytest.raises(ValueError, match="space"):
            await func.score(tmp_path)


# ---------------------------------------------------------------------------
# Rubric-file parsing
# ---------------------------------------------------------------------------


class TestRubricSpaceParsing:
    def test_toml_criterion_space(self, tmp_path: Path) -> None:
        path = tmp_path / "rubric.toml"
        path.write_text(
            """\
[[criterion]]
description = "Tool choice quality"
space = "action"

[[criterion]]
description = "Final answer"
"""
        )

        rubric = load_rubric_toml(path)
        assert rubric.criteria[0].space == "action"
        # Unset -> default output for back-compat.
        assert rubric.criteria[1].space == "output"

    def test_json_criterion_space(self, tmp_path: Path) -> None:
        path = tmp_path / "rubric.json"
        path.write_text(
            '{"criteria": ['
            '{"id": "c1", "match_criteria": "tool quality", "space": "action"},'
            '{"id": "c2", "match_criteria": "final answer"}'
            "]}"
        )

        rubric = load_rubric_json(path)
        assert rubric.criteria[0].space == "action"
        assert rubric.criteria[1].space == "output"

    def test_toml_invalid_space_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "rubric.toml"
        path.write_text(
            """\
[[criterion]]
description = "anything"
space = "not-a-real-space"
"""
        )

        with pytest.raises(ValueError, match="space"):
            load_rubric_toml(path)

    def test_criterion_dataclass_default_space(self) -> None:
        """Constructing a ``Criterion`` directly defaults to ``"output"`` so
        existing programmatic uses keep working."""
        c = Criterion(description="anything")
        assert c.space == "output"


# ---------------------------------------------------------------------------
# Regression: the exact symptom from #396
# ---------------------------------------------------------------------------


class TestIssue396Repro:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_issue_repro_no_longer_emits_terminal_output_tag(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """The exact repro from #396: a dense criterion event must no longer
        serialize as ``space="output"`` / ``granularity="terminal"``."""
        mock_judge.return_value = (
            '```json\n{"verdict": "pass", "score": 1.0, "reasoning": "ok"}\n```'
        )
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(
            criteria=[
                {"description": "Check action quality", "id": "action_quality"},
            ],
        )
        await func.score(tmp_path)

        ev = func.events[0]
        # The bug: ``granularity="terminal"`` paired with a non-None ``step``.
        # Post-fix: ``step`` -> ``granularity="step"``, never both.
        assert (ev.step, ev.granularity) == (0, "step")
        # ``space`` may still default to ``"output"`` (back-compat) — what
        # mattered was that the *granularity* mistag is gone.
        assert ev.source == "criterion:action_quality"
