"""Tests for document-declared user loop compilation."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.rollout import RolloutConfig
from benchflow.sandbox.user import RoundResult
from benchflow.task import Task, compile_document_user_loop, parse_stop_rule_max_rounds
from benchflow.task.user_loop import DocumentSimulatedUser

PROMPT_USER_SEMANTICS_TASK = Path(
    "docs/examples/task-standard/benchflow-wanted-features/prompt-user-semantics"
)


class TestParseStopRule:
    @pytest.mark.parametrize(
        ("stop_rule", "expected"),
        [
            ("satisfied-or-5-rounds", 5),
            ("done-or-3-rounds", 3),
            ("SATISFIED-OR-2-ROUNDS", 2),
        ],
    )
    def test_parses_round_cap(self, stop_rule: str, expected: int) -> None:
        assert parse_stop_rule_max_rounds(stop_rule) == expected

    @pytest.mark.parametrize(
        "stop_rule",
        ["satisfied", "until-done", "5-rounds", ""],
    )
    def test_rejects_unparseable_rules(self, stop_rule: str) -> None:
        assert parse_stop_rule_max_rounds(stop_rule) is None


class TestCompileDocumentUserLoop:
    def test_prompt_user_semantics_dogfood_compiles(self) -> None:
        """Guards prompt-user-semantics dogfood user loop compilation."""
        task = Task(PROMPT_USER_SEMANTICS_TASK)
        compiled = compile_document_user_loop(task)

        assert compiled is not None
        assert compiled.executable is True
        assert compiled.max_user_rounds == 5
        assert isinstance(compiled.user, DocumentSimulatedUser)
        assert compiled.user.user_persona is not None
        assert "targeted clarification" in compiled.user.user_persona

    def test_nudge_budget_sets_max_rounds_when_stop_rule_absent(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "task.md").write_text(
            """---
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
user:
  private_facts:
    hidden_need: "Reveal only when asked."
benchflow:
  nudges:
    mode: simulated-user
    nudge_budget: 6
---
## prompt

Solve it.
"""
        )
        (tmp_path / "task.toml").write_text(
            'version = "1.0"\n[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n'
        )
        (tmp_path / "instruction.md").write_text("Solve it.\n")
        env = tmp_path / "environment"
        env.mkdir()
        (env / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test.sh").write_text("#!/bin/bash\nexit 0\n")

        compiled = compile_document_user_loop(Task(tmp_path))

        assert compiled is not None
        assert compiled.max_user_rounds == 6
        assert isinstance(compiled.user, DocumentSimulatedUser)

    def test_missing_user_block_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "task.md").write_text(
            """---
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
---
## prompt

Solve it.
"""
        )
        (tmp_path / "task.toml").write_text(
            'version = "1.0"\n[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 120\n'
        )
        (tmp_path / "instruction.md").write_text("Solve it.\n")
        env = tmp_path / "environment"
        env.mkdir()
        (env / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test.sh").write_text("#!/bin/bash\nexit 0\n")

        task = Task(tmp_path)
        assert compile_document_user_loop(task) is None


class TestDocumentSimulatedUser:
    @pytest.mark.asyncio
    async def test_round_zero_returns_instruction_without_private_facts(self) -> None:
        user = DocumentSimulatedUser(
            user_persona="Verifier-only persona text.",
            private_facts={"hidden_need": "secret requirement"},
            stop_rule="satisfied-or-5-rounds",
            max_rounds=5,
        )
        prompt = await user.run(0, "Solve the task.")
        assert prompt == "Solve the task."
        assert "secret requirement" not in prompt
        assert "Verifier-only persona" not in prompt

    @pytest.mark.asyncio
    async def test_stops_when_verifier_is_satisfied(self) -> None:
        user = DocumentSimulatedUser(
            user_persona=None,
            private_facts={},
            stop_rule="satisfied-or-5-rounds",
            max_rounds=5,
        )
        passing = RoundResult(round=0, rewards={"reward": 1.0})
        assert await user.run(1, "Solve the task.", passing) is None

    @pytest.mark.asyncio
    async def test_reveals_private_facts_on_clarification_question(self) -> None:
        user = DocumentSimulatedUser(
            user_persona="Persona stays verifier-scoped.",
            private_facts={"hidden_need": "Document user loops must be executable."},
            stop_rule="satisfied-or-5-rounds",
            max_rounds=5,
        )
        clarifying = RoundResult(
            round=0,
            trajectory=[{"content": "Could you clarify the hidden need requirement?"}],
        )
        prompt = await user.run(1, "Solve the task.", clarifying)
        assert prompt is not None
        assert "hidden_need" in prompt
        assert "Document user loops must be executable." in prompt
        assert "Persona stays verifier-scoped." not in prompt

    @pytest.mark.asyncio
    async def test_stops_after_nudge_budget_round_cap(self) -> None:
        user = DocumentSimulatedUser(
            user_persona=None,
            private_facts={},
            stop_rule="nudge-budget-3-rounds",
            max_rounds=3,
        )
        working = RoundResult(round=0, trajectory=[{"content": "Still working."}])
        assert await user.run(1, "Solve the task.", working) == (
            "Please continue working on the task."
        )
        assert await user.run(2, "Solve the task.", working) is None

    @pytest.mark.asyncio
    async def test_private_facts_stay_hidden_without_clarification(self) -> None:
        user = DocumentSimulatedUser(
            user_persona=None,
            private_facts={"hidden_need": "secret requirement"},
            stop_rule="satisfied-or-5-rounds",
            max_rounds=5,
        )
        working = RoundResult(round=0, trajectory=[{"content": "Running tests now."}])
        prompt = await user.run(1, "Solve the task.", working)
        assert prompt == "Please continue working on the task."
        assert "secret requirement" not in prompt


class TestRolloutConfigUserLoopWiring:
    def test_auto_wires_compatible_single_scene_task(self, tmp_path: Path) -> None:
        (tmp_path / "task.md").write_text(
            """---
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    roles: [solver]
user:
  stop_rule: satisfied-or-4-rounds
  private_facts:
    hidden_need: "Keep this private until asked."
---
## prompt

Solve it.
"""
        )

        config = RolloutConfig(task_path=tmp_path)

        assert config.user is not None
        assert isinstance(config.user, DocumentSimulatedUser)
        assert config.max_user_rounds == 4

    def test_skips_auto_wire_for_multi_scene_dogfood(self) -> None:
        """Multi-scene prompt-user-semantics compiles but keeps scene rollout."""
        config = RolloutConfig(task_path=PROMPT_USER_SEMANTICS_TASK)

        assert config.user is None
        assert len(config.effective_scenes) > 1
        compiled = compile_document_user_loop(Task(PROMPT_USER_SEMANTICS_TASK))
        assert compiled is not None
        assert compiled.executable is True

    def test_explicit_user_is_not_overridden(self, tmp_path: Path) -> None:
        from benchflow.sandbox.user import PassthroughUser

        (tmp_path / "task.md").write_text(
            """---
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
agents:
  roles:
    solver:
      agent: codex
scenes:
  - name: solve
    roles: [solver]
user:
  stop_rule: satisfied-or-4-rounds
---
## prompt

Solve it.
"""
        )

        explicit = PassthroughUser()
        config = RolloutConfig(task_path=tmp_path, user=explicit)

        assert config.user is explicit
