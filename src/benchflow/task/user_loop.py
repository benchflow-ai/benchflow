"""Compile document-declared simulated-user metadata into rollout user loops."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from benchflow.contracts.user import BaseUser, RoundResult

if TYPE_CHECKING:
    from benchflow.task.task import Task

_STOP_RULE_MAX_ROUNDS_RE = re.compile(r"-(\d+)-rounds$", re.IGNORECASE)
_CLARIFICATION_HINTS = (
    "clarif",
    "question",
    "what do you need",
    "what should",
    "could you explain",
    "can you explain",
    "more detail",
    "more information",
    "hidden need",
    "requirement",
)


@dataclass(frozen=True)
class CompiledUserLoop:
    """Result of compiling ``task.md`` user metadata."""

    user: BaseUser
    max_user_rounds: int
    executable: bool


def parse_stop_rule_max_rounds(stop_rule: str) -> int | None:
    """Parse ``satisfied-or-5-rounds`` style stop rules into a round cap."""
    match = _STOP_RULE_MAX_ROUNDS_RE.search(stop_rule.strip())
    if match is None:
        return None
    return int(match.group(1))


def _normalize_private_facts(raw: Any) -> dict[str, str] | None:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        return None
    facts: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            return None
        facts[key] = str(value)
    return facts


def _reward_score(rewards: dict[str, Any] | None) -> float | None:
    if not rewards:
        return None
    for key in ("reward", "exact_match"):
        if key in rewards:
            try:
                return float(rewards[key])
            except (TypeError, ValueError):
                return None
    return None


def _is_satisfied(round_result: RoundResult | None) -> bool:
    if round_result is None:
        return False
    score = _reward_score(round_result.rewards)
    return score is not None and score >= 1.0


def _trajectory_text(round_result: RoundResult | None) -> str:
    if round_result is None:
        return ""
    chunks: list[str] = []
    for event in round_result.trajectory:
        if not isinstance(event, dict):
            continue
        for key in ("content", "text", "message", "prompt"):
            value = event.get(key)
            if isinstance(value, str):
                chunks.append(value)
    if round_result.verifier_output:
        chunks.append(round_result.verifier_output)
    return "\n".join(chunks).lower()


def _agent_asked_clarification(round_result: RoundResult | None) -> bool:
    text = _trajectory_text(round_result)
    if "?" not in text:
        return False
    return any(hint in text for hint in _CLARIFICATION_HINTS)


class DocumentSimulatedUser(BaseUser):
    """Rule-based simulated user compiled from ``task.md`` user metadata."""

    def __init__(
        self,
        *,
        user_persona: str | None,
        private_facts: dict[str, str],
        stop_rule: str,
        max_rounds: int,
    ) -> None:
        self.user_persona = user_persona
        self._private_facts = dict(private_facts)
        self._stop_rule = stop_rule
        self._max_rounds = max_rounds
        self._revealed_fact_keys: set[str] = set()

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        if round == 0:
            return instruction

        if _is_satisfied(round_result):
            return None

        if round >= self._max_rounds - 1:
            return None

        if (
            round_result is not None
            and self._private_facts
            and _agent_asked_clarification(round_result)
        ):
            return self._reveal_private_facts()

        return "Please continue working on the task."

    def _reveal_private_facts(self) -> str:
        pending = [
            f"{key}: {value}"
            for key, value in self._private_facts.items()
            if key not in self._revealed_fact_keys
        ]
        if not pending:
            return "Please continue working on the task."
        for key in self._private_facts:
            if key not in self._revealed_fact_keys:
                self._revealed_fact_keys.add(key)
        return "Clarification:\n" + "\n".join(pending)


def _compile_from_document(
    *,
    user_block: dict[str, Any],
    user_persona: str | None,
) -> CompiledUserLoop | None:
    if not user_block:
        return None

    stop_rule = user_block.get("stop_rule")
    if not isinstance(stop_rule, str) or not stop_rule.strip():
        return None

    max_rounds = parse_stop_rule_max_rounds(stop_rule)
    if max_rounds is None or max_rounds < 1:
        return None

    private_facts = _normalize_private_facts(user_block.get("private_facts"))
    if private_facts is None:
        return None

    user = DocumentSimulatedUser(
        user_persona=user_persona,
        private_facts=private_facts,
        stop_rule=stop_rule,
        max_rounds=max_rounds,
    )
    return CompiledUserLoop(
        user=user,
        max_user_rounds=max_rounds,
        executable=True,
    )


def compile_document_user_loop(task: Task) -> CompiledUserLoop | None:
    """Compile ``task.md`` user metadata into a concrete rollout user loop."""
    document = task.document
    if document is None:
        return None
    return _compile_from_document(
        user_block=document.user,
        user_persona=document.user_persona,
    )


def user_loop_rollout_compatible(scenes: list[Any]) -> bool:
    """Return whether compiled user loops can drive rollout execution."""
    if len(scenes) != 1:
        return False
    scene = scenes[0]
    roles = getattr(scene, "roles", None)
    return isinstance(roles, list) and len(roles) == 1


__all__ = [
    "CompiledUserLoop",
    "DocumentSimulatedUser",
    "compile_document_user_loop",
    "parse_stop_rule_max_rounds",
    "user_loop_rollout_compatible",
]
