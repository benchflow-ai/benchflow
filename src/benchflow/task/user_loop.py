"""Compile document-declared simulated-user metadata into rollout user loops."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from benchflow._types import Role, Scene, Turn
from benchflow.contracts.user import BaseUser, RoundResult
from benchflow.scenes import DEFAULT_SCENE_PROMPT

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


@dataclass(frozen=True)
class UserLoopRolloutPlan:
    """Where a compiled user loop plugs into a multi-scene rollout."""

    pre_scenes: tuple[Scene, ...]
    user_loop_scene: Scene
    user_loop_role: Role
    user_loop_prompt: str
    post_scene: Scene | None = None


def parse_stop_rule_max_rounds(stop_rule: str) -> int | None:
    """Parse ``satisfied-or-5-rounds`` style stop rules into a round cap."""
    match = _STOP_RULE_MAX_ROUNDS_RE.search(stop_rule.strip())
    if match is None:
        return None
    return int(match.group(1))


def branchable_simulated_user_nudges(nudges: Any) -> bool:
    """Return whether nudges declare branchable simulated-user rollout splitting."""
    if not isinstance(nudges, dict):
        return False
    return nudges.get("mode") == "simulated-user" and nudges.get("branchable") is True


def _nudge_budget_max_rounds(nudges: Any) -> int | None:
    if not isinstance(nudges, dict):
        return None
    budget = nudges.get("nudge_budget")
    if isinstance(budget, bool) or budget is None:
        return None
    try:
        max_rounds = int(budget)
    except (TypeError, ValueError):
        return None
    return max_rounds if max_rounds >= 1 else None


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
    nudges: Any = None,
) -> CompiledUserLoop | None:
    if not user_block:
        return None

    stop_rule = user_block.get("stop_rule")
    max_rounds: int | None = None
    if isinstance(stop_rule, str) and stop_rule.strip():
        max_rounds = parse_stop_rule_max_rounds(stop_rule)
    else:
        stop_rule = ""

    if max_rounds is None:
        max_rounds = _nudge_budget_max_rounds(nudges)
        if max_rounds is not None and not stop_rule:
            stop_rule = f"nudge-budget-{max_rounds}-rounds"

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
    benchflow = document.benchflow if isinstance(document.benchflow, dict) else {}
    return _compile_from_document(
        user_block=document.user,
        user_persona=document.user_persona,
        nudges=benchflow.get("nudges"),
    )


def infer_user_loop_scene_name(document: Any) -> str | None:
    """Return the scene that should host a document-declared user loop."""
    scene_prompts = getattr(document, "scene_prompts", None)
    if isinstance(scene_prompts, dict):
        for scene_name in scene_prompts:
            normalized = scene_name.lower().replace("_", "-")
            if "user-loop" in normalized:
                return scene_name

    scenes = getattr(document, "scenes", None)
    if isinstance(scenes, list):
        for scene in scenes:
            name = getattr(scene, "name", None)
            if isinstance(name, str) and "user-loop" in name.lower().replace("_", "-"):
                return name
    return None


def resolve_user_loop_rollout_plan(
    scenes: list[Scene],
    *,
    user_loop_scene_name: str | None = None,
    default_prompt: str | None = None,
    nudges: Any = None,
) -> UserLoopRolloutPlan | None:
    """Return how a compiled user loop should execute across document scenes."""
    if not scenes:
        return None

    fallback_prompt = default_prompt or DEFAULT_SCENE_PROMPT

    if len(scenes) == 1:
        scene = scenes[0]
        if len(scene.roles) != 1 or not scene.turns:
            return None
        role = scene.roles[0]
        if scene.turns[0].role != role.name:
            return None
        prompt = (
            scene.turns[0].prompt
            if scene.turns[0].prompt is not None
            else fallback_prompt
        )
        return UserLoopRolloutPlan(
            pre_scenes=(),
            user_loop_scene=scene,
            user_loop_role=role,
            user_loop_prompt=prompt,
        )

    if user_loop_scene_name is None:
        return None

    anchor_index = next(
        (
            index
            for index, scene in enumerate(scenes)
            if scene.name == user_loop_scene_name
        ),
        None,
    )
    if anchor_index is None:
        return None

    anchor_scene = scenes[anchor_index]
    if not anchor_scene.turns:
        return None

    role_map = {role.name: role for role in anchor_scene.roles}
    first_turn = anchor_scene.turns[0]
    anchor_role = role_map.get(first_turn.role)
    if anchor_role is None:
        return None

    user_loop_prompt = (
        first_turn.prompt if first_turn.prompt is not None else fallback_prompt
    )
    user_loop_scene = Scene(
        name=anchor_scene.name,
        roles=[anchor_role],
        turns=[Turn(role=first_turn.role, prompt=first_turn.prompt)],
        skills_dir=anchor_scene.skills_dir,
    )

    post_scene = None
    if len(anchor_scene.turns) > 1:
        if not branchable_simulated_user_nudges(nudges):
            return None
        post_turns = anchor_scene.turns[1:]
        post_role_names = list(dict.fromkeys(turn.role for turn in post_turns))
        post_roles = [role_map[name] for name in post_role_names if name in role_map]
        post_scene = Scene(
            name=anchor_scene.name,
            roles=post_roles,
            turns=post_turns,
            skills_dir=anchor_scene.skills_dir,
        )

    return UserLoopRolloutPlan(
        pre_scenes=tuple(scenes[:anchor_index]),
        user_loop_scene=user_loop_scene,
        user_loop_role=anchor_role,
        user_loop_prompt=user_loop_prompt,
        post_scene=post_scene,
    )


def user_loop_rollout_compatible(
    scenes: list[Any],
    *,
    user_loop_scene_name: str | None = None,
    default_prompt: str | None = None,
    nudges: Any = None,
) -> bool:
    """Return whether compiled user loops can drive rollout execution."""
    typed_scenes = [scene for scene in scenes if isinstance(scene, Scene)]
    if len(typed_scenes) != len(scenes):
        return False
    return (
        resolve_user_loop_rollout_plan(
            typed_scenes,
            user_loop_scene_name=user_loop_scene_name,
            default_prompt=default_prompt,
            nudges=nudges,
        )
        is not None
    )


def user_loop_rollout_compatible_for_task(task: Task) -> bool:
    """Return whether *task* scenes and nudges support a document user loop."""
    document = task.document
    if document is None:
        return False
    benchflow = document.benchflow if isinstance(document.benchflow, dict) else {}
    scene_name = infer_user_loop_scene_name(document)
    return user_loop_rollout_compatible(
        document.scenes,
        user_loop_scene_name=scene_name,
        nudges=benchflow.get("nudges"),
    )


__all__ = [
    "CompiledUserLoop",
    "DocumentSimulatedUser",
    "UserLoopRolloutPlan",
    "branchable_simulated_user_nudges",
    "compile_document_user_loop",
    "infer_user_loop_scene_name",
    "parse_stop_rule_max_rounds",
    "resolve_user_loop_rollout_plan",
    "user_loop_rollout_compatible",
    "user_loop_rollout_compatible_for_task",
]
