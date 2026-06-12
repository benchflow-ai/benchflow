"""Prompt and user-runtime compilation for native task documents."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from benchflow._types import Scene, Turn
from benchflow.contracts.user import BaseUser, DocumentNudgeUser, ModelDocumentNudgeUser
from benchflow.task.document import TaskDocument

PromptComposition = Literal["append", "replace"]
PromptPartKind = Literal["base", "role", "scene", "turn"]
TeamHandoffKind = Literal["none", "sequential-shared"]
BranchExecutionKind = Literal["none", "option-kinds-preserved", "unsupported"]

_APPEND_DEFAULT_ORDER: tuple[PromptPartKind, ...] = ("base", "role", "scene", "turn")
_REPLACE_DEFAULT_ORDER: tuple[PromptPartKind, ...] = ("turn", "scene", "role", "base")
_ALLOWED_PARTS = frozenset({"base", "role", "scene", "turn"})


@dataclass(frozen=True)
class PromptPart:
    """One contributing prompt fragment in a compiled turn prompt."""

    kind: PromptPartKind
    text: str


@dataclass(frozen=True)
class CompiledPromptTurn:
    """The executable prompt for one scene turn."""

    scene: str | None
    role: str | None
    prompt: str
    parts: tuple[PromptPart, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UserRuntimeContract:
    """Redacted summary of document-declared simulated-user semantics."""

    status: Literal["none", "supported", "unsupported"]
    reason: str | None
    model: str | None
    stop_rule: str | None
    max_rounds: int | None
    persona_present: bool
    private_fact_keys: tuple[str, ...]
    nudge_mode: str | None
    nudge_budget: int | None
    branchable: bool
    branch_execution: BranchExecutionKind
    confirmation_policy: str | None
    runtime_kind: Literal["none", "scripted-linear", "model-linear"] | None
    handoff_kind: TeamHandoffKind
    handoff_team: str | None
    handoff_workspace_visibility: str | None
    handoff_trajectory_visibility: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _TeamHandoffPolicy:
    team: str
    workspace_visibility: str
    trajectory_visibility: str | None


@dataclass(frozen=True)
class CompiledUserRuntime:
    """Concrete user runtime compiled from document-declared user metadata."""

    contract: UserRuntimeContract
    user: BaseUser | None
    max_rounds: int | None


@dataclass(frozen=True)
class TaskPromptPlan:
    """Executable prompt plan selected from a task package."""

    composition: PromptComposition
    order: tuple[PromptPartKind, ...]
    turns: tuple[CompiledPromptTurn, ...]
    user_runtime: UserRuntimeContract

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_task_prompt_plan(
    document: TaskDocument | None,
    *,
    fallback_prompt: str,
    scenes: Iterable[Scene],
) -> TaskPromptPlan:
    """Compile a task document into deterministic runtime prompt turns.

    Legacy split-layout tasks do not have role/scene/user document syntax, so
    they compile to one fallback prompt. Native tasks may declare append or
    replace composition through ``benchflow.prompt``.
    """

    if document is None:
        return TaskPromptPlan(
            composition="replace",
            order=("base",),
            turns=(
                CompiledPromptTurn(
                    scene=None,
                    role=None,
                    prompt=fallback_prompt.strip(),
                    parts=(PromptPart(kind="base", text=fallback_prompt.strip()),),
                ),
            ),
            user_runtime=_empty_user_runtime(),
        )

    composition, order = _prompt_policy(document)
    compiled_turns = tuple(
        _compile_document_turns(document, composition=composition, order=order)
    )
    if not compiled_turns:
        base = document.instruction.strip()
        compiled_turns = (
            CompiledPromptTurn(
                scene=None,
                role=None,
                prompt=base,
                parts=(PromptPart(kind="base", text=base),),
            ),
        )

    return TaskPromptPlan(
        composition=composition,
        order=order,
        turns=compiled_turns,
        user_runtime=compile_document_user_runtime(document).contract,
    )


def materialize_prompt_plan_scenes(
    document: TaskDocument,
    *,
    fallback_prompt: str | None = None,
) -> list[Scene]:
    """Return document scenes with each turn prompt replaced by the prompt plan."""

    plan = compile_task_prompt_plan(
        document,
        fallback_prompt=fallback_prompt or document.instruction,
        scenes=document.scenes,
    )
    compiled_iter = iter(plan.turns)
    materialized: list[Scene] = []
    for scene in document.scenes:
        turns: list[Turn] = []
        for turn in scene.turns:
            compiled = next(compiled_iter)
            turns.append(Turn(role=turn.role, prompt=compiled.prompt))
        materialized.append(
            Scene(
                name=scene.name,
                roles=list(scene.roles),
                turns=turns,
                skills_dir=scene.skills_dir,
            )
        )
    return materialized


def compile_document_user_runtime(document: TaskDocument) -> CompiledUserRuntime:
    """Compile supported document user metadata into a concrete ``BaseUser``."""

    raw_nudges = document.benchflow.get("nudges")
    raw_teams = document.benchflow.get("teams")
    handoff_policy, handoff_unsupported = _compile_supported_team_handoff(document)
    nudges = raw_nudges if isinstance(raw_nudges, dict) else {}
    private_facts = document.user.get("private_facts")
    private_fact_keys: tuple[str, ...] = ()
    fact_values: dict[str, str] = {}
    unsupported: list[str] = list(handoff_unsupported)

    if raw_nudges is not None and not isinstance(raw_nudges, dict):
        unsupported.append("benchflow.nudges must be a mapping")
    if raw_teams is not None and not (
        document.user or document.user_persona or raw_nudges
    ):
        unsupported.append("benchflow.teams handoff requires document user runtime")

    if isinstance(private_facts, dict):
        private_fact_keys = tuple(sorted(str(key) for key in private_facts))
        for key, value in private_facts.items():
            if not isinstance(value, str):
                unsupported.append("user.private_facts values must be strings")
                continue
            fact_values[str(key)] = value
    elif private_facts is not None:
        unsupported.append("user.private_facts must be a mapping")

    raw_model = document.user.get("model")
    raw_stop_rule = document.user.get("stop_rule")
    model = _str_or_none(raw_model)
    stop_rule = _str_or_none(raw_stop_rule)
    raw_nudge_budget = nudges.get("nudge_budget")
    nudge_budget = (
        raw_nudge_budget
        if isinstance(raw_nudge_budget, int) and not isinstance(raw_nudge_budget, bool)
        else None
    )
    if raw_model is not None and model is None:
        unsupported.append("user.model must be a string")
    if raw_stop_rule is not None and stop_rule is None:
        unsupported.append("user.stop_rule must be a string")
    model_kind = _document_user_model_kind(model)
    if model_kind is None:
        unsupported.append(
            "document user model execution is not implemented; use model: scripted "
            "or claude-*/gpt-*/gemini-* for linear simulated users"
        )

    raw_mode = nudges.get("mode")
    mode = _str_or_none(raw_mode)
    if raw_mode is not None and mode is None:
        unsupported.append("benchflow.nudges.mode must be a string")
    if mode not in {None, "simulated-user"}:
        unsupported.append("benchflow.nudges.mode must be simulated-user")
    branchable = nudges.get("branchable") is True
    raw_branch_execution = nudges.get("branch_execution")
    branch_execution = _str_or_none(raw_branch_execution)
    branch_execution_supported = True
    if raw_branch_execution is not None and branch_execution is None:
        branch_execution_supported = False
        unsupported.append("benchflow.nudges.branch_execution must be a string")
    elif branch_execution is not None:
        if branch_execution != "option-kinds-preserved":
            branch_execution_supported = False
            unsupported.append(
                "benchflow.nudges.branch_execution supports only "
                "option-kinds-preserved; forked branch execution is not implemented"
            )
        elif not branchable:
            branch_execution_supported = False
            unsupported.append(
                "benchflow.nudges.branch_execution requires branchable: true"
            )
    branch_execution_kind = _branch_execution_kind(
        branchable,
        supported=branch_execution_supported,
    )
    confirmation_policy = _confirmation_policy_tier(nudges.get("confirmation_policy"))
    if "confirmation_policy" in nudges and confirmation_policy is None:
        unsupported.append(
            "document nudge confirmation_policy must be a string or a mapping "
            "whose values are human"
        )

    max_rounds = _max_user_rounds(
        stop_rule=stop_rule,
        nudge_budget=raw_nudge_budget,
        unsupported=unsupported,
    )
    scene_turn_count = 0
    if document.scenes:
        for scene_index, scene in enumerate(document.scenes):
            scene_turn_count += len(scene.turns)
            if len(scene.roles) != 1:
                if handoff_policy is None:
                    unsupported.append(
                        "document user runtime requires each scene to have exactly "
                        "one role unless sequential shared team handoff is declared"
                    )
                elif not _scene_has_explicit_turns(document, scene_index):
                    unsupported.append(
                        "sequential team handoff requires explicit turns for "
                        "multi-role scenes"
                    )
                continue
            role_name = scene.roles[0].name
            if any(turn.role != role_name for turn in scene.turns):
                unsupported.append(
                    "document user runtime requires scene turns to use the "
                    "scene's single role"
                )
        if scene_turn_count > max_rounds:
            unsupported.append(
                "document user runtime max rounds must cover all scene turns"
            )

    has_user_semantics = bool(
        document.user or document.user_persona or raw_nudges or raw_teams
    )
    if has_user_semantics and not fact_values:
        unsupported.append("document user runtime requires string user.private_facts")
    if not has_user_semantics and not unsupported:
        return CompiledUserRuntime(
            contract=_empty_user_runtime(),
            user=None,
            max_rounds=None,
        )

    if unsupported:
        return CompiledUserRuntime(
            contract=UserRuntimeContract(
                status="unsupported",
                reason="; ".join(dict.fromkeys(unsupported)),
                model=model,
                stop_rule=stop_rule,
                max_rounds=max_rounds,
                persona_present=bool(document.user_persona),
                private_fact_keys=private_fact_keys,
                nudge_mode=mode,
                nudge_budget=nudge_budget,
                branchable=branchable,
                branch_execution=branch_execution_kind,
                confirmation_policy=confirmation_policy,
                runtime_kind=None,
                handoff_kind=_handoff_kind(handoff_policy),
                handoff_team=handoff_policy.team if handoff_policy else None,
                handoff_workspace_visibility=(
                    handoff_policy.workspace_visibility if handoff_policy else None
                ),
                handoff_trajectory_visibility=(
                    handoff_policy.trajectory_visibility if handoff_policy else None
                ),
            ),
            user=None,
            max_rounds=None,
        )

    if model_kind == "model-linear":
        assert model is not None
        user: BaseUser = ModelDocumentNudgeUser(
            model=model,
            persona=document.user_persona,
            private_facts=fact_values,
            branchable=branchable,
            branch_execution=branch_execution_kind,
            confirmation_policy=confirmation_policy,
            handoff_kind=_handoff_kind(handoff_policy),
            handoff_team=handoff_policy.team if handoff_policy else None,
        )
    else:
        user = DocumentNudgeUser(
            persona=document.user_persona,
            private_facts=fact_values,
            branchable=branchable,
            branch_execution=branch_execution_kind,
            confirmation_policy=confirmation_policy,
            handoff_kind=_handoff_kind(handoff_policy),
            handoff_team=handoff_policy.team if handoff_policy else None,
        )
    return CompiledUserRuntime(
        contract=UserRuntimeContract(
            status="supported",
            reason=None,
            model=model,
            stop_rule=stop_rule,
            max_rounds=max_rounds,
            persona_present=bool(document.user_persona),
            private_fact_keys=private_fact_keys,
            nudge_mode=mode,
            nudge_budget=nudge_budget,
            branchable=branchable,
            branch_execution=branch_execution_kind,
            confirmation_policy=confirmation_policy,
            runtime_kind=model_kind,
            handoff_kind=_handoff_kind(handoff_policy),
            handoff_team=handoff_policy.team if handoff_policy else None,
            handoff_workspace_visibility=(
                handoff_policy.workspace_visibility if handoff_policy else None
            ),
            handoff_trajectory_visibility=(
                handoff_policy.trajectory_visibility if handoff_policy else None
            ),
        ),
        user=user,
        max_rounds=max_rounds,
    )


def _compile_supported_team_handoff(
    document: TaskDocument,
) -> tuple[_TeamHandoffPolicy | None, list[str]]:
    raw_teams = document.benchflow.get("teams")
    if raw_teams is None:
        return None, []
    if not isinstance(raw_teams, dict):
        return None, ["benchflow.teams must be a mapping"]
    if len(raw_teams) != 1:
        return None, ["benchflow.teams runtime supports exactly one team handoff"]

    team_name, raw_team = next(iter(raw_teams.items()))
    if not isinstance(team_name, str):
        return None, ["benchflow.teams keys must be team names"]
    if not isinstance(raw_team, dict):
        return None, [f"benchflow.teams.{team_name} must be a mapping"]
    team_mapping = cast(dict[str, Any], raw_team)

    team_keys = set(team_mapping)
    unsupported_team_keys = sorted(team_keys - {"handoff"})
    issues: list[str] = []
    if unsupported_team_keys:
        issues.append(
            f"benchflow.teams.{team_name} supports only handoff for runtime; "
            f"unsupported keys: {', '.join(unsupported_team_keys)}"
        )

    raw_handoff = team_mapping.get("handoff")
    if not isinstance(raw_handoff, dict):
        issues.append(f"benchflow.teams.{team_name}.handoff must be a mapping")
        return None, issues

    unsupported_handoff_keys = sorted(
        set(raw_handoff) - {"mode", "workspace_visibility", "trajectory_visibility"}
    )
    if unsupported_handoff_keys:
        issues.append(
            f"benchflow.teams.{team_name}.handoff supports only mode, "
            "workspace_visibility, and trajectory_visibility for runtime; "
            f"unsupported keys: {', '.join(unsupported_handoff_keys)}"
        )

    mode = raw_handoff.get("mode")
    if mode != "sequential":
        issues.append(f"benchflow.teams.{team_name}.handoff.mode must be sequential")

    workspace_visibility = raw_handoff.get("workspace_visibility")
    if workspace_visibility != "shared":
        issues.append(
            f"benchflow.teams.{team_name}.handoff.workspace_visibility must be shared"
        )

    trajectory_visibility = raw_handoff.get("trajectory_visibility")
    if trajectory_visibility not in {None, "none", "metadata"}:
        issues.append(
            f"benchflow.teams.{team_name}.handoff.trajectory_visibility must be "
            "none or metadata"
        )

    if issues:
        return None, issues
    return (
        _TeamHandoffPolicy(
            team=team_name,
            workspace_visibility="shared",
            trajectory_visibility=(
                trajectory_visibility
                if isinstance(trajectory_visibility, str)
                else None
            ),
        ),
        [],
    )


def _prompt_policy(
    document: TaskDocument,
) -> tuple[PromptComposition, tuple[PromptPartKind, ...]]:
    raw_policy = document.benchflow.get("prompt")
    if raw_policy is None:
        return "replace", _REPLACE_DEFAULT_ORDER
    if not isinstance(raw_policy, dict):
        raise ValueError("benchflow.prompt must be a mapping")

    raw_composition = raw_policy.get("composition", "replace")
    if raw_composition not in {"append", "replace"}:
        raise ValueError("benchflow.prompt.composition must be append or replace")
    composition = raw_composition

    default_order = (
        _APPEND_DEFAULT_ORDER if composition == "append" else _REPLACE_DEFAULT_ORDER
    )
    raw_order = raw_policy.get("order", list(default_order))
    if not isinstance(raw_order, list) or not all(
        isinstance(part, str) for part in raw_order
    ):
        raise ValueError("benchflow.prompt.order must be a list of prompt part names")
    if not raw_order:
        raise ValueError("benchflow.prompt.order must not be empty")
    unknown = [part for part in raw_order if part not in _ALLOWED_PARTS]
    if unknown:
        raise ValueError(
            "benchflow.prompt.order contains unknown prompt parts: "
            + ", ".join(sorted(unknown))
        )
    return composition, tuple(raw_order)  # type: ignore[return-value]


def _compile_document_turns(
    document: TaskDocument,
    *,
    composition: PromptComposition,
    order: tuple[PromptPartKind, ...],
) -> list[CompiledPromptTurn]:
    turns: list[CompiledPromptTurn] = []
    raw_scenes = document.frontmatter.get("scenes")
    raw_scene_items = raw_scenes if isinstance(raw_scenes, list) else []
    for scene_index, scene in enumerate(document.scenes):
        raw_scene = (
            raw_scene_items[scene_index]
            if scene_index < len(raw_scene_items)
            and isinstance(raw_scene_items[scene_index], dict)
            else {}
        )
        raw_turns = raw_scene.get("turns")
        raw_turn_items = raw_turns if isinstance(raw_turns, list) else []
        for turn_index, turn in enumerate(scene.turns):
            explicit_turn_prompt = _raw_turn_prompt(raw_turn_items, turn_index)
            part_text = {
                "base": document.instruction,
                "role": document.role_prompts.get(turn.role),
                "scene": document.scene_prompts.get(scene.name),
                "turn": explicit_turn_prompt,
            }
            parts = _select_parts(
                part_text,
                composition=composition,
                order=order,
            )
            prompt = "\n\n".join(part.text for part in parts)
            turns.append(
                CompiledPromptTurn(
                    scene=scene.name,
                    role=turn.role,
                    prompt=prompt,
                    parts=parts,
                )
            )
    return turns


def _raw_turn_prompt(raw_turns: list[Any], turn_index: int) -> str | None:
    if turn_index >= len(raw_turns):
        return None
    raw_turn = raw_turns[turn_index]
    if not isinstance(raw_turn, dict):
        return None
    value = raw_turn.get("prompt")
    return value if isinstance(value, str) else None


def _scene_has_explicit_turns(document: TaskDocument, scene_index: int) -> bool:
    raw_scenes = document.frontmatter.get("scenes")
    if not isinstance(raw_scenes, list) or scene_index >= len(raw_scenes):
        return False
    raw_scene = raw_scenes[scene_index]
    if not isinstance(raw_scene, dict):
        return False
    raw_turns = raw_scene.get("turns")
    return isinstance(raw_turns, list) and bool(raw_turns)


def _select_parts(
    part_text: dict[str, str | None],
    *,
    composition: PromptComposition,
    order: tuple[PromptPartKind, ...],
) -> tuple[PromptPart, ...]:
    available = tuple(
        PromptPart(kind=part, text=text.strip())
        for part in order
        if (text := part_text.get(part)) and text.strip()
    )
    if composition == "append":
        return available
    return available[:1]


def _max_user_rounds(
    *,
    stop_rule: str | None,
    nudge_budget: object,
    unsupported: list[str],
) -> int:
    candidates: list[int] = []
    if stop_rule is not None:
        prefix, separator, suffix = stop_rule.rpartition("-")
        if not separator or suffix != "rounds":
            unsupported.append("user.stop_rule must end with -<n>-rounds")
        else:
            count_text = prefix.rsplit("-", 1)[-1]
            if count_text.isdigit() and int(count_text) > 0:
                candidates.append(int(count_text))
            else:
                unsupported.append("user.stop_rule must include a positive round count")
    if nudge_budget is not None:
        if isinstance(nudge_budget, int) and not isinstance(nudge_budget, bool):
            if nudge_budget > 0:
                candidates.append(nudge_budget)
            else:
                unsupported.append("benchflow.nudges.nudge_budget must be positive")
        else:
            unsupported.append("benchflow.nudges.nudge_budget must be an integer")
    return min(candidates) if candidates else 5


def _empty_user_runtime() -> UserRuntimeContract:
    return UserRuntimeContract(
        status="none",
        reason=None,
        model=None,
        stop_rule=None,
        max_rounds=None,
        persona_present=False,
        private_fact_keys=(),
        nudge_mode=None,
        nudge_budget=None,
        branchable=False,
        branch_execution="none",
        confirmation_policy=None,
        runtime_kind="none",
        handoff_kind="none",
        handoff_team=None,
        handoff_workspace_visibility=None,
        handoff_trajectory_visibility=None,
    )


def _handoff_kind(policy: _TeamHandoffPolicy | None) -> TeamHandoffKind:
    if policy is None:
        return "none"
    return "sequential-shared"


def _branch_execution_kind(
    branchable: bool,
    *,
    supported: bool = True,
) -> BranchExecutionKind:
    if not supported:
        return "unsupported"
    if branchable:
        return "option-kinds-preserved"
    return "none"


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _document_user_model_kind(
    model: str | None,
) -> Literal["scripted-linear", "model-linear"] | None:
    if model in {None, "scripted", "deterministic"}:
        return "scripted-linear"
    if model.startswith(
        (
            "claude-",
            "anthropic/",
            "gpt-",
            "o1",
            "o3",
            "o4",
            "openai/",
            "gemini",
            "google/",
        )
    ):
        return "model-linear"
    return None


def _confirmation_policy_tier(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value == "human" else None
    if isinstance(value, dict):
        if value and all(item == "human" for item in value.values()):
            return "human"
        return None
    return None
