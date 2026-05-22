"""Structured run-visibility policy for the dashboard jobs tree."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_GENERIC_TASK_NAMES = {
    "task",
    "acp-smoke",
    "hello-world",
    "hello-world-task",
}
_HOSTED_ENV_TOKENS = {"daytona", "modal"}


@dataclass(frozen=True)
class TokenRule:
    label: str
    tokens_any: tuple[str, ...] = ()
    phrases_any: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class RunTaskEvidence:
    path: str
    name: str
    rollout: str
    agent: str
    model: str
    environment: str
    outcome: str | None
    source_repo: str | None
    source_path: str | None
    reward_present: bool
    memory_score_present: bool
    trajectory_events: int
    artifact_count: int

    @property
    def has_source(self) -> bool:
        return bool(self.source_repo or self.source_path)

    @property
    def is_generic(self) -> bool:
        name_key = _task_name_key(self.name)
        if name_key:
            return name_key in _GENERIC_TASK_NAMES
        return _task_name_key(self.rollout.split("__", 1)[0]) in _GENERIC_TASK_NAMES


@dataclass(frozen=True)
class RunVisibilityContext:
    group_name: str
    run_id: str
    run_path: str
    summary: Mapping[str, Any]
    tasks: Sequence[RunTaskEvidence]


@dataclass(frozen=True)
class RunVisibilityDecision:
    archived: bool
    archive_reason: str | None
    signals: tuple[str, ...]
    targets: tuple[str, ...]


TARGET_RULES: tuple[TokenRule, ...] = (
    TokenRule(
        "Harvey LAB",
        tokens_any=("harvey",),
        phrases_any=(("harvey", "lab"),),
    ),
    TokenRule("ProgramBench", tokens_any=("programbench",)),
    TokenRule(
        "SkillsBench",
        tokens_any=("skillsbench",),
        phrases_any=(("skill", "eval"),),
    ),
    TokenRule(
        "Terminal-Bench",
        tokens_any=("terminalbench", "tb2"),
        phrases_any=(("terminal", "bench"),),
    ),
    TokenRule(
        "Trace-to-task",
        tokens_any=("opentraces", "huggingface"),
        phrases_any=(("trace", "to", "task"),),
    ),
    TokenRule("Adapters", tokens_any=("adapter", "adapters", "inbound")),
    TokenRule(
        "Sandbox/agent",
        tokens_any=("sandbox",),
        phrases_any=(
            ("sandbox", "agent"),
            ("agent", "decoupling"),
            ("sandbox", "decoupling"),
        ),
    ),
    TokenRule(
        "v0.5",
        tokens_any=("v05",),
        phrases_any=(("v0", "5"), ("feature", "rollouts")),
    ),
)


def decide_run_visibility(context: RunVisibilityContext) -> RunVisibilityDecision:
    """Return the dashboard visibility decision for one run."""
    if not context.tasks:
        return RunVisibilityDecision(
            archived=True,
            archive_reason="no rollout tasks",
            signals=(),
            targets=(),
        )
    if not any(task.artifact_count for task in context.tasks):
        return RunVisibilityDecision(
            archived=True,
            archive_reason="no auditable artifacts",
            signals=(),
            targets=(),
        )

    tokens = _context_tokens(context)
    targets = _matching_labels(TARGET_RULES, tokens)
    signals: set[str] = set()

    if _summary_has_source(context.summary) or any(
        task.has_source for task in context.tasks
    ):
        signals.add("provenance")
    if _has_hosted_env(context):
        signals.add("hosted-env")
    if _summary_concurrency(context.summary) >= 32:
        signals.add("high-concurrency")

    non_generic = [task for task in context.tasks if not task.is_generic]
    if any(task.trajectory_events for task in non_generic):
        signals.add("trajectory")
    if any(task.memory_score_present for task in non_generic):
        signals.add("memory")
    if any(task.outcome in {"errored", "verifier_errored"} for task in non_generic):
        signals.add("failure-evidence")
    if any(task.reward_present for task in non_generic):
        signals.add("reward")

    ordered_signals = tuple(sorted(signals))
    active = bool(targets or ordered_signals)
    return RunVisibilityDecision(
        archived=not active,
        archive_reason=None if active else "no target benchmark tag or result signal",
        signals=ordered_signals,
        targets=targets,
    )


def _context_tokens(context: RunVisibilityContext) -> tuple[str, ...]:
    fields: list[Any] = [context.group_name, context.run_id, context.run_path]
    fields.extend(_flatten(context.summary))
    for task in context.tasks:
        fields.extend(
            (
                task.path,
                task.name,
                task.rollout,
                task.agent,
                task.model,
                task.environment,
                task.outcome,
                task.source_repo,
                task.source_path,
            )
        )
    return _tokens(fields)


def _flatten(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _flatten(item)
    elif isinstance(value, str) or not isinstance(value, Iterable):
        yield value
    else:
        for item in value:
            yield from _flatten(item)


def _tokens(values: Iterable[Any]) -> tuple[str, ...]:
    tokens: list[str] = []
    for value in values:
        if value is None:
            continue
        tokens.extend(_TOKEN_RE.findall(str(value).lower()))
    return tuple(tokens)


def _matching_labels(
    rules: Sequence[TokenRule], tokens: Sequence[str]
) -> tuple[str, ...]:
    token_set = set(tokens)
    return tuple(
        rule.label
        for rule in rules
        if any(token in token_set for token in rule.tokens_any)
        or any(_has_phrase(tokens, phrase) for phrase in rule.phrases_any)
    )


def _has_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    last_start = len(tokens) - len(phrase)
    return any(
        tuple(tokens[start : start + len(phrase)]) == tuple(phrase)
        for start in range(last_start + 1)
    )


def _summary_has_source(summary: Mapping[str, Any]) -> bool:
    return bool(summary.get("source"))


def _summary_concurrency(summary: Mapping[str, Any]) -> int:
    try:
        return int(summary.get("concurrency") or 0)
    except (TypeError, ValueError):
        return 0


def _has_hosted_env(context: RunVisibilityContext) -> bool:
    envs = [context.summary.get("environment")]
    envs.extend(task.environment for task in context.tasks)
    return any(str(env).lower() in _HOSTED_ENV_TOKENS for env in envs if env)


def _task_name_key(value: str) -> str:
    return value.strip().lower().replace("_", "-")
