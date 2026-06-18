#!/usr/bin/env python3
"""Deterministic integration-test matrix planner (7-set taxonomy).

Maps a PR's changed files to the cheapest defensible integration matrix and
emits the matrix JSON (the MATRIX CELL SCHEMA + top-level planner output). Pure
stdlib + optional pyyaml; no secrets, no ``benchflow`` import, so it runs under
bare ``python3`` from trusted main (the planner is CI infrastructure and must
never run a PR's own copy).

Pipeline (all pure functions, individually unit-tested):

* :func:`load_maps`        — parse scope_map.yml + scope_defaults.yml.
* :func:`classify_buckets` — changed files -> matched Default-config-rules.
* :func:`resolve_scope`    — matched rules (+ explicit override) -> task-set,
                             unioned axes, level, trust/cheat/network flags.
* :func:`expand_set_to_cells` — task-set + axes + affected agents -> deduped
                             matrix cells (MATRIX CELL SCHEMA).
* :func:`enforce_caps`     — hard ceiling (exit 2 + rejected_overflow, never
                             silently drop) + aggregate-concurrency clamp.

scope=auto derives the set from the diff via the Default-config-rules table; an
explicit ``--scope`` overrides. The hard ceiling is fail-closed: if the matrix
exceeds ``caps.max_cells`` the planner sets ``rejected_overflow`` and exits 2.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pyyaml is optional; fall back to a tiny loader for our flat data files.
    import yaml as _yaml
except ImportError:  # pragma: no cover - exercised only without pyyaml
    _yaml = None

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
INTEGRATION_DIR = REPO_ROOT / ".github" / "integration"
SCOPE_MAP_PATH = INTEGRATION_DIR / "scope_map.yml"
SCOPE_DEFAULTS_PATH = INTEGRATION_DIR / "scope_defaults.yml"

EXIT_OK = 0
EXIT_OVERFLOW = 2

# The 7 named task-sets (plus operator-only custom) in strength order. `none`
# is the docs-only sentinel: NO rollout, level L0.
SCOPE_CHOICES = (
    "auto",
    "citation",
    "low-smoke",
    "low-3",
    "medium-3",
    "high-3",
    "nine",
    "expanded",
    "custom",
)

# Enforcement-level ordering (max wins across matched rules).
_LEVEL_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One Default-config-rules row from scope_map.yml."""

    id: str
    globs: tuple[str, ...]
    required_set: str
    custom_tasks: tuple[str, ...]
    sandboxes: tuple[str, ...]
    skill_modes: tuple[str, ...]
    extra: tuple[str, ...]
    level: str
    trust_boundary: bool
    cheat_on_trust: bool
    network_lane: bool


@dataclass(frozen=True)
class Caps:
    max_cells: int
    max_agents: int
    max_tasks: int
    per_agent_concurrency: int
    aggregate_concurrency: int
    agent_idle_timeout: int
    comment_trials_cap: int


@dataclass
class Maps:
    """Loaded scope_map.yml + scope_defaults.yml."""

    rules: list[Rule]
    task_sets: dict[str, list[str]]
    set_rank: list[str]
    agents: list[str]
    affected_agent_map: list[tuple[str, str]]
    network_trigger_globs: list[str]
    caps: Caps
    baseline_agent: str
    baseline_model: str
    canonical_high_task: str
    citation_vehicle: str
    judge_model: str
    default_model: str
    agent_models: dict[str, str]
    # Breadth-tiered roster SUBSET emitted at L2 (all-agents-subset). Defaults to
    # [baseline_agent] when absent. The FULL DeepSeek roster runs at L3/expanded.
    roster_subset: list[str]
    # The DeepSeek-only roster fanned in the BROAD lanes (all-agents at
    # L3/expanded). Gated native agents (claude-agent-acp, codex-acp) are NOT in
    # this list — they reach the matrix only via affected-agent. Defaults to
    # [baseline_agent] when absent.
    deepseek_roster: list[str]
    timeout_minutes: dict[str, int]
    # DeepSeek task-difficulty tiering: high-difficulty tasks on the flash lane
    # are promoted to the pro model. Empty pro_model disables tiering.
    pro_model: str = ""
    pro_tasks: frozenset[str] = frozenset()

    def model_for(self, agent: str) -> str:
        return self.agent_models.get(agent, self.default_model)


@dataclass(frozen=True)
class Resolution:
    """The resolved scope: task-set + unioned axes + flags + driver rules."""

    scope: str
    tasks: list[str]
    sandboxes: list[str]
    skill_modes: list[str]
    extra: list[str]
    level: str
    trust_boundary: bool
    cheat: bool
    network_lane: bool
    rule_ids: list[str]
    affected_agents: list[str]
    baseline: str  # pinned | rerun-base


@dataclass
class Cell:
    """One matrix cell (the MATRIX CELL SCHEMA)."""

    id: str
    level: str  # light | scope | final
    task: str
    agent: str
    model: str
    judge_model: str
    sandbox: str
    skill_mode: str
    network_mode: str  # default-off | allowlist
    timeout_minutes: int
    agent_idle_timeout: int
    audit_skills: bool
    expect_reward: str  # ==1.0 | <1.0 | any

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "task": self.task,
            "agent": self.agent,
            "model": self.model,
            "judge_model": self.judge_model,
            "sandbox": self.sandbox,
            "skill_mode": self.skill_mode,
            "network_mode": self.network_mode,
            "timeout_minutes": self.timeout_minutes,
            "agent_idle_timeout": self.agent_idle_timeout,
            "audit_skills": self.audit_skills,
            "expect_reward": self.expect_reward,
        }


@dataclass
class Plan:
    """A full matrix plan, serializable to the planner top-level output."""

    head_sha: str
    base_ref: str
    scope: str
    buckets: list[str]
    trust_boundary: bool
    cheat: bool
    network_lane: bool
    baseline: str
    caps: Caps
    matrix: list[Cell]
    residual_risk: list[str] = field(default_factory=list)
    rejected_overflow: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "head_sha": self.head_sha,
            "base_ref": self.base_ref,
            "scope": self.scope,
            "buckets": list(self.buckets),
            "trust_boundary": self.trust_boundary,
            "cheat": self.cheat,
            "network_lane": self.network_lane,
            "baseline": self.baseline,
            "caps": {
                "max_cells": self.caps.max_cells,
                "max_agents": self.caps.max_agents,
                "max_tasks": self.caps.max_tasks,
                "per_agent_concurrency": self.caps.per_agent_concurrency,
                "aggregate_concurrency": self.caps.aggregate_concurrency,
                "agent_idle_timeout": self.caps.agent_idle_timeout,
                "comment_trials_cap": self.caps.comment_trials_cap,
            },
            "matrix": [cell.to_json() for cell in self.matrix],
            "residual_risk": list(self.residual_risk),
            "rejected_overflow": self.rejected_overflow,
        }


class ScopeError(RuntimeError):
    """Configuration / consistency error that fails the planner closed."""


# ------------------------------------------------------------------
# YAML loading (pyyaml when present, minimal fallback otherwise)
# ------------------------------------------------------------------


def _load_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if _yaml is not None:
        return _yaml.safe_load(text)
    return _MiniYaml(text).parse()


class _MiniYaml:
    """A deliberately tiny YAML subset loader for our own flat data files.

    Supports the constructs scope_map.yml / scope_defaults.yml actually use:
    nested mappings, block sequences, ``&anchor`` / ``*alias`` on full nodes,
    quoted/bareword scalars, and ``# comments``. It is NOT a general YAML
    parser; it exists only so the planner runs without a pyyaml install.
    """

    def __init__(self, text: str) -> None:
        self._lines = self._strip(text)
        self._anchors: dict[str, Any] = {}
        self._pos = 0

    @staticmethod
    def _strip(text: str) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for raw in text.splitlines():
            no_comment = _MiniYaml._drop_comment(raw)
            if not no_comment.strip():
                continue
            indent = len(no_comment) - len(no_comment.lstrip(" "))
            out.append((indent, no_comment.strip()))
        return out

    @staticmethod
    def _drop_comment(line: str) -> str:
        in_single = in_double = False
        for i, ch in enumerate(line):
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif (
                ch == "#"
                and not in_single
                and not in_double
                and (i == 0 or line[i - 1] in " \t")
            ):
                return line[:i]
        return line

    def parse(self) -> Any:
        if not self._lines:
            return None
        return self._parse_block(self._lines[0][0])

    def _parse_block(self, indent: int) -> Any:
        if self._pos >= len(self._lines):
            return None
        _first_indent, first = self._lines[self._pos]
        if first.startswith("- "):
            return self._parse_sequence(indent)
        return self._parse_mapping(indent)

    def _parse_sequence(self, indent: int) -> list[Any]:
        items: list[Any] = []
        while self._pos < len(self._lines):
            cur_indent, cur = self._lines[self._pos]
            if cur_indent < indent or not cur.startswith("- "):
                break
            content = cur[2:].strip()
            self._pos += 1
            if content.startswith("*"):
                items.append(self._resolve_alias(content))
            elif ":" in content and not self._is_scalar_with_colon(content):
                # Inline first key of a mapping item; re-inject as a mapping line.
                self._lines.insert(self._pos, (indent + 2, content))
                items.append(self._parse_mapping(indent + 2))
            else:
                items.append(self._scalar(content))
        return items

    def _parse_mapping(self, indent: int) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        while self._pos < len(self._lines):
            cur_indent, cur = self._lines[self._pos]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                break
            if cur.startswith("- "):
                break
            key, _, rest = cur.partition(":")
            key = key.strip()
            rest = rest.strip()
            anchor = None
            self._pos += 1
            if rest.startswith("&"):
                anchor, _, rest = rest.partition(" ")
                anchor = anchor[1:]
                rest = rest.strip()
            if rest.startswith("*"):
                value: Any = self._resolve_alias(rest)
            elif rest:
                value = self._scalar(rest)
            else:
                value = self._parse_child(indent)
            if anchor is not None:
                self._anchors[anchor] = value
            mapping[key] = value
        return mapping

    def _parse_child(self, parent_indent: int) -> Any:
        if self._pos >= len(self._lines):
            return None
        child_indent, child = self._lines[self._pos]
        if child_indent <= parent_indent:
            return None
        if child.startswith("- ") and child_indent == parent_indent:
            return self._parse_sequence(child_indent)
        return self._parse_block(child_indent)

    def _resolve_alias(self, token: str) -> Any:
        name = token[1:].strip()
        if name not in self._anchors:
            raise ScopeError(f"unknown YAML alias *{name}")
        return self._anchors[name]

    @staticmethod
    def _is_scalar_with_colon(content: str) -> bool:
        # "a: b" -> mapping; "https://x" -> scalar. Treat as scalar only if the
        # colon is inside quotes or has no following space.
        idx = content.find(":")
        if idx == -1:
            return True
        after = content[idx + 1 : idx + 2]
        return after not in ("", " ")

    @staticmethod
    def _scalar(token: str) -> Any:
        if (token.startswith('"') and token.endswith('"')) or (
            token.startswith("'") and token.endswith("'")
        ):
            return token[1:-1]
        if token == "[]":
            return []
        if token == "{}":
            return {}
        low = token.lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
        if low in ("null", "~", ""):
            return None
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            pass
        return token


# ------------------------------------------------------------------
# load_maps()
# ------------------------------------------------------------------


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _as_list(value: Any) -> list[str]:
    return list(_as_tuple(value))


def load_maps(
    scope_map_path: Path = SCOPE_MAP_PATH,
    scope_defaults_path: Path = SCOPE_DEFAULTS_PATH,
) -> Maps:
    """Load and validate scope_map.yml + scope_defaults.yml into a ``Maps``."""
    raw_map = _load_yaml(scope_map_path) or {}
    raw_defaults = _load_yaml(scope_defaults_path) or {}

    rules: list[Rule] = []
    for entry in raw_map.get("rules", []) or []:
        axes = entry.get("required_axes") or {}
        rules.append(
            Rule(
                id=str(entry["id"]),
                globs=_as_tuple(entry.get("globs")),
                required_set=str(entry.get("required_set", "none")),
                custom_tasks=_as_tuple(entry.get("custom_tasks")),
                sandboxes=_as_tuple(axes.get("sandboxes")),
                skill_modes=_as_tuple(axes.get("skill_modes")),
                extra=_as_tuple(axes.get("extra")),
                level=str(entry.get("level", "L1")),
                trust_boundary=bool(entry.get("trust_boundary", False)),
                cheat_on_trust=bool(entry.get("cheat_on_trust", False)),
                network_lane=bool(entry.get("network_lane", False)),
            )
        )

    task_sets: dict[str, list[str]] = {}
    for name, tasks in (raw_map.get("task_sets") or {}).items():
        task_sets[str(name)] = _as_list(tasks)

    affected_agent_map: list[tuple[str, str]] = []
    for entry in raw_map.get("affected_agent_map", []) or []:
        affected_agent_map.append((str(entry["glob"]), str(entry["agent"])))

    caps_raw = raw_defaults.get("caps") or {}
    caps = Caps(
        max_cells=int(caps_raw["max_cells"]),
        max_agents=int(caps_raw["max_agents"]),
        max_tasks=int(caps_raw["max_tasks"]),
        per_agent_concurrency=int(caps_raw["per_agent_concurrency"]),
        aggregate_concurrency=int(caps_raw["aggregate_concurrency"]),
        agent_idle_timeout=int(caps_raw.get("agent_idle_timeout", 240)),
        comment_trials_cap=int(caps_raw.get("comment_trials_cap", 3)),
    )

    agent_models: dict[str, str] = {
        str(k): str(v) for k, v in (raw_defaults.get("agent_models") or {}).items()
    }
    baseline_agent = str(raw_defaults["baseline_agent"])
    # Breadth-tiered L2 subset; default to the baseline agent alone when absent.
    roster_subset = _as_list(raw_defaults.get("roster_subset")) or [baseline_agent]
    # DeepSeek-only broad-lane roster; default to the baseline agent alone.
    deepseek_roster = _as_list(raw_defaults.get("deepseek_roster")) or [baseline_agent]
    timeout_minutes: dict[str, int] = {
        str(k): int(v) for k, v in (raw_defaults.get("timeout_minutes") or {}).items()
    }

    # DeepSeek model tiering: high-difficulty tasks use the pro model. Falls
    # back to the high-3 task set when pro_tasks is not listed explicitly.
    tiering = raw_defaults.get("deepseek_tiering") or {}
    pro_model = str(tiering.get("pro_model", ""))
    pro_tasks = frozenset(
        str(t) for t in (tiering.get("pro_tasks") or task_sets.get("high-3", []))
    )

    maps = Maps(
        rules=rules,
        task_sets=task_sets,
        set_rank=_as_list(raw_map.get("set_rank")),
        agents=_as_list(raw_map.get("roster")),
        affected_agent_map=affected_agent_map,
        network_trigger_globs=_as_list(raw_map.get("network_trigger_globs")),
        caps=caps,
        baseline_agent=baseline_agent,
        baseline_model=str(raw_defaults["baseline_model"]),
        canonical_high_task=str(raw_defaults["canonical_high_task"]),
        citation_vehicle=str(raw_defaults["citation_vehicle"]),
        judge_model=str(raw_defaults["judge_model"]),
        default_model=str(raw_defaults.get("default_model", "")),
        agent_models=agent_models,
        roster_subset=roster_subset,
        deepseek_roster=deepseek_roster,
        timeout_minutes=timeout_minutes,
        pro_model=pro_model,
        pro_tasks=pro_tasks,
    )
    _validate_caps_consistency(maps)
    return maps


def _validate_caps_consistency(maps: Maps) -> None:
    """Fail closed if caps cannot hold the largest single-axis enumeration.

    The worst-case ``nine`` per-(agent,task) expansion is max_agents * max_tasks
    cells, so max_cells must be at least that. Aggregate concurrency must respect
    the documented Daytona ceiling of 24.
    """
    caps = maps.caps
    floor = caps.max_agents * caps.max_tasks
    if caps.max_cells < floor:
        raise ScopeError(
            "inconsistent caps: max_cells "
            f"({caps.max_cells}) < max_agents*max_tasks ({floor}); "
            "a nine matrix needs one cell per (agent, task)"
        )
    if caps.max_agents > len(maps.agents):
        raise ScopeError(
            f"max_agents ({caps.max_agents}) exceeds roster size ({len(maps.agents)})"
        )
    if caps.per_agent_concurrency < 1:
        raise ScopeError("per_agent_concurrency must be >= 1")
    if caps.aggregate_concurrency > 24:
        raise ScopeError(
            "aggregate_concurrency ceiling "
            f"({caps.aggregate_concurrency}) exceeds the documented Daytona "
            "ceiling of 24"
        )
    for name in ("nine", "expanded", "low-3", "medium-3", "high-3", "citation"):
        if name not in maps.task_sets:
            raise ScopeError(f"task_sets missing required set {name!r}")


# ------------------------------------------------------------------
# classify_buckets()
# ------------------------------------------------------------------


def _matches(path: str, glob: str) -> bool:
    norm = path.replace("\\", "/")
    # Strip a single leading "./" prefix only (NOT a char-set lstrip, which would
    # eat a leading dot from ".agents/..." and break dotfile-dir globs).
    if norm.startswith("./"):
        norm = norm[2:]
    if fnmatch.fnmatch(norm, glob):
        return True
    # Treat "dir/**" as also matching "dir/file" (fnmatch needs the trailing /).
    if glob.endswith("/**") and fnmatch.fnmatch(norm, glob[:-3] + "/*"):
        return True
    # Treat "**/x" as also matching a top-level "x".
    return glob.startswith("**/") and fnmatch.fnmatch(norm, glob[3:])


def classify_buckets(files: Sequence[str], maps: Maps) -> list[Rule]:
    """Return the Default-config-rules matched by the changed ``files``.

    Stable order (scope_map.yml rule order). A file may match several rules; the
    planner unions them in :func:`resolve_scope`.
    """
    matched: list[Rule] = []
    for rule in maps.rules:
        if any(_matches(f, glob) for f in files for glob in rule.globs):
            matched.append(rule)
    return matched


def affected_agents(files: Sequence[str], maps: Maps) -> list[str]:
    """Derive affected agent names from changed src/benchflow/agents/<name> paths."""
    found: list[str] = []
    for glob, agent in maps.affected_agent_map:
        if any(_matches(f, glob) for f in files) and agent not in found:
            found.append(agent)
    return found


def network_triggered(files: Sequence[str], maps: Maps) -> bool:
    """True if any changed file hits a Q3 network trigger glob."""
    return any(_matches(f, glob) for f in files for glob in maps.network_trigger_globs)


# ------------------------------------------------------------------
# resolve_scope()
# ------------------------------------------------------------------


def _set_rank(maps: Maps, name: str) -> int:
    """Strength rank of a task-set; unknown/custom sets sort below `nine`."""
    if name in maps.set_rank:
        return maps.set_rank.index(name)
    # custom rides at the strength of its size relative to known sets; we treat
    # it as just below expanded so it is not overridden by a weaker named set.
    if name == "expanded":
        return len(maps.set_rank)
    if name == "custom":
        return len(maps.set_rank) - 1
    return -1


def _strongest_rule(maps: Maps, rules: Sequence[Rule]) -> Rule | None:
    """The matched rule whose required_set has the highest strength rank."""
    runnable = [r for r in rules if r.required_set != "none"]
    if not runnable:
        return None
    return max(runnable, key=lambda r: (_set_rank(maps, r.required_set), r.id))


def _union_axis(rules: Sequence[Rule], attr: str) -> list[str]:
    seen: list[str] = []
    for rule in rules:
        for value in getattr(rule, attr):
            if value not in seen:
                seen.append(value)
    return seen


def _set_tasks(maps: Maps, scope: str, driver: Rule | None) -> list[str]:
    """The ordered task list for a resolved scope.

    ``custom`` reads the driver rule's ``custom_tasks``; named sets read the
    task-set taxonomy. ``low-3-plus`` extra means low-3 ⋃ the set (skill /
    sandbox rules pin medium-3 but also want low-3); ``one-high`` appends the
    canonical high task (agents/ rule).
    """
    if scope == "none":
        return []
    if scope == "custom":
        if driver is None or not driver.custom_tasks:
            raise ScopeError("custom scope requires custom_tasks on the driver rule")
        return list(driver.custom_tasks)
    if scope not in maps.task_sets:
        raise ScopeError(f"unknown task-set {scope!r}")
    tasks = list(maps.task_sets[scope])
    extra = set(driver.extra) if driver else set()
    if "low-3-plus" in extra:
        for task in reversed(maps.task_sets["low-3"]):
            if task not in tasks:
                tasks.insert(0, task)
    if "one-high" in extra and maps.canonical_high_task not in tasks:
        tasks.append(maps.canonical_high_task)
    return tasks


def resolve_scope(
    buckets: Sequence[Rule],
    maps: Maps,
    *,
    override: str = "auto",
    custom_tasks: Sequence[str] = (),
    sandboxes_override: Sequence[str] = (),
    skill_modes_override: Sequence[str] = (),
    affected: Sequence[str] = (),
    network_lane_diff: bool = False,
) -> Resolution:
    """Resolve matched rules (+ explicit override) into a concrete scope.

    scope=auto derives the set from the Default-config-rules union; an explicit
    override forces the set. Axes are the union of matched rules' axes (or the
    explicit overrides). trust/cheat/network flags OR across rules.
    """
    runnable = [r for r in buckets if r.required_set != "none"]
    driver = _strongest_rule(maps, buckets)

    if override != "auto":
        scope = override
        # Synthesize a driver carrying custom_tasks if the override is custom.
        if scope == "custom":
            driver = Rule(
                id="cli-custom",
                globs=(),
                required_set="custom",
                custom_tasks=tuple(custom_tasks),
                sandboxes=(),
                skill_modes=(),
                extra=(),
                level="L2",
                trust_boundary=False,
                cheat_on_trust=False,
                network_lane=False,
            )
    elif driver is None:
        # No RUNNABLE rule matched. If some rule matched but all are explicit
        # `none` (docs-only non-runtime), stay `none` -> NO rollout. If nothing
        # matched at all (unmapped path), fall to the single-cell citation floor.
        explicit_none = any(r.required_set == "none" for r in buckets)
        scope = "none" if explicit_none else "citation"
    else:
        scope = driver.required_set

    tasks = _set_tasks(maps, scope, driver)

    # Axis union: explicit CLI overrides win; otherwise union the matched rules.
    if sandboxes_override:
        sandboxes = list(sandboxes_override)
    else:
        sandboxes = _union_axis(runnable, "sandboxes") or ["docker"]
    if skill_modes_override:
        skill_modes = list(skill_modes_override)
    else:
        skill_modes = _union_axis(runnable, "skill_modes") or ["no-skill"]
    extra = _union_axis(runnable, "extra")
    if driver is not None and driver.extra:
        for value in driver.extra:
            if value not in extra:
                extra.append(value)

    level = "L0"
    for rule in buckets:
        if _LEVEL_RANK.get(rule.level, 0) > _LEVEL_RANK.get(level, 0):
            level = rule.level
    if scope != "none" and level == "L0":
        level = "L1"

    trust = any(r.trust_boundary for r in buckets)
    cheat = any(r.cheat_on_trust for r in buckets if r.trust_boundary)
    network_lane = network_lane_diff or any(r.network_lane for r in buckets)

    baseline = "rerun-base" if scope == "expanded" else "pinned"

    return Resolution(
        scope=scope,
        tasks=tasks,
        sandboxes=sandboxes,
        skill_modes=skill_modes,
        extra=extra,
        level=level,
        trust_boundary=trust,
        cheat=cheat,
        network_lane=network_lane,
        rule_ids=[r.id for r in buckets],
        affected_agents=list(affected),
        baseline=baseline,
    )


# ------------------------------------------------------------------
# expand_set_to_cells()
# ------------------------------------------------------------------


def _cell_level(res: Resolution) -> str:
    """The workflow lane a cell belongs to (light | scope | final)."""
    if res.scope in ("citation", "low-smoke"):
        return "light"
    if res.level == "L3":
        return "final"
    return "scope"


# Scopes whose VARYING axis is the agent -> fan the full 9-agent roster. Every
# other lane varies a non-agent axis (skill_mode, sandbox, task) and runs the
# single baseline agent as its representative vehicle, keeping the cell count
# (and aggregate concurrency) bounded.
_FULL_ROSTER_SCOPES = frozenset({"nine", "expanded"})


def _matrix_agents(res: Resolution, maps: Maps) -> list[str]:
    """The agents that drive the matrix for this resolution.

    The BROAD lanes fan the DeepSeek roster ONLY. The gated native agents
    (claude-agent-acp via Bedrock, codex-acp via OpenAI) cannot use DeepSeek and
    are "currently blocked" from the default fan — they reach the matrix solely
    via affected-agent (a PR touching their own adapter). Concretely:

    - ``all-agents`` and ``nine`` / ``expanded`` fan ``deepseek_roster`` (the
      agent IS the varying axis, but only over the DeepSeek lane).
    - ``all-agents-subset`` is the breadth-tiered L2 (auto-on-push) variant: a
      representative DeepSeek SUBSET plus the baseline, so a registry / provider
      change is probed across launcher families without the full roster.
    - the agents/ rule (affected-agent) runs the affected agents — which MAY be
      a gated native (codex/claude) when its source changed — plus the baseline.
    - every other lane (citation/low/medium/high/custom) varies a non-agent axis
      and runs the single baseline agent.
    """
    if "all-agents" in res.extra:
        return list(maps.deepseek_roster)
    # nine / expanded (the heavy, manually-dispatched L3 lane) fan the full
    # DeepSeek roster and OVERRIDE the L2 breadth-tiered subset — so `expanded`
    # on an agent-infra / provider change gives every DeepSeek agent, not just
    # the subset. This MUST be checked before all-agents-subset/affected-agent,
    # since those rules also match the change and would otherwise win at L3.
    if res.scope in _FULL_ROSTER_SCOPES:
        return list(maps.deepseek_roster)
    if "all-agents-subset" in res.extra:
        agents: list[str] = []
        for agent in (*maps.roster_subset, maps.baseline_agent):
            if agent not in agents:
                agents.append(agent)
        return agents
    if "affected-agent" in res.extra and res.affected_agents:
        agents = []
        for agent in (*res.affected_agents, maps.baseline_agent):
            if agent not in agents:
                agents.append(agent)
        return agents
    return [maps.baseline_agent]


def _clamp_per_agent_concurrency(caps: Caps, n_daytona_agents: int) -> int:
    """Lower per-agent concurrency so aggregate (daytona) stays under ceiling."""
    if n_daytona_agents <= 0:
        return caps.per_agent_concurrency
    allowed = max(1, caps.aggregate_concurrency // n_daytona_agents)
    return min(caps.per_agent_concurrency, allowed)


def expand_set_to_cells(
    res: Resolution,
    maps: Maps,
) -> list[Cell]:
    """Enumerate the deduped matrix cells for a resolved scope.

    Cell-count discipline (keeps the matrix under the hard ceiling without ever
    dropping a required axis):

    * The AGENT axis fans across all matrix agents ONLY on the primary
      (sandbox, skill_mode) combo.
    * Each ADDITIONAL sandbox / skill_mode is covered by the single baseline
      agent (parity / skill-pair probes), not re-fanned across the full roster.

    So a full-roster ``expanded`` lane is ``tasks * agents`` agent cells plus a
    bounded set of baseline-agent axis-coverage cells, rather than the full
    ``tasks * sandboxes * skill_modes * agents`` cartesian product.

    Adds the network allowlist variant (network_lane), the cheat/null-patch
    anti-hack cell (cheat), and pins network_mode as the EXPECTED value
    (default-off unless allowlist). The cell set is deduped by id; cap
    enforcement happens in :func:`enforce_caps`.
    """
    cells: list[Cell] = []
    if res.scope == "none":
        return cells

    cell_level = _cell_level(res)
    timeout = maps.timeout_minutes.get(res.scope, 90)
    judge = maps.judge_model
    agents = _matrix_agents(res, maps)
    primary_sandbox = res.sandboxes[0]
    primary_skill = res.skill_modes[0]
    audit = "audit-skills" in res.extra

    seen: set[str] = set()

    def add(cell: Cell) -> None:
        if cell.id in seen:
            return
        seen.add(cell.id)
        cells.append(cell)

    def make(task: str, sandbox: str, skill_mode: str, agent: str) -> Cell:
        return Cell(
            id=f"{task}-{sandbox}-{skill_mode}-{agent}",
            level=cell_level,
            task=task,
            agent=agent,
            model=_model_for_cell(res, maps, agent, task),
            judge_model=judge,
            sandbox=sandbox,
            skill_mode=skill_mode,
            network_mode="default-off",
            timeout_minutes=timeout,
            agent_idle_timeout=maps.caps.agent_idle_timeout,
            audit_skills=audit,
            expect_reward=_expect_reward(task, skill_mode),
        )

    # The affected-agent lane is small (affected + baseline agents only), so it
    # fully fans every (sandbox, skill_mode) across BOTH agents — the SPEC wants
    # no-skill AND with-skill on the affected and baseline agent. Larger rosters
    # use the bounded axis-coverage scheme below.
    full_fan = (
        "affected-agent" in res.extra and res.affected_agents and len(agents) <= 3
    )
    multi_agent = len(agents) > 1
    for task in res.tasks:
        if full_fan:
            for sandbox in res.sandboxes:
                for skill_mode in res.skill_modes:
                    for agent in agents:
                        add(make(task, sandbox, skill_mode, agent))
            continue
        # Agent fan-out on the primary (sandbox, skill_mode) combo.
        for agent in agents:
            add(make(task, primary_sandbox, primary_skill, agent))
        # Additional axes: covered by the baseline agent only when the roster is
        # multi-agent (otherwise the single agent already fans every combo).
        cover_agent = maps.baseline_agent if multi_agent else agents[0]
        for sandbox in res.sandboxes:
            for skill_mode in res.skill_modes:
                if sandbox == primary_sandbox and skill_mode == primary_skill:
                    continue
                add(make(task, sandbox, skill_mode, cover_agent))

    # Network allowlist VARIANT: the citation vehicle, network_mode=allowlist,
    # docker/no-skill. Carried as EXPECTED only (never passed to bench).
    if res.network_lane:
        vehicle = maps.citation_vehicle
        agent = maps.baseline_agent
        add(
            Cell(
                id=f"{vehicle}-docker-no-skill-{agent}-allowlist",
                level=cell_level,
                task=vehicle,
                agent=agent,
                model=_model_for_cell(res, maps, agent, vehicle),
                judge_model=judge,
                sandbox="docker",
                skill_mode="no-skill",
                network_mode="allowlist",
                timeout_minutes=maps.timeout_minutes.get("citation", timeout),
                agent_idle_timeout=maps.caps.agent_idle_timeout,
                audit_skills=False,
                expect_reward="any",
            )
        )

    # Anti-hack cheat/null-patch lane on the canonical high task (one cell).
    if res.cheat:
        task = (
            maps.canonical_high_task
            if maps.canonical_high_task in res.tasks
            else res.tasks[-1]
        )
        agent = maps.baseline_agent
        add(
            Cell(
                id=f"{task}-docker-no-skill-{agent}-cheat",
                level=cell_level,
                task=task,
                agent=agent,
                model=_model_for_cell(res, maps, agent, task),
                judge_model=judge,
                sandbox="docker",
                skill_mode="no-skill",
                network_mode="default-off",
                timeout_minutes=timeout,
                agent_idle_timeout=maps.caps.agent_idle_timeout,
                audit_skills=False,
                expect_reward="<1.0",
            )
        )

    # Stamp the clamped concurrency budget onto the plan via caps (the workflow
    # reads caps.per_agent_concurrency). Recompute from the cells actually
    # emitted (distinct daytona agents), and halve again when a release-critical
    # lane asked for reduced concurrency.
    distinct_daytona = {c.agent for c in cells if c.sandbox == "daytona"}
    per_agent = _clamp_per_agent_concurrency(maps.caps, len(distinct_daytona))
    if "reduced-concurrency" in res.extra:
        per_agent = max(1, per_agent // 2)
    if per_agent != maps.caps.per_agent_concurrency:
        maps.caps = Caps(
            max_cells=maps.caps.max_cells,
            max_agents=maps.caps.max_agents,
            max_tasks=maps.caps.max_tasks,
            per_agent_concurrency=per_agent,
            aggregate_concurrency=maps.caps.aggregate_concurrency,
            agent_idle_timeout=maps.caps.agent_idle_timeout,
            comment_trials_cap=maps.caps.comment_trials_cap,
        )

    return cells


def _model_for_cell(res: Resolution, maps: Maps, agent: str, task: str) -> str:
    base = (
        maps.baseline_model if agent == maps.baseline_agent else maps.model_for(agent)
    )
    # DeepSeek difficulty tiering: a high-difficulty task on the flash lane is
    # promoted to the pro model. Other agents' models are untouched.
    if maps.pro_model and base == maps.baseline_model and task in maps.pro_tasks:
        return maps.pro_model
    return base


def _expect_reward(task: str, skill_mode: str) -> str:
    """Expected reward band for a cell (advisory; the grader checks bands)."""
    return "any"


# ------------------------------------------------------------------
# enforce_caps()
# ------------------------------------------------------------------


def enforce_caps(plan: Plan) -> Plan:
    """Apply the hard ceiling + aggregate-concurrency cap (fail-closed).

    If the matrix exceeds ``caps.max_cells`` the plan is marked with
    ``rejected_overflow`` and the cells are left intact (never silently
    dropped); the CLI then exits 2. Also re-asserts cell-id uniqueness and the
    aggregate daytona concurrency ceiling.
    """
    ids = [c.id for c in plan.matrix]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ScopeError(f"duplicate cell ids after dedup: {dupes}")

    distinct_daytona_agents = {
        c.agent for c in plan.matrix if c.sandbox == "daytona" and c.agent
    }
    aggregate = plan.caps.per_agent_concurrency * max(1, len(distinct_daytona_agents))
    if aggregate > plan.caps.aggregate_concurrency:
        raise ScopeError(
            "aggregate concurrency "
            f"{aggregate} exceeds cap {plan.caps.aggregate_concurrency} "
            f"({len(distinct_daytona_agents)} daytona agents x "
            f"{plan.caps.per_agent_concurrency})"
        )

    if len(plan.matrix) > plan.caps.max_cells:
        plan.rejected_overflow = (
            f"enumerated {len(plan.matrix)} cells exceeds max_cells "
            f"{plan.caps.max_cells}; refusing to silently drop cells"
        )
    return plan


# ------------------------------------------------------------------
# Top-level orchestration
# ------------------------------------------------------------------


def _residual_risk(res: Resolution) -> list[str]:
    risk: list[str] = []
    if res.scope in ("citation", "low-smoke"):
        risk.append(
            "light lane: no full agent-matrix coverage; lifecycle/hardening "
            "rely on residual + codex review"
        )
    if res.trust_boundary and not res.cheat:
        risk.append(
            "trust-boundary surface without a cheat lane; reward-hacking "
            "coverage relies on the codex equivalence judge"
        )
    if res.network_lane:
        risk.append(
            "network lane carries network_mode as EXPECTED only; the allowlist "
            "variant is not passed to bench (no --network flag exists)"
        )
    if res.baseline == "rerun-base":
        risk.append("expanded scope re-runs the base SHA for before/after equivalence")
    return risk


def build_plan(
    files: Sequence[str],
    maps: Maps,
    *,
    base_ref: str,
    head_sha: str,
    override: str = "auto",
    custom_tasks: Sequence[str] = (),
    sandboxes_override: Sequence[str] = (),
    skill_modes_override: Sequence[str] = (),
) -> Plan:
    matched = classify_buckets(files, maps)
    affected = affected_agents(files, maps)
    network_diff = network_triggered(files, maps)
    res = resolve_scope(
        matched,
        maps,
        override=override,
        custom_tasks=custom_tasks,
        sandboxes_override=sandboxes_override,
        skill_modes_override=skill_modes_override,
        affected=affected,
        network_lane_diff=network_diff,
    )
    cells = expand_set_to_cells(res, maps)
    plan = Plan(
        head_sha=head_sha,
        base_ref=base_ref,
        # `none` scope (docs-only non-runtime) is the NO-rollout sentinel; the
        # matrix is empty and the workflow reports a green no-op.
        scope=res.scope,
        buckets=res.rule_ids,
        trust_boundary=res.trust_boundary,
        cheat=res.cheat,
        network_lane=res.network_lane,
        baseline=res.baseline,
        caps=maps.caps,
        matrix=cells,
        residual_risk=_residual_risk(res),
    )
    return enforce_caps(plan)


def _changed_files_from_git(base_ref: str, head_sha: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...{head_sha}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic integration-test matrix planner (7-set).",
    )
    parser.add_argument("--pr-number", default=None)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        dest="changed_files",
        metavar="PATH",
        help="A changed file path (repeatable).",
    )
    parser.add_argument(
        "--diff-from-git",
        action="store_true",
        help="Resolve changed files via git diff BASE...HEAD.",
    )
    parser.add_argument(
        "--scope",
        choices=SCOPE_CHOICES,
        default="auto",
        help="auto derives the set from the diff; explicit value overrides.",
    )
    parser.add_argument(
        "--custom-tasks",
        default="",
        help="Comma-separated task ids for --scope custom.",
    )
    parser.add_argument(
        "--sandboxes",
        default="",
        help="Comma-separated sandbox override (docker,daytona).",
    )
    parser.add_argument(
        "--skill-modes",
        default="",
        help="Comma-separated skill-mode override (no-skill,with-skill,self-gen).",
    )
    parser.add_argument("--out", default="matrix.json")
    return parser.parse_args(list(argv))


def _resolve_files(args: argparse.Namespace) -> list[str]:
    files = list(args.changed_files)
    if args.diff_from_git:
        files.extend(_changed_files_from_git(args.base_ref, args.head_sha))
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _is_github_output(path: str) -> bool:
    return os.environ.get("GITHUB_OUTPUT") == path or path.endswith("GITHUB_OUTPUT")


def _emit(plan: Plan, out_path: str) -> None:
    """Write the matrix JSON. If --out is the GITHUB_OUTPUT path, ALSO append
    ``matrix=<json>`` and ``head_sha=<sha>`` step-output lines."""
    payload = json.dumps(plan.to_json(), indent=2) + "\n"
    if _is_github_output(out_path):
        compact = json.dumps(plan.to_json(), separators=(",", ":"))
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write(f"matrix={compact}\n")
            handle.write(f"head_sha={plan.head_sha}\n")
        return
    Path(out_path).write_text(payload, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        maps = load_maps()
        files = _resolve_files(args)
        plan = build_plan(
            files,
            maps,
            base_ref=args.base_ref,
            head_sha=args.head_sha,
            override=args.scope,
            custom_tasks=_split_csv(args.custom_tasks),
            sandboxes_override=_split_csv(args.sandboxes),
            skill_modes_override=_split_csv(args.skill_modes),
        )
    except ScopeError as exc:
        print(f"::error::matrix planner failed closed: {exc}", file=sys.stderr)
        return 1

    _emit(plan, args.out)

    if plan.rejected_overflow:
        print(f"::error::{plan.rejected_overflow}", file=sys.stderr)
        return EXIT_OVERFLOW
    print(
        f"matrix plan: scope={plan.scope} cells={len(plan.matrix)} "
        f"level{'/'.join(sorted({c.level for c in plan.matrix})) or '-'} "
        f"buckets={','.join(plan.buckets) or '(none)'}"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
