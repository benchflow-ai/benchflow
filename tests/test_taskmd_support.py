"""Every key the reference tools document gets a decision: honored, refused by name, or recorded."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from benchflow.taskmd import plan
from benchflow.taskmd._vendor import taskmd as ref

VALUES = {  # a plausible value per key name; anything else gets "x"
    "timeout": "1m",
    "build_timeout": "1m",
    "interval": "5s",
    "start_period": "0s",
    "start_interval": "5s",
    "memory": "1 GB",
    "disk": "1 GB",
    "cpus": 1,
    "gpus": 0,
    "retries": 3,
    "samples": 1,
    "min_samples": 1,
    "network": "none",
    "env": {},
    "outputs": ["/work/a"],
    "services": [{"name": "db", "image": "postgres"}],
    "ready": {"run": "true"},
    "snapshot": [{"run": "true", "reads": ["/data"]}],
    "budget": {"tool_calls": 1},
    "tokens": 100,
    "tool_calls": 1,
    "lazy": True,
    "models": False,
    "mount": "/verifier",
    "isolation": "shared",
    "gpu_types": ["a100"],
    "keywords": ["k"],
    "authors": ["A <a@b.c>"],
    "mounts": [],
    "trials": 1,
    "unlock": "on_submit",
    "params": {},
    "splits": {},
    "seed_param": "seed",
    "generator": "family/gen.py",
    "fork": True,
    "max_concurrent": 1,
    "require": ["tool-calls"],
    "on_timeout": "grade",
    "timeout_basis": "wall",
    "boundary": "container",
    "os": "linux",
    "feedback": "none",
    "views": [],
    "resources": {},
    "controls": {},
    "answers": [],
    "combine_stages": "mean",
    "unreached_stages": "zero",
    "services_judge": False,
}


@dataclass
class _Document:
    config: dict
    blocks: list = field(default_factory=list)
    rubric: dict | None = None
    behaviors: dict | None = None
    canary: str | None = None


def _expand(path: tuple) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = [()]
    for part in path:
        options = part if isinstance(part, tuple) else (part,)
        out = [(*p, o) for p in out for o in options]
    return out


PLACEHOLDERS = ("*", "<name>", "<table>", "<format>")


def _build(parts: tuple[str, ...], leaf: object) -> object:
    if not parts:
        return leaf
    head, inner = parts[0], _build(parts[1:], leaf)
    if head == "[]":
        return [inner if isinstance(inner, dict) else {"v": inner}]
    if head in PLACEHOLDERS:
        return {"k": inner}
    if head == "<role>":
        return {"agent": inner}
    return {head: inner}


def _config_for(path: tuple[str, ...]) -> dict:
    """A config holding one documented key, with a plausible value."""
    leaf = "x" if path[-1] in PLACEHOLDERS else VALUES.get(path[-1], "x")
    config = _build(path, leaf)
    assert isinstance(config, dict)
    return config


DOCUMENTED = sorted({p for row in ref.KEY_DOCS for p in _expand(row[0])})


def test_every_documented_key_has_a_decision(tmp_path: Path) -> None:
    undecided = []
    for path in DOCUMENTED:
        if path[0].startswith("x-"):
            continue
        config = _config_for(path)
        result = plan.plan_package(_Document(config), tmp_path)
        silent = [
            f
            for f in result.refused
            if "does not implement this setting yet" in f.detail
        ]
        if silent:
            undecided.append((path, [str(f) for f in silent]))
    assert undecided == []


def test_nothing_is_dropped_silently(tmp_path: Path) -> None:
    """A key no handler names is refused, never passed over."""
    result = plan.plan_package(
        _Document({"sandbox": {"future_key": 1}, "agent": {"another": 2}}), tmp_path
    )
    fields = {f.field for f in result.refused}
    assert {"[sandbox] future_key", "[agent] another"} <= fields


@pytest.mark.parametrize("status", sorted({row[1] for row in plan.SUPPORT}))
def test_the_support_table_uses_known_statuses(status: str) -> None:
    assert status in (
        plan.HONORED,
        plan.REFUSED,
        plan.PARTIAL,
        plan.AGENT_REFUSED,
        plan.RECORDED,
    )


def test_the_support_table_names_every_table() -> None:
    text = " ".join(row[0] for row in plan.SUPPORT)
    for table in sorted(ref.TABLES):
        assert f"[{table}" in text or f"[[{table}" in text, table


LABELS = {
    plan.HONORED: "Honored",
    plan.PARTIAL: "Partly",
    plan.REFUSED: "Refused",
    plan.AGENT_REFUSED: "Refused for agents",
    plan.RECORDED: "Recorded",
}


def test_the_docs_print_the_support_table() -> None:
    docs = Path(__file__).parent.parent / "docs" / "task-authoring-taskmd-v2.md"
    text = docs.read_text()
    for pattern, status, detail in plan.SUPPORT:
        assert f"| {pattern} | {LABELS[status]} | {detail} |" in text, pattern
