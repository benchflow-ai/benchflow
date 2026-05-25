"""Skill evaluation — generate tasks from evals.json, run with/without skill, compare.

Usage:
    from benchflow.skill_eval import SkillEvaluator
    evaluator = SkillEvaluator(skill_dir="my-skill/")
    result = await evaluator.run(agents=["claude-agent-acp"], environment="docker")

The package is split into:

- :mod:`._core` — dataclasses, dataset loading, task generation, runner
- :mod:`.schema` — Pydantic models that validate ``evals.json``
- :mod:`.gepa_export` — GEPA-format trace export
"""

from ._core import (
    JUDGE_API_ENV_KEYS,
    TEMPLATES_DIR,
    AgentLift,
    CaseResult,
    EvalCase,
    EvalDataset,
    SkillEvalResult,
    SkillEvaluator,
    cleanup_tasks,
    generate_tasks,
    load_eval_dataset,
)
from .gepa_export import export_gepa_traces
from .schema import DEFAULT_SKILL_MOUNT_DIR

__all__ = [
    "DEFAULT_SKILL_MOUNT_DIR",
    "JUDGE_API_ENV_KEYS",
    "TEMPLATES_DIR",
    "AgentLift",
    "CaseResult",
    "EvalCase",
    "EvalDataset",
    "SkillEvalResult",
    "SkillEvaluator",
    "cleanup_tasks",
    "export_gepa_traces",
    "generate_tasks",
    "load_eval_dataset",
]
