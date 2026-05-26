"""GEPA-compatible trace export for skill-eval results.

GEPA reads execution traces paired with scores to evolve the skill text.
This module is the seam between BenchFlow's ``SkillEvalResult`` and the
on-disk artifact tree GEPA consumes.

All JSON writes route through ``dumps_finite`` so non-finite floats
(``NaN`` / ``±Infinity``) are normalized to ``null`` and any that slip
through fail loudly via ``allow_nan=False`` (#426).
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
from pathlib import Path

from benchflow._paths import assert_within, safe_path_segment
from benchflow._utils.json_safe import dumps_finite

from ._core import CaseResult, EvalCase, EvalDataset, SkillEvalResult

logger = logging.getLogger(__name__)


def _load_acp_trajectory(rollout_dir: Path) -> list[dict] | None:
    """Read the ACP event trajectory written by rollout.

    Rollout writes JSONL at ``trajectory/acp_trajectory.jsonl``. We return
    a parsed list of events so GEPA exports can include the real
    execution trace (#425). Returns ``None`` when no trajectory file is
    present (e.g. very old rollout layout).
    """
    traj_file = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    if not traj_file.exists():
        # Fall back to the legacy /logs/agent/acp_trajectory.jsonl path
        # the judge reads, in case future layouts move things.
        traj_file = rollout_dir / "agent" / "acp_trajectory.jsonl"
    if not traj_file.exists():
        return None
    events: list[dict] = []
    for line in traj_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip malformed lines rather than abort the whole export.
            continue
    return events or None


def _load_prompt(rollout_dir: Path, case: EvalCase) -> str | None:
    """Return the prompt that was fed to the agent for ``case``.

    Rollout dirs that include a ``prompts.json`` use that; otherwise we
    fall back to the case's ``question`` so trace consumers always have
    something to pair the trajectory against.
    """
    prompts_file = rollout_dir / "prompts.json"
    if prompts_file.exists():
        with contextlib.suppress(json.JSONDecodeError, KeyError, ValueError):
            data = json.loads(prompts_file.read_text())
            if isinstance(data, dict):
                if isinstance(data.get("prompt"), str):
                    return data["prompt"]
                if isinstance(data.get("messages"), list) and data["messages"]:
                    first = data["messages"][0]
                    if isinstance(first, dict) and isinstance(
                        first.get("content"), str
                    ):
                        return first["content"]
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and isinstance(first.get("content"), str):
                    return first["content"]
    return case.question


def export_gepa_traces(
    result: SkillEvalResult,
    dataset: EvalDataset,
    output_dir: str | Path,
) -> Path:
    """Export skill eval results in GEPA-compatible format.

    GEPA reads execution traces paired with scores to evolve the skill text.

    Output structure:
        output_dir/
        ├── skill.md              # current SKILL.md content
        ├── traces/               # one file per case × agent
        │   ├── case-001-claude.json
        │   └── ...
        └── summary.json          # aggregate scores

    Each trace file now embeds the real execution trajectory (ACP events
    + prompt + tool-call count) when one is available from the underlying
    rollout (#425). Trainers/reviewers no longer get a summary-only
    artifact under a name that promises a trace.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)

    # Copy current SKILL.md
    skill_md = dataset.skill_dir / "SKILL.md"
    if skill_md.exists():
        shutil.copy2(skill_md, output_dir / "skill.md")

    skill_text = skill_md.read_text() if skill_md.exists() else None

    # Index cases for prompt fallback when CaseResult.prompt is missing.
    case_by_id = {c.id: c for c in dataset.cases}

    # Write per-case traces
    for cr in result.case_results:
        # Reject case ids that would path-traverse out of traces_dir via the
        # generated filename. load_eval_dataset already rejects unsafe ids,
        # but CaseResult objects can be constructed independently (e.g. by a
        # caller assembling a SkillEvalResult by hand).
        safe_path_segment(cr.case_id, kind="case id")
        agent_label = cr.agent.split("/")[-1] if "/" in cr.agent else cr.agent
        mode = "with" if cr.with_skill else "without"
        trace_file = traces_dir / f"{cr.case_id}-{agent_label}-{mode}.json"
        assert_within(trace_file, traces_dir)

        # Derive trace fields from CaseResult; fall back to the dataset's
        # question when no rollout-side prompt was captured so consumers
        # always see what the agent was asked.
        case = case_by_id.get(cr.case_id)
        prompt = cr.prompt or (case.question if case else None)
        trajectory = cr.trajectory or []
        tool_calls = [
            event
            for event in trajectory
            if isinstance(event, dict)
            and (
                event.get("type") in ("tool_use", "tool_call", "tool_result")
                or "tool" in event
            )
        ]

        trace_payload = {
            "case_id": cr.case_id,
            "agent": cr.agent,
            "model": cr.model,
            "with_skill": cr.with_skill,
            "score": cr.reward,
            "rubric_results": cr.rubric_results,
            "n_tool_calls": cr.n_tool_calls,
            "error": cr.error,
            "skill_text": skill_text,
            # GEPA-shaped trace fields (#425). ``trajectory`` is the
            # full ACP event list; ``tool_calls`` is a derived
            # convenience view; ``prompt`` pairs the trace with the
            # question that produced it.
            "prompt": prompt,
            "trajectory": trajectory,
            "tool_calls": tool_calls,
        }

        trace_file.write_text(dumps_finite(trace_payload, indent=2))

    # Write summary
    summary = {
        "skill_name": result.skill_name,
        "n_cases": result.n_cases,
        "agents": result.agents,
        "lifts": [
            {
                "agent": lift.agent,
                "model": lift.model,
                "with_skill_score": lift.with_skill_score,
                "baseline_score": lift.baseline_score,
                "lift": lift.lift,
                "with_skill_passed": lift.with_skill_passed,
                "baseline_passed": lift.baseline_passed,
                "baseline_ran": lift.baseline_ran,
            }
            for lift in result.agent_lifts
        ],
    }
    (output_dir / "summary.json").write_text(dumps_finite(summary, indent=2))

    logger.info(f"GEPA traces exported to {output_dir}")
    return output_dir


# Re-export the type used by callers reaching into the dataclass directly.
__all__ = ["export_gepa_traces", "CaseResult"]
