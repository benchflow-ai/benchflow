"""Prompt rendering for the rubric reviewer.

The reviewer instruction is assembled host-side from a template plus the
rubric's guidance lines and the structured-output schema, then baked into the
wrapper task's instruction body.  Templates are rendered with
``str.format_map`` over a defaulting mapping, so a custom template that omits
a placeholder renders instead of crashing.

Available placeholders:

- ``{trial_path}`` — absolute in-sandbox path of the read-only rollout evidence copy.
- ``{task_section}`` — pre-rendered paragraph describing the task-definition
  copy (or its absence).
- ``{criteria_guidance}`` — one ``- name: guidance`` line per criterion.
- ``{result_path}`` / ``{output_schema}`` — output-contract details.
- ``{trial_results}`` — job-summary template only; replaced verbatim.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from benchflow.review.config import Rubric, build_criteria_guidance

TRIAL_MOUNT = "/evidence/trial"
TASK_MOUNT = "/evidence/task"

REVIEW_TEMPLATE = """You are reviewing one finished agent run. Judge the run against each criterion listed under Guidance, giving a short rationale for every judgment.

The run's records are at {trial_path}. Read them with paths under that directory (for example "{trial_path}/result.json" or "{trial_path}/trajectory/").

{task_section}

Before judging, read every relevant record:

Run records:
- {trial_path}/result.json — outcome, rewards, and error details
- {trial_path}/trajectory/ — the agent's recorded actions
- {trial_path}/verifier/ — test output, when present
- {trial_path}/config.json — how the run was configured

Work through the criteria one at a time. For each criterion, weigh the evidence before deciding, and cite the specific files or recorded steps that support your judgment in its explanation.

Also write a "summary": three to five sentences covering what the agent attempted, the main problems it hit, and how close it came to finishing (for example: passed part of the tests, had a sound approach but stalled, or failed before making progress).

Do not modify anything under {trial_path}.

Guidance:
{criteria_guidance}
"""

TASK_SECTION_TEMPLATE = """The task the agent attempted is at {task_path}. Read its files first so you know what was required:
- {task_path}/task.md or {task_path}/instruction.md — what the agent was asked to do
- {task_path}/verifier/ or {task_path}/tests/ — the checks its work was graded by
- {task_path}/oracle/ or {task_path}/solution/ — a reference solution, when present"""

TASK_SECTION_MISSING = (
    "The task definition is not available for this run. Infer what was "
    "required from the run's own records and test output."
)

OUTPUT_TEMPLATE = """When you are done, write your answer as JSON to {result_path}. The file must contain a single object matching this schema exactly:

{output_schema}

"trial_name" must be exactly "{trial_name}". Every criterion listed in the schema must appear in "checks" with an "outcome" of "pass", "fail", or "not_applicable" and a non-empty "explanation". Write the file and nothing else; do not print the JSON instead of writing it."""

JOB_SUMMARY_TEMPLATE = """Several runs of the same kind were each reviewed independently. Combine their reviews into one short report: recurring failure patterns, systemic issues with the task or environment, and anything that appears in several runs. Three to eight sentences, plain prose, no headings.

Per-run reviews:

{trial_results}
"""


def render_task_section(task_path: str | None) -> str:
    """Render the paragraph pointing the reviewer at the task copy, if any."""

    if task_path is None:
        return TASK_SECTION_MISSING
    return TASK_SECTION_TEMPLATE.format_map(defaultdict(str, task_path=task_path))


def render_review_instruction(
    rubric: Rubric,
    *,
    template: str | None = None,
    trial_path: str = TRIAL_MOUNT,
    task_path: str | None = TASK_MOUNT,
    result_path: str = "/app/review-result.json",
    trial_name: str = "",
    output_schema: dict[str, Any] | None = None,
) -> str:
    """Render the full wrapper-task instruction body."""

    body = (template or REVIEW_TEMPLATE).format_map(
        defaultdict(
            str,
            trial_path=trial_path,
            task_section=render_task_section(task_path),
            criteria_guidance=build_criteria_guidance(rubric),
        )
    )
    output = OUTPUT_TEMPLATE.format_map(
        defaultdict(
            str,
            result_path=result_path,
            trial_name=trial_name,
            output_schema=json.dumps(output_schema or {}, indent=2),
        )
    )
    return f"{body.rstrip()}\n\n{output.strip()}\n"


def render_job_summary_prompt(trial_results: list[str]) -> str:
    """Render the job-level aggregation prompt.

    Uses ``str.replace`` rather than ``format`` so review text containing
    braces (code snippets, JSON) cannot break rendering.
    """

    return JOB_SUMMARY_TEMPLATE.replace("{trial_results}", "\n\n".join(trial_results))
