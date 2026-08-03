"""Rubric-review engine — free functions taking a Rollout (same engine
convention as ``rollout_branch.py`` / ``rollout._user_loop``).

Runs after ``verify()`` while the sandbox is still alive:

1. Discover ``rubric.json`` next to the verifier (native ``verifier/`` or
   legacy ``tests/``). A file is claimed only when it carries
   ``schema_version`` — an llm-judge rubric.json never triggers a review.
2. Upload a **redacted snapshot** of the captured trajectory files into the
   workspace at ``<workspace>/.benchflow-review/trajectory/`` so the reviewer
   can consult what the agent actually did (inside the workspace because some
   harnesses refuse file reads outside their workspace root; post-verify, so
   it cannot influence the execution reward).
3. ``connect_as()`` a reviewer role — any registered agent harness + model —
   in the task sandbox. The reviewer's session events stream to
   ``review/reviewer_trajectory.jsonl``, never into the solver's trajectory
   artifacts, and the solver's session counters are re-synced after every
   reviewer turn so ``disconnect()``'s partial-capture cannot fold reviewer
   events into ``self._trajectory``.
4. Prompt the reviewer (one turn per criterion in ``individual`` mode, one
   turn total in ``batched`` mode), parse the verdicts, retry once with the
   parse error fed back, and mark anything still unparsable **unscored** —
   a reviewer failure is never scored against the model.
5. Aggregate (gates → signed-weight mean → clamp) and merge ``review``,
   ``review_passed``, and ``review/<id>`` into the rewards dict. The primary
   ``reward`` key is never touched. Full verdicts, reviewer identity, and the
   rubric digest land in ``review/review_details.json``.

Review failures never fail the rollout: every error path degrades to a
``review_details.json`` with a status and leaves the rewards dict alone.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow._types import Role
from benchflow.review.config import (
    REVIEW_RUBRIC_FILENAME,
    ReviewCriterion,
    ReviewParams,
    ReviewRubric,
    ReviewRubricError,
    find_review_rubric,
    load_review_rubric,
)
from benchflow.review.prompts import (
    REVIEW_SNAPSHOT_DIRNAME,
    render_retry_prompt,
    render_review_prompt,
)
from benchflow.review.scoring import (
    STATUS_CONFIG_ERROR,
    STATUS_ERROR,
    STATUS_SCORED,
    CriterionVerdict,
    ReviewOutcome,
    aggregate,
    parse_reviewer_message,
)
from benchflow.trajectories._capture import TrajectoryWriter, make_trajectory_sink

if TYPE_CHECKING:  # pragma: no cover - typing only
    from benchflow.rollout import Rollout

logger = logging.getLogger(__name__)

REVIEWER_ROLE_NAME = "rubric-reviewer"
_DEFAULT_REVIEWER_AGENT = "claude-agent-acp"
_MAX_PARSE_RETRIES = 1
_MAX_TRAJECTORY_UPLOAD_BYTES = 50 * 1024 * 1024

_SECRET_KEY_RE = re.compile(
    r"(?i)(api[-_]?key|authorization|secret|password|bearer|token)"
)


def resolve_review_params(config_review: ReviewParams | None) -> ReviewParams:
    return config_review if config_review is not None else ReviewParams()


async def run_review_engine(rollout: Rollout) -> None:
    """Run the rubric review for one rollout. Never raises."""
    params = resolve_review_params(getattr(rollout._config, "review", None))
    if params.enabled is False:
        return

    task = getattr(rollout, "_task", None)
    if task is None:
        return
    verifier_dir = task.paths.tests_dir
    rubric_path = find_review_rubric(verifier_dir)

    if rubric_path is None and params.enabled is None:
        return  # auto mode: no rubric, no review.

    started_at = datetime.now()
    review_dir = rollout._require_rollout_dir() / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    rubric: ReviewRubric | None = None
    reviewer_agent = ""
    reviewer_model: str | None = None
    reviewer_meta: dict[str, Any] = {}
    # Pre-seeded breadcrumb: the finally below always writes details, so even
    # an escape that ``except Exception`` cannot catch (e.g. CancelledError
    # during shutdown) leaves a diagnosable artifact instead of a bare
    # review/ directory.
    outcome = ReviewOutcome(
        status=STATUS_ERROR, error="review aborted before completion"
    )
    try:
        if rubric_path is None:
            # Explicit --review with a present-but-unclaimed rubric.json:
            # loading it surfaces the precise problem (invalid JSON, missing
            # schema_version) instead of a misleading "no rubric" message.
            candidate = verifier_dir / REVIEW_RUBRIC_FILENAME
            if candidate.is_file():
                load_review_rubric(candidate)
            raise ReviewRubricError(
                f"--review was requested but no review rubric.json (with "
                f"schema_version) exists under {verifier_dir}"
            )
        rubric = load_review_rubric(rubric_path)
        reviewer_agent, reviewer_model = _resolve_reviewer(params, rubric)
        outcome, reviewer_meta = await _run_reviewer_session(
            rollout,
            rubric,
            reviewer_agent=reviewer_agent,
            reviewer_model=reviewer_model,
            review_dir=review_dir,
        )
    except ReviewRubricError as e:
        logger.error("Rubric review configuration error: %s", e)
        outcome = ReviewOutcome(status=STATUS_CONFIG_ERROR, error=str(e))
    except Exception as e:
        logger.error("Rubric review failed", exc_info=True)
        outcome = ReviewOutcome(status=STATUS_ERROR, error=str(e))
    finally:
        try:
            updates = outcome.reward_updates(rubric)
            # Merge only into an existing rewards dict: when the verifier
            # errored (rewards is None), inventing a rewards dict here would
            # flip the rollout's completion classification. The review still
            # speaks through review_details.json in that case.
            if updates and rollout._rewards is not None:
                rollout._rewards = {**rollout._rewards, **updates}
            _write_review_details(
                review_dir / "review_details.json",
                outcome=outcome,
                rubric=rubric,
                rubric_path=rubric_path,
                reviewer_agent=reviewer_agent,
                reviewer_model=reviewer_model,
                reviewer_meta=reviewer_meta,
                started_at=started_at,
            )
        except Exception:
            logger.error("Failed to write review_details.json", exc_info=True)

    if outcome.status == STATUS_SCORED:
        logger.info(
            "Rubric review scored %.4f (passed=%s, gates_failed=%s)",
            outcome.review,
            outcome.passed,
            outcome.failed_gates or "none",
        )
    else:
        logger.warning("Rubric review %s: %s", outcome.status, outcome.error)


def _resolve_reviewer(
    params: ReviewParams, rubric: ReviewRubric
) -> tuple[str, str | None]:
    """CLI/config params override rubric defaults, which override registry."""
    agent = params.agent or rubric.reviewer.agent or _DEFAULT_REVIEWER_AGENT
    model = params.model or rubric.reviewer.model
    if model is None:
        from benchflow.evaluation import effective_model

        model = effective_model(agent, None)
    return agent, model


async def _run_reviewer_session(
    rollout: Rollout,
    rubric: ReviewRubric,
    *,
    reviewer_agent: str,
    reviewer_model: str | None,
    review_dir: Path,
) -> tuple[ReviewOutcome, dict[str, Any]]:
    trajectory_files = await _upload_trajectory_snapshot(rollout)

    role = Role(
        name=REVIEWER_ROLE_NAME,
        agent=reviewer_agent,
        model=reviewer_model,
        timeout_sec=int(rubric.reviewer.timeout),
    )
    await rollout.connect_as(role)
    reviewer_events = 0
    reviewer_tools = 0
    try:
        # Reviewer events must not stream into the solver's
        # trajectory/acp_trajectory.jsonl (connect_as wired that sink).
        if rollout._session is not None:
            rollout._session.on_change = make_trajectory_sink(
                TrajectoryWriter(review_dir / "reviewer_trajectory.jsonl"), []
            )

        batches = (
            [[c] for c in rubric.criteria]
            if rubric.reviewer.mode == "individual"
            else [list(rubric.criteria)]
        )
        task_prompt = (getattr(rollout._task, "instruction", "") or "").strip()

        verdicts: list[CriterionVerdict] = []
        for index, batch in enumerate(batches):
            prompt = render_review_prompt(
                batch,
                workspace=rollout._agent_cwd,
                task_prompt=task_prompt,
                trajectory_dir=(
                    _snapshot_dir(rollout) if trajectory_files else None
                ),
                trajectory_files=trajectory_files,
                first_batch=index == 0,
            )
            batch_verdicts, events, tools = await _grade_batch(
                rollout, rubric, batch, prompt
            )
            verdicts.extend(batch_verdicts)
            reviewer_events += events
            reviewer_tools += tools
    finally:
        try:
            await rollout.disconnect()
        except Exception as e:
            logger.warning("Reviewer disconnect failed: %s", e)

    outcome = aggregate(rubric, verdicts)
    meta = {
        "n_events": reviewer_events,
        "n_tool_calls": reviewer_tools,
        "trajectory_snapshot_files": trajectory_files,
    }
    return outcome, meta


async def _grade_batch(
    rollout: Rollout,
    rubric: ReviewRubric,
    batch: list[ReviewCriterion],
    prompt: str,
) -> tuple[list[CriterionVerdict], int, int]:
    """One reviewer turn (plus at most one corrective retry) for ``batch``."""
    events_used = 0
    tools_used = 0
    error: str | None = None
    for attempt in range(_MAX_PARSE_RETRIES + 1):
        message = prompt if attempt == 0 else render_retry_prompt(error or "")
        reply, events, tools = await _reviewer_turn(rollout, rubric, message)
        events_used += events
        tools_used += tools
        verdicts, error = parse_reviewer_message(reply, batch)
        if error is None:
            return verdicts, events_used, tools_used

    unscored = [
        CriterionVerdict(
            criterion_id=criterion.id,
            verdict=None,
            unscored_reason=f"reviewer reply could not be parsed: {error}",
        )
        for criterion in batch
    ]
    return unscored, events_used, tools_used


async def _reviewer_turn(
    rollout: Rollout, rubric: ReviewRubric, prompt: str
) -> tuple[str, int, int]:
    """Send one prompt through the reviewer session; return its final message.

    Deliberately bypasses ``Rollout.execute()``: that method commits session
    events into the solver's trajectory, tool counts, and rollout tree. The
    reviewer talks through the planes primitive, and the session counters are
    re-synced afterwards so ``disconnect()``'s partial-capture sees nothing
    new.
    """
    timeout = rubric.reviewer.timeout
    prev_events = rollout._session_traj_count
    prev_tools = rollout._session_tool_count
    if getattr(rollout, "_is_session_factory", False):
        trajectory, n_tool_calls = (
            await rollout._planes.execute_prompts_session_factory(
                rollout._session,
                [prompt],
                timeout,
                idle_timeout=rollout._config.agent_idle_timeout,
            )
        )
    else:
        trajectory, n_tool_calls = await rollout._planes.execute_prompts(
            rollout._acp_client,
            rollout._session,
            [prompt],
            timeout,
            idle_timeout=rollout._config.agent_idle_timeout,
        )
    new_events = trajectory[prev_events:]
    # Keep the solver's partial-capture pointer in sync: everything this
    # session produced is reviewer traffic and is captured separately.
    rollout._session_traj_count = len(trajectory)
    rollout._session_tool_count = n_tool_calls

    # A single reviewer turn may be committed as several agent_message events
    # (streamed chunks); join them so a verdicts object split across chunks
    # still parses.
    reply = "\n".join(
        str(event["text"])
        for event in new_events
        if event.get("type") == "agent_message" and event.get("text")
    )
    return reply, len(new_events), max(0, n_tool_calls - prev_tools)


async def _upload_trajectory_snapshot(rollout: Rollout) -> list[str]:
    """Upload redacted trajectory copies into the reviewer-visible snapshot dir.

    Returns the uploaded filenames. The snapshot is taken *before* the
    reviewer connects, so it contains only solver traffic. It lands **inside
    the workspace** (``<workspace>/.benchflow-review/trajectory/``) because
    several harnesses (e.g. gemini) refuse file reads outside their workspace
    root; the review runs after ``verify()``, so a workspace addition can no
    longer influence the execution reward.
    """
    rollout_dir = rollout._require_rollout_dir()
    trajectory_dir = rollout_dir / "trajectory"
    sources: list[Path] = []
    if trajectory_dir.is_dir():
        sources = sorted(p for p in trajectory_dir.glob("*.jsonl") if p.is_file())

    staging = rollout_dir / "review" / "trajectory_snapshot"
    staging.mkdir(parents=True, exist_ok=True)

    remote_dir = _snapshot_dir(rollout)
    uploaded: list[str] = []
    await rollout._env.exec(
        f"mkdir -p {shlex.quote(remote_dir)}",
        user="root",
        timeout_sec=30,
    )
    for source in sources:
        try:
            if source.stat().st_size > _MAX_TRAJECTORY_UPLOAD_BYTES:
                logger.warning(
                    "Skipping trajectory upload for %s (>%d bytes)",
                    source.name,
                    _MAX_TRAJECTORY_UPLOAD_BYTES,
                )
                continue
            redacted = staging / source.name
            redacted.write_text(
                _redact_jsonl(source.read_text(encoding="utf-8", errors="replace")),
                encoding="utf-8",
            )
            await rollout._env.upload_file(redacted, f"{remote_dir}/{source.name}")
            uploaded.append(source.name)
        except Exception as e:
            logger.warning("Trajectory upload failed for %s: %s", source.name, e)

    # The in-memory solver trajectory is authoritative even when no live
    # jsonl writer ran (e.g. oracle runs).
    if rollout._trajectory:
        try:
            events_path = staging / "solver_events.jsonl"
            events_path.write_text(
                "\n".join(
                    json.dumps(_redact_value(event), ensure_ascii=False)
                    for event in rollout._trajectory
                )
                + "\n",
                encoding="utf-8",
            )
            await rollout._env.upload_file(
                events_path, f"{remote_dir}/solver_events.jsonl"
            )
            uploaded.append("solver_events.jsonl")
        except Exception as e:
            logger.warning("Solver-event upload failed: %s", e)

    if uploaded:
        # Uploads land root-owned; the reviewer usually runs as sandbox_user.
        await rollout._env.exec(
            f"chmod -R a+rX {shlex.quote(_snapshot_root(rollout))}",
            user="root",
            timeout_sec=30,
        )
    return uploaded


def _snapshot_root(rollout: Rollout) -> str:
    return f"{rollout._agent_cwd.rstrip('/')}/{REVIEW_SNAPSHOT_DIRNAME}"


def _snapshot_dir(rollout: Rollout) -> str:
    return f"{_snapshot_root(rollout)}/trajectory"


def _redact_jsonl(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue  # never ship unparseable (possibly secret-bearing) lines
        lines.append(json.dumps(_redact_value(payload), ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if _SECRET_KEY_RE.search(str(key)) else _redact_value(v))
            for key, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _write_review_details(
    path: Path,
    *,
    outcome: ReviewOutcome,
    rubric: ReviewRubric | None,
    rubric_path: Path | None,
    reviewer_agent: str,
    reviewer_model: str | None,
    reviewer_meta: dict[str, Any],
    started_at: datetime,
) -> None:
    criteria_by_id = {c.id: c for c in (rubric.criteria if rubric else ())}
    details: dict[str, Any] = {
        "status": outcome.status,
        "review": outcome.review,
        "review_passed": outcome.passed,
        "pass_threshold": rubric.pass_threshold if rubric else None,
        "failed_gates": outcome.failed_gates,
        "error": outcome.error,
        "reviewer": {
            "agent": reviewer_agent or None,
            "model": reviewer_model,
            "mode": rubric.reviewer.mode if rubric else None,
            "timeout": rubric.reviewer.timeout if rubric else None,
            **reviewer_meta,
        },
        "rubric": {
            "path": str(rubric_path) if rubric_path else None,
            "sha256": (
                hashlib.sha256(rubric_path.read_bytes()).hexdigest()
                if rubric_path and rubric_path.is_file()
                else None
            ),
            "n_criteria": len(rubric.criteria) if rubric else 0,
        },
        "timing": {
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - started_at).total_seconds(),
        },
        "verdicts": [
            {
                "id": verdict.criterion_id,
                "verdict": verdict.verdict,
                "score": verdict.score,
                "reasoning": verdict.reasoning,
                "evidence": list(verdict.evidence),
                "unscored_reason": verdict.unscored_reason,
                "required": (
                    criteria_by_id[verdict.criterion_id].required
                    if verdict.criterion_id in criteria_by_id
                    else None
                ),
                "weight": (
                    criteria_by_id[verdict.criterion_id].weight
                    if verdict.criterion_id in criteria_by_id
                    else None
                ),
                "tags": (
                    list(criteria_by_id[verdict.criterion_id].tags)
                    if verdict.criterion_id in criteria_by_id
                    else []
                ),
            }
            for verdict in outcome.verdicts
        ],
    }
    path.write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
