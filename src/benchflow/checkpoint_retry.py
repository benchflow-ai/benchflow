"""Retry a failed or timed-out trial from its last automatic checkpoint.

``bench eval run --checkpoints … --retry-from-checkpoint on-failure|on-timeout
[--retry-prompt TEXT]``. After a trial finishes, if it failed (scored, not a
pass) or its agent timed out, and it kept at least one checkpoint, one retry
child is forked from the *last kept* checkpoint instead of rerunning the whole
trial: an isolated child (``rollout_branch._run_isolated_children``) with its
own sandbox created from that snapshot, sent the prompts after the checkpoint
(or ``--retry-prompt``) in a fresh agent session, and verified by the task's
own verifier.

Reward-hacking guards: the retry child keeps the snapshot's *original*
pre-agent verifier baseline (captured before any agent ran), so a build file
the failed attempt tampered with before the checkpoint is still restored at
verification; it is verified exactly like the trial. Rewards are never
merged: result.json keeps the trial's own ``rewards``; the retry's reward is
reported next to it in a ``retry`` block (and in ``tree.json`` as a ``retry``
fork), and the job summary counts both.

Agent errors and infrastructure failures are not retried here; the
evaluation's existing retry loop reruns those from scratch.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from benchflow.review.persistence import write_json_atomic

logger = logging.getLogger(__name__)

_HELP = "--retry-from-checkpoint takes on-failure, on-timeout, or both comma-separated"
TIMEOUT_CATEGORIES = frozenset({"timeout", "idle_timeout"})


@dataclass(frozen=True)
class RetryPolicy:
    on_failure: bool = False
    on_timeout: bool = False
    prompt: str | None = None
    # Resume the failed trial's conversation (the checkpoint's session id).
    resume_session: bool = False


def parse_retry_policy(
    spec: str, *, prompt: str | None, resume_session: bool = False
) -> RetryPolicy:
    parts = {part.strip() for part in spec.split(",") if part.strip()}
    if not parts or parts - {"on-failure", "on-timeout"}:
        raise ValueError(f"{_HELP}; got {spec!r}")
    if prompt is not None and not prompt.strip():
        raise ValueError("--retry-prompt is empty")
    return RetryPolicy(
        on_failure="on-failure" in parts,
        on_timeout="on-timeout" in parts,
        prompt=prompt,
        resume_session=resume_session,
    )


INSTRUCTION_TOKEN = "@instruction"
FEEDBACK_TOKEN = "@verifier_feedback"
_FEEDBACK_TAIL = 3000


def expand_retry_prompt(
    prompt: str, *, instruction: str, rollout_dir: Any, reward: Any
) -> str:
    """Expand ``@instruction`` (the task instruction) and ``@verifier_feedback``
    (the failed trial's reward and the last ~3,000 characters of its
    verifier output) in a ``--retry-prompt``."""
    if INSTRUCTION_TOKEN in prompt:
        prompt = prompt.replace(INSTRUCTION_TOKEN, instruction)
    if FEEDBACK_TOKEN in prompt:
        from pathlib import Path

        text = ""
        try:
            path = Path(rollout_dir) / "verifier" / "test-stdout.txt"
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - _FEEDBACK_TAIL))
                text = handle.read().decode("utf-8", errors="replace").strip()
        except (OSError, TypeError):
            text = ""
        feedback = f"The verifier gave reward {reward}." + (
            f" The end of its output:\n{text}"
            if text
            else " (no verifier output was recorded)"
        )
        prompt = prompt.replace(FEEDBACK_TOKEN, feedback)
    return prompt


def retry_reason(result: Any, policy: RetryPolicy) -> str | None:
    """``timeout`` / ``failure`` when the policy asks to retry this result."""
    if result.error_category in TIMEOUT_CATEGORIES:
        return "timeout" if policy.on_timeout else None
    if result.error or result.verifier_error:
        return None
    if result.score_outcome == "failed":
        return "failure" if policy.on_failure else None
    return None


def _publish(rollout: Any, result: Any, block: dict[str, Any]) -> None:
    """Add the retry block (and the updated branches summary) to result.json."""
    from benchflow.branch_lineage import branch_summary

    result.retry = block
    path = rollout._rollout_dir / "result.json"
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    document["retry"] = block
    branches = branch_summary(getattr(rollout, "_branch_forks", []) or [])
    if branches is not None:
        document["branches"] = branches
    write_json_atomic(path, document)


async def run_checkpoint_retry(rollout: Any, result: Any, policy: RetryPolicy) -> None:
    """Fork one retry child from the last kept checkpoint when the policy
    applies; record it without touching the trial's reward."""
    from benchflow.branch_lineage import ForkRecord, fork_cost
    from benchflow.checkpoints import load_checkpoints
    from benchflow.rollout_branch import _run_isolated_children
    from benchflow.sandbox.protocol import SandboxImage

    reason = retry_reason(result, policy)
    run_dir = getattr(rollout, "_rollout_dir", None)
    if (
        reason is None
        or run_dir is None
        or rollout._config.primary_agent in ("oracle", "nop")
    ):
        return
    rows = [
        row
        for row in load_checkpoints(run_dir)
        if row.get("status") == "kept" and row.get("ref")
    ]
    if not rows:
        _publish(rollout, result, {"status": "no_checkpoint", "reason": reason})
        return
    row = rows[-1]
    parent = next(
        (node for node in rollout._tree.nodes() if node.id == row.get("node_id")),
        rollout._tree.root,
    )
    resolved = [str(p) for p in rollout._resolved_prompts]
    after = int(row.get("after_prompt") or 0)
    task = getattr(rollout, "_task", None)
    instruction = getattr(task, "instruction", None)
    if not isinstance(instruction, str) or not instruction:
        instruction = resolved[0] if resolved else ""
    prompts = (
        [
            expand_retry_prompt(
                policy.prompt,
                instruction=instruction,
                rollout_dir=run_dir,
                reward=result.reward,
            )
        ]
        if policy.prompt
        else (resolved[after:] or resolved)
    )
    request = (
        f"retry prompt ({len(policy.prompt)} characters, sha256:"
        f"{hashlib.sha256(policy.prompt.encode()).hexdigest()[:12]})"
        if policy.prompt
        else f"prompts after the checkpoint ({len(prompts)})"
    )
    resume_id = row.get("agent_session_id") if policy.resume_session else None
    if policy.resume_session and not resume_id:
        logger.warning(
            "Checkpoint %s recorded no agent session; the retry starts a fresh one",
            row["id"],
        )
    fork_id = uuid.uuid4().hex
    event = ForkRecord(
        rollout,
        run_dir,
        fork_id,
        parent,
        frozenset({"sandbox"}),
        1,
        ["retry"],
        [request],
    )
    event.record.update(
        kind="retry",
        reason=reason,
        checkpoint=row["id"],
        children_mode={"isolated": True, "concurrency": 1, "prewarm": 1},
        # The trial's sandbox is gone; nothing to restore.
        parent_restore="not_needed",
    )
    event.record["snapshot"].update(
        captured_layers=["sandbox"],
        sandbox={"provider": row["provider"], "ref": row["ref"], "digest": None},
        # Governed by the trial's --checkpoint-keep, not by this fork.
        retention="kept",
        agent_session="resumed" if resume_id else "fresh",
    )
    if resume_id:
        event.record["snapshot"]["excluded"].remove("agent_session")
    event.persist()
    fork_dir = run_dir / "branches" / fork_id
    fork_dir.mkdir(parents=True, exist_ok=False)

    work: dict[str, int | None] = {"tool_calls": None}

    async def run_retry(node: Any, *, child: Any) -> float | None:
        sub = child.rollout
        rewards = None
        try:
            await sub.connect()
            executed = await sub.execute(prompts, node=node)
            if (
                isinstance(executed, tuple)
                and len(executed) == 2
                and isinstance(executed[1], int)
            ):
                work["tool_calls"] = executed[1]
            rewards = await sub.verify()
        finally:
            await sub.disconnect()
        return (rewards or {}).get("reward")

    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        await _run_isolated_children(
            rollout,
            parent,
            SandboxImage(provider=row["provider"], ref=row["ref"]),
            event,
            fork_dir,
            run_retry,
            labels=["retry"],
            concurrency=1,
            resume_session_id=resume_id,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Retry from checkpoint %s failed: %s", row["id"], exc)
    [child] = event.record["children"]
    event.record["cost"] = fork_cost(
        event.record, wall_seconds=loop.time() - started, isolated=True
    )
    # The trial's sandbox did not exist during the retry.
    event.record["cost"].update(
        parent_sandbox_seconds=0.0,
        sandbox_seconds=event.record["cost"]["children_sandbox_seconds"],
    )
    event.record["status"] = "completed" if child["status"] == "scored" else "failed"
    event.record["value"] = child["reward"]
    event.persist()
    _publish(
        rollout,
        result,
        {
            "status": event.record["status"],
            "reason": reason,
            "checkpoint": row["id"],
            "fork_id": fork_id,
            "reward": child["reward"],
            "original_reward": result.reward,
            "path": child["artifacts"]["path"],
            # A retry that made no tool calls did no work (for example a
            # fresh session sent only "try again").
            "tool_calls": work["tool_calls"],
            "no_work": work["tool_calls"] == 0,
        },
    )
    if work["tool_calls"] == 0:
        logger.warning(
            "Retry from %s made no tool calls (reward %s): it probably did no "
            "work. A fresh session knows only the retry prompt; include "
            "@instruction (and @verifier_feedback) in --retry-prompt, or use "
            "--retry-resume-session.",
            row["id"],
            child["reward"],
        )


def retry_summary(results: Any) -> dict[str, int] | None:
    """Job-summary counts of checkpoint retries, reported next to the
    original score (which they never change); None when none applied."""
    # Result objects, or result.json documents (the job summary's input).
    blocks = [
        result.get("retry")
        if isinstance(result, dict)
        else getattr(result, "retry", None)
        for result in results
    ]
    blocks = [block for block in blocks if isinstance(block, dict)]
    if not blocks:
        return None
    ran = [b for b in blocks if b.get("status") in {"completed", "failed"}]
    return {
        "attempted": len(ran),
        "passed": sum(
            1
            for b in ran
            if isinstance(b.get("reward"), int | float) and b["reward"] >= 1.0
        ),
        "no_checkpoint": sum(1 for b in blocks if b.get("status") == "no_checkpoint"),
        "failed_to_run": sum(1 for b in ran if b.get("status") == "failed"),
        "no_work": sum(1 for b in ran if b.get("no_work") is True),
    }
