"""Price a Claude Code rollout from its own session log when the gateway could not.

A rollout's ``cost_usd`` comes from BenchFlow's model gateway, which prices
the calls it proxies. A Claude subscription (``CLAUDE_CODE_OAUTH_TOKEN``), or
a run with usage tracking off, sends Claude Code's calls around the gateway:
its tokens still arrive over ACP, but they have no price. At cleanup, before
the sandbox stops, such a rollout's Claude Code session log (the sandbox
user's ``~/.claude/projects``) is copied into the trial's
``agent/claude-sessions/`` (credentials redacted) and priced by
:func:`benchflow._utils.session_cost.estimate_claude_code_cost`. The result
records it as an estimate: ``cost_usd`` holds the figure, ``price_source`` is
``agent_session_log`` and ``usage_details.cost_estimate`` says how it was
computed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PRICE_SOURCE = "agent_session_log"
SESSION_DIR = "claude-sessions"  # under the trial's agent/ folder
# Where Claude Code keeps its session logs, under the agent's home.
CLAUDE_SESSION_LOGS = ".claude/projects"
_MAX_BYTES = 64 * 1024 * 1024
_MAX_ENTRIES = 5000
_TIMEOUT_SEC = 120


def _home(cfg: Any) -> str:
    """The home of the user the agent runs as in the sandbox."""
    return f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"


def _runs_claude_code(agent: str) -> bool:
    name = agent.rsplit("/", 1)[-1]
    return name in ("claude-agent-acp", "claude")


def _secrets(rollout: Any) -> list[str]:
    """Credential values the agent was given, to scrub from its log."""
    env = getattr(rollout, "_agent_env", None) or {}
    values: list[str] = [v for v in env.values() if isinstance(v, str) and len(v) >= 8]
    values.sort(key=len, reverse=True)  # longest first: no partial replacement
    return values


def _redact(path: Path, secrets: list[str]) -> None:
    """Redact one copied log file: the trajectory patterns, then exact values."""
    from benchflow.trajectories.types import (
        redact_trajectory_obj,
        redact_trajectory_text,
    )

    text = path.read_text(errors="replace")
    if path.suffix == ".jsonl":
        lines = []
        for line in text.splitlines():
            try:
                lines.append(json.dumps(redact_trajectory_obj(json.loads(line))))
            except ValueError:
                lines.append(redact_trajectory_text(line))
        clean = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    else:
        clean = redact_trajectory_text(text)
    for secret in secrets:
        clean = clean.replace(secret, "[redacted]")
    if clean != text:
        path.write_text(clean)


async def _copy_out(env: Any, source: str, target: Path, secrets: list[str]) -> bool:
    """Copy ``source`` from the sandbox to ``target``, redacted; False when absent."""
    from benchflow.review.evidence import capture_workspace

    probe = await env.exec(
        f"test -d {shlex.quote(source)}", user="root", timeout_sec=30
    )
    if probe.return_code != 0:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".session-log-", dir=target.parent) as temp:
        bundle = Path(temp) / "bundle"
        await capture_workspace(
            env,
            source,
            bundle,
            max_bytes=_MAX_BYTES,
            max_entries=_MAX_ENTRIES,
            timeout_sec=_TIMEOUT_SEC,
        )
        tree = bundle / "workspace"
        for path in sorted(tree.rglob("*")):
            if path.is_file() and not path.is_symlink():
                _redact(path, secrets)
        if target.exists():
            shutil.rmtree(target)
        tree.rename(target)
    return True


async def price_from_session_log(rollout: Any) -> None:
    """Estimate an unpriced Claude Code rollout's USD from its session log.

    Does nothing when the gateway priced the rollout, the agent is not
    Claude Code, the rollout branched (its children's sessions share the
    log), or no log is found. Never fails the rollout.
    """
    metrics = getattr(rollout, "_usage_metrics", None) or {}
    cfg = rollout._config
    if metrics.get("cost_usd") is not None or not _runs_claude_code(
        cfg.primary_agent or ""
    ):
        return
    paths = getattr(rollout, "_rollout_paths", None)
    if rollout._env is None or paths is None:
        return
    if getattr(rollout, "_branch_child_active", False) or getattr(
        rollout, "_branch_forks", None
    ):
        return
    target = paths.agent_dir / SESSION_DIR
    source = f"{_home(cfg)}/{CLAUDE_SESSION_LOGS}"
    try:
        copied = await asyncio.wait_for(
            _copy_out(rollout._env, source, target, _secrets(rollout)),
            timeout=_TIMEOUT_SEC + 60,
        )
    except Exception as exc:  # the sandbox may be gone; the cost stays unknown
        logger.info("Could not copy Claude Code's session log: %s", exc)
        return
    if not copied:
        return
    from benchflow._utils.session_cost import estimate_claude_code_cost

    estimate = estimate_claude_code_cost(target)
    if estimate is None:
        return
    details = dict(metrics.get("usage_details") or {})
    details["cost_estimate"] = {
        "source": estimate["source"],
        "method": estimate["method"],
        "path": f"agent/{SESSION_DIR}",
        "sessions": estimate["sessions"],
        "responses": estimate["responses"],
        "models": estimate["models"],
        "context_1m": estimate["context_1m"],
    }
    rollout._usage_metrics = {
        **metrics,
        "cost_usd": estimate["usd"],
        "price_source": PRICE_SOURCE,
        "usage_details": details,
    }
    logger.info(
        "Cost estimated from Claude Code's session log: $%.4f (%s)",
        estimate["usd"],
        estimate["method"],
    )
