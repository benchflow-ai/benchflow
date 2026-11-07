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
from pathlib import Path
from typing import Any

from benchflow._utils.reachable_write import (
    ReachablePath,
    private_folder,
)
from benchflow.sandbox._exec_output import exec_raw

logger = logging.getLogger(__name__)

PRICE_SOURCE = "agent_session_log"
SESSION_DIR = "claude-sessions"  # under the trial's agent/ folder
# Where Claude Code keeps its session logs, under the agent's home.
CLAUDE_SESSION_LOGS = ".claude/projects"
CODEX_SESSION_DIR = "codex-sessions"  # under the trial's agent/ folder
# Where Codex keeps its session rollouts, under its home (CODEX_HOME).
CODEX_SESSION_LOGS = "sessions"
#: Keep the agent's session log even when the proxy saw its calls (a check
#: of the proxy's records against the agent's own).
KEEP_SESSION_LOGS_ENV = "BENCHFLOW_KEEP_AGENT_SESSION_LOGS"
_MAX_BYTES = 64 * 1024 * 1024
_MAX_ENTRIES = 5000
_TIMEOUT_SEC = 120


def _home(cfg: Any) -> str:
    """The home of the user the agent runs as in the sandbox."""
    return f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"


def _runs_claude_code(agent: str) -> bool:
    name = agent.rsplit("/", 1)[-1]
    return name in ("claude-agent-acp", "claude")


def _runs_codex(agent: str) -> bool:
    name = agent.rsplit("/", 1)[-1]
    return name in ("codex-acp", "codex")


def _contributors(rollout: Any) -> set[str]:
    """The primary agent and every native role that made the trial's calls."""
    return {rollout._config.primary_agent or ""} | getattr(
        rollout, "_native_usage_agents", set()
    )


def _secrets(rollout: Any) -> list[str]:
    """Credential values the agent was given, to scrub from its log.

    Only the credential-bearing entries, by the same rule that keeps them out
    of the rollout's ``config.json``. Scrubbing every value instead would take
    the public ones with it -- a model name passed as ``ANTHROPIC_MODEL``, a
    base URL -- and the estimate reads the model of every logged response, so
    a scrubbed model name leaves it unpriceable.
    """
    from benchflow._utils.config_redaction import _should_record_env_entry

    env = getattr(rollout, "_agent_env", None) or {}
    values: list[str] = [
        value
        for name, value in env.items()
        if isinstance(value, str)
        and len(value) >= 8
        and not _should_record_env_entry(str(name), value)
    ]
    values.sort(key=len, reverse=True)  # longest first: no partial replacement
    return values


def _redact(path: Path, secrets: list[str]) -> None:
    """Redact one copied log file: the trajectory patterns, then exact values."""
    from benchflow._utils.reachable_write import destination
    from benchflow.trajectories.types import (
        redact_trajectory_obj,
        redact_trajectory_text,
    )

    held = destination(path)
    text = held.read_text(errors="replace")
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
        held.write_text(clean)


async def _copy_out(
    env: Any, source: str, target: ReachablePath, secrets: list[str]
) -> bool:
    """Copy ``source`` from the sandbox to ``target``, redacted; False when absent."""
    from benchflow.review.evidence import capture_workspace

    probe = await exec_raw(
        env.exec, f"test -d {shlex.quote(source)}", user="root", timeout_sec=30
    )
    if probe.return_code != 0:
        return False
    # target.parent is agent/, /logs/agent in a Docker sandbox: the copy is
    # captured and redacted in private custody, then placed there by one
    # rename through the held folder (reachable_write, R4-1).
    target.parent.mkdir(parents=True, exist_ok=True)
    with private_folder(target.parent) as temp:
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
        target.publish(tree)
    return True


async def price_from_session_log(rollout: Any) -> None:
    """Estimate an unpriced Claude Code rollout's USD from its session log.

    Does nothing when the gateway supplied usage (even unpriced or mixed
    with native usage), any contributing agent is not Claude Code, the
    rollout's sandbox holds another rollout's turns, or no log is found.
    Never fails the rollout.

    A rollout is left out when its turns cannot be told apart from turns
    someone else is already charged for: a fork's in-place child writes into
    the parent's own log, and a rollout whose sandbox came from a snapshot --
    an isolated branch child, or a run continued from a kept checkpoint --
    finds the source rollout's session file already in it, so it would be
    charged for turns the source's own record already carries.
    """
    metrics = getattr(rollout, "_usage_metrics", None) or {}
    cfg = rollout._config
    # Null provider cost means unknown coverage, not permission to replace
    # it with an agent-owned log. In particular, a Claude log cannot price
    # other roles or deduplicate their provider subtotal in mixed usage.
    if metrics.get("usage_source") == "provider_response":
        return
    contributors = _contributors(rollout)
    if not contributors or any(not _runs_claude_code(agent) for agent in contributors):
        return
    if metrics.get("cost_usd") is not None or not _runs_claude_code(
        cfg.primary_agent or ""
    ):
        return
    paths = getattr(rollout, "_rollout_paths", None)
    if rollout._env is None or paths is None:
        return
    if (
        getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_branch_forks", None)
        # Started from a snapshot (an isolated branch child, or a continued
        # checkpoint): the sandbox carries the source rollout's session log.
        or getattr(rollout, "_from_branch_snapshot", False)
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
    from benchflow._utils.reported_models import session_log_models
    from benchflow._utils.session_cost import estimate_claude_code_cost

    # The models Claude Code says answered (P6, G-28): the run asked for an
    # alias, and only the session log names the model behind it.
    rollout._session_reported_models = session_log_models(target)

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


async def keep_session_logs(rollout: Any) -> None:
    """Copy the agent's own session log when the proxy saw none of its calls
    and that agent made every call in the trial.

    Claude Code's (``~/.claude/projects``) goes to ``agent/claude-sessions/``
    and Codex's (``$CODEX_HOME/sessions``, ``~/.codex/sessions`` by default)
    to ``agent/codex-sessions/``, redacted like the price's copy, so a run
    around the proxy, with no OpenTelemetry from the agent, still has a
    record of each model call (``trajectories/model_calls.py``). With
    ``BENCHFLOW_KEEP_AGENT_SESSION_LOGS=1`` the log is kept for proxied runs
    too, and for a trial with a token or turn budget, whose subagents' share
    of it only the agent's own log tells (``trial_budget.subagents``). Never
    fails the rollout.
    """
    import os

    metrics = getattr(rollout, "_usage_metrics", None) or {}
    proxied = metrics.get("usage_source") == "provider_response"
    if (
        proxied
        and os.environ.get(KEEP_SESSION_LOGS_ENV) != "1"
        and getattr(rollout, "_trial_budget", None) is None
    ):
        return
    cfg = rollout._config
    paths = getattr(rollout, "_rollout_paths", None)
    if rollout._env is None or paths is None:
        return
    if (
        getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_branch_forks", None)
        or getattr(rollout, "_from_branch_snapshot", False)
    ):
        # The sandbox holds another rollout's turns (see price_from_session_log).
        return
    # A log is the trial's record only when its agent made every call: with
    # another agent's turns in the trial, Claude's log would give its calls,
    # and its cost, as the trial's (the price above refuses the same).
    agents = _contributors(rollout)
    if all(_runs_claude_code(agent) for agent in agents):
        source, target = (
            f"{_home(cfg)}/{CLAUDE_SESSION_LOGS}",
            paths.agent_dir / SESSION_DIR,
        )
    elif all(_runs_codex(agent) for agent in agents):
        env = getattr(rollout, "_agent_env", None) or {}
        home = env.get("CODEX_HOME") or f"{_home(cfg)}/.codex"
        source, target = (
            f"{home}/{CODEX_SESSION_LOGS}",
            paths.agent_dir / CODEX_SESSION_DIR,
        )
    else:
        return
    if target.exists():
        return
    try:
        await asyncio.wait_for(
            _copy_out(rollout._env, source, target, _secrets(rollout)),
            timeout=_TIMEOUT_SEC + 60,
        )
    except Exception as exc:  # the sandbox may be gone; the calls stay unrecorded
        logger.info("Could not copy the agent's session log: %s", exc)


def compactions_in(lines: list[str]) -> dict[str, Any]:
    """The compactions Claude Code's session log records: each line it
    writes at a compaction (``"subtype": "compact_boundary"``), with what
    triggered it and the context's tokens before it (``compactMetadata``)."""
    pre_tokens: list[int] = []
    count = auto = manual = subagent = 0
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("subtype") != "compact_boundary":
            continue
        if row.get("isSidechain") is True:
            subagent += 1
            continue
        count += 1
        meta = row.get("compactMetadata")
        meta = meta if isinstance(meta, dict) else {}
        trigger = meta.get("trigger")
        auto += trigger == "auto"
        manual += trigger == "manual"
        tokens = meta.get("preTokens")
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            pre_tokens.append(tokens)
    return {
        "count": count,
        "auto": auto,
        "manual": manual,
        "subagent": subagent,
        "pre_tokens": pre_tokens,
        "source": "claude-code-session-log",
    }


async def count_compactions(rollout: Any) -> None:
    """Count Claude Code's compactions in this trial (``_compactions``, for
    result.json): from the log copied into ``agent/claude-sessions/`` when
    there is one, else the lines that mark them, read in the sandbox while
    it is up. Left None when another agent made calls, the sandbox holds
    another rollout's turns, or the log cannot be read. Never fails the
    rollout."""
    agents = _contributors(rollout)
    paths = getattr(rollout, "_rollout_paths", None)
    if (
        not agents
        or not all(_runs_claude_code(agent) for agent in agents)
        or rollout._env is None
        or paths is None
        or getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_branch_forks", None)
        or getattr(rollout, "_from_branch_snapshot", False)
    ):
        return
    copied = paths.agent_dir / SESSION_DIR
    lines: list[str] = []
    try:
        if copied.is_dir():
            for path in sorted(copied.rglob("*.jsonl")):
                if path.is_file() and not path.is_symlink():
                    lines += [
                        line
                        for line in path.read_text(errors="replace").splitlines()
                        if "compact_boundary" in line
                    ]
        else:
            source = f"{_home(rollout._config)}/{CLAUDE_SESSION_LOGS}"
            found = await exec_raw(
                rollout._env.exec,
                f"test -d {shlex.quote(source)} || exit 3; "
                f"grep -rhs --include='*.jsonl' compact_boundary {shlex.quote(source)}"
                " || true",
                user="root",
                timeout_sec=30,
            )
            if found.return_code == 3:
                return
            lines = str(found.stdout or "").splitlines()
    except Exception as exc:  # the sandbox may be gone; the count stays unknown
        logger.info("Could not count Claude Code's compactions: %s", exc)
        return
    rollout._compactions = compactions_in(lines)
