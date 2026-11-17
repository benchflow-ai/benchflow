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
from benchflow.usage_tracking import USAGE_SOURCE_AGENT_NATIVE_ACP

logger = logging.getLogger(__name__)

PRICE_SOURCE = "agent_session_log"
SESSION_DIR = "claude-sessions"  # under the trial's agent/ folder
# Where Claude Code keeps its session logs, under the agent's home.
CLAUDE_SESSION_LOGS = ".claude/projects"
CODEX_SESSION_DIR = "codex-sessions"  # under the trial's agent/ folder
# Where Codex keeps its session rollouts, under its home (CODEX_HOME).
CODEX_SESSION_LOGS = "sessions"
#: ``usage_details.token_count.source`` of a Codex trial counted from
#: Codex's own log (``count_codex_tokens``).
CODEX_TOKENS_SOURCE = "codex-session-log"
_TOKEN_FIELDS = (
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_creation_tokens",
    "n_cache_write_5m_tokens",
    "n_cache_write_1h_tokens",
    "total_tokens",
)
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


async def copy_codex_session_log(rollout: Any) -> None:
    """Copy Codex's session log (``$CODEX_HOME/sessions``,
    ``~/.codex/sessions`` by default) into the trial's
    ``agent/codex-sessions/``, redacted like Claude Code's, when the proxy
    saw none of the trial's calls and Codex made all of them: the
    one record of each model call a ChatGPT-login trial has. Never fails the
    rollout."""
    metrics = getattr(rollout, "_usage_metrics", None) or {}
    if metrics.get("usage_source") == "provider_response":
        return
    agents = _contributors(rollout)
    if not agents or not all(_runs_codex(agent) for agent in agents):
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
    env = getattr(rollout, "_agent_env", None) or {}
    home = env.get("CODEX_HOME") or f"{_home(cfg)}/.codex"
    target = paths.agent_dir / CODEX_SESSION_DIR
    if target.exists():
        return
    try:
        await asyncio.wait_for(
            _copy_out(
                rollout._env, f"{home}/{CODEX_SESSION_LOGS}", target, _secrets(rollout)
            ),
            timeout=_TIMEOUT_SEC + 60,
        )
    except Exception as exc:  # the sandbox may be gone; the calls stay unrecorded
        logger.info("Could not copy Codex's session log: %s", exc)


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
    contributors = {cfg.primary_agent or ""} | getattr(
        rollout, "_native_usage_agents", set()
    )
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
    from benchflow.review.evidence import EvidenceLimitError

    try:
        copied = await asyncio.wait_for(
            _copy_out(rollout._env, source, target, _secrets(rollout)),
            timeout=_TIMEOUT_SEC + 60,
        )
    except EvidenceLimitError as exc:
        # A long run's log can pass the copy's limits (64 MiB, 5000 entries).
        # 0.9.3 keeps no copy made while the turns ran: the record says why
        # the log was not read (benchflow-ai/benchflow#1240).
        _record_copy_failure(rollout, f"over the copy limit: {exc}")
        logger.info("Claude Code's session log is over the copy limit: %s", exc)
        return
    except Exception as exc:  # the sandbox may be gone; the cost stays unknown
        _record_copy_failure(rollout, f"{type(exc).__name__}: {exc}")
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
        **{
            key: estimate[key]
            for key in ("claude_code_usd", "repriced_models", "unpriced_models")
            if key in estimate
        },
    }
    if estimate["usd"] is None:
        # No list price for a model Claude Code does not price: the cost is
        # unknown, never Claude Code's default rate (benchflow#1240).
        rollout._usage_metrics = {**metrics, "usage_details": details}
        logger.info(
            "No list price for %s, which Claude Code does not price: the "
            "trial's cost stays unknown (Claude Code's own figure: $%.4f)",
            ", ".join(estimate.get("unpriced_models") or []),
            estimate.get("claude_code_usd") or 0.0,
        )
        return
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


def _record_copy_failure(rollout: Any, failed: str) -> None:
    """Record why teardown's copy of Claude Code's session log failed
    (``usage_details.session_log_copy``): the trial's count and cost are
    not the log's. 0.9.3 keeps no copy made while the turns ran, so
    ``used`` is None."""
    metrics = getattr(rollout, "_usage_metrics", None) or {}
    details = dict(metrics.get("usage_details") or {})
    details["session_log_copy"] = {"full_copy": f"failed: {failed}", "used": None}
    rollout._usage_metrics = {**metrics, "usage_details": details}


async def count_codex_tokens(rollout: Any) -> None:
    """Count a Codex trial's tokens from Codex's own session log.

    codex-acp answers each prompt with the usage of the turn's last model
    call (``PromptResponse.usage`` is Codex's ``tokenUsage.last``), and its
    ``usage_update`` notifications carry only how full the context is
    (``used``, ``size``). The running total Codex keeps (``tokenUsage.total``)
    never crosses ACP, so a trial of 75 tool calls recorded one call's
    tokens (ported from 0.10, #105). Codex logs every response's
    usage in its session log, which :func:`copy_codex_session_log` has
    copied into ``agent/codex-sessions/``. Their
    sum replaces the ACP figures, ``usage_source`` is ``session_log``, and ``usage_details.token_count`` names the log and keeps what
    ACP reported. The log holds the generating responses only: its ``scope``
    says so, and ``usage_details.warmup`` records Codex's generate=false
    warm-up apart, its tokens and cost unknown.

    The native harness's turns report Codex's running total, but a turn cut
    off reports nothing, so its trial is counted from the log too, always:
    one count on both harnesses, the session log's. The log's sum
    replaces the end-of-turn figure; it is never added to it.

    Does nothing when the proxy metered the trial, another agent made calls,
    the sandbox holds another rollout's turns, or there is no log to read.
    The reported figure then stays, without ``token_count``. Never fails the
    rollout.
    """
    from benchflow._utils.result_metadata import (
        CODEX_GENERATING_SCOPE,
        codex_warmup,
    )
    from benchflow.rollout._cost import codex_session_slices, codex_session_tokens
    from benchflow.usage_tracking import USAGE_SESSION_LOG, is_token_usage_available

    metrics = getattr(rollout, "_usage_metrics", None) or {}
    if metrics.get("usage_source") == "provider_response" or not all(
        _runs_codex(agent) for agent in _contributors(rollout)
    ):
        return
    reported = is_token_usage_available(metrics)
    paths = getattr(rollout, "_rollout_paths", None)
    if (
        paths is None
        or getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_branch_forks", None)
        or getattr(rollout, "_from_branch_snapshot", False)
    ):
        # The sandbox holds another rollout's turns (see price_from_session_log).
        return
    try:
        # Read as the model-call record reads it: from the trial's folder,
        # never through a link (agent/ is a folder the sandbox shares).
        totals = await asyncio.to_thread(codex_session_tokens, Path(paths.rollout_dir))
        slices = await asyncio.to_thread(codex_session_slices, Path(paths.rollout_dir))
    except Exception as exc:  # the trial keeps what ACP reported
        logger.info("Could not read Codex's session log: %s", exc)
        return
    if totals is None:
        logger.info(
            "No Codex session log to count: the trial's tokens are the ones "
            "Codex reported at the end of its turns"
        )
        return
    details = dict(metrics.get("usage_details") or {})
    acp_reported: dict[str, Any] | None = None
    if reported:
        acp_reported = {field: metrics.get(field) for field in _TOKEN_FIELDS}
        acp_reported["thought_tokens"] = details.get("thought_tokens")
    details["thought_tokens"] = totals["thought_tokens"]
    details["token_count"] = {
        "source": CODEX_TOKENS_SOURCE,
        "path": f"agent/{CODEX_SESSION_DIR}",
        "calls": totals["calls"],
        "acp_reported": acp_reported,
        "scope": CODEX_GENERATING_SCOPE,
    }
    # The log never sees Codex's generate=false warm-up: recorded
    # apart, unknown, and out of the counts above.
    details["warmup"] = codex_warmup()
    counted = {
        **metrics,
        **{field: totals[field] for field in _TOKEN_FIELDS},
        "usage_source": USAGE_SESSION_LOG,
        "usage_source_legacy": USAGE_SOURCE_AGENT_NATIVE_ACP,
        "usage_details": details,
    }
    if metrics.get("price_source") == "list_price":
        counted.update(cost_usd=None, price_source=None, cost_source=None)
    rollout._usage_metrics = counted
    rollout._native_usage_metrics = dict(counted)
    rollout._usage_slices = slices
    rollout._usage_salvaged = True
    logger.info(
        "Tokens counted from Codex's session log: %d calls, %d tokens "
        "(Codex reported %s at the end of its turns)",
        totals["calls"],
        totals["total_tokens"],
        acp_reported["total_tokens"] if acp_reported else "nothing",
    )
