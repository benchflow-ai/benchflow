"""Orchestrate ``benchflow continue`` — boot, replay, continue live, stitch.

Reuses benchflow's normal :class:`~benchflow.rollout.Rollout` machinery (so the
new run produces a standard HF-compatible folder) but injects a record-replay
proxy in front of OpenHands:

- Host proxy mode uses ``usage_tracking="off"`` so ``ensure_litellm_runtime`` is
  a no-op and the
  ``agent_env`` we pass through (``LLM_BASE_URL`` → the replay proxy) is left
  untouched (``providers/litellm_runtime.py``).
- Sandbox proxy mode starts the provider LiteLLM proxy inside the sandbox and
  keeps the replay proxy on sandbox loopback, so remote environments such as
  Daytona do not need host-loopback connectivity.
- The original first prompt (task instruction) starts OpenHands; the proxy then
  drives the exact replay and, past the cut-point, the live continuation.

The async :func:`continue_run` is the integration entrypoint (it needs Docker +
the openhands install); the helpers it composes (config assembly, live
forwarder request-building, trajectory stitching) are pure and unit-tested.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from benchflow._utils.text import describe_exception
from benchflow.continue_run.replay_proxy import (
    REQUEST_DIGEST_BASIS,
    ReplayCutPointError,
    ReplayProxy,
    ReplayRouter,
    comparable_request_digest,
    validate_max_exchanges,
)
from benchflow.continue_run.run_folder import RunFolder, RunFolderError, load_run_folder
from benchflow.continue_run.sandbox_proxy import (
    SandboxReplayProxy,
    sandbox_replay_base_url,
)
from benchflow.contracts import AgentProtocolError, SandboxStartupFailure
from benchflow.sandbox.providers import SANDBOX_MODEL_PROXY_PROVIDERS
from benchflow.scenes import compile_scenes_to_steps
from benchflow.trajectories.types import LLMExchange, redact_trajectory_text

logger = logging.getLogger(__name__)

# agent_env values handed to OpenHands so it talks to the replay proxy instead
# of a real provider. The model name is irrelevant (the proxy serves by index),
# but litellm needs an ``openai/`` prefix to treat LLM_BASE_URL as an
# OpenAI-compatible endpoint.
_REPLAY_API_KEY = "sk-benchflow-replay"
_REPLAY_MODEL = "openai/replay"
# Providers whose replay and model proxies must run inside the sandbox. This is
# the same placement contract used by the normal LiteLLM runtime.
_SANDBOX_LOCAL_REPLAY_ENVIRONMENTS = SANDBOX_MODEL_PROXY_PROVIDERS
_PROXY_MODES = frozenset({"auto", "host", "sandbox"})


@dataclass
class ContinueResult:
    """Outcome of a ``benchflow continue`` run."""

    rollout_dir: Path
    rewards: dict[str, Any] | None
    error: str | None
    n_recorded: int
    n_live: int
    divergences: int


def select_proxy_mode(requested: str, environment: str) -> str:
    """Resolve the replay proxy placement for a continuation run."""
    if requested not in _PROXY_MODES:
        raise RunFolderError(
            f"invalid proxy mode {requested!r}; expected one of "
            f"{', '.join(sorted(_PROXY_MODES))}."
        )
    if requested != "auto":
        return requested
    return "sandbox" if environment in _SANDBOX_LOCAL_REPLAY_ENVIRONMENTS else "host"


def continued_rollout_name(run: RunFolder) -> str:
    """Derive a stable, collision-resistant rollout name from the source run."""
    source = run.path.name or run.task_name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", source).strip("-")
    if not cleaned:
        cleaned = run.task_name or "run"
    return f"{cleaned}__continued"


def stage_tags_from_run(run: RunFolder, cut_stage: str) -> dict[str, int]:
    """The recorded stage→exchange-index tags backing a ``--cut-stage`` request.

    The run-folder half of the RFC §3.5 stage-tagged cut: stage captures
    record their completed-exchange index into the run's
    ``stage_snapshots.json`` (see ``benchflow.branch_policy.capture_stage``),
    and this resolves that registry for ``cut_stage``, failing closed with a
    typed :class:`ReplayCutPointError` that tells the caller what *was*
    recorded:

    * no registry at all — the run never captured a stage;
    * ``cut_stage`` absent — names the stages the run did record;
    * recorded without an index (``exchanges_completed: null``) — the usage
      gateway could not count at capture time, so the stage cannot name a cut;
    * recorded at index 0 — the stage closed before the first exchange, so
      there is no recorded prefix to replay.

    An explicit SDK ``stage_tags`` mapping bypasses this entirely (it stays
    the override seam); the CLI resolves through here.
    """
    registry = run.stage_registry
    if not registry:
        raise ReplayCutPointError(
            f"--cut-stage {cut_stage!r}: the run folder recorded no stage "
            "snapshots (no stage_snapshots.json) — re-run the task with stage "
            "capture (RolloutConfig.snapshot_stages / Rollout.mark_stage()) "
            "or cut by number with --max-exchanges."
        )
    if cut_stage not in registry:
        raise ReplayCutPointError(
            f"unknown cut stage {cut_stage!r}; this run recorded: "
            f"{', '.join(sorted(registry))}."
        )
    tags = run.stage_exchange_tags
    if cut_stage not in tags:
        raise ReplayCutPointError(
            f"stage {cut_stage!r} was recorded without an exchange index "
            "(exchanges_completed is null — the run's usage gateway could not "
            "count exchanges at capture time), so it cannot name a cut; use "
            "--max-exchanges instead."
        )
    if tags[cut_stage] == 0:
        raise ReplayCutPointError(
            f"stage {cut_stage!r} closed before the first LLM exchange "
            "(exchanges_completed is 0) — there is no recorded prefix to "
            "replay; run the task fresh instead of continuing it."
        )
    return tags


def resolve_cut_point(
    max_exchanges: int | None,
    *,
    stage_tags: dict[str, int] | None = None,
    cut_stage: str | None = None,
) -> tuple[int | None, str | None]:
    """Resolve an optional stage-named cut into a ``max_exchanges`` value.

    ``stage_tags[stage]`` is the 1-based count of exchanges that had completed
    when the stage closed — a cut at that stage replays exactly that many
    exchanges (rollout-branching RFC §3.5). The mapping comes from the run
    folder's recorded registry (:func:`stage_tags_from_run`) unless the SDK
    caller supplies its own override dict. With ``cut_stage`` given, the cut
    resolves to ``stage_tags[cut_stage]``; every misuse — combining
    ``cut_stage`` with ``max_exchanges``, a missing ``stage_tags`` mapping, an
    unknown stage, or a tag below 1 — fails closed with
    :class:`ReplayCutPointError`. Returns ``(max_exchanges, branch_stage)``.
    """
    if cut_stage is None:
        return max_exchanges, None
    if max_exchanges is not None:
        raise ReplayCutPointError(
            "pass either max_exchanges or cut_stage, not both — a stage-named "
            "cut resolves max_exchanges from stage_tags."
        )
    if not stage_tags:
        raise ReplayCutPointError(
            f"cut_stage={cut_stage!r} given but no stage_tags were provided; a "
            "stage-named cut needs the stage-name → completed-exchange-count "
            "mapping."
        )
    if cut_stage not in stage_tags:
        raise ReplayCutPointError(
            f"unknown cut stage {cut_stage!r}; available stages: "
            f"{', '.join(sorted(stage_tags))}."
        )
    resolved = stage_tags[cut_stage]
    if resolved < 1:
        raise ReplayCutPointError(
            f"stage_tags[{cut_stage!r}] must be >= 1 — a stage tag is the "
            f"1-based count of exchanges completed when the stage closed, got "
            f"{resolved}."
        )
    return resolved, cut_stage


def cut_point_provenance(
    exchanges: list[LLMExchange],
    *,
    max_exchanges: int | None = None,
    branch_stage: str | None = None,
) -> dict[str, Any]:
    """The configured ``cut_point`` block recorded in source_provenance (RFC §3.5).

    ``n_replayed_exchanges`` is the replay prefix the run is *configured* to
    serve (the natural end when ``max_exchanges`` is ``None``) and
    ``recorded_request_digest`` the comparable digest of the last replayed
    exchange's *recorded* request — both computed from the recording at
    config time, so the block carries ``accounting: "configured"`` and every
    field only the live run could observe is recorded honestly as null:
    ``served_request_digest`` (no incoming request exists yet),
    ``divergences`` (not observed on this basis) and ``workspace_digest``
    (no sandbox is reachable at the cut point on this basis — the in-sandbox
    replay proxy crosses the cut without notifying the host). In sandbox
    proxy mode the uploaded recording is truncated to exactly this prefix,
    so configured accounting is the honest basis there; in host proxy mode
    the orchestrator reconciles the block after the run with what the live
    router actually served (:func:`served_cut_point`). ``branch_stage`` is
    recorded only for a stage-named cut.
    """
    n_replayed = validate_max_exchanges(max_exchanges, len(exchanges))
    block: dict[str, Any] = {
        "n_replayed_exchanges": n_replayed,
        "recorded_request_digest": (
            comparable_request_digest(exchanges[n_replayed - 1].request.body)
            if n_replayed > 0
            else None
        ),
        "served_request_digest": None,
        "request_digest_basis": REQUEST_DIGEST_BASIS,
        "accounting": "configured",
        "divergences": None,
        "workspace_digest": None,
        "workspace_digest_reason": (
            "configured-basis block computed from the recording alone; no "
            "live sandbox is reachable at the cut point on this basis"
        ),
    }
    if branch_stage is not None:
        block["branch_stage"] = branch_stage
    return block


def served_cut_point(
    router: ReplayRouter,
    *,
    configured_max_exchanges: int | None = None,
    branch_stage: str | None = None,
) -> dict[str, Any]:
    """The post-run ``cut_point`` block reconciled with what was served.

    Host proxy mode keeps the live :class:`ReplayRouter` in-process, so after
    the run the block records the replay prefix the proxy *actually* served
    (``accounting: "served"``): ``served_request_digest`` is the comparable
    digest of the ACTUAL request the agent sent at the cut,
    ``recorded_request_digest`` the recorded request it answered for — named
    separately so a reader can compare them — plus every divergence event the
    router detected and the workspace digest taken as the run crossed the cut
    (null with a reason when none could be, never fabricated). When an
    explicit cut was requested, ``configured_max_exchanges`` preserves the
    requested K so a served/configured divergence (e.g. the agent stopped
    asking before the cut) stays visible; ``branch_stage`` is kept for a
    stage-named cut.
    """
    block: dict[str, Any] = {
        "n_replayed_exchanges": router.n_replayed_exchanges,
        "served_request_digest": router.served_request_digest,
        "recorded_request_digest": router.recorded_request_digest,
        "request_digest_basis": REQUEST_DIGEST_BASIS,
        "accounting": "served",
        "divergences": list(router.divergence_events),
        "workspace_digest": router.workspace_digest,
    }
    if router.workspace_digest is None:
        block["workspace_digest_reason"] = router.workspace_digest_reason
    if configured_max_exchanges is not None:
        block["configured_max_exchanges"] = configured_max_exchanges
    if branch_stage is not None:
        block["branch_stage"] = branch_stage
    return block


@dataclass(frozen=True)
class ContinuedUsageSummary:
    """Token usage recovered from the stitched LLM trajectory."""

    n_input_tokens: int
    n_output_tokens: int
    n_cache_read_tokens: int
    n_cache_creation_tokens: int
    total_tokens: int
    recorded_total_tokens: int
    live_total_tokens: int
    usage_source: str

    def as_agent_result_patch(self) -> dict[str, Any]:
        return {
            "n_input_tokens": self.n_input_tokens,
            "n_output_tokens": self.n_output_tokens,
            "n_cache_read_tokens": self.n_cache_read_tokens,
            "n_cache_creation_tokens": self.n_cache_creation_tokens,
            "total_tokens": self.total_tokens,
            "usage_source": self.usage_source,
            "usage_details": {
                "source": "stitched_llm_trajectory",
                "recorded_total_tokens": self.recorded_total_tokens,
                "live_total_tokens": self.live_total_tokens,
            },
        }


def resolve_task_path(run: RunFolder, tasks_dir: str | Path | None) -> Path:
    """Find the task directory (instruction + verifier + image) to run against.

    The uploaded run folder does **not** ship the task files, so we need the
    original task source: ``--tasks-dir/<task_name>`` if given, else the
    ``task_path`` recorded in ``config.json`` if it still exists on disk.
    """
    if tasks_dir is not None:
        candidate = Path(tasks_dir).expanduser() / run.task_name
        if candidate.is_dir():
            return candidate
        raise RunFolderError(
            f"--tasks-dir given but {candidate} does not exist; expected the "
            f"task source for {run.task_name!r} there."
        )
    recorded = Path(run.task_path) if run.task_path else None
    if recorded is not None and recorded.is_dir():
        return recorded
    raise RunFolderError(
        f"cannot locate task source for {run.task_name!r}. Pass --tasks-dir "
        "pointing at the directory that contains the task (its instruction and "
        "verifier are needed to re-run and re-verify)."
    )


def build_agent_env(proxy_base_url: str) -> dict[str, str]:
    """Env that points OpenHands' LiteLLM client at the replay proxy."""
    return {
        "LLM_BASE_URL": proxy_base_url,
        "LLM_API_KEY": _REPLAY_API_KEY,
        "LLM_MODEL": _REPLAY_MODEL,
    }


def build_rollout_config(
    run: RunFolder,
    *,
    task_path: Path,
    live_model: str | None,
    agent_env: dict[str, str],
    timeout: int | None,
    output_dir: str | Path,
    rollout_name: str,
    max_exchanges: int | None = None,
    branch_stage: str | None = None,
) -> Any:
    """Assemble the RolloutConfig that re-runs the task through the proxy.

    Imported lazily so ``benchflow.continue_run`` stays importable without
    pulling the full rollout stack (keeps unit tests light).
    """
    from benchflow.rollout import RolloutConfig

    return RolloutConfig.from_legacy(
        task_path=task_path,
        agent="openhands",
        # model=None on purpose: the agent targets the replay proxy purely via
        # agent_env LLM_*. A real model here would make resolve_agent_env run
        # provider resolution and *validate the model's API key* — which would
        # wrongly fail the key-free replay phase and could clobber LLM_BASE_URL.
        # The live model lives in the forwarder + provenance instead.
        model=None,
        reasoning_effort=run.reasoning_effort,
        # values are str (a subset of str | None); list invariance needs the cast.
        prompts=cast("list[str | None] | None", run.prompts or None),
        environment=run.environment,
        sandbox_user=run.sandbox_user,
        agent_env=agent_env,
        # The seam that stops benchflow starting its own gateway / rewriting
        # LLM_BASE_URL — our replay proxy stays in front of the agent.
        usage_tracking="off",
        timeout=timeout,
        agent_idle_timeout=run.agent_idle_timeout_sec,
        jobs_dir=str(output_dir),
        rollout_name=rollout_name,
        source_provenance={
            "kind": "benchflow-continue",
            "continued_from": str(run.path),
            "original_error_category": run.error_category,
            "original_model": run.model,
            "live_model": live_model,
            "n_recorded_exchanges": run.n_recorded_exchanges,
            "cut_point": cut_point_provenance(
                run.exchanges,
                max_exchanges=max_exchanges,
                branch_stage=branch_stage,
            ),
        },
    )


class LiteLLMLiveForwarder:
    """Live forwarder backed by the in-process LiteLLM SDK.

    Past the replay cut-point the agent's request is sent to the real model and
    the final (non-streamed) ChatCompletion is returned for the proxy to emit.
    Resolving the provider route once keeps each call cheap; litellm reads
    provider credentials (e.g. ``GEMINI_API_KEY``) from the environment.
    """

    # Request fields worth forwarding verbatim when present.
    _PASSTHROUGH = (
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "response_format",
        "reasoning_effort",
        "parallel_tool_calls",
    )

    def __init__(self, model: str, *, env: dict[str, str] | None = None) -> None:
        from benchflow.providers.litellm_config import resolve_litellm_route

        merged_env = {**os.environ, **(env or {})}
        route = resolve_litellm_route(model, merged_env)
        self.model = model
        self.upstream_model = route.upstream_model

    def build_kwargs(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """Translate an OpenAI request body into ``litellm.completion`` kwargs."""
        kwargs: dict[str, Any] = {
            "model": self.upstream_model,
            "messages": request_body.get("messages") or [],
            "stream": False,
        }
        for key in self._PASSTHROUGH:
            if request_body.get(key) is not None:
                kwargs[key] = request_body[key]
        return kwargs

    def __call__(self, request_body: dict[str, Any]) -> dict[str, Any]:
        import litellm

        response = litellm.completion(**self.build_kwargs(request_body))
        dump = getattr(response, "model_dump", None)
        return dump() if callable(dump) else dict(response)


def stitched_trajectory_lines(
    recorded_lines: list[str],
    live_exchanges: list[LLMExchange],
    *,
    max_recorded: int | None = None,
) -> list[str]:
    """Build the continuous llm_trajectory: recorded prefix + live suffix.

    ``recorded_lines`` are the *parsed* exchanges' verbatim source lines
    (:func:`~benchflow.continue_run.run_folder.load_llm_exchanges_with_lines`
    — already redacted and byte-identical to what the agent replayed). The
    replay counts parsed exchanges, so ``max_recorded`` selects exactly the
    first K replayed exchanges' lines — a malformed (never-replayed) line in
    the source file can neither appear in the stitched prefix nor shift the
    cut. The live suffix is the exchanges the proxy captured after the
    cut-point, redacted on the way out.

    ``max_recorded`` is the prefix that was *served*, which is the configured
    cut only when the agent consumed all of it — callers pass the router's
    served count in host proxy mode, so a run the agent abandoned early does
    not get recorded responses it never received appended to its trajectory.
    """
    lines = list(recorded_lines)
    if max_recorded is not None:
        lines = lines[:max_recorded]
    for exchange in live_exchanges:
        raw = json.dumps(exchange.model_dump(mode="json"), default=str)
        lines.append(redact_trajectory_text(raw))
    return lines


def write_stitched_trajectory(
    rollout_dir: Path,
    recorded_lines: list[str],
    live_exchanges: list[LLMExchange],
    *,
    max_recorded: int | None = None,
) -> Path:
    """Write the stitched continuous trajectory into the new rollout folder."""
    out = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = stitched_trajectory_lines(
        recorded_lines, live_exchanges, max_recorded=max_recorded
    )
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    return out


def _usage_int(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def summarize_llm_trajectory_usage(
    trajectory_path: Path, *, n_recorded: int
) -> ContinuedUsageSummary:
    """Recover aggregate token usage from provider responses in a trajectory.

    ``n_recorded`` is the recorded/live boundary inside the stitched file and
    must be the same prefix length that produced it (the served count in host
    proxy mode, the configured one in sandbox mode) — otherwise live tokens
    are billed as replayed, or the reverse.
    """
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_creation = 0
    total_tokens = 0
    recorded_total = 0
    live_total = 0
    for idx, raw in enumerate(trajectory_path.read_text().splitlines()):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        response = payload.get("response")
        if not isinstance(response, dict):
            continue
        body = response.get("body")
        if not isinstance(body, dict):
            continue
        usage = body.get("usage")
        if not isinstance(usage, dict):
            continue
        prompt = _usage_int(usage, "prompt_tokens", "input_tokens")
        completion = _usage_int(usage, "completion_tokens", "output_tokens")
        total = _usage_int(usage, "total_tokens")
        if total <= 0:
            total = prompt + completion
        input_tokens += prompt
        output_tokens += completion
        total_tokens += total
        cache_read += _usage_int(usage, "cache_read_input_tokens")
        cache_creation += _usage_int(usage, "cache_creation_input_tokens")
        if idx < n_recorded:
            recorded_total += total
        else:
            live_total += total
    return ContinuedUsageSummary(
        n_input_tokens=input_tokens,
        n_output_tokens=output_tokens,
        n_cache_read_tokens=cache_read,
        n_cache_creation_tokens=cache_creation,
        total_tokens=total_tokens,
        recorded_total_tokens=recorded_total,
        live_total_tokens=live_total,
        usage_source="provider_response" if total_tokens > 0 else "unavailable",
    )


def update_continued_metadata(
    rollout_dir: Path,
    *,
    live_model: str | None,
    usage: ContinuedUsageSummary,
    environment: str,
    cut_point: dict[str, Any] | None = None,
) -> None:
    """Patch output metadata that Rollout could not know while model=None.

    ``benchflow continue`` intentionally runs Rollout with ``model=None`` so
    provider resolution does not clobber the replay proxy. After the run, the
    HF-compatible artifacts still need the actual live model and token usage.
    The stitched LLM trajectory is authoritative for provider usage.

    ``cut_point`` (host proxy mode) replaces the configured ``cut_point``
    block in both files' ``source`` with the served-accounting block
    (:func:`served_cut_point`) — reconciling provenance with what the live
    router actually replayed.
    """
    config_path = rollout_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        config["model"] = live_model
        config.setdefault("source", {})["usage_source"] = "stitched_llm_trajectory"
        if cut_point is not None:
            config.setdefault("source", {})["cut_point"] = cut_point
        config["usage_tracking"] = {
            "requested": "required",
            "status": (
                "captured_from_stitched_llm_trajectory"
                if usage.total_tokens > 0
                else "unavailable"
            ),
            "environment": environment,
            "endpoint_kind": "sandbox"
            if environment in _SANDBOX_LOCAL_REPLAY_ENVIRONMENTS
            else "host",
            "usage_source": usage.usage_source,
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    result_path = rollout_dir / "result.json"
    if not result_path.is_file():
        return
    result = json.loads(result_path.read_text())
    result["model"] = live_model
    if cut_point is not None:
        source = result.get("source")
        if not isinstance(source, dict):
            source = {}
            result["source"] = source
        source["cut_point"] = cut_point
    agent_result = result.setdefault("agent_result", {})
    if isinstance(agent_result, dict):
        agent_result.update(usage.as_agent_result_patch())
    result["final_metrics"] = {
        "total_prompt_tokens": usage.n_input_tokens,
        "total_completion_tokens": usage.n_output_tokens,
        "total_cached_tokens": usage.n_cache_read_tokens,
        "total_cost_usd": agent_result.get("cost_usd")
        if isinstance(agent_result, dict)
        else None,
    }
    usage_tracking = result.setdefault("usage_tracking", {})
    if isinstance(usage_tracking, dict):
        usage_tracking["requested"] = "required"
        usage_tracking["usage_source"] = usage.usage_source
        usage_tracking["environment"] = environment
        usage_tracking["endpoint_kind"] = (
            "sandbox" if environment in _SANDBOX_LOCAL_REPLAY_ENVIRONMENTS else "host"
        )
        usage_tracking["status"] = (
            "captured_from_stitched_llm_trajectory"
            if usage.total_tokens > 0
            else usage_tracking.get("status", "off")
        )
    result_path.write_text(json.dumps(result, indent=2) + "\n")


def write_continuation_artifacts(
    rollout_dir: Path,
    run: RunFolder,
    live_exchanges: list[LLMExchange],
    *,
    n_recorded: int,
    cut_point: dict[str, Any],
    live_model: str | None,
) -> Path:
    """Write the stitched trajectory and its metadata on one replay basis.

    The three artifacts a continuation leaves behind only make sense together:
    the stitched ``llm_trajectory.jsonl`` (recorded prefix + live suffix), the
    recorded-vs-live token split derived from it, and the ``cut_point``
    provenance block. Producing them here, from a single ``n_recorded``,
    is what keeps them from disagreeing — a prefix length used for two of the
    three and a different one for the last leaves artifacts no experiment can
    be run on.

    The caller chooses the basis and passes the matching block: host proxy
    mode reads the prefix the live router actually *served*
    (:func:`served_cut_point`), sandbox proxy mode uploads a recording already
    truncated to the *configured* prefix and keeps that
    (:func:`cut_point_provenance`). The block's ``accounting`` field is what
    tells a reader which of the two produced these numbers.
    """
    stitched_path = write_stitched_trajectory(
        rollout_dir,
        run.exchange_lines,
        live_exchanges,
        max_recorded=n_recorded,
    )
    update_continued_metadata(
        rollout_dir,
        live_model=live_model,
        usage=summarize_llm_trajectory_usage(stitched_path, n_recorded=n_recorded),
        environment=run.environment,
        cut_point=cut_point,
    )
    return stitched_path


def _host_workspace_digest_fn(
    rollout: Any, loop: asyncio.AbstractEventLoop
) -> Callable[[], dict[str, Any]]:
    """The router's cut-point workspace hook, bound to a live host-mode run.

    Called by :class:`ReplayRouter` on the proxy's HTTP handler thread the
    moment the first live-leg request arrives — the workspace the replay just
    rebuilt is still live in the rollout's sandbox at that moment, so the
    digest is scheduled onto the run's event loop
    (:func:`~benchflow.sandbox.workspace_digest.compute_workspace_digest`)
    and awaited from the handler thread. Raises — with the reason — when no
    sandbox is attached or the hook is somehow invoked on the loop thread
    itself (waiting there would deadlock the run); the router records the
    reason and never fabricates a digest.
    """
    from benchflow.sandbox.workspace_digest import compute_workspace_digest

    def _digest() -> dict[str, Any]:
        sandbox = getattr(rollout, "env", None)
        if sandbox is None:
            raise RuntimeError(
                "no live sandbox is attached to the continuation rollout"
            )
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            raise RuntimeError(
                "cut point crossed on the run's own event-loop thread; "
                "waiting for the digest here would deadlock the run"
            )
        future = asyncio.run_coroutine_threadsafe(
            compute_workspace_digest(sandbox), loop
        )
        return future.result(timeout=180)

    return _digest


def _host_proxy_binding(environment: str) -> tuple[str, str]:
    """(bind_host, advertise_host) so a Docker agent can reach the host proxy.

    Reuses benchflow's existing bridge-gateway logic so the container can talk
    back to the host-run proxy; non-docker environments use loopback.
    """
    from benchflow.providers.litellm_runtime import (
        _agent_endpoint_for_environment,
        _host_bind_address,
    )

    bind = _host_bind_address(environment)
    endpoint = _agent_endpoint_for_environment(0, environment, bind)
    advertise = endpoint.agent_base_url.split("//", 1)[-1].rsplit(":", 1)[0]
    return bind, advertise


async def _safe_sandbox_continuation_teardown(
    *,
    rollout: Any,
    replay_proxy: SandboxReplayProxy | None,
    provider_runtime: Any | None,
    stop_provider_runtime: Callable[[Any], Awaitable[None]],
    before_cleanup: Callable[[], Awaitable[None]] | None = None,
) -> list[str]:
    """Stop continuation sidecars but always let Rollout write final artifacts."""
    errors: list[str] = []

    async def _capture(label: str, awaitable: Awaitable[Any]) -> None:
        try:
            await awaitable
        except Exception as exc:
            message = f"{label}: {describe_exception(exc)}"
            errors.append(message)
            logger.warning("Continuation teardown step failed: %s", message)

    if replay_proxy is not None:
        await _capture("sandbox replay proxy stop", replay_proxy.stop())
    if provider_runtime is not None:
        await _capture("provider runtime stop", stop_provider_runtime(provider_runtime))

    if errors and getattr(rollout, "_error", None) is None:
        rollout._error = "Continuation teardown warning: " + "; ".join(errors)

    if before_cleanup is not None:
        await _capture("continuation artifact write", before_cleanup())

    await _capture("rollout cleanup", rollout.cleanup())

    return errors


def _result_after_sandbox_teardown(rollout: Any) -> Any | None:
    """Return a Rollout result, forcing artifact emission after cleanup warnings."""
    result = rollout.result
    if result is not None:
        return result
    if getattr(rollout, "_rollout_dir", None) is None:
        return None
    if getattr(rollout, "_phase", None) not in ("verified", "cleaned"):
        rollout._phase = "cleaned"
    return rollout.result


async def continue_run(
    folder: str | Path,
    *,
    tasks_dir: str | Path | None = None,
    model: str | None = None,
    timeout: int | None = None,
    output_dir: str | Path | None = None,
    require_timeout: bool = False,
    strict_divergence: bool = False,
    replay_only: bool = False,
    proxy_mode: str = "auto",
    max_exchanges: int | None = None,
    stage_tags: dict[str, int] | None = None,
    cut_stage: str | None = None,
) -> ContinueResult:
    """Resume ``folder`` to completion via record-replay + live continuation.

    ``model`` overrides the live-continuation model (tests pass
    ``gemini-3.1-flash-lite-preview``); defaults to the original run's model.
    ``replay_only`` skips the live leg (rebuild-and-stop) — useful for testing
    the replay without provider credentials.
    ``max_exchanges`` replays only the first K recorded exchanges, then goes
    live (the RFC §3.5 cut-point). ``cut_stage`` names the cut by recorded
    stage instead: the run folder's ``stage_snapshots.json`` carries the
    1-based count of exchanges that had completed when each recorded stage
    closed (:func:`stage_tags_from_run`), and a cut at that stage replays
    exactly that many exchanges. ``stage_tags`` overrides the recorded
    registry when the SDK caller supplies its own stage→count mapping.
    """
    run = load_run_folder(folder, require_timeout=require_timeout)
    if cut_stage is not None and stage_tags is None:
        stage_tags = stage_tags_from_run(run, cut_stage)
    max_exchanges, branch_stage = resolve_cut_point(
        max_exchanges, stage_tags=stage_tags, cut_stage=cut_stage
    )
    # Fail closed on an out-of-range cut here, before anything runs. Each proxy
    # mode then resolves its own replay basis: host mode reads the served count
    # off the live router after the run, sandbox mode uploads a recording
    # truncated to the configured prefix and keeps that basis.
    validate_max_exchanges(max_exchanges, run.n_recorded_exchanges)
    task_path = resolve_task_path(run, tasks_dir)

    live_model = model or run.model
    live_forwarder = None
    if not replay_only:
        if live_model is None:
            raise RunFolderError(
                "no live-continuation model: the run recorded no model and "
                "--model was not given."
            )
        live_forwarder = LiteLLMLiveForwarder(live_model)

    resolved_proxy_mode = select_proxy_mode(proxy_mode, run.environment)
    if replay_only and resolved_proxy_mode == "sandbox":
        raise RunFolderError("replay-only mode is only supported with host proxy mode.")

    out_root = Path(output_dir) if output_dir else run.path.parent / "continued"
    rollout_name = continued_rollout_name(run)

    if resolved_proxy_mode == "sandbox":
        if live_model is None:
            raise RunFolderError(
                "no live-continuation model: the run recorded no model and "
                "--model was not given."
            )
        return await _continue_run_with_sandbox_proxy(
            run,
            task_path=task_path,
            live_model=live_model,
            timeout=timeout,
            output_dir=out_root,
            rollout_name=rollout_name,
            strict_divergence=strict_divergence,
            max_exchanges=max_exchanges,
            branch_stage=branch_stage,
        )

    router = ReplayRouter(
        run.exchanges,
        live_forwarder=live_forwarder,
        strict_divergence=strict_divergence,
        max_exchanges=max_exchanges,
    )

    bind_host, advertise_host = _host_proxy_binding(run.environment)
    proxy = ReplayProxy(router, host=bind_host, advertise_host=advertise_host).start()

    try:
        from benchflow.rollout import Rollout

        config = build_rollout_config(
            run,
            task_path=task_path,
            live_model=live_model,
            agent_env=build_agent_env(proxy.base_url),
            timeout=timeout if timeout is not None else run.timeout_sec,
            output_dir=out_root,
            rollout_name=rollout_name,
            max_exchanges=max_exchanges,
            branch_stage=branch_stage,
        )
        rollout = await Rollout.create(config)
        # RFC §3.5 workspace accounting: the router digests the continuation
        # workspace the moment the run crosses the cut into the live leg —
        # the one moment the replay-rebuilt workspace is both complete and
        # still live in the sandbox. If the run never crosses (or the digest
        # fails), the cut_point block records null with the reason.
        router.workspace_digest_fn = _host_workspace_digest_fn(
            rollout, asyncio.get_running_loop()
        )
        result = await rollout.run()
        rollout_dir = Path(rollout._rollout_dir or (out_root / rollout_name))
    finally:
        proxy.stop()

    # The served prefix, not the configured one, is what the agent actually
    # consumed: an agent that exits or errors before reaching the cut leaves
    # ``n_replayed_exchanges < n_replay``. Stitching the configured prefix
    # would append recorded responses the agent never received, and the usage
    # split would bill those tokens as "replayed" — while the cut_point block
    # below reported the smaller served count, contradicting both. One basis
    # for all three (``accounting: "served"`` names it in the artifact).
    n_served = router.n_replayed_exchanges
    write_continuation_artifacts(
        rollout_dir,
        run,
        router.live_exchanges,
        n_recorded=n_served,
        # Reconcile the configured cut_point block with what the live router
        # actually served — host mode's post-run accounting.
        cut_point=served_cut_point(
            router,
            configured_max_exchanges=max_exchanges,
            branch_stage=branch_stage,
        ),
        live_model=live_model,
    )

    return ContinueResult(
        rollout_dir=rollout_dir,
        rewards=getattr(result, "rewards", None),
        error=getattr(result, "error", None),
        n_recorded=n_served,
        n_live=len(router.live_exchanges),
        divergences=router.divergences,
    )


async def _continue_run_with_sandbox_proxy(
    run: RunFolder,
    *,
    task_path: Path,
    live_model: str,
    timeout: int | None,
    output_dir: Path,
    rollout_name: str,
    strict_divergence: bool,
    max_exchanges: int | None = None,
    branch_stage: str | None = None,
) -> ContinueResult:
    """Run continuation with replay and provider proxies inside the sandbox."""
    from benchflow.providers.runtime import (
        ensure_litellm_runtime,
        stop_provider_runtime,
    )
    from benchflow.rollout import Rollout

    # The in-sandbox proxy replays exactly the recording it is given, so a cut
    # is threaded by truncating the uploaded prefix — the switch to live then
    # happens exactly as if the recording had ended there.
    n_replay = validate_max_exchanges(max_exchanges, run.n_recorded_exchanges)
    config = build_rollout_config(
        run,
        task_path=task_path,
        live_model=live_model,
        agent_env=build_agent_env(sandbox_replay_base_url()),
        timeout=timeout if timeout is not None else run.timeout_sec,
        output_dir=output_dir,
        rollout_name=rollout_name,
        max_exchanges=max_exchanges,
        branch_stage=branch_stage,
    )
    rollout = await Rollout.create(config)
    replay_proxy: SandboxReplayProxy | None = None
    provider_runtime: Any | None = None
    result: Any | None = None
    pending_acp_error: AgentProtocolError | None = None
    agent_timed_out = False
    rollout_dir: Path | None = None
    live_exchanges: list[LLMExchange] = []
    artifacts_written = False

    async def _write_artifacts_before_cleanup() -> None:
        nonlocal artifacts_written, live_exchanges, result, rollout_dir
        if artifacts_written:
            return
        result = _result_after_sandbox_teardown(rollout)
        if result is None:
            return
        rollout_dir = Path(rollout._rollout_dir or (output_dir / rollout_name))
        live_exchanges = replay_proxy.live_exchanges if replay_proxy is not None else []
        write_continuation_artifacts(
            rollout_dir,
            run,
            live_exchanges,
            n_recorded=n_replay,
            # No live router on this side to read a served count off: the
            # in-sandbox proxy is handed a recording already truncated to the
            # configured prefix, so configured accounting is the honest basis
            # here — and the artifact says so, rather than leaving the reader
            # to infer which basis produced the numbers beside it.
            cut_point=cut_point_provenance(
                run.exchanges,
                max_exchanges=max_exchanges,
                branch_stage=branch_stage,
            ),
            live_model=live_model,
        )
        artifacts_written = True

    try:
        await rollout.setup()
        await rollout.start()
        await rollout.install_agent()

        provider_env = rollout._planes.resolve_agent_env("openhands", live_model, {})
        provider_env, provider_runtime = await ensure_litellm_runtime(
            agent="openhands",
            agent_env=provider_env,
            model=live_model,
            runtime=None,
            environment=run.environment,
            session_id=rollout_name,
            usage_tracking="required",
            sandbox=rollout.env,
        )
        replay_proxy = await SandboxReplayProxy.start(
            sandbox=rollout.env,
            recorded=run.exchanges[:n_replay],
            upstream_url=provider_env["LLM_BASE_URL"],
            upstream_api_key=provider_env["LLM_API_KEY"],
            upstream_model=provider_env["LLM_MODEL"],
            strict_divergence=strict_divergence,
        )

        try:
            if config.user is not None:
                await rollout._run_user_loop()
            else:
                await rollout._run_steps(
                    compile_scenes_to_steps(
                        config.effective_scenes,
                        default_prompt=(
                            rollout._resolved_prompts[0]
                            if rollout._resolved_prompts
                            else None
                        ),
                    )
                )
        except TimeoutError as exc:
            agent_timed_out = True
            detail = str(exc).strip()
            rollout._error = detail or f"Agent timed out after {rollout._timeout}s"
            rollout._diagnostics.capture_idle(exc)
            logger.error(rollout._error)

        if not config.skip_verify:
            await rollout.verify()
            if (
                agent_timed_out
                and rollout._rewards is None
                and rollout._verifier_error is None
            ):
                rollout._rewards = {"reward": 0.0}
                rollout._verifier_error = None

    except TimeoutError as exc:
        detail = str(exc).strip()
        rollout._error = detail or f"Agent timed out after {rollout._timeout}s"
        rollout._diagnostics.capture_idle(exc)
        logger.error(rollout._error)
    except ConnectionError as exc:
        rollout._error = str(exc)
        rollout._diagnostics.capture_transport(exc)
        await rollout._probe_sandbox_health()
        logger.error("Agent connection lost: %s", rollout._error)
    except SandboxStartupFailure as exc:
        rollout._error = f"Sandbox startup failed: {exc}"
        rollout._diagnostics.set(exc.diagnostic)
        logger.error(rollout._error)
    except AgentProtocolError as exc:
        pending_acp_error = exc
        rollout._error = str(exc)
        logger.error(str(exc))
    except Exception as exc:
        rollout._error = str(exc)
        logger.error("Run failed", exc_info=True)
    finally:
        await _safe_sandbox_continuation_teardown(
            rollout=rollout,
            replay_proxy=replay_proxy,
            provider_runtime=provider_runtime,
            stop_provider_runtime=stop_provider_runtime,
            before_cleanup=_write_artifacts_before_cleanup,
        )

    if pending_acp_error is not None:
        rollout._error = rollout._classify_acp_error(pending_acp_error)
        logger.error(rollout._error)

    await _write_artifacts_before_cleanup()
    if rollout_dir is None:
        rollout_dir = Path(rollout._rollout_dir or (output_dir / rollout_name))

    return ContinueResult(
        rollout_dir=rollout_dir,
        rewards=getattr(result, "rewards", None),
        error=getattr(result, "error", None),
        n_recorded=n_replay,
        n_live=len(live_exchanges),
        divergences=0,
    )
