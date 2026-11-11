"""LiteLLM proxy runtime orchestration for host and sandbox runs."""

from __future__ import annotations

import asyncio
import atexit
import base64
import contextlib
import contextvars
import hashlib
import importlib
import inspect
import json
import logging
import os
import posixpath
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import yaml

from benchflow._root_only import root_only_path_check, root_only_tree_check
from benchflow._utils import timing_spans as _timing_spans
from benchflow._utils.text import describe_exception
from benchflow._uv_pin import (
    PYTHON_INSTALL_DIR,
    PYTHON_VERSION,
    UV_BIN,
    UV_VERSION,
    managed_python_exports,
    managed_python_install_command,
    private_tmpdir_exports,
    uv_install_command,
    uv_lock_file_command,
    uv_lock_seal_command,
)
from benchflow.agents.codex_config import apply_codex_provider_config
from benchflow.agents.env import (
    uses_direct_anthropic_endpoint,
    uses_native_subscription_auth,
)
from benchflow.agents.registry import (
    AGENTS,
    GEMINI_NATIVE_BASE_URL_ENV,
    OPENCODE_GOOGLE_BASE_URL_ENV,
    agent_routes_own_model,
    is_scripted_agent,
    routed_model,
    routes_gemini_natively,
)
from benchflow.providers.litellm_bedrock_preflight import (
    BEDROCK_PATCH_PREFLIGHT_SOURCE,
    BedrockPatchPreflightError,
    preflight_host_bedrock_patch,
    preflight_sandbox_bedrock_patch,
    route_requires_bedrock_patch,
)
from benchflow.providers.litellm_config import (
    LITELLM_MASTER_KEY_ENV,
    LITELLM_MODEL_ALIAS_ENV,
    LITELLM_MODEL_VIA_ENV,
    LiteLLMRoute,
    litellm_proxy_config,
    resolve_litellm_route,
    strip_provider_prefix,
)
from benchflow.providers.litellm_logging import (
    CallbackLogReader,
    callback_module_source,
    extract_usage_from_trajectory,
    trajectory_from_litellm_callback_log,
)
from benchflow.providers.proxy_requirements import (
    ProxyInstall,
    ensure_host_venv,
    install_commands,
    proxy_install,
)
from benchflow.sandbox._base import (
    BASH_STARTUP_PREFIX,
    BASH_STARTUP_VARS,
    SYSTEM_SHELL_UNSET_VARS,
)
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw
from benchflow.sandbox.env_layers import (
    PROXY_LAYER_REQUIREMENTS_DIR,
    PROXY_LAYER_STAMP,
    PROXY_LAYER_VENV,
    LayerResult,
    ProxyLayerRecipe,
)
from benchflow.sandbox.providers import SANDBOX_MODEL_PROXY_PROVIDERS
from benchflow.trajectories._capture import ProviderToolEvidence
from benchflow.trajectories._llm_capture import (
    LiveLLMTrajectoryWriter,
    exchange_record,
    published_exchange,
)
from benchflow.trajectories.types import Trajectory
from benchflow.trial_budget import TrialBudget, budget_refusal
from benchflow.trust.state import TRUSTED_TRANSFER, admit
from benchflow.usage_tracking import UsageTrackingConfig, usage_unavailable

if TYPE_CHECKING:
    from benchflow._proxy_zygote import ProxyProcess
    from benchflow.contracts.planes import LiveUsageGateway
    from benchflow.providers.runtime import ProviderRuntime

logger = logging.getLogger(__name__)

LITELLM_VERSION_SPEC = "litellm[proxy]==1.103.3"
# In BenchFlow's reserved runtime tree, which no workspace grant covers: an
# honest /tmp workspace keeps its proxy (PY8-PT-R8-9).
LITELLM_SANDBOX_ROOT = "/opt/runtime/proxy"
_CALLBACK_MODULE = "benchflow_litellm_callback"
# Read by the callback's model gate (litellm_logging.callback_module_source).
_SERVED_MODELS_ENV = "BENCHFLOW_LITELLM_SERVED_MODELS"
# Beside the callback log: the callback's _note_request adds a line for each
# request that reaches the proxy (litellm_logging.callback_module_source).
_REQUEST_MARK_SUFFIX = ".requests"
# Beside it: the call id of each request whose record the callback wrote
# (litellm_logging's _note_logged).
_LOGGED_MARK_SUFFIX = ".logged"
_PATCH_MODULE = "benchflow_litellm_bedrock_patch"
_GEMINI_PATCH_MODULE = "benchflow_litellm_gemini_passthrough_patch"
_TOKEN_CAPTURE_PATCH_MODULE = "benchflow_litellm_token_capture_patch"
# MiMo's transient "Invalid request parameters" 400s are retried in the proxy, twice (litellm_provider_retry_patch).
_PROVIDER_RETRY_PATCH_MODULE = "benchflow_litellm_provider_retry_patch"
_REQUEST_GUARD_MODULE = "benchflow_litellm_request_guard"
# Host proxies only: route analysis and model builds deferred to first use
# (litellm_lazy_start). ``0`` starts the proxy as before.
_LAZY_START_MODULE = "benchflow_litellm_lazy_start"
PROXY_LAZY_START_ENV = "BENCHFLOW_PROXY_LAZY_START"
# Non-secret route facts the proxy callback uses to decide which token-capture
# fields to request (see ``litellm_logging.callback_module_source``).
_ROUTE_PROVIDER_ENV = "BENCHFLOW_LITELLM_ROUTE_PROVIDER"
_UPSTREAM_MODEL_ENV = "BENCHFLOW_LITELLM_UPSTREAM_MODEL"

# The proxy is an internal single-route gateway — it must never register the
# FastAPI Swagger docs route. litellm's `_get_docs_url()` honours an inherited
# `DOCS_URL` env *before* `NO_DOCS`, so a stray non-"/" `DOCS_URL` baked into a
# sandbox base image makes litellm call `add_route(DOCS_URL, ...)` and crash with
# `Routed paths must start with '/'` at startup. Force `DOCS_URL=""` (falsy, so
# it's ignored) and `NO_DOCS=true` so the docs route is skipped regardless of the
# inherited environment.
_PROXY_DOCS_DISABLE_ENV = {"DOCS_URL": "", "NO_DOCS": "true"}
_SKILL_CATALOG_GATE_AGENT_ENV = "BENCHFLOW_SKILL_CATALOG_GATE_AGENT"
_REQUIRED_SKILL_NAMES_ENV = "BENCHFLOW_REQUIRED_SKILL_NAMES_JSON"
# Live callback-log capture budgets. The reader tails the gateway's
# callback.jsonl by byte range, one command per read, and each read on the
# sandbox path costs a full transient-exec round trip (Daytona: create
# session, run, poll on a 1s interval, fetch logs, delete) — so the per-read
# *yield* is what sets the tail's throughput ceiling, not the poll cadence.
# The shipped 24KB-per-`dd bs=1` read put that ceiling near the rate a
# full-message provider log grows at (#965 follow-up: DeepSeek writes the whole
# conversation per record), and any single failed read silently cost the whole
# tick — which is how the dashboard's token figure froze at a plausible number
# for 20 minutes of a live run. 512KB per read (~700KB of base64 on the wire)
# over 8 reads/tick raises the ceiling by ~20x per round trip while cutting
# round trips per tick from 64 to 8.
_LIVE_CAPTURE_CHUNK_BYTES = 512 * 1024
_LIVE_CAPTURE_MAX_READS_PER_TICK = 8
# dd block size inside the sandbox: the range is selected with
# iflag=skip_bytes,count_bytes so the read is a seek plus a handful of
# full-size reads, instead of `bs=1`'s two syscalls per byte.
_LIVE_CAPTURE_READ_BLOCK_BYTES = 64 * 1024
# Per-read sandbox deadline. A read that trips it advances nothing, so it must
# clear the transport comfortably: Daytona polls session commands on a 1s
# interval and this payload is ~700KB of base64.
_LIVE_CAPTURE_READ_TIMEOUT_SEC = 20
_LIVE_CAPTURE_INTERVAL_SEC = 1.0
# Consecutive zero-progress polls (~1s each) before the stall is escalated from
# debug to a single WARNING. A capture that is behind but still advancing is
# normal backlog; one that has stopped advancing is the state that renders a
# frozen-but-plausible token count.
_LIVE_CAPTURE_STALL_WARN_TICKS = 30

# Agents that cannot make model calls through LiteLLM. ``oracle`` has no model
# at all. Gemini is routable through LiteLLM's native Google GenerateContent
# endpoints; keeping it behind the proxy is required for no-web reviewer runs.
# Providers whose mandatory LiteLLM proxy runs inside the sandbox. Keeping this
# placement policy in the canonical provider registry prevents a new backend from
# accidentally handing an in-sandbox agent a host-loopback endpoint.
_SANDBOX_LOCAL_ENVIRONMENTS = SANDBOX_MODEL_PROXY_PROVIDERS


def proxy_runs_on_host(environment: str, *, force_sandbox_local: bool = False) -> bool:
    """Whether a proxy this run starts is a host process (``_start_host_litellm``),
    which runs nothing in the sandbox; the one placement rule
    ``ensure_litellm_runtime`` uses."""
    return not (force_sandbox_local or environment in _SANDBOX_LOCAL_ENVIRONMENTS)


@dataclass(frozen=True)
class LiteLLMEndpoint:
    """Endpoint visible to the agent plus host-local endpoint for health checks."""

    agent_base_url: str
    local_base_url: str


@dataclass(frozen=True)
class _CallbackChunk:
    """One byte-ranged read of the gateway's callback log.

    ``size`` is the log's size observed by the same command that produced
    ``data``, or None when the read failed. It carries the two facts the
    previous bare-``bytes`` return could not express, both of which the live
    token counter depends on:

    * an empty read is "caught up" only when the reader can see it is standing
      at EOF (``offset >= size``). Without the size, a *failed* read looks
      identical to a drained log — which is how a broken tail kept reporting a
      stale total as if it were current.
    * how far behind the tail is, in bytes, so the lag is a number in the log
      rather than an invisible condition.
    """

    data: bytes
    size: int | None = None


def _warn_refused_requests(trajectory: Trajectory | None) -> None:
    """Say which models the gateway refused, from its callback log's records."""
    if trajectory is None:
        return
    refused = trajectory.metadata.get("refused_requests") or []
    counts = Counter(
        str(entry.get("request_model"))
        for entry in refused
        # A call the trial's budget no longer allowed is the trial's ending.
        if isinstance(entry, dict) and entry.get("rule") != "trial-budget"
    )
    if counts:
        logger.warning(
            "LiteLLM gateway refused %d request(s) for model(s) this run does not "
            "serve: %s",
            sum(counts.values()),
            ", ".join(f"{model} x{n}" for model, n in sorted(counts.items())),
        )


async def _capture_to_thread[T](
    work: Callable[[], T], publish: Callable[[T], None] | None = None
) -> T:
    """Run a capture step in a worker thread, and drain it before a
    cancellation lets teardown proceed. ``publish`` takes the step's result on
    the event loop, also when a cancellation arrived while the step ran: what
    the worker already wrote to disk is then also in the live view."""
    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        # Cancelling to_thread does not stop its thread. Final reconciliation
        # and scratch cleanup must wait for the active writer, even when
        # shutdown cancels the polling task more than once (public #1055).
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            except Exception:
                break
        if publish is not None and not task.cancelled() and task.exception() is None:
            publish(task.result())
        with contextlib.suppress(Exception):
            task.result()
        raise
    if publish is not None:
        publish(result)
    return result


class LiteLLMProcess:
    """Common interface for a running LiteLLM proxy."""

    route: LiteLLMRoute
    trajectory: Trajectory | None
    session_id: str
    agent_name: str
    # Live-capture token counter (class-level defaults so reads are plain
    # attribute access even before start_live_capture ever runs).
    _live_usage_total_tokens: int = 0
    _live_usage_seen: bool = False
    # Live-capture lag bookkeeping. ``_live_capture_lag_bytes`` is 0 when the
    # tail reached EOF on the last poll, a byte count when it is behind, and
    # None when it is behind by an unknown amount (the read itself failed).
    _live_capture_lag_bytes: int | None = 0
    _live_capture_stall_ticks: int = 0
    _live_capture_stall_warned: bool = False
    # Polls the live capture has finished, and the first call the proxy
    # refused for the trial's budget (benchflow.trial_budget) that one read.
    _live_capture_passes: int = 0
    _trial_budget_refusal: dict[str, Any] | None = None

    @property
    def base_url(self) -> str:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def is_running(self) -> bool:
        raise NotImplementedError

    def start_live_capture(
        self,
        path: Path,
        *,
        preceding_trajectories: tuple[Trajectory, ...] | None = None,
    ) -> None:
        """Mirror completed callback records into a redacted local artifact."""
        existing_path = getattr(self, "_live_trajectory_path", None)
        task = getattr(self, "_live_capture_task", None)
        if existing_path == Path(path) and task is not None and not task.done():
            return
        if task is not None and not task.done():
            task.cancel()
        # Runtime reuse may restart polling or change the output path without
        # supplying the ledger again. Its earlier roles still belong here.
        if preceding_trajectories is not None:
            self._preceding_trajectories = preceding_trajectories
        self._live_trajectory_path = Path(path)
        self._live_writer = LiveLLMTrajectoryWriter(
            Path(path),
            preceding_trajectories=getattr(self, "_preceding_trajectories", ()),
        )
        self._live_trajectory = Trajectory(
            session_id=self.session_id,
            agent_name=self.agent_name,
        )
        self._live_callback_offset = 0
        self._live_callback_remainder = b""
        self._live_reconciled = False
        self._request_events = {}
        # Clear ``seen`` BEFORE the total so a racing render-thread read can
        # only observe None (fresh capture) — never a transient "0 tokens".
        self._live_usage_seen = False
        self._live_usage_total_tokens = 0
        self._live_capture_lag_bytes = 0
        self._live_capture_stall_ticks = 0
        self._live_capture_stall_warned = False
        self._live_capture_task = asyncio.create_task(self._live_capture_loop())

    def live_usage_tokens(self) -> int | None:
        """Cumulative provider tokens the live callback capture has mirrored.

        Read-only, O(1) — safe to call from the dashboard's render thread while
        the capture loop appends on the event loop (two monotonic attribute
        reads; a torn read is at worst one record stale). None until a
        usage-bearing gateway record has been parsed, so callers can tell "no
        signal yet" from a genuine zero. The value updates per completed LLM
        *request* — mid-prompt — which is exactly the granularity the ACP
        per-completed-prompt snapshots lack (a single-prompt rollout otherwise
        shows no tokens until scoring). It may lag the gateway log by the
        capture cadence/backlog and is display-only: trusted usage still comes
        from the full log import at ``stop()``.
        """
        if not self._live_usage_seen:
            return None
        return self._live_usage_total_tokens

    def trial_budget_refusal(self) -> dict[str, Any] | None:
        """The proxy's record of the first call it refused because the
        trial's budget was spent, as the live capture has read it; None
        before (or without) one. O(1), like :meth:`live_usage_tokens`."""
        return self._trial_budget_refusal

    async def settle_live_capture(self, timeout: float = 5.0) -> None:
        """Wait until the live capture has read the callback log as it
        stands now (two polls that reached its end), or ``timeout`` seconds:
        an agent's turn can end on a call before the next poll reads it."""
        task = getattr(self, "_live_capture_task", None)
        if task is None or task.done():
            return
        start = self._live_capture_passes
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (
                self._live_capture_passes >= start + 2
                and self._live_capture_lag_bytes == 0
            ):
                return
            await asyncio.sleep(0.1)

    def _capture_request_events(self, text: str) -> None:
        path = getattr(self, "_live_trajectory_path", None)
        if path is None:
            return
        # Final imports run in a worker. Readers iterate an immutable snapshot
        # while this method builds the next one.
        events = dict(self._request_events)
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("event") not in {
                "request_start",
                "response_start",
                "request_end",
                "upstream_start",
                "upstream_end",
            }:
                continue
            key = (row["event"], row.get("attempt_id", row.get("request_id")))
            if key in events:
                continue
            events[key] = row
            with path.with_name("provider_requests.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        self._request_events = events

    def model_request_activity(self) -> tuple[int, float, float]:
        """Active requests, oldest age, last event time; no extra sandbox poll."""
        events = getattr(self, "_request_events", {})
        starts = [
            row["time"]
            for (kind, key), row in events.items()
            if kind == "request_start" and ("request_end", key) not in events
        ]
        latest = max((row["time"] for row in events.values()), default=0.0)
        return (
            len(starts),
            max(0.0, time.time() - min(starts)) if starts else 0.0,
            latest,
        )

    def stalled_model_requests(self) -> tuple[int, float]:
        """Requests the proxy has not begun to answer: count, oldest age."""
        events = getattr(self, "_request_events", {})
        starts = [
            row["time"]
            for (kind, key), row in events.items()
            if kind == "request_start"
            and ("request_end", key) not in events
            and ("response_start", key) not in events
        ]
        if not starts:
            return 0, 0.0
        return len(starts), max(0.0, time.time() - min(starts))

    def request_summary(self) -> tuple[dict[str, float], str | None]:
        events = getattr(self, "_request_events", {})
        ends = [row for (kind, _), row in events.items() if kind == "request_end"]
        intervals = sorted(
            (events[("request_start", row["request_id"])]["time"], row["time"])
            for row in ends
            if ("request_start", row["request_id"]) in events
        )
        waiting = 0.0
        through = 0.0
        for start, end in intervals:
            waiting += max(0.0, end - max(start, through))
            through = max(through, end)
        timing = {
            "provider_wait": waiting,
            "provider_retry_wait": sum(row["retry_wait_seconds"] for row in ends),
        }
        last = max(ends, key=lambda row: row["time"], default=None)
        invalid = [
            row
            for row in events.values()
            if row.get("invalid_provider_response")
            and last is not None
            and row["request_id"] == last["request_id"]
            and (last["status"] is None or last["status"] >= 400)
        ]
        error = None
        if invalid:
            row = invalid[-1]
            error = (
                f"invalid provider response: upstream HTTP {row['status']}, "
                f"non-JSON body ({row['response_bytes']} bytes), "
                f"upstream {row['upstream_seconds']:.3f}s; "
                "next: inspect trajectory/provider_requests.jsonl and retry the provider"
            )
        return timing, error

    async def _read_callback_chunk(self, offset: int, limit: int) -> _CallbackChunk:
        raise NotImplementedError

    async def _capture_live_records(self) -> None:
        trajectory = getattr(self, "_live_trajectory", None)
        writer = getattr(self, "_live_writer", None)
        if trajectory is None or writer is None:
            return

        start_offset = int(getattr(self, "_live_callback_offset", 0))
        caught_up = False
        log_size: int | None = None
        for _ in range(_LIVE_CAPTURE_MAX_READS_PER_TICK):
            offset = int(getattr(self, "_live_callback_offset", 0))
            try:
                chunk = await self._read_callback_chunk(
                    offset, _LIVE_CAPTURE_CHUNK_BYTES
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A raising read (the sandbox's exec poll hitting its deadline,
                # a teardown race) is a lag like any other failed read. Catch it
                # HERE rather than letting the loop's blanket handler swallow
                # it, so the tick still reaches the lag accounting below instead
                # of stalling invisibly — the exact shape of the frozen counter
                # this fix exists to surface.
                logger.debug("Live LLM callback read failed: %s", exc)
                chunk = _CallbackChunk(b"", None)
            if chunk.size is not None:
                log_size = chunk.size
            if not chunk.data:
                # Empty means "drained" ONLY when the reader could see the
                # log's size and is standing at it. A failed read (size None)
                # is a lag, and must not be recorded as having caught up.
                caught_up = chunk.size is not None and offset >= chunk.size
                break
            self._live_callback_offset = offset + len(chunk.data)
            data = getattr(self, "_live_callback_remainder", b"") + chunk.data
            lines = data.split(b"\n")
            self._live_callback_remainder = lines.pop()
            for raw_line in lines:
                if not raw_line.strip():
                    continue
                await _capture_to_thread(
                    lambda raw_line=raw_line: self._parse_live_record(raw_line, writer),
                    lambda result: self._publish_live_record(trajectory, result),
                )
            if len(chunk.data) < _LIVE_CAPTURE_CHUNK_BYTES:
                caught_up = True
                break
        self._note_live_capture_lag(
            start_offset=start_offset, caught_up=caught_up, log_size=log_size
        )
        self._live_capture_passes += 1

    def _publish_live_record(
        self, trajectory: Trajectory, result: tuple[Trajectory, str | None]
    ) -> None:
        """On the event loop: one parsed record's request event, exchanges and
        tokens into the live view."""
        parsed, event = result
        if event is not None:
            self._capture_request_events(event)
        if self._trial_budget_refusal is None:
            self._trial_budget_refusal = budget_refusal(
                parsed.metadata.get("refused_requests")
            )
        if parsed.exchanges:
            # Each exchange is published once, as it arrives; the mirror keeps
            # only what a stopped proxy is read for.
            trajectory.exchanges.extend(parsed.exchanges)
            # Live token counter for the eval dashboard: accumulate the same
            # canonical per-exchange accounting scoring sums at stop()
            # (Trajectory.total_provider_tokens over the same parser's output),
            # so the live figure converges to the trusted total as the capture
            # catches up. Failure records carry no usage and leave the counter
            # untouched.
            if parsed.has_provider_usage:
                # Total before ``seen``: a racing reader can never observe
                # seen=True with a not-yet-updated total.
                self._live_usage_total_tokens += int(parsed.total_provider_tokens)
                self._live_usage_seen = True

    def _parse_live_record(
        self, raw_line: bytes, writer: LiveLLMTrajectoryWriter
    ) -> tuple[Trajectory, str | None]:
        """Parse, redact and publish off-loop; return only the compact mirror."""
        text = raw_line.decode("utf-8", errors="replace")
        parsed = trajectory_from_litellm_callback_log(
            text, session_id=self.session_id, agent_name=self.agent_name
        )
        if parsed.exchanges:
            writer.append(parsed.exchanges)
            parsed.exchanges = [published_exchange(e) for e in parsed.exchanges]
            return parsed, None
        # Lifecycle records are small. Only those return to the event loop;
        # an unrecognised record's potentially large body stays in the worker.
        try:
            row = json.loads(text)
        except ValueError:
            return parsed, None
        event = (
            text
            if isinstance(row, dict)
            and row.get("event")
            in {
                "request_start",
                "response_start",
                "request_end",
                "upstream_start",
                "upstream_end",
            }
            else None
        )
        return parsed, event

    def _note_live_capture_lag(
        self, *, start_offset: int, caught_up: bool, log_size: int | None
    ) -> None:
        """Record — and, once it persists, surface — a tail that is behind.

        A poll that ends without reaching EOF is exactly the state that made
        the dashboard's token figure lie: the cell keeps rendering a plausible
        cumulative number that has silently stopped tracking the run. Every
        behind poll is logged at debug with the byte lag; a tail that is behind
        AND has stopped advancing altogether earns one WARNING (latched — a
        stalled capture must not turn into a per-second log flood), because a
        frozen-but-plausible spend figure is worse than a visibly absent one:
        users read it to decide whether to kill a run.

        The counter itself is deliberately left alone. It is a cumulative lower
        bound at all times (the tail always trails the log by some amount), so
        blanking it on a stall would trade a slightly stale number for no
        number, and the dashboard already reconciles it as
        ``max(acp_snapshot, gateway_live)`` — a None here would just hand the
        cell to the other, equally lagging signal.
        """
        offset = int(getattr(self, "_live_callback_offset", 0))
        if caught_up:
            self._live_capture_lag_bytes = 0
            self._live_capture_stall_ticks = 0
            return

        lag = max(log_size - offset, 0) if log_size is not None else None
        self._live_capture_lag_bytes = lag
        lag_text = "an unknown number of" if lag is None else str(lag)
        if offset > start_offset:
            # Behind but still draining: normal backlog after a burst.
            self._live_capture_stall_ticks = 0
            logger.debug(
                "Live LLM capture is %s bytes behind the gateway log "
                "(read through byte %d this poll).",
                lag_text,
                offset,
            )
            return

        self._live_capture_stall_ticks += 1
        logger.debug(
            "Live LLM capture made no progress at byte %d (%s bytes behind, "
            "%d consecutive stalled polls).",
            offset,
            lag_text,
            self._live_capture_stall_ticks,
        )
        if (
            self._live_capture_stall_ticks >= _LIVE_CAPTURE_STALL_WARN_TICKS
            and not self._live_capture_stall_warned
        ):
            self._live_capture_stall_warned = True
            logger.warning(
                "Live token counter has stalled: the model-gateway log tail "
                "has not advanced past byte %d in %d polls (%s bytes behind). "
                "The live token/cost figures now UNDERCOUNT this run; the "
                "final totals, imported from the full log, are unaffected.",
                offset,
                self._live_capture_stall_ticks,
                lag_text,
            )

    async def _live_capture_loop(self) -> None:
        while True:
            try:
                await self._capture_live_records()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Live LLM trajectory mirror failed: %s", exc)
            await asyncio.sleep(_LIVE_CAPTURE_INTERVAL_SEC)

    async def _stop_live_capture(self) -> None:
        task = getattr(self, "_live_capture_task", None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self._live_capture_task = None
        with contextlib.suppress(Exception):
            await self._capture_live_records()

    def _reconcile_live_capture(self) -> None:
        writer = getattr(self, "_live_writer", None)
        if writer is None or getattr(self, "_live_reconciled", False):
            return
        # The import's reconcile wrote the whole file but could not replace
        # the old one: publish it now.
        try:
            if writer.retry_pending():
                self._live_reconciled = True
                return
        except OSError as exc:
            logger.warning(
                "Could not publish %s; it holds what the live mirror wrote: %s",
                writer.path,
                exc,
            )
            return
        writer.reconcile(self.trajectory)

    def _import_callback_lines(self, lines: Iterable[str]) -> None:
        """Import the stopped proxy's callback log, one line at a time.

        The trajectory is what :func:`trajectory_from_litellm_callback_log`
        makes of the whole log. When the live mirror publishes this run's
        ``llm_trajectory.jsonl``, each exchange goes whole to the mirror's
        reconcile as it is read (a line the mirror already published is not
        redacted again), its provider tool evidence is read from it, and only
        the published exchange stays in memory: a long run's log is never
        held in the runner, let alone twice.
        """
        writer = getattr(self, "_live_writer", None)
        evidence = ProviderToolEvidence() if writer is not None else None
        reader = CallbackLogReader(
            session_id=self.session_id,
            agent_name=self.agent_name,
            keep=published_exchange if writer is not None else None,
        )

        def records() -> Iterator[dict[str, Any]]:
            for line in lines:
                self._capture_request_events(line)
                exchange = reader.add_line(line)
                if exchange is not None and evidence is not None:
                    evidence.add(exchange)
                    yield exchange_record(exchange)

        imported = records()
        published = writer is not None
        if writer is not None:
            try:
                writer.reconcile_records(imported)
            except OSError as exc:
                published = False
                logger.warning("Could not publish %s: %s", writer.path, exc)
        for _ in imported:
            pass
        trajectory = reader.finish()
        if evidence is not None:
            trajectory._tool_evidence = evidence.items()
            self._live_reconciled = published
        self.trajectory = trajectory
        _warn_refused_requests(trajectory)

    def _import_callback_file(self, path: Path) -> None:
        with path.open() as lines:
            self._import_callback_lines(lines)


def _live_usage_gateway_conformance(process: LiteLLMProcess) -> LiveUsageGateway:
    """Static assertion: LiteLLMProcess satisfies the contract the rollout
    kernel's dashboard read binds to (``contracts.planes.LiveUsageGateway``).

    The kernel cannot import this module (#515 architecture invariant), so
    its ``server.live_usage_tokens()`` call is dynamically typed there; this
    return-type check is what makes a rename of the accessor — on either the
    protocol or the process side — fail ``ty`` instead of silently blanking
    the live token signal. Never called at runtime.
    """
    return process


class HostLiteLLMProcess(LiteLLMProcess):
    def __init__(
        self,
        *,
        route: LiteLLMRoute,
        process: subprocess.Popen[bytes] | ProxyProcess,
        runtime_dir: Path,
        endpoint: LiteLLMEndpoint,
        log_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        session_id: str,
        agent_name: str,
    ) -> None:
        self.route = route
        self.process = process
        self.runtime_dir = runtime_dir
        self.endpoint = endpoint
        self.log_path = log_path
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.session_id = session_id
        self.agent_name = agent_name
        self.trajectory: Trajectory | None = None

    @property
    def base_url(self) -> str:
        return self.endpoint.agent_base_url

    async def is_running(self) -> bool:
        return self.process.poll() is None

    async def stop(self) -> None:
        await self._stop_live_capture()
        await _await_log_stable(
            self._log_size, requested=self._requested, logged=self._all_logged
        )
        if self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, 10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait, 10)
        _HOST_PROXY_PROCESSES.pop(self.process, None)
        await _capture_to_thread(self._load_callback_log)
        await _capture_to_thread(self._reconcile_live_capture)
        with contextlib.suppress(Exception):
            shutil.rmtree(self.runtime_dir, ignore_errors=True)

    def _log_size(self) -> int:
        try:
            return self.log_path.stat().st_size
        except OSError:
            return -1

    def requests_in_flight(self) -> int | None:
        """Requests the proxy has taken and not yet logged: the lines of its
        request mark file less those of its logged mark file. None when they
        cannot be read."""
        counts = []
        for suffix in (_REQUEST_MARK_SUFFIX, _LOGGED_MARK_SUFFIX):
            try:
                counts.append(Path(f"{self.log_path}{suffix}").read_text().count("\n"))
            except FileNotFoundError:
                counts.append(0)
            except OSError:
                return None
        return max(0, counts[0] - counts[1])

    def _requested(self) -> bool:
        """Whether any request reached the proxy (the callback's mark file)."""
        return Path(f"{self.log_path}{_REQUEST_MARK_SUFFIX}").exists()

    def _all_logged(self) -> bool:
        """Whether every request that reached the proxy has its whole record
        in the callback log (:func:`_marks_all_logged`)."""
        try:
            requests = Path(f"{self.log_path}{_REQUEST_MARK_SUFFIX}").read_text()
            logged = Path(f"{self.log_path}{_LOGGED_MARK_SUFFIX}").read_text()
        except OSError:
            return False
        return _marks_all_logged(requests, logged)

    async def _read_callback_chunk(self, offset: int, limit: int) -> _CallbackChunk:
        def _read() -> _CallbackChunk:
            try:
                with self.log_path.open("rb") as handle:
                    size = os.fstat(handle.fileno()).st_size
                    handle.seek(offset)
                    return _CallbackChunk(handle.read(limit), size)
            except FileNotFoundError:
                # The proxy has not written its first record yet: nothing to
                # read, and nothing behind — not a failed read.
                return _CallbackChunk(b"", 0)
            except OSError:
                return _CallbackChunk(b"", None)

        return await asyncio.to_thread(_read)

    def _load_callback_log(self) -> None:
        if not self.log_path.exists():
            self.trajectory = Trajectory(
                session_id=self.session_id,
                agent_name=self.agent_name,
            )
            return
        with self.log_path.open() as lines:
            self._import_callback_lines(lines)

    def log_tail(self) -> str:
        chunks: list[str] = []
        for label, path in (("stdout", self.stdout_path), ("stderr", self.stderr_path)):
            with contextlib.suppress(Exception):
                text = path.read_text()[-4000:]
                if text.strip():
                    chunks.append(f"{label}: {text.strip()}")
        return "\n".join(chunks)


class SandboxLiteLLMProcess(LiteLLMProcess):
    def __init__(
        self,
        *,
        sandbox: Any,
        route: LiteLLMRoute,
        runtime_dir: str,
        endpoint: LiteLLMEndpoint,
        log_path: str,
        pid_path: str,
        stdout_path: str,
        stderr_path: str,
        session_id: str,
        agent_name: str,
    ) -> None:
        self.sandbox = sandbox
        self.route = route
        self.runtime_dir = runtime_dir
        self.endpoint = endpoint
        self.log_path = log_path
        self.pid_path = pid_path
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.session_id = session_id
        self.agent_name = agent_name
        self.trajectory: Trajectory | None = None
        # Whether the proxy ran from the sandbox's cached layer ("from layer")
        # or was installed in the trial, and why; the trial's config.json
        # gets it under ``layers.proxy``. None when no layer was asked for.
        self.layer_use: dict[str, Any] | None = None

    @property
    def base_url(self) -> str:
        return self.endpoint.agent_base_url

    async def is_running(self) -> bool:
        result = await exec_raw(
            self.sandbox.exec,
            (
                f"if [ -s {shlex.quote(self.pid_path)} ] && "
                f"kill -0 $(cat {shlex.quote(self.pid_path)}) 2>/dev/null; "
                "then echo yes; else echo no; fi"
            ),
            timeout_sec=5,
        )
        return result.return_code == 0 and (result.stdout or "").strip() == "yes"

    async def stop(self) -> None:
        await self._stop_live_capture()
        # The host proxy's settle rule: without the request/logged
        # marks, an agent that never called this proxy held every Daytona
        # trial's teardown for the whole 12 s deadline. One exec per poll
        # reads the log size and both marks; an unreadable probe keeps the
        # old wait.
        marks = _RemoteMarks(size=-1, requested=True, all_logged=False)

        async def _probe() -> int:
            nonlocal marks
            marks = await self._remote_marks()
            return marks.size

        await _await_log_stable(
            _probe, requested=lambda: marks.requested, logged=lambda: marks.all_logged
        )
        # Checked again at use: root kills the pid the file names and removes
        # the folder (FF-install-C2, C10).
        guard = _runtime_use_guard(self.runtime_dir, self.pid_path)
        with contextlib.suppress(Exception):
            await exec_raw(
                self.sandbox.exec,
                (
                    f"{guard} && if [ -s {shlex.quote(self.pid_path)} ]; then "
                    f"kill -TERM $(cat {shlex.quote(self.pid_path)}) 2>/dev/null || true; "
                    "fi"
                ),
                timeout_sec=10,
            )
        await self._load_callback_log()
        await _capture_to_thread(self._reconcile_live_capture)
        with contextlib.suppress(Exception):
            await exec_raw(
                self.sandbox.exec,
                f"{guard} && rm -rf {shlex.quote(self.runtime_dir)}",
                timeout_sec=10,
            )

    async def _remote_marks(self) -> _RemoteMarks:
        """The callback log's size, whether any request reached the proxy
        (``<log>.requests`` exists) and whether every one of them is whole in
        the log (``<log>.logged``, :func:`_marks_all_logged`), in one exec.
        ``base64 -w 0`` is already required by the chunk reads below."""
        log = shlex.quote(self.log_path)
        requests = shlex.quote(f"{self.log_path}{_REQUEST_MARK_SUFFIX}")
        logged = shlex.quote(f"{self.log_path}{_LOGGED_MARK_SUFFIX}")
        command = (
            f"stat -c %s {log} 2>/dev/null || echo -1; "
            f"if [ -e {requests} ]; then echo R; base64 -w 0 {requests} 2>/dev/null; echo; "
            f"base64 -w 0 {logged} 2>/dev/null; echo; else echo N; fi"
        )
        with contextlib.suppress(Exception):
            result = await exec_raw(self.sandbox.exec, command, timeout_sec=5)
            return _parse_remote_marks(result.stdout or "")
        return _RemoteMarks(size=-1, requested=True, all_logged=False)

    async def _remote_log_size(self) -> int:
        with contextlib.suppress(Exception):
            result = await exec_raw(
                self.sandbox.exec,
                f"stat -c %s {shlex.quote(self.log_path)} 2>/dev/null || echo -1",
                timeout_sec=5,
            )
            return int((result.stdout or "-1").strip() or -1)
        return -1

    async def _read_callback_chunk(self, offset: int, limit: int) -> _CallbackChunk:
        # One command per read, emitting ``<size>\n<base64 payload>``: the log
        # size rides along for free so the caller can tell "drained" from
        # "the read failed" and can quantify the lag without a second exec.
        #
        # The range is selected with iflag=skip_bytes,count_bytes so dd seeks
        # and then reads in 64KB blocks. The previous `bs=1 skip=<offset>`
        # form made dd issue two syscalls per byte transferred, which — on top
        # of a full transient-exec round trip per read — is what held the tail
        # to ~24KB per second-plus and let a fast-growing provider log outrun
        # it permanently.
        #
        # Userland floor: `base64 -w 0` on this same line already requires GNU
        # coreutils (BusyBox base64 has no -w), so the only new constraint is
        # the version — skip_bytes/count_bytes landed in coreutils 8.11 (2011).
        # An image that somehow lacks them now fails *visibly*: dd emits
        # nothing, the size line still arrives, and the read is classified as a
        # lag that escalates to a warning instead of freezing the counter.
        path = shlex.quote(self.log_path)
        command = (
            f"stat -c %s {path} 2>/dev/null || echo 0; "
            f"dd if={path} iflag=skip_bytes,count_bytes,fullblock "
            f"bs={_LIVE_CAPTURE_READ_BLOCK_BYTES} skip={offset} count={limit} "
            "2>/dev/null | base64 -w 0"
        )
        # Daytona otherwise retains one session wrapper shell for every
        # one-second live-capture poll. Long MAX rollouts can accumulate
        # thousands of orphaned shells and eventually wedge the agent's own
        # terminal calls. Providers without a transient-exec path keep the
        # existing behavior.
        execute = getattr(self.sandbox, "exec_transient", self.sandbox.exec)
        result = await exec_raw(
            execute, command, timeout_sec=_LIVE_CAPTURE_READ_TIMEOUT_SEC
        )
        if result.return_code != 0:
            return _CallbackChunk(b"", None)
        head, _, payload = (result.stdout or "").strip().partition("\n")
        try:
            size = int(head)
        except ValueError:
            return _CallbackChunk(b"", None)
        # Join before validating: the transport may re-wrap a long line, but a
        # payload that is genuinely not base64 must stay a failed read rather
        # than decode to silent garbage that would desync the byte offset.
        encoded = "".join(payload.split())
        if not encoded:
            return _CallbackChunk(b"", size)
        try:
            return _CallbackChunk(base64.b64decode(encoded, validate=True), size)
        except ValueError:
            return _CallbackChunk(b"", None)

    async def _load_callback_log(self) -> None:
        download_file = getattr(self.sandbox, "download_file", None)
        if download_file is not None:
            with tempfile.NamedTemporaryFile("r", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                await download_file(self.log_path, tmp_path)
                if tmp_path.stat().st_size:
                    # Read from the downloaded file line by line, never as
                    # one string (_import_callback_lines).
                    await _capture_to_thread(
                        lambda: self._import_callback_file(tmp_path)
                    )
                    return
            except Exception as exc:
                logger.debug("LiteLLM callback download failed: %s", exc)
            finally:
                tmp_path.unlink(missing_ok=True)
        text = ""
        result = await exec_raw(
            self.sandbox.exec,
            f"cat {shlex.quote(self.log_path)} 2>/dev/null || true",
            timeout_sec=15,
        )
        if result.return_code == 0:
            text = result.stdout or ""
        await _capture_to_thread(lambda: self._import_callback_lines(text.splitlines()))

    async def log_tail(self) -> str:
        chunks: list[str] = []
        for label, path in (("stdout", self.stdout_path), ("stderr", self.stderr_path)):
            with contextlib.suppress(Exception):
                result = await exec_raw(
                    self.sandbox.exec,
                    f"tail -c 4000 {shlex.quote(path)} 2>/dev/null || true",
                    timeout_sec=5,
                )
                text = (diagnostic_result(result).stdout.render() or "").strip()
                if text:
                    chunks.append(f"{label}: {text}")
        return "\n".join(chunks)


def needs_litellm_runtime(agent: str, model: str | None) -> bool:
    """True when an agent/model pair should be routed through LiteLLM.

    Never for an agent that routes its own model calls (``model_route =
    "agent"``): its model, if any, is a label.
    """
    return bool(routed_model(agent, model)) and not is_scripted_agent(agent)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _marks_all_logged(requests: str, logged: str) -> bool:
    """Whether every request that reached a proxy has its whole record in the
    callback log: each call id in the request marks is in the logged marks.
    A request without a call id ("?") never counts as logged, and a mark
    file still being written (no final newline) is not read as complete."""
    if not requests.endswith("\n") or not logged.endswith("\n"):
        return False
    wanted = set(requests.splitlines())
    return bool(wanted) and "?" not in wanted and wanted <= set(logged.splitlines())


@dataclass(frozen=True)
class _RemoteMarks:
    """One read of a sandbox proxy's callback log and its marks."""

    size: int
    requested: bool
    all_logged: bool


def _parse_remote_marks(stdout: str) -> _RemoteMarks:
    """Parse ``_remote_marks``'s output: the size line, then ``N`` (no request
    reached the proxy) or ``R`` with both mark files as base64. Anything else
    is read as "a request may be in flight" so the caller keeps waiting."""
    lines = stdout.split("\n")
    try:
        size = int(lines[0].strip() or -1)
    except ValueError:
        size = -1
    flag = lines[1].strip() if len(lines) > 1 else ""
    if flag == "N":
        return _RemoteMarks(size=size, requested=False, all_logged=False)
    if flag != "R" or len(lines) < 4:
        return _RemoteMarks(size=size, requested=True, all_logged=False)
    try:
        requests = base64.b64decode(lines[2].strip(), validate=True).decode("utf-8")
        logged = base64.b64decode(lines[3].strip(), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return _RemoteMarks(size=size, requested=True, all_logged=False)
    return _RemoteMarks(
        size=size, requested=True, all_logged=_marks_all_logged(requests, logged)
    )


async def _await_log_stable(
    get_size: Callable[[], int | Awaitable[int]],
    *,
    deadline_s: float = 12.0,
    quiet_s: float = 0.5,
    requested: Callable[[], bool] | None = None,
    logged: Callable[[], bool] | None = None,
) -> None:
    """Wait until the callback JSONL stops growing.

    LiteLLM invokes its CustomLogger hooks fire-and-forget (``asyncio.create_task``)
    *after* returning the provider response, so the final — often largest —
    exchange may still be mid-append when BenchFlow tears the proxy down. A fixed
    sleep either truncates that record (undercounting usage while silently passing
    the ``required`` gate) or wastes time. Poll the log size until it has been
    stable for ``quiet_s`` (the final record landed) or ``deadline_s`` elapses.

    An empty log waits for its first record, unless ``requested`` (the
    callback's per-request mark, written before any upstream call) says no
    request ever reached the proxy: then nothing can still be on its way,
    and an empty log quiet for ``quiet_s`` is final. An agent that talks to
    its model past the proxy, or makes no call, otherwise held every
    trial's teardown for the whole deadline (12 s of run-08's 13.9 s).

    ``logged`` is the signal for the common case: every request that
    reached the proxy has its whole record (the callback marks the call id
    after writing it), so the log is final at once, with no quiet window
    (0.6 s or more of every real-agent trial's teardown, PERF-B).
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    last_size = -2
    stable_since: float | None = None
    while loop.time() - start < deadline_s:
        if logged is not None and logged():
            return
        raw: Any = get_size()
        if inspect.isawaitable(raw):
            raw = await raw
        size = cast(int, raw)
        settled = size > 0 or (requested is not None and not requested())
        if settled and size == last_size:
            if stable_since is None:
                stable_since = loop.time()
            elif loop.time() - stable_since >= quiet_s:
                return
        else:
            stable_since = None
        last_size = size
        await asyncio.sleep(0.2)


def _host_litellm_executable() -> str:
    sibling = Path(sys.executable).with_name("litellm")
    if sibling.exists():
        return str(sibling)
    found = shutil.which("litellm")
    if found:
        return found
    raise RuntimeError(
        f"LiteLLM CLI is not installed. Install {LITELLM_VERSION_SPEC} in this environment."
    )


async def _host_litellm_for(
    install: ProxyInstall, *, venv_root: Path | None = None
) -> str:
    """The host proxy's LiteLLM CLI for one install.

    With no extras it is the LiteLLM installed with BenchFlow (its pinned
    core dependency). Extras never go into BenchFlow's own environment: the
    proxy then runs from a venv of the locked, hashed set plus the extras
    under it (``proxy_requirements.ensure_host_venv``), fresh for this proxy.
    """
    if not install.extra_requirements:
        return _host_litellm_executable()
    await asyncio.to_thread(install.check)
    venv = await asyncio.to_thread(ensure_host_venv, install, root=venv_root)
    return str(venv / "bin" / "litellm")


def _docker_host_address() -> str:
    import platform

    if platform.system().lower() != "linux":
        return "host.docker.internal"
    try:
        out = subprocess.check_output(
            [
                "docker",
                "network",
                "inspect",
                "bridge",
                "--format",
                "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
            ],
            text=True,
            timeout=10,
        ).strip()
        if out:
            return out
    except Exception:
        logger.debug("Could not detect Docker bridge gateway", exc_info=True)
    return "host.docker.internal"


def _address_is_local(address: str) -> bool:
    """True when this host can actually bind the address.

    Under Docker Desktop's WSL2 backend the bridge gateway (172.17.0.1) lives in
    the Docker VM, not in the WSL distro, so it resolves but cannot be bound.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((address, 0))
            return True
        except OSError:
            return False


def _host_bind_address(environment: str) -> str:
    """Interface the host LiteLLM proxy binds to.

    Local runs need no off-box reachability -> loopback only. Docker runs must be
    reachable from the agent container, so bind the concrete bridge-gateway IP
    (Linux) — reachable from containers but not the public network — falling back
    to 0.0.0.0 only when the gateway resolves to a hostname (e.g.
    host.docker.internal on macOS) that cannot be bound directly.
    """
    if environment != "docker":
        return "127.0.0.1"
    address = _docker_host_address()
    try:
        socket.inet_aton(address)
    except OSError:
        return "0.0.0.0"
    # A routable-looking gateway that this host cannot bind means the docker
    # daemon lives elsewhere (Docker Desktop's WSL2 / Hyper-V VM). Fall back to
    # the same path macOS uses.
    if not _address_is_local(address):
        return "0.0.0.0"
    return address


def _agent_endpoint_for_environment(
    port: int, environment: str, bind: str
) -> LiteLLMEndpoint:
    if environment == "docker":
        if bind != "0.0.0.0":
            agent_host = bind
        else:
            detected = _docker_host_address()
            # If we fell back to 0.0.0.0 because the gateway was unbindable,
            # the container cannot reach us on it either - use the name Docker
            # Desktop publishes for the host.
            agent_host = (
                detected if _address_is_local(detected) else "host.docker.internal"
            )
        health_host = "127.0.0.1" if bind == "0.0.0.0" else bind
    else:
        agent_host = health_host = "127.0.0.1"
    return LiteLLMEndpoint(
        agent_base_url=f"http://{agent_host}:{port}",
        local_base_url=f"http://{health_host}:{port}",
    )


def _route_env(route: LiteLLMRoute) -> dict[str, str]:
    """Route facts exported to the proxy process for token capture."""
    return {
        _ROUTE_PROVIDER_ENV: route.provider_name,
        _UPSTREAM_MODEL_ENV: route.upstream_model,
    }


def _native_passthrough_env(route: LiteLLMRoute, env: dict[str, str]) -> dict[str, str]:
    """Make LiteLLM's native Gemini path use the same resolved deployment.

    The vendor pass-through reads environment variables instead of the model
    deployment's api_base/api_key. Overwrite both from that deployment so an
    inherited value cannot silently send traffic to a different endpoint.
    """
    if route.upstream_model.startswith("vertex_ai/"):
        # Native clients still speak AI Studio paths and use only the gateway
        # key. The boundary maps those paths into this controller-owned Vertex
        # resource; LiteLLM performs ADC authentication inside the private proxy.
        return {
            "DEFAULT_VERTEXAI_PROJECT": str(
                route.litellm_params.get("vertex_project", "")
            ),
            "DEFAULT_VERTEXAI_LOCATION": str(
                route.litellm_params.get("vertex_location") or "us-central1"
            ),
            "DEFAULT_GOOGLE_APPLICATION_CREDENTIALS": env.get(
                "GOOGLE_APPLICATION_CREDENTIALS", ""
            ),
            "BENCHFLOW_LITELLM_VERTEX_NATIVE": json.dumps(
                {
                    "model": route.upstream_model.removeprefix("vertex_ai/"),
                    "project": route.litellm_params.get("vertex_project", ""),
                    "location": route.litellm_params.get("vertex_location")
                    or "us-central1",
                }
            ),
        }
    if not route.upstream_model.startswith("gemini/"):
        return {}
    api_key = str(route.litellm_params.get("api_key", ""))
    if api_key.startswith("os.environ/"):
        api_key = env.get(api_key.removeprefix("os.environ/"), "")
    api_base = str(
        route.litellm_params.get("api_base")
        or "https://generativelanguage.googleapis.com"
    )
    # FF-ACP-CONF-1: translation appends /models/... to a versioned base,
    # while native passthrough appends the client's /v1[beta]/models/... .
    # Only exact official forms are normalized; custom gateway prefixes keep
    # their existing meaning under explicit operator admission.
    from benchflow.providers.restricted_models import OFFICIAL_ENDPOINTS

    if api_base.rstrip("/") in OFFICIAL_ENDPOINTS["gemini"]:
        api_base = "https://generativelanguage.googleapis.com"
    return {
        "GEMINI_API_BASE": api_base,
        "GEMINI_API_KEY": api_key,
        "GOOGLE_API_KEY": api_key,
    }


def _served_models_env(config: dict[str, object]) -> dict[str, str]:
    """Every model name one proxy config routes, for the callback's model gate.

    Read from the final config, so a model another layer registers (a
    companion served next to the run's model) is served too.
    """
    names: list[str] = []
    model_list = config.get("model_list")
    for entry in model_list if isinstance(model_list, list) else []:
        if not isinstance(entry, Mapping):
            continue
        name = cast(Mapping[str, Any], entry).get("model_name")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    names += [m for m in _config_upstream_models(config) if m not in names]
    bridges = []
    for entry in model_list if isinstance(model_list, list) else []:
        if not isinstance(entry, Mapping):
            continue
        item = cast(Mapping[str, Any], entry)
        params = item.get("litellm_params")
        name = item.get("model_name")
        if (
            isinstance(params, Mapping)
            and str(cast(Mapping[str, Any], params).get("model", "")).startswith(
                "openai/chat_completions/"
            )
            and isinstance(name, str)
        ):
            bridges.append(name)
    return {
        _SERVED_MODELS_ENV: json.dumps(names),
        "BENCHFLOW_LITELLM_CHAT_BRIDGE_MODELS": json.dumps(bridges),
    }


def _config_upstream_models(config: dict[str, object]) -> list[str]:
    """Upstream model ids (``litellm_params.model``) served by one proxy config."""
    models: list[str] = []
    model_list = config.get("model_list")
    for entry in model_list if isinstance(model_list, list) else []:
        if not isinstance(entry, Mapping):
            continue
        params = cast(Mapping[str, Any], entry).get("litellm_params")
        if not isinstance(params, Mapping):
            continue
        model = cast(Mapping[str, Any], params).get("model")
        if isinstance(model, str) and model and model not in models:
            models.append(model)
    return models


def _sitecustomize_source(
    config: dict[str, object], *, lazy_start: bool = False
) -> str:
    """Proxy-process startup: apply the LiteLLM patches for this route.

    Registering the route's Claude effort capabilities here, before LiteLLM
    parses the config, keeps ``output_config.effort`` on the wire for Claude
    ids missing from LiteLLM's model map (e.g. ``claude-fable-5-1``).
    ``lazy_start`` first defers the proxy's route analysis and model builds
    to first use (``litellm_lazy_start``), before anything imports LiteLLM.
    """
    models = _config_upstream_models(config)
    lazy = (
        f"import {_LAZY_START_MODULE}\n{_LAZY_START_MODULE}.install()\n"
        if lazy_start
        else ""
    )
    # LiteLLM's proxy modules add folders found from the working folder to
    # the import path when they are imported (proxy_server: its "../..",
    # proxy_cli: the folder itself). The import path is put back once they
    # are imported, so a cold proxy and a forked one (whose parent did the
    # same) import from the runtime folder and installed packages only
    # .
    return (
        lazy
        + "import sys as _bf_sys\n"
        + "_bf_path = list(_bf_sys.path)\n"
        + f"import {_PATCH_MODULE}\n"
        f"import {_GEMINI_PATCH_MODULE}\n"
        f"import {_TOKEN_CAPTURE_PATCH_MODULE}\n"
        f"import {_PROVIDER_RETRY_PATCH_MODULE}\n"
        f"{_PATCH_MODULE}.register_claude_effort_capabilities({models!r})\n"
        "try:\n"
        f"    import {_REQUEST_GUARD_MODULE}\n"
        f"    {_REQUEST_GUARD_MODULE}.install()\n"
        "except Exception as exc:\n"
        "    raise SystemExit('BenchFlow proxy request boundary failed') from exc\n"
        "import litellm.proxy.proxy_cli\n"
        "_bf_sys.path[:] = _bf_path\n"
        "del _bf_sys, _bf_path\n"
    )


def _write_runtime_files(
    runtime_dir: Path,
    *,
    config: dict[str, object],
    lazy_start: bool = False,
) -> tuple[Path, Path, Path]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    callback_path = runtime_dir / f"{_CALLBACK_MODULE}.py"
    patch_path = runtime_dir / f"{_PATCH_MODULE}.py"
    gemini_patch_path = runtime_dir / f"{_GEMINI_PATCH_MODULE}.py"
    sitecustomize_path = runtime_dir / "sitecustomize.py"
    config_path = runtime_dir / "config.yaml"
    callback_path.write_text(callback_module_source())
    patch_source = Path(__file__).with_name("litellm_bedrock_patch.py").read_text()
    patch_path.write_text(patch_source)
    gemini_patch_path.write_text(
        Path(__file__).with_name("litellm_gemini_passthrough_patch.py").read_text()
    )
    (runtime_dir / f"{_TOKEN_CAPTURE_PATCH_MODULE}.py").write_text(
        Path(__file__).with_name("litellm_token_capture_patch.py").read_text()
    )
    (runtime_dir / f"{_PROVIDER_RETRY_PATCH_MODULE}.py").write_text(
        Path(__file__).with_name("litellm_provider_retry_patch.py").read_text()
    )
    (runtime_dir / f"{_REQUEST_GUARD_MODULE}.py").write_text(
        Path(__file__).with_name("litellm_request_guard.py").read_text()
    )
    if lazy_start:
        (runtime_dir / f"{_LAZY_START_MODULE}.py").write_text(
            Path(__file__).with_name("litellm_lazy_start.py").read_text()
        )
    sitecustomize_path.write_text(_sitecustomize_source(config, lazy_start=lazy_start))
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path, callback_path, patch_path


# How long to wait for the *host* per-run LiteLLM proxy to become healthy.
# litellm's cold start runs tens of seconds, and when many runs launch in parallel
# (max-parallel sweeps) the proxies cold-start simultaneously and contend for CPU,
# pushing well past a fixed ~30s budget — the old ``range(120)`` x 0.25s. That
# surfaced as "LiteLLM did not become healthy: All connection attempts failed" for
# otherwise-fine agents on Docker under load. Use a generous, time-based, env-tunable
# deadline so a slow-but-fine proxy is not killed; a genuine crash still fails fast
# via the process-exit check below.
#
# Scope: this deadline covers host and sandbox LiteLLM readiness. The sandbox
# pollers also use it as their overall deadline; a single slow Daytona
# ``sandbox.exec`` probe is treated as one failed attempt, not a fatal startup
# failure.
def _health_deadline_sec() -> float:
    # Parse defensively: a malformed BENCHFLOW_LITELLM_HEALTH_TIMEOUT_SEC must not
    # crash ``import benchflow``. Floor at one poll cycle so a 0/negative value still
    # checks process liveness and attempts health once instead of raising immediately.
    try:
        deadline = float(os.environ.get("BENCHFLOW_LITELLM_HEALTH_TIMEOUT_SEC", "180"))
    except (TypeError, ValueError):
        deadline = 180.0
    return max(deadline, 1.0)


_HEALTH_DEADLINE_SEC = _health_deadline_sec()


async def _poll_host_health(
    process: HostLiteLLMProcess, deadline_s: float = _HEALTH_DEADLINE_SEC
) -> None:
    last_error = ""
    loop = asyncio.get_running_loop()
    start = loop.time()
    # One client for every poll: making one (its TLS context) costs about
    # 10 ms of the event loop, and the zygote's proxy is polled every 5 ms
    # while the sandbox starts beside it (SPEED-FIX).
    async with httpx.AsyncClient(timeout=2.0) as client:
        while loop.time() - start < deadline_s:
            if process.process.poll() is not None:
                raise RuntimeError(
                    "LiteLLM exited before becoming healthy.\n" + process.log_tail()
                )
            try:
                for path in ("/health/liveliness", "/health"):
                    response = await client.get(process.endpoint.local_base_url + path)
                    if response.status_code < 500:
                        return
            except Exception as exc:
                last_error = str(exc)
            from benchflow._proxy_zygote import ProxyProcess

            await asyncio.sleep(
                0.005 if isinstance(process.process, ProxyProcess) else 0.25
            )
    raise RuntimeError(
        f"LiteLLM did not become healthy after {deadline_s:.0f}s: "
        f"{last_error}\n{process.log_tail()}"
    )


def _proxy_process_env(source: dict[str, str], runtime_dir: str) -> dict[str, str]:
    """Privileged imports use private code and installed packages only.

    The proxy reads no ``.env`` file: BenchFlow passes every value it needs
    here, and python-dotenv's own switch stops LiteLLM's import-time
    ``load_dotenv()`` calls from searching.
    """
    from benchflow._proxy_zygote import DOTENV_OFF

    env = {
        key: value
        for key, value in source.items()
        if key not in SYSTEM_SHELL_UNSET_VARS and not key.startswith(("PYTHON", "LD_"))
    }
    env.update(
        PYTHONPATH=runtime_dir,
        PYTHONSAFEPATH="1",
        PYTHONNOUSERSITE="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        **DOTENV_OFF,
    )
    return env


def _daemon_proxy(
    *, env: dict[str, str], runtime_dir: Path, args: list[str], executable: str
) -> ProxyProcess | None:
    """Use the daemon's audited parent only for the installed core proxy.

    Extra-requirement venvs keep the existing cold executable. An unavailable
    parent falls back before launch, never after a launch might have happened.
    """
    if os.environ.get("_BENCHFLOW_DAEMON_WORKER") != "1":
        return None
    if Path(executable).parent != Path(sys.executable).parent:
        return None
    from benchflow._daemon import request
    from benchflow._daemon_wire import connect, receive, send
    from benchflow._proxy_zygote import ProxyProcess

    try:
        path = request({"action": "zygote", "env": env}, timeout=90)["path"]
        if not path:
            return None
        conn = connect(Path(path))
    except (OSError, EOFError, KeyError, ValueError):
        return None
    with conn:
        send(
            conn,
            {"action": "launch", "env": env, "cwd": str(runtime_dir), "args": args},
        )
        # From here a proxy may already be bound to this trial's port: a
        # lost answer fails the start instead of launching a second one.
        try:
            response = receive(conn)
        except (OSError, EOFError, ValueError) as exc:
            raise RuntimeError(
                f"The local runtime's proxy parent did not answer: {exc}"
            ) from exc
        if "refused" in response:
            logger.debug("Proxy parent refused this launch: %s", response["refused"])
            return None
        return ProxyProcess(Path(path), response["pid"])


async def _start_host_litellm(
    *,
    route: LiteLLMRoute,
    master_key: str,
    agent_env: dict[str, str],
    environment: str,
    session_id: str,
    agent_name: str,
) -> HostLiteLLMProcess:
    runtime_dir = Path(tempfile.mkdtemp(prefix="benchflow-litellm-"))
    # A fresh extras venv belongs to this process, including on failed start.
    # Build it before writing config, which holds the master key.
    try:
        litellm_executable = await _host_litellm_for(
            proxy_install(route.provider_name), venv_root=runtime_dir
        )
    except BaseException:
        shutil.rmtree(runtime_dir, ignore_errors=True)
        raise
    log_path = runtime_dir / "callback.jsonl"
    stdout_path = runtime_dir / "stdout.log"
    stderr_path = runtime_dir / "stderr.log"
    port = _find_free_port()
    bind = await asyncio.to_thread(_host_bind_address, environment)
    config = litellm_proxy_config(route, master_key=master_key)
    config_path, _, _ = _write_runtime_files(
        runtime_dir, config=config, lazy_start=_proxy_lazy_start()
    )
    env = dict(os.environ)
    env.update(agent_env)
    if route.upstream_model.startswith("vertex_ai/"):
        adc_json = env.pop("GOOGLE_APPLICATION_CREDENTIALS_JSON", None)
        if adc_json:
            adc_path = runtime_dir / "application_default_credentials.json"
            adc_path.write_text(adc_json)
            adc_path.chmod(0o600)
            env["GOOGLE_APPLICATION_CREDENTIALS"] = str(adc_path)
    env = _proxy_process_env(env, str(runtime_dir))
    env.update(
        {
            "PYTHONPATH": str(runtime_dir),
            "PYTHONSAFEPATH": "1",
            "PYTHONNOUSERSITE": "1",
            "LITELLM_MASTER_KEY": master_key,
            "BENCHFLOW_LITELLM_LOG_PATH": str(log_path),
            **_route_env(route),
            **_native_passthrough_env(route, env),
            **_served_models_env(config),
            **_PROXY_DOCS_DISABLE_ENV,
        }
    )
    stdout = stdout_path.open("ab")
    stderr = stderr_path.open("ab")
    try:
        process = _daemon_proxy(
            env=env,
            runtime_dir=runtime_dir,
            args=["--config", str(config_path), "--host", bind, "--port", str(port)],
            executable=litellm_executable,
        ) or subprocess.Popen(
            [
                litellm_executable,
                "--config",
                str(config_path),
                "--host",
                bind,
                "--port",
                str(port),
            ],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=env,
            cwd=runtime_dir,
        )
    finally:
        stdout.close()
        stderr.close()
    _HOST_PROXY_PROCESSES[process] = runtime_dir
    if host_proxy_ahead():
        _prewarm_litellm_import()
    runner = HostLiteLLMProcess(
        route=route,
        process=process,
        runtime_dir=runtime_dir,
        endpoint=_agent_endpoint_for_environment(port, environment, bind),
        log_path=log_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        session_id=session_id,
        agent_name=agent_name,
    )
    try:
        with _timing_spans.span(_timing_spans.PROXY_HEALTH_WAIT):
            await _poll_host_health(runner)
        if route_requires_bedrock_patch(route):
            # Fail closed before the agent launches when the Bedrock 4.8+
            # thinking patch did not activate (#602).
            await asyncio.to_thread(
                preflight_host_bedrock_patch,
                env=env,
                litellm_executable=litellm_executable,
                effort_models=(route.upstream_model,),
            )
    except BaseException:
        # A proxy that never became healthy still holds provider credentials and
        # an on-disk config with the master key — tear it down, don't leak it.
        with contextlib.suppress(Exception):
            if process.poll() is None:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    await asyncio.to_thread(process.wait, 5)
                if process.poll() is None:
                    process.kill()
        with contextlib.suppress(Exception):
            shutil.rmtree(runtime_dir, ignore_errors=True)
        _HOST_PROXY_PROCESSES.pop(process, None)
        raise
    logger.info("LiteLLM proxy listening on %s", runner.base_url)
    return runner


# Every host proxy this process started and has not stopped, with its runtime
# folder (its config holds the master key). At exit, whatever is left (a pool
# proxy no trial took, a trial's proxy when the job died) is stopped and its
# folder removed: nothing in this process can use it any more.
_HOST_PROXY_PROCESSES: dict[subprocess.Popen[bytes] | ProxyProcess, Path] = {}
# A forked child inherits the registry but not the proxies: only the process
# that started them stops them.
_HOST_PROXY_OWNER_PID = os.getpid()


def _stop_host_proxies_at_exit() -> None:
    if os.getpid() != _HOST_PROXY_OWNER_PID:
        return
    left = list(_HOST_PROXY_PROCESSES.items())
    _HOST_PROXY_PROCESSES.clear()
    for process, _ in left:
        with contextlib.suppress(Exception):
            if process.poll() is None:
                process.terminate()
    for process, runtime_dir in left:
        with contextlib.suppress(Exception):
            try:
                process.wait(5)
            except subprocess.TimeoutExpired:
                process.kill()
        with contextlib.suppress(Exception):
            shutil.rmtree(runtime_dir, ignore_errors=True)


atexit.register(_stop_host_proxies_at_exit)


_litellm_import_started = False


def _prewarm_litellm_import() -> None:
    """Import litellm in a thread of this process while the proxy starts.

    The agent's model metadata (_provider_models_for_proxy_alias) imports
    litellm here right after the health wait: about 1.9 s per trial on a
    build VM, with nothing else running. Started with the proxy, the import
    overlaps its health wait; the later import waits on the module lock for
    what is left and gets the same module. Once per process.
    """
    global _litellm_import_started
    if _litellm_import_started or "litellm" in sys.modules:
        return
    _litellm_import_started = True

    def _import() -> None:
        with contextlib.suppress(Exception):
            importlib.import_module("litellm")

    threading.Thread(target=_import, name="litellm-import", daemon=True).start()


# Warm host proxies. A host proxy is a LiteLLM process of its own for each
# trial (its master key, port, callback log and runtime folder), and starting
# one takes about 4 s (LiteLLM's import and server start, then the health
# wait) of every trial's setup. The job's pool starts it ahead of time: a
# trial asks for one when its sandbox starts (reserve_host_proxy) and takes
# it when its agent connects (ensure_litellm_runtime), so the start overlaps
# the sandbox's; BENCHFLOW_HOST_PROXY_WARM keeps more ready for the job's next
# trials. A proxy is taken by one trial only, and only from exactly the
# launch the trial would start (_host_proxy_key); from then on it is that
# trial's proxy, stopped with it like one it started.

#: ``0`` restores the host proxy's start as it was before the pool: each
#: trial starts its proxy when its agent connects, the sandbox guards run
#: after that start instead of beside it (rollout._guards_beside_proxy), and
#: this process imports litellm after the health wait, not during it.
HOST_PROXY_AHEAD_ENV = "BENCHFLOW_HOST_PROXY_AHEAD"
#: How many started host proxies a job keeps ready per launch beyond the
#: trials that asked for one (default none: every trial asks for its own
#: when its sandbox starts, which already overlaps that start, and a spare
#: no trial takes only costs a process and its stop at the job's end).
HOST_PROXY_WARM_ENV = "BENCHFLOW_HOST_PROXY_WARM"
# Never more started proxies waiting for one launch than this.
_HOST_PROXY_POOL_CAP = 64


def host_proxy_ahead() -> bool:
    return os.environ.get(HOST_PROXY_AHEAD_ENV, "").strip() != "0"


def _proxy_lazy_start() -> bool:
    return os.environ.get(PROXY_LAZY_START_ENV, "").strip() != "0"


def _host_proxy_warm() -> int:
    raw = os.environ.get(HOST_PROXY_WARM_ENV, "").strip()
    try:
        return max(int(raw), 0) if raw else 0
    except ValueError:
        return 0


def _host_proxy_env(
    agent: str,
    agent_env: dict[str, str],
    required_skill_names: tuple[str, ...],
    trial_budget: TrialBudget | None = None,
) -> dict[str, str]:
    """The proxy env ensure_litellm_runtime starts a trial's first host proxy
    with when web tools are allowed (a run that disallows them gets the
    sandbox proxy)."""
    env = _litellm_proxy_env(
        agent=agent, agent_env=agent_env, required_skill_names=required_skill_names
    )
    if trial_budget is not None:
        env.update(trial_budget.proxy_env())
    env["BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS"] = "0"
    return env


# Set in this process by an import, not by anyone's choice: litellm's import
# (the prewarm thread runs it as the first proxy starts) sets
# TIKTOKEN_CACHE_DIR to its own bundled folder, and the proxy's own import of
# the same litellm sets the same value there.
_HOST_PROXY_KEY_IGNORED_ENV = frozenset({"TIKTOKEN_CACHE_DIR"})


def _host_proxy_fields(
    *,
    route: LiteLLMRoute,
    agent: str,
    environment: str,
    proxy_env: Mapping[str, str],
) -> dict[str, str]:
    """What a host proxy is started from, but its own key, port and folder,
    one digest per part: the route and its config, the agent, the
    environment, and each variable of the process environment (provider
    keys included) _start_host_litellm builds on."""

    def digest(value: object) -> str:
        blob = json.dumps(value, sort_keys=True, default=repr).encode()
        return hashlib.sha256(blob).hexdigest()

    fields = {
        "agent": digest(agent),
        "environment": digest(environment),
        "route": digest(route.config_key),
        "config": digest(litellm_proxy_config(route, master_key="")),
    }
    for name, value in {**os.environ, **proxy_env}.items():
        if name not in _HOST_PROXY_KEY_IGNORED_ENV:
            fields[f"env {name}"] = digest(value)
    return fields


def _host_proxy_key(fields: Mapping[str, str]) -> str:
    """Two launches with the same key start the same proxy."""
    blob = json.dumps(sorted(fields.items())).encode()
    return hashlib.sha256(blob).hexdigest()


def _host_proxy_spawner(
    *,
    route: LiteLLMRoute,
    agent: str,
    environment: str,
    proxy_env: dict[str, str],
) -> Callable[[], Awaitable[tuple[Any, str]]]:
    """Start one pool proxy for this launch, with a master key of its own."""
    env = dict(proxy_env)

    async def spawn() -> tuple[Any, str]:
        master_key = f"sk-runtime-{secrets.token_urlsafe(24)}"
        server = await _start_host_litellm(
            route=route,
            master_key=master_key,
            agent_env=dict(env),
            environment=environment,
            session_id="",
            agent_name=agent,
        )
        return server, master_key

    return spawn


class _HostProxyPool:
    """Started host proxies waiting for the trial that takes one.

    One per process, on the event loop that first used it (another loop's
    trials start their own proxies). ``reserve`` counts a trial that will
    want a proxy of a launch and starts one for it; ``claim`` takes one (a
    started one first, else the one that started first) and starts the
    next; ``release`` gives back a reservation no claim answered. Started
    proxies no trial wants beyond the warm count are stopped.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._spares: dict[str, list[asyncio.Task[tuple[Any, str]]]] = {}
        self._spawners: dict[str, Callable[[], Awaitable[tuple[Any, str]]]] = {}
        self._wanted: Counter[str] = Counter()
        # A launch a trial asked for at its sandbox's start, to the one its
        # agent then connected with (an install changed the agent's env in
        # between): the job's next trials ask for that one instead.
        self._redirect: dict[str, str] = {}
        # Each reserved launch's parts (_host_proxy_fields), to say what
        # changed when a trial's connection did not take its reservation.
        self._fields: dict[str, dict[str, str]] = {}
        # Stops of proxies no trial wants, kept until they finish.
        self._stopping: set[asyncio.Task[None]] = set()

    def _here(self) -> bool:
        loop = asyncio.get_running_loop()
        if self._loop is None or self._loop.is_closed():
            self._loop = loop
            self._forget()
        return self._loop is loop

    def _forget(self) -> None:
        self._spares.clear()
        self._spawners.clear()
        self._wanted.clear()
        self._redirect.clear()
        self._fields.clear()

    def reserve(
        self,
        key: str,
        spawner: Callable[[], Awaitable[tuple[Any, str]]],
        fields: Mapping[str, str] | None = None,
    ) -> str | None:
        if not self._here():
            return None
        if fields is not None:
            self._fields.setdefault(key, dict(fields))
        key = self._redirect.get(key, key)
        self._spawners.setdefault(key, spawner)
        self._wanted[key] += 1
        self._fill(key, 0)
        return key

    def _changed(self, reserved: str, fields: Mapping[str, str] | None) -> str:
        """The names (never the values) of the launch parts that differ."""
        before = self._fields.get(reserved)
        if before is None or fields is None:
            return "unknown"
        names = sorted(n for n in {*before, *fields} if before.get(n) != fields.get(n))
        return ", ".join(names) or "nothing"

    def release(self, key: str) -> None:
        if not self._here():
            return
        if self._wanted[key] > 0:
            self._wanted[key] -= 1
        tasks = self._spares.get(key, [])
        while len(tasks) > self._wanted[key] + _host_proxy_warm():
            self._retire(tasks.pop())

    async def claim(
        self,
        key: str,
        spawner: Callable[[], Awaitable[tuple[Any, str]]],
        *,
        reserved: str,
        fields: Mapping[str, str] | None = None,
    ) -> tuple[Any, str] | None:
        if not self._here():
            return None
        if reserved != key and self._redirect.get(reserved) != key:
            logger.debug(
                "The warm host proxy is not for this connection: its launch "
                "changed since the trial asked for it (%s); the job's next "
                "trials ask for this one.",
                self._changed(reserved, fields),
            )
        if reserved != key:
            self._redirect[reserved] = key
        if fields is not None:
            self._fields.setdefault(key, dict(fields))
        self._spawners[key] = spawner
        try:
            return await self._take(key)
        finally:
            # Only now: the trial's own proxy is out of the pool, and what is
            # left beyond the other trials' asks (and the warm count) goes.
            self.release(reserved)
            self._fill(key, _host_proxy_warm())

    async def _take(self, key: str) -> tuple[Any, str] | None:
        tasks = self._spares.get(key, [])
        while tasks:
            task = next((t for t in tasks if t.done()), tasks[0])
            tasks.remove(task)
            try:
                server, master_key = await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    continue
                # This trial was cancelled, not the start: give it back.
                tasks.append(task)
                raise
            except Exception:
                continue
            if await server.is_running():
                return server, master_key
            self._retire(task)
        return None

    def _fill(self, key: str, extra: int) -> None:
        spawner = self._spawners.get(key)
        loop = self._loop
        if spawner is None or loop is None:
            return
        tasks = self._spares.setdefault(key, [])
        tasks[:] = [t for t in tasks if not _failed(t)]
        want = min(self._wanted[key] + extra, _HOST_PROXY_POOL_CAP)
        while len(tasks) < want:
            # Its own context: the start records no span into the trial that
            # happened to ask (the trial that takes it records its wait).
            tasks.append(
                loop.create_task(
                    _spawn_pool_proxy(spawner), context=contextvars.Context()
                )
            )

    def _retire(self, task: asyncio.Task[tuple[Any, str]]) -> None:
        if not task.done():
            # _start_host_litellm stops the process and removes its folder.
            task.cancel()
        elif not _failed(task) and self._loop is not None:
            server, _ = task.result()
            stop = self._loop.create_task(_stop_pool_proxy(server))
            self._stopping.add(stop)
            stop.add_done_callback(self._stopping.discard)

    async def drain(self) -> None:
        """Stop every proxy no trial took (the job's end)."""
        if self._loop is None or not self._here():
            return
        tasks = [t for ts in self._spares.values() for t in ts]
        self._forget()
        for task in tasks:
            if not task.done():
                task.cancel()
        started = await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(
            *(_stop_pool_proxy(s[0]) for s in started if isinstance(s, tuple)),
            *self._stopping,
            return_exceptions=True,
        )


def _failed(task: asyncio.Task[Any]) -> bool:
    return task.done() and (task.cancelled() or task.exception() is not None)


async def _spawn_pool_proxy(
    spawner: Callable[[], Awaitable[tuple[Any, str]]],
) -> tuple[Any, str]:
    try:
        return await spawner()
    except Exception as exc:
        # The trial that would have taken it starts its own proxy instead,
        # and reports why that one fails, if it does.
        logger.debug("A warm host proxy did not start: %s", describe_exception(exc))
        raise


async def _stop_pool_proxy(server: Any) -> None:
    """Stop a proxy no trial took. It never served a request, so there is no
    callback record to wait for or import: kill it and remove its folder
    (which holds its master key)."""
    process = getattr(server, "process", None)
    runtime_dir = getattr(server, "runtime_dir", None)
    if not isinstance(process, subprocess.Popen) or runtime_dir is None:
        with contextlib.suppress(Exception):
            await server.stop()
        return
    with contextlib.suppress(Exception):
        if process.poll() is None:
            process.kill()
        await asyncio.to_thread(process.wait, 10)
    _HOST_PROXY_PROCESSES.pop(process, None)
    with contextlib.suppress(Exception):
        shutil.rmtree(runtime_dir, ignore_errors=True)


_HOST_PROXY_POOL = _HostProxyPool()


@dataclass
class HostProxyReservation:
    """A trial's ask for a warm host proxy (``reserve_host_proxy``)."""

    key: str
    answered: bool = False


# The reservation of the trial whose agent is connecting (set around the
# rollout's ensure_litellm_runtime call); the claim marks it answered.
_HOST_PROXY_RESERVATION: contextvars.ContextVar[HostProxyReservation | None] = (
    contextvars.ContextVar("benchflow_host_proxy_reservation", default=None)
)


def reserve_host_proxy(
    *,
    agent: str,
    agent_env: dict[str, str],
    model: str | None,
    environment: str,
    required_skill_names: tuple[str, ...] = (),
    force_sandbox_local: bool = False,
    trial_budget: TrialBudget | None = None,
) -> HostProxyReservation | None:
    """Start, while the trial's sandbox starts, the host proxy its agent will
    connect through (see ``_HostProxyPool``); None when this connection
    would not start one (a sandbox proxy, no proxy, a key of the caller's)
    or the pool is off. Never raises: ensure_litellm_runtime reports, when
    the agent connects, whatever stops the proxy."""
    if not host_proxy_ahead():
        return None
    try:
        asyncio.get_running_loop()
        if not proxy_runs_on_host(environment, force_sandbox_local=force_sandbox_local):
            return None
        if model is None or agent_env.get(LITELLM_MASTER_KEY_ENV):
            return None
        if uses_native_subscription_auth(
            agent, model, agent_env
        ) or not needs_litellm_runtime(agent, model):
            return None
        route = resolve_litellm_route(model, agent_env)
        if _missing_required_env(route, agent_env):
            return None
        proxy_env = _host_proxy_env(
            agent, agent_env, required_skill_names, trial_budget
        )
        fields = _host_proxy_fields(
            route=route, agent=agent, environment=environment, proxy_env=proxy_env
        )
        reserved = _HOST_PROXY_POOL.reserve(
            _host_proxy_key(fields),
            _host_proxy_spawner(
                route=route,
                agent=agent,
                environment=environment,
                proxy_env=proxy_env,
            ),
            fields,
        )
    except Exception as exc:
        logger.debug("No warm host proxy: %s", describe_exception(exc))
        return None
    return HostProxyReservation(reserved) if reserved else None


@contextlib.contextmanager
def connecting_with(reservation: HostProxyReservation | None) -> Any:
    """Let ensure_litellm_runtime, called inside, take the reserved proxy."""
    token = _HOST_PROXY_RESERVATION.set(reservation)
    try:
        yield
    finally:
        _HOST_PROXY_RESERVATION.reset(token)


def release_host_proxy(reservation: HostProxyReservation | None) -> None:
    """Give back a reservation no connection took (the trial ended first)."""
    if reservation is None or reservation.answered:
        return
    reservation.answered = True
    with contextlib.suppress(RuntimeError):
        _HOST_PROXY_POOL.release(reservation.key)


async def drain_host_proxies() -> None:
    """Stop the started proxies no trial took (a job's end)."""
    with contextlib.suppress(Exception):
        await _HOST_PROXY_POOL.drain()


async def _claim_host_proxy(
    *,
    route: LiteLLMRoute,
    agent: str,
    environment: str,
    proxy_env: dict[str, str],
) -> tuple[Any, str] | None:
    """The reserved trial's warm proxy for this launch, or None."""
    reservation = _HOST_PROXY_RESERVATION.get()
    if reservation is None or reservation.answered:
        return None
    reservation.answered = True
    fields = _host_proxy_fields(
        route=route, agent=agent, environment=environment, proxy_env=proxy_env
    )
    started = _timing_spans.clock()
    claimed = await _HOST_PROXY_POOL.claim(
        _host_proxy_key(fields),
        _host_proxy_spawner(
            route=route, agent=agent, environment=environment, proxy_env=proxy_env
        ),
        reserved=reservation.key,
        fields=fields,
    )
    if claimed is not None:
        # The trial's wait for its proxy: what was left of the start.
        _timing_spans.add(
            _timing_spans.PROXY_HEALTH_WAIT, None, _timing_spans.clock() - started
        )
    return claimed


def _sandbox_launcher_source() -> str:
    # The launcher's route-owned names are route_owned_name's, the C8 rule's
    # one list (bash's start-up names included), written into its text: it
    # runs in the sandbox, where benchflow is not importable.
    names = ("PATH", *SYSTEM_SHELL_UNSET_VARS, *BASH_STARTUP_VARS)
    prefixes = ("LD_", "PYTHON", BASH_STARTUP_PREFIX)
    return _SANDBOX_LAUNCHER.replace("@ROUTE_OWNED_NAMES@", repr(names)).replace(
        "@ROUTE_OWNED_PREFIXES@", repr(prefixes)
    )


_SANDBOX_LAUNCHER = r"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

# This exec does not inherit the installer's umask. Keep the PID/state/log
# files private, including files LiteLLM creates, so guarded cleanup accepts
# our own PID file even when the sandbox's umask is 000 (FF-install-C10).
os.umask(0o077)

# Read launch config from a file (argv[1]), never the command line: provider
# keys live in cfg["env"], and a shared sandbox exposes exec argv via /proc.
cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])

# The child's environment is built here, from the route's (the launcher
# runs on root's plumbing route): no loader, start-up or Python value of the
# image's or the launch's comes back, and its only Python path is the
# checked runtime folder, with no empty entry (C8, PT-C8-R3-7).
def _route_owned(name):
    return name in @ROUTE_OWNED_NAMES@ or name.startswith(@ROUTE_OWNED_PREFIXES@)


env = {k: v for k, v in os.environ.items() if not _route_owned(k)}
env.update({k: v for k, v in cfg["env"].items() if not _route_owned(k)})
env["PATH"] = os.environ.get("PATH", "/usr/sbin:/usr/bin:/sbin:/bin")
env["PYTHONPATH"] = cfg["runtime_dir"]
# No user site either: root's HOME may be one an earlier role wrote
# (FF-install-C14).
env["PYTHONNOUSERSITE"] = "1"
# No .env either: LiteLLM's import would read one from the image's folders
# above the runtime folder.
env["PYTHON_DOTENV_DISABLED"] = "1"
env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
stdout = open(cfg["stdout"], "ab")
stderr = open(cfg["stderr"], "ab")
cmd = [
    cfg["litellm"],
    "--config",
    cfg["config"],
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
]
proc = subprocess.Popen(
    cmd,
    stdin=subprocess.DEVNULL,
    stdout=stdout,
    stderr=stderr,
    env=env,
    cwd=cfg["runtime_dir"],
    start_new_session=True,
)
with open(cfg["pid"], "w", encoding="utf-8") as handle:
    handle.write(str(proc.pid))
with open(cfg["state"], "w", encoding="utf-8") as handle:
    json.dump({"pid": proc.pid, "port": port}, handle)
print(json.dumps({"pid": proc.pid, "port": port}))
"""


async def _upload_text(sandbox: Any, text: str, target_path: str, suffix: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        # Runtime files carry the provider environment and proxy master key;
        # ``mode`` is part of the sandbox protocol, so every backend either
        # honors it or fails loudly — never a silently world-readable secret.
        await sandbox.upload_file(
            tmp_path,
            target_path,
            mode="600",
            admission=admit(
                sandbox, "main", TRUSTED_TRANSFER, reason="BenchFlow's LiteLLM runtime"
            ),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


async def _upload_runtime_files_to_sandbox(
    sandbox: Any,
    *,
    runtime_dir: str,
    config: dict[str, object],
) -> dict[str, str]:
    paths = {
        "config": f"{runtime_dir}/config.yaml",
        "callback": f"{runtime_dir}/{_CALLBACK_MODULE}.py",
        "patch": f"{runtime_dir}/{_PATCH_MODULE}.py",
        "gemini_patch": f"{runtime_dir}/{_GEMINI_PATCH_MODULE}.py",
        "token_capture_patch": f"{runtime_dir}/{_TOKEN_CAPTURE_PATCH_MODULE}.py",
        "provider_retry_patch": f"{runtime_dir}/{_PROVIDER_RETRY_PATCH_MODULE}.py",
        "request_guard": f"{runtime_dir}/{_REQUEST_GUARD_MODULE}.py",
        "sitecustomize": f"{runtime_dir}/sitecustomize.py",
        "launcher": f"{runtime_dir}/launcher.py",
        "stdout": f"{runtime_dir}/stdout.log",
        "stderr": f"{runtime_dir}/stderr.log",
        "log": f"{runtime_dir}/callback.jsonl",
        "pid": f"{runtime_dir}/litellm.pid",
        "state": f"{runtime_dir}/state.json",
        "venv": f"{runtime_dir}/venv",
        "launch_config": f"{runtime_dir}/launch_config.json",
        "preflight": f"{runtime_dir}/bedrock_patch_preflight.py",
    }
    # Root writes the proxy's config and launcher here and runs them: only
    # in a folder that is root's alone (C8; /opt/runtime/proxy is root's),
    # new and private to root (the config holds the proxy's key; a folder
    # that already exists is refused, FF-install-C2).
    result = await exec_raw(
        sandbox.exec,
        f"(umask 022; mkdir -p {shlex.quote(runtime_dir.rpartition('/')[0])}) && "
        f"mkdir -m 700 {shlex.quote(runtime_dir)} && "
        + root_only_tree_check(runtime_dir, "the proxy runtime folder"),
        timeout_sec=20,
    )
    if result.return_code != 0:
        raise RuntimeError(_exec_details("prepare LiteLLM runtime directory", result))
    try:
        await _upload_text(
            sandbox, yaml.safe_dump(config, sort_keys=False), paths["config"], ".yaml"
        )
        await _upload_text(sandbox, callback_module_source(), paths["callback"], ".py")
        for filename, key in (
            ("litellm_bedrock_patch.py", "patch"),
            ("litellm_gemini_passthrough_patch.py", "gemini_patch"),
            ("litellm_token_capture_patch.py", "token_capture_patch"),
            ("litellm_provider_retry_patch.py", "provider_retry_patch"),
        ):
            await _upload_text(
                sandbox,
                Path(__file__).with_name(filename).read_text(),
                paths[key],
                ".py",
            )
        await _upload_text(
            sandbox, _sitecustomize_source(config), paths["sitecustomize"], ".py"
        )
        await _upload_text(
            sandbox,
            Path(__file__).with_name("litellm_request_guard.py").read_text(),
            paths["request_guard"],
            ".py",
        )
        await _upload_text(
            sandbox, _sandbox_launcher_source(), paths["launcher"], ".py"
        )
        await _upload_text(
            sandbox, BEDROCK_PATCH_PREFLIGHT_SOURCE, paths["preflight"], ".py"
        )
    except BaseException:
        await _terminate_sandbox_litellm(
            sandbox, pid_path=paths["pid"], runtime_dir=runtime_dir
        )
        raise
    return paths


# What the proxy's install in a trial leaves in its venv's folder (the venv,
# copied there from uv's cache when the cache is on another disk), measured
# for the locked set on ubuntu:24.04: see _disk_check_command. A task disk
# with less free there fails before the agent with a message naming the
# shortfall, instead of half way through the install (#1169).
PROXY_INSTALL_FOOTPRINT_MB = 600
# The message's advice, also given when the install itself runs out of disk.
_DISK_ADVICE = (
    "Give the task more disk (its environment's storage_mb), or let the proxy "
    "come from its cached layer, which does not count against the task's disk "
    "(Docker and remote Docker without --fresh-install)"
)


def _disk_check_command(folder: str, need_mb: int = PROXY_INSTALL_FOOTPRINT_MB) -> str:
    """sh that stops (exit 28, ENOSPC's number) when ``folder``'s disk has
    less than ``need_mb`` free, naming the shortfall; with no ``df`` (or an
    answer it cannot read) it lets the install try."""
    fq = shlex.quote(folder)
    need_kb = need_mb * 1024
    return (
        f'free_kb="$(df -Pk {fq} 2>/dev/null | '
        '{ read -r _ || true; read -r _ _ _ a _ || true; echo "${a:-}"; })"\n'
        'case "$free_kb" in ""|*[!0-9]*) ;; *)\n'
        f'  if [ "$free_kb" -lt {need_kb} ]; then\n'
        f"    echo \"benchflow: the model proxy's install needs about {need_mb} MB "
        f"free under {folder}; the sandbox has $((free_kb / 1024)) MB free there, "
        f'$(( ({need_kb} - free_kb) / 1024 )) MB short (#1169). {_DISK_ADVICE}" >&2\n'
        "    exit 28\n"
        "  fi ;;\n"
        "esac"
    )


def proxy_layer_recipe(
    install: ProxyInstall, *, timeout_sec: int = 900
) -> ProxyLayerRecipe:
    """The cached layer that holds ``install``'s venv (sandbox.env_layers).

    The same pinned uv, managed CPython and hashed requirement file as the
    install in a trial (``_ensure_sandbox_litellm``), with the venv at
    ``PROXY_LAYER_VENV`` and its bytecode compiled, since every trial on the
    layer starts the proxy from it.
    """
    venv = shlex.quote(PROXY_LAYER_VENV)
    requirements = shlex.quote(f"{PROXY_LAYER_REQUIREMENTS_DIR}/{install.file_name}")
    command = "\n".join(
        [
            "set -eu",
            f"UV={shlex.quote(UV_BIN)}",
            uv_install_command(),
            managed_python_exports(),
            managed_python_install_command(),
            f'"$UV" venv -q --python {PYTHON_VERSION} {venv}',
            uv_lock_file_command(PROXY_LAYER_VENV),
            install_commands(
                uv='"$UV"',
                python=f"{venv}/bin/python",
                requirements_path=requirements,
                extras=install.extra_requirements,
                compile_bytecode=True,
            ),
            uv_lock_seal_command(PROXY_LAYER_VENV),
            f"cd / && {venv}/bin/python -I -c 'import litellm'",
        ]
    )
    return ProxyLayerRecipe(
        group=install.group,
        requirements_name=install.file_name,
        requirements_text=install.text,
        install_cmd=command,
        python=PYTHON_VERSION,
        uv_version=UV_VERSION,
        extra_requirements=install.extra_requirements,
        timeout_sec=timeout_sec,
    )


async def _proxy_from_layer(
    sandbox: Any, install: ProxyInstall, *, bypass_reason: str | None = None
) -> tuple[str | None, dict[str, Any] | None]:
    """The layer's venv python when the sandbox's proxy layer holds ``install``.

    Returns (python, record). ``record`` is None when the sandbox started
    from no proxy layer (none asked for, or a backend without layers); else
    it says ``install: from layer``, or ``in trial`` with the reason, and
    then python is None and the trial installs as before. One command reads
    the layer's stamp in the live sandbox: a layer built for other
    requirements or extras (a second role's route), or a sandbox where the
    venv is gone, is not used.
    """
    layer = getattr(sandbox, "proxy_layer", None)
    if not isinstance(layer, LayerResult):
        return None, None
    if bypass_reason is not None:
        return None, {"install": "in trial", "reason": bypass_reason}
    want = proxy_layer_recipe(install).stamp
    venv = shlex.quote(PROXY_LAYER_VENV)
    # Root runs the venv's litellm, so nobody else may be able to change it
    # or a folder above it (a workspace there): the check stops
    # before the stamp is read, and the trial installs its own. Include the
    # interpreter and stdlib behind uv's bin/python link (FF-install-C1).
    guard = " && ".join(
        [
            root_only_tree_check(PROXY_LAYER_VENV, "the cached proxy venv"),
            root_only_tree_check(PYTHON_INSTALL_DIR, "BenchFlow's managed Python"),
            root_only_path_check(PROXY_LAYER_STAMP, "the proxy layer's stamp"),
        ]
    )
    try:
        # Root runs the layer's Python next: only while the venv is root's
        # alone (C8), else the trial installs its own under its runtime dir.
        result = await exec_raw(
            sandbox.exec,
            f"{guard} && cat {shlex.quote(PROXY_LAYER_STAMP)} && "
            f"[ -x {venv}/bin/python ] && [ -x {venv}/bin/litellm ]",
            timeout_sec=30,
        )
    except Exception as e:
        reason = f"the proxy layer's stamp could not be read: {describe_exception(e)}"
        return None, {"install": "in trial", "reason": reason}
    found = (getattr(result, "stdout", "") or "").strip()
    refused = (getattr(result, "stderr", "") or "").strip()
    if getattr(result, "return_code", 1) != 0 and refused.startswith("benchflow: "):
        reason = f"layer {layer.key}'s venv is not used: {refused.removeprefix('benchflow: ')}"
        return None, {"install": "in trial", "reason": reason}
    if getattr(result, "return_code", 1) != 0 or found != want:
        reason = (
            f"layer {layer.key} holds no {install.group} venv for this install "
            f"that is root's alone (stamp {found or 'missing'}, wanted {want})"
        )
        return None, {"install": "in trial", "reason": reason}
    return f"{PROXY_LAYER_VENV}/bin/python", {"install": "from layer"}


def sandbox_proxy_extra_requirements(provider_name: str | None) -> tuple[str, ...]:
    """The run's extra packages for the in-sandbox proxy venv of a route.

    Vertex's SDK is in the litellm-proxy-vertex locked set, not an extra;
    what is left are the run's own extras (proxy_requirements).
    """
    return proxy_install(provider_name).extra_requirements


def sandbox_litellm_install_command(
    venv_dir: str,
    install: ProxyInstall | None = None,
    *,
    python_install_dir: str | None = None,
) -> str:
    """A self-contained script that installs ``install``'s proxy venv at ``venv_dir``.

    Daytona's cached proxy layer runs it while building the image
    (``sandbox.daytona_layers``): the locked, hashed requirement file is
    written next to the venv from the script itself (a recipe build has no
    upload), then the same pinned uv, managed CPython and ``install_commands``
    as the trial's install and Docker's layer (:func:`proxy_layer_recipe`),
    with bytecode compiled since every trial on the image starts from it.
    """
    install = install or proxy_install(None)
    vq = shlex.quote(venv_dir)
    requirements = f"{venv_dir.rstrip('/')}-requirements/{install.file_name}"
    rq = shlex.quote(requirements)
    folder = shlex.quote(requirements.rpartition("/")[0])
    return "\n".join(
        [
            "set -eu",
            f"mkdir -p {folder}",
            f"cat > {rq} <<'BENCHFLOW_REQUIREMENTS_EOF'",
            install.text.rstrip("\n"),
            "BENCHFLOW_REQUIREMENTS_EOF",
            f"UV={shlex.quote(UV_BIN)}",
            uv_install_command(),
            managed_python_exports(),
            *(
                [f"export UV_PYTHON_INSTALL_DIR={shlex.quote(python_install_dir)}"]
                if python_install_dir is not None
                else []
            ),
            managed_python_install_command(),
            f'"$UV" venv -q --python {PYTHON_VERSION} {vq}',
            uv_lock_file_command(venv_dir),
            install_commands(
                uv='"$UV"',
                python=f"{vq}/bin/python",
                requirements_path=rq,
                extras=install.extra_requirements,
                compile_bytecode=True,
            ),
            uv_lock_seal_command(
                venv_dir, *([python_install_dir] if python_install_dir else [])
            ),
            f"cd / && {vq}/bin/python -I -c 'import litellm'",
        ]
    )


async def _ensure_sandbox_litellm(
    sandbox: Any,
    *,
    venv_dir: str,
    requirements_path: str,
    extra_requirements: tuple[str, ...] = (),
    install_timeout_sec: int = 600,
) -> str:
    vq = shlex.quote(venv_dir)
    # The venv is built by the pinned uv on a uv-managed CPython (see
    # benchflow._uv_pin): never the image's own uv or Python. uv 0.13.0 made
    # Python 3.15 the default when an image has no Python, and fastuuid (a
    # LiteLLM 1.91 dependency) has no cp315 wheel, so an unpinned venv broke
    # every proxy install on TB2's images without Python before the agent
    # started (F3-4). uv also needs no ensurepip and ignores PEP 668 markers,
    # which is why the old stdlib-venv fallback is gone: there is no Python to
    # fall back to on the images this matters for, and a pinned Python on the
    # others is the point. Under an egress policy this install still runs
    # before the policy is armed (connect() starts it after the proxy is up,
    # and it binds only the agent's uid), so it reaches the uv and Python
    # download hosts; when it cannot, it stops here, before the agent, with a
    # message naming them. The venv gets the locked, hashed set uploaded to
    # ``requirements_path`` (benchflow.providers.proxy_requirements), never a
    # fresh resolve, and the run's extras only under it as constraints.
    rq = shlex.quote(requirements_path)
    # This venv serves one trial and is deleted with it, so nothing is
    # compiled ahead (see install_commands); a cached layer compiles.
    uv_install = install_commands(
        uv='"$UV"',
        python=f"{vq}/bin/python",
        requirements_path=rq,
        extras=extra_requirements,
        compile_bytecode=False,
    )
    folder = venv_dir.rstrip("/").rpartition("/")[0] or "/"
    # Docker's upload (docker cp) keeps the host user's uid on every file
    # BenchFlow uploaded here (the config, the patches, the launcher, the
    # requirements), which the tree check refused on every in-trial install by
    # a non-root host (FF-INSTALL-ON-PT-R1). The folder itself is checked
    # first: root made it 700 and only root may write it, so what it holds is
    # BenchFlow's own uploads. Those are made the installing user's, root's
    # in a sandbox (links not followed), then the whole tree is checked like
    # the venv.
    qfolder = shlex.quote(folder)
    command = f"""
set -eu
{root_only_path_check(folder, "the proxy runtime folder")} || exit 1
chown -R -h "$(id -u):$(id -g)" -- {qfolder} && chmod -R go-w -- {qfolder} || exit 1
{root_only_tree_check(folder, "the proxy runtime folder")} || exit 1
{_disk_check_command(folder)}
UV={shlex.quote(UV_BIN)}
{uv_install_command()}
{managed_python_exports()}
{private_tmpdir_exports(f"{folder}/uv-tmp")}
{managed_python_install_command()}
[ -x {vq}/bin/python ] || "$UV" venv -q --python {PYTHON_VERSION} {vq}
{uv_lock_file_command(venv_dir)}
{uv_install}
{uv_lock_seal_command(venv_dir)}
cd / && {vq}/bin/python -I - <<'PY'
import litellm
print(litellm.__version__ if hasattr(litellm, "__version__") else "ok")
PY
"""
    result = await exec_raw(sandbox.exec, command, timeout_sec=install_timeout_sec)
    if result.return_code != 0:
        details = _exec_details("install LiteLLM in sandbox", result)
        if "No space left on device" in details and "#1169" not in details:
            # Out of disk past the check (another disk filled, or the check
            # could not read df): say so before the installer's own output.
            details = (
                f"the sandbox ran out of disk installing the model proxy "
                f"(about {PROXY_INSTALL_FOOTPRINT_MB} MB under {folder}; #1169). "
                f"{_DISK_ADVICE}. {details}"
            )
        raise RuntimeError(details)
    return f"{venv_dir}/bin/python"


async def _wait_for_sandbox_state(
    sandbox: Any,
    *,
    state_path: str,
    stderr_path: str,
    deadline_s: float = _HEALTH_DEADLINE_SEC,
) -> dict[str, Any]:
    last_output = ""
    loop = asyncio.get_running_loop()
    start = loop.time()
    while loop.time() - start < deadline_s:
        try:
            result = await exec_raw(
                sandbox.exec,
                f"cat {shlex.quote(state_path)} 2>/dev/null || true",
                timeout_sec=5,
            )
            output = (result.stdout or "").strip()
            last_output = (diagnostic_result(result).stdout.render() or "").strip()
            if output:
                try:
                    state = json.loads(output)
                    if int(state.get("port") or 0) > 0:
                        return state
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
        except Exception as exc:
            last_output = str(exc)
        await asyncio.sleep(0.25)
    stderr = await exec_raw(
        sandbox.exec,
        f"tail -c 4000 {shlex.quote(stderr_path)} 2>/dev/null || true",
        timeout_sec=5,
    )
    raise RuntimeError(
        "sandbox LiteLLM did not publish its state: "
        f"{last_output or (diagnostic_result(stderr).stdout.render() or '').strip()}"
    )


async def _poll_sandbox_health(
    sandbox: Any,
    *,
    python: str,
    port: int,
    stderr_path: str,
    deadline_s: float = _HEALTH_DEADLINE_SEC,
) -> None:
    probe = (
        # Isolated, from /: the standard library the venv's Python was
        # checked with, never a module in the image's working folder (C8).
        f"cd / && {shlex.quote(python)} -I - <<'PY'\n"
        "import sys, urllib.request\n"
        f"url='http://127.0.0.1:{port}/health/liveliness'\n"
        "try:\n"
        "    urllib.request.urlopen(url, timeout=2).read()\n"
        "except Exception:\n"
        "    urllib.request.urlopen(url.replace('/health/liveliness','/health'), timeout=2).read()\n"
        "PY"
    )
    loop = asyncio.get_running_loop()
    start = loop.time()
    while loop.time() - start < deadline_s:
        try:
            result = await exec_raw(sandbox.exec, probe, timeout_sec=5)
            if result.return_code == 0:
                return
        except Exception:
            pass
        await asyncio.sleep(0.25)
    stderr = await exec_raw(
        sandbox.exec,
        f"tail -c 4000 {shlex.quote(stderr_path)} 2>/dev/null || true",
        timeout_sec=5,
    )
    raise RuntimeError(
        f"sandbox LiteLLM did not become healthy: {diagnostic_result(stderr).stdout.render() or ''}"
    )


def _runtime_use_guard(runtime_dir: str, pid_path: str) -> str:
    """sh: the runtime folder, and its pid file when there is one, are root's
    alone, before root kills the pid it names or removes the folder: a pid
    file someone else may write leaves both alone."""
    q = shlex.quote(pid_path)
    return (
        root_only_path_check(runtime_dir, "the proxy runtime folder")
        + f" && {{ {{ [ ! -e {q} ] && [ ! -L {q} ]; }} || "
        + root_only_path_check(pid_path, "the proxy's pid file")
        + "; }"
    )


async def _terminate_sandbox_litellm(
    sandbox: Any, *, pid_path: str, runtime_dir: str
) -> None:
    # Checked again at use (FF-install-C2, C10).
    guard = _runtime_use_guard(runtime_dir, pid_path)
    with contextlib.suppress(Exception):
        await exec_raw(
            sandbox.exec,
            f"{guard} && if [ -s {shlex.quote(pid_path)} ]; then "
            f"kill -TERM $(cat {shlex.quote(pid_path)}) 2>/dev/null || true; fi",
            timeout_sec=10,
        )
    with contextlib.suppress(Exception):
        await exec_raw(
            sandbox.exec,
            f"{guard} && rm -rf {shlex.quote(runtime_dir)}",
            timeout_sec=10,
        )


async def _start_sandbox_litellm(
    *,
    sandbox: Any,
    route: LiteLLMRoute,
    master_key: str,
    agent_env: dict[str, str],
    session_id: str,
    agent_name: str,
    install_timeout_sec: int = 600,
    install_layer_bypass_reason: str | None = None,
) -> SandboxLiteLLMProcess:
    token = uuid4().hex[:16]
    runtime_dir = f"{LITELLM_SANDBOX_ROOT}/{token}"
    pid_path = f"{runtime_dir}/litellm.pid"
    config = litellm_proxy_config(route, master_key=master_key)
    proxy_agent_env = dict(agent_env)
    # Vertex routes install the litellm-proxy-vertex set (with the Vertex
    # SDK); every route installs the run's extras under its locked set.
    install = proxy_install(route.provider_name)
    await asyncio.to_thread(install.check)
    adc_json = None
    if route.provider_name in {"google-vertex", "anthropic-vertex"}:
        adc_json = proxy_agent_env.pop("GOOGLE_APPLICATION_CREDENTIALS_JSON", None)
    runtime_prepared = False
    try:
        paths = await _upload_runtime_files_to_sandbox(
            sandbox,
            runtime_dir=runtime_dir,
            config=config,
        )
        runtime_prepared = True
        with _timing_spans.span(_timing_spans.LITELLM_INSTALL):
            # From the sandbox's cached proxy layer when it holds this exact
            # install (one command); else installed here, in the trial.
            python, layer_use = await _proxy_from_layer(
                sandbox, install, bypass_reason=install_layer_bypass_reason
            )
            if layer_use is not None and layer_use.get("reason"):
                logger.warning(
                    "Installing the proxy in the trial: %s", layer_use["reason"]
                )
            venv_dir = PROXY_LAYER_VENV if python else paths["venv"]
            if (
                python is None
                and layer_use is None
                and install_layer_bypass_reason is None
            ):
                # A Daytona image built with the cached proxy layer already
                # holds the venv (sandbox.daytona_layers).
                from benchflow.sandbox.daytona_layers import (
                    cached_layer_in_sandbox,
                    proxy_install_command,
                )

                daytona_layer = await cached_layer_in_sandbox(
                    sandbox,
                    "proxy",
                    "litellm",
                    script=proxy_install_command(install),
                )
                if daytona_layer is not None and daytona_layer.venv:
                    venv_dir = daytona_layer.venv
                    python = f"{venv_dir}/bin/python"
            if python is None:
                paths["requirements"] = f"{runtime_dir}/{install.file_name}"
                await _upload_text(sandbox, install.text, paths["requirements"], ".txt")
                python = await _ensure_sandbox_litellm(
                    sandbox,
                    venv_dir=paths["venv"],
                    requirements_path=paths["requirements"],
                    extra_requirements=install.extra_requirements,
                    install_timeout_sec=install_timeout_sec,
                )
        if adc_json:
            adc_path = f"{runtime_dir}/application_default_credentials.json"
            await _upload_text(sandbox, adc_json, adc_path, ".json")
            proxy_agent_env["GOOGLE_APPLICATION_CREDENTIALS"] = adc_path
        env = _proxy_process_env(proxy_agent_env, runtime_dir)
        env.update(
            {
                # Only the checked runtime folder (C8, PT-C8-R3-7): the
                # launcher sets it again and drops every other Python value.
                "PYTHONPATH": runtime_dir,
                "PYTHONSAFEPATH": "1",
                "PYTHONNOUSERSITE": "1",
                "LITELLM_MASTER_KEY": master_key,
                "BENCHFLOW_LITELLM_LOG_PATH": paths["log"],
                **_route_env(route),
                **_native_passthrough_env(route, env),
                **_served_models_env(config),
                **_PROXY_DOCS_DISABLE_ENV,
            }
        )
        launch_config = {
            "cwd": runtime_dir,
            "python": python,
            "litellm": f"{venv_dir}/bin/litellm",
            "config": paths["config"],
            "stdout": paths["stdout"],
            "stderr": paths["stderr"],
            "pid": paths["pid"],
            "state": paths["state"],
            "runtime_dir": runtime_dir,
            "env": env,
        }
        await _upload_text(
            sandbox, json.dumps(launch_config), paths["launch_config"], ".json"
        )
        # The launcher reads launch_config.json (provider env + master key) at
        # startup; unlink it immediately afterwards so the secret does not sit in
        # the sandbox filesystem for the life of the run.
        # Root starts the venv's Python on its runtime files: both checked
        # root's alone in the same command (C8), the venv whichever it is
        # (the trial's, a cached layer's or a Daytona image layer's). The
        # runtime folder was root's alone when it was made, so only
        # BenchFlow's uploads are in it; a backend whose file API writes as
        # another user leaves them that user's, so they become root's first.
        venv_root = posixpath.dirname(posixpath.dirname(python))
        quoted_dir = shlex.quote(runtime_dir)
        command = (
            f"chown -R 0:0 -- {quoted_dir} && chmod -R go-w -- {quoted_dir} && "
            + root_only_tree_check(runtime_dir, "the proxy runtime folder")
            + " && "
            + root_only_tree_check(venv_root, "the proxy venv")
            + " && "
            f"rm -f {shlex.quote(paths['state'])} {shlex.quote(paths['pid'])} "
            f"{shlex.quote(paths['log'])} && "
            f"cd {quoted_dir} && {shlex.quote(python)} -I {shlex.quote(paths['launcher'])} "
            f"{shlex.quote(paths['launch_config'])}; "
            f"rc=$?; rm -f {shlex.quote(paths['launch_config'])}; exit $rc"
        )
        # From launching the proxy to its first healthy answer.
        with _timing_spans.span(_timing_spans.PROXY_HEALTH_WAIT):
            result = await exec_raw(sandbox.exec, command, timeout_sec=20)
            if result.return_code != 0:
                raise RuntimeError(_exec_details("start sandbox LiteLLM", result))
            state = await _wait_for_sandbox_state(
                sandbox,
                state_path=paths["state"],
                stderr_path=paths["stderr"],
            )
            port = int(state["port"])
            await _poll_sandbox_health(
                sandbox,
                python=python,
                port=port,
                stderr_path=paths["stderr"],
            )
        if route_requires_bedrock_patch(route):
            # Fail closed before the agent launches when the Bedrock 4.8+
            # thinking patch did not activate (#602). On Daytona this proxy is
            # the only protection — there is no host proxy to fall back to.
            await preflight_sandbox_bedrock_patch(
                sandbox,
                python=python,
                runtime_dir=runtime_dir,
                preflight_path=paths["preflight"],
                effort_models=(route.upstream_model,),
            )
    except BaseException:
        # Never leak a half-started proxy (provider keys + master_key on disk).
        if runtime_prepared:
            await _terminate_sandbox_litellm(
                sandbox, pid_path=pid_path, runtime_dir=runtime_dir
            )
        raise
    endpoint = LiteLLMEndpoint(
        agent_base_url=f"http://127.0.0.1:{port}",
        local_base_url=f"http://127.0.0.1:{port}",
    )
    logger.info("Sandbox LiteLLM proxy listening on %s", endpoint.agent_base_url)
    process = SandboxLiteLLMProcess(
        sandbox=sandbox,
        route=route,
        runtime_dir=runtime_dir,
        endpoint=endpoint,
        log_path=paths["log"],
        pid_path=paths["pid"],
        stdout_path=paths["stdout"],
        stderr_path=paths["stderr"],
        session_id=session_id,
        agent_name=agent_name,
    )
    process.layer_use = layer_use
    return process


def _exec_details(label: str, result: Any) -> str:
    stdout = (diagnostic_result(result).stdout.render() or "").strip()
    stderr = (diagnostic_result(result).stderr.render() or "").strip()
    details = [f"{label} failed with exit code {getattr(result, 'return_code', '?')}"]
    if stdout:
        details.append(f"stdout: {stdout[:2000]}")
    if stderr:
        details.append(f"stderr: {stderr[:2000]}")
    return "; ".join(details)


def _missing_required_env(route: LiteLLMRoute, env: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for key in route.required_env:
        if key == "AWS_REGION" and (
            env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
        ):
            continue
        if not env.get(key):
            missing.append(key)
    return missing


def _provider_secret_env_names() -> set[str]:
    """Upstream provider credentials the proxy owns and the agent must not see."""
    from benchflow.agents.providers import PROVIDERS

    names = {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AZURE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
    }
    for cfg in PROVIDERS.values():
        if cfg.auth_env:
            names.add(cfg.auth_env)
        for credential in cfg.credential_files:
            names.add(credential["env_source"])
            names.update(credential.get("post_env", {}))
    return names


def _provider_model_id(entry: object) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    entry_id = cast("Mapping[str, object]", entry).get("id")
    return entry_id if isinstance(entry_id, str) else None


def _provider_models_for_proxy_alias(
    *,
    raw: str | None,
    route: LiteLLMRoute,
) -> str | None:
    """Preserve provider metadata when agents see a proxy alias instead of its ID."""
    try:
        entries = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        return None
    if not isinstance(entries, list):
        return None
    if any(_provider_model_id(entry) == route.model_alias for entry in entries):
        return raw
    wanted = {
        route.requested_model,
        strip_provider_prefix(route.requested_model),
        route.upstream_model,
        strip_provider_prefix(route.upstream_model),
    }
    source = next(
        (entry for entry in entries if _provider_model_id(entry) in wanted), None
    )
    if source is None:
        from litellm import get_model_info

        try:
            info = get_model_info(route.upstream_model)
        except Exception as exc:
            # LiteLLM raises plain Exception for unmapped models. Metadata is
            # optional; retain existing behavior when its catalog cannot resolve.
            logger.debug("Proxy model metadata unavailable: %s", type(exc).__name__)
            return None
        source = {}
        if "reasoning_effort" in (info.get("supported_openai_params") or []):
            source["compat"] = {"supportsReasoningEffort": True}
        if isinstance(info.get("supports_reasoning"), bool):
            source["reasoning"] = info["supports_reasoning"]
        for target, key in (
            ("maxTokens", "max_output_tokens"),
            ("contextWindow", "max_input_tokens"),
        ):
            value = info.get(key)
            if type(value) is int and value > 0:
                source[target] = value
        if not source:
            return None
        source["input"] = ["text", "image"] if info.get("supports_vision") else ["text"]
    alias_entry = dict(source)
    alias_entry.update(id=route.model_alias, name=route.model_alias)
    return json.dumps([*entries, alias_entry])


# Caller-supplied provider endpoints. If any of these survive in the agent env,
# the agent could reach a provider directly and bypass the proxy (the exact way
# an Azure ``LLM_BASE_URL`` leaked past the gateway before this hardening). They
# are stripped before the proxy endpoint is wired in; the proxy process keeps
# whatever upstream base_url it needs in its own env / baked config.
_PROVIDER_ENDPOINT_ENV_NAMES = frozenset(
    {
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "AZURE_API_ENDPOINT",
        "AZURE_API_BASE",
        "AZURE_OPENAI_ENDPOINT",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_BEDROCK_BASE_URL",
        "GEMINI_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
    }
)


def _assert_proxy_isolated(agent: str, env: dict[str, str], *, master_key: str) -> None:
    """Fail closed if a *raw* provider secret would still reach the agent.

    Invariant after wiring: a routable agent sees only the proxy endpoint and
    the proxy master key — never an upstream provider credential. Some agents
    legitimately carry the master key in a provider-named slot (codex/opencode
    put it in ``OPENAI_API_KEY``, claude in ``ANTHROPIC_AUTH_TOKEN``), so a
    provider-named var holding exactly ``master_key`` is the proxy credential and
    is allowed; anything else in those slots is a raw upstream key. If one
    survives (e.g. an agent registry ``env_mapping`` re-exposed it), refuse to
    run so the agent cannot authenticate directly against a provider.
    """
    from benchflow.agents.launch_env import model_credential_env_names

    leaked = sorted(
        name
        for name in _provider_secret_env_names() | model_credential_env_names()
        if env.get(name) and env.get(name) != master_key
    )
    if leaked:
        raise RuntimeError(
            f"LiteLLM proxy isolation breached for agent {agent!r}: raw provider "
            f"secrets would reach the agent ({', '.join(leaked)}). Refusing to run "
            "to prevent direct-to-provider traffic."
        )


def _apply_litellm_agent_env(
    *,
    agent: str,
    agent_env: dict[str, str],
    route: LiteLLMRoute,
    base_url: str,
    master_key: str,
) -> dict[str, str]:
    """Rewrite the agent env to talk only to the proxy, then verify isolation."""
    from benchflow.agents.launch_env import agent_launch_env

    updated = _wire_litellm_agent_env(
        agent=agent,
        agent_env=agent_env,
        route=route,
        base_url=base_url,
        master_key=master_key,
    )
    # Native logins of other routes (CODEX_AUTH_JSON, CLAUDE_CODE_OAUTH_TOKEN,
    # ...) are not provider secrets the wiring drops: admit only the proxy key.
    updated = agent_launch_env(agent, route.requested_model, updated)
    _assert_proxy_isolated(agent, updated, master_key=master_key)
    return updated


def _litellm_proxy_env(
    *, agent: str, agent_env: dict[str, str], required_skill_names: tuple[str, ...]
) -> dict[str, str]:
    updated = dict(agent_env)
    updated["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    updated.pop(_SKILL_CATALOG_GATE_AGENT_ENV, None)
    updated.pop(_REQUIRED_SKILL_NAMES_ENV, None)
    expected = sorted(set(required_skill_names))
    if agent == "opencode" and expected:
        updated[_SKILL_CATALOG_GATE_AGENT_ENV] = agent
        updated[_REQUIRED_SKILL_NAMES_ENV] = json.dumps(expected)
    return updated


def _wire_litellm_agent_env(
    *,
    agent: str,
    agent_env: dict[str, str],
    route: LiteLLMRoute,
    base_url: str,
    master_key: str,
) -> dict[str, str]:
    updated = dict(agent_env)
    updated.pop(_SKILL_CATALOG_GATE_AGENT_ENV, None)
    updated.pop(_REQUIRED_SKILL_NAMES_ENV, None)
    # Isolation: the agent must reach providers only through the proxy. Drop raw
    # upstream provider secrets AND endpoints so a compromised or curious agent
    # cannot bypass the gateway (and its usage metering) or read live keys. The
    # proxy process holds them via its own env. (In sandbox-local mode the proxy
    # shares the agent's sandbox, so this reduces — but cannot fully remove — key
    # visibility.)
    for secret_key in _provider_secret_env_names():
        updated.pop(secret_key, None)
    for endpoint_key in _PROVIDER_ENDPOINT_ENV_NAMES:
        updated.pop(endpoint_key, None)
    openai_base_url = f"{base_url.rstrip('/')}/v1"
    updated.update(
        {
            "BENCHFLOW_PROVIDER_NAME": "litellm",
            "BENCHFLOW_PROVIDER_BASE_URL": openai_base_url,
            "BENCHFLOW_PROVIDER_API_KEY": master_key,
            "BENCHFLOW_PROVIDER_MODEL": route.model_alias,
            "BENCHFLOW_PROVIDER_PROTOCOL": "openai-completions",
            LITELLM_MODEL_ALIAS_ENV: route.model_alias,
            LITELLM_MASTER_KEY_ENV: master_key,
        }
    )
    alias_models = _provider_models_for_proxy_alias(
        raw=agent_env.get("BENCHFLOW_PROVIDER_MODELS"), route=route
    )
    if alias_models:
        updated["BENCHFLOW_PROVIDER_MODELS"] = alias_models
    # Generic model-via-env: an agent whose registration maps
    # BENCHFLOW_PROVIDER_MODEL into an agent-native env var AND declares
    # supports_acp_set_model=False states, in data, that launch/env config owns
    # model selection. Set the via-env flag so the ACP layer
    # (_model_selection_owned_by_env) skips driving set_model AND any
    # capability-advertised model config option — several agents (qwen-code,
    # kilo, dimcode, ...) validate foreign model ids against their own catalog
    # and reject the gateway alias with -32603. This generalizes the hardcoded
    # codex-acp/openhands/claude-agent-acp branches below to every agent that
    # declares the env-owned-model shape (e.g. the benchflow-ai/agents
    # manifests), instead of growing the special-case list per agent.
    _cfg = AGENTS.get(agent)
    if (
        _cfg is not None
        and not _cfg.supports_acp_set_model
        and _cfg.env_mapping.get("BENCHFLOW_PROVIDER_MODEL")
    ):
        updated[LITELLM_MODEL_VIA_ENV] = "1"
    if agent == "codex-acp":
        updated["OPENAI_BASE_URL"] = openai_base_url
        updated["OPENAI_API_KEY"] = master_key
        updated[LITELLM_MODEL_VIA_ENV] = "1"
        # The bare slug, not route.model_alias: Codex looks its model metadata
        # up by slug and falls back to a reduced tool surface for the alias
        # (#1145). The proxy serves both names.
        apply_codex_provider_config(
            updated,
            base_url=openai_base_url,
            model=strip_provider_prefix(route.requested_model),
            provider_name="litellm",
            strict=True,
        )
        return updated
    if agent == "opencode":
        if route.upstream_model.lower().startswith("gemini/"):
            # OpenCode's own Gemini client through LiteLLM's byte-preserving
            # Gemini pass-through, as for Gemini CLI below: the chat-completions
            # alias would drop Gemini's thought signatures and thinking config.
            # The -proxy wrapper points OpenCode's google provider at it.
            updated.pop(LITELLM_MODEL_ALIAS_ENV, None)
            updated[OPENCODE_GOOGLE_BASE_URL_ENV] = (
                f"{base_url.rstrip('/')}/gemini/v1beta"
            )
            updated["GOOGLE_GENERATIVE_AI_API_KEY"] = master_key
            return updated
        updated["OPENAI_BASE_URL"] = openai_base_url
        updated["OPENAI_API_KEY"] = master_key
        return updated
    if agent == "openhands":
        updated["LLM_BASE_URL"] = openai_base_url
        updated["LLM_API_KEY"] = master_key
        updated["LLM_MODEL"] = f"openai/{route.model_alias}"
        updated[LITELLM_MODEL_VIA_ENV] = "1"
        return updated
    if routes_gemini_natively(_cfg):
        # Gemini CLI and the Antigravity CLI speak Google's native
        # GenerateContent protocol. Use LiteLLM's byte-preserving Gemini
        # pass-through route: its translated GenerateContent route can corrupt
        # streamed, multi-tool responses. The gateway authenticates the
        # reviewer with ``master_key`` and swaps in the upstream Gemini key
        # server-side.
        updated.pop(LITELLM_MODEL_ALIAS_ENV, None)
        updated[GEMINI_NATIVE_BASE_URL_ENV] = f"{base_url.rstrip('/')}/gemini"
        # Gemini CLI recognizes several equivalent credential names, with the
        # selected alias varying by model family and CLI release. Point every
        # accepted alias at the gateway so Gemma cannot inherit a real Google
        # key from the sandbox and silently bypass usage/evidence capture.
        for key in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_GENERATIVE_AI_API_KEY",
        ):
            updated[key] = master_key
        return updated
    if agent == "claude-agent-acp":
        updated["ANTHROPIC_BASE_URL"] = base_url.rstrip("/")
        updated["ANTHROPIC_AUTH_TOKEN"] = master_key
        updated["ANTHROPIC_MODEL"] = route.model_alias
        updated[LITELLM_MODEL_VIA_ENV] = "1"
        for key in (
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
            "ANTHROPIC_BEDROCK_BASE_URL",
        ):
            updated.pop(key, None)
        return updated
    if agent == "pi-acp":
        updated["BENCHFLOW_PROVIDER_PROTOCOL"] = "openai-completions"
        updated["BENCHFLOW_PROVIDER_BASE_URL"] = openai_base_url
        updated["BENCHFLOW_PROVIDER_API_KEY"] = master_key
        updated["BENCHFLOW_PROVIDER_MODEL"] = route.model_alias
        updated["BENCHFLOW_PROVIDER_NAME"] = "litellm"
        return updated

    agent_cfg = AGENTS.get(agent)
    if agent_cfg and agent_cfg.env_mapping:
        for src, dst in agent_cfg.env_mapping.items():
            if src in updated:
                updated[dst] = updated[src]
    return updated


def _warn_token_capture_without_gateway(agent_env: dict[str, str], why: str) -> None:
    """Token capture happens in the gateway; say so when a run skips it."""
    flag = agent_env.get("BENCHFLOW_CAPTURE_TOKEN_LOGPROBS", "")
    if flag.strip().lower() in {"1", "true", "yes", "on"}:
        logger.warning(
            "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS is set but %s, so no token ids or "
            "logprobs will be recorded (capture runs in the gateway). Use an API "
            "key route, e.g. a vllm/ model, for token capture.",
            why,
        )


def proxy_install_record(
    *,
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
    environment: str,
    force_sandbox_local: bool = False,
) -> dict[str, object] | None:
    """Check the run's proxy extras and say what the proxy will install.

    None when the agent does not go through the proxy (no model, a scripted
    agent, a subscription login). Otherwise the extras
    (``BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS``) are checked against the
    route's locked set, raising ``ProxyExtraRequirementError`` naming the
    extra and the locked package, and the result is the ``litellm_proxy``
    block of the trial's config.json. Called before any sandbox starts.
    """
    if (
        model is None
        or not needs_litellm_runtime(agent, model)
        or uses_native_subscription_auth(agent, model, agent_env)
        or uses_direct_anthropic_endpoint(agent, model, agent_env)
    ):
        return None
    try:
        provider: str | None = resolve_litellm_route(model, agent_env).provider_name
    except ValueError:
        # The proxy start reports an unroutable model; the extras are still
        # checked against the plain set here.
        provider = None
    install = proxy_install(provider)
    install.check()
    sandbox_local = force_sandbox_local or environment in _SANDBOX_LOCAL_ENVIRONMENTS
    return install.record(location="sandbox" if sandbox_local else "host")


def _name_own_endpoint_model(
    agent: str, model: str | None, agent_env: dict[str, str]
) -> dict[str, str]:
    """Claude Code off the proxy on another provider's Anthropic-compatible
    endpoint (a direct provider such as MiMo or Kimi): the provider's model id
    in ``ANTHROPIC_MODEL`` when nothing set it there.

    claude-agent-acp 0.81 lists only Claude names for its model option and
    refuses any other, so the id goes in the env, whose model is the
    adapter's ``default``; connect then leaves the option unset
    (``acp.runtime._model_left_to_env``). Against Anthropic itself, or with
    the variable set, the env is unchanged.
    """
    cfg = AGENTS.get(agent)
    mapping = getattr(cfg, "env_mapping", None) or {}
    model_env = mapping.get("BENCHFLOW_PROVIDER_MODEL")
    url_env = mapping.get("BENCHFLOW_PROVIDER_BASE_URL")
    if (
        not model
        or getattr(cfg, "api_protocol", "") != "anthropic-messages"
        or not model_env
        or not url_env
        or agent_env.get(model_env)
    ):
        return agent_env
    host = (urlparse(agent_env.get(url_env) or "").hostname or "").lower()
    if not host or host == "anthropic.com" or host.endswith(".anthropic.com"):
        return agent_env
    return {**agent_env, model_env: strip_provider_prefix(model)}


async def _skip_litellm_runtime(
    agent_env: dict[str, str],
    runtime: Any | None,
    *,
    reason: str | None = None,
) -> tuple[dict[str, str], Any | None]:
    if runtime is not None:
        await stop_litellm_runtime(runtime)
    if reason:
        logger.info("Skipping LiteLLM proxy: %s", reason)
    return agent_env, None


async def _raise_litellm_unavailable(
    *,
    runtime: Any | None,
    error: str,
) -> NoReturn:
    """Fail closed when a routable agent cannot be put behind the proxy.

    BenchFlow always routes routable agents through the LiteLLM proxy so
    provider traffic is metered and captured (``llm_trajectory.jsonl``).
    Silently falling back to direct provider access would leak the raw key,
    bypass usage/cost tracking, and lose the trainable trajectory — so any
    proxy unavailability is fatal, independent of ``usage_tracking`` mode.
    """
    if runtime is not None:
        await stop_litellm_runtime(runtime)
    raise RuntimeError(error)


async def ensure_litellm_runtime(
    *,
    agent: str,
    agent_env: dict[str, str],
    model: str | None,
    runtime: Any | None,
    environment: str,
    session_id: str = "",
    usage_tracking: UsageTrackingConfig | dict[str, Any] | str | None = None,
    sandbox: Any | None = None,
    sandbox_setup_timeout: int = 120,
    required_skill_names: tuple[str, ...] = (),
    live_trajectory_path: Path | None = None,
    prior_runtimes: tuple[ProviderRuntime, ...] = (),
    force_sandbox_local: bool = False,
    install_layer_bypass_reason: str | None = None,
    disallow_web_tools: bool = False,
    trial_budget: TrialBudget | None = None,
) -> tuple[dict[str, str], Any | None]:
    """Start/reuse LiteLLM and rewrite the agent env to talk to it.

    Every LiteLLM-routable agent is *always* routed through the proxy so
    provider traffic is metered and captured (``llm_trajectory.jsonl``) and the
    raw provider key never reaches the agent. ``usage_tracking`` no longer gates
    whether the proxy runs — it only governs whether trusted telemetry is
    *required* (``required`` fails closed when usage cannot be captured at all).
    The only agents that skip the proxy are those that physically cannot be
    routed through it: ``oracle`` (no model) and native-subscription auth (no
    API key to proxy), and explicit token/URL pairs for non-Claude models in
    Claude Code. Direct endpoints cannot satisfy required capture or hosted-tool
    restrictions. Gemini uses LiteLLM's native GenerateContent endpoints.
    """
    # Re-entrant connects pass back proxy-owned env, which cannot reconstruct
    # upstream routing or credentials. Restore controller-held source config.
    if (
        runtime is not None
        and getattr(runtime, "kind", None) == "litellm"
        and getattr(runtime, "source_agent", None) == agent
        and getattr(runtime, "source_model", None) == model
        and agent_env.get(LITELLM_MASTER_KEY_ENV)
        == getattr(runtime, "master_key", None)
        and getattr(runtime, "source_env", None) is not None
    ):
        agent_env = dict(runtime.source_env)

    usage_cfg = UsageTrackingConfig.coerce(usage_tracking).with_env_defaults()

    if uses_direct_anthropic_endpoint(agent, model, agent_env):
        if usage_cfg.mode == "required" or disallow_web_tools:
            await _raise_litellm_unavailable(
                runtime=runtime,
                error="A direct Claude Code endpoint cannot enforce required usage "
                "capture or hosted-tool restrictions; configure a proxy provider route.",
            )
        _warn_token_capture_without_gateway(
            agent_env, "explicit direct Claude Code endpoint"
        )
        return await _skip_litellm_runtime(
            _name_own_endpoint_model(agent, model, agent_env),
            runtime,
            reason="explicit direct Claude Code endpoint; no proxy call capture or cost accounting",
        )

    if uses_native_subscription_auth(agent, model, agent_env):
        _warn_token_capture_without_gateway(
            agent_env, "the agent uses subscription (OAuth) auth"
        )
        return await _skip_litellm_runtime(
            _name_own_endpoint_model(agent, model, agent_env),
            runtime,
            reason="native subscription auth will use agent ACP usage telemetry",
        )

    if not needs_litellm_runtime(agent, model):
        if usage_cfg.mode == "required" and not is_scripted_agent(agent):
            raise RuntimeError(
                "Token usage tracking is required, but agent "
                f"{agent!r} cannot be routed through LiteLLM."
            )
        _warn_token_capture_without_gateway(
            agent_env, f"agent {agent!r} is not routed through the LiteLLM gateway"
        )
        return await _skip_litellm_runtime(
            agent_env,
            runtime,
            reason=(
                f'agent {agent!r} routes its own model calls (model_route = "agent")'
                if agent_routes_own_model(agent)
                else None
            ),
        )
    assert model is not None

    sandbox_local = not proxy_runs_on_host(
        environment, force_sandbox_local=force_sandbox_local
    )
    if sandbox_local and sandbox is None:
        raise RuntimeError("sandbox-local LiteLLM requires a sandbox handle")

    try:
        route = resolve_litellm_route(model, agent_env)
    except ValueError as exc:
        await _raise_litellm_unavailable(
            runtime=runtime,
            error=(
                "LiteLLM proxy is mandatory but cannot resolve "
                f"model {model!r}: {exc}. BenchFlow never sends provider traffic "
                "directly — register the provider/model or fix the route."
            ),
        )
    missing = _missing_required_env(route, agent_env)
    if missing:
        missing_text = ", ".join(missing)
        await _raise_litellm_unavailable(
            runtime=runtime,
            error=(
                f"LiteLLM route for model {model!r} requires {missing_text}. "
                "Export them in your shell or put them in a .env file (the "
                "proxy is mandatory; traffic is never sent directly to the "
                "provider)."
            ),
        )
    # With web access restricted, only a positively identified model runs
    # . Decided once, at launch, for both proxy placements, from
    # the controller-resolved model; the operator entry is read from the
    # controller's own environment, never the agent's.
    admission = None
    restricted_endpoint_env = {}
    if disallow_web_tools:
        from benchflow.providers.restricted_models import (
            RestrictedModelRefused,
            admit_restricted_route,
        )

        try:
            admitted, route, restricted_endpoint_env = admit_restricted_route(
                route, agent_env
            )
            admission = admitted.record()
        except RestrictedModelRefused as exc:
            await _raise_litellm_unavailable(runtime=runtime, error=str(exc))

    master_key = (
        agent_env.get(LITELLM_MASTER_KEY_ENV)
        or f"sk-runtime-{secrets.token_urlsafe(24)}"
    )
    skill_gate_key = json.dumps(
        sorted(set(required_skill_names)), separators=(",", ":")
    )
    proxy_location = "sandbox" if sandbox_local else "host"
    config_key = (
        f"{environment}:{proxy_location}:{route.config_key}:{agent}:"
        f"{session_id}:{skill_gate_key}:{disallow_web_tools}"
    )
    if runtime is not None and getattr(runtime, "kind", None) == "litellm":
        server = getattr(runtime, "server", None)
        if (
            getattr(runtime, "config_key", None) == config_key
            and getattr(runtime, "source_env", None) == agent_env
            and server is not None
        ):
            is_running = await server.is_running()
            if is_running:
                if live_trajectory_path is not None:
                    server.start_live_capture(live_trajectory_path)
                return (
                    _apply_litellm_agent_env(
                        agent=agent,
                        agent_env=agent_env,
                        route=route,
                        base_url=runtime.base_url,
                        master_key=getattr(runtime, "master_key", master_key),
                    ),
                    runtime,
                )
        await stop_litellm_runtime(runtime)

    try:
        proxy_env = _litellm_proxy_env(
            agent=agent,
            agent_env=agent_env,
            required_skill_names=required_skill_names,
        )
        proxy_env.update(restricted_endpoint_env)
        if trial_budget is not None:
            # The proxy enforces the trial's budget, counting on from what the
            # trial's earlier proxies (another scene role's) already counted.
            proxy_env.update(trial_budget.proxy_env(_budget_used(prior_runtimes)))
        # Controller policy overrides any role/env spelling before launch.
        proxy_env["BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS"] = (
            "1" if disallow_web_tools else "0"
        )
        if sandbox_local:
            server = await _start_sandbox_litellm(
                sandbox=sandbox,
                route=route,
                master_key=master_key,
                agent_env=proxy_env,
                session_id=session_id,
                agent_name=agent,
                install_timeout_sec=max(600, int(sandbox_setup_timeout)),
                install_layer_bypass_reason=install_layer_bypass_reason,
            )
        else:
            claimed = None
            if not disallow_web_tools and not agent_env.get(LITELLM_MASTER_KEY_ENV):
                # The proxy the trial reserved when its sandbox started, if it
                # is of this launch (_HostProxyPool); it has its own key.
                claimed = await _claim_host_proxy(
                    route=route,
                    agent=agent,
                    environment=environment,
                    proxy_env=proxy_env,
                )
            if claimed is not None:
                server, master_key = claimed
                server.session_id = session_id
                server.agent_name = agent
                logger.info(
                    "LiteLLM proxy listening on %s (started ahead)", server.base_url
                )
            else:
                server = await _start_host_litellm(
                    route=route,
                    master_key=master_key,
                    agent_env=proxy_env,
                    environment=environment,
                    session_id=session_id,
                    agent_name=agent,
                )
    except BedrockPatchPreflightError:
        raise
    except Exception as exc:
        await _raise_litellm_unavailable(
            runtime=None,
            error=(
                f"LiteLLM proxy failed to start for model {model!r}: "
                f"{describe_exception(exc)}. "
                "BenchFlow never sends provider traffic directly, so this is fatal."
            ),
        )

    from benchflow.providers.runtime import ProviderRuntime

    new_runtime = ProviderRuntime(
        kind="litellm",
        agent_base_url=server.base_url,
        backend_model=route.upstream_model,
        server=server,
        config_key=config_key,
        master_key=master_key,
        source_agent=agent,
        source_model=model,
        source_env=dict(agent_env),
        restricted_web_model=admission,
    )
    if live_trajectory_path is not None:
        # stop() may replace a sandbox process's trajectory with its imported
        # callback log. Read the runtime references only after that import.
        preceding = tuple(
            trajectory
            for prior in prior_runtimes
            if (
                trajectory := getattr(
                    getattr(prior, "server", None), "trajectory", None
                )
            )
            is not None
        )
        server.start_live_capture(
            live_trajectory_path, preceding_trajectories=preceding
        )
    return (
        _apply_litellm_agent_env(
            agent=agent,
            agent_env=agent_env,
            route=route,
            base_url=new_runtime.base_url,
            master_key=master_key,
        ),
        new_runtime,
    )


def _budget_used(runtimes: Iterable[Any]) -> dict[str, int]:
    """What the trial's stopped proxies counted against its budget: their
    calls (turns) and tokens, from their imported callback logs."""
    used = {"tokens": 0, "turns": 0}
    for prior in runtimes:
        trajectory = getattr(getattr(prior, "server", None), "trajectory", None)
        if trajectory is None:
            continue
        used["turns"] += len(trajectory.exchanges)
        used["tokens"] += int(trajectory.total_provider_tokens)
    return used


async def stop_litellm_runtime(runtime: Any | None) -> None:
    if runtime is None or getattr(runtime, "stopped", False):
        return
    server = getattr(runtime, "server", None)
    if getattr(runtime, "kind", None) == "litellm" and server is not None:
        await server.stop()
        # Only a completed import is final. A failed stop stays retryable.
        runtime.stopped = True


def extract_usage(runtime: Any | None) -> dict[str, Any]:
    if runtime is None or getattr(runtime, "kind", None) != "litellm":
        return usage_unavailable()
    server = getattr(runtime, "server", None)
    trajectory = getattr(server, "trajectory", None)
    return extract_usage_from_trajectory(
        trajectory,
        fallback_model=getattr(runtime, "backend_model", None),
    )
