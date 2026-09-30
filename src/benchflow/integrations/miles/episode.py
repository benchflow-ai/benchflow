"""One Miles rollout on a BenchFlow task.

Miles (radixark/miles) trains agents through a session server that keeps the
exact token ids and logprobs of every model call (token-in/token-out, "TITO").
For each rollout, its agent function receives a session-scoped
OpenAI-compatible URL. :func:`run_episode` plays one episode of the RL
cookbooks' shared harness against that URL:

- The task sandbox is a BenchFlow :class:`~benchflow.rollout.TaskRuntime` on
  Daytona or Docker, running as the non-root sandbox user.
- The policy gets the task prompt and two tools, ``run_bash`` and ``submit``:
  the tools, limits and prompt of ``benchflow.integrations.trl`` and
  ``docs/examples/rl/common``, so a policy trained here is evaluated with the
  harness it trained with.
- Every model call goes from this process straight to the session URL, without
  streaming and without any proxy in between, and each assistant message is
  sent back exactly as the session server returned it. That keeps the
  server's append-only history intact; a proxy that re-serializes messages
  would make the server roll back or reject turns.
- The sandbox never talks to the model. Its only traffic is the commands this
  process runs in it, so the policy's code has no route to the session server.
- The episode ends in a BenchFlow verdict, which the attribution rule of
  :mod:`benchflow.integrations.rewards` turns into a training reward or a
  drop. Drops are failures the policy cannot have caused (the sandbox never
  started, the model server failed, the verifier crashed on a sandbox the
  policy never touched); every other failure scores 0.

The outcome carries Miles's vocabulary as well: an ``exit_status`` for the
sample's metadata, and ``dropped`` for the samples the Miles agent function
discards with ``InfraAbort`` (radixark/miles#2801, #2802).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shlex
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import httpx

from benchflow.integrations.rewards import (
    INTEGRITY_VIOLATION,
    MODEL_ENDPOINT,
    NO_REWARD,
    RUN_ERROR,
    SANDBOX_START,
    SCORED,
    TIMEOUT,
    VERIFIER_CRASH_CLEAN_RUN,
    VERIFIER_ERROR,
    RewardDecision,
    dropped,
    reward_from_verify,
    sandbox_start_failure,
    zero,
)
from benchflow.integrations.trl.spec import (
    BashHarnessConfig,
    _truncate,
    bash_tool_schemas,
    write_rollout_record,
)
from benchflow.rollout import TaskRuntime, TaskRuntimeConfig

logger = logging.getLogger(__name__)

# The shared harness of the RL cookbooks (docs/examples/rl/common/harness.py).
# tests/test_miles_integration.py checks that these stay equal to it.
HARNESS_MESSAGE = (
    "\n\nYou are working in a Linux sandbox. Use the run_bash tool to run shell "
    "commands; each call starts in /workdir. When you are done, call the submit tool "
    "once with your final answer: it writes the answer to /workdir/answer.txt for you "
    "and ends the task. For a code fix, submit the word done."
)
BASH_TIMEOUT_SEC = 30
MAX_OUTPUT_CHARS = 2000
MAX_TURNS = 10
SUBMIT_PATH = "/workdir/answer.txt"

# How an episode ended (``ended``), before the verdict.
SUBMITTED = "submitted"
NO_TOOL_CALL = "no_tool_call"
TURN_LIMIT = "turn_limit"
RESPONSE_TRUNCATED = "response_truncated"
CONTEXT_EXHAUSTED = "context_exhausted"
REQUEST_REJECTED = "request_rejected"
TIME_LIMIT = "time_limit"
SANDBOX_NOT_STARTED = "sandbox_start"
MODEL_SERVER_FAILED = "model_endpoint"
GENERATION_ABORTED = "generation_aborted"
CANCELLED = "cancelled"

# Miles exit_status for a scored episode, by how it ended.
_ENDED_EXIT_STATUS = {
    SUBMITTED: "Submitted",
    NO_TOOL_CALL: "NoToolCall",
    TURN_LIMIT: "TurnLimitExceeded",
    RESPONSE_TRUNCATED: "SequenceLengthLimitExceeded",
    CONTEXT_EXHAUSTED: "SequenceLengthLimitExceeded",
    REQUEST_REJECTED: "RequestRejected",
    TIME_LIMIT: "TimeLimitExceeded",
}
# Miles exit_status for a failure scored 0, by the attribution reason.
_ZERO_EXIT_STATUS = {
    TIMEOUT: "TimeLimitExceeded",
    VERIFIER_ERROR: "VerifierError",
    RUN_ERROR: "AgentError",
    NO_REWARD: "NoReward",
    INTEGRITY_VIOLATION: "IntegrityViolation",
}
# Miles exit_status (the InfraAbort cause) for a dropped episode.
_DROP_EXIT_STATUS = {
    SANDBOX_START: "SandboxUnavailable",
    MODEL_ENDPOINT: "ModelEndpointFailed",
    VERIFIER_CRASH_CLEAN_RUN: "VerifierCrashCleanRun",
}
ABORTED_EXIT_STATUS = "Aborted"
GENERATION_ABORTED_EXIT_STATUS = "GenerationAborted"

_CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "prompt is too long",
    "longer than the model",
)
# Session-server answers the policy's own output can cause: 400 invalid messages
# or context overflow, 409 extending a truncated turn, 422 unparseable request,
# 500 token-in/token-out mismatch (a degenerate generation can cause it). When a
# case is ambiguous, it is the policy's: a few false zeros cost less than an
# outcome the policy can learn to trigger to escape a penalty.
_POLICY_CAUSED_STATUS = frozenset({400, 409, 422, 500})
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class EpisodeRequestError(ValueError):
    """The request cannot run: a configuration error, not an episode outcome."""


class _ModelServerFailure(Exception):
    """The session server failed in a way the policy cannot cause."""

    def __init__(self, message: str, *, aborted: bool = False) -> None:
        super().__init__(message)
        self.aborted = aborted


class _PolicyStopped(Exception):
    """The session server refused the conversation for a reason the policy owns."""

    def __init__(self, ended: str, message: str) -> None:
        super().__init__(message)
        self.ended = ended


@dataclass(frozen=True)
class EpisodeSettings:
    """How the environment server runs episodes; one value per server."""

    tasks_dir: Path
    harness: BashHarnessConfig = field(
        default_factory=lambda: BashHarnessConfig(
            environment="daytona",
            jobs_dir="jobs/miles",
            max_output_chars=MAX_OUTPUT_CHARS,
        )
    )
    max_turns: int = MAX_TURNS
    # The session server serves one policy; the name is only echoed back.
    model: str = "default"
    # Merged into every chat request, e.g. {"chat_template_kwargs": {...}}.
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    job_name: str | None = None
    request_timeout_sec: float = 300.0
    transport_retries: int = 2
    # Wall-clock cap from sandbox start to verdict. Hitting it scores 0.
    episode_timeout_sec: float = 1800.0
    # "audit" asks the task runtime for BenchShield's integrity verdict.
    integrity: str | None = None

    def normalized(self) -> EpisodeSettings:
        tasks_dir = Path(self.tasks_dir)
        if not tasks_dir.is_dir():
            raise ValueError(f"tasks_dir is not a directory: {tasks_dir}")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.episode_timeout_sec <= 0 or self.request_timeout_sec <= 0:
            raise ValueError("timeouts must be positive")
        if self.transport_retries < 0:
            raise ValueError("transport_retries must be >= 0")
        reserved = {"model", "messages", "tools", "tool_choice"} & set(self.extra_body)
        if reserved:
            raise ValueError(f"extra_body cannot set {sorted(reserved)}")
        if self.integrity not in (None, "off", "audit", "strict"):
            raise ValueError("integrity must be off, audit or strict")
        integrity = None if self.integrity in (None, "off") else self.integrity
        if integrity is not None and "integrity" not in {
            f.name for f in fields(TaskRuntimeConfig)
        }:
            raise ValueError(
                "this BenchFlow has no integrity audit (TaskRuntimeConfig.integrity); "
                "run with integrity off"
            )
        return replace(
            self,
            tasks_dir=tasks_dir,
            harness=self.harness.normalized(),
            extra_body=dict(self.extra_body),
            integrity=integrity,
        )


@dataclass(frozen=True)
class EpisodeRequest:
    """One rollout, as the Miles agent function sends it."""

    session_url: str
    prompt: list[dict[str, Any]]
    request_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @classmethod
    def from_json(cls, body: Any) -> EpisodeRequest:
        if not isinstance(body, dict):
            raise EpisodeRequestError("the request body must be a JSON object")
        url = body.get("session_url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise EpisodeRequestError("session_url must be an http(s) URL")
        prompt = body.get("prompt")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not all(isinstance(m, dict) and "role" in m for m in prompt)
        ):
            raise EpisodeRequestError(
                "prompt must be a non-empty list of chat messages; build it with "
                "`python -m benchflow.integrations.miles prepare`"
            )
        request_kwargs = body.get("request_kwargs") or {}
        metadata = body.get("metadata") or {}
        if not isinstance(request_kwargs, dict) or not isinstance(metadata, dict):
            raise EpisodeRequestError("request_kwargs and metadata must be objects")
        episode_id = body.get("episode_id")
        return cls(
            session_url=url.rstrip("/"),
            prompt=prompt,
            request_kwargs=request_kwargs,
            metadata=metadata,
            **({"episode_id": str(episode_id)} if episode_id else {}),
        )


def task_id_of(metadata: Mapping[str, Any]) -> str:
    """The task a sample names: ``instance_id`` (Miles's key) or ``task_id``."""

    value = metadata.get("instance_id") or metadata.get("task_id")
    if not isinstance(value, str) or not _SAFE_TASK_ID.match(value):
        raise EpisodeRequestError(
            f"sample metadata needs instance_id naming a task folder, got {value!r}"
        )
    return value


def resolve_task_dir(tasks_dir: Path, task_id: str) -> Path:
    root = tasks_dir.resolve()
    path = (root / task_id).resolve()
    if path.parent != root:
        raise EpisodeRequestError(f"task {task_id!r} escapes the tasks folder")
    if not path.is_dir():
        raise EpisodeRequestError(f"no task folder {task_id!r} under {tasks_dir}")
    return path


@dataclass
class EpisodeOutcome:
    """What one episode produced: the decision plus everything to audit it."""

    episode_id: str
    task_id: str
    decision: RewardDecision | None = None
    ended: str | None = None
    exit_status: str | None = None
    policy_acted: bool = False
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    clipped_replies: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Prompt plus reply tokens of the latest call: the episode's context length.
    context_tokens: int = 0
    rollout_dir: Path | None = None
    verifier: dict[str, Any] = field(default_factory=dict)
    integrity: Any = None
    detail: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def dropped(self) -> bool:
        return self.decision is None or self.decision.dropped

    def response(self) -> dict[str, Any]:
        """The JSON the environment server returns to the Miles agent function."""

        decision = self.decision
        report = {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "ended": self.ended,
            "reason": decision.reason if decision else None,
            "detail": (decision.detail if decision else None) or self.detail,
            "flagged": bool(decision and decision.flagged),
            "policy_acted": self.policy_acted,
            "rollout_dir": str(self.rollout_dir) if self.rollout_dir else None,
            **self.verifier,
        }
        if self.integrity is not None:
            report["integrity"] = _jsonable(self.integrity)
        t = self.timings
        metrics = {
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "clipped_replies": self.clipped_replies,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "context_tokens": self.context_tokens,
            "total_time": t.get("total_sec"),
            "env_setup_time": t.get("sandbox_start_sec"),
            "agent_run_time": t.get("agent_sec"),
            "eval_time": t.get("verify_sec"),
            "model_query_time_sum": t.get("model_sec"),
            "env_execution_time_sum": t.get("tool_sec"),
            "queue_time": t.get("queue_sec"),
            # Wall-clock time spent outside policy generation; Miles subtracts it
            # from throughput accounting (Sample.non_generation_time).
            "total_tool_time": sum(
                t.get(key) or 0.0
                for key in ("sandbox_start_sec", "tool_sec", "verify_sec")
            ),
        }
        return {
            "reward": decision.reward if decision else None,
            "dropped": self.dropped,
            "exit_status": self.exit_status,
            "flagged": bool(decision and decision.flagged),
            "detail": report["detail"],
            "eval_report": report,
            "agent_metrics": {k: v for k, v in metrics.items() if v is not None},
        }

    def record(self) -> dict[str, Any]:
        """The audit record kept in the rollout folder and ``rollouts.jsonl``."""

        decision = self.decision
        return {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "rollout_dir": str(self.rollout_dir) if self.rollout_dir else None,
            "policy_acted": self.policy_acted,
            **(
                decision.as_dict()
                if decision
                else {"reward": None, "dropped": True, "reason": None}
            ),
            "exit_status": self.exit_status,
            "ended": self.ended,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "clipped_replies": self.clipped_replies,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "context_tokens": self.context_tokens,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "integrity": _jsonable(self.integrity),
            "messages": self.messages,
        }


def exit_status_for(decision: RewardDecision | None, ended: str | None) -> str:
    """Miles's ``exit_status`` for an episode: why it ended, or why it failed."""

    if ended == CANCELLED or decision is None:
        return ABORTED_EXIT_STATUS
    if decision.dropped:
        if ended == GENERATION_ABORTED:
            return GENERATION_ABORTED_EXIT_STATUS
        return _DROP_EXIT_STATUS.get(decision.reason, "InfraFailure")
    if decision.reason == SCORED:
        return _ENDED_EXIT_STATUS.get(ended or "", "Submitted")
    return _ZERO_EXIT_STATUS.get(decision.reason, "AgentError")


RuntimeFactory = Callable[[TaskRuntimeConfig], Awaitable[Any]]


async def run_episode(
    request: EpisodeRequest,
    settings: EpisodeSettings,
    *,
    client: httpx.AsyncClient,
    sandbox_slots: asyncio.Semaphore | None = None,
    runtime_factory: RuntimeFactory | None = None,
    closers: set[asyncio.Task] | None = None,
) -> EpisodeOutcome:
    """Run one episode and return its outcome; always releases the sandbox.

    ``settings`` must be :meth:`EpisodeSettings.normalized`. ``sandbox_slots``
    bounds the sandboxes alive at once across episodes. Cancellation (the
    Miles abort hook, or the agent function going away) propagates after the
    sandbox is released; the caller reports it as a drop.
    """

    task_id = task_id_of(request.metadata)
    task_dir = resolve_task_dir(settings.tasks_dir, task_id)
    episode = _Episode(
        request,
        settings,
        client=client,
        task_dir=task_dir,
        task_id=task_id,
        runtime_factory=runtime_factory or TaskRuntime.create,
        closers=closers,
    )
    return await episode.run(sandbox_slots)


class _Episode:
    def __init__(
        self,
        request: EpisodeRequest,
        settings: EpisodeSettings,
        *,
        client: httpx.AsyncClient,
        task_dir: Path,
        task_id: str,
        runtime_factory: RuntimeFactory,
        closers: set[asyncio.Task] | None,
    ) -> None:
        self.request = request
        self.settings = settings
        self.client = client
        self.task_dir = task_dir
        self.runtime_factory = runtime_factory
        self.closers = closers
        self.runtime: Any = None
        self.out = EpisodeOutcome(episode_id=request.episode_id, task_id=task_id)
        self.out.messages = [dict(message) for message in request.prompt]
        # Miles passes --max-seq-len in the sample metadata: the most tokens of
        # one episode it will train on.
        self.max_seq_len = _positive_int(request.metadata.get("max_seq_len"))
        self.reply_budget = _positive_int(request.request_kwargs.get("max_tokens")) or 0

    async def run(self, sandbox_slots: asyncio.Semaphore | None) -> EpisodeOutcome:
        out = self.out
        t0 = time.monotonic()
        slots = sandbox_slots or contextlib.nullcontext()
        try:
            async with slots:
                out.timings["queue_sec"] = time.monotonic() - t0
                try:
                    await asyncio.wait_for(
                        self._play(), timeout=self.settings.episode_timeout_sec
                    )
                except TimeoutError:
                    out.ended = TIME_LIMIT
                    out.decision = zero(
                        TIMEOUT,
                        f"episode exceeded {self.settings.episode_timeout_sec:g}s",
                    )
                finally:
                    await self._close()
        except asyncio.CancelledError:
            out.ended = CANCELLED
            out.decision = None
            out.exit_status = ABORTED_EXIT_STATUS
            out.timings["total_sec"] = time.monotonic() - t0
            self._write_record()
            raise
        out.exit_status = exit_status_for(out.decision, out.ended)
        out.timings["total_sec"] = time.monotonic() - t0
        self._write_record()
        return out

    async def _play(self) -> None:
        out = self.out
        if not await self._start():
            return
        t_agent = time.monotonic()
        try:
            await self._loop()
        except _ModelServerFailure as exc:
            out.ended = GENERATION_ABORTED if exc.aborted else MODEL_SERVER_FAILED
            out.decision = dropped(MODEL_ENDPOINT, exc)
        except _PolicyStopped as exc:
            out.ended = exc.ended
            out.detail = str(exc)[:500]
        finally:
            out.timings["agent_sec"] = time.monotonic() - t_agent
        if out.decision is None:
            await self._verify()

    async def _start(self) -> bool:
        h = self.settings.harness
        kwargs: dict[str, Any] = {}
        if self.settings.integrity is not None:
            kwargs["integrity"] = self.settings.integrity
        config = TaskRuntimeConfig(
            task_path=self.task_dir,
            environment=h.environment,
            sandbox_user=h.sandbox_user,
            jobs_dir=h.jobs_dir,
            job_name=self.settings.job_name,
            planes=h.planes,
            **kwargs,
        )
        t = time.monotonic()
        try:
            self.runtime = await self.runtime_factory(config)
        except Exception as exc:
            self.out.ended = SANDBOX_NOT_STARTED
            self.out.decision = sandbox_start_failure(exc)
            logger.warning("sandbox for %s did not start: %s", self.out.task_id, exc)
            return False
        finally:
            self.out.timings["sandbox_start_sec"] = time.monotonic() - t
        self.out.rollout_dir = getattr(self.runtime, "rollout_dir", None)
        return True

    async def _loop(self) -> None:
        out = self.out
        max_turns = self.settings.max_turns
        # The evaluator's loop (docs/examples/rl/common/evaluate.py): up to
        # max_turns tool-calling turns, and one more model call whose tool calls
        # are not run.
        for turn in range(max_turns + 1):
            if turn and self._context_full():
                # The next request could outgrow what Miles trains on: stop, and
                # let the verifier score the sandbox as the policy left it.
                out.ended = CONTEXT_EXHAUSTED
                return
            choice = await self._chat()
            message = choice.get("message") or {}
            out.messages.append(_replayable(message))
            out.turns += 1
            if choice.get("finish_reason") == "length":
                # The reply was cut at max_tokens. The session server refuses to
                # extend a truncated turn, so the episode ends here, and the
                # verifier scores the sandbox as the policy left it.
                out.clipped_replies += 1
                out.ended = RESPONSE_TRUNCATED
                return
            calls = message.get("tool_calls") or []
            if not calls:
                out.ended = NO_TOOL_CALL
                return
            if turn == max_turns:
                out.ended = TURN_LIMIT
                return
            for call in calls:
                out.tool_calls += 1
                result, done = await self._tool(call)
                out.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": result,
                    }
                )
                if done:
                    out.ended = SUBMITTED
                    return

    def _context_full(self) -> bool:
        """Whether the next request, with its reply, could pass ``max_seq_len``.

        Counts the tokens of the latest call as the server reported them, the
        tool results appended since (about 3 characters per token, rounded up
        to stay on the safe side), and a full reply.
        """

        if not self.max_seq_len or not self.out.context_tokens:
            return False
        pending = 0
        for message in reversed(self.out.messages):
            if message.get("role") == "assistant":
                break
            pending += len(str(message.get("content") or "")) // 3 + 8
        used = self.out.context_tokens + pending + self.reply_budget
        return used > self.max_seq_len

    async def _chat(self) -> dict[str, Any]:
        out = self.out
        s = self.settings
        body = {
            **self.request.request_kwargs,
            **s.extra_body,
            "model": s.model,
            "messages": out.messages,
            "tools": _TOOLS,
            "tool_choice": "auto",
        }
        body.pop("stream", None)
        url = f"{self.request.session_url}/chat/completions"
        last: Exception | None = None
        for attempt in range(s.transport_retries + 1):
            if attempt:
                await asyncio.sleep(min(10.0, 2.0**attempt))
            t = time.monotonic()
            try:
                response = await self.client.post(
                    url, json=body, timeout=s.request_timeout_sec
                )
            except httpx.TransportError as exc:
                # The request may have reached the server; a retry re-sends the
                # same history, which the session server treats as a retry of the
                # latest turn.
                last = exc
                continue
            finally:
                out.timings["model_sec"] = out.timings.get("model_sec", 0.0) + (
                    time.monotonic() - t
                )
            if response.status_code == 502 and attempt < s.transport_retries:
                # The session server lost its SGLang backend for a moment; it
                # accepts the same request again as a retry of the latest turn.
                last = RuntimeError(f"HTTP 502: {response.text[:200]}")
                continue
            return self._parse_reply(response)
        raise _ModelServerFailure(
            f"session server failed after {s.transport_retries + 1} attempts: "
            f"{type(last).__name__}: {last}"
        )

    def _parse_reply(self, response: httpx.Response) -> dict[str, Any]:
        status = response.status_code
        if status == 200:
            try:
                data = response.json()
                choice = data["choices"][0]
                if not isinstance(choice.get("message"), dict):
                    raise TypeError("choice has no message")
            except Exception as exc:
                raise _ModelServerFailure(f"malformed chat response: {exc}") from exc
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            self.out.prompt_tokens += prompt_tokens
            self.out.completion_tokens += completion_tokens
            self.out.context_tokens = prompt_tokens + completion_tokens
            return choice
        text = response.text[:500]
        if status == 400 and any(marker in text.lower() for marker in _CONTEXT_MARKERS):
            raise _PolicyStopped(CONTEXT_EXHAUSTED, f"HTTP 400: {text}")
        if status in _POLICY_CAUSED_STATUS:
            raise _PolicyStopped(REQUEST_REJECTED, f"HTTP {status}: {text}")
        # 503 is the session server saying SGLang aborted the generation, which
        # Miles does when it has enough samples; 404, 502 and other statuses are
        # the trainer's side failing.
        raise _ModelServerFailure(f"HTTP {status}: {text}", aborted=status == 503)

    async def _tool(self, call: Mapping[str, Any]) -> tuple[str, bool]:
        """Run one tool call; return its result text and whether it ended the task."""

        out = self.out
        h = self.settings.harness
        function = call.get("function") or {}
        name = function.get("name")
        t = time.monotonic()
        try:
            raw = function.get("arguments")
            arguments = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            if name == "run_bash":
                command = str(arguments["command"])
                out.policy_acted = True
                result = await self.runtime.bash(
                    command, timeout_sec=h.bash_timeout_sec
                )
                output = result.stdout
                if result.stderr:
                    output = f"{output}{result.stderr}"
                return _truncate(output, h.max_output_chars), False
            if name == "submit":
                answer = shlex.quote(str(arguments["answer"]))
                path = shlex.quote(h.submit_path)
                out.policy_acted = True
                await self.runtime.bash(
                    f"mkdir -p $(dirname {path}) && printf %s {answer} > {path}",
                    timeout_sec=h.bash_timeout_sec,
                )
                return "submission recorded", True
            raise ValueError(f"Tool {name} not found.")
        except Exception as exc:  # the same feedback the TRL adapter gives
            out.tool_errors += 1
            return json.dumps({"error": str(exc)}), False
        finally:
            out.timings["tool_sec"] = out.timings.get("tool_sec", 0.0) + (
                time.monotonic() - t
            )

    async def _verify(self) -> None:
        out = self.out
        runtime = self.runtime
        t = time.monotonic()
        try:
            result = await runtime.verify()
        except Exception as exc:
            out.decision = (
                zero(VERIFIER_ERROR, exc)
                if out.policy_acted
                else dropped(VERIFIER_CRASH_CLEAN_RUN, exc)
            )
            return
        finally:
            out.timings["verify_sec"] = time.monotonic() - t
        out.rollout_dir = getattr(result, "rollout_dir", None) or out.rollout_dir
        out.verifier = {
            "rewards": getattr(result, "rewards", None),
            "verifier_error": getattr(result, "verifier_error", None),
            "error": getattr(result, "error", None),
        }
        out.integrity = getattr(result, "integrity", None)
        try:
            out.decision = reward_from_verify(
                result, policy_acted=out.policy_acted, integrity=out.integrity
            )
        except ValueError as exc:
            # A verdict without a boolean ``exploited`` is not a finding; keep the
            # verifier's decision and say so.
            logger.warning("unreadable integrity verdict for %s: %s", out.task_id, exc)
            out.decision = reward_from_verify(result, policy_acted=out.policy_acted)
            out.detail = f"integrity verdict ignored: {exc}"

    async def _close(self) -> None:
        runtime, self.runtime = self.runtime, None
        if runtime is None:
            return
        closing = asyncio.ensure_future(_close_quietly(runtime, self.out.task_id))
        if self.closers is not None:
            self.closers.add(closing)
            closing.add_done_callback(self.closers.discard)
        # Shielded: a cancelled episode still releases its sandbox.
        await asyncio.shield(closing)

    def _write_record(self) -> None:
        try:
            write_rollout_record(self.out.record(), self.settings.harness.jobs_dir)
        except OSError as exc:  # never let audit bookkeeping fail an episode
            logger.warning("could not record episode %s: %s", self.out.episode_id, exc)


async def _close_quietly(runtime: Any, task_id: str) -> None:
    try:
        await runtime.close()
    except Exception as exc:
        logger.warning("closing the sandbox of %s failed: %s", task_id, exc)


def _replayable(message: Mapping[str, Any]) -> dict[str, Any]:
    """The assistant message to send back, exactly as the server returned it.

    Only the fields a chat template reads are kept (role, content,
    reasoning_content, tool_calls); the session server compares those when it
    matches a replayed history against its stored turns.
    """

    replay: dict[str, Any] = {
        "role": message.get("role") or "assistant",
        "content": message.get("content"),
    }
    if message.get("reasoning_content") is not None:
        replay["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        replay["tool_calls"] = message["tool_calls"]
    return replay


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool | list | dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


_TOOLS: Sequence[dict[str, Any]] = bash_tool_schemas()


__all__ = [
    "ABORTED_EXIT_STATUS",
    "BASH_TIMEOUT_SEC",
    "HARNESS_MESSAGE",
    "MAX_OUTPUT_CHARS",
    "MAX_TURNS",
    "SUBMIT_PATH",
    "EpisodeOutcome",
    "EpisodeRequest",
    "EpisodeRequestError",
    "EpisodeSettings",
    "exit_status_for",
    "resolve_task_dir",
    "run_episode",
    "task_id_of",
]
