"""OpenReward / ORS hosted-environment driver (v0.5 Track-2).

This is the OpenReward sibling of the PrimeIntellect ``vf-eval`` path in
:mod:`benchflow.hosted_env`. Instead of shelling out to ``vf-eval`` against
Prime's hub, it drives an OpenReward environment directly through the
``openreward`` Python client's session loop:

    c = openreward.OpenReward(api_key=...)
    env = c.environments.get("<owner>/<name>")
    task = env.get_task(split, index)
    with env.session(task=task) as session:
        session.get_prompt()
        session.list_tools()
        out = session.call_tool(name, input)   # -> ToolOutput
        # loop until out.finished; final reward is out.reward

The reward of the loop is ``ToolOutput.reward`` (NOT a ``RunResult`` /
``ToolResult``). ``client.rollout.create(...)`` is telemetry-only and is **not**
used here — it is not the env-interaction handle.

The loop is driven by a :class:`Policy`. We ship a :class:`ScriptedPolicy`
(schema-driven, no LLM) so the path is fully exercisable offline with a fake
session; :class:`ModelPolicy` is left as a clearly-marked seam for a real LLM
agent (PR3+).

Artifacts: the final reward is lifted into a canonical ``VerifyResult`` via
:func:`benchflow.rewards.node.verify_result_from_reward_map` (the single
dict→VerifyResult conversion point — we do not duplicate ``adapters/ors.py``),
and the BenchFlow rollout artifact contract is reconstructed by reusing the
same writers the rest of the codebase uses: the legacy ``result.json`` /
``rewards.jsonl`` / ``config.json`` / ``timing.json`` / ``prompts.json`` /
``trajectory/acp_trajectory.jsonl`` shape from :mod:`benchflow.hosted_env`,
plus the canonical ``verifier/verify_result.json`` and
``trainer/verifiers.jsonl`` from :mod:`benchflow.rollout` /
:mod:`benchflow.trajectories.export`. Lineage is stamped
``trajectory_source="openreward"``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from benchflow.diagnostics import RolloutDiagnostics
from benchflow.hosted_env import (
    HostedEnvError,
    HostedEnvRunConfig,
    HostedEnvRunResult,
    _hosted_source_provenance,
    _write_hosted_rewards_jsonl,
    normalize_verifiers_model,
)

if TYPE_CHECKING:
    from benchflow.rewards.protocol import VerifyResult

logger = logging.getLogger(__name__)

OPENREWARD_API_KEY_ENV_VARS = ("OPENREWARD_API_KEY", "ORS_API_KEY")

# Guard rail: a runaway env would otherwise loop forever if it never returns
# ``finished``. Keep this small — the scripted policy makes one structured call
# per tool, and real environments converge well under this.
_DEFAULT_MAX_STEPS = 50


class OpenRewardSession(Protocol):
    """The slice of ``openreward.environments...Session`` the driver uses.

    Declared structurally so the loop is testable with a fake session that
    records ``call_tool`` and returns scripted ``ToolOutput``-shaped objects —
    no network, no platform, no paid rollout.
    """

    def get_prompt(self) -> Any: ...
    def list_tools(self, format: Any = ...) -> Any: ...
    def call_tool(self, tool_name: str, input: Any = ...) -> Any: ...


class Policy(Protocol):
    """Chooses the next ``(tool_name, input)`` given the prompt and tools.

    Returning ``None`` ends the loop early (the policy has nothing more to do);
    otherwise the driver keeps calling tools until the environment reports
    ``finished`` or the step budget is exhausted.
    """

    def act(
        self,
        prompt_text: str,
        tools: list[Any],
        last_output: Any | None,
        step: int,
    ) -> tuple[str, dict[str, Any]] | None: ...


class ScriptedPolicy:
    """Deterministic, schema-driven policy — no LLM in the loop.

    Drives the environment by calling the available tools in order, sending an
    empty input by default. When an answer/submit-style tool is present (a tool
    whose name suggests it terminates the episode) it is preferred and called
    with ``answer`` populated from ``answer`` (if supplied) so an offline run
    can exercise the full terminate-on-``finished`` path. This is intentionally
    minimal: it exists to drive the loop in tests and as a smoke policy, not to
    solve tasks. A real agent plugs in via :class:`ModelPolicy`.
    """

    # Substrings that flag a tool as episode-terminating, in priority order.
    _SUBMIT_HINTS = ("submit", "answer", "finish", "done", "respond")

    def __init__(self, answer: str | None = None) -> None:
        self._answer = answer

    def _tool_name(self, tool: Any) -> str:
        # ToolSpec is a dataclass with ``.name``; dicts use ``["name"]``.
        if isinstance(tool, dict):
            return str(tool.get("name", ""))
        return str(getattr(tool, "name", ""))

    def _build_input(self, tool: Any) -> dict[str, Any]:
        name = self._tool_name(tool).lower()
        if self._answer is not None and any(h in name for h in self._SUBMIT_HINTS):
            return {"answer": self._answer}
        return {}

    def act(
        self,
        prompt_text: str,
        tools: list[Any],
        last_output: Any | None,
        step: int,
    ) -> tuple[str, dict[str, Any]] | None:
        if not tools:
            return None
        # Prefer a submit/answer-style tool so the episode can terminate.
        for tool in tools:
            name = self._tool_name(tool)
            if any(h in name.lower() for h in self._SUBMIT_HINTS):
                return name, self._build_input(tool)
        # Otherwise walk tools in order, one per step, then stop.
        if step < len(tools):
            tool = tools[step]
            return self._tool_name(tool), self._build_input(tool)
        return None


class ModelPolicy:
    """Seam for a real LLM-driven policy (PR3+).

    Not implemented offline. A future change wires a BenchFlow agent / model
    endpoint here: read ``prompt_text`` + ``tools`` (already provider-formatted
    via ``session.list_tools(format=...)``), ask the model for a tool call, and
    map its response to ``(tool_name, input)``. Kept as an explicit class so the
    driver's policy seam is visible and the scripted path stays the only thing
    shipped in PR2.
    """

    def __init__(self, model: str) -> None:
        self.model = model

    def act(
        self,
        prompt_text: str,
        tools: list[Any],
        last_output: Any | None,
        step: int,
    ) -> tuple[str, dict[str, Any]] | None:
        raise NotImplementedError(
            "ModelPolicy is a seam for a real LLM agent and is not wired yet "
            "(PR3+). Use ScriptedPolicy for offline runs."
        )


def _prompt_to_text(prompt: Any) -> str:
    """Render an openreward prompt (list of TextBlock/ImageBlock) to text.

    Best-effort: blocks expose ``.text`` (TextBlock) or are dict-shaped; images
    and unknown blocks are summarised, never raised on.
    """
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return str(prompt)
    parts: list[str] = []
    for block in prompt:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
        elif getattr(block, "type", None) == "image" or (
            isinstance(block, dict) and block.get("type") == "image"
        ):
            parts.append("[image]")
    return "\n".join(parts)


def _tool_output_to_text(output: Any) -> str:
    """Render a ToolOutput's blocks to a single text string (best-effort)."""
    blocks = getattr(output, "blocks", None)
    if blocks is None and isinstance(output, dict):
        blocks = output.get("blocks")
    if not isinstance(blocks, list):
        return ""
    return _prompt_to_text(blocks)


def _read_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def run_hosted_env_openreward(
    config: HostedEnvRunConfig,
    *,
    policy: Policy | None = None,
    split: str = "train",
    index: int = 0,
    session_factory: Any | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> HostedEnvRunResult:
    """Run a single OpenReward environment task and write BenchFlow artifacts.

    Drives the openreward session loop with *policy* (defaults to
    :class:`ScriptedPolicy`), terminating when a ``ToolOutput`` reports
    ``finished`` (or the step budget is hit). The final ``ToolOutput.reward``
    is lifted to a canonical ``VerifyResult`` and the full rollout artifact
    contract is written to ``run_dir``.

    ``session_factory`` is the offline/test seam: when supplied it is called as
    ``session_factory(config, split, index) -> context-manager yielding a
    session`` instead of opening a real ``OpenReward`` client. Production leaves
    it ``None`` and opens the client via :func:`_open_openreward_session`.

    Namespace/owner is REQUIRED for openreward (a bare name 400s on the
    platform) — raises :class:`HostedEnvError` before any session is opened.
    """
    ref = config.source_env
    if ref.provider != "openreward":
        raise HostedEnvError(
            f"run_hosted_env_openreward called with provider {ref.provider!r}; "
            "expected 'openreward'"
        )
    if ref.owner is None:
        raise HostedEnvError(
            "OpenReward requires an explicit owner/namespace "
            "(e.g. openreward:GeneralReasoning/KellyBench). A bare environment "
            "name is rejected by the platform."
        )

    policy = policy or ScriptedPolicy()
    run_dir = _make_run_dir(config)
    normalized_model = normalize_verifiers_model(config.model) if config.model else ""

    started_at = datetime.now(UTC)
    trajectory: list[dict[str, Any]] = []
    prompts: list[str] = []
    final_reward: float | None = None
    n_tool_calls = 0
    error: str | None = None

    factory = session_factory or _open_openreward_session
    try:
        with factory(config, split, index) as session:
            prompt = session.get_prompt()
            prompt_text = _prompt_to_text(prompt)
            if prompt_text:
                prompts.append(prompt_text)
                trajectory.append(
                    {
                        "type": "user_message",
                        "ts": started_at.isoformat(),
                        "content": prompt_text,
                    }
                )
            tools = list(session.list_tools())

            last_output: Any | None = None
            for step in range(max_steps):
                action = policy.act(prompt_text, tools, last_output, step)
                if action is None:
                    break
                tool_name, tool_input = action
                output = session.call_tool(tool_name, tool_input)
                n_tool_calls += 1
                last_output = output

                ts = datetime.now(UTC).isoformat()
                trajectory.append(
                    {
                        "type": "tool_call",
                        "ts": ts,
                        "title": tool_name,
                        "kind": tool_name,
                        "content": [{"type": "text", "text": json.dumps(tool_input)}],
                    }
                )
                out_text = _tool_output_to_text(output)
                if out_text:
                    trajectory.append(
                        {"type": "agent_message", "ts": ts, "content": out_text}
                    )

                reward = _read_attr(output, "reward")
                if isinstance(reward, (int, float)) and not isinstance(reward, bool):
                    final_reward = float(reward)
                if _read_attr(output, "finished", False):
                    break
            else:
                error = (
                    f"environment did not report finished within {max_steps} steps"
                )
    except HostedEnvError:
        raise
    except Exception as e:  # surface any driver/client failure as run error
        error = f"{type(e).__name__}: {e}"
        logger.warning("openreward run failed: %s", error)

    finished_at = datetime.now(UTC)

    result = HostedEnvRunResult(
        source_env=ref,
        run_dir=run_dir,
        command=["openreward", ref.env_id, f"split={split}", f"index={index}"],
        returncode=0 if error is None else 1,
        stdout="",
        stderr=error or "",
        model=config.model,
        normalized_model=normalized_model,
        reward=final_reward,
        total_tool_calls=n_tool_calls,
        verifiers_error=error,
    )

    _write_openreward_artifacts(
        result,
        config,
        trajectory=trajectory,
        prompts=prompts,
        started_at=started_at,
        finished_at=finished_at,
        split=split,
        index=index,
        error=error,
    )
    return result


def _make_run_dir(config: HostedEnvRunConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d__%H-%M-%S-%f")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.source_env.env_id)
    jobs_dir = config.jobs_dir.expanduser().resolve()
    run_id = f"{timestamp}__pid-{os.getpid()}__{uuid4().hex[:8]}"
    run_dir = jobs_dir / "hosted-env" / f"{safe_name}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _open_openreward_session(
    config: HostedEnvRunConfig,
    split: str,
    index: int,
) -> Any:
    """Open a real OpenReward session (production path; not used in tests).

    Returns a context manager that yields a live session. The ``OpenReward``
    client is created here and closed in the wrapper's ``__exit__`` (try/finally
    via ``close()``), so the event-loop thread the sync client spins up is torn
    down even on error.
    """
    return _OpenRewardSessionCtx(config, split, index)


class _OpenRewardSessionCtx:
    """Context manager: open client → get env → get task → open session.

    Owns the ``OpenReward`` client lifecycle (``close()`` in ``__exit__``).
    """

    def __init__(self, config: HostedEnvRunConfig, split: str, index: int) -> None:
        self._config = config
        self._split = split
        self._index = index
        self._client: Any = None
        self._session_cm: Any = None

    def __enter__(self) -> Any:
        import openreward

        api_key = next(
            (os.environ[k] for k in OPENREWARD_API_KEY_ENV_VARS if os.environ.get(k)),
            None,
        )
        if not api_key:
            raise HostedEnvError(
                "OPENREWARD_API_KEY is required to run an openreward environment"
            )
        self._client = openreward.OpenReward(api_key=api_key)
        try:
            env = self._client.environments.get(self._config.source_env.env_id)
            task = env.get_task(self._split, self._index)
            self._session_cm = env.session(task=task)
            return self._session_cm.__enter__()
        except Exception:
            self._close_client()
            raise

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._session_cm is not None:
                self._session_cm.__exit__(*exc)
        finally:
            self._close_client()

    def _close_client(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # best-effort teardown
                    logger.debug("openreward client close() failed", exc_info=True)


def _write_openreward_artifacts(
    result: HostedEnvRunResult,
    config: HostedEnvRunConfig,
    *,
    trajectory: list[dict[str, Any]],
    prompts: list[str],
    started_at: datetime,
    finished_at: datetime,
    split: str,
    index: int,
    error: str | None,
) -> None:
    """Write the BenchFlow rollout artifact contract for an openreward run.

    Mirrors :func:`benchflow.hosted_env._write_run_artifacts` (legacy
    ``result.json`` / ``rewards.jsonl`` / ``config.json`` / ``timing.json`` /
    ``prompts.json`` / ``trajectory/acp_trajectory.jsonl``) and adds the
    canonical Reward-plane artifacts (``verifier/verify_result.json``,
    ``trainer/verifiers.jsonl``) by reusing the shared writers. Lineage is
    stamped ``trajectory_source="openreward"``.
    """
    from benchflow.rewards.node import verify_result_from_reward_map

    run_dir = result.run_dir
    for sub in ("trajectory", "agent", "verifier", "artifacts"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    rewards = {"reward": float(result.reward)} if result.reward is not None else None
    verify_result = verify_result_from_reward_map(rewards, error=error)

    # trajectory/acp_trajectory.jsonl
    (run_dir / "trajectory" / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(e, default=str) for e in trajectory)
        + ("\n" if trajectory else "")
    )

    if not prompts:
        prompts = [f"<openreward:{result.source_env.env_uid} split={split} index={index}>"]

    timing = {"total": round((finished_at - started_at).total_seconds(), 1)}
    source_provenance = _hosted_source_provenance(
        result.source_env,
        runner="openreward",
        env_args=config.env_args,
    )
    source_provenance["split"] = split
    source_provenance["index"] = index

    result_payload: dict[str, Any] = {
        "task_name": result.source_env.env_uid,
        "rollout_name": run_dir.name,
        "rewards": rewards,
        "agent": config.agent or None,
        "agent_name": "openreward",
        "model": result.normalized_model or result.model or None,
        "n_tool_calls": result.total_tool_calls or 0,
        "n_prompts": len(prompts),
        "agent_result": {
            "n_tool_calls": result.total_tool_calls or 0,
            "n_prompts": len(prompts),
            "n_input_tokens": None,
            "n_output_tokens": None,
            "n_cache_read_tokens": None,
            "n_cache_creation_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
            "usage_source": "unavailable",
            "price_source": None,
        },
        "error": result.error,
        "error_category": None,
        "verifier_error": None,
        "verifier_error_category": None,
        **RolloutDiagnostics().to_result_fields(),
        "partial_trajectory": False,
        "trajectory_source": "openreward" if trajectory else None,
        "started_at": str(started_at),
        "finished_at": str(finished_at),
        "timing": timing,
        "scenes": [],
        "source": source_provenance,
    }
    (run_dir / "result.json").write_text(json.dumps(result_payload, indent=2))
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (run_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))

    config_payload: dict[str, Any] = {
        "task_path": None,
        "agent": config.agent or None,
        "model": result.normalized_model or result.model or None,
        "environment": "openreward",
        "skills_dir": None,
        "sandbox_user": None,
        "sandbox_locked_paths": None,
        "sandbox_setup_timeout": None,
        "context_root": None,
        "timeout_sec": None,
        "concurrency": config.concurrency,
        "agent_idle_timeout_sec": None,
        "started_at": str(started_at),
        "agent_env": {},
        "scenes": [],
        "source": source_provenance,
        "hosted_env": {
            "provider": result.source_env.provider,
            "env_uid": result.source_env.env_uid,
            "runner": "openreward",
            "split": split,
            "index": index,
            "num_examples": config.num_examples,
            "rollouts_per_example": config.rollouts_per_example,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "sampling_args": config.sampling_args,
            "env_args": config.env_args,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    if rewards:
        _write_hosted_rewards_jsonl(run_dir, rewards, finished_at)

    # Canonical Reward-plane artifacts — reuse the shared writers so the schema
    # never drifts from native rollouts.
    _write_verify_result_json(run_dir, verify_result)
    _write_verifiers_jsonl(
        run_dir,
        task_id=result.source_env.env_uid,
        prompts=prompts,
        trajectory=trajectory,
        verify_result=verify_result,
        model=result.normalized_model or result.model or None,
        is_completed=error is None,
    )


def _write_verify_result_json(
    run_dir: Path, verify_result: VerifyResult | None
) -> None:
    """Persist the canonical VerifyResult to ``verifier/verify_result.json``.

    Mirrors ``benchflow.rollout._write_verify_result_json`` (serialize the
    dataclass via ``asdict``) without importing it — ``benchflow.rollout``
    pulls heavy ACP/sandbox deps at import time, which this lightweight driver
    must not drag in.
    """
    if verify_result is None:
        return
    out = run_dir / "verifier" / "verify_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(verify_result), indent=2, default=str))


def _write_verifiers_jsonl(
    run_dir: Path,
    *,
    task_id: str,
    prompts: list[str],
    trajectory: list[dict[str, Any]],
    verify_result: VerifyResult,
    model: str | None,
    is_completed: bool,
) -> None:
    """Emit ``trainer/verifiers.jsonl`` via the shared trainer-export writer."""
    from benchflow.trajectories.export import write_rollout_verifiers_jsonl

    write_rollout_verifiers_jsonl(
        run_dir,
        task_id=task_id,
        prompts=prompts,
        trajectory=trajectory,
        verify_result=verify_result,
        model=model,
        environment="openreward",
        is_completed=is_completed,
    )
