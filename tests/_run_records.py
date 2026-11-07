"""Trial folders as a run writes them, for the stats tests.

A trial starts from its rawest record, the model proxy's log of each request
(``trajectory/llm_trajectory.jsonl``, one line per request in the proxy's
shape). BenchFlow's own writers then make the rest, as a rollout does:
``write_model_calls`` the per-call records (``model-calls.jsonl``) and their
totals (``model-calls.json``), ``settings_sent`` / ``effort_outcome`` /
``agent_outcome`` result.json's blocks. The harness receipt has the shape
``agents.receipt.probe_receipt`` records. Nothing of BenchFlow's is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchflow.run_checks import read_calls, settings_sent
from benchflow.trajectories.model_calls import (
    agent_outcome,
    effort_outcome,
    write_model_calls,
)

AGENT = "claude-agent-acp"
CLAUDE_ADAPTER = "@agentclientprotocol/claude-agent-acp"
CLAUDE_CLI = "@anthropic-ai/claude-code"


def receipt(
    agent: str = AGENT,
    adapter: tuple[str, str] | None = (CLAUDE_ADAPTER, "0.88.0"),
    cli: tuple[str, str] | None = (CLAUDE_CLI, "2.1.257"),
) -> dict[str, Any]:
    """A harness receipt (result.json's ``harness_versions``)."""

    def part(found: tuple[str, str] | None) -> dict[str, Any]:
        if found is None:
            return {"version": None, "source": "unavailable"}
        return {"package": found[0], "version": found[1], "source": "installed-package"}

    return {
        "schema": 1,
        "identity": {"agent": agent, "harness": "acp", "install_sha256": "0" * 64},
        "adapter": part(adapter),
        "cli": part(cli),
        "engine": part(cli),
    }


def anthropic_call(
    *,
    model: str,
    served: str,
    route: str,
    effort: str | None,
    status: int = 200,
    seed: int | None = None,
) -> dict[str, Any]:
    """One proxy log line for an Anthropic Messages request."""
    body: dict[str, Any] = {"model": model, "max_tokens": 128000, "messages": []}
    if effort is not None:
        body["output_config"] = {"effort": effort}
    if seed is not None:
        body["seed"] = seed
    answer = (
        {
            "model": served,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        }
        if status == 200
        else {"error": {"message": "upstream"}}
    )
    return {
        "metadata": {
            "call_purpose": "agent",
            "request_model": model,
            "provider_model": route,
        },
        "request": {"body": body, "timestamp": "2026-10-01T12:00:00+00:00"},
        "response": {"status_code": status, "body": answer},
        "duration_ms": 1200.0,
    }


def responses_call(
    *, model: str, served: str, route: str, effort: str | None, status: int = 200
) -> dict[str, Any]:
    """One proxy log line for an OpenAI Responses request (codex-acp)."""
    body: dict[str, Any] = {"model": model, "input": []}
    if effort is not None:
        body["reasoning"] = {"effort": effort}
    answer = (
        {
            "model": served,
            "status": "completed",
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        }
        if status == 200
        else {"error": {"message": "upstream"}}
    )
    return {
        "metadata": {
            "call_purpose": "agent",
            "request_model": model,
            "provider_model": route,
        },
        "request": {"body": body, "timestamp": "2026-10-01T12:00:00+00:00"},
        "response": {"status_code": status, "body": answer},
        "duration_ms": 1500.0,
    }


def chat_call(
    *, model: str, served: str, route: str, effort: str | None, status: int = 200
) -> dict[str, Any]:
    """One proxy log line for a Chat Completions request."""
    body: dict[str, Any] = {"model": model, "messages": []}
    if effort is not None:
        body["reasoning_effort"] = effort
    answer = (
        {
            "model": served,
            "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
        }
        if status == 200
        else {"error": {"message": "upstream"}}
    )
    return {
        "metadata": {
            "call_purpose": "agent",
            "request_model": model,
            "provider_model": route,
        },
        "request": {"body": body, "timestamp": "2026-10-01T12:00:00+00:00"},
        "response": {"status_code": status, "body": answer},
        "duration_ms": 900.0,
    }


class Template:
    """One trial's model-call records, written once by BenchFlow's own
    writers and copied into every trial of a row that made the same calls
    (thousands of trials of one setting carry the same records)."""

    def __init__(self, root: Path, calls: list[dict[str, Any]], model: str) -> None:
        folder = root / "template"
        (folder / "trajectory").mkdir(parents=True)
        (folder / "trajectory" / "llm_trajectory.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in calls)
        )
        self.totals = write_model_calls(folder)
        self.calls = (folder / "trajectory" / "model-calls.jsonl").read_text()
        self.summary = (folder / "model-calls.json").read_text()
        self.settings_sent = settings_sent(read_calls(folder), model)


def trial(
    job: Path,
    task: str,
    index: int,
    *,
    reward: float | None,
    model: str,
    effort: str | None,
    calls: list[dict[str, Any]] | None = None,
    agent: str = AGENT,
    harness_versions: dict[str, Any] | None = None,
    benchflow_version: str = "0.10.0",
    applied: str | None = None,
    error: str | None = None,
    error_category: str | None = None,
    dataset: tuple[str, str] | None = None,
    task_digest: str | None = None,
    extra_config: dict[str, Any] | None = None,
    template: Template | None = None,
) -> Path:
    """One trial folder: proxy log, per-call records, result.json, config.json.

    ``calls`` are proxy log lines (:func:`anthropic_call`); None or empty is a
    trial that made no model call (it errored first). ``template`` gives the
    per-call records a :class:`Template` wrote instead.
    """
    folder = job / f"{task}__{index:08x}"
    (folder / "trajectory").mkdir(parents=True)
    totals = None
    sent: dict[str, Any] | None = None
    if template is not None:
        (folder / "trajectory" / "model-calls.jsonl").write_text(template.calls)
        (folder / "model-calls.json").write_text(template.summary)
        totals, sent = template.totals, template.settings_sent
    elif calls:
        (folder / "trajectory" / "llm_trajectory.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in calls)
        )
        totals = write_model_calls(folder)
        sent = settings_sent(read_calls(folder), model)
    harness = harness_versions if harness_versions is not None else receipt(agent)
    result = {
        "task_name": task,
        "rollout_name": folder.name,
        "rewards": None if reward is None else {"reward": reward},
        "agent": agent,
        "agent_name": agent,
        "model": model,
        "harness_versions": harness,
        "n_tool_calls": 12,
        "agent_outcome": agent_outcome(
            totals, stop_reason="end_turn" if error is None else None
        ),
        "effort": effort_outcome(totals, requested=effort, applied=applied),
        "settings_sent": sent,
        "error": error,
        "error_category": error_category,
        "verifier_error": None,
        "started_at": "2026-10-01T12:00:00+00:00",
        "finished_at": "2026-10-01T12:30:00+00:00",
    }
    if task_digest is not None:
        result["task_digest"] = task_digest
    (folder / "result.json").write_text(json.dumps(result))
    config: dict[str, Any] = {
        "benchflow_version": benchflow_version,
        "task_path": task,
        "agent": agent,
        "model": model,
        "reasoning_effort": effort,
        "harness_mode": "acp",
        "timeout_sec": 28800,
        "harness_versions": harness,
        **({"task_digest": task_digest} if task_digest is not None else {}),
        **(extra_config or {}),
    }
    if dataset is not None:
        config["dataset_name"], config["dataset_version"] = dataset
    (folder / "config.json").write_text(json.dumps(config))
    return folder


def evaluation_job(job: Path, tasks: list[str], **config: Any) -> Path:
    """Mark ``job`` an Evaluation job (evaluation.json), as a run does."""
    job.mkdir(parents=True, exist_ok=True)
    (job / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "config": config, "planned_tasks": tasks})
    )
    return job
