"""Harness for BenchFlow's deterministic integration tier.

Every scenario runs the real ``bench`` CLI against a real sandbox with
``claude-agent-acp`` whose model is a scripted fake provider
(``task/environment/fake_llm``), so agent behaviour is exercised end to end
without model credentials or cost. See ``tests/integration/README.md``.

The fake is the task's environment-plane service (``task/environment.toml``)
at ``http://127.0.0.1:8911`` inside the sandbox: BenchFlow starts it after
sandbox start and restarts it after every branch/checkpoint restore. Two
routes reach it:

- ``proxy`` (API-key route, BenchFlow's default for ``ANTHROPIC_API_KEY``): the
  agent talks to BenchFlow's LiteLLM proxy, whose upstream is the fake, set
  with ``BENCHFLOW_PROVIDER_BASE_URL``. This route records provider usage and
  cost. The fake must run where the proxy runs: in the sandbox on Daytona (and
  every ``model_proxy=sandbox`` provider); on Docker the proxy runs on the
  host, so the harness runs the same fake on the host too.
- ``vllm`` / ``sglang`` (self-hosted policy routes, ``--model vllm/fake-policy``
  or ``sglang/fake-policy``): like ``proxy``, but the fake answers OpenAI chat
  completions with token ids and logprobs shaped like vLLM or SGLang, and
  token capture is on, so ``llm_trajectory.jsonl`` carries per-call ids.
- ``native`` (subscription route, ``ANTHROPIC_AUTH_TOKEN``): no proxy; Claude
  Code calls ``ANTHROPIC_BASE_URL`` itself, the in-sandbox fake. Branching is
  only allowed without a provider runtime (``rollout_branch``: "Branching an
  active provider runtime needs a runtime fork contract"), so the branch
  scenarios take this route.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TEMPLATE_TASK = HERE / "task"
FAKE_LLM_DIR = TEMPLATE_TASK / "environment" / "fake_llm"
GOLDEN_DIR = HERE / "golden"
IN_SANDBOX_FAKE_URL = "http://127.0.0.1:8911"

AGENT = "claude-agent-acp"
MODEL = "claude-haiku-4-5"
# Codex runs against the fake's Responses API route through the proxy. gpt-5.4
# is in the model catalog of the Codex release both harnesses pin (0.156.1),
# so Codex offers its full tool surface.
CODEX_AGENT = "codex-acp"
CODEX_MODEL = "gpt-5.4"
# Not a credential: the fake provider accepts any key. The LiteLLM route needs
# one to be present, and it never leaves the sandbox/host proxy.
DUMMY_KEY = "fake-deterministic-key"

SANDBOX_ENV = "BENCHFLOW_DETERMINISTIC_SANDBOX"
UPDATE_GOLDEN_ENV = "BENCHFLOW_UPDATE_GOLDEN"
KEEP_JOBS_ENV = "BENCHFLOW_DETERMINISTIC_JOBS_DIR"

# Fixed by fake_llm.USAGE / SIDE_USAGE and LiteLLM's claude-haiku-4-5 prices.
MAIN_CALL_USAGE = {"input": 1000, "output": 50}
SIDE_CALL_USAGE = {"input": 0, "output": 1}
PRICE_PER_TOKEN = {"input": 1e-06, "output": 5e-06}


# ---------------------------------------------------------------------------
# Sandbox selection
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=20, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def select_sandbox() -> tuple[str | None, str]:
    """Return (sandbox, reason). ``sandbox`` is None when the tier must skip.

    ``BENCHFLOW_DETERMINISTIC_SANDBOX`` picks the backend explicitly
    (``docker``, ``daytona``, or ``off``). Unset, Docker is used when a daemon
    answers; Daytona is never chosen implicitly because it spends real
    sandbox time.
    """
    requested = os.environ.get(SANDBOX_ENV, "").strip().lower()
    if requested in {"off", "none", "0", "false"}:
        return (
            None,
            f"{SANDBOX_ENV}={requested}: deterministic integration tier disabled",
        )
    if requested == "daytona":
        if not os.environ.get("DAYTONA_API_KEY"):
            return None, f"{SANDBOX_ENV}=daytona but DAYTONA_API_KEY is not set"
        return "daytona", "daytona (requested)"
    if requested == "docker" or not requested:
        if _docker_available():
            return "docker", "docker"
        why = (
            "Docker was requested but `docker info` failed"
            if requested
            else "no Docker daemon answers `docker info`"
        )
        return None, (
            f"deterministic integration tier needs a sandbox: {why}. "
            f"Start Docker, or set {SANDBOX_ENV}=daytona with DAYTONA_API_KEY."
        )
    return None, f"{SANDBOX_ENV}={requested!r} is not one of docker, daytona, off"


def proxy_runs_in_sandbox(sandbox: str) -> bool:
    from benchflow.sandbox.providers import SANDBOX_MODEL_PROXY_PROVIDERS

    return sandbox in SANDBOX_MODEL_PROXY_PROVIDERS


# ---------------------------------------------------------------------------
# Host-side fake provider (Docker: the LiteLLM proxy runs on the host)
# ---------------------------------------------------------------------------


def _load_fake_llm_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "benchflow_det_fake_llm", FAKE_LLM_DIR / "fake_llm.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def host_fake_llm(log_path: Path | None = None) -> Iterator[str]:
    """Run the fake provider on the host; yield its base URL."""
    fake = _load_fake_llm_module()
    fake._Handler.scripts = json.loads((FAKE_LLM_DIR / "scripts.json").read_text())
    fake._Handler.log_path = log_path
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake._Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Scenario tasks
# ---------------------------------------------------------------------------

VERIFIER_ERROR_TEST_SH = """#!/bin/bash
# Deliberately broken verifier: exits non-zero without writing a reward.
echo "verifier crashed on purpose" >&2
exit 3
"""


@dataclass(frozen=True)
class TaskVariant:
    """One scenario's copy of the template task.

    Only the instruction's ``[[fake-llm:NAME]]`` marker, the agent timeout and
    the verifier differ; ``environment/`` is byte-identical, so every variant
    shares one image.
    """

    name: str
    script: str
    agent_timeout_sec: float | None = None
    broken_verifier: bool = False


def materialize_task(variant: TaskVariant, dest_root: Path) -> Path:
    dest = dest_root / variant.name
    shutil.copytree(TEMPLATE_TASK, dest)
    task_md = dest / "task.md"
    text = task_md.read_text()
    text, n = re.subn(
        r"\[\[fake-llm:[A-Za-z0-9_.-]+\]\]", f"[[fake-llm:{variant.script}]]", text
    )
    assert n == 1, "template task.md must carry exactly one fake-llm marker"
    if variant.agent_timeout_sec is not None:
        text, n = re.subn(
            r"(?m)^(agent:\n  timeout_sec: )[0-9.]+$",
            rf"\g<1>{variant.agent_timeout_sec}",
            text,
        )
        assert n == 1, "template task.md must set agent.timeout_sec"
    task_md.write_text(text)
    if variant.broken_verifier:
        test_sh = dest / "tests" / "test.sh"
        test_sh.write_text(VERIFIER_ERROR_TEST_SH)
        test_sh.chmod(0o755)
    return dest


def marker(script: str) -> str:
    return f"[[fake-llm:{script}]]"


# ---------------------------------------------------------------------------
# Running the CLI
# ---------------------------------------------------------------------------


@dataclass
class CliRun:
    args: list[str]
    returncode: int
    output: str
    seconds: float
    jobs_dir: Path

    def job_dirs(self) -> list[Path]:
        return sorted(p for p in self.jobs_dir.iterdir() if p.is_dir())

    def trial_dirs(self) -> list[Path]:
        out: list[Path] = []
        for job in self.job_dirs():
            out.extend(sorted(p.parent for p in job.glob("*/result.json")))
        return out

    def trial(self, task_name: str) -> Path:
        matches = [p for p in self.trial_dirs() if p.name.startswith(f"{task_name}__")]
        assert len(matches) == 1, (
            f"expected one trial for {task_name}, found {matches}\n{self.output[-4000:]}"
        )
        return matches[0]


def bench_executable() -> str:
    sibling = Path(sys.executable).with_name("bench")
    return str(sibling) if sibling.exists() else "bench"


# Self-hosted policy routes: the fake answers OpenAI chat completions with
# token ids and logprobs, shaped like vLLM under /v1 and SGLang under
# /sglang/v1, and the gateway captures them (BENCHFLOW_CAPTURE_TOKEN_LOGPROBS).
POLICY_ROUTES = {
    "vllm": ("vllm/fake-policy", "/v1"),
    "sglang": ("sglang/fake-policy", "/sglang/v1"),
}


def route_model(route: str) -> str:
    """The ``--model`` a route runs with."""
    if route == "codex":
        return CODEX_MODEL
    return POLICY_ROUTES[route][0] if route in POLICY_ROUTES else MODEL


def route_agent(route: str) -> str:
    """The ``--agent`` a route runs (Codex for the ``codex`` route)."""
    return CODEX_AGENT if route == "codex" else AGENT


def route_env(route: str, sandbox: str, host_fake_url: str | None) -> dict[str, str]:
    """The agent env that points ``claude-agent-acp`` at the fake provider."""
    if route in POLICY_ROUTES:
        base = IN_SANDBOX_FAKE_URL if proxy_runs_in_sandbox(sandbox) else host_fake_url
        assert base, "a host LiteLLM proxy needs the host fake provider URL"
        return {
            "BENCHFLOW_PROVIDER_BASE_URL": base + POLICY_ROUTES[route][1],
            "BENCHFLOW_PROVIDER_API_KEY": DUMMY_KEY,
            "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS": "1",
        }
    if route == "native":
        return {
            "ANTHROPIC_AUTH_TOKEN": DUMMY_KEY,
            "ANTHROPIC_BASE_URL": IN_SANDBOX_FAKE_URL,
        }
    if route == "codex":
        # The proxy's upstream is the fake's Responses API (openai/ routes
        # append /responses to the base URL).
        base = IN_SANDBOX_FAKE_URL if proxy_runs_in_sandbox(sandbox) else host_fake_url
        assert base, "a host LiteLLM proxy needs the host fake provider URL"
        return {"OPENAI_API_KEY": DUMMY_KEY, "BENCHFLOW_PROVIDER_BASE_URL": base + "/v1"}
    assert route == "proxy", route
    base = IN_SANDBOX_FAKE_URL if proxy_runs_in_sandbox(sandbox) else host_fake_url
    assert base, "a host LiteLLM proxy needs the host fake provider URL"
    return {"ANTHROPIC_API_KEY": DUMMY_KEY, "BENCHFLOW_PROVIDER_BASE_URL": base}


def check_route_is_hermetic(route: str, agent_env: dict[str, str]) -> None:
    """Refuse a route that would upload host credentials or skip the fake.

    With no explicit token BenchFlow falls back to host subscription files
    (``~/.claude/.credentials.json``); an explicit dummy token must prevent
    that, and the resolved base URL must be the fake.
    """
    from benchflow.agents.env import resolve_agent_env, uses_native_subscription_auth

    model = route_model(route)
    agent = route_agent(route)
    resolved = resolve_agent_env(agent, model, dict(agent_env))
    assert "_BENCHFLOW_SUBSCRIPTION_AUTH" not in resolved, (
        "would upload host credentials"
    )
    native = uses_native_subscription_auth(agent, model, resolved)
    assert native == (route == "native"), (route, native)
    if route == "native":
        assert resolved.get("ANTHROPIC_BASE_URL") == IN_SANDBOX_FAKE_URL
    else:
        assert (
            resolved.get("BENCHFLOW_PROVIDER_BASE_URL")
            == agent_env["BENCHFLOW_PROVIDER_BASE_URL"]
        )


def agent_env_args(agent_env: dict[str, str]) -> list[str]:
    args: list[str] = []
    for key, value in agent_env.items():
        args += ["--agent-env", f"{key}={value}"]
    return args


def bench_command(
    subcommand: Sequence[str],
    *,
    tasks_dir: Path,
    jobs_dir: Path,
    sandbox: str,
    host_fake_url: str | None,
    route: str = "proxy",
    extra: Sequence[str] = (),
) -> tuple[list[str], dict[str, str]]:
    """The ``bench`` argv and process env for one scripted run (checked hermetic)."""
    agent_env = route_env(route, sandbox, host_fake_url)
    args = [
        bench_executable(),
        *subcommand,
        "--tasks-dir",
        str(tasks_dir),
        "--agent",
        route_agent(route),
        "--model",
        route_model(route),
        "--sandbox",
        sandbox,
        "--jobs-dir",
        str(jobs_dir),
        *agent_env_args(agent_env),
        *extra,
    ]
    env = dict(os.environ)
    # A developer's real provider settings must not leak into a scripted run.
    for key in list(env):
        if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_", "CODEX_")) or key in {
            "BENCHFLOW_PROVIDER_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY",
        }:
            env.pop(key)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["BENCHFLOW_SKIP_UPDATE_CHECK"] = "1"
    # Every request each harness sends into the proxy, for wire parity.
    env["BENCHFLOW_LITELLM_WIRE_LOG_DIR"] = str(jobs_dir / "wire")
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k not in env}
    try:
        check_route_is_hermetic(route, agent_env)
    finally:
        os.environ.update(saved)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return args, env


def run_bench(
    subcommand: Sequence[str],
    *,
    tasks_dir: Path,
    jobs_dir: Path,
    sandbox: str,
    host_fake_url: str | None,
    route: str = "proxy",
    extra: Sequence[str] = (),
    timeout_sec: float = 1500,
) -> CliRun:
    args, env = bench_command(
        subcommand,
        tasks_dir=tasks_dir,
        jobs_dir=jobs_dir,
        sandbox=sandbox,
        host_fake_url=host_fake_url,
        route=route,
        extra=extra,
    )
    started = time.monotonic()
    proc = subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )
    seconds = time.monotonic() - started
    (jobs_dir / "cli-output.txt").write_text(proc.stdout)
    return CliRun(args, proc.returncode, proc.stdout, seconds, jobs_dir)


# ---------------------------------------------------------------------------
# Reading and checking artifacts
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def llm_calls(trial_dir: Path) -> list[dict[str, Any]]:
    """The provider calls the LiteLLM proxy captured, in order, normalised.

    ``main`` calls carry tools (Claude Code's agent loop); ``side`` calls do
    not (session-title generation and similar). Ids and timestamps dropped.
    """
    path = trial_dir / "trajectory" / "llm_trajectory.jsonl"
    if not path.is_file():
        return []
    calls = []
    for row in read_jsonl(path):
        request = (row.get("request") or {}).get("body") or {}
        response = (row.get("response") or {}).get("body") or {}
        if isinstance(response.get("output"), list):
            calls.append(_responses_call(request, response))
            continue
        usage = response.get("usage") or {}
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        calls.append(
            {
                "kind": "main" if request.get("tools") else "side",
                "n_messages": len(request.get("messages") or []),
                "finish_reason": choice.get("finish_reason"),
                "text": message.get("content"),
                "tool_calls": [
                    {
                        "id": tc.get("id"),
                        "name": (tc.get("function") or {}).get("name"),
                        "arguments": json.loads(
                            (tc.get("function") or {}).get("arguments") or "{}"
                        ),
                    }
                    for tc in tool_calls
                ],
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            }
        )
    return calls


def _responses_call(request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """One Responses API exchange (Codex) in :func:`llm_calls`' shape."""
    usage = response.get("usage") or {}
    output = [item for item in response.get("output") or [] if isinstance(item, dict)]
    texts = [
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content") or []
        if isinstance(part, dict)
    ]
    calls = [item for item in output if item.get("type") == "function_call"]
    return {
        "kind": "main" if request.get("tools") else "side",
        "n_messages": len(request.get("input") or []),
        "finish_reason": "tool_calls" if calls else "stop",
        "text": "".join(texts) or None,
        "tool_calls": [
            {
                "id": call.get("call_id"),
                "name": call.get("name"),
                "arguments": json.loads(call.get("arguments") or "{}"),
            }
            for call in calls
        ],
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def expected_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    n_main = sum(c["kind"] == "main" for c in calls)
    n_side = len(calls) - n_main
    tokens_in = n_main * MAIN_CALL_USAGE["input"] + n_side * SIDE_CALL_USAGE["input"]
    tokens_out = n_main * MAIN_CALL_USAGE["output"] + n_side * SIDE_CALL_USAGE["output"]
    cost = tokens_in * PRICE_PER_TOKEN["input"] + tokens_out * PRICE_PER_TOKEN["output"]
    return {
        "n_input_tokens": tokens_in,
        "n_output_tokens": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "cost_usd": round(cost, 9),
    }


_ACP_VOLATILE = {"receipt", "ts", "started_at", "finished_at", "timestamp"}


def normalized_acp(trial_dir: Path) -> list[dict[str, Any]]:
    path = trial_dir / "trajectory" / "acp_trajectory.jsonl"
    if not path.is_file():
        return []
    return [_strip(event, _ACP_VOLATILE) for event in read_jsonl(path)]


def normalized_atif(trial_dir: Path) -> dict[str, Any] | None:
    path = trial_dir / "trainer" / "atif.json"
    if not path.is_file():
        return None
    doc = read_json(path)
    doc.pop("session_id", None)
    return _strip(doc, {"timestamp"})


def _strip(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [_strip(v, keys) for v in value]
    return value


def atif_problems(doc: dict[str, Any]) -> list[str]:
    """ATIF-v1.7 structural rules (the ones ``export_atif`` promises)."""
    problems: list[str] = []
    if doc.get("schema_version") != "ATIF-v1.7":
        problems.append(f"schema_version={doc.get('schema_version')!r}")
    agent = doc.get("agent")
    if not isinstance(agent, dict) or not agent.get("name"):
        problems.append("agent.name missing")
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        return [*problems, "steps must be a non-empty array"]
    for index, step in enumerate(steps, start=1):
        where = f"steps[{index - 1}]"
        if step.get("step_id") != index:
            problems.append(
                f"{where}.step_id={step.get('step_id')!r}, expected {index}"
            )
        source = step.get("source")
        if source not in {"user", "agent", "system"}:
            problems.append(f"{where}.source={source!r}")
        if source != "agent":
            for key in ("tool_calls", "reasoning_content", "metrics"):
                if key in step:
                    problems.append(f"{where}.{key} only allowed on agent steps")
        call_ids = {tc.get("tool_call_id") for tc in step.get("tool_calls") or []}
        for result in (step.get("observation") or {}).get("results") or []:
            ref = result.get("source_call_id")
            if ref is not None and ref not in call_ids:
                problems.append(
                    f"{where}.observation source_call_id {ref!r} not in this step"
                )
    metrics = doc.get("final_metrics")
    if isinstance(metrics, dict) and metrics.get("total_steps") not in (
        None,
        len(steps),
    ):
        problems.append("final_metrics.total_steps does not match steps")
    return problems


def trial_document_problems(trial_dir: Path) -> list[str]:
    """Validate the public ``benchflow.trial`` export against its JSON schema."""
    import jsonschema

    import benchflow as bf

    schema = read_json(
        REPO_ROOT / "docs/reference/schemas/benchflow-trial.v1.schema.json"
    )
    document = bf.load_trial(trial_dir).to_json_dict()
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(map(str, e.absolute_path))}: {e.message}"
        for e in sorted(validator.iter_errors(document), key=str)
    ]


def release_kept_checkpoints(trial_dir: Path, sandbox: str) -> list[str]:
    """Delete the sandbox snapshots a trial kept (``checkpoints.json``).

    Kept checkpoints outlive the run on purpose (``--from-checkpoint``); the
    tier deletes its own so no Daytona snapshot or Docker image is left.
    Returns the refs that could not be deleted.
    """
    path = trial_dir / "checkpoints.json"
    if not path.is_file():
        return []
    refs = [c["ref"] for c in read_json(path).get("checkpoints", []) if c.get("ref")]
    failed: list[str] = []
    for ref in refs:
        try:
            if sandbox == "daytona":
                from benchflow.sandbox.daytona import build_sync_client

                client = build_sync_client()
                client.snapshot.delete(client.snapshot.get(ref))
            elif sandbox == "docker":
                subprocess.run(
                    ["docker", "image", "rm", "-f", ref],
                    capture_output=True,
                    check=True,
                    timeout=120,
                )
        except Exception:
            failed.append(ref)
    return failed


TIMING_KEYS = {
    "environment_setup",
    "agent_setup",
    "agent_execution",
    "verifier",
    "total",
}


# ---------------------------------------------------------------------------
# Golden files
# ---------------------------------------------------------------------------


@dataclass
class Golden:
    name: str
    path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.path = GOLDEN_DIR / f"{self.name}.json"

    def check(self, actual: dict[str, Any]) -> None:
        rendered = json.dumps(actual, indent=2, sort_keys=True) + "\n"
        if os.environ.get(UPDATE_GOLDEN_ENV) == "1":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(rendered)
            return
        assert self.path.is_file(), (
            f"golden file {self.path.relative_to(REPO_ROOT)} missing; "
            f"create it with {UPDATE_GOLDEN_ENV}=1 and review the diff"
        )
        expected = json.loads(self.path.read_text())
        assert actual == expected, (
            f"{self.path.relative_to(REPO_ROOT)} differs from this run. If the "
            f"change is intended, regenerate with {UPDATE_GOLDEN_ENV}=1 and "
            "review `git diff` before committing.\n--- actual ---\n" + rendered
        )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
