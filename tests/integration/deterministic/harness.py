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
#: The fake as a compose service of its own (fake_in_sidecar), on an internal
#: network main shares: what BenchFlow's in-sandbox proxy reaches when a model
#: agent runs under a restricted policy, where a declared service in main is
#: refused before the turn.
SIDECAR_FAKE_URL = "http://fake-llm:8911"
SIDECAR_COMPOSE = """\
services:
  main:
    networks: [default, fakenet]
  fake-llm:
    build: .
    command: ["python3", "/opt/fake_llm/fake_llm.py", "--host", "0.0.0.0", "--port", "8911", "--log", "/tmp/fake_llm.jsonl"]
    networks: [fakenet]
networks:
  fakenet:
    internal: true
"""

AGENT = "claude-agent-acp"
MODEL = "claude-haiku-4-5"
# Codex runs against the fake's Responses API route through the proxy. gpt-5.5
# is in the model catalog of the Codex release both harnesses pin (0.159.3)
# and is not code-mode-only, so Codex offers its full tool surface with a
# shell tool the fake drives. (gpt-5.4 was the model until codex 0.158
# dropped it from the catalog; Codex then falls back to plain function tools.)
CODEX_AGENT = "codex-acp"
CODEX_MODEL = "gpt-5.5"
# A model the pinned Codex catalog lacks: Codex runs it on fallback metadata
# and warns as the turn starts, which codex-acp sends as a session notice.
CODEX_UNCATALOGUED_MODEL = "gpt-5.4"
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


def docker_bridge_host() -> str:
    """The host's address on Docker's default bridge: a container's way to the
    host, for BenchFlow's in-sandbox proxy (root, which the agent's network
    rules do not hold) to reach a fake provider on the host."""
    import subprocess

    done = subprocess.run(
        [
            "docker",
            "network",
            "inspect",
            "bridge",
            "-f",
            "{{(index .IPAM.Config 0).Gateway}}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return done.stdout.strip()


@contextmanager
def host_fake_llm(
    log_path: Path | None = None,
    wire_log_path: Path | None = None,
    *,
    host: str = "127.0.0.1",
) -> Iterator[str]:
    """Run the fake provider on the host; yield its base URL.

    ``wire_log_path`` keeps every request body as the provider received it,
    after the proxy. ``host``: the address it listens on
    (:func:`docker_bridge_host` for a proxy in a container).
    """
    fake = _load_fake_llm_module()
    fake._Handler.scripts = json.loads((FAKE_LLM_DIR / "scripts.json").read_text())
    fake._Handler.log_path = log_path
    fake._Handler.wire_log_path = wire_log_path
    server = ThreadingHTTPServer((host, 0), fake._Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{server.server_address[1]}"
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
    # An agent network allowlist (agent.network_mode: allowlist): only these
    # hosts and the model gateway are reachable for the agent.
    allowed_hosts: tuple[str, ...] = ()
    # metadata.x-allow-cloud-metadata: true, the opt-out from refusing the
    # cloud metadata endpoints under the open network.
    allow_cloud_metadata: bool = False
    # test.sh first probes the mocked cloud metadata endpoints as root
    # (ROOT_PROBE_TEST_SH), then scores as the template does.
    root_probe: bool = False
    # test.sh first kills the in-sandbox model proxy and rewrites its usage
    # log, as root (TAMPER_PROXY_TEST_SH), then scores as the
    # template does.
    tamper_proxy: bool = False
    # test.sh first rewrites every token count in the agent's Claude Code
    # session logs to 1, as root (TAMPER_SESSION_LOG_TEST_SH), then
    # scores as the template does.
    tamper_session_log: bool = False
    # test.sh first plants a session log in /logs/agent/<this folder>
    # (claude-sessions or codex-sessions), as root (plant_session_log_test_sh),
    # then scores as the template does.
    plant_sessions: str | None = None
    # The fake provider in a compose service of its own (fake_in_sidecar),
    # for a model agent under a restricted policy; an allowlist variant
    # always has it, unless the fake is on the host instead.
    fake_in_sidecar: bool = False
    # No fake in the task at all (drop_manifest): the run's fake is on the
    # host (docker_bridge_host), and the task stays a single image.
    fake_on_host: bool = False
    # With fake_on_host: keep the task's environment, a manifest that
    # declares no service (no_declared_services), so it still provisions.
    keep_environment: bool = False


# Root in the verifier phase, as code the agent left could do: kill
# each in-sandbox model proxy and rewrite its usage log to 1 prompt token a
# call, then score as the template does. It writes what it
# found to /logs/verifier/tamper.txt.
TAMPER_PROXY_TEST_SH = r"""#!/bin/bash
n=0
for pid in /opt/runtime/proxy/*/litellm.pid; do
  [ -s "$pid" ] && kill -9 "$(cat "$pid")" 2>/dev/null && n=$((n+1))
done
m=0
for log in /opt/runtime/proxy/*/callback.jsonl; do
  [ -f "$log" ] || continue
  sed -i 's/"prompt_tokens":[0-9]*/"prompt_tokens":1/g' "$log" && m=$((m+1))
done
echo "killed=$n rewrote=$m" > /logs/verifier/tamper.txt
REWARD=0
if [ -f /app/hello.txt ] && [ "$(tr -d '\n' < /app/hello.txt)" = "Hello, world!" ]; then
    REWARD=1
fi
echo "$REWARD" > /logs/verifier/reward.txt
"""


# Root in the verifier phase, as code the agent left could do:
# rewrite every token count in the agent's Claude Code session logs to 1 and
# its cost-state figures to $0.000001, then score as the template does. It writes how many logs it rewrote to
# /logs/verifier/tamper.txt.
TAMPER_SESSION_LOG_TEST_SH = r"""#!/bin/bash
n=0
for log in /root/.claude/projects/*/*.jsonl /home/*/.claude/projects/*/*.jsonl; do
  [ -f "$log" ] || continue
  sed -i -E -e 's/("(input_tokens|output_tokens|cache_read_input_tokens|cache_creation_input_tokens|inputTokens|outputTokens)": ?)[0-9]+/\11/g' \
    -e 's/("(totalCostUSD|costUSD)": ?)[0-9.eE+-]+/\10.000001/g' "$log" && n=$((n+1))
done
echo "rewrote=$n" > /logs/verifier/tamper.txt
REWARD=0
if [ -f /app/hello.txt ] && [ "$(tr -d '\n' < /app/hello.txt)" = "Hello, world!" ]; then
    REWARD=1
fi
echo "$REWARD" > /logs/verifier/reward.txt
"""


# What the planted log holds: three compactions and five one-token calls
# in Claude Code's format, and one one-token count in Codex's.
PLANTED_SESSION_LOG = "".join(
    line + "\n"
    for line in [
        '{"type":"system","subtype":"compact_boundary",'
        '"compactMetadata":{"trigger":"auto","preTokens":150000}}'
    ]
    * 3
    + [
        f'{{"type":"assistant","requestId":"req_planted{n}","message":'
        f'{{"id":"msg_planted{n}","model":"{MODEL}","usage":'
        '{"input_tokens":1,"output_tokens":1}}}'
        for n in range(5)
    ]
    + [
        '{"type":"event_msg","payload":{"type":"token_count","info":'
        '{"total_token_usage":{"input_tokens":1,"cached_input_tokens":0,'
        '"output_tokens":1,"reasoning_output_tokens":0,"total_tokens":2},'
        '"last_token_usage":{"input_tokens":1,"cached_input_tokens":0,'
        '"output_tokens":1,"reasoning_output_tokens":0,"total_tokens":2}}}}'
    ]
)


def plant_session_log_test_sh(folder: str) -> str:
    """test.sh that, as root, plants PLANTED_SESSION_LOG at
    /logs/agent/<folder>/-app/planted.jsonl (/logs/agent is the trial's
    agent/), then scores as the template does. It writes
    ``planted=<files>`` to /logs/verifier/plant.txt."""
    return (
        "#!/bin/bash\n"
        f"d=/logs/agent/{folder}/-app\n"
        'mkdir -p "$d"\n'
        "cat > \"$d/planted.jsonl\" <<'JSONL'\n"
        f"{PLANTED_SESSION_LOG}"
        "JSONL\n"
        'echo "planted=$(ls "$d" | wc -l)" > /logs/verifier/plant.txt\n'
        "REWARD=0\n"
        "if [ -f /app/hello.txt ] && [ \"$(tr -d '\\n' < /app/hello.txt)\" = "
        '"Hello, world!" ]; then\n'
        "    REWARD=1\n"
        "fi\n"
        'echo "$REWARD" > /logs/verifier/reward.txt\n'
    )


def fake_in_sidecar(task: Path) -> Path:
    """Move a task's fake provider out of main: no declared
    ``[[environment.services]]`` (environment.toml), and a compose service of
    its own, built from the same image, on an internal network main shares,
    at :data:`SIDECAR_FAKE_URL`. A service declared in main beside a model
    agent under a restricted policy is refused before the turn."""
    drop_manifest(task)
    (task / "environment" / "docker-compose.yaml").write_text(SIDECAR_COMPOSE)
    return task


def no_declared_services(task: Path) -> Path:
    """Keep the task's environment manifest, without the fake as a service of
    its own: ``owns_lifecycle = true`` and no ``[[environment.services]]``.
    The environment still provisions; nothing is declared in main."""
    manifest = task / "environment.toml"
    text = manifest.read_text()
    head, sep, _ = text.partition("[[environment.services]]")
    assert sep, "template environment.toml must declare the fake as a service"
    head, n = re.subn(
        r"(?m)^owns_lifecycle\s*=\s*false$", "owns_lifecycle = true", head
    )
    assert n == 1, "template environment.toml must set owns_lifecycle = false"
    manifest.write_text(head.rstrip() + "\n")
    return task


def drop_manifest(task: Path) -> Path:
    """A task without its declared fake: no environment.toml, and no
    ``benchflow.environment.manifest`` in task.md."""
    (task / "environment.toml").unlink()
    task_md = task / "task.md"
    text, n = re.subn(
        r"(?m)^benchflow:\n  environment:\n    manifest: environment.toml\n",
        "",
        task_md.read_text(),
    )
    assert n == 1, "template task.md must name its manifest"
    task_md.write_text(text)
    return task


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
    if variant.allowed_hosts:
        hosts = "".join(f"\n  - {host}" for host in variant.allowed_hosts)
        text, n = re.subn(
            r"(?m)^(agent:\n  timeout_sec: [0-9.]+)$",
            rf"\g<1>\n  network_mode: allowlist\n  allowed_hosts:{hosts}",
            text,
        )
        assert n == 1, "template task.md must set agent.timeout_sec"
    if variant.allow_cloud_metadata:
        text, n = re.subn(
            r"(?m)^metadata:\n", "metadata:\n  x-allow-cloud-metadata: true\n", text
        )
        assert n == 1, "template task.md must have a metadata section"
    task_md.write_text(text)
    if variant.fake_on_host:
        if variant.keep_environment:
            no_declared_services(dest)
        else:
            drop_manifest(dest)
    elif variant.fake_in_sidecar or variant.allowed_hosts:
        fake_in_sidecar(dest)
    if variant.broken_verifier:
        test_sh = dest / "tests" / "test.sh"
        test_sh.write_text(VERIFIER_ERROR_TEST_SH)
        test_sh.chmod(0o755)
    if variant.root_probe:
        test_sh = dest / "tests" / "test.sh"
        test_sh.write_text(root_probe_test_sh())
        test_sh.chmod(0o755)
    if variant.tamper_proxy:
        test_sh = dest / "tests" / "test.sh"
        test_sh.write_text(TAMPER_PROXY_TEST_SH)
        test_sh.chmod(0o755)
    if variant.tamper_session_log:
        test_sh = dest / "tests" / "test.sh"
        test_sh.write_text(TAMPER_SESSION_LOG_TEST_SH)
        test_sh.chmod(0o755)
    if variant.plant_sessions:
        test_sh = dest / "tests" / "test.sh"
        test_sh.write_text(plant_session_log_test_sh(variant.plant_sessions))
        test_sh.chmod(0o755)
    return dest


def root_probe_test_sh() -> str:
    """test.sh that, as root, connects to each mocked IPv4 cloud metadata
    endpoint on port 80 and on port 53 (DNS, which the rules leave open) and
    closes the connection (never a request), writing ``root-probe <address>
    <port> <refused|reached|timeout|unreachable|failed>`` lines to
    /logs/verifier/root-probe.txt; an endpoint that is not a local address of
    main's namespace (no mock) is never connected to (``root-probe <address>
    - not-mocked``). Then it scores as the template does."""
    from benchflow.sandbox.cloud_metadata import CLOUD_METADATA_V4

    return (
        "#!/bin/bash\n"
        'local_addr() { awk -v a="$1" \'$1 == "|--" && $2 == a {f = 1; next} '
        "f && /host LOCAL/ {print a; exit} {f = 0}' /proc/net/fib_trie; }\n"
        f"for a in {' '.join(CLOUD_METADATA_V4)}; do\n"
        '  if [ -z "$(local_addr "$a")" ]; then echo "root-probe $a - not-mocked"; continue; fi\n'
        "  for p in 80 53; do\n"
        '    e=$(timeout 3 bash -c "exec 3<>/dev/tcp/$a/$p" 2>&1); r=$?\n'
        "    case $r in 0) s=reached;; 124) s=timeout;; *) case $e in "
        "*[Rr]efused*) s=refused;; *[Uu]nreachable*) s=unreachable;; *) s=failed;; "
        "esac;; esac\n"
        '    echo "root-probe $a $p $s"\n'
        "  done\n"
        "done > /logs/verifier/root-probe.txt 2>&1\n"
        "REWARD=0\n"
        "if [ -f /app/hello.txt ] && "
        '[ "$(tr -d \'\\n\' < /app/hello.txt)" = "Hello, world!" ]; then REWARD=1; fi\n'
        'echo "$REWARD" > /logs/verifier/reward.txt\n'
    )


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

    @property
    def wire_dir(self) -> Path:
        return wire_dir(self.jobs_dir)


def wire_dir(jobs_dir: Path) -> Path:
    """Where the proxy writes a run's requests (beside the jobs folder, not in
    it, so the folder holds only jobs)."""
    return jobs_dir.with_name(f"{jobs_dir.name}-wire")


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


# Codex routes: through the proxy wherever the backend runs it, or with the
# proxy inside the sandbox (a network policy moves it there on every backend).
CODEX_ROUTES = ("codex", "codex-in-sandbox", "codex-uncatalogued")


def route_model(route: str) -> str:
    """The ``--model`` a route runs with."""
    if route == "codex-uncatalogued":
        return CODEX_UNCATALOGUED_MODEL
    if route in CODEX_ROUTES:
        return CODEX_MODEL
    return POLICY_ROUTES[route][0] if route in POLICY_ROUTES else MODEL


def route_agent(route: str) -> str:
    """The ``--agent`` a route runs (Codex for the codex routes)."""
    return CODEX_AGENT if route in CODEX_ROUTES else AGENT


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
    if route in CODEX_ROUTES:
        # The proxy's upstream is the fake's Responses API (openai/ routes
        # append /responses to the base URL).
        if route == "codex-in-sandbox":
            # Its jobs are allowlisted: the fake is their compose service.
            base = SIDECAR_FAKE_URL
        else:
            in_sandbox = proxy_runs_in_sandbox(sandbox)
            base = IN_SANDBOX_FAKE_URL if in_sandbox else host_fake_url
        assert base, "a host LiteLLM proxy needs the host fake provider URL"
        return {
            "OPENAI_API_KEY": DUMMY_KEY,
            "BENCHFLOW_PROVIDER_BASE_URL": base + "/v1",
        }
    assert route == "proxy", route
    base = IN_SANDBOX_FAKE_URL if proxy_runs_in_sandbox(sandbox) else host_fake_url
    assert base, "a host LiteLLM proxy needs the host fake provider URL"
    return {"ANTHROPIC_API_KEY": DUMMY_KEY, "BENCHFLOW_PROVIDER_BASE_URL": base}


def check_route_is_hermetic(
    route: str,
    agent_env: dict[str, str],
    agent: str | None = None,
    model: str | None = None,
) -> None:
    """Refuse a route that would upload host credentials or skip the fake.

    With no explicit token BenchFlow falls back to host subscription files
    (``~/.claude/.credentials.json``); an explicit dummy token must prevent
    that, and the resolved base URL must be the fake.
    """
    from benchflow.agents.env import resolve_agent_env, uses_native_subscription_auth

    model = model or route_model(route)
    agent = agent or route_agent(route)
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
    agent: str | None = None,
    model: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """The ``bench`` argv and process env for one scripted run (checked hermetic).

    ``agent`` replaces the route's agent (a plugin agent on the same route),
    ``model`` the route's model.
    """
    agent_env = route_env(route, sandbox, host_fake_url)
    args = [
        bench_executable(),
        *subcommand,
        "--tasks-dir",
        str(tasks_dir),
        "--agent",
        agent or route_agent(route),
        "--model",
        model or route_model(route),
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
        if key.startswith(
            ("ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_", "CODEX_")
        ) or key in {
            "BENCHFLOW_PROVIDER_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY",
        }:
            env.pop(key)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["BENCHFLOW_SKIP_UPDATE_CHECK"] = "1"
    # The tier's providers are its fake, which has no hosted web search:
    # under a restricted network (the allowlist and denylist scenarios) they
    # are admitted the documented way, as operator-run providers
    # (a58792161's admission matches the provider-qualified id).
    env["BENCHFLOW_RESTRICTED_WEB_MODELS"] = "anthropic/*,openai/*"
    # Every request each harness sends into the proxy, for wire parity.
    env["BENCHFLOW_LITELLM_WIRE_LOG_DIR"] = str(wire_dir(jobs_dir))
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k not in env}
    try:
        check_route_is_hermetic(route, agent_env, agent, model)
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
    agent: str | None = None,
    env_extra: dict[str, str] | None = None,
    model: str | None = None,
) -> CliRun:
    args, env = bench_command(
        subcommand,
        tasks_dir=tasks_dir,
        jobs_dir=jobs_dir,
        sandbox=sandbox,
        host_fake_url=host_fake_url,
        route=route,
        extra=extra,
        agent=agent,
        model=model,
    )
    env.update(env_extra or {})
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


def _responses_call(
    request: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
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
# trajectories.types.stamp_observation_ids: random per run, shared by every
# sighting of one tool call.
_OBSERVATION_ID = re.compile(r"obs-[0-9a-f]{32}")


def normalized_acp(trial_dir: Path) -> list[dict[str, Any]]:
    """The ACP trajectory without run-dependent values.

    Each tool call's random ``observation_id`` becomes ``obs-<n>``, numbered
    in order of first sighting, so the golden still says which events share
    one identity and that every tool call has one.
    """
    path = trial_dir / "trajectory" / "acp_trajectory.jsonl"
    if not path.is_file():
        return []
    events = [_strip(event, _ACP_VOLATILE) for event in read_jsonl(path)]
    ordinals: dict[str, str] = {}
    for event in events:
        identity = event.get("observation_id")
        if identity is None:
            continue
        assert isinstance(identity, str) and _OBSERVATION_ID.fullmatch(identity), event
        event["observation_id"] = ordinals.setdefault(
            identity, f"obs-{len(ordinals) + 1}"
        )
    return events


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
