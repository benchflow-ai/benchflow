"""B4d/B5d: setup trials for two or three arms, interleaved, every span to CSV.

Each trial is one ``bench eval run`` of one task with one agent, run with an
arm's own ``bench``; arms alternate trial by trial (the order rotates each
round). After each trial the script reads its ``timing.json`` and appends a
CSV row with every phase and span (absent spans stay empty), the reward and
the error. Agents: ``claude-agent-acp`` or ``codex-acp`` on the scripted fake
model provider (no key, nothing billed), or the ``nop`` and ``oracle``
controls. Trials and logs go under ``--work``, in a folder named after the
run (agent, proxy, backend, task variant); the script refuses to start when an
earlier run already wrote there, and never deletes a trial.

The fake-provider route, one for every speed number: on Docker the
fake runs on the build VM, bound to the Docker bridge's gateway address on a
port from 8900 to 8999, started by this script from the deterministic tier's
``fake_llm.py`` with the script's own Python. The task image needs no Python
for it. With ``--proxy host`` (Docker's default) BenchFlow's host LiteLLM
proxy calls the fake; with ``--proxy sandbox`` the task carries an egress
policy (``agent.network_mode: allowlist``), which moves the proxy into the
sandbox, and the in-sandbox proxy calls the fake at the bridge address: the
policy confines only the agent's uid, which reaches nothing but the proxy on
loopback. On Daytona the VM is out of reach, so the proxy (always in the
sandbox there) calls the deterministic task's own fake service at
``http://127.0.0.1:8911``; only ``--task-variant deterministic`` runs there.

    uv run python -m tools.perf.setup_trials \\
        --arm base=2ea2ad00:$W/venv-base --arm tip=abc12345:$W/venv-tip \\
        --agent claude-agent-acp --proxy sandbox --backend docker --count 5 \\
        --work $W/perf/b4d-work --out $W/perf/b4d.csv
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from tools.perf._common import (
    REPO,
    Arm,
    append_rows,
    check_arms,
    image_digest,
    interleaved,
    output,
    parse_arm,
    provenance,
    report_provenance,
)

TEMPLATE_TASK = REPO / "tests" / "integration" / "deterministic" / "task"
FAKE_LLM = TEMPLATE_TASK / "environment" / "fake_llm" / "fake_llm.py"
IN_SANDBOX_FAKE_URL = "http://127.0.0.1:8911"
# Not a credential: the fake accepts any key; the proxy route needs one set.
DUMMY_KEY = "fake-perf-key"
FAKE_PORTS = range(8900, 9000)

MODEL_AGENTS = {"claude-agent-acp": "claude-haiku-4-5", "codex-acp": "gpt-5.5"}
CONTROLS = ("nop", "oracle")

PHASES = (
    "environment_setup",
    "agent_setup",
    "agent_execution",
    "verifier",
    "total",
)
SPANS = (
    "slot_wait",
    "start_pace_wait",
    "install_agent",
    "litellm_install",
    "proxy_health_wait",
    "acp_connect",
    "cloud_metadata_block",
    "firewall_power",
    "cleanup",
)
HARDEN_STEPS = (
    "total",
    "kill_sandbox_user_procs",
    "clear_verifier_output",
    "prepare_app_dir",
    "reclaim_disk",
    "restore_workspace",
    "freeze_workspace",
    "cleanup_conftests",
    "build_verifier_env",
)
# The parts of ``total`` that do not overlap: what is left is unnamed time.
# (litellm_install, proxy_health_wait, acp_connect, cloud_metadata_block and
# firewall_power sit inside agent_setup, harden_before_verify inside verifier;
# slot_wait is before the trial starts.)
DISJOINT = (
    "environment_setup",
    "install_agent",
    "agent_setup",
    "agent_execution",
    "verifier",
    "cleanup",
)
FIELDS = (
    "arm",
    "commit",
    "round",
    "position",
    "agent",
    "proxy",
    "backend",
    "task_variant",
    "exit_code",
    "wall_seconds",
    "reward",
    "error",
    *PHASES,
    *SPANS,
    *(f"harden_before_verify.{step}" for step in HARDEN_STEPS),
    "unnamed",
    "trial_dir",
)

NO_PYTHON_DOCKERFILE = """FROM ubuntu:24.04

# No Python: the fake model provider runs outside the sandbox (tools/perf
# README, "The fake-provider route"). curl and CA certificates are what the
# agent install needs.
RUN apt-get update -qq \\
    && apt-get install -y -qq --no-install-recommends curl ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
"""


def run_key(*, agent: str, proxy: str, backend: str, variant: str) -> str:
    """What a run is: its trials and logs live under this name in --work, so
    runs with different agents, proxies, backends or tasks never collide."""
    return f"{agent}-{proxy}-{backend}-{variant}"


def trial_paths(work: Path, key: str, arm: str, round_index: int) -> tuple[Path, Path]:
    """The jobs folder and the log of one trial."""
    return (
        work / "jobs" / key / arm / f"{round_index:03d}",
        work / "logs" / key / f"{arm}-{round_index:03d}.txt",
    )


def check_fresh(paths: list[Path]) -> None:
    """Refuse to start when an earlier run's trial or log is in the way: the
    script never deletes or overwrites a trial (it is the run's evidence)."""
    used = [path for path in paths if path.exists()]
    if used:
        raise SystemExit(
            f"{used[0]} already exists ({len(used)} trial paths in use): an earlier "
            "run with the same agent, proxy, backend and task wrote there; pass a "
            "fresh --work (nothing was deleted)"
        )


def materialize_task(dest_root: Path, *, variant: str, proxy: str) -> Path:
    """The deterministic task, with the egress policy when the proxy is in
    the sandbox, and without Python in its image for ``no-python``."""
    dest = dest_root / f"perf-{variant}-{proxy}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(TEMPLATE_TASK, dest)
    task_md = dest / "task.md"
    text = task_md.read_text()
    if proxy == "sandbox":
        text, n = re.subn(
            r"(?m)^(agent:\n  timeout_sec: [0-9.]+)$",
            r"\g<1>\n  network_mode: allowlist\n  allowed_hosts:\n  - example.com",
            text,
        )
        if n != 1:
            raise SystemExit("deterministic task.md: agent.timeout_sec not found")
    if variant == "no-python":
        text, n = re.subn(
            r"(?m)^benchflow:\n  environment:\n    manifest: environment.toml\n",
            "",
            text,
        )
        if n != 1:
            raise SystemExit("deterministic task.md: environment manifest not found")
        (dest / "environment.toml").unlink()
        shutil.rmtree(dest / "environment" / "fake_llm")
        (dest / "environment" / "Dockerfile").write_text(NO_PYTHON_DOCKERFILE)
    task_md.write_text(text)
    return dest


def bridge_address() -> str:
    address = output(
        [
            "docker",
            "network",
            "inspect",
            "bridge",
            "--format",
            "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
        ]
    )
    if not address:
        raise SystemExit("no Docker bridge gateway (is Docker running here?)")
    return address


def _free_port(host: str) -> int:
    for port in FAKE_PORTS:
        with socket.socket() as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise SystemExit(f"no free port from 8900 to 8999 on {host}")


@contextlib.contextmanager
def fake_provider(host: str, log_dir: Path) -> Iterator[str]:
    """Run the deterministic fake on ``host``; yield its base URL."""
    port = _free_port(host)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "fake-stdout.txt").open("wb") as out:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(FAKE_LLM),
                "--host",
                host,
                "--port",
                str(port),
                "--log",
                str(log_dir / "fake-calls.jsonl"),
            ],
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 15
        while True:
            with socket.socket() as sock:
                if sock.connect_ex((host, port)) == 0:
                    break
            if process.poll() is not None or time.monotonic() > deadline:
                raise SystemExit(f"the fake provider did not start on {host}:{port}")
            time.sleep(0.1)
        yield f"http://{host}:{port}"
    finally:
        # The fake this script started, by its own handle.
        process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(10)
        if process.poll() is None:
            process.kill()


def agent_args(agent: str, base_url: str | None) -> list[str]:
    if agent in CONTROLS:
        return ["--agent", agent]
    model = MODEL_AGENTS[agent]
    if agent == "codex-acp":
        env = {
            "OPENAI_API_KEY": DUMMY_KEY,
            "BENCHFLOW_PROVIDER_BASE_URL": f"{base_url}/v1",
        }
    else:
        env = {
            "ANTHROPIC_API_KEY": DUMMY_KEY,
            "BENCHFLOW_PROVIDER_BASE_URL": f"{base_url}",
        }
    args = ["--agent", agent, "--model", model]
    for key, value in env.items():
        args += ["--agent-env", f"{key}={value}"]
    return args


def clean_env() -> dict[str, str]:
    """The caller's env without provider settings (they must not leak in)."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_", "CODEX_"))
        and key not in {"BENCHFLOW_PROVIDER_BASE_URL", "BENCHFLOW_PROVIDER_API_KEY"}
    }
    env["BENCHFLOW_SKIP_UPDATE_CHECK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def timing_row(timing: Mapping[str, Any]) -> dict[str, float | None]:
    """Every phase and span of one timing.json, flat, plus unnamed time."""
    row: dict[str, float | None] = {}
    for key in (*PHASES, *SPANS):
        row[key] = _number(timing.get(key))
    steps = timing.get("harden_before_verify")
    steps = steps if isinstance(steps, Mapping) else {}
    for step in HARDEN_STEPS:
        row[f"harden_before_verify.{step}"] = _number(steps.get(step))
    total = row["total"]
    row["unnamed"] = (
        None if total is None else total - sum(row[key] or 0.0 for key in DISJOINT)
    )
    return row


def read_trial(jobs_dir: Path) -> dict[str, Any]:
    """The one trial a run wrote under ``jobs_dir``."""
    trials = sorted(p.parent for p in jobs_dir.glob("*/*/result.json"))
    if len(trials) != 1:
        return {"error": f"expected one trial under {jobs_dir}, found {len(trials)}"}
    trial = trials[0]
    result = json.loads((trial / "result.json").read_text())
    timing_path = trial / "timing.json"
    timing = (
        json.loads(timing_path.read_text())
        if timing_path.is_file()
        else result.get("timing") or {}
    )
    rewards = result.get("rewards") or {}
    return {
        "trial_dir": str(trial),
        "reward": _number(rewards.get("reward")) if isinstance(rewards, dict) else None,
        "error": result.get("error") or result.get("verifier_error"),
        **timing_row(timing),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", type=parse_arm, action="append", required=True)
    parser.add_argument(
        "--agent", choices=[*MODEL_AGENTS, *CONTROLS], default="claude-agent-acp"
    )
    parser.add_argument("--proxy", choices=("host", "sandbox"), default="sandbox")
    parser.add_argument("--backend", choices=("docker", "daytona"), default="docker")
    parser.add_argument(
        "--task-variant",
        choices=("deterministic", "no-python"),
        default="deterministic",
    )
    parser.add_argument("--count", type=int, default=5, help="trials per arm")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    arms: list[Arm] = args.arm
    if not 1 <= len(arms) <= 3:
        raise SystemExit("one to three arms")
    check_arms(arms)
    if args.backend == "daytona":
        if args.task_variant != "deterministic":
            raise SystemExit("on Daytona only the deterministic task runs")
        if args.proxy != "sandbox":
            raise SystemExit("on Daytona the proxy always runs in the sandbox")
    key = run_key(
        agent=args.agent,
        proxy=args.proxy,
        backend=args.backend,
        variant=args.task_variant,
    )
    planned = {
        (round_index, arm.name): trial_paths(args.work, key, arm.name, round_index)
        for round_index in range(args.count)
        for arm in arms
    }
    check_fresh([path for pair in planned.values() for path in pair])
    args.work.mkdir(parents=True, exist_ok=True)
    task_dir = materialize_task(
        args.work / "tasks", variant=args.task_variant, proxy=args.proxy
    )
    base_image = "ubuntu:24.04"
    if args.backend == "docker":
        subprocess.run(["docker", "pull", "-q", base_image], check=False)
    env = clean_env()

    with contextlib.ExitStack() as stack:
        if args.agent in CONTROLS:
            base_url = None
        elif args.backend == "daytona":
            base_url = IN_SANDBOX_FAKE_URL
        else:
            base_url = stack.enter_context(
                fake_provider(bridge_address(), args.work / "fake" / key)
            )
        report_provenance(
            provenance(
                "setup_trials",
                arms=arms,
                images={base_image: image_digest(base_image)}
                if args.backend == "docker"
                else {},
                extra={
                    "agent": args.agent,
                    "proxy": args.proxy,
                    "backend": args.backend,
                    "task_variant": args.task_variant,
                    "count": args.count,
                    "fake_url": base_url,
                },
            ),
            args.out,
        )
        for round_index in range(args.count):
            for position, arm in enumerate(interleaved(arms, round_index)):
                jobs_dir, log = planned[(round_index, arm.name)]
                argv_ = [
                    str(arm.exe("bench")),
                    "eval",
                    "run",
                    "--tasks-dir",
                    str(task_dir),
                    *agent_args(args.agent, base_url),
                    "--sandbox",
                    args.backend,
                    "--jobs-dir",
                    str(jobs_dir),
                    "--concurrency",
                    "1",
                    "--retry-attempts",
                    "0",
                ]
                started = time.monotonic()
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("xb") as out:
                    try:
                        code = subprocess.run(
                            argv_,
                            stdout=out,
                            stderr=subprocess.STDOUT,
                            env=env,
                            timeout=args.timeout,
                            check=False,
                        ).returncode
                    except subprocess.TimeoutExpired:
                        code = -1
                row: dict[str, Any] = {
                    "arm": arm.name,
                    "commit": arm.commit,
                    "round": round_index,
                    "position": position,
                    "agent": args.agent,
                    "proxy": args.proxy,
                    "backend": args.backend,
                    "task_variant": args.task_variant,
                    "exit_code": code,
                    "wall_seconds": time.monotonic() - started,
                    **read_trial(jobs_dir),
                }
                append_rows(args.out, FIELDS, [row])
                print(
                    f"{arm.name} round {round_index}: exit {code}, "
                    f"reward {row.get('reward')}, total {row.get('total')}",
                    flush=True,
                )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
