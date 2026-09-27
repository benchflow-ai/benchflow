"""Fresh native CLI sessions on the operator host, without Docker isolation."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

from benchflow.models import RolloutResult

from . import client
from .bridge import write_json


def command_line(agent: str, model: str, effort: str, workspace: Path) -> list[str]:
    if agent == "codex":
        return [
            "codex",
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--json",
            "--color",
            "never",
            "--sandbox",
            "workspace-write",
            "-c",
            "sandbox_workspace_write.network_access=true",
            "-c",
            f'model_reasoning_effort="{effort}"',
            "--model",
            model,
            "--cd",
            str(workspace),
            "-",
        ]
    if agent == "claude":
        return [
            "claude",
            "--print",
            "--safe-mode",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            f"Bash({sys.executable} {workspace}/robot.py *)",
            "Read",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            model,
            "--effort",
            effort,
        ]
    raise ValueError("Host mode supports codex and claude")


def parse_events(path: Path, agent: str) -> dict:
    usage: dict = {}
    completed = False
    failed = False
    messages = []
    native_cost = None
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if agent == "codex":
            if event.get("type") == "turn.completed":
                completed = True
                for key, value in event.get("usage", {}).items():
                    if isinstance(value, int):
                        usage[key] = usage.get(key, 0) + value
            if event.get("type") in {"turn.failed", "error"}:
                failed = True
            item = event.get("item", {})
            if (
                event.get("type") == "item.completed"
                and item.get("type") == "agent_message"
            ):
                messages.append(item.get("text", ""))
        elif event.get("type") == "result":
            completed = True
            failed = event.get("is_error", False)
            usage = event.get("usage", {})
            native_cost = event.get("total_cost_usd")
            messages.append(event.get("result", ""))
    return {
        "completed": completed,
        "failed": failed,
        "usage": usage,
        "native_cost_usd": native_cost,
        "final_text": "\n\n".join(messages),
    }


async def run_host_agent(
    *,
    output: Path,
    workspace: Path,
    bridge,
    setup: dict,
    task_path: Path,
    agent: str,
    model: str,
    effort: str,
    timeout: int,
    prompt: str,
    provider_env: dict[str, str],
) -> RolloutResult:
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "robot.py").write_text(Path(client.__file__).read_text())
    write_json(
        workspace / "setup.json",
        {"setup_id": setup["setup_id"], **setup["public_facts"]},
    )
    connection = workspace / "robot-connection.json"
    write_json(
        connection,
        {"url": f"http://127.0.0.1:{bridge.server.server_port}", "token": bridge.token},
    )
    connection.chmod(0o600)
    prompt = prompt.replace("/app/", str(workspace) + "/").replace(
        "python ", sys.executable + " "
    )
    prompt += "\nUse only robot.py for robot access. Do not inspect host histories, past trials, credentials, or harness internals. Do not spawn other agents.\n"
    (output / "agent-prompt.md").write_text(prompt)
    env = dict(os.environ)
    env.update(provider_env)
    env["ROBOT_CONNECTION"] = str(connection)
    env["ROBOT_OBSERVATIONS"] = str(workspace / "observations")
    argv = command_line(agent, model, effort, workspace)
    write_json(
        output / "host-launch.json",
        {
            "argv": argv,
            "workspace": str(workspace),
            "fresh_session": True,
            "isolation": "host workspace; no container filesystem isolation",
        },
    )
    process = None
    started = time.monotonic()
    error = None
    try:
        bridge.activate()
        with (
            (output / "agent-events.jsonl").open("wb") as stdout,
            (output / "agent-stderr.log").open("wb") as stderr,
        ):
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=workspace,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            communicate = asyncio.create_task(process.communicate(prompt.encode()))
            while not communicate.done():
                await asyncio.wait({communicate}, timeout=0.25)
                if time.monotonic() - started > timeout:
                    error = "Host agent time budget exhausted"
                    break
                if (output / "STOP").exists():
                    error = "Operator stopped host trial"
                    break
                if not bridge.recording_healthy():
                    bridge.halted = True
                    bridge.halt_reason = "recording_lost"
                    error = "Camera recording lost"
                    break
            if error:
                bridge.closed = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(communicate, 10)
                except TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await communicate
            else:
                await communicate
    finally:
        bridge.closed = True
        if process is not None and process.returncode is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), 10)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        connection.unlink(missing_ok=True)
    parsed = parse_events(output / "agent-events.jsonl", agent)
    (output / "agent-final.md").write_text(parsed.pop("final_text"))
    usage = parsed["usage"]
    write_json(
        output / "host-agent-summary.json",
        {
            **parsed,
            "exit_code": process.returncode if process else None,
            "agent_wall_s": time.monotonic() - started,
            "usage_source": "agent_native_cli" if usage else "unavailable",
        },
    )
    if not error and (
        not parsed["completed"] or parsed["failed"] or process.returncode
    ):
        error = "Host agent failed; inspect agent-events.jsonl and agent-stderr.log"
    return RolloutResult(
        task_name=task_path.name,
        agent=agent,
        model=model,
        error=error,
        n_input_tokens=usage.get("input_tokens"),
        n_output_tokens=usage.get("output_tokens"),
        n_cache_read_tokens=usage.get(
            "cached_input_tokens", usage.get("cache_read_input_tokens")
        ),
        n_cache_creation_tokens=usage.get("cache_creation_input_tokens"),
        cost_usd=parsed["native_cost_usd"],
        usage_source="unavailable",
        usage_details={
            "native_cli": parsed,
            "agent_wall_s": time.monotonic() - started,
        },
    )
