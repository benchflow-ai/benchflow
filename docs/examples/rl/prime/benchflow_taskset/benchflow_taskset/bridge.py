"""The BenchFlow side of ``benchflow-taskset``: BenchFlow tasks over stdin/stdout.

BenchFlow and Verifiers v1 cannot share one Python environment (BenchFlow's
``litellm[proxy]`` needs ``mcp<2``; Verifiers needs ``mcp==2.0.0``). So this
file runs in a BenchFlow venv, imports only the standard library and BenchFlow's
public API, and the Verifiers side (``env.py``) talks to it through a pipe. It
is never imported by the Verifiers side.

Two modes::

    python bridge.py tasks --tasks-dir DIR [--include NAME]... [--exclude NAME]...
        One JSON object per task on stdout: name, id, task_dir, prompt, metadata.

    python bridge.py session [--idle-timeout SECONDS]
        One task session. Requests are JSON objects, one per line on stdin; each
        gets one JSON reply line on stdout:

        {"op": "start", "task_dir": ..., "environment": "daytona", "sandbox_user": "agent",
         "jobs_dir": ..., "rollout_name": ...}
            -> {"ok": true, "workspace": "/workdir", "rollout_dir": ...}
            -> {"ok": false, "decision": <drop, reason sandbox_start>, "error": ...}
        {"op": "bash", "command": ..., "timeout_sec": 60}
            -> {"ok": true, "return_code": 0, "stdout": ..., "stderr": ..., "timed_out": false}
            -> {"ok": false, "error": ..., "transient": true|false}
        {"op": "write", "path": "/workdir/answer.txt", "text": ...}
        {"op": "verify", "policy_acted": true}
            -> {"ok": true, "decision": {"reward", "dropped", "reason", "detail"}, "result": {...}}
        {"op": "close"} -> {"ok": true}

        The sandbox is closed on "close", at end of input (the caller went away),
        on SIGTERM or SIGINT, and after --idle-timeout seconds without a request.

The reward decision (a reward, or a drop for an infrastructure failure the
policy could not have caused) comes from ``benchflow.integrations.rewards``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Commands run under `timeout`; the exec call waits this much longer, so a command
# that runs out of time comes back as exit code 124 rather than an exec error.
EXEC_GRACE_SEC = 15
MAX_STREAM_CHARS = 256_000


def _clip(text: str, limit: int = MAX_STREAM_CHARS) -> str:
    if len(text) <= limit:
        return text
    return (
        text[: limit // 2]
        + "\n[... output clipped by the bridge ...]\n"
        + text[-limit // 2 :]
    )


def _describe(exc: BaseException) -> str:
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text[:2000]


def _is_transient(exc: BaseException) -> bool:
    """A Daytona transport blip, as BenchFlow itself marks it."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ == "TransientSandboxTransportError":
            return True
        current = current.__cause__ or current.__context__
    return False


# --- tasks ------------------------------------------------------------------------------


def list_tasks(
    tasks_dir: Path, include: list[str], exclude: list[str]
) -> list[dict[str, Any]]:
    import benchflow as bf

    root = tasks_dir.expanduser().resolve()
    if (root / "task.md").is_file() or (root / "task.toml").is_file():
        candidates = [root]
    else:
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
    rows = []
    for task_dir in candidates:
        if include and task_dir.name not in include:
            continue
        if task_dir.name in exclude:
            continue
        if not ((task_dir / "task.md").is_file() or (task_dir / "task.toml").is_file()):
            continue
        task = bf.Task(task_dir)
        prompt = (task.instruction or "").strip()
        if not prompt:
            raise ValueError(f"{task_dir}: the task has no prompt")
        config = task.config
        metadata = getattr(config, "metadata", None)
        if hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump(mode="json")
        verifier = getattr(config, "verifier", None)
        agent = getattr(config, "agent", None)
        rows.append(
            {
                "name": task_dir.name,
                "id": str(task.name),
                "task_dir": str(task_dir),
                "prompt": prompt,
                "metadata": metadata if isinstance(metadata, dict) else {},
                "verifier_timeout_sec": getattr(verifier, "timeout_sec", None),
                "agent_timeout_sec": getattr(agent, "timeout_sec", None),
            }
        )
    return rows


# --- one session ------------------------------------------------------------------------


class Session:
    def __init__(self) -> None:
        self.runtime: Any = None
        self.verified = False
        self.closed = False

    async def start(self, request: dict[str, Any]) -> dict[str, Any]:
        from benchflow.integrations import rewards
        from benchflow.rollout import TaskRuntime, TaskRuntimeConfig

        if self.runtime is not None:
            return {"ok": False, "error": "already started"}
        config = TaskRuntimeConfig(
            task_path=request["task_dir"],
            environment=request.get("environment", "daytona"),
            sandbox_user=request.get("sandbox_user", "agent"),
            jobs_dir=request.get("jobs_dir", "jobs/benchflow-taskset"),
            job_name=request.get("job_name"),
            rollout_name=request.get("rollout_name"),
            sandbox_setup_timeout=int(request.get("sandbox_setup_timeout", 300)),
        )
        runtime = TaskRuntime(config)
        try:
            await runtime.start()
        except Exception as exc:  # the policy has not acted: an infrastructure failure
            decision = rewards.sandbox_start_failure(_describe(exc))
            return {
                "ok": False,
                "error": _describe(exc),
                "decision": decision.as_dict(),
            }
        self.runtime = runtime
        return {
            "ok": True,
            "workspace": runtime.workspace,
            "rollout_dir": str(runtime.rollout_dir),
        }

    async def bash(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.runtime is None or self.verified:
            return {"ok": False, "error": "no running sandbox", "transient": False}
        timeout_sec = max(1, int(request.get("timeout_sec", 60)))
        command = str(request.get("command", ""))
        # Kill the command (and, after 5 more seconds, anything it left) at the limit.
        wrapped = f"timeout -k 5 {timeout_sec} bash -c {shlex.quote(command)}"
        started = time.monotonic()
        try:
            result = await self.runtime.bash(
                wrapped, timeout_sec=timeout_sec + EXEC_GRACE_SEC
            )
        except Exception as exc:
            # A transport blip can also say "timed out"; it is not the command's timeout.
            if _is_transient(exc):
                return {"ok": False, "error": _describe(exc), "transient": True}
            if "timed out" in str(exc).lower():
                return {
                    "ok": True,
                    "return_code": 124,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": True,
                    "elapsed_sec": round(time.monotonic() - started, 3),
                }
            return {"ok": False, "error": _describe(exc), "transient": False}
        return {
            "ok": True,
            "return_code": result.return_code,
            "stdout": _clip(result.stdout),
            "stderr": _clip(result.stderr),
            "timed_out": result.return_code in (124, 137)
            and result.elapsed_sec >= timeout_sec,
            "elapsed_sec": result.elapsed_sec,
        }

    async def write(self, request: dict[str, Any]) -> dict[str, Any]:
        path = str(request["path"])
        if not path.startswith("/"):
            return {"ok": False, "error": "path must be absolute", "transient": False}
        text = str(request.get("text", ""))
        quoted = shlex.quote(path)
        # The same write the TRL harness's `submit` does, as the sandbox user.
        command = f'mkdir -p "$(dirname {quoted})" && printf %s {shlex.quote(text)} > {quoted}'
        reply = await self.bash({"command": command, "timeout_sec": 30})
        if reply.get("ok") and reply.get("return_code") != 0:
            detail = (reply.get("stderr") or reply.get("stdout") or "").strip()[-500:]
            return {
                "ok": False,
                "error": f"write failed (exit {reply['return_code']}): {detail}",
                "transient": False,
            }
        return reply

    async def verify(self, request: dict[str, Any]) -> dict[str, Any]:
        from benchflow.integrations import rewards

        if self.runtime is None:
            return {"ok": False, "error": "no running sandbox"}
        if self.verified:
            return {"ok": False, "error": "already verified"}
        policy_acted = bool(request.get("policy_acted", True))
        self.verified = True
        try:
            outcome = await self.runtime.verify()
        except Exception as exc:
            # The verifier did not produce a result. After the policy acted this may be
            # its doing (a scored 0); on an untouched sandbox it is infrastructure.
            if policy_acted:
                decision = rewards.zero(rewards.VERIFIER_ERROR, _describe(exc))
            else:
                decision = rewards.dropped(
                    rewards.VERIFIER_CRASH_CLEAN_RUN, _describe(exc)
                )
            return {
                "ok": True,
                "decision": decision.as_dict(),
                "result": {"exception": _describe(exc)},
            }
        run = outcome.result
        decision = rewards.reward_from_verify(run, policy_acted=policy_acted)
        return {
            "ok": True,
            "decision": decision.as_dict(),
            "result": {
                "reward": outcome.reward,
                "rewards": outcome.rewards,
                "verifier_error": outcome.verifier_error,
                "error": outcome.error,
                "error_category": getattr(run, "error_category", None),
                "verifier_error_category": getattr(
                    run, "verifier_error_category", None
                ),
                "rollout_dir": str(outcome.rollout_dir),
            },
        }

    async def close(self) -> dict[str, Any]:
        if self.closed:
            return {"ok": True}
        self.closed = True
        runtime, self.runtime = self.runtime, None
        if runtime is not None:
            try:
                await asyncio.wait_for(runtime.close(), timeout=180)
            except Exception as exc:
                return {"ok": False, "error": f"close failed: {_describe(exc)}"}
        return {"ok": True}


async def serve_session(idle_timeout: float) -> int:
    # Replies go to the real stdout; anything BenchFlow or its dependencies print goes
    # to stderr, so it can never corrupt the protocol.
    reply_fd = os.dup(1)
    os.dup2(2, 1)
    replies = os.fdopen(reply_fd, "w", buffering=1)
    session = Session()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    reader = asyncio.StreamReader(limit=64 * 1024 * 1024)
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
    )

    handlers = {
        "start": session.start,
        "bash": session.bash,
        "write": session.write,
        "verify": session.verify,
    }
    try:
        while not stop.is_set():
            line_task = asyncio.ensure_future(reader.readline())
            stop_task = asyncio.ensure_future(stop.wait())
            done, _ = await asyncio.wait(
                {line_task, stop_task},
                timeout=idle_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            stop_task.cancel()
            if line_task not in done:
                line_task.cancel()
                if not done:
                    print(
                        f"bridge: idle for {idle_timeout:g}s; closing the sandbox",
                        file=sys.stderr,
                    )
                break
            line = line_task.result()
            if not line:
                break  # end of input: the caller is gone
            try:
                request = json.loads(line)
                op = request.get("op")
                if op == "close":
                    reply = await session.close()
                    replies.write(json.dumps(reply) + "\n")
                    break
                handler = handlers.get(op)
                if handler is None:
                    reply = {"ok": False, "error": f"unknown op {op!r}"}
                else:
                    # A signal during a long call must still close the sandbox.
                    call = asyncio.ensure_future(handler(request))
                    stop_task = asyncio.ensure_future(stop.wait())
                    await asyncio.wait(
                        {call, stop_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    stop_task.cancel()
                    if not call.done():
                        call.cancel()
                        break
                    reply = call.result()
            except Exception as exc:  # a bad request must not kill the session
                reply = {"ok": False, "error": _describe(exc), "transient": False}
            replies.write(json.dumps(reply) + "\n")
    finally:
        await session.close()
        replies.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    tasks = sub.add_parser("tasks")
    tasks.add_argument("--tasks-dir", type=Path, required=True)
    tasks.add_argument("--include", action="append", default=[])
    tasks.add_argument("--exclude", action="append", default=[])
    session = sub.add_parser("session")
    session.add_argument("--idle-timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if args.mode == "tasks":
        for row in list_tasks(args.tasks_dir, args.include, args.exclude):
            print(json.dumps(row))
        return 0
    return asyncio.run(serve_session(args.idle_timeout))


if __name__ == "__main__":
    sys.exit(main())
