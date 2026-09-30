"""Check the BenchFlow side of the Miles connector, without a GPU.

Run it against a running environment server (``python -m
benchflow.integrations.miles serve``). It starts a stand-in for Miles's
session server that plays a scripted policy and refuses any request that does
not extend the session's history exactly (the check Miles's session server
makes), then runs real episodes, in real sandboxes, through the server:

    python docs/examples/rl/miles/smoke.py --env-url http://127.0.0.1:12100 \\
        --tasks-dir tasks/train --answers-dir tasks-with-answers/train \\
        --task sql-000000 --task log-000001 --task bugfix-000003

What it checks:

- Each task's episode: the policy runs ``env; id -u; ls -la /workdir``, then
  submits the expected answer when ``--answers-dir`` holds the task's
  ``verifier/expected.json`` (a question task then scores 1), else ``0``. Its
  assistant turns carry ``reasoning_content``, so the replay check covers that
  field too.
- No credential reaches the policy: the output of ``env`` in the sandbox must
  not contain the value of any variable named by ``--secret-env`` (default
  ``DAYTONA_API_KEY``), and the policy must not run as root.
- A model-server failure after the policy acted (the stand-in answers the
  second call with HTTP 502) comes back discarded as ``ModelEndpointFailed``.
- Abort: an episode stuck in a sleeping command, cancelled through ``/abort``
  (or the agent function's ``abort`` hook), comes back discarded as
  ``Aborted``, and no sandbox is left held.

``--via-agent-function`` drives the episodes through the Miles example's
``benchflow_agent_function`` instead of calling ``/run`` directly; put a Miles
checkout and ``examples/experimental/benchflow`` on ``PYTHONPATH`` (only
torch-free Miles modules are imported). Exits 0 when every check passes.
Costs a few sandbox-minutes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from benchflow.integrations.miles import dataset_rows

LOOK = "env; id -u; ls -la /workdir"


class StandIn:
    """A scripted policy behind Miles's session-server routes."""

    def __init__(self, answers_dir: Path | None, secrets: dict[str, str]) -> None:
        self.answers_dir = answers_dir
        self.secrets = secrets
        self.sessions: dict[str, dict[str, Any]] = {}
        self.problems: list[str] = []
        self.calls = 0
        self.uids: set[str] = set()

    def answer(self, task: str) -> str:
        if self.answers_dir is None:
            return "0"
        path = self.answers_dir / task / "verifier" / "expected.json"
        try:
            expected = json.loads(path.read_text())
        except (OSError, ValueError):
            return "0"
        if expected.get("type") == "bugfix":
            return "done"
        return str(expected.get("answer", "0"))

    def _inspect_tool_results(self, task: str, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            if message.get("role") != "tool":
                continue
            text = str(message.get("content") or "")
            for name, value in self.secrets.items():
                if value and value in text:
                    self.problems.append(f"{task}: the policy could read {name}")
            lines = text.splitlines()
            for index, line in enumerate(lines):
                # `id -u` prints the uid on the line before `ls -la`'s "total N".
                if (
                    line.startswith("total ")
                    and index > 0
                    and lines[index - 1].isdigit()
                ):
                    self.uids.add(lines[index - 1])

    def app(self) -> Any:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def create(request: Request) -> JSONResponse:
            body = await request.json()
            session_id = uuid.uuid4().hex[:12]
            self.sessions[session_id] = {
                "task": body["task"],
                "mode": body.get("mode", "solve"),
                "history": None,
                "turn": 0,
            }
            return JSONResponse({"session_id": session_id})

        async def chat(request: Request) -> JSONResponse:
            self.calls += 1
            session = self.sessions[request.path_params["session_id"]]
            body = await request.json()
            messages = body["messages"]
            stored = session["history"]
            task, turn, mode = session["task"], session["turn"], session["mode"]
            if stored is not None and (
                messages[: len(stored)] != stored
                or any(m.get("role") == "assistant" for m in messages[len(stored) :])
            ):
                problem = f"{task} turn {turn}: history not extended exactly"
                self.problems.append(problem)
                return JSONResponse({"error": problem}, status_code=400)
            self._inspect_tool_results(task, messages[len(stored or []) :])
            if mode == "fail" and turn == 1:
                return JSONResponse(
                    {"error": "backend transport error"}, status_code=502
                )
            if mode == "sleep":
                call = ("run_bash", {"command": "sleep 120"})
            elif turn == 0:
                call = ("run_bash", {"command": LOOK})
            else:
                call = ("submit", {"answer": self.answer(task)})
            message = {
                "role": "assistant",
                "content": "",
                "reasoning_content": f"turn {turn}: {call[0]}",
                "tool_calls": [
                    {
                        "id": f"call_{turn}",
                        "index": 0,
                        "type": "function",
                        "function": {"name": call[0], "arguments": json.dumps(call[1])},
                    }
                ],
            }
            session["history"] = [*messages, message]
            session["turn"] = turn + 1
            reply = {
                "id": f"chatcmpl-{turn}",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": "tool_calls"}
                ],
                "usage": {"prompt_tokens": 200 * (turn + 1), "completion_tokens": 20},
            }
            return JSONResponse(reply)

        return Starlette(
            routes=[
                Route("/sessions", create, methods=["POST"]),
                Route(
                    "/sessions/{session_id}/v1/chat/completions", chat, methods=["POST"]
                ),
            ]
        )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve_in_thread(app: Any, port: int) -> None:
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 20
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the stand-in session server did not start")
        time.sleep(0.1)


class Runner:
    """Plays one episode through /run, or through the Miles agent function."""

    def __init__(self, args: argparse.Namespace, standin_url: str) -> None:
        self.env_url = args.env_url.rstrip("/")
        self.standin_url = standin_url
        self.headers: dict[str, str] = {}
        if args.token_file:
            self.headers["Authorization"] = (
                f"Bearer {args.token_file.read_text().strip()}"
            )
        self.agent = None
        if args.via_agent_function:
            os.environ["BENCHFLOW_ENV_URL"] = self.env_url
            if args.token_file:
                os.environ["BENCHFLOW_ENV_TOKEN_FILE"] = str(args.token_file)
            import benchflow_agent_function  # the Miles example, from PYTHONPATH

            self.agent = benchflow_agent_function

    async def episode(
        self, http: httpx.AsyncClient, row: dict[str, Any], mode: str = "solve"
    ) -> dict[str, Any]:
        task = row["metadata"]["instance_id"]
        created = await http.post(
            f"{self.standin_url}/sessions", json={"task": task, "mode": mode}
        )
        base_url = f"{self.standin_url}/sessions/{created.json()['session_id']}"
        request_kwargs = {"temperature": 1.0, "max_tokens": 256}
        metadata = {**row["metadata"], "max_seq_len": 16384}
        if self.agent is None:
            body = {
                "session_url": f"{base_url}/v1",
                "prompt": row["prompt"],
                "request_kwargs": request_kwargs,
                "metadata": metadata,
            }
            response = await http.post(
                f"{self.env_url}/run", json=body, headers=self.headers, timeout=None
            )
            response.raise_for_status()
            out = response.json()
            return {"task": task, "discarded": out["dropped"], **out}
        try:
            out = await self.agent.run(
                base_url=base_url,
                prompt=row["prompt"],
                request_kwargs=request_kwargs,
                metadata=metadata,
            )
        except Exception as exc:  # InfraAbort, on a Miles that has it
            if type(exc).__name__ != "InfraAbort":
                raise
            return {
                "task": task,
                "discarded": True,
                "exit_status": exc.exit_status,
                "reward": None,
            }
        return {"task": task, "discarded": out.get("reward") is None, **out}

    async def abort(self, http: httpx.AsyncClient) -> Any:
        if self.agent is not None:
            await self.agent.abort(SimpleNamespace())
            return "agent function abort hook"
        return (await http.post(f"{self.env_url}/abort", headers=self.headers)).json()


async def main_async(args: argparse.Namespace) -> int:
    rows = dataset_rows(args.tasks_dir, include_tasks=tuple(args.task))
    if not rows:
        print("no tasks selected", file=sys.stderr)
        return 2
    secrets = {name: os.environ.get(name, "") for name in args.secret_env}
    standin = StandIn(args.answers_dir, secrets)
    port = _free_port()
    _serve_in_thread(standin.app(), port)
    runner = Runner(args, f"http://127.0.0.1:{port}")
    failures: list[str] = []
    async with httpx.AsyncClient(timeout=30) as http:
        started = time.monotonic()
        outcomes = await asyncio.gather(
            *(runner.episode(http, row) for row in rows),
            runner.episode(http, rows[0], mode="fail"),
        )
        *solved, failed = outcomes
        print(f"{len(outcomes)} episodes in {time.monotonic() - started:.1f}s")
        for out in outcomes:
            metrics = out.get("agent_metrics") or {}
            print(
                f"  {out['task']:<16} {out['exit_status']:<22} reward={out.get('reward')} "
                f"turns={metrics.get('turns')} start={metrics.get('env_setup_time', 0):.1f}s "
                f"verify={metrics.get('eval_time', 0):.1f}s total={metrics.get('total_time', 0):.1f}s"
            )
        for out in solved:
            if out["discarded"]:
                failures.append(f"{out['task']} was discarded ({out['exit_status']})")
            elif out["exit_status"] != "Submitted":
                failures.append(
                    f"{out['task']} ended {out['exit_status']}, not Submitted"
                )
            elif (
                args.answers_dir is not None
                and not out["task"].startswith("bugfix")
                and out["reward"] != 1.0
            ):
                failures.append(
                    f"{out['task']} scored {out['reward']} with the right answer"
                )
        if not (failed["discarded"] and failed["exit_status"] == "ModelEndpointFailed"):
            failures.append(
                f"a 502 after the policy acted gave {failed['exit_status']}, not a discard"
            )
        if not standin.uids or "0" in standin.uids:
            failures.append(
                f"sandbox uid {sorted(standin.uids)}: the policy must not be root"
            )
        print(
            f"policy uid in the sandbox: {sorted(standin.uids)}; secrets checked: {sorted(secrets)}"
        )

        # Abort: an episode stuck in a sleeping command must come back discarded.
        calls_before = standin.calls
        sleeper = asyncio.create_task(runner.episode(http, rows[0], mode="sleep"))
        for _ in range(120):
            await asyncio.sleep(1)
            if standin.calls > calls_before:
                break
        await asyncio.sleep(2)  # the sleeping command is running now
        aborted_at = time.monotonic()
        how = await runner.abort(http)
        result = await sleeper
        print(
            f"abort ({how}): the episode came back as {result['exit_status']} after {time.monotonic() - aborted_at:.1f}s"
        )
        if not (result["discarded"] and result["exit_status"] == "Aborted"):
            failures.append(f"abort returned {result['exit_status']}, not a discard")
        health = (await http.get(f"{runner.env_url}/health")).json()
        if health["sandboxes_in_use"] or health["in_flight"]:
            failures.append(f"sandboxes still held after the run: {health}")
    failures.extend(standin.problems)
    for problem in failures:
        print(f"FAIL: {problem}")
    print(
        "smoke passed" if not failures else f"smoke failed ({len(failures)} problems)"
    )
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-url", default="http://127.0.0.1:12100")
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument(
        "--answers-dir",
        type=Path,
        help="task folders with verifier/expected.json, to submit right answers",
    )
    parser.add_argument("--task", action="append", default=[], help="task id (repeat)")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument(
        "--secret-env",
        action="append",
        default=["DAYTONA_API_KEY"],
        help="environment variable whose value must never reach the policy (repeat)",
    )
    parser.add_argument(
        "--via-agent-function",
        action="store_true",
        help="play the episodes through the Miles example's benchflow_agent_function",
    )
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
