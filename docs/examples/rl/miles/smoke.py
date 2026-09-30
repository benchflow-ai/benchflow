"""Check the BenchFlow side of the Miles connector, without a GPU.

Run it against a running environment server (``python -m
benchflow.integrations.miles serve``). It starts a stand-in for Miles's
session server that plays a scripted policy and refuses any request that does
not extend the session's history exactly (the check Miles's session server
makes), then runs real episodes through the environment server's ``/run``,
and one through ``/abort``:

    python docs/examples/rl/miles/smoke.py --env-url http://127.0.0.1:12100 \\
        --tasks-dir tasks/train --answers-dir tasks-with-answers/train \\
        --task sql-000000 --task log-000001 --task bugfix-000003

The scripted policy lists ``/workdir`` with ``run_bash``, then submits the
expected answer when ``--answers-dir`` holds the task's
``verifier/expected.json`` (a question task then scores 1), else ``0``. Its
assistant turns carry ``reasoning_content``, so the check also covers that
field. The abort check runs one more episode whose command sleeps, calls
``/abort``, and expects the episode to come back discarded (``Aborted``) with
its sandbox released.

Exits 0 when every check passes. Costs a few sandbox-minutes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from benchflow.integrations.miles import dataset_rows


class StandIn:
    """A scripted policy behind Miles's session-server routes."""

    def __init__(self, answers_dir: Path | None) -> None:
        self.answers_dir = answers_dir
        self.sessions: dict[str, dict[str, Any]] = {}
        self.violations: list[str] = []
        self.calls = 0

    def answer(self, task: str) -> str:
        if self.answers_dir is None:
            return "0"
        path = self.answers_dir / task / "verifier" / "expected.json"
        try:
            expected = json.loads(path.read_text())
        except (OSError, ValueError):
            return "0"
        return (
            "done"
            if expected.get("type") == "bugfix"
            else str(expected.get("answer", "0"))
        )

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
            if stored is not None and (
                messages[: len(stored)] != stored
                or any(m.get("role") == "assistant" for m in messages[len(stored) :])
            ):
                problem = f"{session['task']} turn {session['turn']}: history not extended exactly"
                self.violations.append(problem)
                return JSONResponse({"error": problem}, status_code=400)
            turn = session["turn"]
            if session["mode"] == "sleep":
                call = ("run_bash", {"command": "sleep 120"})
            elif turn == 0:
                call = ("run_bash", {"command": "ls -la /workdir"})
            else:
                call = ("submit", {"answer": self.answer(session["task"])})
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
            return JSONResponse(
                {
                    "id": f"chatcmpl-{turn}",
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": "tool_calls"}
                    ],
                    "usage": {
                        "prompt_tokens": 200 * (turn + 1),
                        "completion_tokens": 20,
                    },
                }
            )

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


async def _episode(
    http: httpx.AsyncClient,
    env_url: str,
    standin_url: str,
    row: dict[str, Any],
    headers: dict[str, str],
    *,
    mode: str = "solve",
) -> dict[str, Any]:
    task = row["metadata"]["instance_id"]
    created = await http.post(
        f"{standin_url}/sessions", json={"task": task, "mode": mode}
    )
    session_url = f"{standin_url}/sessions/{created.json()['session_id']}/v1"
    body = {
        "session_url": session_url,
        "prompt": row["prompt"],
        "request_kwargs": {"temperature": 1.0, "max_tokens": 256},
        "metadata": {**row["metadata"], "max_seq_len": 16384},
    }
    response = await http.post(
        f"{env_url}/run", json=body, headers=headers, timeout=None
    )
    response.raise_for_status()
    return {"task": task, **response.json()}


async def main_async(args: argparse.Namespace) -> int:
    env_url = args.env_url.rstrip("/")
    headers = {}
    if args.token_file:
        headers["Authorization"] = f"Bearer {args.token_file.read_text().strip()}"
    rows = dataset_rows(args.tasks_dir, include_tasks=tuple(args.task))
    if not rows:
        print("no tasks selected", file=sys.stderr)
        return 2
    standin = StandIn(args.answers_dir)
    port = _free_port()
    _serve_in_thread(standin.app(), port)
    standin_url = f"http://127.0.0.1:{port}"
    failures: list[str] = []
    async with httpx.AsyncClient(timeout=30) as http:
        started = time.monotonic()
        outcomes = await asyncio.gather(
            *(_episode(http, env_url, standin_url, row, headers) for row in rows)
        )
        print(f"{len(outcomes)} episodes in {time.monotonic() - started:.1f}s")
        for out in outcomes:
            metrics = out.get("agent_metrics") or {}
            print(
                f"  {out['task']:<16} {out['exit_status']:<22} reward={out['reward']} "
                f"turns={metrics.get('turns')} start={metrics.get('env_setup_time', 0):.1f}s "
                f"verify={metrics.get('eval_time', 0):.1f}s total={metrics.get('total_time', 0):.1f}s"
            )
            if out["dropped"]:
                failures.append(f"{out['task']} was dropped: {out.get('detail')}")
            elif out["exit_status"] != "Submitted":
                failures.append(
                    f"{out['task']} ended {out['exit_status']}, not Submitted"
                )
            expected_one = args.answers_dir is not None and not out["task"].startswith(
                "bugfix"
            )
            if expected_one and out["reward"] != 1.0:
                failures.append(
                    f"{out['task']} scored {out['reward']} with the right answer"
                )

        # Abort: an episode stuck in a sleeping command must come back discarded.
        sleeper = asyncio.create_task(
            _episode(http, env_url, standin_url, rows[0], headers, mode="sleep")
        )
        for _ in range(120):
            await asyncio.sleep(1)
            health = (await http.get(f"{env_url}/health")).json()
            if health["in_flight"] and standin.calls > sum(
                o["agent_metrics"].get("turns", 0) for o in outcomes
            ):
                break
        await asyncio.sleep(2)  # the sleeping command is running now
        aborted_at = time.monotonic()
        cancelled = (await http.post(f"{env_url}/abort", headers=headers)).json()
        result = await sleeper
        print(
            f"abort: {cancelled}, the episode came back as {result['exit_status']} "
            f"after {time.monotonic() - aborted_at:.1f}s"
        )
        if not (result["dropped"] and result["exit_status"] == "Aborted"):
            failures.append(f"abort returned {result['exit_status']}, not a discard")
        health = (await http.get(f"{env_url}/health")).json()
        if health["sandboxes_in_use"] or health["in_flight"]:
            failures.append(f"sandboxes still held after the run: {health}")
    failures.extend(standin.violations)
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
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
