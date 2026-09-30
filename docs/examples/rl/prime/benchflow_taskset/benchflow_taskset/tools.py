"""The episode's two tools, ``run_bash`` and ``submit``, served over MCP in-process.

Verifiers' own toolsets run as separate processes and are torn down before a
rollout is scored, so they cannot own a sandbox that must still be alive for the
verifier. This server instead runs in the env worker's event loop for one
episode, next to the ``BenchFlowSession`` it calls, on ``127.0.0.1`` behind an
unguessable path. The model's commands never execute here: ``run_bash`` forwards
them to the BenchFlow sandbox.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import socket
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchflow_taskset.session import BenchFlowSession

RUN_BASH_DESCRIPTION = (
    "Run a bash command in the task sandbox, in the task's working directory. "
    "Returns the combined standard output and standard error (truncated if long), "
    "followed by the exit code when it is not zero."
)
SUBMIT_DESCRIPTION = (
    "Submit your final answer and end the episode. The answer, if given, is written "
    "to {submit_path}, replacing that file; leave it empty if you already wrote the "
    "file. Call this once, when you are done."
)
SUBMIT_DESCRIPTION_NO_PATH = "End the episode. Call this once, when you are done."


def build_server(session: BenchFlowSession, token: str):
    from mcp.server.mcpserver import MCPServer
    from mcp.server.transport_security import TransportSecuritySettings

    server = MCPServer("benchflow")

    async def run_bash(command: str) -> str:
        return await session.run_bash(command)

    async def submit(answer: str = "") -> str:
        return await session.submit(answer)

    server.add_tool(run_bash, name="run_bash", description=RUN_BASH_DESCRIPTION)
    server.add_tool(
        submit,
        name="submit",
        description=(
            SUBMIT_DESCRIPTION.format(submit_path=session.submit_path)
            if session.submit_path
            else SUBMIT_DESCRIPTION_NO_PATH
        ),
    )
    return server.streamable_http_app(
        streamable_http_path=f"/{token}/mcp",
        json_response=True,
        stateless_http=True,
        # Reached over loopback only, never from a browser.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )


def _server_class():
    import uvicorn

    class QuietServer(uvicorn.Server):
        """uvicorn without signal capture. Its default swaps the process's SIGINT and
        SIGTERM handlers while it serves and restores them after; with several episode
        servers overlapping in one env worker, those restores would leave stale handlers."""

        @contextlib.contextmanager
        def capture_signals(self):
            yield

    return QuietServer


@contextlib.asynccontextmanager
async def serve_tools(session: BenchFlowSession) -> AsyncIterator[str]:
    """Serve the session's tools for the duration of the block; yields the MCP URL."""
    import uvicorn

    token = secrets.token_urlsafe(24)
    app = build_server(session, token)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = _server_class()(uvicorn.Config(app, log_level="critical", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(500):
            if server.started or task.done():
                break
            await asyncio.sleep(0.01)
        if task.done():
            task.result()  # raise the startup failure
            raise RuntimeError("the tool server stopped during startup")
        if not server.started:
            raise RuntimeError("the tool server did not start within 5 seconds")
        yield f"http://127.0.0.1:{port}/{token}/mcp"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.shield(task), 10)
        except (asyncio.TimeoutError, Exception):
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        sock.close()
