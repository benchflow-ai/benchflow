"""The BenchFlow environment server a Miles agent function calls.

Run it next to the Miles rollout workers, in its own Python environment
(BenchFlow and Miles pin different versions of shared packages):

    python -m benchflow.integrations.miles serve --tasks-dir tasks/train \\
        --sandbox daytona --max-sandboxes 32

Endpoints (JSON):

- ``POST /run``: one episode. Body: ``session_url`` (the session's
  OpenAI-compatible root, ending in ``/v1``), ``prompt``, ``request_kwargs``,
  ``metadata`` (``instance_id`` names the task folder). Answers with the
  outcome: ``reward``, ``dropped``, ``exit_status``, ``flagged``,
  ``eval_report``, ``agent_metrics``. A client that disconnects cancels its
  episode, and the sandbox is released.
- ``POST /abort``: cancel every episode in flight (Miles calls its agent
  function's ``abort`` hook when a rollout step has enough samples).
- ``GET /tasks``: the task ids this server can run.
- ``GET /health``: counts of episodes, sandboxes and exit statuses.

The server binds to 127.0.0.1 by default. Binding another address needs a
bearer token (``--token-file``), because anyone who reaches ``/run`` can
start sandboxes on your account and make this process call a URL of their
choice.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from benchflow.integrations.miles.episode import (
    ABORTED_EXIT_STATUS,
    EpisodeRequest,
    EpisodeRequestError,
    EpisodeSettings,
    run_episode,
)
from benchflow.integrations.trl.spec import _discover_task_dirs

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass
class ServerState:
    settings: EpisodeSettings
    max_sandboxes: int
    token: str | None = None
    tasks: frozenset[str] = frozenset()
    slots: asyncio.Semaphore | None = None
    client: httpx.AsyncClient | None = None
    in_flight: dict[str, asyncio.Task] = field(default_factory=dict)
    closers: set[asyncio.Task] = field(default_factory=set)
    exit_status: Counter = field(default_factory=Counter)
    episodes: int = 0
    started_at: float = field(default_factory=time.time)


def load_tasks(settings: EpisodeSettings) -> frozenset[str]:
    """The runnable task ids under ``settings.tasks_dir``; refuses an empty folder."""

    tasks = frozenset(
        path.name
        for path in _discover_task_dirs(
            settings.tasks_dir, include_tasks=set(), exclude_tasks=set()
        )
    )
    if not tasks:
        raise ValueError(f"no runnable BenchFlow tasks under {settings.tasks_dir}")
    return tasks


def create_app(
    settings: EpisodeSettings,
    *,
    max_sandboxes: int,
    token: str | None = None,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> Any:
    """The Starlette app; ``settings`` is normalized here.

    ``client_factory`` makes the HTTP client episodes call the session server
    with (tests pass a scripted one).
    """

    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    if max_sandboxes < 1:
        raise ValueError("max_sandboxes must be >= 1")
    state = ServerState(settings=settings.normalized(), max_sandboxes=max_sandboxes)
    state.token = token or None
    state.tasks = load_tasks(state.settings)

    @contextlib.asynccontextmanager
    async def lifespan(app: Any):
        state.slots = asyncio.Semaphore(max_sandboxes)
        state.client = (
            client_factory()
            if client_factory is not None
            else httpx.AsyncClient(
                limits=httpx.Limits(max_connections=max(64, 4 * max_sandboxes)),
                timeout=httpx.Timeout(state.settings.request_timeout_sec),
            )
        )
        logger.info(
            "BenchFlow environment for Miles: %d tasks under %s, %s sandboxes (at most %d)",
            len(state.tasks),
            state.settings.tasks_dir,
            state.settings.harness.environment,
            max_sandboxes,
        )
        try:
            yield
        finally:
            await _cancel_all(state)
            if state.closers:
                await asyncio.wait(set(state.closers), timeout=120)
            await state.client.aclose()

    def authorized(request: Request) -> bool:
        if state.token is None:
            return True
        header = request.headers.get("authorization", "")
        return hmac.compare_digest(header, f"Bearer {state.token}")

    async def health(request: Request) -> JSONResponse:
        slots = state.slots
        return JSONResponse(
            {
                "status": "ok",
                "tasks": len(state.tasks),
                "sandbox": state.settings.harness.environment,
                "in_flight": len(state.in_flight),
                "max_sandboxes": state.max_sandboxes,
                "sandboxes_in_use": (
                    state.max_sandboxes - slots._value if slots is not None else 0
                ),
                "episodes": state.episodes,
                "exit_status": dict(state.exit_status),
                "uptime_sec": round(time.time() - state.started_at, 1),
            }
        )

    async def tasks(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return JSONResponse({"tasks": sorted(state.tasks)})

    async def run(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            episode = EpisodeRequest.from_json(await request.json())
        except EpisodeRequestError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except ValueError as exc:
            return JSONResponse({"error": f"invalid JSON: {exc}"}, status_code=400)
        task_id = episode.metadata.get("instance_id") or episode.metadata.get("task_id")
        if task_id not in state.tasks:
            return JSONResponse(
                {"error": f"unknown task {task_id!r}; GET /tasks lists this server's tasks"},
                status_code=400,
            )
        if episode.episode_id in state.in_flight:
            return JSONResponse(
                {"error": f"episode {episode.episode_id} is already running"},
                status_code=409,
            )
        assert state.client is not None
        job = asyncio.create_task(
            run_episode(
                episode,
                state.settings,
                client=state.client,
                sandbox_slots=state.slots,
                closers=state.closers,
            )
        )
        state.in_flight[episode.episode_id] = job
        watcher = asyncio.create_task(_cancel_on_disconnect(request, job))
        try:
            outcome = await job
        except asyncio.CancelledError:
            if not job.cancelled():
                raise  # this handler itself is being cancelled
            _count(state, ABORTED_EXIT_STATUS)
            return JSONResponse(_aborted_response(episode, task_id))
        except EpisodeRequestError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        finally:
            watcher.cancel()
            state.in_flight.pop(episode.episode_id, None)
        _count(state, outcome.exit_status or "unknown")
        body = outcome.response()
        logger.info(
            "episode %s %s: %s reward=%s turns=%d tools=%d %.1fs",
            outcome.episode_id,
            outcome.task_id,
            outcome.exit_status,
            body["reward"],
            outcome.turns,
            outcome.tool_calls,
            outcome.timings.get("total_sec", 0.0),
        )
        return JSONResponse(body)

    async def abort(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        cancelled = await _cancel_all(state)
        logger.info("abort: cancelled %d episodes in flight", cancelled)
        return JSONResponse({"cancelled": cancelled})

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/tasks", tasks, methods=["GET"]),
            Route("/run", run, methods=["POST"]),
            Route("/abort", abort, methods=["POST"]),
        ],
        lifespan=lifespan,
    )


def serve(
    settings: EpisodeSettings,
    *,
    host: str = "127.0.0.1",
    port: int = 12100,
    max_sandboxes: int = 16,
    token: str | None = None,
) -> None:
    """Run the environment server until interrupted."""

    import uvicorn

    if host not in _LOOPBACK_HOSTS and not token:
        raise ValueError(
            f"binding {host} exposes the server beyond this machine; pass --token-file "
            "(a bearer token the Miles agent function sends), or bind 127.0.0.1"
        )
    app = create_app(settings, max_sandboxes=max_sandboxes, token=token)
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)


async def _cancel_all(state: ServerState) -> int:
    jobs = [job for job in state.in_flight.values() if not job.done()]
    for job in jobs:
        job.cancel()
    if jobs:
        await asyncio.wait(jobs, timeout=120)
    return len(jobs)


async def _cancel_on_disconnect(request: Any, job: asyncio.Task) -> None:
    """Cancel the episode when the agent function goes away (its task was cancelled)."""

    while not job.done():
        if await request.is_disconnected():
            logger.info("client went away; cancelling its episode")
            job.cancel()
            return
        await asyncio.sleep(1.0)


def _count(state: ServerState, exit_status: str) -> None:
    state.episodes += 1
    state.exit_status[exit_status] += 1


def _aborted_response(episode: EpisodeRequest, task_id: Any) -> dict[str, Any]:
    return {
        "reward": None,
        "dropped": True,
        "exit_status": ABORTED_EXIT_STATUS,
        "flagged": False,
        "detail": "episode cancelled before it finished (abort or client disconnect)",
        "eval_report": {"task_id": task_id, "episode_id": episode.episode_id},
        "agent_metrics": {},
    }


__all__ = ["ServerState", "create_app", "load_tasks", "serve"]
