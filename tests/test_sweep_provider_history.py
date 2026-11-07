"""Guards the fix at 363d815b: every role contributes once.

Earlier roles live in rollout's ledger of retired runtimes (the same finding
as in rollout, fixed there on integration). These drive the real
provider switch with the ledger kept as rollout keeps it: each runtime holds
only its own role, so the ledger counts every role exactly once.
"""

import pytest

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers.litellm_logging import trajectory_from_litellm_callback_log
from tests.test_litellm_runtime import FakeLiteLLMServer


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["local", "remote-docker"])
@pytest.mark.parametrize("middle_skip", [False, True])
async def test_proxy_switch_keeps_all_usage_and_exchanges(
    monkeypatch, environment, middle_skip
):
    import json

    totals = iter([100, 200, 300])

    async def start(**kwargs):
        server = FakeLiteLLMServer("http://gateway.invalid", kwargs["route"])
        server.trajectory = trajectory_from_litellm_callback_log(
            json.dumps(
                {
                    "event": "success",
                    "request_model": kwargs["route"].requested_model,
                    "request": {"body": {}},
                    "response": {"model": kwargs["route"].requested_model},
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": next(totals) - 1,
                    },
                    "response_cost": 0.25,
                }
            ),
            session_id="s",
            agent_name="codex-acp",
        )
        return server

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    monkeypatch.setattr(runtime, "_start_sandbox_litellm", start)
    current = None
    ledger = []

    def retain(runtime_):
        if runtime_ is not None and all(old is not runtime_ for old in ledger):
            ledger.append(runtime_)

    for model in ["anthropic/claude-a", "anthropic/claude-b", "anthropic/claude-c"]:
        if middle_skip and current is not None:
            retain(current)
            _, current = await runtime.ensure_litellm_runtime(
                agent="oracle",
                agent_env={},
                model=None,
                runtime=current,
                environment=environment,
            )
        retain(current)
        _, current = await runtime.ensure_litellm_runtime(
            agent="codex-acp",
            agent_env={"ANTHROPIC_API_KEY": "dummy"},
            model=model,
            runtime=current,
            environment=environment,
            sandbox=object(),
            prior_runtimes=tuple(ledger),
        )
    retain(current)
    for _ in range(2):
        for runtime_ in ledger:
            await runtime.stop_litellm_runtime(runtime_)
    assert len(ledger) == 3
    assert all(len(r.server.trajectory.exchanges) == 1 for r in ledger)
    assert sum(runtime.extract_usage(r)["total_tokens"] for r in ledger) == 600
    assert sum(r.server.trajectory.total_cost_usd for r in ledger) == pytest.approx(
        0.75
    )


@pytest.mark.asyncio
async def test_live_and_final_history_survive_route_switch_and_last_nonproxy_role(
    tmp_path, monkeypatch
):
    import json

    from tests.test_live_llm_trajectory import _callback_line

    class MemoryServer(runtime.LiteLLMProcess):
        def __init__(self, route, tokens):
            self.route = route
            self.session_id = "s"
            self.agent_name = "codex-acp"
            self.trajectory = None
            self.raw = _callback_line(
                content=str(tokens),
                usage={"prompt_tokens": tokens, "completion_tokens": 0},
            ).encode()
            self.stopped = False

        @property
        def base_url(self):
            return "http://gateway.invalid"

        async def is_running(self):
            return not self.stopped

        async def _read_callback_chunk(self, offset, limit):
            return runtime._CallbackChunk(
                self.raw[offset : offset + limit], len(self.raw)
            )

        async def stop(self):
            await self._stop_live_capture()
            self.trajectory = trajectory_from_litellm_callback_log(
                self.raw.decode(), session_id="s", agent_name=self.agent_name
            )
            self.stopped = True
            self._reconcile_live_capture()

    amounts = iter([100, 200, 300])

    async def start(**kwargs):
        return MemoryServer(kwargs["route"], next(amounts))

    monkeypatch.setattr(runtime, "_start_host_litellm", start)
    path = tmp_path / "llm.jsonl"
    current = None
    ledger = []
    for count, model in enumerate(
        ["anthropic/claude-a", "anthropic/claude-b", "anthropic/claude-c"], 1
    ):
        if current is not None:
            ledger.append(current)
        _, current = await runtime.ensure_litellm_runtime(
            agent="codex-acp",
            agent_env={"ANTHROPIC_API_KEY": "dummy"},
            model=model,
            runtime=current,
            environment="local",
            live_trajectory_path=path,
            prior_runtimes=tuple(ledger),
        )
        await current.server._capture_live_records()
        # The live file holds every role; the process counts only its own.
        assert current.server.live_usage_tokens() == [100, 200, 300][count - 1]
        assert len(path.read_text().splitlines()) == count
    ledger.append(current)
    _, current = await runtime.ensure_litellm_runtime(
        agent="oracle", agent_env={}, model=None, runtime=current, environment="local"
    )
    assert current is None
    for runtime_ in ledger:
        await runtime.stop_litellm_runtime(runtime_)
    assert sum(runtime.extract_usage(r)["total_tokens"] for r in ledger) == 600
    assert len([json.loads(line) for line in path.read_text().splitlines()]) == 3
