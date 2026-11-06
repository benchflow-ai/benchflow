"""DFX-SCALE round 3: the review of 71eee926d (M1, M3, M4, L1-L7)."""

import json
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from benchflow.acp.session import ACPSession
from benchflow.acp.watchdog import IdleWatchdog
from benchflow.evaluation import RetryConfig
from benchflow.providers import litellm_request_guard as guard
from benchflow.providers.litellm_runtime import LiteLLMProcess

# p3's own errors (from its result.json files), cut.
GUARD = (
    "SandboxDaemonGuardError: sandbox startup refused: the agent user could not "
    "be kept away from Daytona's root toolbox daemon: the guard did not report "
    "(no report). Running anyway would let the agent reach root and rewrite its "
    "own grade."
)
GUARD_124 = "SandboxDaemonGuardError: sandbox startup refused: exec status 124"
FREEZER = (
    "KeptServiceFreezerUnavailable: pypi-server keeps services on port 8080 "
    "(x-keep-services, from BenchFlow's Terminal-Bench 2 list (pypi-server)). "
    "BenchFlow holds them frozen in a cgroup of root's from the kill after the "
    "agent's turn until the verifier starts, which a daytona sandbox cannot give"
)
SEALING = (
    "RuntimeError: BenchFlow could not prepare its protected runtime "
    "(/opt/benchflow/rt) in the sandbox"
)
API_502 = "provider api error [server_error/transient] HTTP 502 on 3/3 requests"
ACP_JSON = "ACP error -32603: Internal error: Expecting value: line 60 column 1"
INVALID = (
    "invalid provider response: upstream HTTP 502, non-JSON body (8 bytes), "
    "upstream 0.100s; next: inspect trajectory/provider_requests.jsonl and "
    "retry the provider"
)


def _replaced(original):
    from benchflow._utils.scoring import ORIGINAL_ERROR

    return f"{INVALID}{ORIGINAL_ERROR}{original}"


# What dc509087a decided for each error (should_retry, reruns_on_resume),
# RetryConfig() defaults; the review's table, plus freezer and sealing.
@pytest.mark.parametrize(
    ("error", "category", "retried"),
    [
        (GUARD, "guard_timeout", True),
        (GUARD_124, "guard_timeout", True),
        (FREEZER, "kept_service_freezer", False),
        (SEALING, "runtime_sealing", False),
        ("API", "invalid_provider_response", True),
        ("ACP", "invalid_provider_response", True),
    ],
)
def test_reporting_categories_keep_retry_and_resume(error, category, retried):
    """M1: the four categories change neither retry nor resume."""
    from benchflow._utils.scoring import classify_error

    error = {"API": _replaced(API_502), "ACP": _replaced(ACP_JSON)}.get(error, error)
    assert classify_error(error) == category
    retry = RetryConfig()
    assert retry.should_retry(error, category=category) is retried
    assert retry.reruns_on_resume(error, category=category) is retried
    # A stored result carries the category; a bare error is classified.
    assert retry.should_retry(error) is retried


def test_excluded_original_category_stays_excluded():
    """M1: a run that excludes api_error from retries still excludes it."""
    retry = RetryConfig(exclude_categories={"api_error"})
    error = _replaced(API_502)
    assert not retry.should_retry(error, category="invalid_provider_response")
    retry = RetryConfig(retry_on_infra=False)
    assert not retry.should_retry(GUARD, category="guard_timeout")


class _Server:
    def __init__(self, timing, invalid):
        self._timing, self._invalid = timing, invalid

    def request_summary(self):
        return dict(self._timing), self._invalid


def _rollout(tmp_path):
    from benchflow.rollout import Rollout, RolloutConfig
    from tests.test_keep_services_tb2 import _copy

    return Rollout(
        RolloutConfig(
            task_path=_copy(tmp_path),
            agent="claude-agent-acp",
            environment="docker",
            jobs_dir=tmp_path / "jobs",
        )
    )


async def test_teardown_counts_a_reused_runtime_once_and_keeps_the_first(tmp_path):
    """L5 and M1: a runtime retired and still current adds its wait once; the
    first invalid response is kept, and the error it replaces stays readable
    for retry."""
    rollout = _rollout(tmp_path)
    first = SimpleNamespace(server=_Server({"provider_wait": 2.0}, INVALID))
    reused = SimpleNamespace(server=_Server({"provider_wait": 3.0}, None))
    rollout._retired_usage_runtimes = [first, reused]
    rollout._usage_runtime = reused
    rollout._planes = MagicMock(
        stop_provider_runtime=AsyncMock(), extract_usage=MagicMock(return_value={})
    )
    rollout._timing = {"agent_execution": 10.0}
    rollout._error = API_502
    await rollout._teardown_usage_runtime()
    assert rollout._timing["provider_wait"] == 5.0
    assert rollout._timing["agent_without_provider_wait"] == 5.0
    assert rollout._invalid_provider_response_cached == INVALID
    assert rollout._error == _replaced(API_502)
    assert RetryConfig().should_retry(rollout._error)


def _process(events):
    process = LiteLLMProcess.__new__(LiteLLMProcess)
    process._request_events = {
        (row["event"], row.get("attempt_id", row["request_id"])): row for row in events
    }
    return process


def test_stalled_requests_are_those_without_a_response():
    now = time.time()
    process = _process(
        [
            {"event": "request_start", "request_id": "a", "time": now - 90},
            {"event": "request_start", "request_id": "b", "time": now - 30},
            {"event": "response_start", "request_id": "b", "time": now - 20},
            {"event": "request_start", "request_id": "c", "time": now - 200},
            {"event": "request_end", "request_id": "c", "time": now - 10},
        ]
    )
    count, age = process.stalled_model_requests()
    assert count == 1 and 89 < age < 95
    # Both still count as in flight for the idle deadline.
    assert process.model_request_activity()[0] == 2


@pytest.fixture
def capture(tmp_path, monkeypatch):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    return log


async def test_guard_records_when_the_response_starts(capture):
    """M3: the proxy's answer starts the response; until then it is a stall."""
    from tests.test_dfx_scale import _request

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"{}"})

    await _request(app)
    rows = [json.loads(line) for line in capture.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "request_start",
        "response_start",
        "request_end",
    ]
    assert rows[1]["status"] == 200
    assert rows[1]["request_id"] == rows[0]["request_id"]


async def test_lifecycle_write_never_fails_the_request(capture):
    """L4: a log that cannot be written (here a folder) leaves the request
    and its outcome alone."""
    from tests.test_dfx_scale import _request

    capture.mkdir()
    sent = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 503})
        await send({"type": "http.response.body", "body": b"busy"})
        sent.append(True)

    await _request(app)
    assert sent == [True]
    attempt = guard._UpstreamAttempt(httpx.Request("POST", "https://x.test"))
    attempt.end()


def _timeout(idle):
    from benchflow.diagnostics import (
        AgentPromptTimeoutDiagnostic,
        AgentPromptTimeoutError,
        IdleTimeoutDiagnostic,
        IdleTimeoutError,
    )

    if idle:
        return IdleTimeoutError(
            "Agent idle for 30s with no new tool call",
            IdleTimeoutDiagnostic(idle_timeout_sec=30),
        )
    return AgentPromptTimeoutError(
        "Agent exceeded wall-clock budget of 180s",
        trajectory=[],
        diagnostic=AgentPromptTimeoutDiagnostic(timeout_sec=180),
    )


@pytest.mark.parametrize("idle", [False, True])
async def test_unanswered_request_is_a_retried_provider_stall(tmp_path, idle):
    """M3: the deadline ends while a model request has no response: the
    result names the stall and its age, is unscored, and is retried and
    resumed as a transient provider error."""
    from benchflow._utils.scoring import PROVIDER_STALL

    rollout = _rollout(tmp_path)
    process = _process(
        [{"event": "request_start", "request_id": "a", "time": time.time() - 181.0}]
    )
    rollout._usage_runtime = SimpleNamespace(server=process)
    rollout._executed_prompts = ["hello"]
    rollout._config.agent_idle_timeout = 30
    rollout._record_agent_timeout(_timeout(idle), agent_phase=True)
    rollout._rewards = {"reward": 0.0}  # the verifier's score of the timeout
    rollout._maybe_classify_api_error()
    assert rollout._error.startswith(PROVIDER_STALL)
    assert "1 model request(s) had no response" in rollout._error
    assert "pending 181." in rollout._error
    assert rollout._rewards is None
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir()
    rollout._started_at = datetime.now()
    result = rollout._build_result(result_filename=None)
    assert result.error_category == "api_error"
    assert result.rewards is None
    retry = RetryConfig()
    assert retry.should_retry(result.error, category=result.error_category)
    assert retry.reruns_on_resume(result.error, category=result.error_category)


@pytest.mark.parametrize(
    "events",
    [
        # A response on its way to the agent.
        [("request_start", 90), ("response_start", 80)],
        # A request younger than the idle budget: a busy agent's wall timeout.
        [("request_start", 20)],
    ],
)
async def test_other_pending_requests_keep_the_timeout(tmp_path, events):
    """M3: only a request unanswered for longer than the idle budget is a
    provider stall; the wall timeout stays scored as at dc509087a."""
    rollout = _rollout(tmp_path)
    now = time.time()
    rollout._usage_runtime = SimpleNamespace(
        server=_process(
            [
                {"event": event, "request_id": "a", "time": now - ago}
                for event, ago in events
            ]
        )
    )
    rollout._config.agent_idle_timeout = 30
    rollout._record_agent_timeout(_timeout(False), agent_phase=True)
    assert rollout._error == "Agent exceeded wall-clock budget of 180s"
    assert not rollout._provider_stall


def test_pending_request_counts_for_a_bounded_time():
    """L6: a request whose end never comes holds the idle deadline off for
    three idle budgets, not for the rest of the prompt."""
    session = ACPSession("test")
    session.model_request_activity = lambda: (1, 0.0, 1.0)
    watchdog = IdleWatchdog.start(
        session, idle_timeout_sec=30, wall_timeout_sec=1000, now=0
    )
    for now in range(10, 90, 5):
        session.model_request_activity = lambda now=now: (1, float(now), 1.0)
        watchdog.observe(session, now=now)
        assert not watchdog.idle_expired(now + 29)
    session.model_request_activity = lambda: (1, 95.0, 1.0)
    watchdog.observe(session, now=95)
    assert not watchdog.idle_expired(110)  # 85 s was the last that counted
    assert watchdog.idle_expired(115)


def test_expired_pending_tool_call_still_ends_the_prompt():
    """L6: a pending model request does not hide an expired pending tool call."""
    session = ACPSession("test")
    session.pending_tool_call_state = lambda: (("call-1", 0),)
    session.model_request_activity = lambda: (1, 5.0, 1.0)
    watchdog = IdleWatchdog.start(
        session,
        idle_timeout_sec=30,
        wall_timeout_sec=10000,
        now=0,
        pending_grace_sec=90,
    )
    watchdog.observe(session, now=91)
    assert watchdog.expired_pending_ids(91) == ("call-1",)
    assert watchdog.idle_expired(91)


async def test_rollout_wires_the_proxy_activity_into_the_session(tmp_path):
    """L7: execute() gives the session the usage proxy's own activity."""
    rollout = _rollout(tmp_path)
    now = time.time()
    server = _process([{"event": "request_start", "request_id": "a", "time": now - 5}])
    rollout._usage_runtime = SimpleNamespace(server=server)
    rollout._session = ACPSession("wired")
    rollout._acp_client = SimpleNamespace()
    rollout._resolved_prompts = ["hello"]
    seen = []

    async def execute_prompts(client, session, prompts, timeout, *, idle_timeout):
        seen.append(session.model_request_activity())
        raise RuntimeError("stop after the wiring")

    rollout._planes = SimpleNamespace(execute_prompts=execute_prompts)
    with pytest.raises(RuntimeError, match="stop after the wiring"):
        await rollout.execute()
    assert seen and seen[0][0] == 1 and seen[0][1] >= 5


# --- M4: the cached layers only where they pay off ---


@pytest.fixture
def layers(monkeypatch):
    from benchflow.sandbox import daytona_layers

    monkeypatch.setattr(daytona_layers, "_BUILT", set())
    monkeypatch.setattr(daytona_layers, "_BUILDING", set())
    monkeypatch.delenv(daytona_layers.LAYER_REUSE_ENV, raising=False)
    monkeypatch.delenv("BENCHFLOW_DAYTONA_SNAPSHOTS", raising=False)
    return daytona_layers


def test_claims_build_once_per_recipe_and_reuse_what_was_built(layers, monkeypatch):
    keys = {"proxy-litellm": "a", "agent-x": "b"}
    assert layers.claim_layer_build(keys) is None  # one trial: install fresh
    monkeypatch.setenv(layers.LAYER_REUSE_ENV, "1")
    assert layers.claim_layer_build(keys) == "build"
    assert layers.claim_layer_build(keys) is None  # no trial waits on another
    layers.layer_build_finished(keys, built=True)
    monkeypatch.delenv(layers.LAYER_REUSE_ENV)
    assert layers.claim_layer_build(keys) == "built"
    layers.layer_build_finished(keys, built=False)
    assert layers.claim_layer_build(keys) is None


async def test_single_trial_installs_in_the_sandbox(layers, tmp_path, monkeypatch):
    """M4: with no reuse planned and nothing built, the recipe is the task
    image alone (dc509087a's effective path) and sandbox.json says why."""
    pytest.importorskip("daytona")
    from tests.test_daytona_install_layers import FakeDaytona, _start

    fake = _fake_client(monkeypatch, FakeDaytona)
    sandbox = await _start(fake, tmp_path / "trial-1", only_if_reused=True)
    recipe = fake.requests[0].image.dockerfile()
    assert "RUN " not in recipe
    assert sandbox._install_layer_keys == {}
    record = json.loads((tmp_path / "trial-1" / "sandbox.json").read_text())
    assert record["install_layers"]["mode"] == "fresh"
    assert "no other trial" in record["install_layers"]["fresh_reason"]


def _fake_client(monkeypatch, fake_type, **kwargs):
    from benchflow.sandbox import daytona as daytona_mod

    fake = fake_type(**kwargs)
    manager = SimpleNamespace(get_client=AsyncMock(return_value=fake))
    monkeypatch.setattr(
        daytona_mod,
        "DaytonaClientManager",
        SimpleNamespace(get_instance=AsyncMock(return_value=manager)),
    )
    monkeypatch.delenv("BENCHFLOW_FRESH_INSTALL", raising=False)
    return fake


async def test_reused_image_builds_once_then_reuses(layers, tmp_path, monkeypatch):
    """M4: with reuse planned, the first trial builds the layers and the next
    ones carry them; a build that fails is not counted as built."""
    pytest.importorskip("daytona")
    from tests.test_daytona_install_layers import FakeDaytona, _start

    monkeypatch.setenv(layers.LAYER_REUSE_ENV, "1")
    fake = _fake_client(monkeypatch, FakeDaytona)
    first = await _start(fake, tmp_path / "trial-1", only_if_reused=True)
    assert "RUN " in fake.requests[0].image.dockerfile()
    monkeypatch.delenv(layers.LAYER_REUSE_ENV)
    second = await _start(fake, tmp_path / "trial-2", only_if_reused=True)
    assert second._install_layer_keys == first._install_layer_keys != {}

    failing = _fake_client(monkeypatch, FakeDaytona, fail_layered_builds=True)
    monkeypatch.setattr(layers, "_BUILT", set())
    monkeypatch.setenv(layers.LAYER_REUSE_ENV, "1")
    fallback = await _start(failing, tmp_path / "trial-3", only_if_reused=True)
    assert fallback._install_layer_keys == {}
    assert not layers._BUILT and not layers._BUILDING


def test_more_than_one_trial_plans_reuse(layers, monkeypatch):
    """M4: --trials N > 1 (and a job config's n_attempts) plan reuse for the
    loop's length only."""
    import os

    from benchflow.cli.eval_artifacts import _plan_layer_reuse

    restore = _plan_layer_reuse(1)
    assert layers.LAYER_REUSE_ENV not in os.environ
    restore()
    restore = _plan_layer_reuse(4)
    assert os.environ[layers.LAYER_REUSE_ENV] == "1"
    restore()
    assert layers.LAYER_REUSE_ENV not in os.environ


async def test_rollout_asks_for_layers_only_if_reused(tmp_path, monkeypatch):
    """M4: the trial path plans its Daytona layers with only_if_reused."""
    from benchflow.rollout import Rollout, RolloutConfig
    from tests.test_keep_services_tb2 import _copy

    calls = []

    class Planned:
        def configure_install_layers(self, **kwargs):
            calls.append(kwargs)

    rollout = Rollout(
        RolloutConfig(
            task_path=_copy(tmp_path),
            agent="claude-agent-acp",
            environment="daytona",
            jobs_dir=tmp_path / "jobs",
        )
    )
    rollout._env = Planned()
    rollout._configure_daytona_layers()
    assert calls and calls[0]["only_if_reused"] is True


# --- L1-L3 ---


@pytest.mark.parametrize(
    ("base_image", "refused"), [(None, True), ("python:3.13-slim", False)]
)
def test_listed_task_refused_early_unless_its_image_is_replaced(
    tmp_path, base_image, refused
):
    """L1: BenchFlow measured the listed TB2 tasks' verifiers root; a
    --base-image replaces that image, so its own USER decides, after setup."""
    from benchflow.sandbox.protocol import KeptServiceFreezerUnavailable
    from benchflow.task import Task, keep_services
    from tests.test_keep_services_tb2 import _copy

    path = _copy(tmp_path)

    def early():
        keep_services.preflight_daytona(
            Task(path),
            path,
            sandbox="daytona",
            agent="claude-agent-acp",
            sandbox_user="agent",
            base_image=base_image,
        )

    if refused:
        with pytest.raises(KeptServiceFreezerUnavailable):
            early()
    else:
        early()


@pytest.mark.parametrize(
    ("verifier_user", "refused"), [("root", True), (0, False), ("0", False)]
)
def test_declared_root_spellings(tmp_path, verifier_user, refused):
    """L1: only the name "root" is sure before the sandbox exists."""
    from benchflow.sandbox.protocol import KeptServiceFreezerUnavailable
    from benchflow.task import Task, keep_services
    from tests.test_dfx_scale import _kept_task

    path = _kept_task(tmp_path, verifier_user=verifier_user)

    def early():
        keep_services.preflight_daytona(
            Task(path),
            path,
            sandbox="daytona",
            agent="claude-agent-acp",
            sandbox_user="agent",
        )

    if refused:
        with pytest.raises(KeptServiceFreezerUnavailable):
            early()
    else:
        early()


@pytest.mark.parametrize(
    ("error", "key", "words"),
    [
        (FREEZER, "kept_service_freezer", "--sandbox docker"),
        (
            "KeptServiceFreezerUnavailable: This trial keeps services "
            "(x-keep-services) for an agent that runs as a non-root user, ...; "
            "rootless. No agent code ran. Use Docker Engine 28 or later with "
            "cgroup v2, not rootless and without user-namespace remapping.",
            "kept_service_freezer:engine",
            "Docker Engine 28",
        ),
        (
            "KeptServiceFreezerUnavailable: This task keeps services "
            "(x-keep-services), which BenchFlow holds in a frozen cgroup v2 of "
            "root's from the kill after the agent's turn until the verifier "
            "starts, and this sandbox cannot (the check's reason is below).",
            "kept_service_freezer:probe",
            "cgroup v2",
        ),
    ],
)
def test_each_freezer_cause_has_its_own_remedy(error, key, words):
    """L2: the Engine and the cgroup probe are not told to leave Daytona."""
    from benchflow.cli.explain import find
    from benchflow.failures import cause_keys, cause_of

    cause = cause_of(
        {
            "task_name": "t",
            "error": error,
            "error_category": "kept_service_freezer",
            "rewards": None,
        }
    )
    assert cause.key == key and key in cause_keys()
    assert words in cause.next_step
    if key != "kept_service_freezer":
        assert "--sandbox docker" not in cause.next_step
        assert "Daytona" not in find(key).body


def test_guard_timeout_remedy_assumes_no_cause():
    """L3: the guard's own error says what it needs; the remedy does not guess."""
    from benchflow.failures import describe

    step = describe("guard_timeout").next_step
    assert "firewall" not in step and "install" not in step
    assert "bench eval resume" in step
