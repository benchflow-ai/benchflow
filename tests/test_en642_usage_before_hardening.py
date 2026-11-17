"""an in-sandbox model proxy's usage log is read at the end of the
agent's turn, before hardening and the verifier, and a log that may miss
calls is never read as complete.

The proxy here is a real local process running BenchFlow's request boundary
(``litellm_request_guard``: the log's count and its SIGUSR1 answer). The host
side is the real ``SandboxLiteLLMProcess.stop()``, its shell commands run on
this machine with the test's user standing in for root
(``tests._root_stand_in``), so these are the commands a Docker or Daytona
sandbox runs, without the sandbox. Every value here is made up.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers.litellm_logging import (
    callback_module_source,
    extract_usage_from_trajectory,
)
from benchflow.providers.runtime import ProviderRuntime
from tests._root_stand_in import run_sh

GUARD_SOURCE = Path(runtime.__file__).with_name("litellm_request_guard.py")
PROXY_REASON = "the in-sandbox model proxy's usage log may miss calls: "

# One provider call as the callback writes it: 11 prompt and 4 completion
# tokens, $0.001.
CALL = {
    "event": "success",
    "request": {
        "method": "POST",
        "path": "/v1/chat/completions",
        "body": {
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "hi"}],
        },
    },
    "response": {
        "model": "openai/MiniMax-M3",
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    },
    "response_cost": 0.001,
    "start_time": "2026-11-14T10:00:00",
    "end_time": "2026-11-14T10:00:01",
    "duration_ms": 1000,
}

# The proxy: BenchFlow's request boundary, which sitecustomize loads first in
# the real proxy, appending records through its count as the callback and the
# boundary do for a request. A boundary without the count (before that change)
# appends them as the callback did, so the trial tests run on both.
DUMMY_PROXY = r"""
import json, os, signal, sys, time
from pathlib import Path

folder = Path(sys.argv[1])
sys.path.insert(0, str(folder))
import benchflow_litellm_request_guard as guard


def count(name, *args):
    step = getattr(guard, name, None)
    if step is not None:
        step(*args)


def append(log, line):
    if hasattr(guard, "append_record"):
        guard.append_record(log, line)
    else:
        with open(log, "ab") as handle:
            handle.write(line)


if os.environ.get("DUMMY_TAKES_SIGNAL") == "1":
    count("_install_tally_signal")
plan = json.loads((folder / "plan.json").read_text())
log = os.environ["BENCHFLOW_LITELLM_LOG_PATH"]
for n, record in enumerate(plan["records"]):
    call = f"call-{n}"
    count("_count_open", 1)
    guard._lifecycle("request_start", request_id=call, request_bytes=1)
    count("await_record", call)
    append(log, (json.dumps(record, separators=(",", ":")) + "\n").encode())
    count("recorded", call)
    guard._lifecycle("request_end", request_id=call, status=200)
    count("_count_open", -1)
for call in plan["awaiting"]:
    count("await_record", call)


def stop(*_):
    (folder.parent / "stopped").write_text("stopped")
    sys.exit(0)


signal.signal(signal.SIGTERM, stop)
(folder.parent / "ready").write_text("ready")
while True:
    time.sleep(0.05)
"""


class _Sandbox:
    """Root's commands run here, as the test's user (run_sh)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    async def exec(self, command, timeout_sec=None, **_kwargs):
        done = run_sh(
            command,
            env={"PATH": "/usr/bin:/bin", "HOME": str(self.root)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        return SimpleNamespace(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )

    async def download_file(self, source_path, target_path) -> None:
        shutil.copyfile(source_path, target_path)


def _wait_for(path: Path, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def start_proxy(tmp_path):
    children: list[subprocess.Popen] = []

    def start(records=(CALL,), *, awaiting=(), takes_signal=True):
        folder = tmp_path / "runtime"
        folder.mkdir(mode=0o700)
        shutil.copyfile(GUARD_SOURCE, folder / "benchflow_litellm_request_guard.py")
        (folder / "dummy_proxy.py").write_text(DUMMY_PROXY)
        (folder / "plan.json").write_text(
            json.dumps({"records": list(records), "awaiting": list(awaiting)})
        )
        log = folder / "callback.jsonl"
        env = {
            **os.environ,
            "BENCHFLOW_LITELLM_LOG_PATH": str(log),
            "DUMMY_TAKES_SIGNAL": "1" if takes_signal else "0",
        }
        umask = os.umask(0o077)
        try:
            child = subprocess.Popen(
                [sys.executable, "-I", str(folder / "dummy_proxy.py"), str(folder)],
                env=env,
                start_new_session=True,
            )
        finally:
            os.umask(umask)
        children.append(child)
        pid_path = folder / "litellm.pid"
        pid_path.write_text(str(child.pid))
        pid_path.chmod(0o600)
        assert _wait_for(tmp_path / "ready"), "the dummy proxy did not start"
        process = runtime.SandboxLiteLLMProcess(
            sandbox=_Sandbox(tmp_path),
            route=None,  # type: ignore[arg-type]
            runtime_dir=str(folder),
            endpoint=runtime.LiteLLMEndpoint(
                "http://127.0.0.1:1", "http://127.0.0.1:1"
            ),
            log_path=str(log),
            pid_path=str(pid_path),
            stdout_path=str(folder / "stdout.log"),
            stderr_path=str(folder / "stderr.log"),
            session_id="trial",
            agent_name="claude-agent-acp",
        )
        return process, child, log

    yield start
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def _kill(child: subprocess.Popen) -> None:
    os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=10)


def _tokens(trajectory) -> tuple[int | None, int | None]:
    usage = extract_usage_from_trajectory(trajectory)
    return usage["n_input_tokens"], usage["n_output_tokens"]


# ---------------------------------------------------------------------------
# The proxy's stop(): the log read back against the proxy's own count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_clean_proxy_is_read_whole_and_still_stopped_gracefully(
    start_proxy, tmp_path
):
    process, _child, _log = start_proxy()
    await process.stop()
    assert process.usage_log_shortfall is None
    assert _tokens(process.trajectory) == (11, 4)
    # The count is asked for before the stop; the stop is still SIGTERM.
    assert (tmp_path / "stopped").exists()
    assert not (tmp_path / "runtime").exists()


@pytest.mark.asyncio
async def test_a_proxy_killed_during_the_agent_s_turn_is_a_shortfall(start_proxy):
    process, child, _log = start_proxy()
    _kill(child)
    await process.stop()
    assert process.usage_log_shortfall == (
        "the proxy was not running at the end of the agent's turn (killed or exited)"
    )
    # What its log holds is still counted, as a floor.
    assert _tokens(process.trajectory) == (11, 4)


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [0, 40])
async def test_a_truncated_log_is_a_shortfall(start_proxy, keep):
    process, _child, log = start_proxy(records=(CALL, CALL))
    whole = log.read_bytes()
    log.write_bytes(whole[:keep])
    await process.stop()
    records = whole.count(b"\n")
    assert process.usage_log_shortfall == (
        f"the log read back holds 0 record(s) ({keep} bytes), not the {records} "
        f"({len(whole)} bytes) the proxy wrote (truncated or rewritten)"
    )


@pytest.mark.asyncio
async def test_a_log_rewritten_to_the_same_size_is_a_shortfall(start_proxy):
    process, _child, log = start_proxy()
    whole = log.read_bytes()
    edited = whole.replace(b'"prompt_tokens":11', b'"prompt_tokens":99')
    assert edited != whole and len(edited) == len(whole)
    log.write_bytes(edited)
    await process.stop()
    assert process.usage_log_shortfall == (
        f"the log read back is not the one the proxy wrote: same size "
        f"({len(whole)} bytes), different content (rewritten)"
    )


@pytest.mark.asyncio
async def test_a_request_with_no_record_yet_is_a_shortfall(start_proxy):
    process, _child, _log = start_proxy(awaiting=("call-late",))
    await process.stop()
    # RUSAGE's L-b: the call cut off is named, not a flush blamed.
    assert process.usage_log_shortfall == (
        "the proxy was stopped with model calls cut off, so its log has no "
        "record of them: 1 call(s) LiteLLM had not logged (call-late)"
    )


def test_the_calls_cut_off_are_named_and_counted_apart():
    """Requests still being answered and calls with no record are counted
    apart; only a well-formed call id is quoted."""
    tally = runtime._ProxyTally(
        "answered",
        report={
            "records": 1,
            "bytes": 10,
            "sha256": "0" * 64,
            "open": 1,
            "awaiting": 2,
            "awaiting_ids": ["call-9", "x; rm -rf /"],
        },
    )
    assert runtime._usage_log_shortfall(tally, None) == (
        "the proxy was stopped with model calls cut off, so its log has no "
        "record of them: 1 request(s) still being answered, 2 call(s) LiteLLM "
        "had not logged (call-9)"
    )


@pytest.mark.asyncio
async def test_a_proxy_that_does_not_take_the_signal_is_never_sent_it(
    start_proxy, tmp_path
):
    """SIGUSR1's default action ends a process: one that does not catch it
    is not sent it, and still gets its graceful SIGTERM."""
    process, _child, _log = start_proxy(takes_signal=False)
    await process.stop()
    assert process.usage_log_shortfall == (
        "the proxy did not report its own count of the log "
        "(it does not take the count signal)"
    )
    assert (tmp_path / "stopped").exists()


@pytest.mark.asyncio
async def test_a_count_left_from_before_is_not_taken_for_an_answer(
    start_proxy, tmp_path
):
    """Only the live proxy can answer: a .tally already in its folder (one
    root wrote while the proxy was down) is removed before the question."""
    process, child, log = start_proxy()
    forged = {
        "pid": child.pid,
        "records": 1,
        "bytes": log.stat().st_size,
        "sha256": "0" * 64,
        "open": 0,
        "awaiting": 0,
    }
    _kill(child)
    Path(f"{log}.tally").write_text(json.dumps(forged))
    await process.stop()
    assert "was not running" in (process.usage_log_shortfall or "")


def test_the_host_reads_the_count_the_proxy_writes():
    """One name for the count's file, on both sides."""
    namespace: dict[str, object] = {}
    exec(compile(GUARD_SOURCE.read_text(), str(GUARD_SOURCE), "exec"), namespace)
    assert namespace["_TALLY_SUFFIX"] == runtime._TALLY_SUFFIX
    assert "import benchflow_litellm_request_guard as _bf_tally" in (
        callback_module_source()
    )
    assert runtime._REQUEST_GUARD_MODULE == "benchflow_litellm_request_guard"


# ---------------------------------------------------------------------------
# The callback's records go through the count
# ---------------------------------------------------------------------------


@pytest.fixture
def callback_with_count(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "benchflow_litellm_request_guard", GUARD_SOURCE
    )
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setitem(sys.modules, "benchflow_litellm_request_guard", guard)
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    monkeypatch.setenv("BENCHFLOW_LITELLM_SERVED_MODELS", json.dumps(["served"]))
    namespace: dict[str, object] = {}
    exec(callback_module_source(), namespace)
    return namespace, guard, log


def _tally(guard, log: Path) -> dict:
    guard._write_tally()
    return json.loads(Path(f"{log}.tally").read_text())


@pytest.mark.asyncio
async def test_the_callback_s_records_are_counted_byte_for_byte(callback_with_count):
    namespace, guard, log = callback_with_count
    logger = namespace["proxy_handler_instance"]
    assert namespace["_bf_tally"] is guard

    await logger.async_pre_call_hook(  # type: ignore[attr-defined]
        None, None, {"model": "served", "litellm_call_id": "c1"}, "acompletion"
    )
    assert _tally(guard, log)["awaiting"] == 1
    logger._write({"event": "success", "usage": {"total_tokens": 3}})  # type: ignore[attr-defined]
    namespace["_note_logged"]({"litellm_call_id": "c1"})  # type: ignore[operator]
    # A refused request has its record: nothing more is due for it.
    with pytest.raises(namespace["ModelNotServed"]):  # type: ignore[arg-type]
        await logger.async_pre_call_hook(  # type: ignore[attr-defined]
            None, None, {"model": "other", "litellm_call_id": "c2"}, "acompletion"
        )
    # A pass-through request: its id is its logging object's, and a failed
    # one gets no failure record, only the post-call failure hook.
    passthrough = {"litellm_logging_obj": SimpleNamespace(litellm_call_id="c3")}
    await logger.async_pre_call_hook(None, None, passthrough, "pass_through_endpoint")  # type: ignore[attr-defined]
    assert _tally(guard, log)["awaiting"] == 1
    await logger.async_post_call_failure_hook(  # type: ignore[attr-defined]
        {"litellm_call_id": "c3"}, RuntimeError("upstream 500"), None
    )

    report = _tally(guard, log)
    data = log.read_bytes()
    assert report["awaiting"] == 0 and report["open"] == 0
    assert (report["records"], report["bytes"]) == (data.count(b"\n"), len(data))
    assert report["sha256"] == runtime._read_back(data).sha256
    assert report["pid"] == os.getpid()
    # The same bytes as before the count: one compact JSON line per record.
    first = json.loads(data.splitlines()[0])
    assert data.splitlines()[0] == json.dumps(first, separators=(",", ":")).encode()


# ---------------------------------------------------------------------------
# The trial: the import before the verifier, and the marker in every record
# ---------------------------------------------------------------------------


def _trial_with(tmp_path, monkeypatch, process):
    from tests.test_en374_salvage import _rollout

    home = tmp_path / "home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._usage_runtime = ProviderRuntime(
        kind="litellm",
        agent_base_url="http://127.0.0.1:1",
        backend_model="openai/MiniMax-M3",
        server=process,
    )
    return rollout, trial


def _agent_result(rollout, trial) -> dict:
    rollout._build_result()
    return json.loads((trial / "result.json").read_text())["agent_result"]


def _root_test_sh(child: subprocess.Popen, log: Path):
    """What test.sh, run as root with code the agent wrote, can do to the
    proxy in the verifier phase: stop it, and rewrite its log to one cheap
    call (or remove it)."""

    async def verifier(self):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        cheap = json.loads(json.dumps(CALL))
        cheap["response"]["usage"] = {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
        cheap["response_cost"] = 0.000001
        if log.parent.exists():
            log.write_text(json.dumps(cheap, separators=(",", ":")) + "\n")
        self._rewards = {"reward": 1.0}
        return self._rewards

    return verifier


@pytest.mark.asyncio
async def test_a_clean_trial_imports_the_same_records(
    start_proxy, tmp_path, monkeypatch
):
    """Passes before and after that change: the clean trial's record is unchanged."""
    process, _child, _log = start_proxy()
    rollout, trial = _trial_with(tmp_path, monkeypatch, process)
    await rollout._cleanup_before_teardown()
    agent = _agent_result(rollout, trial)
    assert (agent["n_input_tokens"], agent["n_output_tokens"]) == (11, 4)
    assert agent["cost_usd"] == pytest.approx(0.001)
    assert agent["usage_source"] == "proxy"
    assert "usage_incomplete" not in agent and "cost_incomplete" not in agent
    calls = (trial / "trajectory" / "llm_trajectory.jsonl").read_text().splitlines()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_verifier_that_stops_the_proxy_or_rewrites_its_log_changes_nothing(
    start_proxy, tmp_path, monkeypatch
):
    """Fails before that change: the log was imported at cleanup, after test.sh,
    and the trial recorded the rewritten call (1 + 1 tokens, $0.000001)."""
    from benchflow.rollout import Rollout

    process, child, log = start_proxy()
    rollout, trial = _trial_with(tmp_path, monkeypatch, process)
    monkeypatch.setattr(Rollout, "_score_finished_solver", _root_test_sh(child, log))

    assert await rollout.verify() == {"reward": 1.0}
    await rollout._cleanup_before_teardown()
    agent = _agent_result(rollout, trial)
    assert (agent["n_input_tokens"], agent["n_output_tokens"]) == (11, 4)
    assert agent["cost_usd"] == pytest.approx(0.001)
    assert "usage_incomplete" not in agent and "cost_incomplete" not in agent


@pytest.mark.asyncio
async def test_a_proxy_killed_during_the_agent_s_turn_is_incomplete_in_every_record(
    start_proxy, tmp_path, monkeypatch
):
    """Fails before that change: the trial's 11 + 4 tokens and $0.001 read as
    its whole usage, with nothing saying the proxy had been killed."""
    from rich.console import Console
    from typer.testing import CliRunner

    import benchflow as bf
    import benchflow.cli._shared as shared
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.cli.main import app
    from benchflow.evaluation import RUN_CARD_USAGE_FIELDS
    from benchflow.models import RolloutResult

    process, child, _log = start_proxy()
    rollout, trial = _trial_with(tmp_path, monkeypatch, process)
    _kill(child)
    await rollout._cleanup_before_teardown()
    data_agent = _agent_result(rollout, trial)
    reason = (
        PROXY_REASON + "the proxy was not running at the end of the agent's turn "
        "(killed or exited)"
    )

    # result.json: the counted part, said to be partial, tokens and cost.
    assert (data_agent["n_input_tokens"], data_agent["n_output_tokens"]) == (11, 4)
    assert data_agent["usage_incomplete"] is True
    assert data_agent["usage_incomplete_reason"] == reason
    assert data_agent["cost_incomplete"] is True
    data = json.loads((trial / "result.json").read_text())
    extra = data["final_metrics"]["extra"]
    assert (extra["usage_incomplete"], extra["cost_incomplete"]) == (True, True)
    assert extra["usage_incomplete_reason"] == reason
    loaded = RolloutResult.load(trial)
    assert (loaded.usage_incomplete, loaded.usage_incomplete_reason) == (True, reason)

    # summary.json and the run card.
    summary = usage_summary({trial.name: data})
    assert (summary["trials_usage_incomplete"], summary["trials_cost_incomplete"]) == (
        1,
        1,
    )
    assert {"trials_usage_incomplete", "trials_cost_incomplete"} <= set(
        RUN_CARD_USAGE_FIELDS
    )

    # The trial export and `bench results`.
    usage = bf.load_trial(trial).to_json_dict()["usage"]
    assert (usage["usage_incomplete"], usage["cost_incomplete"]) == (True, True)
    assert usage["usage_incomplete_reason"] == reason
    shown = CliRunner().invoke(
        app, ["results", str(trial.parent)], env={"COLUMNS": "400"}
    )
    assert shown.exit_code == 0, shown.output
    table = " ".join(re.sub("[─-╿]", " ", shown.stdout).split())
    assert "1 trial(s) counted only some of their calls" in table
    assert "the proxy was not running at the end of the agent's turn" in table

    # The closing Cost: line of `bench eval run`.
    console = Console(file=io.StringIO(), width=240)
    monkeypatch.setattr(shared, "console", console)
    shared._report_outcomes(SimpleNamespace(results={"t": loaded}), trial.parent)
    plain = " ".join(re.sub("[─-╿]", " ", console.file.getvalue()).split())
    assert (
        "1 trial(s) counted only some of their calls, so the total is a floor" in plain
    )


@pytest.mark.asyncio
async def test_a_truncated_log_is_incomplete_even_when_nothing_was_counted(
    start_proxy, tmp_path, monkeypatch
):
    """Fails before that change: an emptied log read as a trial that counted
    nothing, with no reason."""
    process, _child, log = start_proxy()
    rollout, trial = _trial_with(tmp_path, monkeypatch, process)
    log.write_bytes(b"")
    await rollout._cleanup_before_teardown()
    agent = _agent_result(rollout, trial)
    assert agent["n_input_tokens"] is None and agent["cost_usd"] is None
    assert agent["usage_incomplete"] is True
    assert agent["usage_incomplete_reason"].startswith(PROXY_REASON)
    assert "truncated or rewritten" in agent["usage_incomplete_reason"]
    # An unknown cost is unknown, not a floor.
    assert "cost_incomplete" not in agent


class _StandInServer:
    def __init__(self, *, in_sandbox: bool, fail: bool = False) -> None:
        self.runs_in_sandbox = in_sandbox
        self.fail = fail
        self.trajectory = None
        self.usage_log_shortfall: str | None = None


@pytest.mark.asyncio
@pytest.mark.parametrize("in_sandbox", [True, False])
async def test_only_an_in_sandbox_proxy_stops_before_the_verifier(
    tmp_path, monkeypatch, in_sandbox
):
    from benchflow.rollout import Rollout

    rollout, _trial = _trial_with(
        tmp_path, monkeypatch, _StandInServer(in_sandbox=in_sandbox)
    )
    order: list[str] = []

    async def stop(runtime_, **_kwargs):
        order.append("proxy stop")
        runtime_.stopped = True

    async def verifier(self):
        order.append("verifier")
        return None

    monkeypatch.setattr(rollout._planes, "stop_provider_runtime", stop)
    monkeypatch.setattr(Rollout, "_score_finished_solver", verifier)
    await rollout.verify()
    assert order == (["proxy stop", "verifier"] if in_sandbox else ["verifier"])


@pytest.mark.asyncio
async def test_a_proxy_that_cannot_be_read_before_the_verifier_is_incomplete(
    tmp_path, monkeypatch
):
    from benchflow.rollout import Rollout

    server = _StandInServer(in_sandbox=True)
    rollout, _trial = _trial_with(tmp_path, monkeypatch, server)

    async def stop(_runtime, **_kwargs):
        raise RuntimeError("exec channel lost")

    async def verifier(self):
        return None

    monkeypatch.setattr(rollout._planes, "stop_provider_runtime", stop)
    monkeypatch.setattr(Rollout, "_score_finished_solver", verifier)
    await rollout.verify()
    assert server.usage_log_shortfall == (
        "it could not be stopped and read before the verifier ran "
        "(RuntimeError: exec channel lost)"
    )
    assert rollout._usage_runtime.stopped is False  # cleanup tries again


@pytest.mark.asyncio
async def test_an_earlier_role_s_proxy_shortfall_is_kept(tmp_path, monkeypatch):
    """A scene's earlier role ran through a proxy that was killed: its
    retired runtime's shortfall marks the trial, whatever the last role's."""
    from benchflow.rollout._cost import mark_incomplete_usage

    earlier = _StandInServer(in_sandbox=True)
    earlier.usage_log_shortfall = "the proxy was not running"
    rollout, _trial = _trial_with(
        tmp_path, monkeypatch, _StandInServer(in_sandbox=True)
    )
    rollout._usage_runtime.stopped = True
    retired = ProviderRuntime(kind="litellm", agent_base_url="x", server=earlier)  # type: ignore[arg-type]
    retired.stopped = True
    rollout._retired_usage_runtimes = [retired]

    await rollout._teardown_usage_runtime()
    rollout._usage_metrics = {"usage_source": "provider_response", "total_tokens": 9}
    mark_incomplete_usage(rollout)
    assert rollout._usage_metrics["usage_incomplete"] is True
    assert rollout._usage_metrics["usage_incomplete_reason"] == (
        PROXY_REASON + "the proxy was not running"
    )


# ---------------------------------------------------------------------------
# With a log that cannot be read at the stop before the verifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_lost_sandbox_at_the_stop_before_the_verifier_keeps_what_was_read(
    tmp_path, monkeypatch
):
    """Nothing can read the log later: the stop before the verifier keeps
    what the live capture read, with the reason."""
    from tests.test_en649_daytona_loss_usage import (
        CELLS,
        _DaytonaSandbox,
        _no_wait,
        _proxy,
        _record,
    )

    monkeypatch.setattr(runtime, "_LIVE_CAPTURE_INTERVAL_SEC", 0.02)
    monkeypatch.setattr(runtime, "_await_log_stable", _no_wait)
    sandbox = _DaytonaSandbox(_record(*CELLS["pi-acp"][0]).encode() + b"\n")
    proxy = _proxy(sandbox, "pi-acp")
    proxy.start_live_capture(tmp_path / "llm_trajectory.jsonl")
    await proxy.settle_live_capture(timeout=5)
    sandbox.lost = True
    await proxy.stop(retry_later=True)
    assert len(proxy.trajectory.exchanges) == 1
    assert (proxy.usage_log_shortfall or "").startswith(
        "the sandbox was lost before the log could be read at the end"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", [True, False])
async def test_a_log_read_after_test_sh_is_checked_against_the_live_capture(
    tmp_path, monkeypatch, rewrite
):
    """RUSAGE's L-a: the stop before the verifier could not read the log on
    a sandbox that answers, so cleanup reads it after test.sh. The log must
    begin with the bytes the live capture read while the turn ran; test.sh
    rewrote them, and the reason says so. Fails before: the reason did not
    say the log was rewritten."""
    from types import SimpleNamespace

    from benchflow.rollout import Rollout
    from tests.test_en649_daytona_loss_usage import (
        CELLS,
        _AnsweringSandbox,
        _no_wait,
        _proxy,
        _record,
    )

    class Rereads(_AnsweringSandbox):
        """Its log reads (cat) fail only while ``lost``."""

        async def exec(self, command: str, timeout_sec=None, **_kwargs):
            if not self.lost and command.startswith("cat ") and "callback" in command:
                return SimpleNamespace(
                    return_code=0, stdout=self.log.decode(), stderr=""
                )
            return await super().exec(command, timeout_sec=timeout_sec)

    monkeypatch.setattr(runtime, "_LIVE_CAPTURE_INTERVAL_SEC", 0.02)
    monkeypatch.setattr(runtime, "_await_log_stable", _no_wait)
    log = _record(*CELLS["pi-acp"][0]).encode() + b"\n"
    sandbox = Rereads(log)
    proxy = _proxy(sandbox, "pi-acp")
    proxy.start_live_capture(tmp_path / "llm_trajectory.jsonl")
    await proxy.settle_live_capture(timeout=5)
    sandbox.lost = True  # the read before the verifier fails
    rollout, trial = _trial_with(tmp_path, monkeypatch, proxy)
    cheap = re.sub(rb'("prompt_tokens":\s*)\d+', rb"\g<1>1", log)
    assert cheap != log

    async def verifier(self):
        # test.sh, as root: one prompt token a call.
        if rewrite:
            sandbox.log = cheap
        sandbox.lost = False
        return None

    monkeypatch.setattr(Rollout, "_score_finished_solver", verifier)
    await rollout.verify()
    await rollout._cleanup_before_teardown()
    agent = _agent_result(rollout, trial)
    first = (
        "it could not be stopped and read before the verifier ran (RuntimeError: "
        "the callback log could not be read (exit 255))"
    )
    assert agent["usage_incomplete"] is True
    assert agent["usage_incomplete_reason"] == PROXY_REASON + first + (
        f"; the log read back does not begin with the {len(log)} bytes the live "
        "capture read from it during the agent's turn (rewritten or truncated "
        "after them): only the 1 call(s) the live capture read during the agent's "
        "turn are counted"
        if rewrite
        else ""
    )
    # What the live capture read, not test.sh's rewrite (1 prompt token).
    assert agent["total_tokens"] == 1396 + 84


def _counting_sandbox(log: bytes):
    """A sandbox whose proxy answers the count of what it wrote (``written``,
    with ``open_requests`` requests still being answered) until the stop's
    SIGTERM, when it writes ``at_term`` (an in-flight call's record). The log
    read (cat) fails only while ``lost``, the live capture's reads while
    ``frozen``."""
    import hashlib
    from types import SimpleNamespace

    from tests.test_en649_daytona_loss_usage import _AnsweringSandbox

    def done(code: int, out: str = ""):
        return SimpleNamespace(return_code=code, stdout=out, stderr="")

    class Counting(_AnsweringSandbox):
        def __init__(self, data: bytes) -> None:
            super().__init__(data)
            self.written = data
            self.answers = True
            self.open_requests = 0
            self.at_term = b""
            self.terminated = False
            self.frozen = False

        async def exec(self, command: str, timeout_sec=None, **_kwargs):
            if "BF_TALLY_PID" in command:
                if self.terminated:
                    return done(0, "BF_TALLY_DEAD\n")
                if not self.answers:
                    return done(0, "BF_TALLY_NOHANDLER\n")
                report = {
                    "pid": 4242,
                    "records": self.written.count(b"\n"),
                    "bytes": len(self.written),
                    "sha256": hashlib.sha256(self.written).hexdigest(),
                    "open": self.open_requests,
                    "awaiting": 0,
                }
                return done(0, f"BF_TALLY_PID 4242\n{json.dumps(report)}\n")
            if "kill -TERM" in command:
                if not self.terminated:
                    self.terminated = True
                    self.written += self.at_term
                    self.log += self.at_term
                return done(0)
            if self.frozen and "dd if=" in command:
                return done(1)
            if not self.lost and command.startswith("cat ") and "callback" in command:
                return done(0, self.log.decode())
            return await super().exec(command, timeout_sec=timeout_sec)

        # The live capture reads through exec_transient.
        exec_transient = exec

    return Counting(log)


def _call(prompt: int = 1396) -> bytes:
    """One call's record: ``prompt`` + 84 tokens."""
    from tests.test_en649_daytona_loss_usage import _record

    return (
        _record("/v1/chat/completions", prompt, 84, 1280, 4.3344e-05).encode() + b"\n"
    )


ONE_CALL = 1396 + 84
_RETRY = (
    "it could not be stopped and read before the verifier ran (RuntimeError: "
    "the callback log could not be read (exit 255))"
)


async def _retried(tmp_path, monkeypatch, sandbox, test_sh, after_live=None) -> dict:
    rollout, trial = await _retried_trial(
        tmp_path, monkeypatch, sandbox, test_sh, after_live
    )
    return _agent_result(rollout, trial)


async def _retried_trial(tmp_path, monkeypatch, sandbox, test_sh, after_live=None):
    """The stop before the verifier takes the count and cannot read the log on
    a sandbox that answers; ``test_sh`` runs as root; cleanup reads the log
    again. ``after_live`` runs once the live capture has read the log."""
    from benchflow.rollout import Rollout
    from tests.test_en649_daytona_loss_usage import _no_wait, _proxy

    monkeypatch.setattr(runtime, "_LIVE_CAPTURE_INTERVAL_SEC", 0.02)
    monkeypatch.setattr(runtime, "_await_log_stable", _no_wait)
    proxy = _proxy(sandbox, "pi-acp")
    proxy.start_live_capture(tmp_path / "llm_trajectory.jsonl")
    await proxy.settle_live_capture(timeout=5)
    if after_live is not None:
        after_live()
    rollout, trial = _trial_with(tmp_path, monkeypatch, proxy)

    async def verifier(self):
        test_sh()
        sandbox.lost = False
        return None

    monkeypatch.setattr(Rollout, "_score_finished_solver", verifier)
    sandbox.lost = True
    await rollout.verify()
    await rollout._cleanup_before_teardown()
    return rollout, trial


def _counted(records: int, size: int) -> str:
    return (
        f"the {records} record(s) ({size} bytes) the proxy had written when the "
        "stop before the verifier asked it"
    )


@pytest.mark.asyncio
async def test_a_call_appended_after_the_count_is_neither_counted_nor_missed(
    tmp_path, monkeypatch
):
    """RUSAGE's (d) probe: root appends one made-up call of 900,000 prompt
    tokens before cleanup reads the log again. Only the bytes the kept count
    vouches for are counted. Fails before: flagged, but counted (901,564)."""
    log = _call()
    sandbox = _counting_sandbox(log)

    def test_sh():
        sandbox.log += _call(900000)

    agent = await _retried(tmp_path, monkeypatch, sandbox, test_sh)
    assert agent["total_tokens"] == ONE_CALL
    assert agent["usage_incomplete_reason"] == PROXY_REASON + _RETRY + (
        "; the log read back at cleanup holds more call records after "
        f"{_counted(1, len(log))} than the 0 call(s) it then had in flight; none "
        "after it is counted (written after the count, or by test.sh)"
    )


@pytest.mark.asyncio
async def test_a_call_appended_beside_one_in_flight_is_neither_counted_nor_missed(
    tmp_path, monkeypatch
):
    """RUSAGE's L1 probe: a call was in flight at the count and its record
    lands at the SIGTERM; root appends a made-up call too. Two call records
    for one call in flight: none after the counted bytes is counted. Fails
    before: neither flagged nor cut (903,044 tokens)."""
    log = _call()
    sandbox = _counting_sandbox(log)
    sandbox.open_requests = 1
    sandbox.at_term = _call()

    def test_sh():
        sandbox.log += _call(900000)

    agent = await _retried(tmp_path, monkeypatch, sandbox, test_sh)
    assert agent["total_tokens"] == ONE_CALL
    assert agent["usage_incomplete_reason"] == PROXY_REASON + _RETRY + _IN_FLIGHT + (
        "; the log read back at cleanup holds more call records after "
        f"{_counted(1, len(log))} than the 1 call(s) it then had in flight; none "
        "after it is counted (written after the count, or by test.sh)"
    )


# The call in flight at the count, named in L-b's cut-call words.
_IN_FLIGHT = (
    "; the proxy was stopped with model calls cut off, so its log has no record "
    "of them: 1 request(s) still being answered"
)


def _slot_reason(log: bytes, records: int = 1) -> str:
    return _IN_FLIGHT + (
        f"; the log read back at cleanup holds {records} call record(s) after "
        f"{_counted(1, len(log))}; they are not counted (written at the stop, or "
        "by test.sh)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("lands", [True, False])
async def test_a_call_in_flight_at_the_count_is_named_not_counted(
    tmp_path, monkeypatch, lands
):
    """The call in flight at the count writes its record at the SIGTERM,
    after the agent's turn, where test.sh's cannot be told from it: it is not
    counted, and the reason says so. When it writes none, the reason names
    the call cut off."""
    log = _call()
    sandbox = _counting_sandbox(log)
    sandbox.open_requests = 1
    sandbox.at_term = _call() if lands else b""
    agent = await _retried(tmp_path, monkeypatch, sandbox, lambda: None)
    assert agent["total_tokens"] == ONE_CALL
    assert agent["usage_incomplete_reason"] == PROXY_REASON + _RETRY + (
        _slot_reason(log) if lands else _IN_FLIGHT
    )
    assert (agent["usage_incomplete"], agent["cost_incomplete"]) == (True, True)


@pytest.mark.asyncio
async def test_a_call_in_flight_at_the_stop_is_a_floor_in_every_record(
    tmp_path, monkeypatch
):
    """F2b's condition for naming, not counting, a call in flight at the stop
    whose record lands after the count: the trial is a floor, usage and cost,
    in result.json, summary.json, report.md, the export, `bench results` and
    the closing Cost: line, and the reason names the call in the cut-call
    words. Fails on 72d64c55: the call was counted (2,960), unmarked beyond
    the early-stop reason."""
    from rich.console import Console
    from typer.testing import CliRunner

    import benchflow as bf
    import benchflow.cli._shared as shared
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.cli.main import app
    from benchflow.models import RolloutResult
    from benchflow.report import build_report

    log = _call()
    sandbox = _counting_sandbox(log)
    sandbox.open_requests = 1
    sandbox.at_term = _call()
    rollout, trial = await _retried_trial(tmp_path, monkeypatch, sandbox, lambda: None)
    agent = _agent_result(rollout, trial)
    reason = PROXY_REASON + _RETRY + _slot_reason(log)
    assert agent["total_tokens"] == ONE_CALL
    assert (agent["usage_incomplete"], agent["cost_incomplete"]) == (True, True)
    assert agent["usage_incomplete_reason"] == reason
    data = json.loads((trial / "result.json").read_text())
    extra = data["final_metrics"]["extra"]
    assert (extra["usage_incomplete"], extra["cost_incomplete"]) == (True, True)
    loaded = RolloutResult.load(trial)
    assert (loaded.usage_incomplete, loaded.usage_incomplete_reason) == (True, reason)
    summary = usage_summary({trial.name: data})
    assert (summary["trials_usage_incomplete"], summary["trials_cost_incomplete"]) == (
        1,
        1,
    )
    report = " ".join(build_report([trial.parent], "t").split())
    assert "1 trial(s) counted only some of their calls, so the total is a floor" in (
        report
    )
    usage = bf.load_trial(trial).to_json_dict()["usage"]
    assert (usage["usage_incomplete"], usage["cost_incomplete"]) == (True, True)
    shown = CliRunner().invoke(
        app, ["results", str(trial.parent)], env={"COLUMNS": "400"}
    )
    assert shown.exit_code == 0, shown.output
    table = " ".join(re.sub("[─-╿]", " ", shown.stdout).split())
    assert "1 trial(s) counted only some of their calls" in table
    assert "model calls cut off" in table
    console = Console(file=io.StringIO(), width=240)
    monkeypatch.setattr(shared, "console", console)
    shared._report_outcomes(SimpleNamespace(results={"t": loaded}), trial.parent)
    plain = " ".join(re.sub("[─-╿]", " ", console.file.getvalue()).split())
    assert (
        "1 trial(s) counted only some of their calls, so the total is a floor" in plain
    )


@pytest.mark.asyncio
async def test_a_made_up_call_in_an_empty_in_flight_slot_is_not_counted(
    tmp_path, monkeypatch
):
    """RUSAGE's (b): a call was in flight at the count and left no record;
    root writes a made-up 900,000-token call into its slot. Fails on
    6d71fc1e: counted, unflagged (901,564)."""
    log = _call()
    sandbox = _counting_sandbox(log)
    sandbox.open_requests = 1

    def test_sh():
        sandbox.log += _call(900000)

    agent = await _retried(tmp_path, monkeypatch, sandbox, test_sh)
    assert agent["total_tokens"] == ONE_CALL
    assert agent["usage_incomplete_reason"] == PROXY_REASON + _RETRY + _slot_reason(log)
    assert (agent["usage_incomplete"], agent["cost_incomplete"]) == (True, True)


@pytest.mark.asyncio
async def test_a_call_appended_with_no_count_kept_is_not_counted(tmp_path, monkeypatch):
    """The proxy did not answer the count (it does not take the signal), the
    stop before the verifier could not read the log, and root appends a
    made-up 900,000-token call. With no count to check the log by, only what
    the live capture read until the end of the turn is counted. Fails on
    6d71fc1e: the append was counted, unflagged (901,564)."""
    log = _call()
    sandbox = _counting_sandbox(log)
    sandbox.answers = False
    fake = _call(900000)

    def test_sh():
        sandbox.log += fake

    agent = await _retried(tmp_path, monkeypatch, sandbox, test_sh)
    assert agent["total_tokens"] == ONE_CALL
    assert agent["usage_incomplete_reason"] == PROXY_REASON + _RETRY + (
        f"; the log read back at cleanup holds {len(fake)} byte(s) after the "
        f"{len(log)} the live capture read during the agent's turn; they are not "
        "counted (written after the turn, or by test.sh)"
    )


@pytest.mark.asyncio
async def test_cleanup_reads_nothing_more_into_the_live_capture(tmp_path, monkeypatch):
    """RUSAGE's (a): the proxy's second record lands after the live capture's
    last read of the turn; test.sh rewrites it into a 900,000-token call, and
    the live capture could read again at cleanup. The counted bytes are not
    intact, so the live capture's records are counted: only those read by the
    end of the turn. Fails on 6d71fc1e: cleanup drained test.sh's bytes into
    the live capture (901,564, and "2 call(s) ... during the agent's turn")."""
    first, second = _call(), _call()
    sandbox = _counting_sandbox(first)

    def after_live():
        sandbox.frozen = True
        sandbox.written = sandbox.log = first + second

    def test_sh():
        sandbox.log = first + _call(900000)
        sandbox.frozen = False

    agent = await _retried(tmp_path, monkeypatch, sandbox, test_sh, after_live)
    assert agent["total_tokens"] == ONE_CALL
    assert agent["usage_incomplete_reason"] == PROXY_REASON + _RETRY + (
        f"; the log read back at cleanup is not {_counted(2, len(first + second))} "
        "(rewritten after the live read): only the 1 call(s) the live capture "
        "read during the agent's turn are counted"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", [True, False])
async def test_a_change_in_the_counted_tail_falls_back_to_the_live_capture(
    tmp_path, monkeypatch, rewrite
):
    """RUSAGE's tail probe: the proxy wrote a second record after the live
    capture's last read, and test.sh cheapens it. The counted bytes are not
    intact, so none of the log is counted, only what the live capture read.
    Fails before: flagged, but the cheapened record was counted. Untouched,
    both records count, unflagged."""
    first, second = _call(), _call()
    sandbox = _counting_sandbox(first)
    cheap = _call(1)

    def after_live():
        sandbox.frozen = True
        sandbox.written = sandbox.log = first + second

    def test_sh():
        if rewrite:
            sandbox.log = first + cheap

    agent = await _retried(tmp_path, monkeypatch, sandbox, test_sh, after_live)
    assert agent["total_tokens"] == ONE_CALL * (1 if rewrite else 2)
    assert agent["usage_incomplete_reason"] == PROXY_REASON + _RETRY + (
        f"; the log read back at cleanup is not {_counted(2, len(first + second))} "
        "(rewritten after the live read): only the 1 call(s) the live capture "
        "read during the agent's turn are counted"
        if rewrite
        else ""
    )


@pytest.mark.asyncio
async def test_a_line_that_is_not_the_proxy_s_after_the_count_is_cut(
    tmp_path, monkeypatch
):
    log = _call()
    sandbox = _counting_sandbox(log)

    def test_sh():
        sandbox.log += b"made up\n"

    agent = await _retried(tmp_path, monkeypatch, sandbox, test_sh)
    assert agent["total_tokens"] == ONE_CALL
    assert agent["usage_incomplete_reason"] == PROXY_REASON + _RETRY + (
        f"; the log read back at cleanup holds lines after {_counted(1, len(log))} "
        "that are not the proxy's records; they are not counted (rewritten after "
        "the live read)"
    )


@pytest.mark.asyncio
async def test_an_unreadable_log_on_a_sandbox_that_answers_is_read_again_at_cleanup(
    tmp_path, monkeypatch
):
    """F2b's ruling for #243 with #249: the stop before the verifier does not
    settle for what the live capture read while the sandbox still answers.
    It raises, the trial is marked, and cleanup reads the log again; when that
    read fails too, what the live capture read is counted, with the reason."""
    from benchflow.rollout import Rollout
    from tests.test_en649_daytona_loss_usage import (
        CELLS,
        _AnsweringSandbox,
        _no_wait,
        _proxy,
        _record,
    )

    monkeypatch.setattr(runtime, "_LIVE_CAPTURE_INTERVAL_SEC", 0.02)
    monkeypatch.setattr(runtime, "_await_log_stable", _no_wait)
    sandbox = _AnsweringSandbox(_record(*CELLS["pi-acp"][0]).encode() + b"\n")
    proxy = _proxy(sandbox, "pi-acp")
    proxy.start_live_capture(tmp_path / "llm_trajectory.jsonl")
    await proxy.settle_live_capture(timeout=5)
    sandbox.lost = True  # only the log read fails now
    rollout, trial = _trial_with(tmp_path, monkeypatch, proxy)

    async def verifier(self):
        return None

    monkeypatch.setattr(Rollout, "_score_finished_solver", verifier)
    await rollout.verify()
    assert rollout._usage_runtime.stopped is False
    assert proxy.usage_log_shortfall == (
        "it could not be stopped and read before the verifier ran "
        "(RuntimeError: the callback log could not be read (exit 255))"
    )

    await rollout._cleanup_before_teardown()
    agent = _agent_result(rollout, trial)
    assert agent["total_tokens"] == 1480
    assert agent["usage_incomplete"] is True
    assert agent["usage_incomplete_reason"] == (
        PROXY_REASON + "the log could not be read at the end, though the sandbox "
        "answers (RuntimeError: the callback log could not be read (exit 255)): the "
        "1 call(s) read from it while the turn ran are counted, any after the last "
        "read are not"
    )
