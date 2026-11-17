"""A native Claude Code turn cut off at its time cap, and a lost sandbox
(EN-374, 0.9.3).

X2's run of 2026-11-13 on 0.9.2 (``bench run tasks/timeout -a
claude-agent-acp -m claude-sonnet-5-5 --harness native``, a task that echoes
``started`` and then sleeps 90 s against a 15 s cap) recorded 0 tokens with
null cost against 27,017 in Claude Code's session log. The CLI then said
"none: no model tokens were used". On rc3 a lost sandbox removed both
measurements: the session log is in the sandbox, and the turn never sent its
report.

Here the real native client runs a stand-in CLI
(``fixtures/native_harness/replay_cli.py``). The stand-in replays two
responses in Claude Code 2.1.280's stream shape, built from the recorded
``cancelled.jsonl`` sample, and then hangs in its ``sleep 90``. The
kernel's wall clock cuts the turn. The rollout's teardown then has to count
what the session log holds, from the stream log that BenchFlow writes on
the host as each event arrives. With the sandbox gone, no copy-out is
possible.
"""

from __future__ import annotations

import asyncio
import copy
import json
import shutil
from pathlib import Path

import pytest

from benchflow.acp.runtime import execute_prompts
from benchflow.diagnostics import AgentPromptTimeoutError
from benchflow.native_harness.client import NativeCLIClient
from benchflow.native_harness.harnesses import CLAUDE_CODE
from tests.test_en374_salvage import _rollout
from tests.test_native_harness_client import FIXTURES, _harness, _LocalSandbox

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/environ").exists(), reason="needs Linux /proc"
)

RECORDED = FIXTURES / "claude-code-2.1.280" / "cancelled.jsonl"
SONNET = "claude-sonnet-5-5"
# Sonnet 5.5 at the 2026-11-04 snapshot: input, output, cache read, 5m write.
SONNET_RATES = (2e-6, 10e-6, 0.1e-6, 2.5e-6)
# Each response: (uncached input, cache read, cache write, output). Their
# sum is X2's 0.9.2 measurement, 27,017.
ECHO = ("msg_echo", "toolu_echo", "echo started", (3, 0, 13412, 58))
SLEEP = ("msg_sleep", "toolu_sleep", "sleep 90", (1, 13412, 79, 52))
RAW_TOTAL = 27_017
EXPECTED_USD = (
    (3 + 1) * SONNET_RATES[0]
    + (58 + 52) * SONNET_RATES[1]
    + 13412 * SONNET_RATES[2]
    + (13412 + 79) * SONNET_RATES[3]
)


def _usage(counts: tuple[int, int, int, int], output: int | None = None) -> dict:
    fresh, read, write, out = counts
    return {
        "input_tokens": fresh,
        "cache_creation_input_tokens": write,
        "cache_read_input_tokens": read,
        "output_tokens": out if output is None else output,
        "cache_creation": {
            "ephemeral_5m_input_tokens": write,
            "ephemeral_1h_input_tokens": 0,
        },
    }


def _recorded() -> list[dict]:
    return [json.loads(line) for line in RECORDED.read_text().splitlines()]


def _response(lines: list[dict], spec: tuple) -> list[dict]:
    """One model response as the recorded sample streams it, from
    message_start to message_stop (lines 2-12), with this response's id,
    tool call and usage. The partial ``assistant`` events carry the stale
    output count of 1 that Claude Code streams. Only the final
    ``message_delta`` has the real output count."""
    message_id, tool_id, command, counts = spec
    events = copy.deepcopy(lines[2:13])
    start = events[0]["event"]["message"]
    start.update(id=message_id, model=SONNET, usage=_usage(counts, output=1))
    for event in events:
        if event.get("type") == "assistant":
            message = event["message"]
            message.update(id=message_id, model=SONNET, usage=_usage(counts, output=1))
            for block in message["content"]:
                if block.get("type") == "tool_use":
                    block.update(id=tool_id, input={"command": command})
        inner = event.get("event") or {}
        if inner.get("type") == "content_block_start":
            block = inner["content_block"]
            if block.get("type") == "tool_use":
                block["id"] = tool_id
        if inner.get("type") == "content_block_delta":
            delta = inner["delta"]
            if delta.get("type") == "input_json_delta":
                delta["partial_json"] = json.dumps({"command": command})
        if inner.get("type") == "message_delta":
            inner["usage"] = {"output_tokens": counts[3]}
    return events


def _sample(tmp_path: Path) -> Path:
    """The stream up to the cut: init, the echo response, its result, the
    sleep response, and the tool running. The stand-in then hangs."""
    lines = _recorded()
    init, status = copy.deepcopy(lines[0]), copy.deepcopy(lines[1])
    init["model"] = SONNET
    echoed = copy.deepcopy(lines[16])
    echoed["message"]["content"] = [
        {"type": "tool_result", "content": "started", "tool_use_id": ECHO[1]}
    ]
    started = copy.deepcopy(lines[13])
    started["tool_use_id"] = SLEEP[1]
    progress = copy.deepcopy(lines[14])
    progress.update(tool_use_id=f"{SLEEP[1]}-heartbeat-0", parent_tool_use_id=SLEEP[1])
    events = [
        init,
        status,
        *_response(lines, ECHO),
        echoed,
        *_response(lines, SLEEP),
        started,
        progress,
    ]
    path = tmp_path / "timeout-sample.jsonl"
    path.write_text(
        "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events)
    )
    return path


def _session_log(home: Path) -> None:
    """Claude Code's session log in the sandbox: one assistant line per
    content block, each carrying the response's usage, and the tool result
    between the two responses."""
    path = home / ".claude" / "projects" / "-work" / "session-1.jsonl"
    path.parent.mkdir(parents=True)
    rows: list[dict] = []
    for index, (message_id, tool_id, command, counts) in enumerate((ECHO, SLEEP)):
        for block in (
            {"type": "text", "text": "Running it."},
            {
                "type": "tool_use",
                "id": tool_id,
                "name": "Bash",
                "input": {"command": command},
            },
        ):
            rows.append(
                {
                    "type": "assistant",
                    "requestId": f"req_{index}",
                    "message": {
                        "id": message_id,
                        "model": SONNET,
                        "role": "assistant",
                        "content": [block],
                        "usage": _usage(counts),
                    },
                }
            )
        if index == 0:
            rows.append(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "started",
                                "tool_use_id": tool_id,
                            }
                        ],
                    },
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _raw_session_total(home: Path) -> int:
    """X2's measurement: every response in the session log, once per
    message id, input + cache read + cache write + output."""
    per_message: dict[str, int] = {}
    for path in (home / ".claude" / "projects").rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("type") != "assistant":
                continue
            usage = row["message"]["usage"]
            per_message[row["message"]["id"]] = (
                usage["input_tokens"]
                + usage["cache_read_input_tokens"]
                + usage["cache_creation_input_tokens"]
                + usage["output_tokens"]
            )
    return sum(per_message.values())


def _client(tmp_path: Path, sandbox: _LocalSandbox) -> NativeCLIClient:
    harness = _harness(CLAUDE_CODE, tmp_path)
    return NativeCLIClient(
        env=sandbox,
        harness=harness,
        agent=harness.agent,
        launch_env={"REPLAY_SAMPLE": str(_sample(tmp_path)), "REPLAY_HANG": "1"},
        sandbox_user=None,
        cwd=str(tmp_path),
        rollout_dir=tmp_path / "trial",
    )


def _host_stream(tmp_path: Path) -> list[dict]:
    path = tmp_path / "trial" / "agent" / "claude-code.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _assert_counted(usage: dict, raw: int) -> None:
    assert usage["total_tokens"] == raw == RAW_TOTAL
    assert usage["n_input_tokens"] == 3 + 1
    assert usage["n_output_tokens"] == 58 + 52
    assert usage["n_cache_read_tokens"] == 13412
    assert usage["n_cache_creation_tokens"] == 13412 + 79
    assert usage["n_cache_write_5m_tokens"] == 13412 + 79
    assert usage["n_cache_write_1h_tokens"] == 0
    assert usage["usage_source"] == "stream_log"
    assert usage["cost_usd"] == pytest.approx(EXPECTED_USD)
    assert usage["cost_source"] == "list_price_estimate"


class _GoneSandbox:
    """A sandbox deleted under the run: every call fails, as Daytona's
    ``not found: sandbox ... may have been deleted`` does."""

    def _gone(self, *_args, **_kwargs):
        raise ConnectionError("not found: sandbox may have been deleted or stopped")

    async def exec(self, *args, **kwargs):
        self._gone()

    async def download_file(self, *args, **kwargs):
        self._gone()

    async def download_dir(self, *args, **kwargs):
        self._gone()

    async def stop(self, *, delete: bool = True) -> None:
        pass


@pytest.mark.asyncio
async def test_a_native_turn_cut_at_its_cap_counts_what_the_session_log_holds(
    tmp_path, monkeypatch
):
    import benchflow.cli._shared as shared
    from benchflow.evaluation import EvaluationConfig, EvaluationResult
    from tests.test_cli_outcome_summary import _render, _trial

    home = tmp_path / "sandbox-home"
    _session_log(home)
    raw = _raw_session_total(home)
    sandbox = _LocalSandbox()
    client = _client(tmp_path, sandbox)
    try:
        with pytest.raises(AgentPromptTimeoutError):
            await execute_prompts(
                client,
                client.session,
                ["Echo started, then sleep 90."],
                timeout=3,
                idle_timeout=None,
            )
    finally:
        await client.close()
    # 0.9.2's record of this turn: no report came, so no usage at all.
    (turn,) = client.turns
    assert not turn.get("usage")

    rollout, _ = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )
    rollout._usage_report_missing = True  # execute() saw the timeout
    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    _assert_counted(usage, raw)

    # The CLI's summary line counts it and never says no model was used.
    job_dir = tmp_path / "jobs" / "run"
    job_dir.mkdir(parents=True)
    trial = _trial(
        job_dir,
        "timeout",
        error="Agent timed out after 3s",
        agent_result={
            key: usage[key]
            for key in ("total_tokens", "cost_usd", "usage_source", "cost_source")
        },
    )
    result = EvaluationResult(
        job_name="run",
        config=EvaluationConfig(),
        total=1,
        errored=1,
        job_dir=job_dir,
        results={"timeout": trial},
    )
    out = _render(monkeypatch, result, job_dir)
    assert "no model tokens" not in out
    assert f"Cost:      ${EXPECTED_USD:.2f}, {RAW_TOTAL:,} tokens" in out
    assert shared._counted_tokens(trial) == RAW_TOTAL


@pytest.mark.asyncio
async def test_a_lost_sandbox_keeps_the_count_the_host_wrote_as_the_turn_ran(
    tmp_path, monkeypatch
):
    home = tmp_path / "sandbox-home"
    _session_log(home)
    raw = _raw_session_total(home)
    sandbox = _LocalSandbox()
    client = _client(tmp_path, sandbox)
    task = asyncio.create_task(client.prompt("Echo started, then sleep 90."))
    try:
        # The host's stream log holds the second response's final output
        # count while the turn still runs: BenchFlow writes each event as it
        # arrives, not a copy out of the sandbox at the end.
        for _ in range(400):
            deltas = [
                e
                for e in _host_stream(tmp_path)
                if (e.get("event") or {}).get("type") == "message_delta"
            ]
            if len(deltas) == 2:
                break
            await asyncio.sleep(0.05)
        assert len(deltas) == 2
        assert not task.done()
        # The sandbox is deleted mid-turn; the caller stops waiting.
        sandbox.fail_exec = True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        sandbox.fail_exec = False
        await client.close()  # stops the local stand-in
    shutil.rmtree(home)  # its session log went with the sandbox

    rollout, _ = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )
    rollout._env = _GoneSandbox()
    rollout._usage_report_missing = True
    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    _assert_counted(usage, raw)
    assert usage["price_table_date"] == "2026-11-04"
    assert not (tmp_path / "trial" / "agent" / "claude-sessions").exists()
