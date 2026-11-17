"""A Claude Code trial's tokens count its subagents (EN-498,
benchflow-ai/benchflow#1241, reported by Benjamin-eecs).

claude-agent-acp's end-of-turn report (``PromptResponse.usage``) counts the
main session only, and result.json's ``agent_result`` took it as the
trial's tokens: a subagent's calls (the Agent/Task tool) were left out. On
benchflow 0.7.8, fable52 d1 QNTK-research 2d46c7d6 recorded 545,698 output
tokens where Claude Code's OpenTelemetry counted 546,925 for the main agent
and 355,860 for a general-purpose subagent (902,785). The fixture below
has those output counts. Whenever teardown copies Claude Code's session
log, its count (main session, subagents, and Claude Code's own side calls
when its totals cover them) is the trial's, and
``usage_details.token_count`` names its scope and splits it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.rollout._cost import claude_session_usage
from tests.test_en374_salvage import _rollout

# usage_details.token_count: what the count was taken from, and covers
COUNTED_FROM_TOTALS = "claude-code-totals"
COUNTED_FROM_RESPONSES = "responses"
SCOPE_ALL = "main session, subagents and Claude Code's side calls"
SCOPE_LOGGED = "main session and subagents"
SCOPE_MAIN = "main session only"
OPUS = "claude-opus-5-5"
MAIN_OUTPUT = (300_000, 246_925)  # 546,925
SUBAGENT_OUTPUT = (200_000, 155_860)  # 355,860
REPORTED_OUTPUT = 545_698  # agent_result on 0.7.8


def _usage(fresh: int, output: int, read: int = 0, write: int = 0) -> dict:
    return {
        "input_tokens": fresh,
        "output_tokens": output,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
    }


def _line(prefix: str, index: int, usage: dict) -> dict:
    return {
        "type": "assistant",
        "requestId": f"req_{prefix}{index}",
        "message": {"id": f"msg_{prefix}{index}", "model": OPUS, "usage": usage},
    }


def _write(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def _qntk_log(home: Path, *, state_output: int | None) -> Path:
    """The main session's two responses, a subagent's two, and (unless
    None) a cost-state counting ``state_output`` output tokens."""
    root = home / ".claude" / "projects" / "-app"
    main = [
        _line("m", i, _usage(10, out, 1000, 100)) for i, out in enumerate(MAIN_OUTPUT)
    ]
    if state_output is not None:
        main.append(
            {
                "type": "cost-state",
                "totalCostUSD": 20.0,
                "modelUsage": {
                    OPUS: {
                        "inputTokens": 40,
                        "outputTokens": state_output,
                        "cacheReadInputTokens": 4000,
                        "cacheCreationInputTokens": 400,
                        "costUSD": 20.0,
                    }
                },
            }
        )
    _write(root / "session-1.jsonl", main)
    _write(
        root / "session-1" / "subagents" / "agent-general.jsonl",
        [
            _line("s", i, _usage(10, out, 1000, 100))
            for i, out in enumerate(SUBAGENT_OUTPUT)
        ],
    )
    return home / ".claude" / "projects"


def test_the_count_splits_main_session_and_subagents(tmp_path):
    counted = claude_session_usage(_qntk_log(tmp_path, state_output=None))
    assert counted is not None
    assert counted["counted_from"] == COUNTED_FROM_RESPONSES
    assert counted["scope"] == SCOPE_LOGGED
    assert counted["n_output_tokens"] == 902_785
    assert counted["main"]["n_output_tokens"] == 546_925
    assert counted["subagents"]["n_output_tokens"] == 355_860
    assert (counted["main"]["responses"], counted["subagents"]["responses"]) == (2, 2)


def test_claude_codes_totals_add_what_the_responses_left_out(tmp_path):
    """Its cost-state counts 1,000 more output tokens than the responses
    log (a title, a compaction summary): they count, and are named."""
    counted = claude_session_usage(_qntk_log(tmp_path, state_output=903_785))
    assert counted is not None
    assert counted["counted_from"] == COUNTED_FROM_TOTALS
    assert counted["scope"] == SCOPE_ALL
    assert counted["n_output_tokens"] == 903_785
    assert counted["unlogged"]["n_output_tokens"] == 1_000


def test_a_stale_cost_state_falls_back_to_the_responses(tmp_path):
    counted = claude_session_usage(_qntk_log(tmp_path, state_output=500_000))
    assert counted is not None
    assert counted["counted_from"] == COUNTED_FROM_RESPONSES
    assert counted["n_output_tokens"] == 902_785


def _reported(rollout, output: int) -> None:
    rollout._native_usage_metrics = {
        **rollout._native_usage_metrics,
        "n_input_tokens": 20,
        "n_output_tokens": output,
        "n_cache_read_tokens": 2000,
        "n_cache_creation_tokens": 200,
        "total_tokens": 20 + output + 2000 + 200,
        "usage_source": "agent_native_acp",
        "cost_usd": None,
    }


@pytest.mark.asyncio
async def test_agent_result_counts_the_subagents_tokens(tmp_path, monkeypatch):
    """benchflow#1241: 902,785 output tokens, not the main session's 545,698."""
    home = tmp_path / "sandbox-home"
    _qntk_log(home, state_output=None)
    rollout, _trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-" + "x" * 40}
    _reported(rollout, REPORTED_OUTPUT)

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["n_output_tokens"] == 902_785
    assert usage["usage_source"] == "session_log"
    count = usage["usage_details"]["token_count"]
    assert count["scope"] == SCOPE_LOGGED
    assert count["subagents"]["n_output_tokens"] == 355_860
    assert count["end_of_turn_reported"]["n_output_tokens"] == REPORTED_OUTPUT


@pytest.mark.asyncio
async def test_a_log_shorter_than_the_reports_keeps_the_reports(tmp_path, monkeypatch):
    """A log cut short counts less than the turns reported: the reports
    stay, the log's count beside them."""
    home = tmp_path / "sandbox-home"
    _qntk_log(home, state_output=None)
    rollout, _trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-" + "x" * 40}
    _reported(rollout, 2_000_000)

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["n_output_tokens"] == 2_000_000
    count = usage["usage_details"]["token_count"]
    assert count["scope"] == SCOPE_MAIN
    assert count["session_log"]["n_output_tokens"] == 902_785


@pytest.mark.asyncio
async def test_without_a_log_the_count_says_it_is_the_main_session(
    tmp_path, monkeypatch
):
    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, _trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-" + "x" * 40}
    _reported(rollout, REPORTED_OUTPUT)

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["n_output_tokens"] == REPORTED_OUTPUT
    assert usage["usage_details"]["token_count"]["scope"] == SCOPE_MAIN


@pytest.mark.asyncio
async def test_a_cut_turn_counted_from_the_log_is_not_incomplete(tmp_path, monkeypatch):
    """0.9.3's partial-count marker: a turn cut off before its report,
    counted from the log, subagents included, is a whole count."""
    home = tmp_path / "sandbox-home"
    _qntk_log(home, state_output=None)
    rollout, _trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-" + "x" * 40}
    _reported(rollout, REPORTED_OUTPUT)
    rollout._usage_report_missing = True

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["n_output_tokens"] == 902_785
    assert usage["usage_details"]["token_count"]["scope"] == SCOPE_LOGGED
    assert "usage_incomplete" not in usage


@pytest.mark.asyncio
async def test_a_cut_turn_without_a_log_is_incomplete_and_main_session_only(
    tmp_path, monkeypatch
):
    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, _trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-" + "x" * 40}
    _reported(rollout, REPORTED_OUTPUT)
    rollout._usage_report_missing = True

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["n_output_tokens"] == REPORTED_OUTPUT
    assert usage["usage_incomplete"] is True
    assert usage["usage_details"]["token_count"]["scope"] == SCOPE_MAIN


@pytest.mark.asyncio
async def test_a_log_short_of_one_kind_keeps_the_reports(tmp_path, monkeypatch):
    """a review's input (0.10's per-kind rule, ported): the log has no
    cost-state, its main session's lines logged 35 + 35 of the 1,005 output
    tokens the turns reported (MiMo's lines can be written before a response
    ends), and a subagent's two responses add 20,000 cache reads each, so
    the log's total (42,560) passes the reports' (3,225). The log is short
    of output: the reports stay, the log's count and its short kinds
    beside them."""
    home = tmp_path / "sandbox-home"
    root = home / ".claude" / "projects" / "-app"
    _write(
        root / "session-1.jsonl",
        [_line("m", i, _usage(10, 35, 1000, 100)) for i in range(2)],
    )
    _write(
        root / "session-1" / "subagents" / "agent-general.jsonl",
        [_line("s", i, _usage(10, 25, 20_000, 100)) for i in range(2)],
    )
    rollout, _trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-" + "x" * 40}
    _reported(rollout, 1_005)  # 20 in, 2,000 cache read, 200 written: 3,225

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["usage_source"] != "session_log"
    assert (usage["n_output_tokens"], usage["total_tokens"]) == (1_005, 3_225)
    count = usage["usage_details"]["token_count"]
    assert count["scope"] == SCOPE_MAIN
    log = count["session_log"]
    assert log["short_of"] == ["n_output_tokens"]
    assert (log["n_output_tokens"], log["total_tokens"]) == (120, 42_560)


def test_a_covered_count_prices_each_logged_response_at_its_tier(tmp_path):
    """0.10's test, ported: Claude Code's totals cover a Haiku 5.5 response
    with a 162,000-token prompt (over the 100K tier), its cache writes
    1-hour; the count's slices keep that response's prompt and 1-hour
    writes, and price only what the totals add at the base tier."""
    from benchflow._utils.pricing import price_slices

    haiku = "claude-haiku-5-5"
    usage = {
        "input_tokens": 2_000,
        "output_tokens": 4_000,
        "cache_read_input_tokens": 150_000,
        "cache_creation_input_tokens": 10_000,
        "cache_creation": {"ephemeral_1h_input_tokens": 10_000},
    }
    root = tmp_path / ".claude" / "projects" / "-app"
    line = {
        "type": "assistant",
        "requestId": "req_0",
        "message": {"id": "msg_0", "model": haiku, "usage": usage},
    }
    state = {
        "type": "cost-state",
        "totalCostUSD": 0.031,
        "modelUsage": {
            haiku: {
                "inputTokens": 2_000,
                "outputTokens": 5_000,
                "cacheReadInputTokens": 150_000,
                "cacheCreationInputTokens": 10_000,
                "costUSD": 0.031,
            }
        },
    }
    _write(root / "session-1.jsonl", [line, state])
    counted = claude_session_usage(tmp_path / ".claude" / "projects")
    assert counted is not None and counted["counted_from"] == COUNTED_FROM_TOTALS
    priced = price_slices(counted["slices"])
    assert priced is not None
    over = 2_000 * 0.5e-6 + 4_000 * 2.5e-6 + 150_000 * 0.05e-6 + 10_000 * 1e-6
    assert priced.usd == pytest.approx(over + 1_000 * 0.5e-6)


MIMO = "mimo-v2.6-pro"
FIRST = _usage(1_500, 120, 0, 3_000)
SECOND = _usage(20, 300, 3_000, 400)
ONCE = {
    "n_input_tokens": 1_520,
    "n_output_tokens": 420,
    "n_cache_read_tokens": 3_000,
    "n_cache_creation_tokens": 3_400,
    "total_tokens": 8_340,
}
ONCE_AT_MIMOS_PRICE = 0.0025164  # 1,520 and 3,400 written at $0.435/M, ...


def _response(index: int, usage: dict, model: str) -> dict:
    return {
        "type": "assistant",
        "requestId": f"req_{index}",
        "message": {"id": f"msg_{index}", "model": model, "usage": usage},
    }


def _logged_twice(home: Path, model: str, *, resumed: bool) -> Path:
    """Two responses, then the second one's first block again (its output
    so far: 1 token) after its full line. ``resumed``: a second session file
    logs both responses again, with that partial line (0.10's EN-374 test
    input); else the partial line follows in the same file."""
    root = home / ".claude" / "projects" / "-app"
    lines = [_response(0, FIRST, model), _response(1, SECOND, model)]
    partial = _response(1, {**SECOND, "output_tokens": 1}, model)
    if resumed:
        _write(root / "session-1.jsonl", lines)
        _write(root / "session-2.jsonl", [*lines, partial])
    else:
        _write(root / "session-1.jsonl", [*lines, partial])
    return home / ".claude" / "projects"


@pytest.mark.parametrize("resumed", [True, False], ids=["two-files", "partial-line"])
def test_a_response_counts_once_with_its_most_output(tmp_path, resumed):
    """a review's inputs (0.10's rule, ported): a response is one (message
    id, request id), however many lines and session files log it. It counts
    once, in the session that logged it first, with its line with the most
    output, in the count and in the session-log estimate."""
    from benchflow._utils.session_cost import estimate_claude_code_cost

    folder = _logged_twice(tmp_path, MIMO, resumed=resumed)
    counted = claude_session_usage(folder)
    assert counted is not None
    assert counted["responses"] == 2
    assert {field: counted[field] for field in ONCE} == ONCE
    estimate = estimate_claude_code_cost(folder)
    assert estimate is not None
    assert estimate["usd"] == pytest.approx(ONCE_AT_MIMOS_PRICE)


def _short_output_log(home: Path) -> None:
    """REV3's input (review finding R1 on #164): the main session's lines
    log 35 + 35 output tokens (MiMo's lines are written before a response
    ends), a subagent's two responses 25 output and 20,000 cache read each,
    and no cost-state: 40 input, 120 output, 42,000 cache read and 400
    cache write in all."""
    root = home / ".claude" / "projects" / "-app"
    _write(
        root / "session-1.jsonl",
        [_line("m", i, _usage(10, 35, 1000, 100)) for i in range(2)],
    )
    _write(
        root / "session-1" / "subagents" / "agent-general.jsonl",
        [_line("s", i, _usage(10, 25, 20_000, 100)) for i in range(2)],
    )


# The reports: 20 input, 1,005 output, 2,000 cache read, and 200 cache
# write (one kind the log is short of) or 5,000 (two kinds).
SHORT_OF_ONE = (200, ["n_output_tokens"], "output")
SHORT_OF_TWO = (
    5_000,
    ["n_output_tokens", "n_cache_creation_tokens"],
    "output and cache-write",
)


async def _short_log_trial(tmp_path, monkeypatch, *, write: int, cut: bool):
    home = tmp_path / "sandbox-home"
    _short_output_log(home)
    rollout, _trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-" + "x" * 40}
    _reported(rollout, 1_005)
    rollout._native_usage_metrics.update(
        n_cache_creation_tokens=write, total_tokens=20 + 1_005 + 2_000 + write
    )
    rollout._usage_report_missing = cut
    await rollout._cleanup_before_teardown()
    return rollout._usage_metrics


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [SHORT_OF_ONE, SHORT_OF_TWO], ids=["one", "two"])
async def test_a_cut_turns_short_log_keeps_each_kinds_larger_count(
    tmp_path, monkeypatch, case
):
    """a review P1: a turn ended without its report, and the log is short of
    a kind the turns that reported counted. The log replaced the reports
    whole (120 output tokens of 1,005) and the trial read as complete. Each
    is a lower bound: each kind is now the larger, ``short_of`` names the
    kinds, and the count and its cost are marked incomplete."""
    from benchflow._utils.result_metadata import incomplete_usage_fields

    write, kinds, named = case
    usage = await _short_log_trial(tmp_path, monkeypatch, write=write, cut=True)

    assert usage["usage_source"] == "session_log"
    assert (
        usage["n_input_tokens"],
        usage["n_output_tokens"],
        usage["n_cache_read_tokens"],
        usage["n_cache_creation_tokens"],
    ) == (40, 1_005, 42_000, max(400, write))
    assert usage["total_tokens"] == 40 + 1_005 + 42_000 + max(400, write)
    count = usage["usage_details"]["token_count"]
    assert count["short_of"] == kinds
    assert count["end_of_turn_reported"]["n_output_tokens"] == 1_005
    assert usage["usage_incomplete"] is True
    assert f"fewer {named} tokens" in usage["usage_incomplete_reason"]
    assert usage["cost_usd"] is not None
    assert incomplete_usage_fields(usage)["cost_incomplete"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [SHORT_OF_ONE, SHORT_OF_TWO], ids=["one", "two"])
async def test_reports_kept_beside_a_short_log_are_marked_incomplete(
    tmp_path, monkeypatch, case
):
    """a review P2: every turn reported and the log is short of a kind, so
    the reports stay; the cost was still the log's estimate, priced from
    its 120 output tokens, and read as complete. Now the trial is marked
    incomplete (usage_incomplete, cost_incomplete); the cost is not
    re-priced from the reports, which count the main session only."""
    from benchflow._utils.result_metadata import incomplete_usage_fields

    write, kinds, named = case
    usage = await _short_log_trial(tmp_path, monkeypatch, write=write, cut=False)

    assert usage["usage_source"] == "agent_native_acp"
    assert usage["n_output_tokens"] == 1_005
    assert usage["n_cache_creation_tokens"] == write
    count = usage["usage_details"]["token_count"]
    assert count["scope"] == SCOPE_MAIN
    assert count["session_log"]["short_of"] == kinds
    assert count["session_log"]["n_output_tokens"] == 120
    assert usage["usage_incomplete"] is True
    assert f"fewer {named} tokens" in usage["usage_incomplete_reason"]
    assert usage["cost_usd"] is not None
    assert incomplete_usage_fields(usage)["cost_incomplete"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", [False, True], ids=["reported", "cut-turn"])
async def test_a_complete_log_replaces_the_reports_and_is_not_marked(
    tmp_path, monkeypatch, cut
):
    """A log counting at least each reported kind (REV3's log against
    reports of 20 / 100 / 2,000 / 200) is the count, the reports beside it,
    and nothing is marked incomplete, with or without a cut turn."""
    from benchflow._utils.result_metadata import incomplete_usage_fields

    home = tmp_path / "sandbox-home"
    _short_output_log(home)
    rollout, _trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="acp", home=home
    )
    rollout._agent_env = {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-" + "x" * 40}
    _reported(rollout, 100)
    rollout._usage_report_missing = cut

    await rollout._cleanup_before_teardown()

    usage = rollout._usage_metrics
    assert usage["usage_source"] == "session_log"
    assert usage["n_output_tokens"] == 120
    count = usage["usage_details"]["token_count"]
    assert "short_of" not in count
    assert count["end_of_turn_reported"]["n_output_tokens"] == 100
    assert not usage.get("usage_incomplete")
    assert usage["cost_usd"] is not None
    assert incomplete_usage_fields(usage) == {}
