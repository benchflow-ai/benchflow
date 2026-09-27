"""Separate Claude subagent LLM calls from the parent agent in SFT conversion.

The fixtures under ``tests/fixtures/claude_subagents`` are synthetic captures
shaped like a ``claude-agent-acp`` / ``claude-opus-5`` rollout recorded through
the LiteLLM gateway. The tasks, prompts, answers and tool descriptions are
invented; the request and response layout, the tool-call ids that link the
exchanges and the declared tool names follow the shape of a real capture.

- ``plan_subagent`` (rollout ``demo-plan-task``): exchange 0 is the parent
  calling ``Agent`` (``subagent_type: Plan``); exchanges 1-2 are the Plan
  subagent's own conversation (first user message = the ``Agent`` prompt, no
  ``Agent`` tool); exchange 3 is the parent again.
- ``websearch_helper`` (rollout ``demo-websearch-task``): exchange 0 spawns an
  Explore subagent whose only call (exchange 1) failed with 500; exchange 3 is
  the tool-less model call Claude Code makes inside its ``WebSearch`` tool;
  exchanges 2 and 4 are the parent.

Without subagent separation, ``bench train convert`` treated every captured
exchange as the parent's, so exchange mode emitted the subagent's calls as
parent rows and rollout mode picked whichever exchange was captured last, even
when it was the subagent's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.trajectories.export_prime_sft import (
    convert_benchflow_rollouts_to_prime_sft_rows,
    export_prime_sft_jsonl,
    validate_prime_sft_jsonl,
)
from benchflow.trajectories.export_trl_sft import (
    convert_benchflow_rollouts_to_trl_sft_rows,
    export_trl_sft_jsonl,
    validate_trl_sft_jsonl,
)
from benchflow.trajectories.results import write_rollout_results_jsonl
from benchflow.trajectories.sft_subagents import SubagentAttributionError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "claude_subagents"
SPAWN_ID = "toolu_demo_plan_spawn_0002"
SUBAGENT_TOOL_IDS = ("toolu_demo_plan_read_0003", "toolu_demo_plan_bash_0004")

runner = CliRunner()


def _fixture(name: str) -> list[dict[str, Any]]:
    path = FIXTURES / f"{name}.llm_trajectory.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_rollout(
    job_dir: Path,
    records: list[dict[str, Any]],
    *,
    acp_events: list[dict[str, Any]] | None = None,
    name: str = "demo-plan-task__00000001",
) -> Path:
    rollout = job_dir / name
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps(
            {
                "task_name": name.split("__")[0],
                "agent": "claude-agent-acp",
                "model": "claude-opus-5",
                "rewards": {"reward": 1.0},
            }
        )
    )
    (rollout / "trajectory" / "llm_trajectory.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    if acp_events is not None:
        (rollout / "trajectory" / "acp_trajectory.jsonl").write_text(
            "\n".join(json.dumps(event) for event in acp_events) + "\n"
        )
    return rollout


def _first_user_prompt_block(record: dict[str, Any]) -> dict[str, Any]:
    first_user = next(
        message
        for message in record["request"]["body"]["messages"]
        if message["role"] == "user"
    )
    return first_user["content"][-1]


def _as_compacted_subagent(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite the subagent's first message the way a context compaction does,
    so it no longer equals the ``Agent`` prompt."""
    for record in records[1:3]:
        _first_user_prompt_block(record)["text"] = (
            "This session is being continued from a previous conversation "
            "that ran out of context."
        )
    return records


def _acp_child_events(parent: str) -> list[dict[str, Any]]:
    return [
        {"type": "tool_call", "tool_call_id": SPAWN_ID, "title": "Task"},
        *(
            {
                "type": "tool_call",
                "tool_call_id": call_id,
                "parent_tool_call_id": parent,
            }
            for call_id in SUBAGENT_TOOL_IDS
        ),
    ]


def test_prime_sft_exchange_mode_excludes_claude_subagent_calls(
    tmp_path: Path,
) -> None:
    _write_rollout(tmp_path / "job", _fixture("plan_subagent"))

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        tmp_path / "job", row_mode="exchange"
    )

    assert [row["exchange_index"] for row in rows] == [0, 3]
    assert all("agent_role" not in row for row in rows)
    report = stats.as_dict()
    assert report["rollouts_with_subagents"] == 1
    assert report["subagent_calls_seen"] == 1
    assert report["subagent_exchanges_seen"] == 2
    assert report["subagent_exchanges_excluded"] == 2
    assert report["subagent_rows_written"] == 0
    assert report["rows_written"] == 2


def test_prime_sft_rollout_mode_ignores_subagent_exchange_captured_last(
    tmp_path: Path,
) -> None:
    # The capture ends inside the Plan subagent, as when a rollout hits its
    # wall-clock budget while a subagent is running. The rollout row
    # must be the parent's last call, not the subagent's final report.
    _write_rollout(tmp_path / "job", _fixture("plan_subagent")[:3])

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(tmp_path / "job")

    assert [row["exchange_index"] for row in rows] == [0]
    assert rows[0]["messages"][-1]["tool_calls"][0]["function"]["name"] == "Agent"
    assert stats.as_dict()["subagent_exchanges_excluded"] == 2


def test_trl_sft_exchange_mode_excludes_claude_subagent_calls(
    tmp_path: Path,
) -> None:
    _write_rollout(tmp_path / "job", _fixture("plan_subagent"))

    rows, stats = convert_benchflow_rollouts_to_trl_sft_rows(
        tmp_path / "job", row_mode="exchange"
    )

    assert [row["exchange_index"] for row in rows] == [0, 3]
    report = stats.as_dict()
    assert report["subagent_exchanges_seen"] == 2
    assert report["subagent_exchanges_excluded"] == 2


@pytest.mark.parametrize(
    ("row_mode", "expected"),
    [
        ("exchange", [(0, "parent"), (3, "parent"), (1, "subagent"), (2, "subagent")]),
        ("rollout", [(3, "parent"), (2, "subagent")]),
    ],
)
def test_subagent_rows_opt_in_emits_linked_rows(
    tmp_path: Path, row_mode: str, expected: list[tuple[int, str]]
) -> None:
    _write_rollout(tmp_path / "job", _fixture("plan_subagent"))

    for convert in (
        convert_benchflow_rollouts_to_prime_sft_rows,
        convert_benchflow_rollouts_to_trl_sft_rows,
    ):
        rows, stats = convert(tmp_path / "job", row_mode=row_mode, subagent_rows=True)

        assert [(row["exchange_index"], row["agent_role"]) for row in rows] == expected
        for row in rows:
            if row["agent_role"] == "subagent":
                assert row["parent_tool_call_id"] == SPAWN_ID
                assert row["subagent_type"] == "Plan"
                assert row["subagent_description"] == "Review ringlog design"
            else:
                assert "parent_tool_call_id" not in row
        report = stats.as_dict()
        included = sum(role == "subagent" for _, role in expected)
        assert report["subagent_rows_written"] == included
        assert report["subagent_exchanges_excluded"] == 2 - included


def test_tool_less_nested_model_call_is_not_a_parent_row(tmp_path: Path) -> None:
    _write_rollout(
        tmp_path / "job",
        _fixture("websearch_helper"),
        name="demo-websearch-task__00000002",
    )

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        tmp_path / "job", row_mode="exchange"
    )

    assert [row["exchange_index"] for row in rows] == [0, 2, 4]
    report = stats.as_dict()
    assert report["skipped_helper_calls"] == 1
    assert report["subagent_calls_seen"] == 1
    # The Explore subagent's only call failed upstream, so nothing to exclude.
    assert report["subagent_exchanges_seen"] == 0


def test_unattributable_tool_using_conversation_fails_loudly(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_rollout(job, _as_compacted_subagent(_fixture("plan_subagent")))
    out = tmp_path / "train.jsonl"

    with pytest.raises(SubagentAttributionError, match=r"exchanges \[1, 2\]"):
        export_prime_sft_jsonl(job, out, row_mode="exchange")
    assert not out.exists()

    result = runner.invoke(
        app, ["train", "convert", str(job), "--out", str(out), "--format", "trl-sft"]
    )
    assert result.exit_code == 1
    assert "cannot attribute" in result.output


def test_acp_parent_linkage_attributes_a_compacted_subagent(tmp_path: Path) -> None:
    _write_rollout(
        tmp_path / "job",
        _as_compacted_subagent(_fixture("plan_subagent")),
        acp_events=_acp_child_events(SPAWN_ID),
    )

    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(
        tmp_path / "job", row_mode="exchange", subagent_rows=True
    )

    assert [(row["exchange_index"], row["agent_role"]) for row in rows] == [
        (0, "parent"),
        (3, "parent"),
        (1, "subagent"),
        (2, "subagent"),
    ]
    assert {row.get("parent_tool_call_id") for row in rows[2:]} == {SPAWN_ID}
    assert rows[2]["subagent_type"] == "Plan"


def test_acp_linkage_contradicting_the_spawn_prompt_fails_loudly(
    tmp_path: Path,
) -> None:
    _write_rollout(
        tmp_path / "job",
        _fixture("plan_subagent"),
        acp_events=_acp_child_events("toolu_someOtherSpawn"),
    )

    with pytest.raises(SubagentAttributionError, match="toolu_someOtherSpawn"):
        convert_benchflow_rollouts_to_prime_sft_rows(tmp_path / "job")


def test_parent_that_delegates_its_own_prompt_stays_the_parent(tmp_path: Path) -> None:
    records = _fixture("plan_subagent")
    task_prompt = _first_user_prompt_block(records[0])["text"]
    call = records[0]["response"]["body"]["choices"][0]["message"]["tool_calls"][0]
    arguments = json.loads(call["function"]["arguments"])
    arguments["prompt"] = task_prompt
    call["function"]["arguments"] = json.dumps(arguments)
    for record in records[1:3]:
        _first_user_prompt_block(record)["text"] = task_prompt
    _write_rollout(tmp_path / "job", records)

    rows, _ = convert_benchflow_rollouts_to_prime_sft_rows(
        tmp_path / "job", row_mode="exchange", subagent_rows=True
    )

    assert [(row["exchange_index"], row["agent_role"]) for row in rows] == [
        (0, "parent"),
        (3, "parent"),
        (1, "subagent"),
        (2, "subagent"),
    ]


def test_train_convert_cli_reports_subagent_counts(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_rollout(job, _fixture("plan_subagent"))
    out = tmp_path / "train.jsonl"
    manifest = tmp_path / "manifest.json"

    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(job),
            "--out",
            str(out),
            "--row-mode",
            "exchange",
            "--manifest",
            str(manifest),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "excluded 2 subagent LLM exchange(s)" in result.output
    assert validate_prime_sft_jsonl(out)["rows"] == 2
    assert json.loads(manifest.read_text())["subagent_exchanges_excluded"] == 2

    trl_out = tmp_path / "train.trl.jsonl"
    result = runner.invoke(
        app,
        [
            "train",
            "convert",
            str(job),
            "--out",
            str(trl_out),
            "--format",
            "trl-sft",
            "--row-mode",
            "exchange",
            "--subagent-rows",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "included 2 subagent row(s)" in result.output
    assert validate_trl_sft_jsonl(trl_out)["rows"] == 4


def _results_jsonl(rollout: Path) -> Path:
    write_rollout_results_jsonl(
        rollout,
        task_name="demo-plan-task",
        rollout_name=rollout.name,
        agent="claude-agent-acp",
        agent_name="@agentclientprotocol/claude-agent-acp",
        model="claude-opus-5",
        n_tool_calls=4,
        prompts=["Your task is to implement ringlog."],
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        error=None,
        verifier_error=None,
    )
    return rollout / "results.jsonl"


@pytest.mark.parametrize("export", [export_prime_sft_jsonl, export_trl_sft_jsonl])
def test_results_jsonl_with_subagent_steps_is_refused(tmp_path: Path, export) -> None:
    # results.jsonl keeps every captured call as a trajectory step (and takes
    # the prime-sft prompt/completion from the last one) without the per-call
    # tools or ACP linkage needed to attribute them, so conversion refuses it.
    source = _results_jsonl(_write_rollout(tmp_path / "job", _fixture("plan_subagent")))
    out = tmp_path / "train.jsonl"

    with pytest.raises(SubagentAttributionError, match=r"steps \[1, 2\]"):
        export(source, out)
    assert not out.exists()


def test_results_jsonl_without_subagent_calls_still_converts(tmp_path: Path) -> None:
    records = _fixture("websearch_helper")
    source = _results_jsonl(
        _write_rollout(tmp_path / "job", records, name="demo-websearch-task__00000002")
    )

    assert export_prime_sft_jsonl(source, tmp_path / "train.jsonl").rows_written == 1


@pytest.mark.parametrize("export", [export_prime_sft_jsonl, export_trl_sft_jsonl])
def test_results_jsonl_ending_inside_a_subagent_is_refused(
    tmp_path: Path, export
) -> None:
    """Guards the results.jsonl path when a capture ends inside a subagent.

    The capture ends inside the Plan subagent. The results writer keeps only
    calls a later request consumed, so the parent's spawning call is dropped
    and the row's steps are the subagent's alone. Spawn-prompt evidence read
    back from the row then finds nothing, and the subagent's final report was
    converted as the parent's conversation. The writer now records each step's
    owner while it still has the whole capture, and conversion refuses the row.
    """
    rollout = _write_rollout(tmp_path / "job", _fixture("plan_subagent")[:3])
    source = _results_jsonl(rollout)
    steps = json.loads(source.read_text().splitlines()[0])["trajectory"]
    assert steps and {s["extras"]["agent_role"] for s in steps} == {"subagent"}
    assert {s["extras"]["parent_tool_call_id"] for s in steps} == {SPAWN_ID}
    out = tmp_path / "train.jsonl"

    with pytest.raises(SubagentAttributionError, match="subagent"):
        export(source, out)
    assert not out.exists()


def test_results_jsonl_steps_name_their_owner(tmp_path: Path) -> None:
    source = _results_jsonl(
        _write_rollout(
            tmp_path / "job",
            _fixture("websearch_helper"),
            name="demo-websearch-task__00000002",
        )
    )
    steps = json.loads(source.read_text().splitlines()[0])["trajectory"]
    roles = [s["extras"]["agent_role"] for s in steps]
    assert "parent" in roles and set(roles) <= {"parent", "helper"}
