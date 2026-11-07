"""Guards FF-sw-trajectories SFT findings at 363d815b."""

import copy
import json

import pytest

from benchflow.trajectories.export_prime_sft import (
    convert_benchflow_rollouts_to_prime_sft_rows,
)
from benchflow.trajectories.export_trl_sft import (
    convert_benchflow_rollouts_to_trl_sft_rows,
)

EXPORTERS = [
    convert_benchflow_rollouts_to_prime_sft_rows,
    convert_benchflow_rollouts_to_trl_sft_rows,
]


def write_rollout(tmp_path, exchanges):
    rollout = tmp_path / "task"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "result.json").write_text(
        json.dumps({"task_name": "dummy", "rewards": {"reward": 1}})
    )
    (rollout / "trajectory" / "llm_trajectory.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in exchanges)
    )


@pytest.mark.parametrize("exporter", EXPORTERS)
@pytest.mark.parametrize("row_mode", ["exchange", "rollout"])
@pytest.mark.parametrize(
    "bad",
    [
        {"status": "incomplete"},
        {"status": "failed"},
        {"status": "in_progress"},
        {"status": "queued"},
        {"status": "cancelled"},
        {"incomplete_details": {"reason": "max_tokens"}},
    ],
)
def test_directory_export_selects_only_completed_responses(
    tmp_path, exporter, row_mode, bad
):
    """Guards the fix using the canonical response selector."""
    complete = {
        "request": {"body": {"messages": [{"role": "user", "content": "List files"}]}},
        "response": {
            "status_code": 200,
            "body": {
                "choices": [{"message": {"role": "assistant", "content": "Done"}}]
            },
        },
    }
    incomplete = copy.deepcopy(complete)
    incomplete["response"]["body"].update(bad)
    incomplete["response"]["body"]["choices"][0]["message"]["content"] = "Partial"
    write_rollout(tmp_path, [complete, incomplete])
    rows, _ = exporter(tmp_path, row_mode=row_mode)
    assert len(rows) == 1
    assert rows[0]["exchange_index"] == 0


@pytest.mark.parametrize("exporter", EXPORTERS)
def test_directory_export_omits_unconsumed_retry(tmp_path, exporter):
    """Guards FF-sw-trajectories: a consumed response outranks its late retry."""
    tool = {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    request = {
        "body": {
            "messages": [{"role": "user", "content": "List files"}],
            "tools": [tool],
        }
    }
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "used",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }
        ],
    }
    consumed = {
        "request": request,
        "response": {"status_code": 200, "body": {"choices": [{"message": call}]}},
    }
    followup = {
        "request": {
            "body": {
                "messages": [
                    *request["body"]["messages"],
                    call,
                    {"role": "tool", "tool_call_id": "used", "content": "README"},
                ],
                "tools": [tool],
            }
        },
        "response": {
            "status_code": 200,
            "body": {
                "choices": [{"message": {"role": "assistant", "content": "Done"}}]
            },
        },
    }
    late = copy.deepcopy(consumed)
    late["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["id"] = "unused"
    write_rollout(tmp_path, [consumed, followup, late])
    rows, _ = exporter(tmp_path, row_mode="exchange")
    assert [row["exchange_index"] for row in rows] == [0, 1]


@pytest.mark.parametrize("exporter", EXPORTERS)
@pytest.mark.parametrize(
    "family,reason",
    [
        ("chat", "length"),
        ("chat", "content_filter"),
        ("chat", None),
        ("anthropic", "max_tokens"),
        ("anthropic", "pause_turn"),
        ("anthropic", None),
    ],
)
def test_nonterminal_provider_finish_reasons_are_not_training(
    tmp_path, exporter, family, reason
):
    """Guards FF-sw-trajectories across equivalent provider completion flags."""
    from benchflow.trajectories.results import _training_success_exchange_indices

    body = (
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Partial"},
                    "finish_reason": reason,
                }
            ]
        }
        if family == "chat"
        else {"content": [{"type": "text", "text": "Partial"}], "stop_reason": reason}
    )
    exchange = {
        "request": {"body": {"messages": [{"role": "user", "content": "List files"}]}},
        "response": {"status_code": 200, "body": body},
    }
    assert _training_success_exchange_indices([exchange]) == set()
    write_rollout(tmp_path, [exchange])
    assert exporter(tmp_path, row_mode="exchange")[0] == []
