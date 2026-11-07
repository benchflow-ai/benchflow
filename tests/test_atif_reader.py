"""BenchFlow reads ATIF trajectories, ATIF-v1.0 to ATIF-v1.8.

The fixture under ``fixtures/atif/job`` is a real trial a task.toml-package
runner wrote: its own terminal agent on MiMo v2.6 flash solving the
hello-world task (reward 1.0), with an ATIF-v1.8 ``agent/trajectory.json``.
The agent's long task prompt is replaced by a short placeholder; everything
else is as written. Its calls are terminal keystrokes parsed from the model's
text (no tool definitions), and one observation answers several calls.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from benchflow.trajectories import atif_reader as atif

JOB = Path(__file__).parent / "fixtures" / "atif" / "job"
TRIAL = JOB / "hello-world-task__fZWhoPQ"
REAL = json.loads((TRIAL / "agent" / "trajectory.json").read_text())

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _function_calling(version: str = "ATIF-v1.7") -> dict:
    """A function-calling agent's trajectory, with what ATIF-v1.5 to 1.8 add."""
    return {
        "schema_version": version,
        "agent": {
            "name": "a",
            "version": "1",
            "model_name": "m",
            "tool_definitions": [WEATHER],
        },
        "steps": [
            {"step_id": 1, "source": "system", "message": "be brief"},
            {"step_id": 2, "source": "user", "message": "weather in Paris?"},
            {
                "step_id": 3,
                "source": "agent",
                "message": "earlier turn",
                "is_copied_context": True,
            },
            {
                "step_id": 4,
                "source": "agent",
                "message": "",
                "reasoning_content": "the user wants the weather",
                "tool_calls": [
                    {
                        "tool_call_id": "c1",
                        "function_name": "get_weather",
                        "arguments": {"city": "Paris"},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "c1", "content": "18C"}]
                },
                "metrics": {
                    "prompt_tokens": 50,
                    "completion_tokens": 9,
                    "cached_tokens": 40,
                },
                "llm_call_count": 1,
            },
            {
                "step_id": 5,
                "source": "agent",
                "message": "dispatch",
                "llm_call_count": 0,
            },
            {
                "step_id": 6,
                "source": "agent",
                "message": [{"type": "text", "text": "It is 18C."}],
                "metrics": {"prompt_tokens": 70, "completion_tokens": 5},
            },
        ],
        "final_metrics": {"total_prompt_tokens": 120, "total_completion_tokens": 14},
    }


def test_every_1x_version_up_to_1_8_reads_and_others_are_refused():
    for minor in range(9):
        atif.load_atif(_function_calling(f"ATIF-v1.{minor}"))
    for version in ("ATIF-v1.9", "ATIF-v2.0", None):
        with pytest.raises(atif.AtifError, match="not one BenchFlow reads"):
            atif.load_atif(_function_calling(version))


def test_a_broken_trajectory_says_which_step():
    doc = _function_calling()
    doc["steps"][3]["observation"]["results"][0]["source_call_id"] = "c9"
    with pytest.raises(atif.AtifError, match="step 4: observation answers 'c9'"):
        atif.load_atif(doc)
    doc = _function_calling()
    doc["steps"][1]["step_id"] = 7
    with pytest.raises(atif.AtifError, match="step 2: step_id must be 2"):
        atif.load_atif(doc)


def test_a_function_calling_agent_keeps_its_tool_calls_structured():
    conversation = atif.atif_conversation(atif.load_atif(_function_calling()))
    assert conversation.structured
    roles = [m["role"] for m in conversation.messages]
    assert roles == [
        "system",
        "user",
        "assistant",
        "assistant",
        "tool",
        "assistant",
        "assistant",
    ]
    call = conversation.messages[3]["tool_calls"][0]
    assert call["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    assert conversation.messages[4] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "18C",
    }
    assert all("reasoning_content" not in m for m in conversation.messages)
    # Copied context and the no-call dispatch stay in later prompts, never a call.
    assert conversation.copied_steps == 1
    assert [conversation.messages[i]["content"] for i, _ in conversation.calls] == [
        "",
        "It is 18C.",
    ]


def test_each_model_call_becomes_a_proxy_log_record():
    exchanges = atif.atif_exchanges(atif.load_atif(_function_calling()))
    assert len(exchanges) == 2
    first, second = exchanges
    assert first["request"]["body"]["tools"] == [WEATHER]
    assert [m["role"] for m in first["request"]["body"]["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert first["response"]["body"]["choices"][0]["finish_reason"] == "tool_calls"
    assert first["response"]["body"]["usage"] == {
        "prompt_tokens": 50,
        "completion_tokens": 9,
        "prompt_tokens_details": {"cached_tokens": 40},
    }
    assert second["request"]["body"]["messages"][-1] == {
        "role": "assistant",
        "content": "dispatch",
    }
    assert first["metadata"]["source"] == "atif" and first["metadata"]["step_id"] == 4


def test_media_parts_are_left_out_and_counted():
    doc = _function_calling("ATIF-v1.8")
    doc["steps"][1]["message"] = [
        {"type": "text", "text": "what does this say?"},
        {"type": "image", "source": {"media_type": "image/png", "path": "a.png"}},
        {"type": "audio", "source": {"media_type": "audio/wav", "path": "a.wav"}},
    ]
    conversation = atif.atif_conversation(atif.load_atif(doc))
    assert conversation.messages[1]["content"] == "what does this say?"
    assert conversation.dropped_media == 2


def test_a_terminal_agent_trajectory_renders_its_calls_as_text():
    """No tool definitions: the calls were parsed from the model's text, so
    each assistant turn is the message, then one <tool_call> block per call;
    one observation answering three calls is one user message."""
    trajectory = atif.load_atif(REAL)
    assert trajectory["schema_version"] == "ATIF-v1.8"
    conversation = atif.atif_conversation(trajectory)
    assert not conversation.structured
    roles = [m["role"] for m in conversation.messages]
    assert roles == ["user"] + ["assistant", "user"] * 4
    first = conversation.messages[1]["content"]
    assert first.count("<tool_call>") == 3 and first.startswith("Analysis:")
    call = json.loads(first.split("<tool_call>\n")[1].split("\n</tool_call>")[0])
    assert call == {
        "name": "bash_command",
        "arguments": {
            "keystrokes": "printf 'Hello, world!\\n' > hello.txt\n",
            "duration": 0.1,
        },
    }
    assert "Hello, world!" in conversation.messages[2]["content"]
    assert atif.atif_totals(trajectory) == {
        "schema_version": "ATIF-v1.8",
        "steps": 5,
        "agent_steps": 4,
        "tool_calls": 7,
        "prompt_tokens": 4561,
        "completion_tokens": 648,
        "cached_tokens": 2880,
        "cost_usd": None,
    }


def test_bench_train_convert_reads_another_runners_trial(tmp_path):
    """The trial has no proxy log: its ATIF trajectory is the record, and its
    reward is read from the runner's own result.json."""
    from benchflow.trajectories.export_prime_sft import (
        convert_benchflow_rollouts_to_prime_sft_rows,
    )
    from benchflow.trajectories.export_trl_sft import (
        convert_benchflow_rollouts_to_trl_sft_rows,
    )

    job = tmp_path / "job"
    shutil.copytree(JOB, job)
    rows, stats = convert_benchflow_rollouts_to_trl_sft_rows(job, row_mode="exchange")
    assert stats.rows_written == len(rows) == 4
    assert {(r["source"], r["reward"]) for r in rows} == {("atif", 1.0)}
    assert rows[0]["source_path"].endswith("agent/trajectory.json")
    assert rows[0]["completion"][0]["content"].count("<tool_call>") == 3
    assert [len(r["prompt"]) for r in rows] == [1, 3, 5, 7]
    prime, prime_stats = convert_benchflow_rollouts_to_prime_sft_rows(job)
    assert prime_stats.rows_written == 1 and prime[0]["source"] == "atif"


def test_a_trial_atif_of_another_version_is_not_read(tmp_path):
    trial = tmp_path / TRIAL.name
    shutil.copytree(TRIAL, trial)
    doc = copy.deepcopy(REAL)
    doc["schema_version"] = "ATIF-v9"
    (trial / "agent" / "trajectory.json").write_text(json.dumps(doc))
    assert atif.trial_atif(trial) is None


def test_a_rollout_with_a_proxy_log_never_reads_its_atif(tmp_path):
    from benchflow.trajectories.export_prime_sft import load_rollout_exchanges

    trial = tmp_path / TRIAL.name
    shutil.copytree(TRIAL, trial)
    (trial / "trajectory").mkdir()
    shutil.copy(
        Path(__file__).parent / "fixtures" / "model_calls" / "trunc-proxy.jsonl",
        trial / "trajectory" / "llm_trajectory.jsonl",
    )
    exchanges = load_rollout_exchanges(trial)
    assert len(exchanges) == 30
    assert all((e.get("metadata") or {}).get("source") != "atif" for e in exchanges)
