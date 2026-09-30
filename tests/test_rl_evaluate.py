"""The RL cookbooks' shared evaluator (docs/examples/rl/common/evaluate.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "docs" / "examples" / "rl" / "common"
sys.path.insert(0, str(COMMON))
import evaluate  # noqa: E402
import positive_control  # noqa: E402


def _calls(message: dict) -> list[tuple[str, dict]]:
    return [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for c in evaluate.recover_tool_calls(message)
    ]


def test_recovers_tool_calls_servers_leave_in_the_wrong_channel() -> None:
    """Seen with Qwen3.5-9B and gpt-oss-20b through the HF router on 2026-09-30."""

    qwen35 = {
        "content": "",
        "reasoning_content": "Let me look.\n\n<tool_call>\n<function=run_bash>\n"
        "<parameter=command>\ncat /workdir/helpers.py\n</parameter>\n</function>\n</tool_call>",
    }
    gpt_oss = {"content": "", "reasoning_content": '{"command":"ls -R"}'}
    hermes = {
        "content": '<tool_call>\n{"name": "submit", "arguments": {"answer": "42\\nnorth"}}\n'
        "</tool_call>"
    }
    assert _calls(qwen35) == [("run_bash", {"command": "cat /workdir/helpers.py"})]
    assert _calls(gpt_oss) == [("run_bash", {"command": "ls -R"})]
    assert _calls(hermes) == [("submit", {"answer": "42\nnorth"})]


@pytest.mark.parametrize(
    "message",
    [
        {"content": "The answer is 42."},
        {
            "content": '<tool_call>\n{"name": "run_bash",. "arguments": {}}\n</tool_call>'
        },
        {"content": "", "reasoning_content": '{"path": "/workdir"}'},
        {"content": ""},
    ],
)
def test_does_not_invent_tool_calls(message: dict) -> None:
    """A malformed call is the policy's mistake, as it would be in TRL training."""

    assert evaluate.recover_tool_calls(message) == []


def test_wilson_interval_brackets_the_rate() -> None:
    low, high = evaluate.wilson(5, 10)
    assert low < 0.5 < high
    assert evaluate.wilson(0, 0) is None
    assert evaluate.wilson(10, 10)[1] == 1.0


def test_positive_control_policy_fingerprints_then_acts() -> None:
    """The scripted policy identifies a task by its files, never by its prompt."""

    table = {
        "abc": {
            "expected": {
                "type": "answers",
                "answers": [{"answer": "7"}, {"answer": "x"}],
            },
            "fix": "",
        },
        "def": {"expected": {"type": "bugfix"}, "fix": "sed -i s/a/b/ m.py"},
    }

    def act(messages: list[dict], mode: str = "oracle") -> tuple[str, dict]:
        call = positive_control._reply(table, mode, messages)["choices"][0]["message"][
            "tool_calls"
        ][0]
        return call["function"]["name"], json.loads(call["function"]["arguments"])

    prompt = [{"role": "user", "content": "task"}]
    assert act(prompt) == ("run_bash", {"command": positive_control.FINGERPRINT})
    assert act([*prompt, {"role": "tool", "content": "abc\n"}]) == (
        "submit",
        {"answer": "7\nx"},
    )
    fixing = [*prompt, {"role": "tool", "content": "def"}]
    assert act(fixing) == ("run_bash", {"command": "sed -i s/a/b/ m.py"})
    assert act([*fixing, {"role": "tool", "content": "ok"}]) == (
        "submit",
        {"answer": "done"},
    )
    assert act(prompt, mode="nothing") == ("submit", {"answer": ""})
