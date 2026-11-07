"""Guards PR #54 fixes for issues #897 and #1154 (G30)."""

import sys
from types import SimpleNamespace

import pytest

from tests.test_sw_other_skill_judge import load_judge


@pytest.mark.parametrize(
    "answer,reply,score",
    [
        ("getPageCount()", "Use getPageCount().", 1.0),
        ("(42)", "Answer: (42)", 1.0),
        ("C++", "C++", 1.0),
        ("42", "142", 0.0),
        ("42", "42", 1.0),
        ("getPageCount()", "forgetPageCount()", 0.0),
    ],
)
def test_punctuation_answer_boundaries(tmp_path, monkeypatch, answer, reply, score):
    judge, logs, *_ = load_judge(tmp_path, monkeypatch)
    judge["CASE"]["ground_truth"] = answer
    (logs / "output.txt").write_text(reply)
    assert judge["judge_exact_match"]() == score


@pytest.mark.parametrize("texts", [['{"score":', " 1}"], []])
def test_thinking_blocks_do_not_hide_verdict(tmp_path, monkeypatch, texts):
    monkeypatch.setenv("JUDGE_MODEL", "anthropic/claude-sonnet-4")
    judge, *_ = load_judge(tmp_path, monkeypatch)
    content = [SimpleNamespace(type="thinking", thinking="private reasoning")]
    content += [SimpleNamespace(type="text", text=text) for text in texts]
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(
            Anthropic=lambda: SimpleNamespace(
                messages=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(content=content)
                )
            )
        ),
    )
    if texts:
        assert judge["call_llm"]("prompt") == '{"score": 1}'
    else:
        with pytest.raises(judge["JudgeUnavailableError"]):
            judge["call_llm"]("prompt")


def test_generated_verifier_writes_punctuation_reward(tmp_path, monkeypatch):
    import json
    import subprocess

    _, logs, _, verifier = load_judge(tmp_path, monkeypatch)
    (verifier / "case.json").write_text(json.dumps({"ground_truth": "getPageCount()"}))
    (logs / "output.txt").write_text("Call getPageCount().")
    completed = subprocess.run(
        [sys.executable, str(tmp_path / "judge.py")], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads((verifier / "reward.json").read_text()) == {"reward": 1.0}
