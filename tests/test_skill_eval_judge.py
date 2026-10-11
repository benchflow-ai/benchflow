"""Run the generated skill-eval judge.py end to end against stub provider SDKs.

Guards the fix from PR #1156 for issue #1154: the judge must score with the
configured ``judge_model`` or not at all. Before it, a reply that opened with
a thinking block made ``response.content[0].text`` raise, and the judge then
took its score from Gemini or ``gpt-4o-mini``, whichever key happened to be
set.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from benchflow.skill_eval import generate_tasks, load_eval_dataset

_STUBS = {
    "anthropic/__init__.py": """
        import json, os

        if os.environ.get("STUB_NO_ANTHROPIC_SDK"):
            raise ImportError("No module named 'anthropic'")

        class _Obj:
            def __init__(self, fields):
                self.__dict__.update(fields)

        class Anthropic:
            def __init__(self):
                self.messages = self

            def create(self, model, **kwargs):
                with open(os.environ["STUB_LOG"], "a") as log:
                    log.write(json.dumps(["anthropic", model]) + "\\n")
                if os.environ.get("STUB_KWARGS_LOG"):
                    with open(os.environ["STUB_KWARGS_LOG"], "w") as out:
                        json.dump(kwargs, out)
                if os.environ.get("STUB_ANTHROPIC_RAISE"):
                    raise RuntimeError("overloaded")
                blocks = json.loads(os.environ["STUB_ANTHROPIC_CONTENT"])
                return _Obj({
                    "content": [_Obj(block) for block in blocks],
                    "stop_reason": os.environ.get("STUB_STOP_REASON", "end_turn"),
                })
    """,
    "openai/__init__.py": """
        import json, os
        from types import SimpleNamespace

        class OpenAI:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)

            def create(self, model, **kwargs):
                with open(os.environ["STUB_LOG"], "a") as log:
                    log.write(json.dumps(["openai", model]) + "\\n")
                if "max_tokens" in kwargs:
                    # What OpenAI's reasoning models answer (HTTP 400).
                    raise ValueError("Unsupported parameter: 'max_tokens'")
                message = SimpleNamespace(content='{"score": 0.25}')
                finish = os.environ.get("STUB_OPENAI_FINISH", "stop")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason=finish)]
                )
    """,
    "google/__init__.py": "",
    "google/genai/__init__.py": """
        import json, os
        from types import SimpleNamespace

        class Client:
            def __init__(self, api_key):
                self.models = self

            def generate_content(self, model, contents):
                with open(os.environ["STUB_LOG"], "a") as log:
                    log.write(json.dumps(["gemini", model]) + "\\n")
                finish = SimpleNamespace(name=os.environ.get("STUB_GEMINI_FINISH", "STOP"))
                return SimpleNamespace(
                    text=os.environ.get("STUB_GEMINI_TEXT", '{"score": 0.5}'),
                    candidates=[SimpleNamespace(finish_reason=finish)],
                )
    """,
}

ALL_KEYS = {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "GOOGLE_API_KEY": "g"}
THINKING_THEN_TEXT = [
    {"type": "thinking", "thinking": "Checking each rubric item."},
    {"type": "text", "text": '{"items": [], "score": 1.0, "reasoning": "ok"}'},
]


def _judge_dir(tmp_path, judge_model):
    skill = tmp_path / "calculator"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: calculator\ndescription: A calculator skill\n---\n# Calculator\n"
    )
    (skill / "evals" / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "calculator",
                "defaults": {"judge_model": judge_model},
                "cases": [
                    {
                        "id": "calc-001",
                        "question": "What is 2 + 2?",
                        "ground_truth": "4",
                        "expected_behavior": ["Agent reported 4"],
                    }
                ],
            }
        )
    )
    task_dir = generate_tasks(
        load_eval_dataset(skill), tmp_path / "tasks", with_skill=True
    )[0]
    return task_dir / "verifier"


def _run_judge(tmp_path, judge_model, keys, **stub_env):
    verifier = _judge_dir(tmp_path, judge_model)
    stubs = tmp_path / "stubs"
    for rel, source in _STUBS.items():
        (stubs / rel).parent.mkdir(parents=True, exist_ok=True)
        (stubs / rel).write_text(textwrap.dedent(source))
    logs = tmp_path / "agent-logs"
    logs.mkdir()
    (logs / "agent.txt").write_text("The agent ran calc.py and reported 4.")
    log = tmp_path / "calls.jsonl"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(stubs),
        "BENCHFLOW_VERIFIER_DIR": str(verifier),
        "BENCHFLOW_AGENT_LOG_DIR": str(logs),
        "BENCHFLOW_WORKSPACE": str(tmp_path / "app"),
        "STUB_LOG": str(log),
        **keys,
        **stub_env,
    }
    proc = subprocess.run(
        # -S skips site-packages, so only the stub SDKs are importable.
        [sys.executable, "-S", str(verifier / "judge.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return proc, calls, verifier


def test_thinking_block_before_text_is_scored_by_the_configured_claude(tmp_path):
    proc, calls, verifier = _run_judge(
        tmp_path,
        "claude-sonnet-5-5",
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps(THINKING_THEN_TEXT),
    )
    assert proc.returncode == 0, proc.stderr
    assert calls == [["anthropic", "claude-sonnet-5-5"]]
    assert (verifier / "reward.txt").read_text() == "1.0"
    result = json.loads((verifier / "judge_result.json").read_text())
    assert (result["judge_provider"], result["judge_model"]) == (
        "anthropic",
        "claude-sonnet-5-5",
    )


@pytest.mark.parametrize(
    "stub_env",
    [
        pytest.param({"STUB_ANTHROPIC_RAISE": "1"}, id="api-error"),
        pytest.param(
            {"STUB_ANTHROPIC_CONTENT": json.dumps(THINKING_THEN_TEXT[:1])},
            id="no-text-block",
        ),
        *(
            pytest.param(
                {
                    "STUB_ANTHROPIC_CONTENT": json.dumps(
                        [{"type": "text", "text": '{"items": [], "score": 0.9}'}]
                    ),
                    "STUB_STOP_REASON": reason,
                },
                id=reason,
            )
            for reason in ("max_tokens", "model_context_window_exceeded", "refusal")
        ),
    ],
)
def test_failed_claude_judge_is_a_verifier_error_not_another_models_score(
    tmp_path, stub_env
):
    proc, calls, verifier = _run_judge(
        tmp_path, "claude-sonnet-5-5", ALL_KEYS, **stub_env
    )
    assert proc.returncode == 1
    assert "judge could not run" in proc.stderr
    assert calls == [["anthropic", "claude-sonnet-5-5"]]
    assert not (verifier / "reward.txt").exists()


def test_gemini_judge_without_a_google_key_does_not_fall_back_to_openai(tmp_path):
    proc, calls, verifier = _run_judge(
        tmp_path,
        "gemini-3.1-flash-lite",
        {"OPENAI_API_KEY": "o", "ANTHROPIC_API_KEY": "a"},
    )
    assert proc.returncode == 1
    assert "GOOGLE_API_KEY or GEMINI_API_KEY is not set" in proc.stderr
    assert calls == []
    assert not (verifier / "reward.txt").exists()


@pytest.mark.parametrize(
    ("judge_model", "provider", "reward"),
    [("gemini-3.1-flash-lite", "gemini", "0.5"), ("gpt-5.5", "openai", "0.25")],
)
def test_each_provider_answers_only_for_its_own_models(
    tmp_path, judge_model, provider, reward
):
    proc, calls, verifier = _run_judge(tmp_path, judge_model, ALL_KEYS)
    assert proc.returncode == 0, proc.stderr
    assert calls == [[provider, judge_model]]
    assert (verifier / "reward.txt").read_text() == reward
    result = json.loads((verifier / "judge_result.json").read_text())
    assert (result["judge_provider"], result["judge_model"]) == (provider, judge_model)


@pytest.mark.parametrize(
    ("judge_model", "provider", "api_model"),
    [
        ("o4-mini", "openai", "o4-mini"),
        ("openai/gpt-5.5", "openai", "gpt-5.5"),
        ("google/gemini-3.1-flash-lite", "gemini", "gemini-3.1-flash-lite"),
        ("gemini/gemini-3.1-flash-lite", "gemini", "gemini-3.1-flash-lite"),
        ("anthropic/claude-sonnet-5-5", "anthropic", "claude-sonnet-5-5"),
    ],
)
def test_provider_prefixes_route_like_rewards_llm(
    tmp_path, judge_model, provider, api_model
):
    proc, calls, verifier = _run_judge(
        tmp_path,
        judge_model,
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps(THINKING_THEN_TEXT),
    )
    assert proc.returncode == 0, proc.stderr
    assert calls == [[provider, api_model]]
    result = json.loads((verifier / "judge_result.json").read_text())
    assert (result["judge_provider"], result["judge_model"]) == (provider, judge_model)


def test_unrecognised_judge_model_calls_no_provider(tmp_path):
    proc, calls, verifier = _run_judge(tmp_path, "deepseek-v4-flash", ALL_KEYS)
    assert proc.returncode == 1
    assert "names no supported provider" in proc.stderr
    assert calls == []
    assert not (verifier / "reward.txt").exists()


def test_judge_model_env_override_picks_the_provider(tmp_path):
    proc, calls, _ = _run_judge(
        tmp_path, "claude-sonnet-5-5", ALL_KEYS, JUDGE_MODEL="gemini-3.1-flash-lite"
    )
    assert proc.returncode == 0, proc.stderr
    assert calls == [["gemini", "gemini-3.1-flash-lite"]]


@pytest.mark.parametrize(
    ("judge_model", "stub_env", "message"),
    [
        pytest.param(
            "claude-sonnet-5-5",
            {"STUB_NO_ANTHROPIC_SDK": "1"},
            "ImportError",
            id="missing-sdk",
        ),
        pytest.param(
            "gpt-5.5",
            {"STUB_OPENAI_FINISH": "length"},
            "did not finish normally: length",
            id="openai-length",
        ),
        pytest.param(
            "claude-sonnet-5-5",
            {
                "STUB_ANTHROPIC_CONTENT": json.dumps(
                    [{"type": "text", "text": "I cannot grade this trajectory."}]
                )
            },
            "has 0 JSON objects, not one verdict",
            id="no-score",
        ),
    ],
)
def test_judge_without_a_verdict_is_a_verifier_error(
    tmp_path, judge_model, stub_env, message
):
    proc, _, verifier = _run_judge(tmp_path, judge_model, ALL_KEYS, **stub_env)
    assert proc.returncode == 1
    assert message in proc.stderr
    assert not (verifier / "reward.txt").exists()


def test_fenced_json_reply_is_scored_with_provenance(tmp_path):
    fenced = '```json\n{"items": [], "score": 0.75, "reasoning": "ok"}\n```'
    proc, calls, verifier = _run_judge(
        tmp_path,
        "claude-sonnet-5-5",
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps([{"type": "text", "text": fenced}]),
    )
    assert proc.returncode == 0, proc.stderr
    assert calls == [["anthropic", "claude-sonnet-5-5"]]
    assert (verifier / "reward.txt").read_text() == "0.75"
    result = json.loads((verifier / "judge_result.json").read_text())
    assert result["score"] == 0.75
    assert (result["judge_provider"], result["judge_model"]) == (
        "anthropic",
        "claude-sonnet-5-5",
    )


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param('{"items": [], "score": NaN}', id="nan"),
        pytest.param('{"items": [], "score": Infinity}', id="infinity"),
        # The score is valid, so only the NaN rejection in the parser fails it.
        pytest.param('{"items": [NaN], "score": 0.5}', id="nan-outside-the-score"),
        pytest.param(
            '{"items": [1e999], "score": 0.5}', id="overflow-outside-the-score"
        ),
        pytest.param('{"items": [], "score": 85}', id="out-of-range"),
        pytest.param('{"items": [], "score": -0.5}', id="negative"),
        pytest.param('{"items": [], "score": true}', id="bool"),
        pytest.param('{"items": [], "score": "0.9"}', id="string"),
        pytest.param("[0.3]", id="not-an-object"),
        pytest.param(
            '{"items": [{"criterion": "c", "score": 1.0}], "score": null}',
            id="null-score-with-a-nested-score",
        ),
    ],
)
def test_a_score_outside_0_to_1_is_a_verifier_error_not_clamped(tmp_path, reply):
    proc, _, verifier = _run_judge(
        tmp_path,
        "claude-sonnet-5-5",
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps([{"type": "text", "text": reply}]),
    )
    assert proc.returncode == 1
    assert "judge could not run" in proc.stderr
    assert not (verifier / "reward.txt").exists()


def test_the_fenced_verdict_wins_over_a_score_mentioned_in_prose(tmp_path):
    reply = 'Each "score": 0 means fail.\n```json\n{"items": [], "score": 0.8}\n```'
    proc, _, verifier = _run_judge(
        tmp_path,
        "claude-sonnet-5-5",
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps([{"type": "text", "text": reply}]),
    )
    assert proc.returncode == 0, proc.stderr
    assert (verifier / "reward.txt").read_text() == "0.8"


def test_gemini_reply_cut_off_at_the_token_limit_is_a_verifier_error(tmp_path):
    proc, _, verifier = _run_judge(
        tmp_path,
        "gemini-3.1-flash-lite",
        ALL_KEYS,
        STUB_GEMINI_FINISH="MAX_TOKENS",
        STUB_GEMINI_TEXT='{"items": [], "score": 0.85, "reasoning": "trunc',
    )
    assert proc.returncode == 1
    assert "did not finish normally: MAX_TOKENS" in proc.stderr
    assert not (verifier / "reward.txt").exists()


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(
            'Example: {"score": 1}. Actual verdict:\n```json\n'
            '{"score": 0, "reasoning": "unfinished\n```',
            id="malformed-fence-after-an-example",
        ),
        pytest.param(
            'Example: {"score": 1}. Actual verdict: {"score": 0}', id="two-objects"
        ),
        pytest.param(
            'Here: {"items": [{"criterion": "c", "score": 1.0}], "score": 0.2, "rea',
            id="truncated-verdict-with-a-scored-item",
        ),
        pytest.param(
            '```json\n{"score": 1}\n```\n```json\n{"score": 0}\n```',
            id="two-json-fences",
        ),
        pytest.param(
            '{"score": 1}\n```json\n{"score": 0.2}\n```', id="an-object-and-a-fence"
        ),
        pytest.param(
            'Example: {"score": 1}. Actual verdict: {"score": 0, "rea',
            id="truncated-verdict-after-an-example",
        ),
        pytest.param('Example: {"score": 1}. Verdict: {', id="reply-ends-at-a-brace"),
        pytest.param(
            'Example: {"score": 1}. Verdict: {"reasoning"}',
            id="keyless-object-after-an-example",
        ),
        pytest.param("[" * 100_000, id="only-unclosed-brackets"),
        pytest.param(
            'Format: {"score": 1}\nMy verdict: [{"score": 0.2}]',
            id="verdict-in-an-array-beside-an-example",
        ),
        pytest.param(
            'Format {"score": 1}\nVerdict: {score: 0}',
            id="unquoted-keys-verdict-after-an-example",
        ),
        pytest.param(
            'Example: {"score": 1}\nVerdict: [{"score": 0}, oops]',
            id="broken-array-after-an-example",
        ),
        pytest.param('Verdict: {"score": 0, "score": 1}', id="duplicate-score-keys"),
        pytest.param(
            'The agent typed "[" here. {"score": 0.1} Reply format: "]" {"score": 1}',
            id="quoted-brackets-between-two-objects",
        ),
        pytest.param(
            '[see "note] {"score": 0, "r": "a"} "] {"score": 1}',
            id="quoted-bracket-run-hiding-an-object",
        ),
        pytest.param(
            '```python``` was misused.\n{"score": 0}\nFormat:\n```\n{"score": 1}\n```',
            id="inline-backticks-at-a-line-start",
        ),
        pytest.param(
            '```python\nfoo()\n{"score": 0}\nFormat:\n```\n{"score": 1}\n```',
            id="unclosed-fence-before-the-verdict",
        ),
        pytest.param(
            'Example:\n```python\nd = {"score": 1}\n```\n{"score": 0}',
            id="dict-in-a-code-fence-beside-the-verdict",
        ),
        pytest.param(
            "Format: {\"score\": 1}. My verdict: {'score': 0}",
            id="non-json-verdict-after-an-example",
        ),
        pytest.param(
            'The agent left an unterminated ``` in README.\n{"score": 0}\n'
            'Expected format:\n```json\n{"score": 1}\n```',
            id="stray-backticks-before-the-verdict",
        ),
    ],
)
def test_an_ambiguous_or_broken_verdict_is_a_verifier_error(tmp_path, reply):
    """Each of these used to yield a reward taken from the wrong object."""
    proc, _, verifier = _run_judge(
        tmp_path,
        "claude-sonnet-5-5",
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps([{"type": "text", "text": reply}]),
    )
    assert proc.returncode == 1, proc.stdout
    assert "judge could not run" in proc.stderr
    assert not (verifier / "reward.txt").exists()


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(
            'Verdict: {"score": 0.8, "reasoning": "literal }"}',
            id="close-brace-in-string",
        ),
        pytest.param(
            'Verdict: {"score": 0.8, "reasoning": "literal {"}',
            id="open-brace-in-string",
        ),
        pytest.param('```\n{"score": 0.8}\n```', id="untagged-fence"),
        pytest.param(
            'Done.\n```python\nprint(1)\n```\n```json\n{"score": 0.8}\n```',
            id="json-fence-beside-a-code-fence",
        ),
        pytest.param(
            '{"items": [], "score": 0.8}\nLet me know if you need more.',
            id="verdict-then-prose",
        ),
        pytest.param('[PASS] item 1\n{"score": 0.8}', id="item-list-then-verdict"),
        pytest.param(
            'The agent used the f-string {name}. Verdict: {"score": 0.8}',
            id="prose-braces-before-the-verdict",
        ),
        pytest.param(
            'It returned an empty dict {} on error.\n{"score": 0.8}',
            id="empty-braces-before-the-verdict",
        ),
        pytest.param('Done, {"score": 0.8}', id="comma-before-the-verdict"),
        pytest.param(
            '```json\n{"score": 0.8, "reasoning": "it used ```py fences"}\n```',
            id="backticks-inside-a-fenced-verdict",
        ),
        pytest.param(
            '```json title="verdict"\n{"score": 0.8}\n```', id="fence-info-string"
        ),
        pytest.param(
            '[x] item 1 [ ] item 2 [1/3] passed\n{"score": 0.8}',
            id="checklist-brackets-before-the-verdict",
        ),
        pytest.param(
            '{"score": 0.8}\nInterval [0.4, 0.6) per [1], see [[link]].',
            id="intervals-and-citations-after-the-verdict",
        ),
        pytest.param('Result: [{"score": 0.8}]', id="lone-verdict-inside-an-array"),
        pytest.param('[the 5" screen] {"score": 0.8}', id="inch-mark-in-brackets"),
        pytest.param(
            '[the "best ]" one] {"score": 0.8}', id="bracket-inside-a-quoted-string"
        ),
        pytest.param(
            'The template [uses {name}] placeholders.\n{"score": 0.8}',
            id="prose-braces-inside-prose-brackets",
        ),
        pytest.param(
            "[" * 5000 + "]" * 5000 + '\n{"score": 0.8}',
            id="deep-but-valid-array-before-the-verdict",
        ),
        pytest.param(
            'The agent ran:\n```\nls -la\n```\n{"score": 0.8}',
            id="untagged-command-fence-before-the-verdict",
        ),
    ],
)
def test_one_clear_verdict_is_read_even_with_braces_in_strings(tmp_path, reply):
    proc, _, verifier = _run_judge(
        tmp_path,
        "claude-sonnet-5-5",
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps([{"type": "text", "text": reply}]),
    )
    assert proc.returncode == 0, proc.stderr
    assert (verifier / "reward.txt").read_text() == "0.8"


def test_every_anthropic_text_block_is_joined(tmp_path):
    blocks = [
        {"type": "thinking", "thinking": "hmm"},
        {"type": "text", "text": '{"items": [], '},
        {"type": "text", "text": '"score": 0.6}'},
    ]
    proc, _, verifier = _run_judge(
        tmp_path,
        "claude-sonnet-5-5",
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps(blocks),
    )
    assert proc.returncode == 0, proc.stderr
    assert (verifier / "reward.txt").read_text() == "0.6"


@pytest.mark.parametrize(
    ("judge_model", "stub_env"),
    [
        pytest.param(
            "gpt-5.5",
            {"STUB_OPENAI_FINISH": "content_filter"},
            id="openai-content-filter",
        ),
        pytest.param(
            "gemini-3.1-flash-lite",
            {"STUB_GEMINI_FINISH": "SAFETY"},
            id="gemini-safety",
        ),
    ],
)
def test_a_reply_that_did_not_finish_normally_is_a_verifier_error(
    tmp_path, judge_model, stub_env
):
    """The stub still returns a valid score, so only the finish check can fail it."""
    proc, _, verifier = _run_judge(tmp_path, judge_model, ALL_KEYS, **stub_env)
    assert proc.returncode == 1
    assert "did not finish normally" in proc.stderr
    assert not (verifier / "reward.txt").exists()


def test_claude_judge_leaves_room_for_thinking(tmp_path):
    """Thinking counts against max_tokens, so 2048 could cut off the verdict.

    4096 is the largest output budget every Claude model accepts.
    """
    kwargs_log = tmp_path / "kwargs.json"
    proc, _, _ = _run_judge(
        tmp_path,
        "claude-sonnet-5-5",
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps([{"type": "text", "text": '{"score": 1}'}]),
        STUB_KWARGS_LOG=str(kwargs_log),
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(kwargs_log.read_text())["max_tokens"] == 4096


def test_a_malformed_fenced_verdict_beside_an_example_is_not_scored(tmp_path):
    """Skipping the broken object would leave the example as the only verdict."""
    reply = 'Example: {"score": 1}. Actual verdict:\n```json\n{"score": 0,\n```'
    proc, _, verifier = _run_judge(
        tmp_path,
        "claude-sonnet-5-5",
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps([{"type": "text", "text": reply}]),
    )
    assert proc.returncode == 1
    assert "malformed or truncated JSON object" in proc.stderr
    assert not (verifier / "reward.txt").exists()


def test_deeply_nested_json_is_a_verifier_error_not_a_traceback(tmp_path):
    reply = '{"a": ' + "[" * 100_000  # the JSON decoder raises RecursionError
    proc, _, verifier = _run_judge(
        tmp_path,
        "claude-sonnet-5-5",
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps([{"type": "text", "text": reply}]),
    )
    assert proc.returncode == 1
    assert "malformed or truncated JSON object" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not (verifier / "reward.txt").exists()


def test_a_verdict_too_deep_to_record_is_a_verifier_error(tmp_path):
    """json.dumps recurses in Python, so it fails on nesting the decoder accepts."""
    reply = '{"score": 0.5, "x": ' + "[" * 1500 + "]" * 1500 + "}"
    proc, _, verifier = _run_judge(
        tmp_path,
        "claude-sonnet-5-5",
        ALL_KEYS,
        STUB_ANTHROPIC_CONTENT=json.dumps([{"type": "text", "text": reply}]),
    )
    assert proc.returncode == 1
    assert "nests too deeply to record" in proc.stderr
    assert not (verifier / "reward.txt").exists()
