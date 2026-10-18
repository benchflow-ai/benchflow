"""The skill-eval judge scores only with the configured JUDGE_MODEL (#1154)."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from string import Template

import pytest

TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "benchflow"
    / "templates"
    / "judge.py.tmpl"
)


def load_judge(tmp_path, monkeypatch, judge_model):
    case = {"question": "q", "expected_behavior": ["uses the skill"]}
    (tmp_path / "case.json").write_text(json.dumps(case))
    path = tmp_path / "judge.py"
    path.write_text(Template(TEMPLATE.read_text()).safe_substitute(judge_model=judge_model))
    monkeypatch.setenv("BENCHFLOW_VERIFIER_DIR", str(tmp_path))
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    spec = importlib.util.spec_from_file_location("skill_eval_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def block(kind, text=None):
    return types.SimpleNamespace(type=kind, text=text) if text is not None else types.SimpleNamespace(type=kind)


def fake_anthropic(monkeypatch, content=None, error=None):
    calls = []

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs["model"])
            if error:
                raise error
            return types.SimpleNamespace(content=content)

    class Anthropic:
        messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Anthropic))
    return calls


def fake_openai(monkeypatch):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs["model"])
            message = types.SimpleNamespace(content='{"score": 0.0}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class OpenAI:
        chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    return calls


def fake_gemini(monkeypatch):
    calls = []

    class Models:
        def generate_content(self, model, contents):
            calls.append(model)
            return types.SimpleNamespace(text='{"score": 0.0}')

    class Client:
        def __init__(self, api_key):
            self.models = Models()

    google = types.ModuleType("google")
    google.genai = types.SimpleNamespace(Client=Client)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", google.genai)
    return calls


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("claude-sonnet-5", "anthropic"),
        ("gpt-4o-mini", "openai"),
        ("o3-mini", "openai"),
        ("gemini-3.1-flash-lite", "gemini"),
        ("anthropic/claude-sonnet-5", "anthropic"),
        ("openai/gpt-4o", "openai"),
        ("google/gemini-3.1-flash-lite", "gemini"),
        ("gemini/gemini-3.1-flash-lite", "gemini"),
    ],
)
def test_provider_follows_judge_model(tmp_path, monkeypatch, model, provider):
    judge = load_judge(tmp_path, monkeypatch, model)
    assert judge.judge_provider(model) == provider


def test_claude_judge_ignores_other_provider_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    gemini_calls = fake_gemini(monkeypatch)
    openai_calls = fake_openai(monkeypatch)
    anthropic_calls = fake_anthropic(monkeypatch, content=[block("text", '{"score": 1.0}')])
    judge = load_judge(tmp_path, monkeypatch, "claude-sonnet-5")

    assert judge.call_llm("p") == '{"score": 1.0}'
    assert anthropic_calls == ["claude-sonnet-5"]
    assert gemini_calls == [] and openai_calls == []


def test_thinking_block_before_text_is_skipped(tmp_path, monkeypatch):
    fake_anthropic(
        monkeypatch,
        content=[block("thinking"), block("text", '{"score": 0.5}')],
    )
    judge = load_judge(tmp_path, monkeypatch, "claude-sonnet-5-5")

    assert judge.call_llm("p") == '{"score": 0.5}'


def test_reply_without_text_is_a_judge_failure(tmp_path, monkeypatch):
    fake_anthropic(monkeypatch, content=[block("thinking")])
    judge = load_judge(tmp_path, monkeypatch, "claude-sonnet-5-5")

    with pytest.raises(judge.JudgeUnavailableError):
        judge.call_llm("p")


def test_failed_judge_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    openai_calls = fake_openai(monkeypatch)
    fake_anthropic(monkeypatch, error=RuntimeError("overloaded"))
    judge = load_judge(tmp_path, monkeypatch, "claude-sonnet-5")

    with pytest.raises(judge.JudgeUnavailableError, match="overloaded"):
        judge.call_llm("p")
    assert openai_calls == []


def test_judge_result_records_provider_and_model(tmp_path, monkeypatch):
    fake_anthropic(
        monkeypatch,
        content=[block("text", '{"items": [], "score": 0.75, "reasoning": "ok"}')],
    )
    judge = load_judge(tmp_path, monkeypatch, "claude-sonnet-5")

    assert judge.judge_with_llm("trajectory") == 0.75
    result = json.loads((tmp_path / "judge_result.json").read_text())
    assert result["judge_provider"] == "anthropic"
    assert result["judge_model"] == "claude-sonnet-5"


@pytest.mark.parametrize(
    ("judge_model", "fake", "sent_model"),
    [
        ("anthropic/claude-sonnet-5", "anthropic", "claude-sonnet-5"),
        ("openai/gpt-4o", "openai", "gpt-4o"),
        ("google/gemini-3.1-flash-lite", "gemini", "gemini-3.1-flash-lite"),
        ("gemini/gemini-3.1-flash-lite", "gemini", "gemini-3.1-flash-lite"),
    ],
)
def test_prefixed_judge_model_reaches_its_provider_bare(tmp_path, monkeypatch, judge_model, fake, sent_model):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    calls = {
        "anthropic": fake_anthropic(monkeypatch, content=[block("text", '{"score": 0.0}')]),
        "openai": fake_openai(monkeypatch),
        "gemini": fake_gemini(monkeypatch),
    }
    judge = load_judge(tmp_path, monkeypatch, judge_model)

    judge.call_llm("p")
    assert calls[fake] == [sent_model]
    assert all(c == [] for name, c in calls.items() if name != fake)
