"""Guards FF-sw-other configured judge identity at 363d815b."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.test_sw_other_skill_judge import load_judge


@pytest.mark.parametrize(
    "model,provider",
    [
        ("gpt-4o-mini", "openai"),
        ("o4-mini", "openai"),
        ("o2-mini", "openai"),
        ("o42-mini", "openai"),
        ("openai/gpt-4o-mini", "openai"),
        ("gemini-2.5-flash", "gemini"),
        ("google/gemini-2.5-flash", "gemini"),
        ("claude-sonnet-4", "anthropic"),
        ("anthropic/claude-sonnet-4", "anthropic"),
    ],
)
@pytest.mark.parametrize("outage", [False, True])
def test_configured_model_exclusively_selects_provider(
    tmp_path, monkeypatch, model, provider, outage
):
    monkeypatch.setenv("JUDGE_MODEL", model)
    for name in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.setenv(name, "dummy")
    judge, *_ = load_judge(tmp_path, monkeypatch)
    calls = []

    def call(kind, **kwargs):
        calls.append((kind, kwargs["model"]))
        if outage:
            raise OSError("dummy outage")
        return SimpleNamespace(
            text="ok",
            content=[SimpleNamespace(text="ok")],
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )

    genai = ModuleType("google.genai")
    genai.Client = lambda **kw: SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kw: call("gemini", **kw))
    )
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    import google

    monkeypatch.setattr(google, "genai", genai, raising=False)
    for name in ("openai", "anthropic"):
        mod = ModuleType(name)
        if name == "openai":
            mod.OpenAI = lambda: SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **kw: call("openai", **kw)
                    )
                )
            )
        else:
            mod.Anthropic = lambda: SimpleNamespace(
                messages=SimpleNamespace(create=lambda **kw: call("anthropic", **kw))
            )
        monkeypatch.setitem(sys.modules, name, mod)
    if outage:
        with pytest.raises(judge["JudgeUnavailableError"]):
            judge["call_llm"]("dummy")
    else:
        assert judge["call_llm"]("dummy") == "ok"
    assert calls == [(provider, model.split("/", 1)[-1])]
