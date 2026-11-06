"""Defaults and shipped examples name models their vendors still serve.

Each id in RETIRED was checked on its vendor's own page on 2026-11-04:

- OpenAI (learn.chatgpt.com/docs/models): gpt-5.5 left Codex's ChatGPT sign-in
  on 2026-10-14, and gpt-5.4 and gpt-5.4-mini on 2026-08-31.
- Google (aistudio.google.com/docs/models): gemini-3.1-flash-lite-preview is
  shut down; gemini-2.5-flash is deprecated and closed to new API keys.
- Anthropic (platform.claude.com, model deprecations): the retired Claude ids,
  and claude-sonnet-4-5-20250929, deprecated and retiring on 2026-11-30.

When a vendor retires a model, add its id here: the test then names every
default and example that still uses it.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from benchflow.agents.registry import AGENTS
from benchflow.cli.review import DEFAULT_REVIEW_MODEL
from benchflow.doctor import HEADROOM_MODEL
from benchflow.doctor_smoke import SMOKE_DEFAULT_MODELS
from benchflow.evaluation import DEFAULT_MODEL
from benchflow.rewards.builtins import LLMJudgeRewardFunc
from benchflow.rewards.rubric_config import JudgeConfig
from benchflow.task.config import JudgeVerifierConfig

RETIRED = {
    "gpt-5.5": "left Codex's ChatGPT sign-in on 2026-10-14",
    "gpt-5.4": "left Codex's ChatGPT sign-in on 2026-08-31",
    "gpt-5.4-mini": "left Codex's ChatGPT sign-in on 2026-08-31",
    "gemini-3.1-flash-lite-preview": "shut down by Google",
    "gemini-2.5-flash": "deprecated by Google and closed to new API keys",
    "claude-sonnet-4-5-20250929": "deprecated; Anthropic retires it on 2026-11-30",
    "claude-opus-4-1-20250805": "retired by Anthropic on 2026-08-05",
    "claude-opus-4-20250514": "retired by Anthropic on 2026-06-15",
    "claude-sonnet-4-20250514": "retired by Anthropic on 2026-06-15",
    "claude-3-7-sonnet-20250219": "retired by Anthropic on 2026-02-19",
    "claude-3-5-haiku-20241022": "retired by Anthropic on 2026-02-19",
    "claude-3-haiku-20240307": "retired by Anthropic on 2026-04-20",
}

REPO = Path(__file__).resolve().parents[1]

# Run configs a user copies. Parity records and benchmark metadata describe
# what ran at the time and keep the model it ran with.
EXAMPLE_CONFIGS = [
    "benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml",
    "benchmarks/opaquetoolsbench/opaquetoolsbench-gemini-flash-lite.yaml",
    "benchmarks/programbench/programbench-gemini-flash-lite.yaml",
]


def _bare(model: str) -> str:
    """``gemini/gemini-2.5-flash`` and ``gemini-2.5-flash`` are the same model."""
    return model.rsplit("/", 1)[-1]


def _defaults() -> dict[str, str]:
    found = {
        "evaluation.DEFAULT_MODEL": DEFAULT_MODEL,
        "doctor.HEADROOM_MODEL": HEADROOM_MODEL,
        "cli.review.DEFAULT_REVIEW_MODEL": DEFAULT_REVIEW_MODEL,
        "rubric_config.JudgeConfig.model": JudgeConfig().model,
        "task.config.JudgeVerifierConfig.model": JudgeVerifierConfig.model_fields[
            "model"
        ].default,
        "rewards.builtins.LLMJudgeRewardFunc model": inspect.signature(
            LLMJudgeRewardFunc.__init__
        )
        .parameters["model"]
        .default,
    }
    for agent, model in SMOKE_DEFAULT_MODELS.items():
        found[f"doctor_smoke.SMOKE_DEFAULT_MODELS[{agent}]"] = model
    for name, config in AGENTS.items():
        if config.default_model:
            found[f"agents.registry.AGENTS[{name}].default_model"] = (
                config.default_model
            )
    return found


def _example_models() -> dict[str, str]:
    found: dict[str, str] = {}
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for n, match in enumerate(re.finditer(r"(?:--model\s+|model:\s*)(\S+)", readme)):
        found[f"README.md model #{n + 1}"] = match.group(1).strip("'\"")
    for rel in EXAMPLE_CONFIGS:
        text = (REPO / rel).read_text(encoding="utf-8")
        for match in re.finditer(r"^\s*model:\s*(\S+)", text, re.MULTILINE):
            found[rel] = match.group(1).strip("'\"")
    return found


@pytest.mark.parametrize(
    "where,model",
    sorted({**_defaults(), **_example_models()}.items()),
)
def test_no_default_or_example_names_a_retired_model(where: str, model: str) -> None:
    reason = RETIRED.get(_bare(model))
    assert reason is None, f"{where} is {model!r}, which was {reason}"


def test_the_readme_and_example_configs_were_scanned() -> None:
    examples = _example_models()
    assert any(k.startswith("README.md") for k in examples)
    assert set(EXAMPLE_CONFIGS) <= set(examples)
