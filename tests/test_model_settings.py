"""The model vendor's recommended sampling on the run card (benchflow.model_settings)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.model_settings import vendor_setting
from benchflow.run_card import build_run_card, card_lines


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3.8-27B",
        "hosted_vllm/Qwen/Qwen3.8-27B",
        "openai/qwen3.8-27b-fp8",
        "qwen3.8-27b",
    ],
)
def test_a_qwen3_8_27b_id_finds_its_own_model_card(model):
    setting = vendor_setting(model)
    assert setting is not None and setting.covers == "Qwen3.8-27B"
    # The Qwen3.8 card differs from Qwen3's: temperature 1.0 with thinking on,
    # and a presence penalty with it off.
    assert setting.thinking["temperature"] == 1.0
    assert setting.non_thinking == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
    }


@pytest.mark.parametrize(
    "model", ["Qwen/Qwen3-32B", "openrouter/qwen/qwen3-30b-a3b", "qwen3-0.6b"]
)
def test_a_2025_qwen3_id_finds_the_qwen3_card(model):
    setting = vendor_setting(model)
    assert setting is not None and setting.source.endswith("Qwen3-32B")
    assert setting.thinking["temperature"] == 0.6


@pytest.mark.parametrize(
    "model",
    [
        None,
        "",
        "deepseek/deepseek-v4",
        "claude-haiku-4-5-20251001",
        # The 2507 Instruct models have no thinking mode; Qwen3-Coder has its
        # own setting; neither is on file.
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "qwen/qwen3-coder-480b-a35b",
        "Qwen/Qwen3.8-270B",
    ],
)
def test_a_model_with_no_entry_has_no_setting(model):
    assert vendor_setting(model) is None


def _card(tmp_path: Path, model: str) -> list[str]:
    task = tmp_path / "tasks" / "t"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n[agent]\ntimeout_sec = 600\n')
    (task / "instruction.md").write_text("Write 42 to /app/answer.txt.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\n")
    job = Evaluation(
        tasks_dir=str(tmp_path / "tasks"),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="claude-agent-acp", model=model),
        preflight=False,
    )
    card = build_run_card(job, job._get_task_dirs())
    return card, card_lines(card, tmp_path / "job")


def test_the_card_puts_the_vendor_setting_beside_the_agent_s_own_sampling(tmp_path):
    card, lines = _card(tmp_path, "hosted_vllm/Qwen/Qwen3.8-27B")
    assert card["model"]["vendor_setting"]["covers"] == "Qwen3.8-27B"
    assert (
        "  Vendor:    Qwen recommends for Qwen3.8-27B: thinking on, its default, "
        "temperature 1, top_p 0.95, top_k 20, min_p 0, presence_penalty 0; with "
        "thinking off, temperature 0.7, top_p 0.8, top_k 20, min_p 0, "
        "presence_penalty 1.5 (https://huggingface.co/Qwen/Qwen3.8-27B, read "
        "2026-11-04): set the agent to match, or say why"
    ) in lines
    sampling = next(line for line in lines if line.startswith("  Sampling:"))
    assert "trajectory/llm_trajectory.jsonl" in sampling
    # The platform lead's finding: BenchFlow does set Claude Code's output cap.
    assert (
        "caps Claude Code's output at 128000 tokens (CLAUDE_CODE_MAX_OUTPUT_TOKENS)"
        in sampling
    )


def test_a_model_with_no_entry_says_to_read_its_model_card(tmp_path):
    card, lines = _card(tmp_path, "deepseek/deepseek-v4")
    assert card["model"]["vendor_setting"] is None
    assert (
        "  Vendor:    no recommended setting on file for deepseek/deepseek-v4: "
        "read its model card before you compare"
    ) in lines
