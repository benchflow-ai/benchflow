import importlib.util
import json
from pathlib import Path


def _load_publish_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "skillsbench-fill"
        / "publish.py"
    )
    spec = importlib.util.spec_from_file_location("skillsbench_fill_publish", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safe_bytes_preserves_token_usage_counters(tmp_path):
    """Guards Hugging Face PR #4 token-usage recovery against scrub redaction."""
    publish = _load_publish_module()
    result = {
        "agent_result": {
            "n_input_tokens": 123,
            "n_output_tokens": 45,
            "n_cache_read_tokens": 678,
            "n_cache_creation_tokens": 90,
            "total_tokens": 936,
            "usage_source": "provider_response",
        },
        "final_metrics": {
            "total_prompt_tokens": 123,
            "total_completion_tokens": 45,
            "total_cached_tokens": 678,
            "total_cost_usd": 1.23,
        },
        "HUGGING_FACE_TOKEN": "hf_abcdefghijklmnopqrstuvwxyz",
    }
    src = tmp_path / "result.json"
    src.write_text(json.dumps(result))

    scrubbed = json.loads(publish.safe_bytes(src))

    assert scrubbed["agent_result"]["n_input_tokens"] == 123
    assert scrubbed["agent_result"]["n_output_tokens"] == 45
    assert scrubbed["agent_result"]["n_cache_read_tokens"] == 678
    assert scrubbed["agent_result"]["n_cache_creation_tokens"] == 90
    assert scrubbed["agent_result"]["total_tokens"] == 936
    assert scrubbed["final_metrics"]["total_prompt_tokens"] == 123
    assert scrubbed["final_metrics"]["total_completion_tokens"] == 45
    assert scrubbed["final_metrics"]["total_cached_tokens"] == 678
    assert scrubbed["HUGGING_FACE_TOKEN"] == "[REDACTED]"


def test_safe_bytes_normalizes_config_without_leaking_secret_tokens(tmp_path):
    """Guards Hugging Face PR #4 token-usage recovery against credential leakage."""
    publish = _load_publish_module()
    config = {
        "include_task_skills": False,
        "agent_env": {
            "AWS_BEARER_TOKEN_BEDROCK": "Bearer abcdefghijklmnop",
            "OPENAI_API_KEY": "sk-abcdefghijklmnopqrstuvwxyz",
            "BENCHFLOW_MODEL_MAX_TOKENS": 8192,
        },
    }
    src = tmp_path / "config.json"
    src.write_text(json.dumps(config))

    scrubbed = json.loads(publish.safe_bytes(src, is_config=True, mode="with"))

    assert scrubbed["include_task_skills"] is True
    assert scrubbed["agent_env"]["AWS_BEARER_TOKEN_BEDROCK"] == "[REDACTED]"
    assert scrubbed["agent_env"]["OPENAI_API_KEY"] == "[REDACTED]"
    assert scrubbed["agent_env"]["BENCHFLOW_MODEL_MAX_TOKENS"] == 8192
