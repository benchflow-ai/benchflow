"""Standalone OpenHands settings-writer source embedded into sandboxes."""

OPENHANDS_SETTINGS_WRITER = r"""import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def optional_int(name):
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def optional_non_negative_int(name):
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    if not value.isdigit():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def optional_bool(name):
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def disable_subagents_if_requested():
    if os.environ.get("BENCHFLOW_OPENHANDS_DISABLE_SUBAGENTS", "0") != "1":
        return

    openhands_bin = shutil.which("openhands")
    if not openhands_bin:
        raise RuntimeError("Cannot locate OpenHands executable")
    openhands_python = Path(os.path.realpath(openhands_bin)).parent / "python"
    if not openhands_python.is_file() or not os.access(openhands_python, os.X_OK):
        raise RuntimeError("Cannot locate OpenHands tool interpreter")

    subprocess.run(
        [
            str(openhands_python),
            "-c",
            "from pathlib import Path\n"
            "import openhands_cli.utils as u\n"
            "p = Path(u.__file__)\n"
            "s = p.read_text()\n"
            "old = '        Tool(name=task_tool_name),\\n'\n"
            "new = '        # BenchFlow: delegation disabled for this run.\\n'\n"
            "assert old in s or new in s\n"
            "p.write_text(s.replace(old, new, 1))\n",
        ],
        check=True,
    )


llm = {
    "model": os.environ["LLM_MODEL"],
    "api_key": os.environ["LLM_API_KEY"],
    "usage_id": "agent",
}
for env_name, field_name in (
    ("LLM_BASE_URL", "base_url"),
    ("LLM_API_VERSION", "api_version"),
):
    value = os.environ.get(env_name, "").strip()
    if value:
        llm[field_name] = value

for env_name, field_name in (
    ("LLM_NATIVE_TOOL_CALLING", "native_tool_calling"),
    ("LLM_CACHING_PROMPT", "caching_prompt"),
    ("LLM_DROP_PARAMS", "drop_params"),
    ("LLM_MODIFY_PARAMS", "modify_params"),
):
    value = optional_bool(env_name)
    if value is not None:
        llm[field_name] = value

timeout = optional_non_negative_int("LLM_TIMEOUT")
if timeout is not None:
    llm["timeout"] = timeout

reasoning_effort = os.environ.get("LLM_REASONING_EFFORT", "").strip()
if reasoning_effort == "max":
    llm["litellm_extra_body"] = {"reasoning": {"effort": "max"}}
elif reasoning_effort in {"none", "low", "medium", "high", "xhigh"}:
    llm["reasoning_effort"] = reasoning_effort
    llm["litellm_extra_body"] = {"reasoning_effort": reasoning_effort}
elif reasoning_effort:
    llm["litellm_extra_body"] = {"reasoning_effort": reasoning_effort}

context_limit = optional_int("BENCHFLOW_OPENHANDS_CONTEXT_LIMIT")
output_limit = optional_int("BENCHFLOW_OPENHANDS_OUTPUT_LIMIT")
if context_limit is not None:
    llm["max_input_tokens"] = context_limit
if output_limit is not None:
    llm["max_output_tokens"] = output_limit

condenser = {
    "llm": {**llm, "usage_id": "condenser"},
    "max_size": 80,
    "keep_first": 4,
    "kind": "LLMSummarizingCondenser",
}
if context_limit is not None and output_limit is not None:
    reserve = optional_int("BENCHFLOW_OPENHANDS_CONTEXT_RESERVE") or 4096
    condenser_limit = context_limit - output_limit - reserve
    if condenser_limit <= 0:
        raise ValueError("OpenHands context budget leaves no room for input")
    condenser["max_tokens"] = condenser_limit

settings = {
    "llm": llm,
    "tools": [],
    "condenser": condenser,
    "kind": "Agent",
}
Path(sys.argv[1]).write_text(json.dumps(settings, separators=(",", ":")))
disable_subagents_if_requested()
"""
