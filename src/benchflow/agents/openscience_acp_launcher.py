#!/usr/bin/env python3
"""Prepare an isolated OpenScience configuration and exec its native ACP server."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"openscience launcher requires {name}")
    return value


def _provider_adapter(protocol: str) -> tuple[str, str]:
    """Return the OpenScience AI SDK package and a fallback provider ID."""
    if protocol in ("", "openai-completions"):
        return "@ai-sdk/openai-compatible", "deepseek"
    if protocol == "anthropic-messages":
        return "@ai-sdk/anthropic", "anthropic"
    raise SystemExit(
        "openscience launcher: unsupported BENCHFLOW_PROVIDER_PROTOCOL "
        f"{protocol!r}; supported values are openai-completions and "
        "anthropic-messages"
    )


def _configuration(agent_home: Path) -> dict[str, object]:
    model = (
        os.environ.get("BENCHFLOW_LITELLM_MODEL_ALIAS")
        or os.environ.get("OPENSCIENCE_BENCHFLOW_MODEL")
        or os.environ.get("BENCHFLOW_PROVIDER_MODEL")
        or "deepseek-v4-flash"
    )
    base_url = os.environ.get("BENCHFLOW_PROVIDER_BASE_URL", "").rstrip("/")
    proxy_mode = bool(os.environ.get("BENCHFLOW_LITELLM_MODEL_ALIAS"))
    protocol = os.environ.get("BENCHFLOW_PROVIDER_PROTOCOL", "")
    provider_npm, fallback_provider_id = _provider_adapter(protocol)
    provider_id = (
        "benchflow"
        if proxy_mode
        else os.environ.get("BENCHFLOW_PROVIDER_NAME") or fallback_provider_id
    )

    provider: dict[str, object] = {
        "name": "BenchFlow gateway" if proxy_mode else provider_id,
        "npm": provider_npm,
        "env": [],
        "models": {
            model: {
                "name": model,
                "tool_call": True,
                "reasoning": True,
                "limit": {"context": 131072, "output": 8192},
            }
        },
        "options": {
            "apiKey": "{env:OPENSCIENCE_BENCHFLOW_API_KEY}",
            **({"baseURL": base_url} if base_url else {}),
        },
    }
    config: dict[str, object] = {
        "$schema": "https://syntheticsciences.ai/config.json",
        "model": f"{provider_id}/{model}",
        "provider": {provider_id: provider},
        "instructions": [
            str(
                agent_home
                / ".openscience-benchflow"
                / "config"
                / "benchflow-workspace.md"
            )
        ],
        "sandbox": {"enabled": False},
        "agent": {"title": {"disable": True}},
        "skills": {"paths": [str(agent_home / ".claude" / "skills")]},
        "permission": {
            "*": "deny",
            "read": "allow",
            "edit": "allow",
            "write": "allow",
            "apply_patch": "allow",
            "glob": "allow",
            "grep": "allow",
            "list": "allow",
            "bash": "allow",
            "python": "allow",
            "notebook": "allow",
            "r": "allow",
            "rkernel": "allow",
            "artifact": "allow",
            "experiments": "allow",
            "todowrite": "allow",
            "todoread": "allow",
            "question": "allow",
            "lsp": "allow",
            "skill": "allow",
            # Task-declared MCP servers arrive through ACP session/new. The
            # global deny rule otherwise removes their tools from model input.
            "mcp": "allow",
        },
        "experimental": {"continue_loop_on_deny": True},
    }
    task_mcp_path = agent_home / ".openscience-benchflow" / "config" / "task-mcp.json"
    if task_mcp_path.is_file():
        task_mcp = json.loads(task_mcp_path.read_text(encoding="utf-8"))
        if isinstance(task_mcp, dict) and isinstance(task_mcp.get("mcp"), dict):
            config["mcp"] = task_mcp["mcp"]
            permissions = config["permission"]
            if isinstance(permissions, dict):
                for server_name in task_mcp["mcp"]:
                    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", server_name)
                    permissions[f"{safe_name}_*"] = "allow"
    return config


def main() -> None:
    agent_home = Path(
        os.environ.get("BENCHFLOW_AGENT_HOME") or _required("HOME")
    ).resolve()
    state_root = agent_home / ".openscience-benchflow"
    data_dir = state_root / "data"
    config_dir = state_root / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "openscience.json"
    workspace = Path(os.environ.get("BENCHFLOW_WORKSPACE") or os.getcwd()).resolve()
    instruction_path = config_dir / "benchflow-workspace.md"
    instruction_path.write_text(
        "\n".join(
            [
                "# BenchFlow task workspace",
                "",
                f"The benchmark task workspace is `{workspace}`.",
                "Use this exact directory as `workdir` for shell commands and use absolute paths beneath it for file tools.",
                "OpenScience session scratch is temporary internal state, not the task workspace or a deliverable location.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    instruction_path.chmod(0o600)
    config_path.write_text(
        json.dumps(_configuration(agent_home), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    os.environ.update(
        {
            "HOME": str(agent_home),
            "OPENSCIENCE_DATA_DIR": str(data_dir),
            "OPENSCIENCE_CONFIG_DIR": str(config_dir),
            "OPENSCIENCE_DISABLE_AUTOUPDATE": "1",
            "OPENSCIENCE_DISABLE_LSP_DOWNLOAD": "1",
            "OPENSCIENCE_DISABLE_PROJECT_CONFIG": "1",
            "OPENSCIENCE_SKIP_ENVIRONMENT_BOOTSTRAP": "1",
            "OPENSCIENCE_DISABLE_BUNDLED_SKILLS": "1",
            "OPENSCIENCE_DISABLE_MODELS_FETCH": "1",
            "OPENSCIENCE_DISABLE_TERMINAL_TITLE": "1",
        }
    )
    os.execv(
        "/opt/benchflow/bin/openscience",
        ["openscience", "acp", "--cwd", str(workspace)],
    )


if __name__ == "__main__":
    main()
