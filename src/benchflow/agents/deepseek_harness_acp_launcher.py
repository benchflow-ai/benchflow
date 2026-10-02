#!/usr/bin/env python3
"""Prepare a deterministic DSH profile overlay and exec its native ACP server."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _yaml_scalar(value: str) -> str:
    """Return a JSON string, which is also a valid YAML scalar."""
    return json.dumps(value)


def _dsh_protocol(provider_protocol: str) -> str:
    """Map BenchFlow's API protocol names to DSH's native wire protocols."""
    if provider_protocol in ("", "openai-completions"):
        return "chat-completions"
    if provider_protocol == "anthropic-messages":
        return "messages"
    raise SystemExit(
        "deepseek-harness: unsupported BENCHFLOW_PROVIDER_PROTOCOL "
        f"{provider_protocol!r}; supported values are openai-completions and "
        "anthropic-messages"
    )


def _patch(model: str, base_url: str, skill_dir: Path, protocol: str) -> str:
    llm_config = [
        f"    protocol: {protocol}",
        "    apiKeyEnv: DEEPSEEK_API_KEY",
    ]
    if base_url:
        llm_config.append(f"    baseURL: {_yaml_scalar(base_url.rstrip('/'))}")
    return "\n".join(
        [
            "- id: acp",
            "  config:",
            "    provider: deepseek-official",
            f"    model: {_yaml_scalar(model)}",
            "",
            "- id: agent-default-model",
            "  config:",
            "    provider: deepseek-official",
            f"    model: {_yaml_scalar(model)}",
            "",
            "- id: llm-deepseek",
            "  config:",
            *llm_config,
            "",
            "- id: sandbox-policy",
            "  config:",
            "    mode: danger-full-access",
            "    workspaceRoot: !!js process.cwd()",
            "",
            "- id: approval",
            "  config:",
            "    policy: never",
            "",
            "- id: skill-filesystem",
            "  config:",
            "    includeDefaultRoots: false",
            "    customSkillDirs:",
            f"      - {_yaml_scalar(str(skill_dir))}",
            "    watch: false",
            "",
            "- id: session-telemetry-otel",
            "  disabled: true",
            "",
            "- id: web-search-deepseek",
            "  disabled: true",
            "",
            "- id: tool-web",
            "  disabled: true",
            "",
            "- id: plugin-manager",
            "  disabled: true",
            "",
            "- id: hmr",
            "  disabled: true",
            "",
        ]
    )


def main() -> None:
    agent_home = Path(
        os.environ.get("BENCHFLOW_AGENT_HOME") or os.environ.get("HOME") or "/tmp"
    ).resolve()
    dsh_home = agent_home / ".dsh-benchflow"
    dsh_home.mkdir(parents=True, exist_ok=True)
    skill_dir = agent_home / ".agents" / "skills"
    skill_dir.mkdir(parents=True, exist_ok=True)

    model = (
        os.environ.get("BENCHFLOW_LITELLM_MODEL_ALIAS")
        or os.environ.get("DSH_BENCHFLOW_MODEL")
        or os.environ.get("BENCHFLOW_PROVIDER_MODEL")
        or "deepseek-v4-flash"
    )
    patch_path = dsh_home / "benchflow-acp.patch.yml"
    protocol = _dsh_protocol(os.environ.get("BENCHFLOW_PROVIDER_PROTOCOL", ""))
    patch_path.write_text(
        _patch(
            model,
            os.environ.get("DEEPSEEK_BASE_URL", ""),
            skill_dir,
            protocol,
        ),
        encoding="utf-8",
    )
    patch_path.chmod(0o600)

    os.environ.update(
        {
            "HOME": str(agent_home),
            "DSH_HOME": str(dsh_home),
            "DSH_PERMISSION_MODE": "danger-full-access",
            "DSH_TELEMETRY_MODE": "DISABLED",
        }
    )
    os.execv(
        "/opt/benchflow/bin/dsh",
        ["dsh", "--profile", "acp", "--patch", str(patch_path)],
    )


if __name__ == "__main__":
    main()
