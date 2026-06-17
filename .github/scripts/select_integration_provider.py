#!/usr/bin/env python3
"""Select a live model provider for the integration-eval workflow.

The workflow needs a real LLM for both the rollout and the agent-as-judge
gate. Repo secrets can drift independently from code, so this script probes the
configured candidates with a tiny OpenAI-compatible request and exports the
first usable one to ``GITHUB_ENV``. It prints only candidate names and HTTP
status classes, never secret values or response bodies.
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

ProbeFunc = Callable[["Candidate", Mapping[str, str]], tuple[bool, str]]


@dataclass(frozen=True)
class Candidate:
    name: str
    key_env: str
    base_url: str
    probe_model: str
    rollout_agent: str
    rollout_model: str
    judge_model: str
    exports: Mapping[str, str]
    insecure_tls: bool = False


def _append_github_env(values: Mapping[str, str], env: Mapping[str, str]) -> None:
    path = env.get("GITHUB_ENV")
    if not path:
        return
    marker = "__BENCHFLOW_ENV__"
    with Path(path).open("a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}<<{marker}\n{value}\n{marker}\n")


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def probe_candidate(candidate: Candidate, env: Mapping[str, str]) -> tuple[bool, str]:
    key = env.get(candidate.key_env, "").strip()
    if not key:
        return False, f"missing {candidate.key_env}"

    body = json.dumps(
        {
            "model": candidate.probe_model,
            "messages": [{"role": "user", "content": "Return OK only."}],
            "max_tokens": 8,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        _chat_completions_url(candidate.base_url),
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    context = None
    if candidate.insecure_tls:
        context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(request, timeout=25, context=context) as response:
            if response.status < 400:
                return True, f"HTTP {response.status}"
            return False, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return False, f"{type(reason).__name__}"
    except TimeoutError:
        return False, "timeout"


def candidates(env: Mapping[str, str]) -> list[Candidate]:
    deepseek_base = env.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    glm_base = env.get("GLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
    qwen_base = env.get("QWEN_BASE_URL") or (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    litellm_base = env.get("LITELLM_BASE_URL") or (
        "http://llm-proxy.eval.all-hands.dev/v1"
    )
    github_base = "https://models.github.ai/inference"

    return [
        Candidate(
            name="deepseek",
            key_env="DEEPSEEK_API_KEY",
            base_url=deepseek_base,
            probe_model=env.get("DEEPSEEK_FLASH_MODEL") or "deepseek-v4-flash",
            rollout_agent="openhands",
            rollout_model="deepseek/deepseek-v4-flash",
            judge_model="openai/deepseek-v4-flash",
            exports={
                "DEEPSEEK_API_KEY": env.get("DEEPSEEK_API_KEY", ""),
                "DEEPSEEK_BASE_URL": deepseek_base,
                "OPENAI_API_KEY": env.get("DEEPSEEK_API_KEY", ""),
                "OPENAI_BASE_URL": deepseek_base,
            },
        ),
        Candidate(
            name="glm",
            key_env="GLM_API_KEY",
            base_url=glm_base,
            probe_model=env.get("GLM_MODEL") or "glm-5.1",
            rollout_agent="openhands",
            rollout_model="glm/glm-5.1",
            judge_model="openai/glm-5.1",
            exports={
                "GLM_API_KEY": env.get("GLM_API_KEY", ""),
                "GLM_BASE_URL": glm_base,
                "OPENAI_API_KEY": env.get("GLM_API_KEY", ""),
                "OPENAI_BASE_URL": glm_base,
            },
        ),
        Candidate(
            name="qwen-dashscope",
            key_env="QWEN_API_KEY",
            base_url=qwen_base,
            probe_model=env.get("QWEN_MAX_PREVIEW_MODEL") or "qwen3.6-max-preview",
            rollout_agent="openhands",
            rollout_model="qwen-dashscope/qwen3.6-max-preview",
            judge_model="openai/qwen3.6-max-preview",
            exports={
                "QWEN_API_KEY": env.get("QWEN_API_KEY", ""),
                "QWEN_BASE_URL": qwen_base,
                "OPENAI_API_KEY": env.get("QWEN_API_KEY", ""),
                "OPENAI_BASE_URL": qwen_base,
            },
        ),
        Candidate(
            name="litellm-proxy",
            key_env="LITELLM_API_KEY",
            base_url=litellm_base,
            probe_model=env.get("LITELLM_MODEL") or "gpt-4.1-mini",
            rollout_agent="openhands",
            rollout_model=f"litellm/{env.get('LITELLM_MODEL') or 'gpt-4.1-mini'}",
            judge_model=f"openai/{env.get('LITELLM_MODEL') or 'gpt-4.1-mini'}",
            exports={
                "LITELLM_API_KEY": env.get("LITELLM_API_KEY", ""),
                "LITELLM_BASE_URL": litellm_base,
                "OPENAI_API_KEY": env.get("LITELLM_API_KEY", ""),
                "OPENAI_BASE_URL": litellm_base,
            },
        ),
        Candidate(
            name="openai",
            key_env="OPENAI_API_KEY",
            base_url=env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            probe_model=env.get("OPENAI_MODEL") or "gpt-4o-mini",
            rollout_agent="openhands",
            rollout_model=f"openai/{env.get('OPENAI_MODEL') or 'gpt-4o-mini'}",
            judge_model=f"openai/{env.get('OPENAI_MODEL') or 'gpt-4o-mini'}",
            exports={
                "OPENAI_API_KEY": env.get("OPENAI_API_KEY", ""),
                "OPENAI_BASE_URL": env.get("OPENAI_BASE_URL")
                or "https://api.openai.com/v1",
            },
        ),
        Candidate(
            name="github-models",
            key_env="GITHUB_MODELS_TOKEN",
            base_url=github_base,
            probe_model=env.get("GITHUB_MODELS_MODEL") or "openai/gpt-4.1-mini",
            rollout_agent="openhands",
            rollout_model="github-models/openai/gpt-4.1-mini",
            judge_model="openai/openai/gpt-4.1-mini",
            exports={
                "GITHUB_TOKEN": env.get("GITHUB_MODELS_TOKEN", ""),
                "OPENAI_API_KEY": env.get("GITHUB_MODELS_TOKEN", ""),
                "OPENAI_BASE_URL": github_base,
            },
        ),
    ]


def select_candidate(
    pool: list[Candidate],
    env: Mapping[str, str],
    probe: ProbeFunc = probe_candidate,
) -> tuple[Candidate | None, list[tuple[str, str]]]:
    attempts: list[tuple[str, str]] = []
    for candidate in pool:
        ok, reason = probe(candidate, env)
        attempts.append((candidate.name, reason))
        print(f"Provider candidate {candidate.name}: {reason}")
        if ok:
            return candidate, attempts
    return None, attempts


def main() -> int:
    env = os.environ
    selected, attempts = select_candidate(candidates(env), env)
    if selected is None:
        summary = ", ".join(f"{name}={reason}" for name, reason in attempts)
        print(f"::error::No usable integration LLM provider found ({summary})")
        return 1

    exports = {
        "BENCHFLOW_INTEGRATION_PROVIDER": selected.name,
        "BENCHFLOW_INTEGRATION_AGENT": selected.rollout_agent,
        "BENCHFLOW_INTEGRATION_MODEL": selected.rollout_model,
        "BENCHFLOW_JUDGE_MODEL": selected.judge_model,
        **selected.exports,
    }
    _append_github_env(exports, env)
    print(
        "Selected integration provider "
        f"{selected.name}: agent={selected.rollout_agent}, "
        f"model={selected.rollout_model}, judge={selected.judge_model}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
