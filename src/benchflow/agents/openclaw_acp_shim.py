#!/usr/bin/env python3
"""ACP shim for OpenClaw — wraps `openclaw agent --local` as an ACP server.

openclaw's native ACP bridge requires a gateway with chat-thread sessions.
This shim speaks ACP on stdio and internally calls `openclaw agent --local`
for each prompt, then parses openclaw's session JSONL to emit proper ACP
tool_call and text updates.

Architecture:
  benchflow ACP client ←stdio→ this shim ←subprocess→ openclaw agent --local
                                          ←file read→  ~/.openclaw/agents/main/sessions/*.jsonl

Key details:
  - Workspace: symlinks ~/.openclaw/workspace → task cwd (openclaw ignores subprocess cwd)
  - Skills: if task env has ~/.claude/skills/, also copies to ~/.openclaw/workspace/.claude/skills/
  - Trajectory: parses session JSONL for tool calls, thinking, text → emits ACP session/update
  - Model: set via openclaw config on session/set_model
"""

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only; the shim must stay runnable without benchflow installed.
    from benchflow.agents.providers import ProviderConfig

logger = logging.getLogger(__name__)

_DIAG_TRUNCATE = 2000  # max chars for diagnostic output in ACP updates
_TOOL_RESULT_TRUNCATE = 1000  # max chars for tool result text
_TOOL_INPUT_TRUNCATE = 500  # max chars for tool input echoed in ACP updates
_OPENCLAW_BIN = "/opt/benchflow/bin/openclaw"

_PARAM_MAP = {
    "BENCHFLOW_MODEL_TEMPERATURE": "agents.defaults.params.temperature",
    "BENCHFLOW_MODEL_TOP_P": "agents.defaults.params.topP",
    "BENCHFLOW_MODEL_MAX_TOKENS": "agents.defaults.params.maxTokens",
}


# ── ACP stdio I/O ─────────────────────────────────────────────────────────────


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def recv():
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed")
        line = line.strip()
        if not line:
            continue
        return json.loads(line)


# ── Workspace + auth setup ────────────────────────────────────────────────────


def setup_workspace(cwd: str):
    """Point openclaw's workspace at the task directory and load skills.

    openclaw discovers skills from <workspace>/skills/ (not .claude/skills/).
    SkillsBench tasks bake skills into ~/.claude/skills/ via Dockerfile.
    We symlink/copy them to <workspace>/skills/ so openclaw can find them.
    """
    home = os.environ.get("HOME", os.path.expanduser("~"))
    oc_workspace = Path(home) / ".openclaw" / "workspace"

    if oc_workspace.is_symlink() or oc_workspace.exists():
        if oc_workspace.is_symlink():
            oc_workspace.unlink()
        elif oc_workspace.is_dir():
            shutil.rmtree(oc_workspace)

    oc_workspace.parent.mkdir(parents=True, exist_ok=True)
    oc_workspace.symlink_to(cwd)

    # Load skills: check common skill locations and copy to <workspace>/skills/
    workspace_skills = Path(cwd) / "skills"
    if not workspace_skills.exists():
        # Search for skills in known locations
        skill_sources = [
            Path(cwd) / ".claude" / "skills",  # SkillsBench Claude format
            Path(home) / ".claude" / "skills",  # Home dir Claude skills
            Path(cwd) / ".codex" / "skills",  # Codex format
            Path(cwd) / ".agents" / "skills",  # Generic agent skills
        ]
        for src in skill_sources:
            if src.is_dir() and any(src.iterdir()):
                # Copy skills to workspace/skills/ for openclaw discovery
                workspace_skills.mkdir(parents=True, exist_ok=True)
                for skill_dir in src.iterdir():
                    if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                        dest = workspace_skills / skill_dir.name
                        if not dest.exists():
                            shutil.copytree(skill_dir, dest)
                break  # Use first source found


def setup_openai_auth():
    """Write OPENAI_API_KEY into openclaw's native auth store if present."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return
    agent_dir = Path.home() / ".openclaw" / "agents" / "main" / "agent"
    auth_path = agent_dir / "auth-profiles.json"
    existing = {}
    if auth_path.exists():
        try:
            existing = json.loads(auth_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.debug("Could not read existing auth config at %s", auth_path)
    existing["openai"] = {"apiKey": api_key}
    agent_dir.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps(existing))


def setup_gcloud_adc():
    """Write ADC credentials from env var to disk and enable google plugin for Vertex AI."""
    adc_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not adc_json:
        return
    adc_path = (
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    )
    adc_path.parent.mkdir(parents=True, exist_ok=True)
    adc_path.write_text(adc_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(adc_path)
    # Enable the google plugin so openclaw recognizes google-vertex/ models
    subprocess.run(
        [_OPENCLAW_BIN, "plugins", "enable", "google"],
        capture_output=True,
        timeout=10,
    )


def _get_adc_token() -> str:
    """Get a bearer token from ADC credentials (stdlib only, no google-auth dep).

    Supports both service-account keys (JWT → token exchange) and
    authorized-user credentials (refresh_token → token exchange).
    """
    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not adc_path or not Path(adc_path).exists():
        # Fallback to default ADC location
        adc_path = str(
            Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        )
    with open(adc_path) as f:
        creds = json.load(f)

    cred_type = creds.get("type", "")

    if cred_type == "authorized_user":
        # Refresh token flow
        data = urllib.parse.urlencode(
            {
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"],
                "refresh_token": creds["refresh_token"],
                "grant_type": "refresh_token",
            }
        ).encode()
        req = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["access_token"]

    elif cred_type == "service_account":
        # JWT → access token flow (RS256)
        # Requires PyJWT or manual RSA — use subprocess openssl as fallback
        now = int(time.time())
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
        ).rstrip(b"=")
        payload = base64.urlsafe_b64encode(
            json.dumps(
                {
                    "iss": creds["client_email"],
                    "scope": "https://www.googleapis.com/auth/cloud-platform",
                    "aud": "https://oauth2.googleapis.com/token",
                    "iat": now,
                    "exp": now + 3600,
                }
            ).encode()
        ).rstrip(b"=")
        signing_input = header + b"." + payload

        # Sign with openssl (available in most containers)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as kf:
            kf.write(creds["private_key"])
            key_path = kf.name
        try:
            result = subprocess.run(
                ["openssl", "dgst", "-sha256", "-sign", key_path],
                input=signing_input,
                capture_output=True,
                timeout=10,
            )
            signature = base64.urlsafe_b64encode(result.stdout).rstrip(b"=")
        finally:
            os.unlink(key_path)

        jwt_token = (signing_input + b"." + signature).decode()
        data = urllib.parse.urlencode(
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": jwt_token,
            }
        ).encode()
        req = urllib.request.Request(
            "https://oauth2.googleapis.com/token",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["access_token"]

    else:
        raise ValueError(f"Unsupported ADC credential type: {cred_type!r}")


# ── Provider resolution ───────────────────────────────────────────────────────


def setup_custom_provider(
    provider_name: str,
    base_url: str,
    api_key: str,
    api_protocol: str = "openai-completions",
    models: list[dict] | None = None,
):
    """Configure an openclaw custom provider in ~/.openclaw/openclaw.json.

    This is the generic replacement for per-provider setup functions.
    Any OpenAI-compatible or Anthropic-compatible endpoint can be registered.
    """
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing config so multiple providers can coexist
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.debug("Could not read existing provider config at %s", config_path)

    providers = existing.setdefault("models", {}).setdefault("providers", {})
    providers[provider_name] = {
        "baseUrl": base_url,
        "api": api_protocol,
        "apiKey": api_key,
        "models": models or [],
    }

    config_path.write_text(json.dumps(existing, indent=2))


def _infer_provider_prefix(model: str) -> str:
    """Infer the openclaw provider prefix from a bare model name.

    Resolution order:
      1. The benchflow provider registry — any registered custom provider
         (deepseek/glm/qwen-dashscope/...) that claims this bare model id via
         its declared model_prefixes. This routes prefix-stripped ids like
         "deepseek-v4-flash" to "deepseek" instead of defaulting to anthropic.
      2. The native gemini/gpt heuristics (openclaw handles these directly).
      3. Anthropic as the final fallback.
    """
    m = model.lower()
    # 1. Registry-driven bare-model routing (registry owns provider knowledge).
    try:
        from benchflow.agents.providers import find_provider_for_bare_model

        result = find_provider_for_bare_model(model)
        if result is not None:
            return result[0]
    except ImportError:
        logger.debug("benchflow.agents.providers unavailable; using name heuristics")

    # 2. Native providers openclaw recognizes without registry config.
    if "gemini" in m:
        return "google"
    if "gpt" in m or m.startswith(("o1", "o3")):
        return "openai"
    # 3. Default.
    return "anthropic"


def _setup_provider_from_config(
    provider_name: str, cfg: "ProviderConfig"
) -> str | None:
    """Write a resolved registry ProviderConfig into openclaw.json.

    Shared by the prefix-based (``_find_and_setup_provider``) and bare-model
    (``_setup_bare_custom_provider``) paths so both register a custom provider
    through exactly the same logic. Returns ``provider_name`` on success, or
    ``None`` if required config (base_url / api key) is missing — callers then
    fall through to their next resolution strategy.

    Raises ``KeyError`` if a required ``url_params`` env var is missing, so the
    prefix-based caller can preserve its existing "fall through to the
    BENCHFLOW_PROVIDER_* env-var path" behavior on that specific failure.
    """
    from benchflow.agents.providers import resolve_base_url

    env = dict(os.environ)
    base_url = resolve_base_url(cfg, env)  # may raise KeyError (missing url_params)
    if cfg.auth_type == "adc":
        try:
            api_key = _get_adc_token()
        except Exception:
            logger.debug(
                "ADC token acquisition failed for %s", provider_name, exc_info=True
            )
            return None
    elif cfg.auth_type == "none":
        api_key = ""
    elif cfg.auth_env:
        api_key = env.get(cfg.auth_env, "")
        if not api_key:
            return None
    else:
        return None
    setup_custom_provider(
        provider_name, base_url, api_key, cfg.api_protocol, cfg.models
    )
    return provider_name


def _setup_bare_custom_provider(model: str) -> str | None:
    """Configure the custom provider a BARE model id resolves to, if any.

    Companion to ``_infer_provider_prefix`` for the ``session/set_model``
    "No provider" branch: ``_infer_provider_prefix`` only *names* the provider
    prefix, but a registered custom provider (deepseek/glm/qwen-dashscope/...)
    must also be written into ``~/.openclaw/openclaw.json`` or openclaw gets a
    ``deepseek/...`` model id pointing at a provider it was never told about.

    Resolves ``model`` via ``find_provider_for_bare_model`` (registry-owned
    ``model_prefixes``) and, when that names a custom provider, registers it via
    the shared ``_setup_provider_from_config`` path. Returns the provider name
    if setup succeeded, else ``None`` (openclaw-native ids like gemini/gpt and
    unknown ids resolve to no registry provider and need no custom config).
    """
    try:
        from benchflow.agents.providers import find_provider_for_bare_model

        result = find_provider_for_bare_model(model)
        if result is not None:
            provider_name, cfg = result
            try:
                return _setup_provider_from_config(provider_name, cfg)
            except KeyError:
                # Missing url_params env var — provider can't be configured.
                logger.debug(
                    "Bare model %s resolved to %s but its config env vars are "
                    "unset; leaving provider unconfigured",
                    model,
                    provider_name,
                )
                return None
    except ImportError:
        logger.debug("benchflow.agents.providers unavailable; skipping bare setup")
    return None


def _find_and_setup_provider(model: str) -> str | None:
    """If model matches a custom provider, configure it and return the provider name.

    Returns the registered provider name (e.g. "google-vertex", "custom") so the
    caller can prefix the model for openclaw, or None if no provider was set up.

    Resolution order:
    1. If benchflow is importable, try find_provider(model) for prefix-based match.
    2. Fall back to BENCHFLOW_PROVIDER_* env vars injected by the SDK.
       This handles stripped model names (no prefix) where the SDK already
       resolved the provider and passed config via env vars.
    """
    # 1. Try benchflow provider registry (prefix-based match)
    try:
        from benchflow.agents.providers import find_provider

        result = find_provider(model)
        if result is not None:
            provider_name, cfg = result
            try:
                return _setup_provider_from_config(provider_name, cfg)
            except KeyError:
                pass  # missing url_params env var; fall through to env var path
    except ImportError:
        logger.debug("benchflow.agents.providers not available, using env var fallback")

    # 2. Fall back to BENCHFLOW_PROVIDER_* env vars set by the SDK.
    #    This is the primary path for stripped model names (e.g. "claude-sonnet-4-6"
    #    from "anthropic-vertex/claude-sonnet-4-6") where the SDK already resolved
    #    the provider config.
    base_url = os.environ.get("BENCHFLOW_PROVIDER_BASE_URL")
    api_key = os.environ.get("BENCHFLOW_PROVIDER_API_KEY")
    api_protocol = os.environ.get("BENCHFLOW_PROVIDER_PROTOCOL", "openai-completions")
    models_json = os.environ.get("BENCHFLOW_PROVIDER_MODELS", "[]")
    # If no explicit API key, try ADC (for Vertex AI providers)
    if base_url and not api_key:
        try:
            api_key = _get_adc_token()
        except Exception:
            logger.debug("ADC token fallback failed", exc_info=True)
    if base_url and api_key:
        provider_name = model.split("/")[0] if "/" in model else "custom"
        try:
            models = json.loads(models_json)
        except json.JSONDecodeError:
            models = []
        setup_custom_provider(provider_name, base_url, api_key, api_protocol, models)
        return provider_name
    return None


def _resolve_bare_model_prefix(model: str) -> str:
    """Resolve (and, where possible, configure) the provider for a BARE model id.

    Used by the ``session/set_model`` branch where ``BENCHFLOW_PROVIDER_NAME``
    is absent. Each step falls through on ``None``:

    1. ``_setup_bare_custom_provider`` — the registry claims the bare id and
       its provider-specific config env vars resolve; the provider is written
       into openclaw.json.
    2. ``_find_and_setup_provider`` — generic ``BENCHFLOW_PROVIDER_*`` env
       fallback: a harness may inject endpoint config without the provider
       name. Explicit endpoint config must win over a prefix guess for a
       provider openclaw was never configured with (codex P2 on PR #670).
    3. ``_infer_provider_prefix`` — no config available anywhere; name the
       prefix without registration (openclaw-native gemini/gpt ids, or a
       custom provider whose run cannot work without its key anyway).
    """
    return (
        _setup_bare_custom_provider(model)
        or _find_and_setup_provider(model)
        or _infer_provider_prefix(model)
    )


# ── Session parsing ───────────────────────────────────────────────────────────


def find_session_jsonl() -> Path | None:
    """Find the most recent openclaw session JSONL file."""
    home = os.environ.get("HOME", os.path.expanduser("~"))
    sessions_dir = Path(home) / ".openclaw" / "agents" / "main" / "sessions"
    if not sessions_dir.exists():
        return None

    jsonl_files = sorted(
        sessions_dir.glob("*.jsonl"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    # Skip .lock files
    for f in jsonl_files:
        if not f.name.endswith(".lock"):
            return f
    return None


def parse_session_jsonl(path: Path, session_id: str) -> list[dict]:
    """Parse openclaw session JSONL and convert to ACP session/update events.

    openclaw JSONL format uses {type: "message", message: {role, content}} entries.
    Roles: "user", "assistant", "toolResult"
    Content blocks: text, tool_use, thinking (in assistant messages)
    """
    updates = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # openclaw format: {type: "message", message: {role, content}}
                if entry.get("type") != "message":
                    continue

                msg = entry.get("message", {})
                role = msg.get("role", "")
                content = msg.get("content", [])

                if role == "assistant" and isinstance(content, list):
                    for block in content:
                        block_type = block.get("type", "")

                        if block_type in ("text",):
                            updates.append(
                                {
                                    "jsonrpc": "2.0",
                                    "method": "session/update",
                                    "params": {
                                        "sessionId": session_id,
                                        "update": {
                                            "sessionUpdate": "text_update",
                                            "text": block.get("text", ""),
                                        },
                                    },
                                }
                            )

                        elif block_type in ("tool_use", "toolCall"):
                            _input = block.get("input", block.get("arguments", {}))
                            _title = _input.get(
                                "command",
                                _input.get("description", block.get("name", "tool")),
                            )
                            updates.append(
                                {
                                    "jsonrpc": "2.0",
                                    "method": "session/update",
                                    "params": {
                                        "sessionId": session_id,
                                        "update": {
                                            "sessionUpdate": "tool_call",
                                            "toolCallId": block.get("id", ""),
                                            "kind": block.get("name", "tool"),
                                            "title": _title,
                                            "status": "completed",
                                            "content": [
                                                {
                                                    "type": "content",
                                                    "content": {
                                                        "type": "text",
                                                        "text": json.dumps(_input)[
                                                            :_TOOL_INPUT_TRUNCATE
                                                        ],
                                                    },
                                                }
                                            ],
                                        },
                                    },
                                }
                            )

                        elif block_type == "thinking":
                            updates.append(
                                {
                                    "jsonrpc": "2.0",
                                    "method": "session/update",
                                    "params": {
                                        "sessionId": session_id,
                                        "update": {
                                            "sessionUpdate": "agent_thought",
                                            "text": block.get("thinking", ""),
                                        },
                                    },
                                }
                            )

                elif role == "toolResult":
                    # Emit as tool_call_update (status=completed) to update
                    # the tool_call record created by the tool_use block
                    tool_id = msg.get("toolCallId", "")
                    result_text = ""
                    if isinstance(content, list):
                        result_text = " ".join(
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    elif isinstance(content, str):
                        result_text = content

                    updates.append(
                        {
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "sessionId": session_id,
                                "update": {
                                    "sessionUpdate": "tool_call_update",
                                    "toolCallId": tool_id,
                                    "status": "completed",
                                    "content": [
                                        {
                                            "type": "content",
                                            "content": {
                                                "type": "text",
                                                "text": result_text[
                                                    :_TOOL_RESULT_TRUNCATE
                                                ],
                                            },
                                        }
                                    ],
                                },
                            },
                        }
                    )

    except Exception:
        logger.debug("Failed to parse session JSONL for trajectory", exc_info=True)

    return updates


# ── Main loop ─────────────────────────────────────────────────────────────────


def main():
    setup_openai_auth()
    setup_gcloud_adc()
    session_id = "openclaw-shim"
    cwd = "/app"

    while True:
        try:
            msg = recv()
        except EOFError:
            break

        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": 1,
                        "agentCapabilities": {
                            "loadSession": False,
                            "promptCapabilities": {"image": False, "audio": False},
                        },
                        "agentInfo": {"name": "openclaw", "version": "1.0"},
                    },
                }
            )

        elif method == "session/new":
            cwd = params.get("cwd", "/app")
            setup_workspace(cwd)
            session_id = "openclaw-shim"
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"sessionId": session_id},
                }
            )

        elif method == "session/set_model":
            model = params.get("modelId", "")
            # A provider-resolution / config-write failure here must NOT crash the
            # shim: an unhandled exception exits rc=1, which benchflow sees as the
            # ACP transport dying mid-set_model ("Process closed stdout (rc=1)") —
            # the observed openclaw-on-docker failure. Catch, log the real cause,
            # and still ACK so the run proceeds with the model config that exists.
            try:
                if model:
                    # The SDK strips provider prefixes before set_model and passes
                    # the original provider name via BENCHFLOW_PROVIDER_NAME env var.
                    #
                    # Openclaw natively supports google-vertex/ and anthropic/ prefixes
                    # (via the google plugin enabled at startup). Custom providers like
                    # zai/ and other custom providers need explicit registration via openclaw.json.
                    provider_name = os.environ.get("BENCHFLOW_PROVIDER_NAME", "")

                    # Native Vertex providers — openclaw handles these via google plugin
                    if provider_name in ("google-vertex", "anthropic-vertex"):
                        # Reconstruct the full model name openclaw expects
                        if "/" not in model:
                            model = f"{provider_name}/{model}"
                    # Custom providers — register in openclaw.json
                    elif provider_name:
                        _provider_name = _find_and_setup_provider(model)
                        if _provider_name and "/" not in model:
                            model = f"{_provider_name}/{model}"
                    # No provider env var — resolve the bare id: registry-specific
                    # setup, then the generic BENCHFLOW_PROVIDER_* env fallback,
                    # then prefix heuristics (see _resolve_bare_model_prefix).
                    elif "/" not in model:
                        model = f"{_resolve_bare_model_prefix(model)}/{model}"

                    subprocess.run(
                        [
                            _OPENCLAW_BIN,
                            "config",
                            "set",
                            "agents.defaults.model",
                            model,
                        ],
                        capture_output=True,
                        timeout=10,
                    )

                # Apply model generation parameters from env vars
                for env_key, config_path in _PARAM_MAP.items():
                    val = os.environ.get(env_key)
                    if val:
                        subprocess.run(
                            [_OPENCLAW_BIN, "config", "set", config_path, val],
                            capture_output=True,
                            timeout=10,
                        )
            except Exception as exc:
                print(
                    f"[openclaw-acp-shim] set_model setup failed, continuing: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )

            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        elif method == "session/prompt":
            prompt_parts = params.get("prompt", [])
            text = ""
            for part in prompt_parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part.get("text", "")

            try:
                result = subprocess.run(
                    [
                        _OPENCLAW_BIN,
                        "agent",
                        "--local",
                        "--agent",
                        "main",
                        "--json",
                        "-m",
                        text,
                        "--timeout",
                        "900",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=920,
                    env={**os.environ},
                )

                # Surface stderr as agent thought (for debugging)
                if result.stderr and result.stderr.strip():
                    send(
                        {
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "sessionId": session_id,
                                "update": {
                                    "sessionUpdate": "agent_thought",
                                    "text": f"[openclaw stderr]\n{result.stderr[:_DIAG_TRUNCATE]}",
                                },
                            },
                        }
                    )

                # Parse openclaw's session JSONL for full trajectory
                # Extract session ID from JSON output (may be multi-line)
                oc_session_id = None
                try:
                    # openclaw --json output can be multi-line pretty-printed
                    stdout = result.stdout.strip()
                    if stdout:
                        response_data = json.loads(stdout)
                        oc_session_id = (
                            response_data.get("meta", {})
                            .get("agentMeta", {})
                            .get("sessionId")
                        )
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Try finding sessionId in raw output
                    import re

                    m = re.search(r'"sessionId"\s*:\s*"([^"]+)"', result.stdout or "")
                    if m:
                        oc_session_id = m.group(1)

                # Find session JSONL: try specific ID first, then most recent
                session_jsonl = None
                home = os.environ.get("HOME", os.path.expanduser("~"))
                sessions_dir = Path(home) / ".openclaw" / "agents" / "main" / "sessions"

                if oc_session_id:
                    specific = sessions_dir / f"{oc_session_id}.jsonl"
                    if specific.exists():
                        session_jsonl = specific

                if not session_jsonl:
                    session_jsonl = find_session_jsonl()

                # Fallback: scan directory for most recent JSONL
                if not session_jsonl and sessions_dir.exists():
                    for jf in sorted(
                        sessions_dir.glob("*.jsonl"),
                        key=lambda f: f.stat().st_mtime,
                        reverse=True,
                    ):
                        if jf.name not in ("sessions.json",) and not jf.name.endswith(
                            ".lock"
                        ):
                            session_jsonl = jf
                            break

                if session_jsonl:
                    updates = parse_session_jsonl(session_jsonl, session_id)
                    for update in updates:
                        send(update)

                # If no JSONL trajectory, fall back to text response
                if not session_jsonl:
                    try:
                        response = json.loads(result.stdout)
                        agent_text = response.get("payloads", [{}])[0].get("text", "")
                    except (json.JSONDecodeError, IndexError, KeyError):
                        agent_text = (
                            result.stdout[:_DIAG_TRUNCATE] if result.stdout else ""
                        )

                    if agent_text:
                        send(
                            {
                                "jsonrpc": "2.0",
                                "method": "session/update",
                                "params": {
                                    "sessionId": session_id,
                                    "update": {
                                        "sessionUpdate": "text_update",
                                        "text": agent_text,
                                    },
                                },
                            }
                        )

                send(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"stopReason": "end_turn"},
                    }
                )

            except subprocess.TimeoutExpired:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"stopReason": "end_turn"},
                    }
                )
            except Exception as e:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32603, "message": str(e)},
                    }
                )

        elif method == "session/cancel":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})

        elif method == "session/request_permission":
            options = params.get("options", [])
            option_id = options[0].get("optionId", "default") if options else "default"
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "outcome": {"outcome": "selected", "optionId": option_id}
                    },
                }
            )

        else:
            if req_id:
                send({"jsonrpc": "2.0", "id": req_id, "result": {}})


if __name__ == "__main__":
    main()
