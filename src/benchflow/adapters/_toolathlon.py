"""Toolathlon source adapter materialization helpers."""

from __future__ import annotations

import json
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w
import yaml

_NOOP_EXCLUDE_TAG = "__benchflow_exclude_no_tools__"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TOKEN_PLACEHOLDER_RE = re.compile(r"\$\{token\.([A-Za-z0-9_]+)\}")
_TOOLATHLON_UVX_PACKAGE_PINS = {
    "office-word-mcp-server": "office-word-mcp-server==1.1.11",
}


@dataclass(frozen=True)
class _ToolathlonCredentialSpec:
    path: str
    env: str
    b64_env: str
    trigger_servers: tuple[str, ...] = ()
    token_defaults: tuple[tuple[str, str], ...] = ()
    copy_paths: tuple[str, ...] = ()


_TOOLATHLON_CREDENTIAL_SPECS = (
    _ToolathlonCredentialSpec(
        path="configs/gcp-service_account.keys.json",
        env="TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON",
        b64_env="TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON_B64",
        trigger_servers=("google-cloud",),
        token_defaults=(
            (
                "gcp_service_account_path",
                "/workspace/configs/gcp-service_account.keys.json",
            ),
        ),
    ),
    _ToolathlonCredentialSpec(
        path="configs/google_credentials.json",
        env="TOOLATHLON_GOOGLE_CREDENTIALS_JSON",
        b64_env="TOOLATHLON_GOOGLE_CREDENTIALS_JSON_B64",
        trigger_servers=("google_calendar", "google_sheet"),
        token_defaults=(
            (
                "google_oauth2_credentials_path",
                "/workspace/configs/google_credentials.json",
            ),
            (
                "google_oauth2_token_path",
                "/workspace/agent_workspace/.toolathlon/google_credentials.json",
            ),
        ),
        copy_paths=(
            "agent_workspace/.toolathlon/google_credentials.json",
            "agent_workspace/.toolathlon/calendar_credentials.json",
        ),
    ),
    _ToolathlonCredentialSpec(
        path="configs/gcp-oauth.keys.json",
        env="TOOLATHLON_GCP_OAUTH_KEYS_JSON",
        b64_env="TOOLATHLON_GCP_OAUTH_KEYS_JSON_B64",
        trigger_servers=("google_calendar",),
    ),
)
_TOOLATHLON_CREDENTIALS_BY_PATH = {
    spec.path: spec for spec in _TOOLATHLON_CREDENTIAL_SPECS
}
_TOOLATHLON_CREDENTIAL_REF_RE = re.compile(
    "|".join(re.escape(spec.path) for spec in _TOOLATHLON_CREDENTIAL_SPECS)
)
_TOOLATHLON_TOKEN_DEFAULTS = {
    token: value
    for spec in _TOOLATHLON_CREDENTIAL_SPECS
    for token, value in spec.token_defaults
}


def _safe_name(value: str) -> str:
    slug = _SAFE_NAME_RE.sub("-", value).strip("-._")
    return slug or "task"


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _write_toml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(payload))


def toolathlon_tasks_root(ctx: Any, *, variant: str) -> Path | None:
    candidates = [ctx.source_root]
    if ctx.source_root.name != "finalpool":
        candidates.append(ctx.source_root / "tasks" / "finalpool")
    candidates.append(ctx.repo_root / "tasks" / "finalpool")
    for candidate in candidates:
        if _has_toolathlon_task_children(candidate):
            return candidate
    return None


def _has_toolathlon_task_children(path: Path) -> bool:
    return path.is_dir() and any(
        child.is_dir() and (child / "task_config.json").is_file()
        for child in path.iterdir()
    )


def materialize_toolathlon(ctx: Any, output_dir: Path, *, variant: str) -> None:
    tasks_root = toolathlon_tasks_root(ctx, variant=variant)
    if tasks_root is None:
        raise ValueError(f"{variant} source does not contain tasks/finalpool task dirs")

    for upstream_task in sorted(tasks_root.iterdir()):
        if not (upstream_task / "task_config.json").is_file():
            continue
        task_name = upstream_task.name
        task_dir = output_dir / _safe_name(task_name)
        task_config = json.loads((upstream_task / "task_config.json").read_text())
        credential_refs = _toolathlon_credential_refs_for_task(
            upstream_task, task_config, variant=variant
        )
        mcp_servers = [
            _toolathlon_mcp_server(ctx, server_name, variant=variant)
            for server_name in task_config.get("needed_mcp_servers", [])
        ]

        _write_text(task_dir / "instruction.md", _toolathlon_instruction(upstream_task))
        _write_toml(
            task_dir / "task.toml",
            _toolathlon_task_toml(
                ctx,
                task_name=task_name,
                mcp_servers=mcp_servers,
                variant=variant,
                credential_refs=credential_refs,
            ),
        )
        _write_text(
            task_dir / "environment" / "Dockerfile",
            _toolathlon_dockerfile(ctx, variant=variant),
        )
        if variant == "gym":
            _write_text(
                task_dir / "environment" / "docker-compose.yaml",
                _TOOLATHLON_GYM_COMPOSE,
            )
            db_src = ctx.repo_root / "db" / "init.sql.gz"
            if db_src.is_file():
                db_dst = task_dir / "environment" / "db" / "init.sql.gz"
                db_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(db_src, db_dst)
        _write_text(
            task_dir / "tests" / "test.sh",
            _toolathlon_test_sh(task_name=task_name, variant=variant),
        )


def _toolathlon_referenced_config_paths(task_dir: Path) -> set[str]:
    refs: set[str] = set()
    for root_name in ("preprocess", "evaluation"):
        root = task_dir / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            if path.suffix not in {".py", ".json", ".md", ".txt", ".yaml", ".yml"}:
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            refs.update(_TOOLATHLON_CREDENTIAL_REF_RE.findall(text))
    return refs


def _toolathlon_credential_refs_for_task(
    task_dir: Path, task_config: dict[str, Any], *, variant: str
) -> set[str]:
    if variant != "official":
        return set()
    refs = set(_toolathlon_referenced_config_paths(task_dir))
    servers = set(task_config.get("needed_mcp_servers", []))
    for spec in _TOOLATHLON_CREDENTIAL_SPECS:
        if servers & set(spec.trigger_servers):
            refs.add(spec.path)
    return refs


def _toolathlon_instruction(task_dir: Path) -> str:
    system_prompt = (task_dir / "docs" / "agent_system_prompt.md").read_text()
    prompt = (task_dir / "docs" / "task.md").read_text()
    system_prompt = system_prompt.replace(
        "!!<<<<||||workspace_dir||||>>>>!!", "/workspace/agent_workspace"
    )
    return system_prompt.rstrip() + "\n\n" + prompt.rstrip() + "\n"


def _toolathlon_task_toml(
    ctx: Any,
    *,
    task_name: str,
    mcp_servers: list[dict[str, Any]],
    variant: str,
    credential_refs: set[str],
) -> dict[str, Any]:
    benchmark_name = "toolathlon-gym" if variant == "gym" else "toolathlon"
    setup_commands: list[dict[str, Any]] = []
    credential_command = _toolathlon_credential_setup_command(credential_refs)
    if credential_command:
        setup_commands.append(
            {
                "command": credential_command,
                "cwd": "/workspace",
                "timeout_sec": 60.0,
                "env": _toolathlon_credential_setup_env(credential_refs),
            }
        )
    setup_commands.append(
        {
            "command": _toolathlon_setup_command(task_name=task_name, variant=variant),
            "cwd": "/workspace",
            "timeout_sec": 600.0,
        }
    )
    environment: dict[str, Any] = {
        "cpus": 4,
        "memory_mb": 8192,
        "storage_mb": 10240,
        "workdir": "/workspace/agent_workspace",
        "mcp_servers": mcp_servers,
        "setup_commands": setup_commands,
    }
    if variant == "gym":
        environment["env"] = _TOOLATHLON_GYM_ENV

    metadata: dict[str, Any] = {
        "benchmark": benchmark_name,
        "upstream_task_id": task_name,
        "upstream_repo": ctx.repo,
        "upstream_sha": ctx.resolved_sha,
    }
    if credential_refs:
        metadata["required_credential_files"] = sorted(credential_refs)
        metadata["credential_env_options"] = _toolathlon_credential_env_options(
            credential_refs
        )

    return {
        "schema_version": "1.3",
        "task": {
            "name": f"{benchmark_name}/{_safe_name(task_name)}",
            "description": f"{benchmark_name} task adapted from upstream source",
            "keywords": [benchmark_name, "mcp"],
        },
        "metadata": metadata,
        "agent": {"timeout_sec": 1800.0},
        "verifier": {"timeout_sec": 900.0},
        "environment": environment,
    }


def _toolathlon_required_credential_envs(credential_refs: set[str]) -> list[str]:
    envs: list[str] = []
    for ref in sorted(credential_refs):
        spec = _TOOLATHLON_CREDENTIALS_BY_PATH[ref]
        envs.extend((spec.env, spec.b64_env))
    return envs


def _toolathlon_credential_env_options(
    credential_refs: set[str],
) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for ref in sorted(credential_refs):
        spec = _TOOLATHLON_CREDENTIALS_BY_PATH[ref]
        options.append({"file": ref, "env": spec.env, "base64_env": spec.b64_env})
    return options


def _toolathlon_credential_setup_env(credential_refs: set[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for name in _toolathlon_required_credential_envs(credential_refs):
        env[name] = f"${{{name}:-}}"
    return env


def _toolathlon_credential_setup_command(credential_refs: set[str]) -> str | None:
    specs = [
        {
            "path": spec.path,
            "json_env": spec.env,
            "b64_env": spec.b64_env,
            "copy_paths": list(spec.copy_paths),
        }
        for ref in sorted(credential_refs)
        for spec in (_TOOLATHLON_CREDENTIALS_BY_PATH[ref],)
    ]
    if not specs:
        return None
    specs_json = json.dumps(specs, sort_keys=True)
    return "\n".join(
        [
            "set -e",
            f"SPECS={shlex.quote(specs_json)} /usr/local/bin/uv run python - <<'PY'",
            "import base64, json, os, pathlib, sys",
            "specs = json.loads(os.environ['SPECS'])",
            "workspace = pathlib.Path.cwd()",
            "missing = []",
            "for spec in specs:",
            "    target = workspace / spec['path']",
            "    payload = None",
            "    if target.exists():",
            "        payload = target.read_bytes()",
            "    else:",
            "        raw = os.environ.get(spec['json_env']) or ''",
            "        b64 = os.environ.get(spec['b64_env']) or ''",
            "        if raw:",
            "            payload = raw.encode()",
            "        elif b64:",
            "            try:",
            "                payload = base64.b64decode(b64)",
            "            except Exception as exc:",
            "                sys.stderr.write(",
            "                    f\"BenchFlow Toolathlon credential setup error: invalid base64 for {spec['path']}: {exc}\\n\"",
            "                )",
            "                sys.exit(66)",
            "        else:",
            "            missing.append(",
            "                f\"{spec['path']} ({spec['json_env']} or {spec['b64_env']})\"",
            "            )",
            "            continue",
            "    try:",
            "        json.loads(payload.decode())",
            "    except Exception as exc:",
            "        sys.stderr.write(",
            "            f\"BenchFlow Toolathlon credential setup error: invalid JSON for {spec['path']}: {exc}\\n\"",
            "        )",
            "        sys.exit(66)",
            "    if not target.exists():",
            "        target.parent.mkdir(parents=True, exist_ok=True)",
            "        target.write_bytes(payload)",
            "        target.chmod(0o644)",
            "    for rel in spec.get('copy_paths', []):",
            "        copy = workspace / rel",
            "        copy.parent.mkdir(parents=True, exist_ok=True)",
            "        copy.write_bytes(payload)",
            "        copy.chmod(0o644)",
            "if missing:",
            "    sys.stderr.write(",
            "        'BenchFlow Toolathlon credential setup error: missing '",
            "        + ', '.join(missing)",
            "        + '\\n'",
            "    )",
            "    sys.exit(66)",
            "PY",
        ]
    )


def _toolathlon_setup_command(*, task_name: str, variant: str) -> str:
    python_cmd = (
        "/opt/venv/bin/python3" if variant == "gym" else "/usr/local/bin/uv run python"
    )
    task_path = f"/workspace/tasks/finalpool/{task_name}"
    return "\n".join(
        [
            "set -e",
            "mkdir -p /workspace/agent_workspace",
            f"TASK_DIR={shlex.quote(task_path)}",
            'if [ -d "$TASK_DIR/initial_workspace" ]; then',
            '  cp -a "$TASK_DIR/initial_workspace/." /workspace/agent_workspace/',
            "fi",
            'if [ -f "$TASK_DIR/preprocess/main.py" ]; then',
            '  TASK_DIR="$TASK_DIR" AGENT_WORKSPACE=/workspace/agent_workspace '
            'PYTHONPATH="$TASK_DIR:/workspace:${PYTHONPATH:-}" '
            f"{python_cmd} - <<'PY'",
            "import os, runpy, sys",
            "task_dir = os.environ['TASK_DIR']",
            "sys.path.insert(0, task_dir)",
            "sys.path.insert(0, '/workspace')",
            "sys.argv = ['preprocess.main', '--agent_workspace', os.environ['AGENT_WORKSPACE']]",
            "runpy.run_module('preprocess.main', run_name='__main__')",
            "PY",
            "fi",
            'for private in "$TASK_DIR/groundtruth_workspace" "$TASK_DIR/evaluation"; do',
            '  if [ -e "$private" ]; then',
            '    chown -R root:root "$private"',
            '    chmod -R go-rwx "$private"',
            "  fi",
            "done",
            "chmod -R a+rwX /workspace/agent_workspace",
        ]
    )


def _toolathlon_test_sh(*, task_name: str, variant: str) -> str:
    python_cmd = (
        "/opt/venv/bin/python3" if variant == "gym" else "/usr/local/bin/uv run python"
    )
    task_path = f"/workspace/tasks/finalpool/{task_name}"
    interpreter_check = []
    if variant == "official":
        interpreter_check = [
            "if [ ! -x /usr/local/bin/uv ]; then",
            '  echo "BenchFlow Toolathlon verifier setup error: /usr/local/bin/uv is missing" >&2',
            "  exit 127",
            "fi",
        ]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -u",
            *interpreter_check,
            "mkdir -p /logs/verifier",
            "cd /workspace",
            f"TASK_DIR={shlex.quote(task_path)}",
            "AGENT_WORKSPACE=/workspace/agent_workspace",
            'GROUNDTRUTH="$TASK_DIR/groundtruth_workspace"',
            "RES_LOG=/logs/verifier/toolathlon_result.json",
            "EVAL_LOG=/logs/verifier/toolathlon_evaluator.log",
            "printf '{}' > \"$RES_LOG\"",
            "set +e",
            'TASK_DIR="$TASK_DIR" AGENT_WORKSPACE="$AGENT_WORKSPACE" '
            'GROUNDTRUTH="$GROUNDTRUTH" RES_LOG="$RES_LOG" '
            'PYTHONPATH="$TASK_DIR:/workspace:${PYTHONPATH:-}" '
            f"{python_cmd} - <<'PY' > \"$EVAL_LOG\" 2>&1",
            "import os, runpy, sys",
            "task_dir = os.environ['TASK_DIR']",
            "sys.path.insert(0, task_dir)",
            "sys.path.insert(0, '/workspace')",
            "sys.argv = [",
            "    'evaluation.main',",
            "    '--agent_workspace', os.environ['AGENT_WORKSPACE'],",
            "    '--groundtruth_workspace', os.environ['GROUNDTRUTH'],",
            "    '--res_log_file', os.environ['RES_LOG'],",
            "]",
            "runpy.run_module('evaluation.main', run_name='__main__')",
            "PY",
            "status=$?",
            "set -u",
            'cat "$EVAL_LOG"',
            'if [ "$status" -eq 0 ]; then',
            "  printf '{\"reward\": 1.0}\\n' > /logs/verifier/reward.json",
            "  printf '1.0\\n' > /logs/verifier/reward.txt",
            "elif grep -q 'Traceback (most recent call last):' \"$EVAL_LOG\"; then",
            '  echo "BenchFlow Toolathlon verifier setup error: evaluator crashed" >&2',
            '  exit "$status"',
            "else",
            "  printf '{\"reward\": 0.0}\\n' > /logs/verifier/reward.json",
            "  printf '0.0\\n' > /logs/verifier/reward.txt",
            "fi",
            "",
        ]
    )


_TOOLATHLON_SERVER_ALIASES = {
    "arxiv-latex": "arxiv-latex-mcp",
    "fetch": "npx-fetch",
    "rail_12306": "12306",
    "scholarly": "scholarly_search",
    "youtube-transcript": "youtube_transcript",
}


def _toolathlon_mcp_server(
    ctx: Any, server_name: str, *, variant: str
) -> dict[str, Any]:
    config_path = _toolathlon_mcp_config_path(ctx, server_name)
    data = yaml.safe_load(config_path.read_text())
    params = data.get("params") or {}
    command = _replace_toolathlon_placeholders(
        str(params.get("command", "")), variant=variant
    )
    args = [
        _replace_toolathlon_placeholders(str(arg), variant=variant)
        for arg in params.get("args", [])
    ]
    env = {
        str(key): _replace_toolathlon_placeholders(str(value), variant=variant)
        for key, value in (params.get("env") or {}).items()
    }
    env = _normalize_toolathlon_env(env, variant=variant)
    if variant == "official" and (data.get("name") or server_name) == "google_calendar":
        env = dict(env)
        env["CALENDAR_OAUTH_PATH"] = "/workspace/configs/gcp-oauth.keys.json"
        env["CALENDAR_CREDENTIALS_PATH"] = (
            "/workspace/agent_workspace/.toolathlon/calendar_credentials.json"
        )
    cwd = params.get("cwd")
    cwd = _replace_toolathlon_placeholders(str(cwd), variant=variant) if cwd else None
    if variant == "official" and command in {"uv", "uvx"}:
        command = f"/usr/local/bin/{command}"
        if args:
            args = [_TOOLATHLON_UVX_PACKAGE_PINS.get(arg, arg) for arg in args]
    payload: dict[str, Any] = {
        "name": str(data.get("name") or server_name),
        "transport": "stdio",
        "command": command,
        "args": args,
        "exclude_tags": [_NOOP_EXCLUDE_TAG],
    }
    if cwd:
        payload["cwd"] = cwd
    if env:
        payload["env"] = env
    return payload


def _normalize_toolathlon_env(env: dict[str, str], *, variant: str) -> dict[str, str]:
    if variant != "gym" or not env:
        return env
    pg_keys = {
        "PGHOST",
        "PG_HOST",
        "PGPORT",
        "PG_PORT",
        "PGUSER",
        "PG_USER",
        "PGPASSWORD",
        "PG_PASSWORD",
        "PGDATABASE",
        "PG_DATABASE",
    }
    if not (pg_keys & set(env)):
        return env
    normalized = dict(env)
    normalized.update(
        {
            "PGHOST": "postgres",
            "PG_HOST": "postgres",
            "PGPORT": "5432",
            "PG_PORT": "5432",
            "PGUSER": "eigent",
            "PG_USER": "eigent",
            "PGPASSWORD": "camel",
            "PG_PASSWORD": "camel",
            "PGDATABASE": "toolathlon_gym",
            "PG_DATABASE": "toolathlon_gym",
        }
    )
    return normalized


def _toolathlon_mcp_config_path(ctx: Any, server_name: str) -> Path:
    mapped = _TOOLATHLON_SERVER_ALIASES.get(server_name, server_name)
    candidates = [
        mapped,
        mapped.replace("_", "-"),
        mapped.replace("-", "_"),
        server_name,
        server_name.replace("_", "-"),
        server_name.replace("-", "_"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = ctx.repo_root / "configs" / "mcp_servers" / f"{candidate}.yaml"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find MCP server config for {server_name!r}")


def _replace_toolathlon_placeholders(value: str, *, variant: str) -> str:
    local_servers = (
        "/opt/local_servers" if variant == "gym" else "/workspace/local_servers"
    )
    replacements = {
        "${local_servers_paths}": local_servers,
        "${agent_workspace}": "/workspace/agent_workspace",
        "${task_dir}": "/workspace/tasks/finalpool",
        "${local_binary_paths}": "/workspace/local_binary",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)

    def token_replacement(match: re.Match[str]) -> str:
        token_name = match.group(1)
        if variant == "official" and token_name in _TOOLATHLON_TOKEN_DEFAULTS:
            return _TOOLATHLON_TOKEN_DEFAULTS[token_name]
        name = token_name.upper()
        return f"${{TOOLATHLON_{name}}}"

    return _TOKEN_PLACEHOLDER_RE.sub(token_replacement, value)


def _toolathlon_dockerfile(ctx: Any, *, variant: str) -> str:
    repo_url = f"https://github.com/{ctx.repo}.git"
    if variant == "official":
        return f"""FROM lockon0927/toolathlon-task-image:1016beta
ENV PATH="/usr/local/bin:/root/.local/bin:$PATH"
WORKDIR /workspace
RUN rm -rf /tmp/toolathlon-src \\
    && git clone {repo_url} /tmp/toolathlon-src \\
    && cd /tmp/toolathlon-src \\
    && git checkout {ctx.resolved_sha} \\
    && rsync -a --exclude .git /tmp/toolathlon-src/ /workspace/ \\
    && rm -rf /tmp/toolathlon-src
RUN if [ -f /workspace/configs/global_configs_example.py ] && [ ! -f /workspace/configs/global_configs.py ]; then \\
        cp /workspace/configs/global_configs_example.py /workspace/configs/global_configs.py; \\
    fi
RUN if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi \\
    && cp "$(command -v uv)" /usr/local/bin/uv \\
    && if command -v uvx >/dev/null 2>&1; then cp "$(command -v uvx)" /usr/local/bin/uvx; else ln -sf /usr/local/bin/uv /usr/local/bin/uvx; fi \\
    && chmod 755 /usr/local/bin/uv /usr/local/bin/uvx
RUN [ ! -e /workspace/utils/local_servers ] || chmod -R a+rwX /workspace/utils/local_servers
CMD ["/bin/bash"]
"""
    return f"""FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \\
    curl wget git ca-certificates gnupg python3 python3-pip rsync postgresql-client \\
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libdbus-1-3 \\
    libatspi2.0-0 libx11-6 libxcomposite1 libxdamage1 libxext6 libxfixes3 \\
    libxrandr2 libgbm1 libxcb1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2 \\
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \\
    && cp /root/.local/bin/uv /usr/local/bin/uv \\
    && cp /root/.local/bin/uvx /usr/local/bin/uvx \\
    && chmod 755 /usr/local/bin/uv /usr/local/bin/uvx
ENV PATH="/usr/local/bin:/root/.local/bin:$PATH"
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \\
    && apt-get install -y nodejs \\
    && rm -rf /var/lib/apt/lists/* \\
    && npm install -g npm@latest
WORKDIR /workspace
RUN git init /workspace \\
    && git remote add origin {repo_url} \\
    && git fetch --depth 1 origin {ctx.resolved_sha} \\
    && git checkout FETCH_HEAD
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python \\
    "camel-ai" "anthropic" "psycopg2-binary" "openpyxl" "python-docx" "python-pptx" \\
    "termcolor" "aiofiles" "psutil" "addict" "arxiv" "bibtexparser" "canvasapi" \\
    "prompt_toolkit"
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
RUN playwright install chromium || true
RUN cp -a /workspace/local_servers /opt/local_servers
RUN for dir in \\
        /opt/local_servers/Calendar-Autoauth-MCP-Server \\
        /opt/local_servers/google-forms-mcp \\
        /opt/local_servers/mcp-google-sheets \\
        /opt/local_servers/youtube-mcp-server \\
        /opt/local_servers/filesystem \\
        /opt/local_servers/HowToCook-mcp \\
        /opt/local_servers/servers; do \\
    [ -f "$dir/package.json" ] && \\
        echo "=== $dir ===" && cd "$dir" && npm install && (npm run build 2>/dev/null || true) && cd /workspace || true; \\
done
RUN for dir in \\
        /opt/local_servers/arxiv-mcp-server \\
        /opt/local_servers/arxiv-latex-mcp \\
        /opt/local_servers/yahoo-finance-mcp \\
        /opt/local_servers/emails-mcp \\
        /opt/local_servers/mcp-snowflake-server \\
        /opt/local_servers/mcp-scholarly \\
        /opt/local_servers/Office-Word-MCP-Server \\
        /opt/local_servers/Office-PowerPoint-MCP-Server \\
        /opt/local_servers/excel-mcp-server \\
        /opt/local_servers/pdf-tools-mcp \\
        /opt/local_servers/mcp-youtube-transcript \\
        /opt/local_servers/cli-mcp-server; do \\
    [ -f "$dir/pyproject.toml" ] && \\
        echo "=== $dir ===" && cd "$dir" && uv sync || true && cd /workspace || true; \\
done
RUN chmod -R a+rwX /opt/local_servers
CMD ["/bin/bash"]
"""


_TOOLATHLON_GYM_ENV = {
    "PGHOST": "postgres",
    "PG_HOST": "postgres",
    "PGPORT": "5432",
    "PGUSER": "eigent",
    "PGPASSWORD": "camel",
    "PGDATABASE": "toolathlon_gym",
    "LOCAL_SERVERS_PATH": "/opt/local_servers",
    "PYTHON_BIN": "/opt/venv/bin/python3",
}


_TOOLATHLON_GYM_COMPOSE = """services:
  main:
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      PGHOST: postgres
      PG_HOST: postgres
      PGPORT: "5432"
      PGUSER: eigent
      PGPASSWORD: camel
      PGDATABASE: toolathlon_gym
      LOCAL_SERVERS_PATH: /opt/local_servers
      PYTHON_BIN: /opt/venv/bin/python3
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: toolathlon_gym
      POSTGRES_USER: eigent
      POSTGRES_PASSWORD: camel
    volumes:
      - ./db/init.sql.gz:/docker-entrypoint-initdb.d/init.sql.gz
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U eigent -d toolathlon_gym"]
      interval: 5s
      timeout: 5s
      retries: 10
"""
