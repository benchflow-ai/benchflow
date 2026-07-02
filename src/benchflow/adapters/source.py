"""Source-level benchmark adapters for ``bench eval run --source-repo``.

These adapters sit between remote source resolution and ``Evaluation`` task
discovery. If a source already contains BenchFlow-native task directories, it is
returned unchanged. If it is a known foreign benchmark source, the adapter
materializes a cache of native task directories and returns that cache as the
resolved source path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import shlex
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import tomli_w
import yaml

from benchflow._utils.benchmark_repos import ResolvedSource

_ADAPTER_VERSION = "2026-07-02.3"
_NOOP_EXCLUDE_TAG = "__benchflow_exclude_no_tools__"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TOKEN_PLACEHOLDER_RE = re.compile(r"\$\{token\.([A-Za-z0-9_]+)\}")
_MISSING_CONFIG_REF_RE = re.compile(
    r"configs/(?:gcp-service_account\.keys\.json|google_credentials\.json)"
)
_TOOLATHLON_UVX_PACKAGE_PINS = {
    "office-word-mcp-server": "office-word-mcp-server==1.1.11",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SourceContext:
    resolved: ResolvedSource
    repo: str
    repo_root: Path
    source_path: str
    source_root: Path
    resolved_sha: str


def adapt_resolved_source_if_needed(resolved: ResolvedSource) -> ResolvedSource:
    """Return a BenchFlow-native source, adapting known foreign benchmarks."""

    if _contains_native_task(resolved.path):
        return resolved

    ctx = _SourceContext(
        resolved=resolved,
        repo=str(resolved.provenance.get("repo") or ""),
        repo_root=_infer_repo_root(resolved),
        source_path=str(resolved.provenance.get("path") or ""),
        source_root=resolved.path,
        resolved_sha=str(resolved.provenance.get("resolved_sha") or "unknown"),
    )

    adapter_name = _detect_adapter(ctx)
    if adapter_name is None:
        return resolved

    output_dir = _adapted_output_dir(ctx, adapter_name)
    marker_path = output_dir / ".benchflow-source-adapter.json"
    marker = {
        "adapter": adapter_name,
        "version": _ADAPTER_VERSION,
        "repo": ctx.repo,
        "resolved_sha": ctx.resolved_sha,
        "source_path": ctx.source_path,
    }
    if _marker_matches(marker_path, marker):
        return _adapted_resolved(ctx, output_dir, adapter_name)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    if adapter_name == "mcp-atlas":
        _materialize_mcp_atlas(ctx, output_dir)
    elif adapter_name == "toolathlon-gym":
        _materialize_toolathlon(ctx, output_dir, variant="gym")
    elif adapter_name == "toolathlon":
        _materialize_toolathlon(ctx, output_dir, variant="official")
    else:  # pragma: no cover - impossible without a detector bug
        raise RuntimeError(f"unknown source adapter: {adapter_name}")

    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    return _adapted_resolved(ctx, output_dir, adapter_name)


def _contains_native_task(path: Path) -> bool:
    if (path / "task.toml").is_file() or (path / "task.md").is_file():
        return True
    if not path.is_dir():
        return False
    return any(
        child.is_dir()
        and ((child / "task.toml").is_file() or (child / "task.md").is_file())
        for child in path.iterdir()
    )


def _infer_repo_root(resolved: ResolvedSource) -> Path:
    local_path = resolved.path.resolve(strict=True)
    source_path = str(resolved.provenance.get("path") or "").strip("/")
    if source_path:
        root = local_path
        for _ in Path(source_path).parts:
            root = root.parent
        if (root / ".git").exists():
            return root
    for candidate in [local_path, *local_path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return local_path


def _detect_adapter(ctx: _SourceContext) -> str | None:
    repo = ctx.repo.lower()
    if _mcp_atlas_csv(ctx) is not None:
        return "mcp-atlas"
    has_toolathlon_tasks = _toolathlon_tasks_root(ctx, variant="any") is not None
    if has_toolathlon_tasks and (
        "toolathlon_gym" in repo or (ctx.repo_root / "db" / "init.sql.gz").is_file()
    ):
        return "toolathlon-gym"
    if has_toolathlon_tasks and (
        repo.endswith("/toolathlon")
        or (ctx.repo_root / "global_preparation").is_dir()
        or (ctx.repo_root / "configs" / "users_data.json").is_file()
    ):
        return "toolathlon"
    return None


def _adapted_output_dir(ctx: _SourceContext, adapter_name: str) -> Path:
    key = "|".join(
        [adapter_name, _ADAPTER_VERSION, ctx.repo, ctx.resolved_sha, ctx.source_path]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    repo_slug = _safe_name(ctx.repo.replace("/", "__"))
    return Path.cwd() / ".cache" / "source-adapters" / adapter_name / f"{repo_slug}__{digest}"


def _marker_matches(path: Path, expected: dict[str, Any]) -> bool:
    try:
        return json.loads(path.read_text()) == expected
    except (OSError, json.JSONDecodeError):
        return False


def _adapted_resolved(
    ctx: _SourceContext, output_dir: Path, adapter_name: str
) -> ResolvedSource:
    provenance = dict(ctx.resolved.provenance)
    provenance["local_path"] = str(output_dir)
    provenance["adapter"] = {
        "name": adapter_name,
        "version": _ADAPTER_VERSION,
        "source_local_path": str(ctx.source_root),
        "source_repo_root": str(ctx.repo_root),
    }
    return ResolvedSource(path=output_dir, provenance=provenance)


def _safe_name(value: str) -> str:
    slug = _SAFE_NAME_RE.sub("-", value).strip("-._")
    return slug or "task"


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _write_toml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(payload))


def _adapter_resource_text(name: str) -> str:
    return resources.files("benchflow.adapters.resources").joinpath(name).read_text()


def _mcp_atlas_csv(ctx: _SourceContext) -> Path | None:
    candidates = [
        ctx.source_root / "sample_tasks.csv",
        ctx.source_root / "services" / "mcp_eval" / "sample_tasks.csv",
        ctx.repo_root / "services" / "mcp_eval" / "sample_tasks.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _materialize_mcp_atlas(ctx: _SourceContext, output_dir: Path) -> None:
    csv_path = _mcp_atlas_csv(ctx)
    if csv_path is None:
        raise ValueError("MCP Atlas source is missing services/mcp_eval/sample_tasks.csv")

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"MCP Atlas task CSV is empty: {csv_path}")

    for row in rows:
        task_id = str(row.get("TASK") or "").strip()
        if not task_id:
            continue
        task_dir = output_dir / _safe_name(task_id)
        tools = _json_list(row.get("ENABLED_TOOLS"))
        claims = _json_list(row.get("GTFA_CLAIMS"))
        prompt = str(row.get("PROMPT") or "").strip()
        enabled_servers = _mcp_atlas_enabled_servers(tools)

        _write_text(task_dir / "instruction.md", prompt + "\n")
        _write_toml(
            task_dir / "task.toml",
            {
                "schema_version": "1.3",
                "task": {
                    "name": f"mcp-atlas/{_safe_name(task_id)}",
                    "description": "MCP Atlas tool-use task",
                    "keywords": ["mcp-atlas", "mcp"],
                },
                "metadata": {
                    "benchmark": "mcp-atlas",
                    "upstream_task_id": task_id,
                },
                "agent": {"timeout_sec": 900.0},
                "verifier": {
                    "timeout_sec": 900.0,
                    "env": {
                        "OPENROUTER_API_KEY": "${OPENROUTER_API_KEY}",
                        "MCP_ATLAS_JUDGE_MODEL": "${MCP_ATLAS_JUDGE_MODEL:-qwen/qwen-plus}",
                    },
                },
                "environment": {
                    "cpus": 4,
                    "memory_mb": 8192,
                    "storage_mb": 10240,
                    "workdir": "/workspace",
                    "mcp_servers": [
                        {
                            "name": "mcp-server",
                            "transport": "streamable-http",
                            "url": "http://localhost:18765/mcp",
                            "tools": tools or None,
                        }
                    ],
                },
            },
        )
        _write_text(
            task_dir / "environment" / "Dockerfile",
            "FROM ghcr.io/scaleapi/mcp-atlas:1.2.5\n"
            "RUN uv pip install --system fastmcp==3.4.2 httpx==0.28.1\n"
            "COPY enabled_tools.txt /enabled_tools.txt\n"
            "COPY mcp_bridge.py /mcp_bridge.py\n"
            "ENV MCP_ENABLED_TOOLS_FILE=/enabled_tools.txt\n",
        )
        _write_text(
            task_dir / "environment" / "docker-compose.yaml",
            "services:\n"
            "  main:\n"
            "    environment:\n"
            f"      ENABLED_SERVERS: {json.dumps(','.join(enabled_servers))}\n"
            "    command:\n"
            "      - bash\n"
            "      - -lc\n"
            "      - |\n"
            "        set -e\n"
            "        uv run python -m uvicorn agent_environment.main:app --host 0.0.0.0 --port 1984 &\n"
            "        upstream_pid=$!\n"
            "        trap 'kill ${upstream_pid} 2>/dev/null || true' EXIT\n"
            "        python /mcp_bridge.py\n",
        )
        _write_text(task_dir / "environment" / "enabled_tools.txt", "\n".join(tools) + "\n")
        _write_text(
            task_dir / "environment" / "mcp_bridge.py",
            _adapter_resource_text("mcp_atlas_bridge.py"),
        )
        _write_text(
            task_dir / "tests" / "claims.json",
            json.dumps({"prompt": prompt, "claims": claims}, indent=2, ensure_ascii=False)
            + "\n",
        )
        _write_text(
            task_dir / "tests" / "mcp_atlas_judge.py",
            _adapter_resource_text("mcp_atlas_judge.py"),
        )
        _write_text(
            task_dir / "tests" / "test.sh",
            "#!/usr/bin/env bash\n"
            "set -u\n"
            "python3 /tests/mcp_atlas_judge.py /tests/claims.json\n",
        )


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        return []
    result: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            result.append(item["name"])
    return result


def _mcp_atlas_enabled_servers(tools: list[str]) -> list[str]:
    servers = sorted({tool.split("_", 1)[0] for tool in tools if "_" in tool})
    return servers


def _toolathlon_tasks_root(ctx: _SourceContext, *, variant: str) -> Path | None:
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


def _materialize_toolathlon(
    ctx: _SourceContext, output_dir: Path, *, variant: str
) -> None:
    tasks_root = _toolathlon_tasks_root(ctx, variant=variant)
    if tasks_root is None:
        raise ValueError(f"{variant} source does not contain tasks/finalpool task dirs")

    skipped: list[dict[str, str]] = []
    for upstream_task in sorted(tasks_root.iterdir()):
        if not (upstream_task / "task_config.json").is_file():
            continue
        unsupported_reason = _toolathlon_unsupported_reason(
            ctx, upstream_task, variant=variant
        )
        if unsupported_reason is not None:
            skipped.append({"task_id": upstream_task.name, "reason": unsupported_reason})
            continue
        task_name = upstream_task.name
        task_dir = output_dir / _safe_name(task_name)
        task_config = json.loads((upstream_task / "task_config.json").read_text())
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
    if skipped:
        logger.warning(
            "Skipped %d unsupported %s tasks while adapting %s",
            len(skipped),
            "Toolathlon-GYM" if variant == "gym" else "Toolathlon",
            ctx.repo,
        )
        _write_text(
            output_dir / ".benchflow-source-adapter-skipped.json",
            json.dumps({"skipped": skipped}, indent=2, sort_keys=True) + "\n",
        )


def _toolathlon_unsupported_reason(
    ctx: _SourceContext, task_dir: Path, *, variant: str
) -> str | None:
    if variant != "official":
        return None
    missing = sorted(
        {
            ref
            for ref in _toolathlon_referenced_config_paths(task_dir)
            if not (ctx.repo_root / ref).exists()
        }
    )
    if not missing:
        return None
    return "references missing repo config: " + ", ".join(missing)


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
            refs.update(_MISSING_CONFIG_REF_RE.findall(text))
    return refs


def _toolathlon_instruction(task_dir: Path) -> str:
    system_prompt = (task_dir / "docs" / "agent_system_prompt.md").read_text()
    prompt = (task_dir / "docs" / "task.md").read_text()
    system_prompt = system_prompt.replace(
        "!!<<<<||||workspace_dir||||>>>>!!", "/workspace/agent_workspace"
    )
    return system_prompt.rstrip() + "\n\n" + prompt.rstrip() + "\n"


def _toolathlon_task_toml(
    ctx: _SourceContext,
    *,
    task_name: str,
    mcp_servers: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    benchmark_name = "toolathlon-gym" if variant == "gym" else "toolathlon"
    environment: dict[str, Any] = {
        "cpus": 4,
        "memory_mb": 8192,
        "storage_mb": 10240,
        "workdir": "/workspace/agent_workspace",
        "mcp_servers": mcp_servers,
        "setup_commands": [
            {
                "command": _toolathlon_setup_command(task_name=task_name, variant=variant),
                "cwd": "/workspace",
                "timeout_sec": 600.0,
            }
        ],
    }
    if variant == "gym":
        environment["env"] = _TOOLATHLON_GYM_ENV

    return {
        "schema_version": "1.3",
        "task": {
            "name": f"{benchmark_name}/{_safe_name(task_name)}",
            "description": f"{benchmark_name} task adapted from upstream source",
            "keywords": [benchmark_name, "mcp"],
        },
        "metadata": {
            "benchmark": benchmark_name,
            "upstream_task_id": task_name,
            "upstream_repo": ctx.repo,
            "upstream_sha": ctx.resolved_sha,
        },
        "agent": {"timeout_sec": 1800.0},
        "verifier": {"timeout_sec": 900.0},
        "environment": environment,
    }


def _toolathlon_setup_command(*, task_name: str, variant: str) -> str:
    python_cmd = "/opt/venv/bin/python3" if variant == "gym" else "/usr/local/bin/uv run python"
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
            "  TASK_DIR=\"$TASK_DIR\" AGENT_WORKSPACE=/workspace/agent_workspace "
            "PYTHONPATH=\"$TASK_DIR:/workspace:${PYTHONPATH:-}\" "
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
    python_cmd = "/opt/venv/bin/python3" if variant == "gym" else "/usr/local/bin/uv run python"
    task_path = f"/workspace/tasks/finalpool/{task_name}"
    interpreter_check = []
    if variant == "official":
        interpreter_check = [
            'if [ ! -x /usr/local/bin/uv ]; then',
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
            "TASK_DIR=\"$TASK_DIR\" AGENT_WORKSPACE=\"$AGENT_WORKSPACE\" "
            "GROUNDTRUTH=\"$GROUNDTRUTH\" RES_LOG=\"$RES_LOG\" "
            "PYTHONPATH=\"$TASK_DIR:/workspace:${PYTHONPATH:-}\" "
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
            "cat \"$EVAL_LOG\"",
            "if [ \"$status\" -eq 0 ]; then",
            "  printf '{\"reward\": 1.0}\\n' > /logs/verifier/reward.json",
            "  printf '1.0\\n' > /logs/verifier/reward.txt",
            "elif grep -q 'Traceback (most recent call last):' \"$EVAL_LOG\"; then",
            '  echo "BenchFlow Toolathlon verifier setup error: evaluator crashed" >&2',
            "  exit \"$status\"",
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
    ctx: _SourceContext, server_name: str, *, variant: str
) -> dict[str, Any]:
    config_path = _toolathlon_mcp_config_path(ctx, server_name)
    data = yaml.safe_load(config_path.read_text())
    params = data.get("params") or {}
    command = _replace_toolathlon_placeholders(str(params.get("command", "")), variant=variant)
    args = [
        _replace_toolathlon_placeholders(str(arg), variant=variant)
        for arg in params.get("args", [])
    ]
    env = {
        str(key): _replace_toolathlon_placeholders(str(value), variant=variant)
        for key, value in (params.get("env") or {}).items()
    }
    env = _normalize_toolathlon_env(env, variant=variant)
    cwd = params.get("cwd")
    cwd = _replace_toolathlon_placeholders(str(cwd), variant=variant) if cwd else None
    if variant == "official" and command in {"uv", "uvx"}:
        command = f"/usr/local/bin/{command}"
        if args:
            args = [
                _TOOLATHLON_UVX_PACKAGE_PINS.get(arg, arg)
                for arg in args
            ]
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


def _toolathlon_mcp_config_path(ctx: _SourceContext, server_name: str) -> Path:
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
    local_servers = "/opt/local_servers" if variant == "gym" else "/workspace/local_servers"
    replacements = {
        "${local_servers_paths}": local_servers,
        "${agent_workspace}": "/workspace/agent_workspace",
        "${task_dir}": "/workspace/tasks/finalpool",
        "${local_binary_paths}": "/workspace/local_binary",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)

    def token_replacement(match: re.Match[str]) -> str:
        name = match.group(1).upper()
        return f"${{TOOLATHLON_{name}}}"

    return _TOKEN_PLACEHOLDER_RE.sub(token_replacement, value)


def _toolathlon_dockerfile(ctx: _SourceContext, *, variant: str) -> str:
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
