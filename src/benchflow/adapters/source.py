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
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import tomli_w

from benchflow._utils.benchmark_repos import ResolvedSource
from benchflow.adapters._toolathlon import materialize_toolathlon, toolathlon_tasks_root

_ADAPTER_VERSION = "2026-07-02.8"
_NOOP_EXCLUDE_TAG = "__benchflow_exclude_no_tools__"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
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
        materialize_toolathlon(ctx, output_dir, variant="gym")
    elif adapter_name == "toolathlon":
        materialize_toolathlon(ctx, output_dir, variant="official")
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
    has_toolathlon_tasks = toolathlon_tasks_root(ctx, variant="any") is not None
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
    return (
        Path.cwd()
        / ".cache"
        / "source-adapters"
        / adapter_name
        / f"{repo_slug}__{digest}"
    )


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
        raise ValueError(
            "MCP Atlas source is missing services/mcp_eval/sample_tasks.csv"
        )

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
        _write_text(
            task_dir / "environment" / "enabled_tools.txt", "\n".join(tools) + "\n"
        )
        _write_text(
            task_dir / "environment" / "mcp_bridge.py",
            _adapter_resource_text("mcp_atlas_bridge.py"),
        )
        _write_text(
            task_dir / "tests" / "claims.json",
            json.dumps(
                {"prompt": prompt, "claims": claims}, indent=2, ensure_ascii=False
            )
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
