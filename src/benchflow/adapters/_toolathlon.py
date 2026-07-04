"""Toolathlon source adapter materialization helpers."""

from __future__ import annotations

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

from benchflow.adapters import _toolathlon_services

logger = logging.getLogger(__name__)

_NOOP_EXCLUDE_TAG = "__benchflow_exclude_no_tools__"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TOOLATHLON_UVX_PACKAGE_PINS = {
    "office-word-mcp-server": "office-word-mcp-server==1.1.11",
}
_TOOLATHLON_NPX_PACKAGE_PINS = {
    # Pin so mcp-remote's on-disk token path (mcp-remote-<ver>/) matches the
    # OAuth dir injected by the notion mcp-auth setup command.
    "mcp-remote": "mcp-remote@0.1.37",
}
# Absolute location the notion_official server reads its OAuth token from (the
# upstream config uses a cwd-relative ./configs/.mcp-auth).
_TOOLATHLON_NOTION_MCP_AUTH_DIR = "/workspace/configs/.mcp-auth"
# The pre-registered OAuth client info (captured at token-authorization time).
# Passing it via --static-oauth-client-info stops mcp-remote from dynamically
# re-registering a new client on each run (which invalidates the stored token
# and forces an interactive re-auth); with it, mcp-remote refreshes the injected
# token headlessly. The hash is derived from the fixed server URL, so it is
# stable as long as the pinned mcp-remote version + notion MCP URL don't change.
_TOOLATHLON_NOTION_CLIENT_INFO = (
    f"{_TOOLATHLON_NOTION_MCP_AUTH_DIR}/mcp-remote-0.1.37/"
    "cb42d1a06ae8db4e5585a26f2e5ca947_client_info.json"
)


@dataclass(frozen=True)
class _ToolathlonCredentialSpec:
    path: str
    env: str
    b64_env: str
    trigger_servers: tuple[str, ...] = ()
    copy_paths: tuple[str, ...] = ()
    # "json" payloads are validated as JSON before writing; "pem" ones (e.g. a
    # Snowflake private key) are opaque bytes written verbatim.
    content_format: str = "json"
    # Private keys must not be world-readable; JSON creds keep 0644.
    file_mode: int = 0o644


_TOOLATHLON_CREDENTIAL_SPECS = (
    _ToolathlonCredentialSpec(
        path="configs/gcp-service_account.keys.json",
        env="TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON",
        b64_env="TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON_B64",
        trigger_servers=("google-cloud",),
    ),
    _ToolathlonCredentialSpec(
        path="configs/google_credentials.json",
        env="TOOLATHLON_GOOGLE_CREDENTIALS_JSON",
        b64_env="TOOLATHLON_GOOGLE_CREDENTIALS_JSON_B64",
        trigger_servers=("google_calendar", "google_sheet"),
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
    _ToolathlonCredentialSpec(
        path="configs/snowflake_rsa_key.p8",
        env="TOOLATHLON_SNOWFLAKE_RSA_KEY",
        b64_env="TOOLATHLON_SNOWFLAKE_RSA_KEY_B64",
        trigger_servers=("snowflake",),
        content_format="pem",
        file_mode=0o600,
    ),
)
_TOOLATHLON_CREDENTIALS_BY_PATH = {
    spec.path: spec for spec in _TOOLATHLON_CREDENTIAL_SPECS
}
_TOOLATHLON_CREDENTIAL_REF_RE = re.compile(
    "|".join(re.escape(spec.path) for spec in _TOOLATHLON_CREDENTIAL_SPECS)
)

# ``token.X`` values baked into the container-side global
# ``token_key_session.py`` (see ``_toolathlon_container._GLOBAL_TOKENS``),
# resolved host-side from BenchFlow's environment/.env at setup time. Most are
# secrets; a few (Snowflake warehouse/schema, the Notion source/eval page URLs)
# are non-secret per-deployment config carried through the same channel.
_TOOLATHLON_TOKEN_SECRET_ENVS = (
    "TOOLATHLON_GCP_PROJECT_ID",
    "TOOLATHLON_MAPS_API_KEY",
    "TOOLATHLON_SERPER_API_KEY",
    "TOOLATHLON_GITHUB_TOKEN",
    "TOOLATHLON_HF_TOKEN",
    "TOOLATHLON_WANDB_API_KEY",
    "TOOLATHLON_NOTION_KEY",
    "TOOLATHLON_NOTION_KEY_EVAL",
    "TOOLATHLON_NOTION_SOURCE_PAGE_URL",
    "TOOLATHLON_NOTION_EVAL_PAGE_URL",
    "TOOLATHLON_GOOGLE_CLIENT_ID",
    "TOOLATHLON_GOOGLE_CLIENT_SECRET",
    "TOOLATHLON_GOOGLE_REFRESH_TOKEN",
    "TOOLATHLON_SNOWFLAKE_ACCOUNT",
    "TOOLATHLON_SNOWFLAKE_USER",
    "TOOLATHLON_SNOWFLAKE_ROLE",
    "TOOLATHLON_SNOWFLAKE_WAREHOUSE",
    "TOOLATHLON_SNOWFLAKE_DATABASE",
    "TOOLATHLON_SNOWFLAKE_SCHEMA",
)

_TOOLATHLON_CONTAINER_MODULE_PATH = "/workspace/.toolathlon/toolathlon_container.py"

# MCP server flags whose value is a working DIRECTORY the server (and the task's
# evaluator) expect to exist. Upstream servers create these lazily, so an agent
# that never exercises the server leaves the directory absent and evaluators
# that ``os.listdir`` it crash. The launcher ``mkdir -p``s these. Explicit
# allow-list, not a heuristic: every entry takes a directory (never a file).
_TOOLATHLON_MCP_DIR_ARG_FLAGS = frozenset(
    {
        "--storage-path",
        "--attachment_download_path",
        "--attachment_upload_path",
        "--email_export_path",
    }
)


def _toolathlon_container_python(variant: str) -> str:
    # Absolute so FastMCP can spawn the launcher even without PATH in the MCP
    # server subprocess environment.
    return "/opt/venv/bin/python3" if variant == "gym" else "/usr/bin/python3"


def _toolathlon_container_module_source() -> str:
    return (
        resources.files("benchflow.adapters")
        .joinpath("_toolathlon_container.py")
        .read_text()
    )


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
        services = (
            _toolathlon_services.required_services(upstream_task)
            if variant == "official"
            else set()
        )
        # Any task using the notion server duplicates its source page in
        # preprocess via notion_official (mcp-remote OAuth), so it needs the
        # injected auth even though notion_official isn't in needed_mcp_servers.
        needs_notion_auth = variant == "official" and "notion" in (
            task_config.get("needed_mcp_servers") or []
        )
        mcp_servers = []
        for server_name in task_config.get("needed_mcp_servers", []):
            try:
                mcp_servers.append(
                    _toolathlon_mcp_server(
                        ctx, server_name, variant=variant, task_name=task_name
                    )
                )
            except FileNotFoundError:
                # ``task_config`` may name a local harness tool (e.g.
                # ``web_search``) rather than an MCP server — upstream resolves
                # those from its agent-tool registry, so no configs/mcp_servers
                # yaml exists. Drop it instead of failing the adaptation.
                logger.warning(
                    "Toolathlon task %s: no MCP config for %r; dropping server",
                    task_name,
                    server_name,
                )

        _write_text(task_dir / "instruction.md", _toolathlon_instruction(upstream_task))
        _write_toml(
            task_dir / "task.toml",
            _toolathlon_task_toml(
                ctx,
                task_name=task_name,
                mcp_servers=mcp_servers,
                variant=variant,
                credential_refs=credential_refs,
                services=services,
                needs_notion_auth=needs_notion_auth,
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
        elif services:
            _toolathlon_services.apply_service_sidecars(task_dir, services)
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


def _toolathlon_notion_mcp_auth_command() -> str:
    """Unpack the mcp-remote OAuth token and patch ``notion_official.yaml`` so the
    preprocess page-duplication (which spawns mcp-remote from that yaml, not from
    benchflow's MCP wiring) pins the version, uses the absolute auth dir, and
    reuses the pre-registered client — refreshing the token headlessly."""
    patch = "\n".join(
        [
            "/usr/bin/python3 - <<'PY'",
            "import yaml",
            "p = '/workspace/configs/mcp_servers/notion_official.yaml'",
            "try:",
            "    d = yaml.safe_load(open(p)) or {}",
            "except FileNotFoundError:",
            "    raise SystemExit(0)",
            "params = d.setdefault('params', {})",
            "args = ['mcp-remote@0.1.37' if a == 'mcp-remote' else a "
            "for a in (params.get('args') or [])]",
            "if '--static-oauth-client-info' not in args:",
            f"    args += ['--static-oauth-client-info', '@{_TOOLATHLON_NOTION_CLIENT_INFO}']",
            "params['args'] = args",
            "params.setdefault('env', {})['MCP_REMOTE_CONFIG_DIR'] = "
            f"'{_TOOLATHLON_NOTION_MCP_AUTH_DIR}'",
            "yaml.safe_dump(d, open(p, 'w'))",
            "print('patched notion_official.yaml for headless reuse')",
            "PY",
        ]
    )
    return "\n".join(
        [
            "set -e",
            f"mkdir -p {_TOOLATHLON_NOTION_MCP_AUTH_DIR}",
            'if [ -n "${TOOLATHLON_NOTION_MCP_AUTH_B64:-}" ]; then',
            f'  printf %s "$TOOLATHLON_NOTION_MCP_AUTH_B64" | base64 -d '
            f"| tar xz -C {_TOOLATHLON_NOTION_MCP_AUTH_DIR}",
            "else",
            '  echo "TOOLATHLON_NOTION_MCP_AUTH_B64 not set; notion page-dup will fail"',
            "fi",
            patch,
        ]
    )


def _toolathlon_task_toml(
    ctx: Any,
    *,
    task_name: str,
    mcp_servers: list[dict[str, Any]],
    variant: str,
    credential_refs: set[str],
    services: set[str] | None = None,
    needs_notion_auth: bool = False,
) -> dict[str, Any]:
    services = services or set()
    benchmark_name = "toolathlon-gym" if variant == "gym" else "toolathlon"
    setup_commands: list[dict[str, Any]] = []
    setup_commands.append(
        {
            "command": _toolathlon_token_setup_command(variant),
            "cwd": "/workspace",
            "timeout_sec": 60.0,
            "env": _toolathlon_token_setup_env(),
        }
    )
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
    if _toolathlon_services.POSTE in services:
        # Point the task's localhost mail configs at the poste sidecar before
        # preprocess (which seeds mailboxes) runs.
        setup_commands.append(
            {
                "command": _toolathlon_services.poste_config_rewrite_command(task_name),
                "cwd": "/workspace",
                "timeout_sec": 60.0,
            }
        )
    if needs_notion_auth:
        # Unpack the pre-authorized notion_official OAuth token so the in-sandbox
        # mcp-remote can duplicate the source page during preprocess.
        setup_commands.append(
            {
                "command": _toolathlon_notion_mcp_auth_command(),
                "cwd": "/workspace",
                "timeout_sec": 60.0,
                "env": {
                    "TOOLATHLON_NOTION_MCP_AUTH_B64": (
                        "${TOOLATHLON_NOTION_MCP_AUTH_B64}"
                    )
                },
            }
        )
    setup_commands.append(
        {
            "command": _toolathlon_setup_command(task_name=task_name, variant=variant),
            "cwd": "/workspace",
            "timeout_sec": 600.0,
        }
    )
    # Service tasks run a DinD compose with sidecar containers alongside main —
    # give the sandbox extra memory/disk for the second image + running service.
    environment: dict[str, Any] = {
        "cpus": 4,
        "memory_mb": 12288 if services else 8192,
        "storage_mb": 24576 if services else 10240,
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
            "raw_env": spec.env,
            "b64_env": spec.b64_env,
            "copy_paths": list(spec.copy_paths),
            "content_format": spec.content_format,
            "file_mode": spec.file_mode,
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
            "        raw = os.environ.get(spec['raw_env']) or ''",
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
            "                f\"{spec['path']} ({spec['raw_env']} or {spec['b64_env']})\"",
            "            )",
            "            continue",
            "    if spec['content_format'] == 'json':",
            "        try:",
            "            json.loads(payload.decode())",
            "        except Exception as exc:",
            "            sys.stderr.write(",
            "                f\"BenchFlow Toolathlon credential setup error: invalid JSON for {spec['path']}: {exc}\\n\"",
            "            )",
            "            sys.exit(66)",
            "    mode = spec['file_mode']",
            "    if not target.exists():",
            "        target.parent.mkdir(parents=True, exist_ok=True)",
            "        target.write_bytes(payload)",
            "        target.chmod(mode)",
            "    for rel in spec.get('copy_paths', []):",
            "        copy = workspace / rel",
            "        copy.parent.mkdir(parents=True, exist_ok=True)",
            "        copy.write_bytes(payload)",
            "        copy.chmod(mode)",
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


def _toolathlon_token_setup_env() -> dict[str, str]:
    """Env for the token-setup command: the container helper (base64) plus the
    secret token values, resolved host-side from BenchFlow's environment/.env."""
    import base64

    module_b64 = base64.b64encode(
        _toolathlon_container_module_source().encode()
    ).decode()
    env: dict[str, str] = {"TOOLATHLON_CONTAINER_MODULE_B64": module_b64}
    for name in _TOOLATHLON_TOKEN_SECRET_ENVS:
        env[name] = f"${{{name}:-}}"
    return env


def _toolathlon_token_setup_command(variant: str) -> str:
    """Stage the container helper and generate the global token_key_session.py.

    The helper is decoded from ``TOOLATHLON_CONTAINER_MODULE_B64`` and its
    ``write-config`` mode bakes the injected secret tokens into
    ``/workspace/configs/token_key_session.py`` (gitignored upstream, so absent
    from the image). MCP servers later resolve ``${token.X}`` against it via the
    same helper's ``launch`` mode.
    """
    python_cmd = _toolathlon_container_python(variant)
    module_path = shlex.quote(_TOOLATHLON_CONTAINER_MODULE_PATH)
    return "\n".join(
        [
            "set -e",
            "mkdir -p /workspace/.toolathlon",
            f"{python_cmd} - <<'PY'",
            "import base64, os, pathlib",
            f"target = pathlib.Path({_TOOLATHLON_CONTAINER_MODULE_PATH!r})",
            "target.write_bytes(",
            "    base64.b64decode(os.environ['TOOLATHLON_CONTAINER_MODULE_B64'])",
            ")",
            "target.chmod(0o755)",
            "PY",
            "chmod -R a+rX /workspace/.toolathlon",
            f"{python_cmd} {module_path} write-config",
            "chmod 0644 /workspace/configs/token_key_session.py",
        ]
    )


def _toolathlon_setup_command(*, task_name: str, variant: str) -> str:
    python_cmd = (
        "/opt/venv/bin/python3" if variant == "gym" else "/usr/local/bin/uv run python"
    )
    task_path = f"/workspace/tasks/finalpool/{task_name}"
    # Run preprocess as the full ``tasks.finalpool.<task>.preprocess.main`` module
    # from /workspace — exactly as upstream does — so ``from ..utils`` relative
    # imports resolve (running it as a bare ``preprocess.main`` breaks them).
    preprocess_module = f"tasks.finalpool.{task_name}.preprocess.main"
    return "\n".join(
        [
            "set -e",
            "mkdir -p /workspace/agent_workspace /workspace/.toolathlon",
            f"TASK_DIR={shlex.quote(task_path)}",
            # A single launch_time shared by preprocess and the verifier, matching
            # upstream's ``datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')``.
            "if [ ! -f /workspace/.toolathlon/launch_time.txt ]; then",
            "  date -u '+%Y-%m-%d %H:%M:%S %A' > /workspace/.toolathlon/launch_time.txt",
            "fi",
            'if [ -d "$TASK_DIR/initial_workspace" ]; then',
            '  cp -a "$TASK_DIR/initial_workspace/." /workspace/agent_workspace/',
            "fi",
            'if [ -f "$TASK_DIR/preprocess/main.py" ]; then',
            '  TASK_DIR="$TASK_DIR" AGENT_WORKSPACE=/workspace/agent_workspace '
            'PYTHONPATH="/workspace:${PYTHONPATH:-}" '
            f"{python_cmd} - <<'PY'",
            "import os, runpy, sys",
            "sys.path.insert(0, '/workspace')",
            "argv = ['preprocess.main', '--agent_workspace', os.environ['AGENT_WORKSPACE']]",
            # Only pass --launch_time to scripts that declare it; argparse errors
            # on unknown args for the ones that don't.
            "src = open(os.environ['TASK_DIR'] + '/preprocess/main.py').read()",
            "if 'launch_time' in src:",
            "    lt = open('/workspace/.toolathlon/launch_time.txt').read().strip()",
            "    argv += ['--launch_time', lt]",
            "sys.argv = argv",
            f"runpy.run_module({preprocess_module!r}, run_name='__main__')",
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
    # Full module path from /workspace (as upstream) so ``from ..`` relative
    # imports in evaluators resolve.
    eval_module = f"tasks.finalpool.{task_name}.evaluation.main"
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
            'PYTHONPATH="/workspace:${PYTHONPATH:-}" '
            f"{python_cmd} - <<'PY' > \"$EVAL_LOG\" 2>&1",
            "import os, runpy, sys",
            "sys.path.insert(0, '/workspace')",
            "argv = [",
            "    'evaluation.main',",
            "    '--agent_workspace', os.environ['AGENT_WORKSPACE'],",
            "    '--groundtruth_workspace', os.environ['GROUNDTRUTH'],",
            "    '--res_log_file', os.environ['RES_LOG'],",
            "]",
            "src = open(os.environ['TASK_DIR'] + '/evaluation/main.py').read()",
            "lt_path = '/workspace/.toolathlon/launch_time.txt'",
            "if 'launch_time' in src and os.path.exists(lt_path):",
            "    argv += ['--launch_time', open(lt_path).read().strip()]",
            "sys.argv = argv",
            f"runpy.run_module({eval_module!r}, run_name='__main__')",
            "PY",
            "status=$?",
            "set -u",
            'cat "$EVAL_LOG"',
            'if [ "$status" -eq 0 ]; then',
            "  printf '{\"reward\": 1.0}\\n' > /logs/verifier/reward.json",
            "  printf '1.0\\n' > /logs/verifier/reward.txt",
            # Upstream evaluators score by exit code: a non-zero exit is a task
            # FAIL (reward 0), and several signal failure by *raising* (e.g.
            # ``raise ValueError('Some tests FAILED')``) rather than returning
            # False. So a traceback alone is not an infra error. Only escalate on
            # signatures that can't be an agent failure — a broken evaluator
            # ENVIRONMENT (missing module/dependency, permission denied) — which
            # a rerun cannot fix and which should surface, not silently score 0.
            "elif grep -qE "
            "'ModuleNotFoundError|ImportError|PermissionError: ' \"$EVAL_LOG\"; then",
            '  echo "BenchFlow Toolathlon verifier setup error: evaluator environment failure" >&2',
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
    ctx: Any, server_name: str, *, variant: str, task_name: str
) -> dict[str, Any]:
    config_path = _toolathlon_mcp_config_path(ctx, server_name)
    data = yaml.safe_load(config_path.read_text())
    params = data.get("params") or {}
    # Static path placeholders (``${agent_workspace}`` …) resolve to fixed
    # in-container paths now; ``${token.X}`` are left intact and resolved at
    # spawn time by the container launcher, the only layer that sees per-task
    # and runtime-computed token values.
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
    if command in {"uv", "uvx"}:
        command = f"/usr/local/bin/{command}"
        if args:
            args = [_TOOLATHLON_UVX_PACKAGE_PINS.get(arg, arg) for arg in args]
    if command == "npx" and args:
        # Pin mcp-remote so its on-disk token path (mcp-remote-<ver>/) matches the
        # injected OAuth dir, and make the auth dir absolute so it resolves
        # regardless of the server's cwd (see the notion mcp-auth setup command).
        args = [_TOOLATHLON_NPX_PACKAGE_PINS.get(arg, arg) for arg in args]
        if "MCP_REMOTE_CONFIG_DIR" in env:
            env["MCP_REMOTE_CONFIG_DIR"] = _TOOLATHLON_NOTION_MCP_AUTH_DIR
            args = [
                *args,
                "--static-oauth-client-info",
                f"@{_TOOLATHLON_NOTION_CLIENT_INFO}",
            ]
    # Wrap the real server in the container launcher so ``${token.X}`` in argv
    # and env resolve at spawn time. ``TOOLATHLON_TASK_DIR`` tells the launcher
    # which task's token_key_session.py overrides the global one.
    env = dict(env)
    env["TOOLATHLON_TASK_DIR"] = f"/workspace/tasks/finalpool/{task_name}"
    ensure_dirs = [
        args[i + 1]
        for i, arg in enumerate(args[:-1])
        if arg in _TOOLATHLON_MCP_DIR_ARG_FLAGS
    ]
    if ensure_dirs:
        # ``:``-joined for the Linux container launcher (not os.pathsep, which
        # would follow the host running the adapter).
        env["TOOLATHLON_ENSURE_DIRS"] = ":".join(ensure_dirs)
    payload: dict[str, Any] = {
        "name": str(data.get("name") or server_name),
        "transport": "stdio",
        "command": _toolathlon_container_python(variant),
        "args": [_TOOLATHLON_CONTAINER_MODULE_PATH, "launch", command, *args],
        "env": env,
        "exclude_tags": [_NOOP_EXCLUDE_TAG],
    }
    if cwd:
        payload["cwd"] = cwd
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
    """Resolve static path placeholders to fixed in-container paths.

    ``${token.X}`` placeholders are intentionally left untouched: they are
    resolved at MCP spawn time by the container launcher (see
    ``_toolathlon_container.py``), the only layer that can see per-task and
    runtime-computed token values.
    """
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
    return value


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
