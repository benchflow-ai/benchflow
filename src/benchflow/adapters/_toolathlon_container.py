#!/usr/bin/env python3
"""Toolathlon container-side helper — runs INSIDE the task sandbox, never
imported by host BenchFlow.

The BenchFlow Toolathlon adapter ships this file into each task container and
invokes it in two roles:

``write-config``
    Generate ``/workspace/configs/token_key_session.py`` (the global
    ``all_token_key_session`` dict) from ``TOOLATHLON_*`` environment variables
    injected at setup time. Upstream gitignores this file; without it the MCP
    servers have no credentials.

``launch <command> [args...]``
    Wrap an MCP server. Resolve ``${token.X}`` / ``${config.X}`` / static path
    placeholders across the server's argv **and** environment, then ``exec`` it.
    Resolution mirrors upstream ``utils/mcp/tool_servers.py`` — the global
    ``all_token_key_session`` merged with, and overridden by, the task's own
    ``token_key_session.py``. This is the only layer that sees per-task and
    runtime-computed tokens (Drive folder ids written by preprocess,
    groundtruth-derived allowlists, per-student Canvas tokens, …), so it is
    where they must be resolved — the host adapter cannot know them.

Kept dependency-free (plain ``python3``, with an ``addict`` shim) so it needs
neither the upstream venv nor a ``uv`` round-trip on every MCP spawn.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import re
import subprocess
import sys
import types
from pathlib import Path

_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")


def _workspace() -> Path:
    return Path(os.environ.get("TOOLATHLON_WORKSPACE", "/workspace"))


class _AttrDict(dict):
    """dict with attribute access, enough to stand in for ``addict.Dict`` when
    executing upstream token files without the real dependency."""

    def __getattr__(self, name: str):  # pragma: no cover - trivial
        return self.get(name)

    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover
        self[name] = value


def _ensure_addict_importable() -> None:
    """Upstream token files do ``from addict import Dict``. Fall back to a
    plain-dict shim when the real package is absent so this helper stays
    dependency-free."""
    if "addict" in sys.modules:
        return
    try:
        import addict  # noqa: F401
    except Exception:
        shim = types.ModuleType("addict")
        shim.__spec__ = importlib.machinery.ModuleSpec("addict", loader=None)
        shim.__dict__["Dict"] = _AttrDict
        sys.modules["addict"] = shim


def _load_all_token_key_session(path: Path) -> dict:
    if not path.is_file():
        return {}
    _ensure_addict_importable()
    spec = importlib.util.spec_from_file_location(f"_tks_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(getattr(module, "all_token_key_session", {}) or {})


# --------------------------------------------------------------------------- #
# launch: resolve placeholders across an MCP server's argv + env, then exec.
# --------------------------------------------------------------------------- #
def _template_vars(task_dir: Path) -> dict[str, str]:
    workspace = _workspace()
    variables: dict[str, str] = {
        "agent_workspace": str(workspace / "agent_workspace"),
        "local_servers_paths": str(workspace / "local_servers"),
        "local_binary_paths": str(workspace / "local_binary"),
        "task_dir": str(workspace / "tasks" / "finalpool"),
    }
    tokens = _load_all_token_key_session(workspace / "configs" / "token_key_session.py")
    tokens.update(_load_all_token_key_session(task_dir / "token_key_session.py"))
    k8s_configs = task_dir / "k8s_configs"
    if k8s_configs.is_dir():
        kubeconfigs = sorted(k8s_configs.glob("*-config.yaml"))
        if len(kubeconfigs) == 1:
            # Some k8s token files derive an instance suffix through PyYAML.
            # The launcher intentionally runs dependency-free, so resolve the
            # token from the kubeconfig that preprocess actually generated.
            tokens["kubeconfig_path"] = str(kubeconfigs[0])
    for key, value in tokens.items():
        if isinstance(value, (str, int, float, bool)):
            variables[f"token.{key}"] = str(value)
    return variables


def _resolve(value: str, variables: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(
        lambda m: variables.get(m.group(1), m.group(0)),
        value,
    )


def _looks_like_jsonrpc_stdout(line: bytes) -> bool:
    """Return True for line-oriented MCP JSON-RPC stdout frames.

    Several upstream Toolathlon MCP servers print human startup logs to stdout
    before emitting real MCP JSON-RPC. MCP clients read stdout as protocol, so
    those logs must be moved to stderr. The official servers used here emit
    compact one-line JSON objects for protocol messages.
    """
    stripped = line.lstrip()
    return stripped.startswith(b"{")


def _run_stdio_server(argv: list[str], env: dict[str, str]) -> int:
    proc = subprocess.Popen(
        argv,
        env=env,
        stdin=sys.stdin,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    assert proc.stdout is not None
    try:
        for line in iter(proc.stdout.readline, b""):
            if _looks_like_jsonrpc_stdout(line):
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            else:
                sys.stderr.buffer.write(line)
                sys.stderr.buffer.flush()
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 130


def _launch(argv: list[str]) -> int:
    if not argv:
        sys.stderr.write("toolathlon launch: no command given\n")
        return 2
    task_dir = Path(os.environ.get("TOOLATHLON_TASK_DIR", str(_workspace())))
    variables = _template_vars(task_dir)
    resolved_argv = [_resolve(arg, variables) for arg in argv]
    resolved_env = dict(os.environ)
    for key, value in list(resolved_env.items()):
        if "${" in value:
            resolved_env[key] = _resolve(value, variables)
    # Create directories the server declares (e.g. arxiv --storage-path, emails
    # attachment dirs) so servers that create them lazily — and evaluators that
    # list them — do not fail when the agent never exercises that path.
    for directory in resolved_env.get("TOOLATHLON_ENSURE_DIRS", "").split(":"):
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
    # FastMCP may spawn this launcher without inheriting PATH; guarantee a sane
    # default so the wrapped server (uvx / npx / node) is still resolvable.
    if not resolved_env.get("PATH"):
        resolved_env["PATH"] = (
            "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin"
        )
    return _run_stdio_server(resolved_argv, resolved_env)


# --------------------------------------------------------------------------- #
# write-config: materialize the global token_key_session.py from env.
# --------------------------------------------------------------------------- #
# (token key, env var, default) — secrets come from env; non-secret defaults
# reproduce configs/token_key_session_example.py. Per-task values (allowlists,
# folder ids, canvas/woo/notion scopes) stay at their "null"/placeholder default
# here and are overridden by each task's own token_key_session.py at launch.
_GLOBAL_TOKENS: tuple[tuple[str, str | None, object], ...] = (
    ("timezone", None, "Asia/Hong_Kong"),
    ("serper_api_key", "TOOLATHLON_SERPER_API_KEY", "XX"),
    ("google_cloud_console_api_key", "TOOLATHLON_MAPS_API_KEY", "XX"),
    ("gcp_project_id", "TOOLATHLON_GCP_PROJECT_ID", "XX"),
    (
        "gcp_service_account_path",
        None,
        "/workspace/configs/gcp-service_account.keys.json",
    ),
    ("google_client_id", "TOOLATHLON_GOOGLE_CLIENT_ID", ""),
    ("google_client_secret", "TOOLATHLON_GOOGLE_CLIENT_SECRET", ""),
    ("google_refresh_token", "TOOLATHLON_GOOGLE_REFRESH_TOKEN", ""),
    ("google_sheets_folder_id", None, "null"),
    (
        "google_oauth2_credentials_path",
        None,
        "/workspace/configs/google_credentials.json",
    ),
    (
        "google_oauth2_token_path",
        None,
        "/workspace/agent_workspace/.toolathlon/google_credentials.json",
    ),
    ("google_cloud_allowed_buckets", None, "null"),
    ("google_cloud_allowed_bigquery_datasets", None, "null"),
    ("google_cloud_allowed_log_buckets", None, "null"),
    ("google_cloud_allowed_instances", None, "null"),
    ("github_token", "TOOLATHLON_GITHUB_TOKEN", "XX"),
    ("github_allowed_repos", None, "null"),
    ("github_read_only", None, "1"),
    ("huggingface_token", "TOOLATHLON_HF_TOKEN", "XX"),
    ("wandb_api_key", "TOOLATHLON_WANDB_API_KEY", "XX"),
    ("notion_integration_key", "TOOLATHLON_NOTION_KEY", "XX"),
    ("notion_integration_key_eval", "TOOLATHLON_NOTION_KEY_EVAL", "XX"),
    ("source_notion_page_url", "TOOLATHLON_NOTION_SOURCE_PAGE_URL", "XX"),
    ("eval_notion_page_url", "TOOLATHLON_NOTION_EVAL_PAGE_URL", "XX"),
    ("snowflake_account", "TOOLATHLON_SNOWFLAKE_ACCOUNT", "XX"),
    ("snowflake_warehouse", "TOOLATHLON_SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    ("snowflake_role", "TOOLATHLON_SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    ("snowflake_user", "TOOLATHLON_SNOWFLAKE_USER", "XX"),
    (
        "snowflake_private_key_path",
        None,
        "/workspace/configs/snowflake_rsa_key.p8",
    ),
    ("snowflake_database", "TOOLATHLON_SNOWFLAKE_DATABASE", "SNOWFLAKE"),
    ("snowflake_schema", "TOOLATHLON_SNOWFLAKE_SCHEMA", "PUBLIC"),
    ("snowflake_op_allowed_databases", None, "null"),
    ("canvas_api_token", None, "canvas_token_victoria_14z"),
    ("canvas_domain", None, "localhost:20001"),
    ("woocommerce_api_key", None, "ck_woocommerce_token_PE0613bf053"),
    ("woocommerce_api_secret", None, "cs_woocommerce_token_PE0613bf053"),
    ("woocommerce_site_url", None, "http://localhost:10003/store100"),
    ("kubeconfig_path", None, "/workspace/deployment/k8s/configs/cluster1-config.yaml"),
    ("emails_config_file", None, "/workspace/configs/example_email_config.json"),
)


def _write_config() -> int:
    tokens: dict[str, object] = {}
    for key, env_var, default in _GLOBAL_TOKENS:
        value = os.environ.get(env_var) if env_var else None
        tokens[key] = value if value not in (None, "") else default
    # Upstream code reads this via ATTRIBUTE access
    # (``all_token_key_session.github_token``) because upstream builds it with
    # ``addict.Dict``. Emit a self-contained attribute-accessible dict so the
    # generated file needs no third-party import in either the preprocess env or
    # the launcher; missing keys return None (addict returns an empty Dict, but
    # nothing here relies on that).
    lines = [
        "class _TokenDict(dict):",
        "    def __getattr__(self, name):",
        "        try:",
        "            return self[name]",
        "        except KeyError:",
        "            return None",
        "",
        "",
        "all_token_key_session = _TokenDict({",
    ]
    lines += [f"    {key!r}: {value!r}," for key, value in tokens.items()]
    lines.append("})")
    target = _workspace() / "configs" / "token_key_session.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    target.chmod(0o644)
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        sys.stderr.write("toolathlon_container: expected 'write-config' or 'launch'\n")
        return 2
    mode, rest = argv[0], argv[1:]
    if mode == "write-config":
        return _write_config()
    if mode == "launch":
        return _launch(rest)
    sys.stderr.write(f"toolathlon_container: unknown mode {mode!r}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
