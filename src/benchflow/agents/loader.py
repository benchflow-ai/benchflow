"""TOML → ``AgentManifest`` loader for BYOA agents.

PERIPHERY — ``AgentManifest`` is intentionally NOT in ``benchflow.contracts``.
It carries the v1 ``agent.toml`` schema (incl. fields like ``schema_version``,
``[smoke_test]``, ``[reporting]``, ``[mcp_servers]`` that have no live reader
yet) and projects onto the semver-stable ``AgentConfig`` via
``Manifest.to_agent_config()``. New TOML fields land here, NOT on ``AgentConfig``.

See ``sandbox/PLAN_V2_byoa.md`` §3 (locked schema) and §9 (contract impact).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.contracts.agent_config import (
    AgentConfig,
    CredentialFile,
    HostAuthFile,
    SubscriptionAuth,
)

SCHEMA_VERSION = 1

# Per-agent rules also live in tests/test_registry_invariants.py; this list is
# the runtime equivalent — the discovery loader raises early on violations.
VALID_PROTOCOLS = {"acp", "cli", "shim"}
VALID_API_PROTOCOLS = {"", "anthropic-messages", "openai-completions"}


class ManifestParseError(ValueError):
    """Raised when an ``agent.toml`` cannot be parsed into an ``AgentManifest``."""


@dataclass(frozen=True)
class SmokeTest:
    version_cmd: str = ""
    version_regex: str = ""
    ping_cmd: str = ""
    ping_timeout: int = 30


@dataclass(frozen=True)
class Reporting:
    cost: str = "none"  # "full" | "totals_only" | "none"
    trajectory: str = "acp"  # "acp" | "file" | "shim" | "none"
    trajectory_path: str = ""
    trajectory_schema: str = ""
    # PR10 (PLAN_V2_byoa.md): name of the env var that, when set to "1",
    # enables the agent's OpenTelemetry export. Empty = no OTel support.
    # Trial sets this var + standard OTEL_EXPORTER_OTLP_* when the
    # BENCHFLOW_OTEL_ENABLE flag is on. Per the OTel survey, claude
    # uses CLAUDE_CODE_ENABLE_TELEMETRY; openclaw + gemini have their
    # own knobs (config-file driven, but the same env-var convention
    # works for opt-in detection).
    otel_enable_env: str = ""


@dataclass(frozen=True)
class ProviderRef:
    name: str = ""
    inline: bool = False


@dataclass(frozen=True)
class McpServer:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()  # frozen kv pairs


@dataclass(frozen=True)
class AgentManifest:
    """v1 ``agent.toml`` payload. Projects to ``AgentConfig`` via :meth:`to_agent_config`.

    Periphery type. New TOML fields land here, never on ``AgentConfig``.
    """

    schema_version: int
    name: str
    install_cmd: str
    launch_cmd: str
    description: str = ""
    protocol: str = "acp"
    api_protocol: str = ""
    default_model: str = ""
    install_timeout: int = 900
    supports_atif: bool = True
    supports_windows: bool = False
    install_target_dir: str = ""
    requires_env: tuple[str, ...] = ()
    env_mapping: tuple[tuple[str, str], ...] = ()
    home_dirs: tuple[str, ...] = ()
    skill_paths: tuple[str, ...] = ()
    credential_files: tuple[CredentialFile, ...] = ()
    subscription_auth: SubscriptionAuth | None = None
    error_taxonomy: tuple[str, ...] = ()
    error_taxonomy_transient: tuple[str, ...] = ()
    error_taxonomy_oom: tuple[int, ...] = ()
    provider: ProviderRef = field(default_factory=ProviderRef)
    smoke_test: SmokeTest = field(default_factory=SmokeTest)
    reporting: Reporting = field(default_factory=Reporting)
    mcp_servers: tuple[McpServer, ...] = ()
    source_dir: Path | None = None  # populated when loaded from a directory

    def to_agent_config(self) -> AgentConfig:
        """Project the manifest onto the semver-stable ``AgentConfig`` shape.

        Lossy by design — fields without a contract reader (``schema_version``,
        ``smoke_test``, ``reporting``, ``mcp_servers``, ``provider``, install
        ``target_dir``, capability flags) stay on the manifest and are accessed
        by periphery callers (tester, hooks, discovery).
        """
        return AgentConfig(
            name=self.name,
            install_cmd=self.install_cmd,
            launch_cmd=self.launch_cmd,
            protocol=self.protocol if self.protocol != "shim" else "acp",
            requires_env=list(self.requires_env),
            description=self.description,
            skill_paths=list(self.skill_paths),
            install_timeout=self.install_timeout,
            default_model=self.default_model,
            api_protocol=self.api_protocol,
            env_mapping=dict(self.env_mapping),
            credential_files=list(self.credential_files),
            home_dirs=list(self.home_dirs),
            subscription_auth=self.subscription_auth,
            error_taxonomy=list(self.error_taxonomy),
        )


def load_agent_toml(path: str | Path) -> AgentManifest:
    """Parse an ``agent.toml`` (or a directory containing one) into a manifest.

    Per-agent shape rules run inline; cross-agent rules (substring uniqueness,
    alias collisions, …) belong to ``benchflow.agents.discovery.validate_agents``.
    """
    p = Path(path)
    if p.is_dir():
        source_dir: Path | None = p
        toml_path = p / "agent.toml"
    else:
        source_dir = p.parent if p.parent != Path() else None
        toml_path = p
    if not toml_path.is_file():
        raise ManifestParseError(f"agent.toml not found at {toml_path}")
    try:
        data = tomllib.loads(toml_path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ManifestParseError(f"{toml_path}: TOML parse error: {exc}") from exc
    return _build_manifest(data, source_dir, toml_path)


def _build_manifest(
    data: dict[str, Any], source_dir: Path | None, toml_path: Path
) -> AgentManifest:
    schema_version = data.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ManifestParseError(
            f"{toml_path}: schema_version must be {SCHEMA_VERSION}, got {schema_version!r}"
        )

    agent = _require_table(data, "agent", toml_path)
    install = _require_table(data, "install", toml_path)
    launch = _require_table(data, "launch", toml_path)
    env_tbl = data.get("env", {}) or {}
    skills_tbl = data.get("skills", {}) or {}
    err_tbl = data.get("error_taxonomy", {}) or {}
    smoke_tbl = data.get("smoke_test", {}) or {}
    reporting_tbl = data.get("trajectory", {}) or {}
    cost_tbl = data.get("reporting", {}) or {}
    provider_tbl = data.get("provider", {}) or {}
    sub_tbl = data.get("subscription_auth")

    name = _require_str(agent, "agent.name", toml_path)
    install_cmd = _resolve_install_cmd(install, source_dir, toml_path)
    launch_cmd = _require_str(launch, "launch.cmd", toml_path)
    protocol = agent.get("protocol", "acp")
    if protocol not in VALID_PROTOCOLS:
        raise ManifestParseError(
            f"{toml_path}: agent.protocol={protocol!r} not in {sorted(VALID_PROTOCOLS)}"
        )
    api_protocol = agent.get("api_protocol", "")
    if api_protocol not in VALID_API_PROTOCOLS:
        raise ManifestParseError(
            f"{toml_path}: agent.api_protocol={api_protocol!r} not in {sorted(VALID_API_PROTOCOLS)}"
        )
    install_timeout = int(agent.get("install_timeout", 900))
    if install_timeout <= 0:
        raise ManifestParseError(
            f"{toml_path}: agent.install_timeout must be positive, got {install_timeout}"
        )

    requires_env = tuple(_require_str_list(env_tbl.get("requires", []), "env.requires", toml_path))
    env_mapping_raw = env_tbl.get("mapping", {}) or {}
    if not isinstance(env_mapping_raw, dict):
        raise ManifestParseError(f"{toml_path}: env.mapping must be a table")
    for key, val in env_mapping_raw.items():
        if not key.startswith("BENCHFLOW_PROVIDER_"):
            raise ManifestParseError(
                f"{toml_path}: env.mapping key {key!r} must start with BENCHFLOW_PROVIDER_"
            )
        if not (isinstance(val, str) and val):
            raise ManifestParseError(
                f"{toml_path}: env.mapping[{key!r}] must be a non-empty string"
            )
    env_mapping = tuple(sorted(env_mapping_raw.items()))

    home_dirs_raw = _require_str_list(env_tbl.get("home_dirs", []), "env.home_dirs", toml_path)
    for d in home_dirs_raw:
        if not d.startswith("."):
            raise ManifestParseError(
                f"{toml_path}: env.home_dirs entry {d!r} must start with '.'"
            )
    home_dirs = tuple(home_dirs_raw)

    skill_paths_raw = _require_str_list(
        skills_tbl.get("mount_paths", []), "skills.mount_paths", toml_path
    )
    for sp in skill_paths_raw:
        if not sp.startswith(("$HOME/", "$WORKSPACE/", "{home}/")):
            raise ManifestParseError(
                f"{toml_path}: skills.mount_paths entry {sp!r} must start with $HOME/, $WORKSPACE/, or {{home}}/"
            )
    skill_paths = tuple(skill_paths_raw)

    credential_files = tuple(_parse_credentials(data.get("credentials", []), toml_path))
    subscription_auth = _parse_subscription_auth(sub_tbl, toml_path)

    error_taxonomy = tuple(
        _require_str_list(err_tbl.get("substrings", []), "error_taxonomy.substrings", toml_path)
    )
    err_transient = tuple(
        _require_str_list(err_tbl.get("transient", []), "error_taxonomy.transient", toml_path)
    )
    err_oom_raw = err_tbl.get("oom", []) or []
    if not isinstance(err_oom_raw, list) or not all(isinstance(x, int) for x in err_oom_raw):
        raise ManifestParseError(
            f"{toml_path}: error_taxonomy.oom must be a list of ints"
        )
    err_oom = tuple(err_oom_raw)

    provider = ProviderRef(
        name=str(provider_tbl.get("name", "")),
        inline=bool(provider_tbl.get("inline", False)),
    )

    smoke_test = SmokeTest(
        version_cmd=str(smoke_tbl.get("version_cmd", "")),
        version_regex=str(smoke_tbl.get("version_regex", "")),
        ping_cmd=str(smoke_tbl.get("ping_cmd", "")),
        ping_timeout=int(smoke_tbl.get("ping_timeout", 30)),
    )

    reporting = Reporting(
        cost=str(cost_tbl.get("cost", "none")),
        trajectory=str(reporting_tbl.get("mode", "acp")),
        trajectory_path=str(reporting_tbl.get("path", "")),
        trajectory_schema=str(reporting_tbl.get("schema", "")),
        otel_enable_env=str(cost_tbl.get("otel_enable_env", "")),
    )

    mcp_servers = tuple(_parse_mcp_servers(data.get("mcp_servers", {}) or {}, toml_path))

    return AgentManifest(
        schema_version=SCHEMA_VERSION,
        name=name,
        install_cmd=install_cmd,
        launch_cmd=launch_cmd,
        description=str(agent.get("description", "")),
        protocol=protocol,
        api_protocol=api_protocol,
        default_model=str(agent.get("default_model", "")),
        install_timeout=install_timeout,
        supports_atif=bool(agent.get("supports_atif", True)),
        supports_windows=bool(agent.get("supports_windows", False)),
        install_target_dir=str(install.get("target_dir", "")),
        requires_env=requires_env,
        env_mapping=env_mapping,
        home_dirs=home_dirs,
        skill_paths=skill_paths,
        credential_files=credential_files,
        subscription_auth=subscription_auth,
        error_taxonomy=error_taxonomy,
        error_taxonomy_transient=err_transient,
        error_taxonomy_oom=err_oom,
        provider=provider,
        smoke_test=smoke_test,
        reporting=reporting,
        mcp_servers=mcp_servers,
        source_dir=source_dir,
    )


def _resolve_install_cmd(
    install: dict[str, Any], source_dir: Path | None, toml_path: Path
) -> str:
    cmd = install.get("cmd")
    script = install.get("script")
    if cmd and script:
        raise ManifestParseError(
            f"{toml_path}: install must specify exactly one of cmd or script, not both"
        )
    if cmd:
        if not (isinstance(cmd, str) and cmd):
            raise ManifestParseError(f"{toml_path}: install.cmd must be a non-empty string")
        return cmd
    if script:
        if not isinstance(script, str):
            raise ManifestParseError(f"{toml_path}: install.script must be a string path")
        if source_dir is None:
            raise ManifestParseError(
                f"{toml_path}: install.script requires the manifest to be in a directory"
            )
        script_path = source_dir / script
        if not script_path.is_file():
            raise ManifestParseError(f"{toml_path}: install.script {script_path} not found")
        return f"bash /opt/agent/{script}"
    raise ManifestParseError(f"{toml_path}: install must specify cmd or script")


def _parse_credentials(raw: Any, toml_path: Path) -> list[CredentialFile]:
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ManifestParseError(f"{toml_path}: [[credentials]] must be a list of tables")
    out: list[CredentialFile] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ManifestParseError(f"{toml_path}: each credentials entry must be a table")
        path = entry.get("path")
        env_source = entry.get("env_source")
        if not (isinstance(path, str) and path):
            raise ManifestParseError(f"{toml_path}: credentials.path must be non-empty")
        if not (isinstance(env_source, str) and env_source):
            raise ManifestParseError(f"{toml_path}: credentials.env_source must be non-empty")
        out.append(
            CredentialFile(
                path=path,
                env_source=env_source,
                template=str(entry.get("template", "")),
                mkdir=bool(entry.get("mkdir", True)),
            )
        )
    return out


def _parse_subscription_auth(raw: Any, toml_path: Path) -> SubscriptionAuth | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ManifestParseError(f"{toml_path}: [subscription_auth] must be a table")
    replaces_env = raw.get("replaces_env")
    detect_file = raw.get("detect_file")
    if not (isinstance(replaces_env, str) and replaces_env):
        raise ManifestParseError(
            f"{toml_path}: subscription_auth.replaces_env must be non-empty"
        )
    if not (isinstance(detect_file, str) and detect_file):
        raise ManifestParseError(
            f"{toml_path}: subscription_auth.detect_file must be non-empty"
        )
    files_raw = raw.get("files", []) or []
    if not isinstance(files_raw, list):
        raise ManifestParseError(f"{toml_path}: subscription_auth.files must be a list")
    files: list[HostAuthFile] = []
    for entry in files_raw:
        if not isinstance(entry, dict):
            raise ManifestParseError(
                f"{toml_path}: each subscription_auth.files entry must be a table"
            )
        host_path = entry.get("host_path")
        container_path = entry.get("container_path")
        if not (isinstance(host_path, str) and host_path):
            raise ManifestParseError(
                f"{toml_path}: subscription_auth.files[].host_path must be non-empty"
            )
        if not (isinstance(container_path, str) and container_path):
            raise ManifestParseError(
                f"{toml_path}: subscription_auth.files[].container_path must be non-empty"
            )
        files.append(HostAuthFile(host_path=host_path, container_path=container_path))
    return SubscriptionAuth(
        replaces_env=replaces_env, detect_file=detect_file, files=files
    )


def _parse_mcp_servers(raw: Any, toml_path: Path) -> list[McpServer]:
    if not isinstance(raw, dict):
        raise ManifestParseError(f"{toml_path}: [mcp_servers] must be a table")
    out: list[McpServer] = []
    for server_name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ManifestParseError(
                f"{toml_path}: mcp_servers.{server_name} must be a table"
            )
        command = entry.get("command")
        if not (isinstance(command, str) and command):
            raise ManifestParseError(
                f"{toml_path}: mcp_servers.{server_name}.command must be non-empty"
            )
        args_raw = entry.get("args", []) or []
        if not isinstance(args_raw, list) or not all(isinstance(a, str) for a in args_raw):
            raise ManifestParseError(
                f"{toml_path}: mcp_servers.{server_name}.args must be a list of strings"
            )
        env_raw = entry.get("env", {}) or {}
        if not isinstance(env_raw, dict):
            raise ManifestParseError(
                f"{toml_path}: mcp_servers.{server_name}.env must be a table"
            )
        out.append(
            McpServer(
                name=server_name,
                command=command,
                args=tuple(args_raw),
                env=tuple(sorted((str(k), str(v)) for k, v in env_raw.items())),
            )
        )
    return out


def _require_table(data: dict[str, Any], key: str, toml_path: Path) -> dict[str, Any]:
    val = data.get(key)
    if not isinstance(val, dict):
        raise ManifestParseError(f"{toml_path}: missing or invalid [{key}] table")
    return val


def _require_str(table: dict[str, Any], qualified_key: str, toml_path: Path) -> str:
    leaf = qualified_key.rsplit(".", 1)[-1]
    val = table.get(leaf)
    if not (isinstance(val, str) and val):
        raise ManifestParseError(f"{toml_path}: {qualified_key} must be a non-empty string")
    return val


def _require_str_list(val: Any, qualified_key: str, toml_path: Path) -> list[str]:
    if val is None:
        return []
    if not isinstance(val, list) or not all(isinstance(x, str) and x for x in val):
        raise ManifestParseError(
            f"{toml_path}: {qualified_key} must be a list of non-empty strings"
        )
    return [str(x) for x in val]
