"""Agent registry — thin facade over filesystem discovery.

PR4 inverts the source of truth: ``_builtins/<name>/agent.toml`` (and
user-installed ``~/.benchflow/agents/<name>/``) are canonical; this module
exposes the discovered agents through the long-standing import surface
(``AGENTS``, ``AGENT_INSTALLERS``, ``AGENT_LAUNCH``, ``AGENT_ALIASES``,
``register_agent``, ``resolve_agent``, ``get_agent``, ``list_agents``,
``parse_agent_spec``, ``infer_env_key_for_model``, ``is_vertex_model``,
``get_sandbox_home_dirs``).

Adding a new agent is now a directory edit:
  1. Create ``src/benchflow/agents/_builtins/<name>/agent.toml`` (and
     optionally ``shim.py``); see PLAN_V2_byoa.md §3 for the schema.
  2. Run ``tools/regenerate_builtin_agents.py`` if you change a Python
     ``AGENTS`` literal — but Python literals are no longer the source.

External callers that want to register an agent at runtime continue to
use :func:`register_agent`.
"""

from __future__ import annotations

import tomllib
from typing import Any

from benchflow.agents.discovery import (
    BUILTINS_DIR,
    AgentRegistryInvalid,
    discover_agents,
)
from benchflow.contracts.agent_config import (
    AgentConfig,
    CredentialFile,
    HostAuthFile,
    SubscriptionAuth,
)

__all__ = [
    "AGENTS",
    "AGENT_INSTALLERS",
    "AGENT_LAUNCH",
    "AGENT_ALIASES",
    "VALID_PROTOCOLS",
    "AgentConfig",
    "CredentialFile",
    "HostAuthFile",
    "SubscriptionAuth",
    "register_agent",
    "resolve_agent",
    "get_agent",
    "list_agents",
    "parse_agent_spec",
    "infer_env_key_for_model",
    "is_vertex_model",
    "get_sandbox_home_dirs",
]

VALID_PROTOCOLS = {"acp", "harbor"}

# Lazy module-level state. Populated by ``_ensure_loaded()`` on first
# access via ``__getattr__`` or ``register_agent``.
_AGENTS: dict[str, AgentConfig] | None = None
_AGENT_INSTALLERS: dict[str, str] | None = None
_AGENT_LAUNCH: dict[str, str] | None = None
_AGENT_ALIASES: dict[str, str] | None = None


def _load_aliases() -> dict[str, str]:
    aliases_path = BUILTINS_DIR / "_aliases.toml"
    if not aliases_path.is_file():
        return {}
    raw = tomllib.loads(aliases_path.read_text())
    return {str(k): str(v) for k, v in raw.items()}


def _ensure_loaded() -> None:
    """Load ``_builtins/`` + user dirs on first use; idempotent."""
    global _AGENTS, _AGENT_INSTALLERS, _AGENT_LAUNCH, _AGENT_ALIASES
    if _AGENTS is not None:
        return
    aliases = _load_aliases()
    manifests = discover_agents(aliases=aliases)
    _AGENTS = {name: m.to_agent_config() for name, m in manifests.items()}
    _AGENT_INSTALLERS = {name: c.install_cmd for name, c in _AGENTS.items()}
    _AGENT_LAUNCH = {name: c.launch_cmd for name, c in _AGENTS.items()}
    _AGENT_ALIASES = aliases


def __getattr__(name: str) -> Any:
    """Module-level lazy bindings (PEP 562). Triggers discovery on first read."""
    if name == "AGENTS":
        _ensure_loaded()
        assert _AGENTS is not None
        return _AGENTS
    if name == "AGENT_INSTALLERS":
        _ensure_loaded()
        assert _AGENT_INSTALLERS is not None
        return _AGENT_INSTALLERS
    if name == "AGENT_LAUNCH":
        _ensure_loaded()
        assert _AGENT_LAUNCH is not None
        return _AGENT_LAUNCH
    if name == "AGENT_ALIASES":
        _ensure_loaded()
        assert _AGENT_ALIASES is not None
        return _AGENT_ALIASES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def parse_agent_spec(spec: str) -> tuple[str, str]:
    """Parse an agent spec like 'acp/claude-agent-acp' or 'claude'.

    Returns (protocol, agent_name) with alias resolution.
    Bare names default to 'acp' protocol.
    """
    _ensure_loaded()
    assert _AGENT_ALIASES is not None
    if "/" in spec:
        protocol, name = spec.split("/", 1)
    else:
        protocol, name = "acp", spec
    name = _AGENT_ALIASES.get(name, name)
    return protocol, name


def resolve_agent(spec: str) -> AgentConfig:
    """Resolve an agent spec to an AgentConfig.

    Supports: bare name, alias, ``protocol/name``. Raises ``KeyError`` with
    suggestions for unknown agents. ``harbor/<name>`` short-circuits to a
    synthesized AgentConfig because Harbor agents bypass the dir model
    (PLAN_V2_byoa.md §10 risk 4).
    """
    _ensure_loaded()
    assert _AGENTS is not None
    protocol, name = parse_agent_spec(spec)
    if protocol not in VALID_PROTOCOLS:
        raise KeyError(
            f"Unknown protocol: {protocol!r}. Valid: {', '.join(sorted(VALID_PROTOCOLS))}"
        )
    if protocol == "harbor":
        return AgentConfig(
            name=name,
            install_cmd="",
            launch_cmd="",
            protocol="harbor",
            requires_env=[],
            description=f"Harbor agent: {name}",
        )
    if name in _AGENTS:
        return _AGENTS[name]
    from difflib import get_close_matches

    close = get_close_matches(name, list(_AGENTS), n=1, cutoff=0.6)
    if close:
        raise KeyError(f"Unknown agent: {name!r}. Did you mean: {close[0]!r}?")
    raise KeyError(
        f"Unknown agent: {name!r}. Available: {', '.join(sorted(_AGENTS))}"
    )


def get_agent(name: str) -> tuple[AgentConfig, str]:
    """Get agent config by name. Returns ``(config, default_model)``."""
    _ensure_loaded()
    assert _AGENTS is not None
    if name not in _AGENTS:
        raise KeyError(
            f"Unknown agent: {name!r}. Available: {', '.join(sorted(_AGENTS))}"
        )
    config = _AGENTS[name]
    return config, config.default_model


def list_agents() -> list[AgentConfig]:
    """List all registered agents."""
    _ensure_loaded()
    assert _AGENTS is not None
    return list(_AGENTS.values())


def register_agent(
    name: str,
    install_cmd: str,
    launch_cmd: str,
    *,
    protocol: str = "acp",
    requires_env: list[str] | None = None,
    description: str = "",
    skill_paths: list[str] | None = None,
    install_timeout: int = 900,
    default_model: str = "",
    api_protocol: str = "",
    env_mapping: dict[str, str] | None = None,
    credential_files: list[CredentialFile] | None = None,
    home_dirs: list[str] | None = None,
    subscription_auth: SubscriptionAuth | None = None,
    error_taxonomy: list[str] | None = None,
) -> AgentConfig:
    """Register a custom agent at runtime.

    Mutates the in-memory registry only. To make an agent persistent,
    create ``~/.benchflow/agents/<name>/agent.toml`` per PLAN_V2_byoa.md §1.
    """
    _ensure_loaded()
    assert _AGENTS is not None and _AGENT_INSTALLERS is not None and _AGENT_LAUNCH is not None
    config = AgentConfig(
        name=name,
        install_cmd=install_cmd,
        launch_cmd=launch_cmd,
        protocol=protocol,
        requires_env=requires_env or [],
        description=description,
        skill_paths=skill_paths or [],
        install_timeout=install_timeout,
        default_model=default_model,
        api_protocol=api_protocol,
        env_mapping=env_mapping or {},
        credential_files=credential_files or [],
        home_dirs=home_dirs or [],
        subscription_auth=subscription_auth,
        error_taxonomy=error_taxonomy or [],
    )
    _AGENTS[name] = config
    _AGENT_INSTALLERS[name] = install_cmd
    _AGENT_LAUNCH[name] = launch_cmd
    return config


def get_sandbox_home_dirs() -> set[str]:
    """Collect all dot-dirs under $HOME that sandbox user setup should copy.

    Derives from three sources across all registered agents:
    - skill_paths: ``$HOME/.foo/...`` → ``.foo``
    - credential_files: ``{home}/.foo/...`` → ``.foo``
    - home_dirs: explicit extras (e.g. ``.openclaw``)

    Always includes ``.local`` (pip scripts, etc.).
    """
    _ensure_loaded()
    assert _AGENTS is not None
    dirs: set[str] = {".local"}
    for cfg in _AGENTS.values():
        for sp in cfg.skill_paths:
            if sp.startswith("$HOME/."):
                dirname = sp.removeprefix("$HOME/").split("/")[0]
                dirs.add(dirname)
        for cf in cfg.credential_files:
            path = cf.path
            if path.startswith("{home}/."):
                dirname = path.removeprefix("{home}/").split("/")[0]
                dirs.add(dirname)
        if cfg.subscription_auth:
            for f in cfg.subscription_auth.files:
                if f.container_path.startswith("{home}/."):
                    dirname = f.container_path.removeprefix("{home}/").split("/")[0]
                    dirs.add(dirname)
        dirs.update(cfg.home_dirs)
    return dirs


def is_vertex_model(model: str) -> bool:
    """True if the model uses Vertex AI (GCP ADC auth, not API keys)."""
    from benchflow.agents.providers import find_provider

    result = find_provider(model)
    if result:
        _, cfg = result
        return cfg.auth_type == "adc"
    return False


def infer_env_key_for_model(model: str) -> str | None:
    """Infer the required API key environment variable from a model ID."""
    from benchflow.agents.providers import resolve_auth_env

    custom = resolve_auth_env(model)
    if custom is not None:
        return custom
    if is_vertex_model(model):
        return None
    m = model.lower()
    if "gemini" in m:
        return "GEMINI_API_KEY"
    if "gpt" in m or m.startswith("o1") or m.startswith("o3"):
        return "OPENAI_API_KEY"
    if "claude" in m or "haiku" in m or "sonnet" in m or "opus" in m:
        return "ANTHROPIC_API_KEY"
    return None


# Re-export the AgentRegistryInvalid for callers wanting to handle it.
__all__.append("AgentRegistryInvalid")
