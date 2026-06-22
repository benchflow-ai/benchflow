"""Thin agent-manifest loader — the consumer half of the decoupling contract.

An agent declares itself in a ``manifest.toml`` (see the agents repo's
``contract/manifest_schema.json`` for the authoritative, strict schema). Core
*consumes* that declaration here, turning it into the ``AgentConfig`` the rest
of BenchFlow already understands — so a new agent can ship as data (a manifest)
instead of a hand-edited entry in ``registry.AGENTS`` (design decision #4), and
a directory of manifests can be merged into the registry at opt-in (#7).

Authoring vs. consuming, on purpose:

* The agents-repo JSON Schema is the *author's* contract — ``additionalProperties:
  false``, every field type-checked. It fails an author loudly for a typo.
* This loader is the *consumer's* contract — it enforces only the major-version
  gate and the four required keys, then maps the declared fields onto
  AgentConfig and **ignores unknown keys**. A v1.x manifest that carries a field
  added in a later minor still loads on this v1 loader (SemVer: a minor bump is a
  backward-compatible addition). The AgentConfig fields outside the contract are
  the shim/credential set (session_factory, credential_files, subscription_auth,
  acp_model_config_id, acp_effort_config_id, disallow_web_tools_*); they keep
  their AgentConfig defaults because they are logic the shim owns, not data
  (see _SHIM_ONLY and the partition test).
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from benchflow.agents.registry import AgentConfig

# This loader speaks contract major version 1. A manifest declaring a different
# major is rejected (its shape may be incompatible); a different minor/patch is
# accepted (additive-only within a major).
SUPPORTED_CONTRACT_MAJOR = 1

# The contract is acp-only ("one protocol", ADR-0001 §1; author schema enum:["acp"]).
# The consumer re-enforces it because the BENCHFLOW_AGENTS_DIR dev override bypasses
# the author-side jsonschema, and register_manifest_agents writes AGENTS directly,
# skipping registry.VALID_PROTOCOLS — so this loader is the last protocol gate.
_SUPPORTED_PROTOCOLS = frozenset({"acp"})

# Directory env override for opt-in filesystem discovery (decision #7).
MANIFEST_DIR_ENV = "BENCHFLOW_AGENTS_DIR"

_REQUIRED = ("contract_version", "name", "install_cmd", "launch_cmd")
_VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$")

# manifest key -> AgentConfig field: the contract's data fields (the PR #14 schema
# minus the meta keys contract_version/aliases). The consumer must cover EVERY
# schema data field with an AgentConfig home; _SHIM_ONLY below holds the rest.
# A test asserts _FIELD_MAP.values() | _SHIM_ONLY partitions AgentConfig exactly,
# so a new field can never be silently dropped (it was, for the bottom three).
_FIELD_MAP = {
    "name": "name",
    "description": "description",
    "install_cmd": "install_cmd",
    "launch_cmd": "launch_cmd",
    "protocol": "protocol",
    "api_protocol": "api_protocol",
    "acp_model_format": "acp_model_format",
    "supports_acp_set_model": "supports_acp_set_model",
    "requires_env": "requires_env",
    "install_timeout": "install_timeout",
    "env_mapping": "env_mapping",
    "default_model": "default_model",
    "skill_paths": "skill_paths",
    "home_dirs": "home_dirs",
}

# AgentConfig fields the contract deliberately does NOT carry: logic/credential
# concerns the SHIM owns (credential-file emission, subscription auth, config-file
# writing, web-tool toggles), never expressed as manifest data. The PR #14 schema
# excludes them via additionalProperties:false; here they keep AgentConfig
# defaults. Together with _FIELD_MAP's values these partition AgentConfig exactly.
_SHIM_ONLY = frozenset(
    {
        # Non-ACP "module:callable" seam, only meaningful when
        # protocol="session-factory"; the acp-only contract can never carry it.
        "session_factory",
        "credential_files",
        "subscription_auth",
        "acp_model_config_id",
        "acp_effort_config_id",
        "disallow_web_tools_setup_cmd",
        "disallow_web_tools_owned_paths",
        "disallow_web_tools_launch_suffix",
    }
)


def _merge_core_shim_only(
    manifest_config: AgentConfig, core_config: AgentConfig
) -> AgentConfig:
    """Take the _SHIM_ONLY fields from *core_config*, everything else from
    *manifest_config*. A data-only manifest cannot carry the shim-only fields
    (subscription_auth, credential_files, acp_model_config_id, disallow_web_tools_*,
    ...); when a manifest overrides an EXISTING core agent in additive/compatible
    mode, those host-side/credential concerns are preserved from the core entry so
    the merged config equals the original — the agents repo owns the 14 data fields,
    core retains the un-shimmable host-side bits (e.g. subscription OAuth copy)."""
    return replace(manifest_config, **{f: getattr(core_config, f) for f in _SHIM_ONLY})


class AgentManifestError(ValueError):
    """A manifest.toml is unreadable, missing a required field, declares an
    unsupported contract major version, or collides with an existing agent."""


@dataclass(frozen=True)
class LoadedManifest:
    """A manifest resolved into the registry's vocabulary: the AgentConfig plus
    the alias names the agent answers to (registered separately from the config,
    which has no alias field)."""

    config: AgentConfig
    aliases: tuple[str, ...]


def _check_contract_version(raw: object) -> None:
    if not isinstance(raw, str) or not _VERSION_RE.match(raw):
        raise AgentManifestError(
            f"contract_version must look like MAJOR.MINOR[.PATCH]; got {raw!r}"
        )
    major = int(raw.split(".", 1)[0])
    if major != SUPPORTED_CONTRACT_MAJOR:
        raise AgentManifestError(
            f"contract_version {raw!r} declares major {major}; this loader speaks "
            f"contract {SUPPORTED_CONTRACT_MAJOR}.x"
        )


def load_agent_manifest(path: str | Path) -> LoadedManifest:
    """Parse + validate one ``manifest.toml`` and return its LoadedManifest.

    Raises AgentManifestError on an unreadable/malformed file, a missing required
    field, or an unsupported contract major version.
    """
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise AgentManifestError(f"cannot read manifest {path}: {exc}") from exc

    for key in _REQUIRED:
        if key not in data:
            raise AgentManifestError(f"manifest {path} is missing required {key!r}")
    _check_contract_version(data["contract_version"])

    protocol = data.get("protocol", "acp")
    if protocol not in _SUPPORTED_PROTOCOLS:
        raise AgentManifestError(
            f"manifest {path}: protocol {protocol!r} is not supported; the contract "
            f"is acp-only ({sorted(_SUPPORTED_PROTOCOLS)}). Non-ACP platforms must "
            "ship an ACP-over-stdio shim and declare protocol='acp'."
        )

    kwargs = {field: data[key] for key, field in _FIELD_MAP.items() if key in data}
    if "install_timeout" in kwargs:
        # TOML has no int/float distinction guarantee across writers; AgentConfig
        # wants seconds as an int.
        kwargs["install_timeout"] = int(kwargs["install_timeout"])

    aliases = tuple(data.get("aliases", ()))
    return LoadedManifest(config=AgentConfig(**kwargs), aliases=aliases)


# Directories never scanned for manifests: build/vendor/VCS noise that could
# otherwise surface a fixture manifest or shadow a real agent.
_DISCOVERY_SKIP_DIRS = frozenset(
    {
        "node_modules",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        "site-packages",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }
)


def discover_manifests(root: str | Path) -> list[Path]:
    """Return the sorted manifest.toml paths under *root*, RECURSIVELY (eve-style:
    each agent is a self-contained directory, and families nest — e.g.
    ``ai-sdk/acp/manifest.toml`` alongside ``mimo-acp/manifest.toml``). A
    manifest directly at *root* is ignored (the discovery root holds agent
    SUBdirectories, not an agent itself), as are paths under build/vendor/VCS
    dirs (_DISCOVERY_SKIP_DIRS). A missing root yields ``[]``."""
    root = Path(root)
    if not root.is_dir():
        return []
    out: list[Path] = []
    for path in root.rglob("manifest.toml"):
        rel = path.relative_to(root)
        if len(rel.parts) < 2:
            continue  # a bare root/manifest.toml is not an agent directory
        if any(part in _DISCOVERY_SKIP_DIRS for part in rel.parts):
            continue
        out.append(path)
    return sorted(out)


def load_agents_from_dir(root: str | Path) -> dict[str, LoadedManifest]:
    """Load every manifest under *root*, keyed by the agent's DECLARED name — not
    the directory name, which may differ (e.g. ``mimo-acp/`` declares ``mimo``).
    Raises AgentManifestError on a duplicate declared name."""
    out: dict[str, LoadedManifest] = {}
    for path in discover_manifests(root):
        loaded = load_agent_manifest(path)
        name = loaded.config.name
        if name in out:
            raise AgentManifestError(
                f"duplicate agent name {name!r} (second declaration at {path})"
            )
        out[name] = loaded
    return out


def register_manifest_agents(
    loaded: Mapping[str, LoadedManifest],
    *,
    agents: dict[str, AgentConfig],
    aliases: dict[str, str],
    installers: dict[str, str],
    launch: dict[str, str],
    override: bool = False,
    merge_shim_only: bool = False,
) -> None:
    """Merge *loaded* manifests into the registry maps in place.

    Writes all four name-keyed maps the registry keeps in lockstep — ``agents``
    (AGENTS), ``installers`` (AGENT_INSTALLERS), ``launch`` (AGENT_LAUNCH), and
    ``aliases`` (AGENT_ALIASES) — because per-agent lookups in install.py and
    rollout_planes.py read the installer/launch projections, not AGENTS. Omitting
    them would leave a manifest agent uninstallable / unlaunchable.

    Fail-loud on collision (an agent name or alias that already exists is an
    ambiguous source of truth, not a silent shadow) unless ``override=True``.
    Collisions are checked across the whole batch BEFORE any mutation, so a
    rejected batch leaves every map untouched (all-or-nothing).

    ``merge_shim_only=True`` is the additive/compatible mode (used by the
    BENCHFLOW_AGENTS_DIR loader): a manifest reproducing an existing core agent
    intentionally overrides it, but its _SHIM_ONLY fields are taken from the core
    entry (which the data-only manifest can't carry), so the merged config equals
    the original. Implies override semantics (no collision raise)."""
    if not override and not merge_shim_only:
        for name, lm in loaded.items():
            if name in agents:
                raise AgentManifestError(
                    f"agent {name!r} already in the registry; ship its manifest as "
                    "the sole source or pass override=True"
                )
            for alias in lm.aliases:
                if alias in aliases:
                    raise AgentManifestError(
                        f"alias {alias!r} (for {name!r}) already maps to "
                        f"{aliases[alias]!r}"
                    )
                if alias in agents:
                    raise AgentManifestError(
                        f"alias {alias!r} (for {name!r}) collides with an existing "
                        "agent name"
                    )
    for name, lm in loaded.items():
        config = lm.config
        if merge_shim_only and name in agents:
            config = _merge_core_shim_only(lm.config, agents[name])
        agents[name] = config
        installers[name] = config.install_cmd
        launch[name] = config.launch_cmd
        for alias in lm.aliases:
            aliases[alias] = name


def register_env_manifest_agents(
    *,
    agents: dict[str, AgentConfig] | None = None,
    aliases: dict[str, str] | None = None,
    installers: dict[str, str] | None = None,
    launch: dict[str, str] | None = None,
) -> list[str]:
    """Register agents from the directory named by ``$BENCHFLOW_AGENTS_DIR`` into
    the registry; return the sorted names registered.

    A no-op returning ``[]`` when the env var is unset — so a default import of
    core is unchanged and the dual-source registry only activates on explicit
    opt-in. The four maps default to the live ``registry`` globals (resolved
    lazily to avoid an import cycle); tests pass throwaway dicts to stay
    hermetic."""
    root = os.environ.get(MANIFEST_DIR_ENV)
    if not root:
        return []
    if None in (agents, aliases, installers, launch):
        from benchflow.agents.registry import (
            AGENT_ALIASES,
            AGENT_INSTALLERS,
            AGENT_LAUNCH,
            AGENTS,
        )

        agents = AGENTS if agents is None else agents
        aliases = AGENT_ALIASES if aliases is None else aliases
        installers = AGENT_INSTALLERS if installers is None else installers
        launch = AGENT_LAUNCH if launch is None else launch
    loaded = load_agents_from_dir(root)
    register_manifest_agents(
        loaded,
        agents=agents,
        aliases=aliases,
        installers=installers,
        launch=launch,
        # Additive/compatible: a manifest reproducing a core agent overrides it but
        # keeps the core entry's host-side _SHIM_ONLY fields (subscription_auth, ...).
        merge_shim_only=True,
    )
    return sorted(loaded)
