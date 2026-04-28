"""Filesystem discovery + cross-agent validator for BYOA.

Walks discovery roots in priority order, parses each ``agent.toml`` via
``loader.load_agent_toml``, then runs cross-agent invariants
(``validate_agents``) before exposing the result.

Discovery order (highest priority first):
  1. Explicit path (``--agent <path>``) — caller's responsibility.
  2. ``~/.benchflow/agents/<name>/``
  3. ``$BENCHFLOW_AGENTS_PATH/<name>/``  (colon-separated, like $PATH)
  4. ``src/benchflow/agents/_builtins/<name>/``  (shipped with the wheel)

Per-agent rules already run in ``loader._build_manifest``. This file owns
the cross-agent rules listed in PLAN_V2_byoa.md §5: substring uniqueness,
reserved-substring guard, alias collisions, host-file collisions.

The validator is shape-only — it never reads agent code, never executes
shims, never touches the network. Runtime liveness checks live in
``benchflow.agents.tester`` (PR6).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from benchflow.agents.loader import (
    AgentManifest,
    ManifestParseError,
    load_agent_toml,
)
from benchflow.contracts.scoring import RESERVED_ERROR_SUBSTRINGS

BUILTINS_DIR = Path(__file__).parent / "_builtins"
USER_DIR = Path("~/.benchflow/agents").expanduser()
ENV_PATH_VAR = "BENCHFLOW_AGENTS_PATH"


class AgentRegistryInvalid(Exception):
    """Raised by :func:`discover_agents` when validation surfaces ≥1 error."""

    def __init__(self, errors: list[AgentValidationError]):
        self.errors = errors
        super().__init__(format_errors(errors))


@dataclass(frozen=True)
class AgentValidationError:
    agent: str  # offending agent name (or "<all>" for cross-cutting)
    rule: str  # stable rule ID for grep + CI message
    detail: str
    related: tuple[str, ...] = ()  # other agents involved in collisions


# ── discovery ──────────────────────────────────────────────────────────────


def discovery_roots(env: dict[str, str] | None = None) -> list[Path]:
    """Ordered list of directories to scan for agent dirs.

    Highest priority first. Caller stops at the first hit per agent name.
    """
    env = env if env is not None else dict(os.environ)
    roots: list[Path] = []
    if USER_DIR.is_dir():
        roots.append(USER_DIR)
    raw = env.get(ENV_PATH_VAR, "")
    for chunk in raw.split(os.pathsep):
        if chunk and Path(chunk).is_dir():
            roots.append(Path(chunk))
    if BUILTINS_DIR.is_dir():
        roots.append(BUILTINS_DIR)
    return roots


def scan_root(root: Path) -> dict[str, Path]:
    """Return ``{agent_name: agent_dir}`` for every direct child dir of *root*
    that contains an ``agent.toml``.

    Names starting with ``_`` are treated as private and skipped.
    """
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "agent.toml").is_file():
            out[child.name] = child
    return out


def discover_manifests(
    env: dict[str, str] | None = None,
) -> tuple[dict[str, AgentManifest], list[AgentValidationError]]:
    """Scan all roots, parse every agent.toml, return loaded manifests + per-agent
    parse errors.

    Cross-agent rules are NOT applied here — that is :func:`validate_agents`.
    Highest-priority root wins on name collision.
    """
    loaded: dict[str, AgentManifest] = {}
    errors: list[AgentValidationError] = []
    for root in discovery_roots(env):
        for name, path in scan_root(root).items():
            if name in loaded:
                continue  # higher-priority root already won
            try:
                manifest = load_agent_toml(path)
            except ManifestParseError as exc:
                errors.append(
                    AgentValidationError(
                        agent=name,
                        rule="agent.parse_error",
                        detail=str(exc),
                    )
                )
                continue
            if manifest.name != name:
                errors.append(
                    AgentValidationError(
                        agent=name,
                        rule="agent.name_matches_dir",
                        detail=(
                            f"directory {path} declares agent.name={manifest.name!r} "
                            f"— must match the directory name"
                        ),
                    )
                )
                continue
            loaded[name] = manifest
    return loaded, errors


# ── cross-agent validator ──────────────────────────────────────────────────


def validate_agents(
    loaded: dict[str, AgentManifest],
    aliases: dict[str, str] | None = None,
) -> list[AgentValidationError]:
    """Cross-agent invariants per PLAN_V2_byoa.md §5.

    Returns ``[]`` on success. Loader composes all errors before raising
    ``AgentRegistryInvalid`` so a malformed registry surfaces every
    violation in one go (not a stop-at-first-error parade).
    """
    errors: list[AgentValidationError] = []
    errors.extend(_check_error_taxonomy_uniqueness(loaded))
    errors.extend(_check_error_taxonomy_reserved(loaded))
    errors.extend(_check_subscription_detect_file_uniqueness(loaded))
    errors.extend(_check_credential_file_path_uniqueness(loaded))
    errors.extend(_check_alias_collision(loaded, aliases or {}))
    errors.extend(_check_alias_target_exists(loaded, aliases or {}))
    errors.extend(_check_home_dir_collision_warn(loaded))
    errors.extend(_check_api_protocol_has_provider(loaded))
    return errors


def _check_error_taxonomy_uniqueness(
    loaded: dict[str, AgentManifest],
) -> list[AgentValidationError]:
    seen: dict[str, str] = {}
    errors: list[AgentValidationError] = []
    for name in sorted(loaded):
        for substring in loaded[name].error_taxonomy:
            other = seen.get(substring)
            if other is not None:
                errors.append(
                    AgentValidationError(
                        agent=name,
                        rule="cross.error_taxonomy_uniqueness",
                        detail=(
                            f"{name!r} and {other!r} both claim error_taxonomy "
                            f"substring {substring!r}"
                        ),
                        related=(other,),
                    )
                )
            else:
                seen[substring] = name
    return errors


def _check_error_taxonomy_reserved(
    loaded: dict[str, AgentManifest],
) -> list[AgentValidationError]:
    errors: list[AgentValidationError] = []
    for name in sorted(loaded):
        for substring in loaded[name].error_taxonomy:
            if substring in RESERVED_ERROR_SUBSTRINGS:
                errors.append(
                    AgentValidationError(
                        agent=name,
                        rule="cross.error_taxonomy_reserved",
                        detail=(
                            f"{name!r} claims reserved substring {substring!r} "
                            f"— already matched globally by classify_error"
                        ),
                    )
                )
    return errors


def _check_subscription_detect_file_uniqueness(
    loaded: dict[str, AgentManifest],
) -> list[AgentValidationError]:
    seen: dict[str, str] = {}
    errors: list[AgentValidationError] = []
    for name in sorted(loaded):
        sa = loaded[name].subscription_auth
        if sa is None:
            continue
        prior = seen.get(sa.detect_file)
        if prior is not None:
            errors.append(
                AgentValidationError(
                    agent=name,
                    rule="cross.subscription_detect_file_uniqueness",
                    detail=(
                        f"{name!r} and {prior!r} both declare "
                        f"subscription_auth.detect_file={sa.detect_file!r} — "
                        f"would race on host-auth import"
                    ),
                    related=(prior,),
                )
            )
        else:
            seen[sa.detect_file] = name
    return errors


def _check_credential_file_path_uniqueness(
    loaded: dict[str, AgentManifest],
) -> list[AgentValidationError]:
    # (container_path) → (agent, env_source)
    seen: dict[str, tuple[str, str]] = {}
    errors: list[AgentValidationError] = []
    for name in sorted(loaded):
        for cf in loaded[name].credential_files:
            prior = seen.get(cf.path)
            if prior is not None:
                prior_name, prior_env = prior
                if prior_env != cf.env_source:
                    errors.append(
                        AgentValidationError(
                            agent=name,
                            rule="cross.credential_file_path_uniqueness",
                            detail=(
                                f"{name!r} writes credentials to {cf.path!r} from "
                                f"env {cf.env_source!r}, but {prior_name!r} writes "
                                f"the same path from {prior_env!r} — silent overwrite risk"
                            ),
                            related=(prior_name,),
                        )
                    )
            else:
                seen[cf.path] = (name, cf.env_source)
    return errors


def _check_alias_collision(
    loaded: dict[str, AgentManifest], aliases: dict[str, str]
) -> list[AgentValidationError]:
    errors: list[AgentValidationError] = []
    for alias, target in aliases.items():
        if alias == target:
            continue  # self-alias is a no-op, not a collision
        if alias in loaded:
            errors.append(
                AgentValidationError(
                    agent=alias,
                    rule="cross.alias_collision",
                    detail=(
                        f"alias {alias!r} → {target!r} collides with a real agent "
                        f"named {alias!r} — alias is unreachable"
                    ),
                    related=(target,),
                )
            )
    return errors


def _check_alias_target_exists(
    loaded: dict[str, AgentManifest], aliases: dict[str, str]
) -> list[AgentValidationError]:
    errors: list[AgentValidationError] = []
    for alias, target in aliases.items():
        if target not in loaded:
            errors.append(
                AgentValidationError(
                    agent=alias,
                    rule="cross.alias_target_exists",
                    detail=(
                        f"alias {alias!r} points to {target!r}, which is not a "
                        f"discovered agent"
                    ),
                )
            )
    return errors


def _check_api_protocol_has_provider(
    loaded: dict[str, AgentManifest],
) -> list[AgentValidationError]:
    """Per PLAN_V2_byoa.md §5: if an agent declares api_protocol, at least one
    central provider must support it (otherwise SDK base-URL resolution silently
    routes to the wrong endpoint at trial time)."""
    errors: list[AgentValidationError] = []
    # Lazy import to keep providers.py off discovery's module-import path.
    from benchflow.agents.providers import PROVIDERS

    available: set[str] = set()
    for cfg in PROVIDERS.values():
        available.add(cfg.api_protocol)
        available.update(cfg.endpoints)
    available.discard("")  # empty is "any/inferred"; not a real protocol
    for name in sorted(loaded):
        ap = loaded[name].api_protocol
        if ap and ap not in available:
            errors.append(
                AgentValidationError(
                    agent=name,
                    rule="cross.api_protocol_has_provider",
                    detail=(
                        f"{name!r} declares api_protocol={ap!r} but no provider "
                        f"in PROVIDERS exposes that endpoint "
                        f"(available: {sorted(available)})"
                    ),
                )
            )
    return errors


def _check_home_dir_collision_warn(
    loaded: dict[str, AgentManifest],
) -> list[AgentValidationError]:
    """Warning, not error. Two agents writing to the same dot-dir is legitimate
    for shared caches (``.cache``, ``.config``); only flag for visibility."""
    seen: dict[str, str] = {}
    errors: list[AgentValidationError] = []
    for name in sorted(loaded):
        for d in loaded[name].home_dirs:
            prior = seen.get(d)
            if prior is not None and prior != name:
                errors.append(
                    AgentValidationError(
                        agent=name,
                        rule="cross.home_dir_collision_warn",
                        detail=(
                            f"{name!r} and {prior!r} both declare home_dirs entry "
                            f"{d!r} — review intent"
                        ),
                        related=(prior,),
                    )
                )
            else:
                seen[d] = name
    return errors


# ── public discovery entrypoint ────────────────────────────────────────────


def discover_agents(
    aliases: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, AgentManifest]:
    """Discover + parse + validate all agents from configured roots.

    Raises :class:`AgentRegistryInvalid` if any error-level violation surfaces.
    Warning-level violations (rules ending in ``_warn``) are reported via
    Python ``warnings`` and do not raise.
    """
    loaded, parse_errors = discover_manifests(env)
    cross = validate_agents(loaded, aliases)
    fatal = [e for e in (*parse_errors, *cross) if not e.rule.endswith("_warn")]
    warns = [e for e in cross if e.rule.endswith("_warn")]
    if warns:
        import warnings

        for w in warns:
            warnings.warn(_format_one(w), RuntimeWarning, stacklevel=2)
    if fatal:
        raise AgentRegistryInvalid(fatal)
    return loaded


def format_errors(errors: Iterable[AgentValidationError]) -> str:
    return "Agent registry validation failed:\n" + "\n".join(
        f"  - {_format_one(e)}" for e in errors
    )


def _format_one(e: AgentValidationError) -> str:
    return f"[{e.rule}] {e.agent}: {e.detail}"
