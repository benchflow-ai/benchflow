"""One-shot loading of declarative agent manifests."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path

from benchflow.agents.manifest import (
    AgentManifestError,
    LoadedManifest,
    ManifestIssueKind,
    _merge_core_shim_only,
    discover_manifests,
    load_agent_manifest,
    register_manifest_agents,
    select_manifest_agents,
)

AGENTS_DIR_ENV = "BENCHFLOW_AGENTS_DIR"
AGENTS_SOURCE_ENV = "BENCHFLOW_AGENTS_SOURCE"
DEFAULT_AGENTS_SOURCE = "benchflow-ai/agents@main"
_OFF_VALUES = frozenset({"off", "0", "none", "disabled", "false"})


@dataclass(frozen=True)
class ManifestIssue:
    kind: ManifestIssueKind
    detail: str
    path: str = ""
    cause: str = ""

    def warning(self) -> str:
        prefix = f"{self.path}: " if self.path else ""
        return f"{prefix}{self.kind.value}: {self.detail}"


@dataclass(frozen=True)
class ManifestCatalog:
    manifests: tuple[LoadedManifest, ...]
    issues: tuple[ManifestIssue, ...]
    source: str
    ref: str
    applied: bool = False

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(issue.warning() for issue in self.issues)

@dataclass(frozen=True)
class _RequestedSource:
    source: str
    ref: str
    raw_source: str
    raw_ref: str
    local_root: Path | None = None


_lock = threading.Lock()
_snapshot: ManifestCatalog | None = None
last_source_description = ""


def _effective_source() -> tuple[str, bool]:
    directory = os.environ.get(AGENTS_DIR_ENV, "").strip()
    if directory:
        return directory, True
    return os.environ.get(AGENTS_SOURCE_ENV, DEFAULT_AGENTS_SOURCE).strip(), False


def _parse_source(spec: str) -> _RequestedSource:
    """Parse source/ref once; strip URL secrets from diagnostics."""
    clean = spec.split("?", 1)[0].split("#", 1)[0]
    local = Path(clean).expanduser()
    if local.is_dir():
        return _RequestedSource(clean, "", clean, "", local)
    source, separator, ref = clean.rpartition("@")
    userinfo_only = bool(
        separator and re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://[^/]*$", source)
    )
    if not separator or not ref or userinfo_only:
        source, ref = clean, ""
    safe_source = re.sub(r"(://)[^/@]+@", r"\1***@", source)
    return _RequestedSource(safe_source, ref, source, ref)


def _source_root(request: _RequestedSource) -> Path:
    if request.local_root is not None:
        return request.local_root
    from benchflow._utils.benchmark_repos import resolve_source

    return resolve_source(request.raw_source, ref=request.raw_ref or None)


def _read_catalog(spec: str) -> tuple[ManifestCatalog, dict[str, LoadedManifest]]:
    requested = _parse_source(spec)
    if not spec or spec.lower() in _OFF_VALUES:
        issue = ManifestIssue(ManifestIssueKind.DISABLED, "agents source disabled")
        return ManifestCatalog((), (issue,), requested.source or "off", ""), {}
    try:
        root = _source_root(requested)
    except Exception as exc:
        issue = ManifestIssue(
            ManifestIssueKind.UNREACHABLE,
            f"catalog unavailable ({type(exc).__name__})",
            cause=type(exc).__name__,
        )
        return ManifestCatalog((), (issue,), requested.source, requested.ref), {}

    loaded: dict[str, LoadedManifest] = {}
    issues: list[ManifestIssue] = []
    for path in discover_manifests(root):
        rel = path.relative_to(root).as_posix()
        try:
            manifest = load_agent_manifest(path)
            loaded[rel] = manifest
        except AgentManifestError as exc:
            issues.append(
                ManifestIssue(
                    exc.kind,
                    f"cannot load manifest ({type(exc).__name__})",
                    rel,
                    type(exc).__name__,
                )
            )
        except Exception as exc:
            issues.append(
                ManifestIssue(
                    ManifestIssueKind.MALFORMED,
                    f"cannot load manifest ({type(exc).__name__})",
                    rel,
                    type(exc).__name__,
                )
            )
    return ManifestCatalog((), tuple(issues), requested.source, requested.ref), loaded


def _select(
    catalog: ManifestCatalog,
    loaded: dict[str, LoadedManifest],
    *,
    local_override: bool,
) -> tuple[ManifestCatalog, dict[str, LoadedManifest]]:
    from benchflow.agents.registry import (
        _CORE_AGENT_CONFIGS,
        AGENT_ALIASES,
        AGENTS,
    )

    eligible = {
        manifest.config.name
        for manifest in loaded.values()
        if local_override
        and manifest.config.name in AGENTS
        and _CORE_AGENT_CONFIGS.get(manifest.config.name)
        == AGENTS[manifest.config.name]
    }
    agents = {name: config for name, config in AGENTS.items() if name not in eligible}
    selected, conflicts = select_manifest_agents(
        loaded, agents=agents, aliases=AGENT_ALIASES
    )
    collision_issues = tuple(
        ManifestIssue(kind, detail, path) for path, kind, detail in conflicts
    )
    for name in eligible & selected.keys():
        manifest = selected[name]
        selected[name] = LoadedManifest(
            _merge_core_shim_only(manifest.config, AGENTS[name]), manifest.aliases
        )

    issues = [*catalog.issues, *collision_issues]
    result = replace(
        catalog,
        manifests=tuple(selected.values()),
        issues=tuple(
            sorted(
                set(issues),
                key=lambda issue: (issue.path, issue.kind.value, issue.detail),
            )
        ),
    )
    return result, selected


def _register_catalog(
    catalog: ManifestCatalog,
    loaded: dict[str, LoadedManifest],
    *,
    local_override: bool,
) -> ManifestCatalog:
    from benchflow.agents.registry import (
        _REGISTRY_LOCK,
        AGENT_ALIASES,
        AGENT_INSTALLERS,
        AGENT_LAUNCH,
        AGENTS,
    )

    with _REGISTRY_LOCK:
        result, selected = _select(catalog, loaded, local_override=local_override)
        register_manifest_agents(
            selected,
            agents=AGENTS,
            aliases=AGENT_ALIASES,
            installers=AGENT_INSTALLERS,
            launch=AGENT_LAUNCH,
            override=True,
            merge_shim_only=local_override,
        )
    return replace(result, applied=True)


def ensure_manifest_catalog() -> ManifestCatalog:
    """Read, register, then publish one runtime catalog result."""
    global _snapshot, last_source_description
    with _lock:
        if _snapshot is None:
            spec, local_override = _effective_source()
            catalog, loaded = _read_catalog(spec)
            result = _register_catalog(catalog, loaded, local_override=local_override)
            _snapshot = result
            last_source_description = (
                "agents source disabled"
                if any(
                    issue.kind is ManifestIssueKind.DISABLED for issue in result.issues
                )
                else f"agents source {result.source!r}"
            )
        return _snapshot


def manifest_catalog_for_listing() -> ManifestCatalog:
    """Return applied result, or uncached/non-mutating preview."""
    with _lock:
        if _snapshot is not None:
            return _snapshot
        spec, local_override = _effective_source()
        catalog, loaded = _read_catalog(spec)
        return _select(catalog, loaded, local_override=local_override)[0]


def autoload_remote_manifest_agents() -> int:
    """Compatibility wrapper for miss-driven callers."""
    return len(ensure_manifest_catalog().manifests)


def _reset_for_tests() -> None:
    """Clear catalog cache only; fixtures own registry restoration."""
    global _snapshot, last_source_description
    with _lock:
        _snapshot = None
        last_source_description = ""
