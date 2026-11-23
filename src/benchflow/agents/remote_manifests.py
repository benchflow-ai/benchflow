"""One-shot loading of declarative agent manifests."""

from __future__ import annotations

import dataclasses
import hashlib
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
# Lets a local agents folder's manifest replace the core agent it names even
# where it differs (agents-repo development); unset, core runs.
OVERRIDE_CORE_ENV = "BENCHFLOW_AGENTS_OVERRIDE_CORE"
# The catalog a release loads when nothing overrides it: a commit, never a
# branch, so a release keeps resolving the manifests it was tested with.
# agents 6f19d628d: agents main 9aaec336b (after #71-#81 merged) plus three
# fixes for the agent's neutral names. mini-swe-acp 0.1.2 installs under
# benchflow._runtime_tree.RUNTIME_ROOT, and mini-swe, OpenClaw and Prime
# Agent read RUNTIME_* before BENCHFLOW_*, so the same commit runs on
# BenchFlow before and after the rename. mini-swe keeps its private managed
# CPython, the mini-swe-agent 2.4.6 / LiteLLM 1.105.1 pins and its cacheable
# install. The deterministic mini-swe test loads the plugin from this same
# commit; the agents commit and this pin ship together.
# BENCHFLOW_AGENTS_SOURCE overrides it (any ref, `benchflow-ai/agents@main`,
# a directory, or `off`); a nonblank BENCHFLOW_AGENTS_DIR overrides both.
DEFAULT_AGENTS_SOURCE = "benchflow-ai/agents@6f19d628d1c761d56bafa0b24add66b403f69c5a"
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
class InvalidManifest:
    """A manifest a local agents folder holds that fails its strict check
    (``agents.manifest_check``): listed as invalid, never run."""

    name: str | None
    path: str
    problems: tuple[str, ...]


@dataclass(frozen=True)
class ManifestCatalog:
    manifests: tuple[LoadedManifest, ...]
    issues: tuple[ManifestIssue, ...]
    source: str
    ref: str
    applied: bool = False
    invalid: tuple[InvalidManifest, ...] = ()
    overrides: tuple[CoreOverride, ...] = ()

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(issue.warning() for issue in self.issues)


@dataclass(frozen=True)
class CoreOverride:
    """A core agent a local manifest replaced under OVERRIDE_CORE_ENV: the
    manifest as the agents folder names it, its bytes' SHA-256 and the fields
    it changes. A run records it as ``agent_source``."""

    name: str
    manifest: str
    sha256: str
    differs_in: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {
            "manifest": self.manifest,
            "sha256": self.sha256,
            "overrides_core": True,
            "differs_in": list(self.differs_in),
        }

    def text(self) -> str:
        return (
            f"{self.name} runs from {self.manifest} ({self.sha256[:23]}), not "
            f"BenchFlow's core {self.name}: {OVERRIDE_CORE_ENV}=1 lets it replace "
            f"the core agent (it differs in {', '.join(self.differs_in)})"
        )


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
    local = Path(spec).expanduser()
    if local.is_dir():
        return _RequestedSource(spec, "", spec, "", local)
    source, separator, ref = spec.rpartition("@")
    userinfo_only = bool(
        separator and re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://[^/]*$", source)
    )
    if not separator or not ref or userinfo_only:
        source, ref = spec, ""
    # Redaction changes diagnostics only. A local name or valid Git ref can
    # contain '#'; trimming it before resolution selects a different catalog.
    safe_source = re.sub(r"(://)[^/@]+@", r"\1***@", source)
    safe_source = safe_source.split("?", 1)[0].split("#", 1)[0]
    safe_ref = ref.split("?", 1)[0].split("#", 1)[0]
    return _RequestedSource(safe_source, safe_ref, source, ref)


def _source_root(request: _RequestedSource) -> Path:
    if request.local_root is not None:
        return request.local_root
    from benchflow._utils.benchmark_repos import resolve_source

    return resolve_source(request.raw_source, ref=request.raw_ref or None)


def _read_catalog(
    spec: str, *, local_override: bool = False
) -> tuple[ManifestCatalog, dict[str, LoadedManifest]]:
    requested = _parse_source(spec)
    if not spec or spec.lower() in _OFF_VALUES:
        issue = ManifestIssue(ManifestIssueKind.DISABLED, "agents source disabled")
        return ManifestCatalog((), (issue,), requested.source or "off", ""), {}
    if local_override and requested.local_root is None:
        # BENCHFLOW_AGENTS_DIR names a folder, never a remote catalog.
        issue = ManifestIssue(
            ManifestIssueKind.MISSING,
            f"{AGENTS_DIR_ENV}={spec} names no folder "
            f"({Path(spec).expanduser().absolute()})",
        )
        return ManifestCatalog((), (issue,), spec, ""), {}
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
    invalid: list[InvalidManifest] = []
    for path in discover_manifests(root):
        rel = path.relative_to(root).as_posix()
        if requested.local_root is not None:
            # A local folder is the author's own: its manifests are checked
            # strictly, and one that fails stays visible as invalid.
            from benchflow.agents.manifest_check import (
                check_manifest,
                manifest_identity,
            )

            shown = (Path(spec) / rel).as_posix()
            problems = check_manifest(path)
            if problems:
                texts = tuple(
                    p.text().replace(path.as_posix(), shown, 1) for p in problems
                )
                name = manifest_identity(path).name
                invalid.append(InvalidManifest(name, shown, texts))
                first = texts[0].replace(f"{shown}:", "line ", 1)
                more = f" and {len(texts) - 1} more" if len(texts) > 1 else ""
                issues.append(
                    ManifestIssue(
                        ManifestIssueKind.MALFORMED,
                        f"agent {name or '?'} is invalid: {first}{more} "
                        "(`bench agents validate` on this file lists every problem)",
                        rel,
                    )
                )
                continue
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
    catalog = ManifestCatalog(
        (), tuple(issues), requested.source, requested.ref, invalid=tuple(invalid)
    )
    return catalog, loaded


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

    # A local folder's manifest may stand for a core agent only when it is
    # that agent: merged with core's host-side fields it equals the core
    # definition, so the launch path, policy plumbing and model-id mapping
    # are core's. One that differs would silently change a working core
    # agent (Claude, Codex, OpenCode, pi, DeepAgents and MiMo broke
    # under the pinned catalog): core runs, and the conflict is said, unless
    # BENCHFLOW_AGENTS_OVERRIDE_CORE asks for the manifest by name.
    override_core = os.environ.get(OVERRIDE_CORE_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    eligible: set[str] = set()
    drift: list[ManifestIssue] = []
    kept_out: set[str] = set()
    overrides: dict[str, CoreOverride] = {}
    for path, manifest in loaded.items():
        name = manifest.config.name
        if not (
            local_override
            and name in AGENTS
            and _CORE_AGENT_CONFIGS.get(name) == AGENTS[name]
        ):
            continue
        core = AGENTS[name]
        merged = _merge_core_shim_only(manifest.config, core)
        fields = [
            f.name
            for f in dataclasses.fields(core)
            if getattr(merged, f.name) != getattr(core, f.name)
        ]
        if not fields:
            eligible.add(name)
        elif override_core:
            eligible.add(name)
            folder = Path(catalog.source)
            overrides[name] = CoreOverride(
                name,
                (folder / path).as_posix(),
                "sha256:"
                + hashlib.sha256((folder.expanduser() / path).read_bytes()).hexdigest(),
                tuple(fields),
            )
            drift.append(
                ManifestIssue(
                    ManifestIssueKind.COLLISION,
                    f"replaces the core agent {name!r} ({OVERRIDE_CORE_ENV}): it "
                    f"differs in {', '.join(fields)}",
                    path,
                )
            )
        else:
            kept_out.add(path)
            drift.append(
                ManifestIssue(
                    ManifestIssueKind.COLLISION,
                    f"names the core agent {name!r} but differs from it in "
                    f"{', '.join(fields)}: BenchFlow runs its core {name!r} "
                    f"(set {OVERRIDE_CORE_ENV}=1 to run this manifest instead)",
                    path,
                )
            )
    loaded = {path: m for path, m in loaded.items() if path not in kept_out}
    agents = {name: config for name, config in AGENTS.items() if name not in eligible}
    selected, conflicts = select_manifest_agents(
        loaded, agents=agents, aliases=AGENT_ALIASES
    )
    collision_issues = (
        *drift,
        *(
            ManifestIssue(kind, detail, path)
            for path, kind, detail in conflicts
            if local_override
            or kind is not ManifestIssueKind.COLLISION
            or loaded[path].config.name not in AGENTS
        ),
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
        overrides=tuple(o for name, o in sorted(overrides.items()) if name in selected),
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
            catalog, loaded = _read_catalog(spec, local_override=local_override)
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
        catalog, loaded = _read_catalog(spec, local_override=local_override)
        return _select(catalog, loaded, local_override=local_override)[0]


def invalid_manifest(name: str) -> InvalidManifest | None:
    """The invalid manifest the configured local agents folder holds for
    ``name``, if any (the catalog is read once, as a run reads it)."""
    catalog = ensure_manifest_catalog()
    return next((m for m in catalog.invalid if m.name == name), None)


def catalog_problem() -> str | None:
    """Why the configured agents source gave no agents at all (a missing
    BENCHFLOW_AGENTS_DIR folder), or None."""
    catalog = ensure_manifest_catalog()
    missing = [i for i in catalog.issues if i.kind is ManifestIssueKind.MISSING]
    return missing[0].detail if missing else None


def missing_manifest(name: str) -> str | None:
    """For an agent the local agents folder has a folder of but no manifest
    in (``agents/<name>/`` without ``manifest.toml``): what is missing."""
    directory = os.environ.get(AGENTS_DIR_ENV, "").strip()
    if not directory:
        return None
    folder = Path(directory).expanduser() / name
    if folder.is_dir() and not (folder / "manifest.toml").is_file():
        return f"{(folder / 'manifest.toml').as_posix()} does not exist"
    return None


def core_override(name: str | None) -> CoreOverride | None:
    """The local manifest that replaced the core agent ``name`` under
    OVERRIDE_CORE_ENV in the catalog as loaded, or None (the core agent, or
    no catalog loaded)."""
    with _lock:
        snapshot = _snapshot
    if snapshot is None or not name:
        return None
    return next((o for o in snapshot.overrides if o.name == name), None)


def agent_source(name: str | None) -> dict[str, object] | None:
    """``agent_source`` for config.json, result.json and the run card: set
    only for a core agent a local manifest replaced."""
    found = core_override(name)
    return found.record() if found else None


def manifest_agent_names() -> frozenset[str]:
    """The agents a manifest declared, from a local agents folder or the
    agents catalog, as loaded: integration_health judges a turn of theirs
    with no ACP event and no model output as a broken handshake."""
    with _lock:
        snapshot = _snapshot
    if snapshot is None:
        return frozenset()
    return frozenset(m.config.name for m in snapshot.manifests)


def autoload_remote_manifest_agents() -> int:
    """Compatibility wrapper for miss-driven callers."""
    return len(ensure_manifest_catalog().manifests)


def autoload_local_manifest_agents() -> None:
    """Apply an explicitly configured local catalog before registry lookup."""
    if os.environ.get(AGENTS_DIR_ENV, "").strip():
        ensure_manifest_catalog()


def _reset_for_tests() -> None:
    """Clear catalog cache only; fixtures own registry restoration."""
    global _snapshot, last_source_description
    with _lock:
        _snapshot = None
        last_source_description = ""
