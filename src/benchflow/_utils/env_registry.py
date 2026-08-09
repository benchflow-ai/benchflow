"""Environment registry — resolve ``name@version`` env specs to manifests.

This is the environment-axis twin of :func:`benchflow._utils.dataset_registry.
resolve_dataset`. The *task* declares which environment it needs by **name** (a
contract); the *run* resolves that name to a concrete, content-addressed
manifest **version** at the command line. That decouples the environment — Han
Lee's ``S`` axis in ``I_eval = {T, A, M, S, C, R, τ}`` — from the task, so it is
swappable per run exactly like ``--agent`` / ``--model`` / ``--sandbox`` already
are, instead of being pinned by a relative path baked into the task file.

Registry layout (a directory of manifest files)::

    <registry>/<name>@<version>.toml   # a pinned environment version
    <registry>/<name>.toml             # optional default ("latest")

Which directory is the registry:

1. an explicit ``registry=`` argument (tests, embedders);
2. else ``$BENCHFLOW_ENV_REGISTRY`` — when set it wins entirely (no fallback
   into the built-in registry for names it does not contain); an empty value
   counts as unset;
3. else the **built-in registry** shipped inside the ``benchflow`` wheel
   (``benchflow/environment/_registry/``), so a bare ``pip install benchflow``
   resolves the committed pins (``env0@prod``, ``env0@outage``) with no
   checkout and no env vars.

Resolution::

    env0          -> <registry>/env0.toml          (default) else newest @version
    env0@v2       -> <registry>/env0@v2.toml        (pinned)

Every resolution is content-addressed (``env_hash = sha256(manifest_bytes)``) so
a run records exactly which world it bound and can be replayed months later — the
reproducibility contract from "Decouple the task from the harness."
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchflow._utils.content_address import sha256_prefixed

if TYPE_CHECKING:
    from benchflow.environment.manifest import EnvironmentManifest

#: Env var naming the local environment registry directory.
ENV_REGISTRY_VAR = "BENCHFLOW_ENV_REGISTRY"

# A spec is ``<name>`` or ``<name>@<version>`` with no path separators.
_SPEC_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:@(?P<version>[A-Za-z0-9][A-Za-z0-9._-]*))?$"
)

#: Manifest extensions the registry resolves, in preference order. TOML first for
#: back-compat; YAML/YML are the canonical going-forward format (consistent with
#: the task / run / job configs).
_MANIFEST_EXTS = (".toml", ".yaml", ".yml")


class EnvironmentRegistryError(ValueError):
    """Raised when an environment spec cannot be resolved."""


@dataclass(frozen=True)
class ResolvedEnvironment:
    """A concrete, content-addressed environment bound for one run."""

    name: str
    version: str
    manifest_path: Path
    env_hash: str

    @property
    def spec(self) -> str:
        return f"{self.name}@{self.version}"


def looks_like_env_spec(value: str) -> bool:
    """True when ``value`` is a registry spec rather than a filesystem path.

    A path (contains a separator or ends in a manifest extension) is never a
    spec, so the historical file-path behavior of :func:`load_manifest` is kept.
    """
    if os.sep in value or "/" in value or value.endswith(_MANIFEST_EXTS):
        return False
    return bool(_SPEC_RE.match(value))


def _builtin_registry_dir() -> Path | None:
    """Directory of the pinned manifests shipped inside the ``benchflow`` wheel.

    ``benchflow/environment/_registry/`` is package data, so a bare
    ``pip install benchflow`` carries the committed pins. Path(__file__)-relative
    like every other package-data lookup in the repo (compose files, default
    rubric, demo task) — zip imports are unsupported package-wide. Returns
    ``None`` when the packaged registry is missing (a stripped install).
    """
    directory = Path(__file__).resolve().parents[1] / "environment" / "_registry"
    return directory if directory.is_dir() else None


def _registry_dir(registry: str | os.PathLike[str] | None) -> Path:
    if registry is not None:
        raw: str | os.PathLike[str] | None = registry
        source = "the registry= argument"
    else:
        raw = os.environ.get(ENV_REGISTRY_VAR)
        source = f"${ENV_REGISTRY_VAR}"
    if not raw:  # unset or empty — empty counts as unset
        builtin = _builtin_registry_dir()
        if builtin is not None:
            return builtin
        raise EnvironmentRegistryError(
            "no environment registry available: the built-in registry shipped "
            f"with benchflow is missing and ${ENV_REGISTRY_VAR} is unset; set "
            f"${ENV_REGISTRY_VAR} or pass --environment-manifest a manifest "
            "file path instead of a name@version spec"
        )
    directory = Path(raw).expanduser()
    if not directory.is_dir():
        raise EnvironmentRegistryError(
            f"environment registry {directory} (from {source}) is not a directory"
        )
    return directory


def _available_specs(directory: Path) -> list[str]:
    """Sorted stems (``name`` / ``name@version``) present in the registry."""
    return sorted({p.stem for ext in _MANIFEST_EXTS for p in directory.glob(f"*{ext}")})


def _find_manifest(directory: Path, stem: str) -> Path | None:
    """Return ``directory/stem.<ext>`` for the first existing manifest extension."""
    for ext in _MANIFEST_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def resolve_environment(
    spec: str, registry: str | os.PathLike[str] | None = None
) -> ResolvedEnvironment:
    """Resolve ``name`` / ``name@version`` to a content-addressed manifest.

    Mirrors :func:`resolve_dataset`: parse the spec, look it up in the registry,
    and return the pinned manifest path plus its content hash for provenance.
    """
    match = _SPEC_RE.match(spec)
    if not match:
        raise EnvironmentRegistryError(
            f"invalid environment spec {spec!r} — expected <name> or <name>@<version>"
        )
    name = match.group("name")
    version = match.group("version")
    directory = _registry_dir(registry)

    if version is not None:
        path = _find_manifest(directory, f"{name}@{version}")
        if path is None:
            available = ", ".join(_available_specs(directory)) or "none"
            raise EnvironmentRegistryError(
                f"environment {spec!r} not found in {directory} "
                f"(looked for {name}@{version}{{.toml,.yaml,.yml}}; "
                f"available: {available})"
            )
    else:
        path = _find_manifest(directory, name)
        if path is not None:
            version = "default"
        else:
            candidates = sorted(
                p for ext in _MANIFEST_EXTS for p in directory.glob(f"{name}@*{ext}")
            )
            if not candidates:
                available = ", ".join(_available_specs(directory)) or "none"
                raise EnvironmentRegistryError(
                    f"no versions of environment {name!r} found in {directory} "
                    f"(available: {available})"
                )
            path = candidates[-1]
            version = path.stem.split("@", 1)[1]

    env_hash = sha256_prefixed(path.read_bytes())
    return ResolvedEnvironment(
        name=name, version=version, manifest_path=path, env_hash=env_hash
    )


def resolve_state(value: str, registry: str | os.PathLike[str] | None = None):
    """Resolve a ``--state`` value (the ``S`` axis) to an ``EnvironmentManifest``.

    Accepts the two forms Han's whiteboard sketched:

    * **inline JSON** — ``{"name": "env0", "tools": ["gmail", "gcal"]}`` resolves
      ``env0`` from the registry and **filters its services to the tool subset**
      (start only gmail + gcal this run).
    * **a ``name@version`` spec or a manifest file path** — resolved as-is.

    The result is the same ``EnvironmentManifest`` the run binds, so ``--state`` is
    a richer front-end to ``--environment-manifest`` (which stays for back-compat).
    """
    from benchflow.environment.manifest import load_manifest

    text = value.strip()
    if not text.startswith("{"):
        return load_manifest(text)  # name / name@version / path

    try:
        spec = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EnvironmentRegistryError(f"invalid --state JSON: {exc}") from None
    if not isinstance(spec, dict) or "name" not in spec:
        raise EnvironmentRegistryError(
            '--state JSON must be a mapping with a "name", e.g. '
            '{"name": "env0", "tools": ["gmail", "gcal"]}'
        )

    manifest: EnvironmentManifest = load_manifest(str(spec["name"]))
    tools = spec.get("tools")
    if tools:
        available = {s.name for s in manifest.services}
        missing = [t for t in tools if t not in available]
        if missing:
            raise EnvironmentRegistryError(
                f"--state tools {missing} not in environment "
                f"{spec['name']!r} (available: {sorted(available)})"
            )
        kept = [s for s in manifest.services if s.name in set(tools)]
        manifest = manifest.model_copy(update={"services": kept})
    return manifest
