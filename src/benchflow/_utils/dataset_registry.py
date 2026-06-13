"""Dataset registry resolution — `bench eval create -d skillsbench@1.1`.

Resolves a published dataset version from the skillsbench registry
(``registry.json``, see ``docs/dataset-versioning.md`` in
benchflow-ai/skillsbench): clones the pinned snapshot, verifies every
task's content digest against the registry entry, and hands back the
task set plus source provenance for result stamping.
"""

import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.request import urlopen

from benchflow._utils import benchmark_repos
from benchflow._utils.task_authoring import task_digest

DEFAULT_REGISTRY_SOURCE = (
    "https://raw.githubusercontent.com/benchflow-ai/skillsbench/main/registry.json"
)

_GITHUB_URL_RE = re.compile(
    r"^(?:https://|git@)github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$"
)


class DatasetResolutionError(ValueError):
    """A dataset spec could not be resolved against the registry."""


class DatasetDigestMismatchError(DatasetResolutionError):
    """Snapshot content does not match the registry's pinned digests."""


@dataclass(frozen=True)
class ResolvedDataset:
    """A registry entry resolved to verified task directories on disk."""

    name: str
    version: str
    bench_version: str | None
    tasks_dir: Path
    task_names: set[str]
    task_digests: dict[str, str]
    provenance: dict[str, Any]

    @property
    def spec(self) -> str:
        return f"{self.name}@{self.version}"


def parse_dataset_spec(spec: str) -> tuple[str, str]:
    """Split ``<name>@<version>`` — the version is mandatory.

    Published versions are immutable, so there is no "latest" default: a
    run must name the exact version it pins.
    """
    name, sep, version = spec.partition("@")
    if not sep or not name or not version:
        raise DatasetResolutionError(
            f"Invalid dataset spec {spec!r} — expected <name>@<version> "
            f"(e.g. skillsbench@1.1)"
        )
    return name, version


def load_registry(source: str) -> list[dict[str, Any]]:
    """Load a dataset registry from an HTTP(S) URL or a local file path."""
    try:
        if source.startswith(("http://", "https://")):
            with urlopen(source, timeout=30) as response:
                raw = response.read().decode("utf-8")
        else:
            raw = Path(source).read_text()
    except OSError as exc:  # URLError + FileNotFoundError both subclass OSError
        raise DatasetResolutionError(
            f"Could not read registry at {source}: {exc}"
        ) from exc
    try:
        registry = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetResolutionError(
            f"Registry at {source} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(registry, list):
        raise DatasetResolutionError(
            f"Registry at {source} is not a list of dataset entries"
        )
    return registry


def bench_version_issue(declared_range: str | None) -> str | None:
    """Warning text when the running bench falls outside the dataset's
    validated ``bench_version`` range, else None. Advisory only — the
    range records what the version was validated against, it is not a
    hard gate."""
    if not declared_range:
        return None
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version

    from benchflow import __version__

    try:
        specifier = SpecifierSet(declared_range)
    except InvalidSpecifier:
        return f"registry declares an unparseable bench_version {declared_range!r}"
    try:
        current = Version(__version__)
    except InvalidVersion:
        return (
            f"cannot compare bench version {__version__!r} against the "
            f"dataset's validated range {declared_range!r}"
        )
    # Compare on the BASE version (strip pre/post/dev) so a release candidate or
    # dev build counts as in-range for a spec that includes its release line.
    # PEP 440 orders 0.6.0rc6 < 0.6.0, so a bare `>=0.6` would otherwise flag the
    # very release being validated (v0.6 ships as 0.6.0rcN) as out-of-range — and
    # the planned bench_version hard-gate would then block every RC user from
    # dataset runs. base_version maps 0.6.0rc6 -> 0.6.0, the line it belongs to.
    if specifier.contains(Version(current.base_version), prereleases=True):
        return None
    return (
        f"bench {__version__} is outside the range this dataset was "
        f"validated against ({declared_range}) — results may not be "
        f"comparable with published runs"
    )


def _github_org_repo(git_url: str) -> str:
    match = _GITHUB_URL_RE.match(git_url)
    if not match:
        raise DatasetResolutionError(
            f"Unsupported git_url {git_url!r} — expected a github.com repository"
        )
    return f"{match.group(1)}/{match.group(2)}"


def resolve_dataset(
    spec: str, registry: str = DEFAULT_REGISTRY_SOURCE
) -> ResolvedDataset:
    """Resolve ``<name>@<version>`` to digest-verified task dirs on disk.

    Clones the snapshot pinned by the registry entry (by commit id, so a
    moved git tag cannot change what runs), then recomputes every task's
    content digest and fails hard on any mismatch — a dataset version is
    an immutable artifact and must byte-match its registry entry.
    """
    name, version = parse_dataset_spec(spec)
    entries = load_registry(registry)
    entry = next(
        (
            e
            for e in entries
            if isinstance(e, dict)
            and e.get("name") == name
            and str(e.get("version")) == version
        ),
        None,
    )
    if entry is None:
        available = (
            ", ".join(
                f"{e.get('name')}@{e.get('version')}"
                for e in entries
                if isinstance(e, dict)
            )
            or "none"
        )
        raise DatasetResolutionError(
            f"Dataset {spec!r} not found in registry (available: {available})"
        )
    tasks = entry.get("tasks") or []
    if not tasks:
        raise DatasetResolutionError(f"Dataset {spec!r} has no tasks in the registry")

    def _require(task: object, key: str) -> str:
        # A registry written by hand / a stale tool can have task entries that
        # are non-objects or miss a key; surface a clean DatasetResolutionError
        # instead of a raw KeyError / TypeError ('string indices must be integers').
        if not isinstance(task, dict) or key not in task:
            raise DatasetResolutionError(
                f"Dataset {spec!r} has a malformed task entry "
                f"(expected an object with {key!r}): {task!r}"
            )
        return str(cast("dict[str, Any]", task)[key])

    snapshots = {(_require(t, "git_url"), _require(t, "git_commit_id")) for t in tasks}
    if len(snapshots) != 1:
        raise DatasetResolutionError(
            f"Dataset {spec!r} spans {len(snapshots)} git snapshots — "
            f"only single-snapshot datasets are supported"
        )
    git_url, commit = next(iter(snapshots))

    parents = {posixpath.dirname(_require(t, "path").rstrip("/")) for t in tasks}
    if len(parents) != 1:
        raise DatasetResolutionError(
            f"Dataset {spec!r} tasks live under {len(parents)} parent "
            f"directories — only a single tasks directory is supported"
        )
    parent = next(iter(parents))

    task_digests: dict[str, str] = {}
    for t in tasks:
        base = posixpath.basename(_require(t, "path").rstrip("/"))
        if base in task_digests:
            raise DatasetResolutionError(
                f"Dataset {spec!r} has two tasks named {base!r}"
            )
        task_digests[base] = _require(t, "digest")

    repo = _github_org_repo(git_url)
    resolved = benchmark_repos.resolve_source_with_metadata(
        repo, path=parent or None, ref=commit
    )
    resolved_sha = resolved.provenance.get("resolved_sha")
    if resolved_sha != commit:
        raise DatasetResolutionError(
            f"Snapshot for {spec!r} resolved to {resolved_sha} but the "
            f"registry pins {commit}"
        )

    mismatches = []
    for base, expected in sorted(task_digests.items()):
        candidate = resolved.path / base
        if not candidate.is_dir():
            mismatches.append(f"{base}: missing from snapshot")
            continue
        actual = task_digest(candidate)
        if actual != expected:
            mismatches.append(f"{base}: computed {actual}, registry pins {expected}")
    if mismatches:
        raise DatasetDigestMismatchError(
            f"Dataset {spec!r} content does not match its registry digests:\n  "
            + "\n  ".join(mismatches)
        )

    return ResolvedDataset(
        name=name,
        version=version,
        bench_version=entry.get("bench_version"),
        tasks_dir=resolved.path,
        task_names=set(task_digests),
        task_digests=task_digests,
        provenance=resolved.provenance,
    )
