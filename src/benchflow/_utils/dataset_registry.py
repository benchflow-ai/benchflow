"""Dataset registry resolution — `bench eval run -d skillsbench@1.1`.

Resolves a published dataset version from a registry (``registry.json``,
see ``docs/dataset-versioning.md`` in benchflow-ai/skillsbench): clones the
pinned snapshot, verifies every task's content digest against the registry
entry, and hands back the task set plus source provenance for result
stamping. With no registry named, BenchFlow's registry is read, and only
it: its published list, then the entries this package ships for datasets
that list does not hold (Terminal-Bench 2.0). A registry named with
``--registry`` is read alone; a task it gives no digest for runs unchecked
and the run says so.
"""

import difflib
import hashlib
import json
import os
import posixpath
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from benchflow._utils import benchmark_repos
from benchflow._utils.task_authoring import task_digest
from benchflow.errors import UserError

DEFAULT_REGISTRY_SOURCE = (
    "https://raw.githubusercontent.com/benchflow-ai/skillsbench/main/registry.json"
)
# BenchFlow's registry entries shipped in the package, every task pinned by
# commit and content digest: the datasets the published list does not hold.
PACKAGED_REGISTRY_SOURCE = str(Path(__file__).with_name("dataset_registry.json"))
# BenchFlow's registry, read in this order when no registry is named.
DEFAULT_REGISTRY_SOURCES = (DEFAULT_REGISTRY_SOURCE, PACKAGED_REGISTRY_SOURCE)
# The next step when BenchFlow's registry does not have a dataset.
OTHER_REGISTRY_HINT = (
    "pick a dataset that BenchFlow's registry lists (bench hub list "
    "--provider benchflow), or name a registry that has it with "
    "--registry <URL or file>"
)
# A git_commit_id that pins nothing: the repository's default branch.
_UNPINNED_COMMITS = {"", "head"}

# Only GitHub's own name characters: org and repo name the source cache's
# folders (`.cache/datasets-v2/<org>/<repo>`), and an entry from a downloaded
# registry is untrusted. `[^/]+` let `https://github.com/../..` through, and
# its cache path was the git root of the user's own project, where the
# refresh then fetched and checked out their origin's default branch.
_GITHUB_URL_RE = re.compile(
    r"^(?:https://|git@)github\.com[:/]"
    r"([A-Za-z0-9][A-Za-z0-9-]*)/([A-Za-z0-9._-]+?)(?:\.git)?/?$"
)
# A pinned git_commit_id is a full commit id (SHA-1 or SHA-256); anything
# else from the registry would reach `git fetch` and `git clone` as given.
_COMMIT_ID_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class DatasetResolutionError(ValueError, UserError):
    """A dataset spec could not be resolved against the registry.

    ``exit_code`` is 2 for a usage mistake (a bad spec, a dataset the registry
    does not hold, a missing local registry) and 1 otherwise (a registry
    whose contents or remote response could not be read).
    """

    exit_code: int = 1


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
    registry: str = DEFAULT_REGISTRY_SOURCE

    @property
    def spec(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def unpinned_tasks(self) -> set[str]:
        """Tasks the registry gives no digest for: run, but not checked."""
        return self.task_names - set(self.task_digests)


def registry_label(source: str) -> str:
    """A registry as messages name it."""
    if source in DEFAULT_REGISTRY_SOURCES:
        return "BenchFlow's registry"
    return f"registry at {source}"


def load_benchflow_registry() -> tuple[list[dict[str, Any]], list[str]]:
    """BenchFlow's registry: its published list, then the packaged entries.

    Returns the entries and the message of each part that could not be read
    (the published list needs the network; the packaged entries do not).
    """
    entries: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for source in DEFAULT_REGISTRY_SOURCES:
        try:
            entries.extend(e for e in load_registry(source) if isinstance(e, dict))
        except DatasetResolutionError as exc:
            unreadable.append(str(exc))
    return entries, unreadable


def parse_dataset_spec(spec: str) -> tuple[str, str]:
    """Split ``<name>@<version>`` — the version is mandatory.

    Published versions are immutable, so there is no "latest" default: a
    run must name the exact version it pins.
    """
    name, sep, version = spec.partition("@")
    if not sep or not name or not version:
        error = DatasetResolutionError(
            f"Invalid dataset spec {spec!r} — expected <name>@<version> "
            f"(e.g. skillsbench@1.1)"
        )
        error.exit_code = 2  # a usage mistake (bench's exit codes)
        raise error
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
        error = DatasetResolutionError(
            f"Could not read {registry_label(source)}: {exc}"
        )
        if isinstance(exc, URLError) and not isinstance(exc, HTTPError):
            # The server was not reached (an HTTP status means it was).
            error.hint = benchmark_repos._OFFLINE_NEXT
        elif isinstance(exc, FileNotFoundError):
            error.exit_code = 2
            error.hint = "pass an existing registry JSON file or URL to --registry"
        raise error from exc
    try:
        registry = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetResolutionError(
            f"{_capital(registry_label(source))} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(registry, list):
        raise DatasetResolutionError(
            f"{_capital(registry_label(source))} is not a list of dataset entries"
        )
    return registry


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _entry(entries: list[Any], name: str, version: str) -> dict[str, Any] | None:
    return next(
        (
            e
            for e in entries
            if isinstance(e, dict)
            and e.get("name") == name
            and str(e.get("version")) == version
        ),
        None,
    )


def _find_entry(
    spec: str, name: str, version: str, registry: str | None
) -> tuple[dict[str, Any], str]:
    """The registry entry for ``name@version`` and the registry it came from.

    A named registry is the only one read. Otherwise only BenchFlow's
    registry is: its published list, then its packaged entries. A dataset it
    does not hold is an error that names BenchFlow's registry and how to
    read another one (``--registry``); no other registry is tried.
    """
    if registry is not None:
        entries = load_registry(registry)
        entry = _entry(entries, name, version)
        if entry is None:
            available = (
                ", ".join(
                    f"{e.get('name')}@{e.get('version')}"
                    for e in entries
                    if isinstance(e, dict)
                )
                or "none"
            )
            error = DatasetResolutionError(
                f"Dataset {spec!r} not found in registry (available: {available})"
            )
            error.exit_code = 2
            raise error
        return entry, registry
    unreadable: list[str] = []
    every: list[dict[str, Any]] = []
    for source in DEFAULT_REGISTRY_SOURCES:
        try:
            entries = load_registry(source)
        except DatasetResolutionError as exc:
            unreadable.append(str(exc))
            continue
        entry = _entry(entries, name, version)
        if entry is not None:
            return entry, source
        every.extend(e for e in entries if isinstance(e, dict))
    versions = sorted({str(e.get("version")) for e in every if e.get("name") == name})
    names = sorted({str(e.get("name")) for e in every if e.get("name")})
    close = difflib.get_close_matches(name, names, n=5, cutoff=0.75)
    if versions:
        found = f" (its versions there: {', '.join(versions)})"
    else:
        found = f"; similar names: {', '.join(close)}" if close else ""
    if unreadable:
        # The part that could not be read may hold the dataset.
        error = DatasetResolutionError(
            f"Dataset {spec!r} could not be looked up in BenchFlow's registry: "
            f"{'; '.join(unreadable)}{found}"
        )
        error.hint = "check this machine's network, or " + OTHER_REGISTRY_HINT
        raise error
    error = DatasetResolutionError(
        f"Dataset {spec!r} not found in BenchFlow's registry{found}"
    )
    error.hint = OTHER_REGISTRY_HINT
    error.exit_code = 2
    raise error


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
    if not match or match.group(2) in {".", ".."}:
        raise DatasetResolutionError(
            f"Unsupported git_url {git_url!r} — expected a github.com repository"
        )
    return f"{match.group(1)}/{match.group(2)}"


def _hub_for(
    name: str, registry: str | None, exc: DatasetResolutionError | None = None
) -> str | None:
    """The Hub to read ``name`` from: the one ``--registry hub:<url>`` names, or, when no registry was named and
    BenchFlow's registry does not hold the dataset (``exc``), ``$BENCHFLOW_HUB_URL`` for an ``org/name`` dataset."""
    from benchflow.hub.client import HUB_URL_ENV, hub_from_registry, normalize_hub_url

    named = hub_from_registry(registry)
    if named is not None or registry is not None:
        return named
    fallback = os.environ.get(HUB_URL_ENV)
    if exc is not None and exc.exit_code == 2 and fallback and "/" in name:
        return normalize_hub_url(fallback)
    return None


def _hub_entry(spec: str, name: str, version: str, hub: str) -> dict[str, Any]:
    """A Hub's dataset document as a registry entry (name, version, bench_version, tasks with digests)."""
    from benchflow.hub.client import HubClient, HubError

    try:
        doc = HubClient.from_env(hub).dataset(name, version)
    except HubError as exc:
        error = DatasetResolutionError(
            f"Dataset {spec!r} could not be read from the Hub at {hub}: {exc}"
        )
        error.hint = exc.hint or "check the name@version with `bench hub show`"
        error.exit_code = 2 if exc.status == 404 else 1
        raise error from None
    return {
        "name": doc.get("name"),
        "version": doc.get("version"),
        "bench_version": doc.get("bench_version"),
        "tasks": [
            {"name": t.get("name"), "digest": t.get("digest")}
            for t in doc.get("tasks") or []
        ],
        "hub": hub,
        "digest": doc.get("digest"),
    }


def dataset_registry_entry(spec: str, *, registry: str | None = None) -> dict[str, Any]:
    """Read a dataset's declared pins without cloning or starting a sandbox.

    Looks the entry up the way ``resolve_dataset`` does (one named registry,
    or BenchFlow's own, then ``$BENCHFLOW_HUB_URL``), so a check sees what a run
    sees.
    """
    name, version = parse_dataset_spec(spec)
    hub = _hub_for(name, registry)
    if hub is not None:
        return _hub_entry(spec, name, version, hub)
    try:
        entry, _source = _find_entry(spec, name, version, registry)
    except DatasetResolutionError as exc:
        hub = _hub_for(name, registry, exc)
        if hub is None:
            raise
        return _hub_entry(spec, name, version, hub)
    return entry


def _checkout_blob_matches(
    candidate: Path, commit: str, name: str, oid: str, actual_digest: str
) -> bool:
    """Compare a built-in checkout conversion using only pinned attributes.

    Git's index flags, worktree/info attributes and configured clean/smudge
    drivers are mutable. An empty private repository reads the pinned tree
    through an alternate object store, with its own index and no filters or
    replacement refs. Materialize only regular pinned attribute files,
    including each file's parents outside the selected task; cat-file's
    conversion on older Git does not fall back to index attributes.
    """
    objects = benchmark_repos._git_stdout(
        candidate, "rev-parse", "--git-path", "objects"
    )
    prefix = benchmark_repos._git_stdout(candidate, "rev-parse", "--show-prefix")
    if not objects or prefix is None:
        return False
    # Git 2.26 has --git-path but not --path-format. Ordinary repositories
    # can return a relative path; linked worktrees usually return an absolute one.
    objects = str((candidate / objects).resolve())
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_ATTR_NOSYSTEM="1",
    )
    try:
        with tempfile.TemporaryDirectory(prefix="bench-checkout-") as directory:
            root = Path(directory)
            # Git before 2.32 ignores GIT_CONFIG_GLOBAL. Isolate both user
            # config roots before even init, so no ambient driver or include
            # can enter the private repository's built-in-only conversion.
            home = root / "home"
            xdg = root / "xdg"
            home.mkdir()
            xdg.mkdir()
            env.update(HOME=str(home), XDG_CONFIG_HOME=str(xdg))
            worktree = root / "worktree"
            worktree.mkdir()

            def git(*args: str) -> bytes:
                return subprocess.run(
                    [
                        "git",
                        "--no-replace-objects",
                        "-c",
                        "core.bare=false",
                        "-c",
                        f"core.attributesFile={os.devnull}",
                        *args,
                    ],
                    cwd=worktree,
                    env=env,
                    pass_fds=benchmark_repos._CACHE_LOCK_FDS.get(),
                    check=True,
                    capture_output=True,
                ).stdout

            git(
                "init",
                "--bare",
                "--template=",
                # SHA-1 is the default on supported Git 2.26, which predates
                # --object-format. SHA-256 repositories already need newer Git.
                *(["--object-format=sha256"] if len(oid) == 64 else []),
                str(root / "repo"),
            )
            env.update(
                GIT_DIR=str(root / "repo"),
                GIT_WORK_TREE=str(worktree),
                GIT_ALTERNATE_OBJECT_DIRECTORIES=objects,
            )
            git("read-tree", commit)
            parent = Path(prefix + name).parent
            parents = (parent, *parent.parents)
            attributes = [str(parent / ".gitattributes") for parent in parents]
            env["GIT_LITERAL_PATHSPECS"] = "1"
            records = git("ls-files", "--stage", "-z", "--", *attributes)
            for record in records.split(b"\0"):
                if not record:
                    continue
                metadata, path = record.split(b"\t", 1)
                mode, attribute_oid, _ = metadata.split()
                # Git does not follow attribute-file symlinks. Never make
                # one in the private worktree either.
                if mode not in {b"100644", b"100755"}:
                    continue
                destination = worktree / os.fsdecode(path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(
                    git("cat-file", "blob", attribute_oid.decode("ascii"))
                )
            contents = git("cat-file", "--filters", f"--path={prefix}{name}", oid)
            digest = hashlib.new("sha256" if len(oid) == 64 else "sha1")
            digest.update(f"blob {len(contents)}\0".encode("ascii"))
            digest.update(contents)
            return digest.hexdigest() == actual_digest
    except (OSError, subprocess.CalledProcessError):
        return False


def _task_matches_commit(candidate: Path, commit: str) -> bool:
    """Compare task bytes and modes to the tree, without trusting the index.

    Status honors assume-unchanged, skip-worktree, stat caches and clean
    filters. Read unreplaced tree object IDs instead and hash actual bytes,
    including ignored/untracked files. Only pinned built-in checkout
    conversions may differ from a blob. Never reset or change index flags.
    """
    listing = benchmark_repos._git_stdout(
        candidate, "--no-replace-objects", "ls-tree", "-r", "-z", commit, "--", "."
    )
    if listing is None or candidate.is_symlink():
        return False
    expected = {}
    for record in listing.split("\0"):
        if not record:
            continue
        meta, _, name = record.partition("\t")
        fields = meta.split()
        if len(fields) != 3 or fields[1] != "blob":
            return False
        expected[name] = (fields[0], fields[2])
    seen = set()
    walk_errors = []
    try:
        for directory, dirs, files in candidate.walk(on_error=walk_errors.append):
            # A registry may name a task at the repository root. Its .git
            # administrative file/directory is not part of the task tree.
            if directory == candidate and (candidate / ".git").exists():
                if ".git" in dirs:
                    dirs.remove(".git")
                if ".git" in files:
                    files.remove(".git")
            for filename in files:
                path = directory / filename
                info = path.lstat()
                name = path.relative_to(candidate).as_posix()
                entry = expected.get(name)
                if entry is None:
                    return False
                mode, oid = entry
                digest = hashlib.new("sha256" if len(oid) == 64 else "sha1")
                if stat.S_ISLNK(info.st_mode):
                    if mode != "120000":
                        return False
                    contents = os.fsencode(os.readlink(path))
                    digest.update(f"blob {len(contents)}\0".encode("ascii"))
                    digest.update(contents)
                elif stat.S_ISREG(info.st_mode):
                    actual_mode = "100755" if info.st_mode & stat.S_IXUSR else "100644"
                    if mode != actual_mode:
                        return False
                    digest.update(f"blob {info.st_size}\0".encode("ascii"))
                    with path.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                else:
                    return False
                actual_digest = digest.hexdigest()
                if actual_digest != oid and (
                    mode == "120000"
                    or not _checkout_blob_matches(
                        candidate, commit, name, oid, actual_digest
                    )
                ):
                    return False
                seen.add(name)
    except OSError:
        return False
    return not walk_errors and seen == expected.keys()


def resolve_dataset(spec: str, registry: str | None = None) -> ResolvedDataset:
    """Resolve ``<name>@<version>`` to digest-verified task dirs on disk.

    Clones the snapshot pinned by the registry entry (by commit id, so a
    moved git tag cannot change what runs), then recomputes every task's
    content digest and fails hard on any mismatch — a dataset version is
    an immutable artifact and must byte-match its registry entry. A task the
    entry gives no digest for (a registry named with ``--registry`` may
    pin none) is run unchecked and listed in ``unpinned_tasks``; an entry whose
    ``git_commit_id`` is ``HEAD`` or missing runs the default branch.

    ``registry`` names the one registry to read; by default only BenchFlow's
    registry is read (its published list and its packaged entries), then the
    Hub at ``$BENCHFLOW_HUB_URL`` for an ``org/name`` dataset it does not hold.
    ``hub:<url>`` names a BenchFlow Hub (:mod:`benchflow.hub.datasets`).
    """
    name, version = parse_dataset_spec(spec)
    hub = _hub_for(name, registry)
    if hub is not None:
        return _resolve_from_hub(spec, name, version, hub)
    try:
        entry, source = _find_entry(spec, name, version, registry)
    except DatasetResolutionError as exc:
        hub = _hub_for(name, registry, exc)
        if hub is None:
            raise
        return _resolve_from_hub(spec, name, version, hub)
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

    def _commit(task: object) -> str:
        # A registry may write HEAD (or nothing) for a dataset that follows
        # its repository's default branch.
        value = cast("dict[str, Any]", task).get("git_commit_id")
        text = "" if value is None else str(value).strip()
        if text.lower() in _UNPINNED_COMMITS:
            return "HEAD"
        if not _COMMIT_ID_RE.fullmatch(text.lower()):
            raise DatasetResolutionError(
                f"Dataset {spec!r} pins git_commit_id {text!r}, which is "
                f"neither a full commit id nor HEAD"
            )
        return text.lower()

    snapshots = {(_require(t, "git_url"), _commit(t)) for t in tasks}
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

    task_names: set[str] = set()
    task_digests: dict[str, str] = {}
    for t in tasks:
        base = posixpath.basename(_require(t, "path").rstrip("/"))
        if base in task_names:
            raise DatasetResolutionError(
                f"Dataset {spec!r} has two tasks named {base!r}"
            )
        task_names.add(base)
        digest = cast("dict[str, Any]", t).get("digest")
        if digest is None or digest == "":
            continue  # not pinned: run, but not checked
        if not isinstance(digest, str):
            raise DatasetResolutionError(
                f"Dataset {spec!r} has a malformed digest for {base!r}: {digest!r}"
            )
        task_digests[base] = digest

    repo = _github_org_repo(git_url)
    pinned = commit != "HEAD"
    # An unpinned entry asks for "HEAD" (the remote's default branch), not no
    # ref: with no ref an existing cache is reused as it stands, at whatever
    # commit an earlier run left it (another dataset's pin in the same repo).
    resolved = benchmark_repos.resolve_source_with_metadata(
        repo, path=parent or None, ref=commit
    )
    resolved_sha = resolved.provenance.get("resolved_sha")
    if pinned and resolved_sha != commit:
        raise DatasetResolutionError(
            f"Snapshot for {spec!r} resolved to {resolved_sha} but the "
            f"registry pins {commit}"
        )

    mismatches = []
    for base in sorted(task_names):
        candidate = resolved.path / base
        if not candidate.is_dir():
            mismatches.append(f"{base}: missing from snapshot")
            continue
        expected = task_digests.get(base)
        if expected is None:
            # A commit name authenticates HEAD, not the working tree. Keep
            # users' edits intact, but do not run them as the registry's task.
            if not isinstance(resolved_sha, str) or not _task_matches_commit(
                candidate, resolved_sha
            ):
                mismatches.append(
                    f"{base}: cached task differs from its commit snapshot"
                )
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
        task_names=task_names,
        task_digests=task_digests,
        provenance=resolved.provenance,
        registry=source,
    )


def _resolve_from_hub(spec: str, name: str, version: str, hub: str) -> ResolvedDataset:
    """Pull ``name@version`` from the Hub at ``hub`` (every archive and task digest checked)."""
    from benchflow.hub.client import HUB_REGISTRY_PREFIX, HubClient, HubError
    from benchflow.hub.datasets import pull_dataset

    try:
        pulled = pull_dataset(HubClient.from_env(hub), name, version)
    except HubError as exc:
        error = DatasetResolutionError(
            f"Dataset {spec!r} could not be pulled from the Hub at {hub}: {exc}"
        )
        damaged = "no longer hash" in str(exc)
        error.hint = exc.hint or (
            ("the Hub's stored copy is damaged, so tell its admin; " if damaged else "")
            + "`bench hub show "
            + spec
            + " --hub "
            + hub
            + "` shows what the Hub holds"
        )
        error.exit_code = 2 if exc.status == 404 else 1
        raise error from None
    return ResolvedDataset(
        name=name,
        version=version,
        bench_version=pulled.bench_version,
        tasks_dir=pulled.tasks_dir,
        task_names=set(pulled.task_digests),
        task_digests=dict(pulled.task_digests),
        provenance=pulled.provenance,
        registry=HUB_REGISTRY_PREFIX + hub,
    )
