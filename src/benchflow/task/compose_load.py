"""The strict reading of a task's Compose files before Compose runs.

Compose's resolved project (``docker compose config``) is what BenchFlow
checks and launches, but it no longer shows the files Compose read to make
it: an ``include``, an ``extends`` file, an ``env_file`` or ``label_file``
(their contents are in the project, their paths are gone), and every
included project's ``.env``. Those reads happen before anything is checked,
so they are checked here, on the staged copy, before Compose runs: each must
be a file of the copied environment folder (:mod:`benchflow.task.staging`),
since a file outside it (``/tmp/op.env``, ``~/.aws/credentials``) would put
its contents in the project with nothing left to show where they came from.

The reading is strict, so no file reads one way here and another way to
Compose's go-yaml: a mapping key must be text (a
``!!binary`` key is ``include`` to Compose and bytes to PyYAML), a value may
carry only the core schema's tags and Compose's ``!reset``/``!override``,
plain scalars resolve as go-yaml v3 resolves them, and the top-level and
service keys must be ones Compose documents. A key BenchFlow does not know
(a newer Compose's) and ``models`` (Compose runs the host's model runner with
the project's environment) are refused unless the operator
allowed host access. Every value a dependency-bearing
key can take is followed, from every mapping a ``<<`` merge brings in, and
from every folder Compose may resolve it from, so a difference in how merges
or bases resolve can only add a refusal.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.task.env import TaskEnvRequest

_CORE = "tag:yaml.org,2002:"
#: The tags a Compose file may carry: YAML's core schema (plus merge keys and
#: plain timestamps, which go-yaml resolves too) and Compose's own.
ALLOWED_TAGS = frozenset(
    {
        *(
            _CORE + name
            for name in (
                "str",
                "int",
                "float",
                "bool",
                "null",
                "map",
                "seq",
                "merge",
                "timestamp",
            )
        ),
        "!reset",
        "!override",
    }
)

#: Compose's top-level keys (``x-`` extensions aside) BenchFlow checks.
TOP_LEVEL_KEYS = frozenset(
    {
        "version",
        "name",
        "services",
        "networks",
        "volumes",
        "secrets",
        "configs",
        "include",
    }
)

#: The compose-spec service keys (Compose 2.40 and v5.6), less ``models``,
#: which Compose serves with this machine's model runner (``provider``,
#: which runs a program here, is refused on the resolved project).
SERVICE_KEYS = frozenset(
    [
        "annotations",
        "attach",
        "blkio_config",
        "build",
        "cap_add",
        "cap_drop",
        "cgroup",
        "cgroup_parent",
        "command",
        "configs",
        "container_name",
        "cpu_count",
        "cpu_percent",
        "cpu_period",
        "cpu_quota",
        "cpu_rt_period",
        "cpu_rt_runtime",
        "cpu_shares",
        "cpus",
        "cpuset",
        "credential_spec",
        "depends_on",
        "deploy",
        "develop",
        "device_cgroup_rules",
        "devices",
        "dns",
        "dns_opt",
        "dns_search",
        "domainname",
        "entrypoint",
        "env_file",
        "environment",
        "expose",
        "extends",
        "external_links",
        "extra_hosts",
        "gpus",
        "group_add",
        "healthcheck",
        "hostname",
        "image",
        "init",
        "ipc",
        "isolation",
        "label_file",
        "labels",
        "links",
        "logging",
        "mac_address",
        "mem_limit",
        "mem_reservation",
        "mem_swappiness",
        "memswap_limit",
        "network_mode",
        "networks",
        "oom_kill_disable",
        "oom_score_adj",
        "pid",
        "pids_limit",
        "platform",
        "ports",
        "post_start",
        "pre_stop",
        "privileged",
        "profiles",
        "provider",
        "pull_policy",
        "read_only",
        "restart",
        "runtime",
        "scale",
        "secrets",
        "security_opt",
        "shm_size",
        "stdin_open",
        "stop_grace_period",
        "stop_signal",
        "storage_opt",
        "sysctls",
        "tmpfs",
        "tty",
        "ulimits",
        "use_api_socket",
        "user",
        "userns_mode",
        "uts",
        "volumes",
        "volumes_from",
        "working_dir",
    ]
)

#: What clears a refusal of a file nothing else clears: rewriting it.
FIXED_REMEDY = (
    "Compose and BenchFlow must read a task's Compose files the same way: "
    "write keys as text and leave out YAML tags other than Compose's "
    "!reset and !override"
)


#: PyYAML's YAML 1.1 resolvers whose plain scalars go-yaml v3 (Compose's
#: parser, YAML 1.2) reads otherwise.
_YAML11_TAGS = frozenset(
    {_CORE + "bool", _CORE + "int", _CORE + "float", _CORE + "value"}
)

#: go-yaml v3's plain booleans and numbers (its ``resolve``: underscores are
#: dropped before a number is parsed, and a decimal it cannot read as an int,
#: such as ``089``, is a float). Ints come before floats, which match them too.
_YAML12_RESOLVERS = (
    (_CORE + "bool", r"^(?:true|True|TRUE|false|False|FALSE)$", "tTfF"),
    (
        _CORE + "int",
        r"""^[-+]?(?:0b[01_]*[01][01_]*|0o[0-7_]*[0-7][0-7_]*
        |0x[0-9a-fA-F_]*[0-9a-fA-F][0-9a-fA-F_]*|0[0-7_]*|[1-9][0-9_]*)$""",
        "-+0123456789",
    ),
    (
        _CORE + "float",
        r"""^(?:[-+]?(?:\.[0-9_]*[0-9][0-9_]*|[0-9][0-9_]*(?:\.[0-9_]*)?)
        (?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$""",
        "-+0123456789.",
    ),
)


def loader_class() -> Any:
    """A PyYAML loader whose plain scalars resolve as go-yaml v3's do."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.yaml_implicit_resolvers = {
        first: [r for r in resolvers if r[0] not in _YAML11_TAGS]
        for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    for tag, pattern, first in _YAML12_RESOLVERS:
        Loader.add_implicit_resolver(tag, re.compile(pattern, re.X), list(first))
    return Loader


def compose_nodes(text: str) -> list[Any] | None:
    """The YAML node tree of every document of *text*, ``None`` when it does
    not parse (tags resolved as go-yaml resolves them, nothing constructed)."""
    import yaml

    try:
        return [node for node in yaml.compose_all(text, Loader=loader_class())]
    except yaml.YAMLError:
        return None


def _scalar_text(node: Any) -> str | None:
    """The text of a plain-text scalar node, else ``None``."""
    import yaml

    if isinstance(node, yaml.ScalarNode) and node.tag == _CORE + "str":
        return node.value
    return None


def _remote(value: str) -> bool:
    return "://" in value or value.startswith(("git@", "github.com/"))


@dataclass
class LoadResult:
    """What the strict reading found."""

    refused: list[TaskEnvRequest] = field(default_factory=list)
    #: The files Compose reads that the reading followed, by path.
    files: list[Path] = field(default_factory=list)


@dataclass
class _Reader:
    """Reads the task's Compose files and the files they make Compose read
    (see the module doc), on the staged copy."""

    #: The staged environment folder: every file read must be in it.
    boundary: Path
    #: Whether the operator allowed host access (refusals it clears are
    #: dropped, and a file outside the folder is read where it is).
    host_access: bool
    #: Turns a file outside *boundary* into one the reading may follow
    #: (host access: the staged copy of the original's file), else None.
    outside: Any = None
    #: What clears a refusal the host-access opt-in clears.
    host_remedy: str = ""
    #: Maps a path of the copy back to the original's
    #: (:meth:`benchflow.task.staging.StagedEnvironment.original`).
    original: Any = None
    #: The VM's uploaded environment path, mapped to the local snapshot
    #: while preflighting Daytona without allocating it.
    path_alias: tuple[str, Path] | None = None
    result: LoadResult = field(default_factory=LoadResult)
    seen: set[tuple[Path, frozenset[Path]]] = field(default_factory=set)

    def refuse(self, label: str, what: str, clearable: bool) -> None:
        """Refuse; a *clearable* refusal is the host-access opt-in's to clear."""
        if clearable and self.host_access:
            return
        remedy = self.host_remedy if clearable else FIXED_REMEDY
        self.result.refused.append(
            TaskEnvRequest(label, "", None, f"{label} {what}", remedy=remedy)
        )

    def label(self, path: Path) -> str:
        try:
            rel = path.relative_to(self.boundary).as_posix()
        except ValueError:
            return str(path)
        return f"environment/{rel}"

    def inside(self, path: Path) -> bool:
        try:
            return Path(os.path.realpath(path)).is_relative_to(self.boundary)
        except (OSError, ValueError):
            return False

    # -- the node tree

    def check_tree(self, label: str, root: Any) -> bool:
        """Refuse a tag, a key or a merge value the reading cannot trust;
        whether the tree is clean."""
        import yaml

        clean = True
        stack = [root]
        visited: set[int] = set()
        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            if node.tag not in ALLOWED_TAGS:
                shown = node.tag.removeprefix(_CORE).removeprefix("tag:")
                self.refuse(
                    label,
                    f"uses the YAML tag {shown[:40]}, which Compose and "
                    "BenchFlow may read differently",
                    False,
                )
                clean = False
                continue
            if isinstance(node, yaml.MappingNode):
                for key, value in node.value:
                    if key.tag == _CORE + "merge":
                        sources = (
                            value.value
                            if isinstance(value, yaml.SequenceNode)
                            else [value]
                        )
                        if not all(isinstance(s, yaml.MappingNode) for s in sources):
                            self.refuse(
                                label,
                                "merges (<<) something that is not a mapping",
                                False,
                            )
                            clean = False
                    elif _scalar_text(key) is None:
                        shown = getattr(key, "value", "")
                        self.refuse(
                            label,
                            f"has a key that is not text ({str(shown)[:40]!r}, "
                            f"{key.tag.removeprefix(_CORE)}), which Compose "
                            "and BenchFlow may read differently",
                            False,
                        )
                        clean = False
                    stack += [key, value]
            elif isinstance(node, yaml.SequenceNode):
                stack += node.value
        return clean

    @staticmethod
    def entries(node: Any) -> Iterator[tuple[str, Any]]:
        """A mapping node's (key, value) pairs, every ``<<`` merge's included
        (all of them: precedence is go-yaml's to decide, the check takes
        every value)."""
        import yaml

        if not isinstance(node, yaml.MappingNode):
            return
        stack = [node]
        seen: set[int] = set()
        while stack:
            mapping = stack.pop()
            if id(mapping) in seen:
                continue
            seen.add(id(mapping))
            for key, value in mapping.value:
                if key.tag == _CORE + "merge":
                    sources = (
                        value.value if isinstance(value, yaml.SequenceNode) else [value]
                    )
                    stack += [s for s in sources if isinstance(s, yaml.MappingNode)]
                elif (text := _scalar_text(key)) is not None:
                    yield text, value

    def values(self, label: str, what: str, node: Any) -> list[str] | None:
        """The paths a dependency key names (a text or a list of texts, or
        mappings with ``path``), each refused unless plain text with no
        variable; ``None`` once one is refused. A mapping's every ``path``
        is taken, its own and each one a ``<<`` merge brings in: which one
        wins is go-yaml's to decide."""
        import yaml

        items = node.value if isinstance(node, yaml.SequenceNode) else [node]
        scalars: list[Any] = []
        for item in items:
            if not isinstance(item, yaml.MappingNode):
                scalars.append(item)
                continue
            for path in (v for k, v in self.entries(item) if k == "path"):
                scalars += path.value if isinstance(path, yaml.SequenceNode) else [path]
        out: list[str] = []
        for item in scalars:
            text = _scalar_text(item)
            if text is None:
                if isinstance(item, yaml.ScalarNode) and item.tag == _CORE + "null":
                    continue
                shown = getattr(item, "value", "")
                self.refuse(
                    label,
                    f"{what} {str(shown)[:40]!r}, which is not text to Compose, "
                    "so BenchFlow cannot check what Compose reads there",
                    False,
                )
                return None
            if "$" in text:
                self.refuse(
                    label,
                    f"{what} {text}, a file named through a variable",
                    True,
                )
                return None
            out.append(text)
        return out

    # -- files

    def join(self, base: Path, raw: str) -> Path:
        if self.path_alias is not None:
            remote, local = self.path_alias
            if raw == remote or raw.startswith(remote + "/"):
                return local / raw[len(remote) :].lstrip("/")
        if raw.startswith("~"):
            return Path(os.path.expanduser(raw))
        return base / raw

    def follow(self, label: str, what: str, path: Path) -> Path | None:
        """A file Compose reads: refused when it lands outside the staged
        folder (an absent one is Compose's to report); where to read it."""
        # A path that climbed out of the copy names the original's file.
        named = Path(self.original(str(path))) if self.original else path
        if not any(p.exists() or p.is_symlink() for p in (path, named)):
            return None
        if self.inside(path):
            return path
        if self.host_access:
            staged = self.outside(path) if self.outside else None
            return staged or path
        self.refuse(
            label,
            f"{what} {named}, outside the task's environment folder",
            True,
        )
        return None

    def compose_file(self, path: Path, bases: frozenset[Path]) -> None:
        import yaml

        key = (Path(os.path.realpath(path)), bases)
        if key in self.seen or not path.is_file():
            return
        self.seen.add(key)
        self.result.files.append(path)
        label = self.label(path)
        documents = compose_nodes(path.read_text(encoding="utf-8", errors="replace"))
        if documents is None:
            self.refuse(
                label,
                "is not a Compose file BenchFlow could read (it does not parse)",
                True,
            )
            return
        own = bases | {path.parent}
        for document in documents:
            if not self.check_tree(label, document):
                continue
            if not isinstance(document, yaml.MappingNode):
                if not (
                    isinstance(document, yaml.ScalarNode)
                    and document.tag == _CORE + "null"
                ):
                    self.refuse(
                        label,
                        "is not a Compose file BenchFlow could read (not a mapping)",
                        True,
                    )
                continue
            for top, value in self.entries(document):
                if top.startswith("x-"):
                    continue
                if top == "models":
                    self.refuse(
                        label,
                        "declares models, which Compose pulls and runs with "
                        "this machine's model runner",
                        True,
                    )
                elif top not in TOP_LEVEL_KEYS:
                    self.refuse(
                        label,
                        f"has the top-level key {top}, which BenchFlow does not check",
                        True,
                    )
                elif top == "include":
                    items = (
                        value.value if isinstance(value, yaml.SequenceNode) else [value]
                    )
                    for item in items:
                        self.include(label, item, bases)
                elif top == "services":
                    self.services(label, value, own)

    def services(self, label: str, node: Any, bases: frozenset[Path]) -> None:
        import yaml

        if isinstance(node, yaml.ScalarNode) and node.tag == _CORE + "null":
            return
        if not isinstance(node, yaml.MappingNode):
            self.refuse(label, "has services that are not a mapping", False)
            return
        for name, service in self.entries(node):
            if not isinstance(service, yaml.MappingNode):
                continue  # null, or Compose refuses it
            where = f"service {name}"
            for key, value in self.entries(service):
                if key.startswith("x-"):
                    continue
                if key == "models":
                    self.refuse(
                        label,
                        f"gives {where} models, which Compose runs with this "
                        "machine's model runner",
                        True,
                    )
                elif key not in SERVICE_KEYS:
                    self.refuse(
                        label,
                        f"gives {where} the key {key}, which BenchFlow does not check",
                        True,
                    )
                elif key in ("env_file", "label_file"):
                    what = (
                        "reads the env file"
                        if key == "env_file"
                        else ("reads the label file")
                    )
                    for raw in self.values(label, what, value) or []:
                        for base in bases:
                            self.follow(label, what, self.join(base, raw))
                elif key == "extends":
                    self.extends(label, value, bases)

    def extends(self, label: str, node: Any, bases: frozenset[Path]) -> None:
        import yaml

        if not isinstance(node, yaml.MappingNode):
            return  # a service of this file
        files = [v for k, v in self.entries(node) if k == "file"]
        for value in files:
            for raw in self.values(label, "extends", value) or []:
                if _remote(raw):
                    self.refuse(
                        label,
                        f"extends {raw}, a remote file BenchFlow cannot check",
                        True,
                    )
                    continue
                for base in bases:
                    target = self.follow(label, "extends", self.join(base, raw))
                    if target is not None:
                        self.compose_file(target, frozenset({target.parent}))

    def include(self, label: str, item: Any, bases: frozenset[Path]) -> None:
        import yaml

        if isinstance(item, yaml.MappingNode):
            entries = list(self.entries(item))
            paths = [
                p
                for k, v in entries
                if k == "path"
                for p in (self.values(label, "includes", v) or [])
            ]
            env_files = [
                p
                for k, v in entries
                if k == "env_file"
                for p in (self.values(label, "includes the env file", v) or [])
            ]
            projects = [
                p
                for k, v in entries
                if k == "project_directory"
                for p in (
                    self.values(label, "includes from the project folder", v) or []
                )
            ]
        else:
            paths = self.values(label, "includes", item) or []
            env_files, projects = [], []
        for raw in paths:
            if _remote(raw):
                # Compose fetches it (git, OCI) and runs it with this
                # machine's docker settings; BenchFlow never sees it.
                self.refuse(
                    label,
                    f"includes {raw}, a remote file BenchFlow cannot check",
                    True,
                )
                return
        folders = {self.join(base, raw) for base in bases for raw in projects}
        for folder in folders:
            if not self.inside(folder) and not self.host_access:
                self.refuse(
                    label,
                    f"includes from the project folder {folder}, outside the "
                    "task's environment folder",
                    True,
                )
                return
        targets = [self.join(base, raw) for base in bases for raw in paths]
        # The included project's folder: its project_directory, else its
        # first file's folder (each file's, to be safe).
        included = frozenset(folders | {target.parent for target in targets})
        for base in bases | included:
            for raw in env_files:
                self.follow(label, "includes the env file", self.join(base, raw))
        for folder in included:
            self.follow(label, "reads the env file", folder / ".env")
        for base in bases | included:
            for raw in paths:
                target = self.follow(label, "includes", self.join(base, raw))
                if target is not None:
                    self.compose_file(target, included)


def read_compose(
    environment_dir: Path,
    compose_file: Path,
    *,
    host_access: bool,
    host_remedy: str,
    outside: Any = None,
    original: Any = None,
    path_alias: tuple[str, Path] | None = None,
) -> LoadResult:
    """Read *compose_file* (in the staged *environment_dir*) and every file
    it makes Compose read, strictly (see the module doc)."""
    boundary = Path(os.path.realpath(environment_dir))
    reader = _Reader(boundary, host_access, outside, host_remedy, original, path_alias)
    reader.follow("environment/.env", "is", environment_dir / ".env")
    reader.compose_file(compose_file, frozenset({environment_dir}))
    return reader.result
