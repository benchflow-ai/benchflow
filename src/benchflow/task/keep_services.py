"""Services the agent leaves running for grading: ``x-keep-services``.

A task whose tests connect to a service the agent starts and leaves running
(a web API on port 5000, a package index on 8080) loses it when BenchFlow
kills the agent user's processes before grading. The task names the TCP
ports of such services, and the kills spare the agent user's processes that
listen on them (IPv4 or IPv6); every other process of the agent is killed as
before, and root's are never touched.

Where the key goes (the same key and places as the Rust ``bench``):

- task.md draft 2: ``[sandbox] x-keep-services``;
- task.toml: ``[metadata] x-keep-services``;
- BenchFlow task.md front matter: ``metadata: x-keep-services:``.

Its value is a list of at most 8 ``{ port, name }`` tables: ``port`` a whole
number from 1 to 65535, ``name`` an optional label for people (BenchFlow keeps
the port, not the name).

A task with no key of its own can still get one from BenchFlow's Terminal-Bench
2 list (``tb2_keep_services.json``, the Rust ``bench``'s list): only when the
content digest of the task folder (:func:`task_digest`) equals the digest
the list was read from, so an edited task, or another task with the same name,
gets none. The trial records where the names came from (:attr:`KeptServices.source`).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

#: The key, in every format.
KEY = "x-keep-services"
#: At most this many services.
MAX_SERVICES = 8
#: The longest ``name``, in characters.
NAME_MAX = 64

#: Where the names came from, in the words the trial's record uses.
SOURCE_TASK = "the task"
SOURCE_TB2 = "BenchFlow's Terminal-Bench 2 list"

# Split-layout files and folders included in the list's content digests.
_PUBLISHED_FILES = ("task.toml", "instruction.md", "README.md", "trajectory.json")
_PUBLISHED_DIRS = ("environment", "tests", "solution", "steps")
# Default ignores for a task without a .gitignore: a folder pattern (trailing
# slash) matches a folder at any depth, the others any path component.
_DEFAULT_IGNORED_DIRS = ("__pycache__",)
_DEFAULT_IGNORED_NAMES = ("*.pyc", ".DS_Store", "*.swp", "*.swo", "*~")
# The list's tasks are each a few kilobytes; a folder whose published files
# are bigger than this is no listed task, so it is not hashed at all.
_DIGEST_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class KeepService:
    """One service the agent leaves running."""

    port: int
    name: str | None = None

    def label(self) -> str:
        return f"{self.port} ({self.name})" if self.name else str(self.port)


@dataclass(frozen=True)
class KeptServices:
    """The services a trial keeps and where their names came from."""

    services: tuple[KeepService, ...]
    #: :data:`SOURCE_TASK` or :data:`SOURCE_TB2`.
    source: str
    #: The Terminal-Bench 2 task the list matched, when the list named them.
    listed_task: str | None = None

    @property
    def ports(self) -> tuple[int, ...]:
        return tuple(s.port for s in self.services)

    def describe_source(self) -> str:
        if self.listed_task:
            return f"{self.source} ({self.listed_task})"
        return self.source


def read(value: Any) -> tuple[list[KeepService], list[tuple[str, str]]]:
    """Read a ``x-keep-services`` value: the services, and every problem.

    A problem is ``(path, message)``: the path under the key (``""`` for the
    value itself, ``[1]`` for an entry, ``[1].port`` for a field). A service
    with a problem is left out; a value naming more than :data:`MAX_SERVICES`
    keeps none.
    """
    services: list[KeepService] = []
    problems: list[tuple[str, str]] = []
    if not isinstance(value, list):
        problems.append(("", "must be a list of { port, name } tables"))
        return services, problems
    if len(value) > MAX_SERVICES:
        problems.append(
            ("", f"names {len(value)} services; at most {MAX_SERVICES} are kept")
        )
    for n, item in enumerate(value):
        at = f"[{n}]"
        if not isinstance(item, dict):
            problems.append((at, "must be a { port, name } table"))
            continue
        service, problem = _entry(item)
        if problem is not None:
            field, message = problem
            problems.append((f"{at}.{field}" if field else at, message))
        elif service is not None and any(s.port == service.port for s in services):
            problems.append((f"{at}.port", f"port {service.port} is named twice"))
        elif service is not None:
            services.append(service)
    if len(value) > MAX_SERVICES:
        services = []
    return services, problems


def _entry(
    entry: dict[Any, Any],
) -> tuple[KeepService | None, tuple[str, str] | None]:
    unknown = sorted(
        str(k)
        for k in entry
        if k not in ("port", "name") and not (isinstance(k, str) and k.startswith("x-"))
    )
    if unknown:
        return None, (
            "",
            f"has an unknown key {unknown[0]!r}: a service holds port and name",
        )
    if "port" not in entry:
        return None, ("", "has no port")
    port = entry["port"]
    # bool is an int in Python; a TOML/YAML true is no port.
    if not (
        isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535
    ):
        return None, ("port", f"must be a whole number from 1 to 65535, not {port!r}")
    name = entry.get("name")
    if name is not None and not (
        isinstance(name, str)
        and name.strip()
        and len(name) <= NAME_MAX
        and not any(unicodedata.category(c) == "Cc" for c in name)
    ):
        return None, (
            "name",
            f"must be a label of 1 to {NAME_MAX} characters on one line",
        )
    return KeepService(port=port, name=name), None


def problems_message(problems: list[tuple[str, str]], where: str) -> str:
    """One line naming every problem, each at its place under ``where``."""
    return "; ".join(f"{where}{path} {message}" for path, message in problems)


def validate(value: Any, where: str = f"metadata.{KEY}") -> list[KeepService]:
    """The services ``value`` names; :class:`ValueError` naming every problem."""
    services, problems = read(value)
    if problems:
        raise ValueError(problems_message(problems, where))
    return services


@dataclass(frozen=True)
class _Listed:
    task: str
    digest: str
    services: tuple[KeepService, ...]


@cache
def tb2_list() -> tuple[_Listed, ...]:
    """BenchFlow's Terminal-Bench 2 list, from its data file."""
    text = (
        resources.files("benchflow.task")
        .joinpath("tb2_keep_services.json")
        .read_text(encoding="utf-8")
    )
    data = json.loads(text)
    return tuple(
        _Listed(
            task=t["task"],
            digest=t["digest"],
            services=tuple(validate(t["services"], f"{t['task']}: services")),
        )
        for t in data["tasks"]
    )


def listed(digest: str | None) -> KeptServices | None:
    """The list's services for the task with this digest, or None.

    Only the digest decides: a task edited in any byte, or another task with
    the same name, gets none.
    """
    if not digest:
        return None
    for entry in tb2_list():
        if entry.digest == digest:
            return KeptServices(entry.services, SOURCE_TB2, entry.task)
    return None


def _ignored_by_default(rel: str) -> bool:
    parts = rel.split("/")
    for i, part in enumerate(parts):
        if i < len(parts) - 1 and any(
            fnmatch.fnmatchcase(part, p) for p in _DEFAULT_IGNORED_DIRS
        ):
            return True
        if any(fnmatch.fnmatchcase(part, p) for p in _DEFAULT_IGNORED_NAMES):
            return True
    return False


def task_digest(task_dir: Path) -> str | None:
    """The split layout's package digest.

    The published files (:data:`_PUBLISHED_FILES` and everything under
    :data:`_PUBLISHED_DIRS`; a link to a file counts as the file, a linked
    folder is not entered) minus the default ignores, sorted by path,
    each adding ``{path}\\0{sha256 hex}\\n`` to an outer SHA-256. The same
    digest the Rust ``bench`` computes for the split layout.

    None when it cannot be the digest of a listed task: the folder has a
    ``.gitignore`` of its own (none of the list's tasks has one, and BenchFlow
    does not read gitignore patterns), its published files are over 64 MiB,
    or a file cannot be read. The listing stops where the files pass 64 MiB,
    so such a folder is neither read nor walked to the end.
    """
    if not task_dir.is_dir() or (task_dir / ".gitignore").exists():
        return None
    files = _published_files(task_dir)
    if files is None:
        return None
    outer = hashlib.sha256()
    try:
        for rel, path in sorted(files):
            outer.update(rel + b"\0")
            with path.open("rb") as f:
                outer.update(hashlib.file_digest(f, "sha256").hexdigest().encode())
            outer.update(b"\n")
    except OSError:
        return None
    return f"sha256:{outer.hexdigest()}"


def _published_files(task_dir: Path) -> list[tuple[bytes, Path]] | None:
    """The published files :func:`task_digest` hashes, as (path bytes, path),
    or None once their sizes add up to more than :data:`_DIGEST_MAX_BYTES`
    or one cannot be measured.

    The relative paths are cut from the walk's own strings (one
    ``Path.relative_to`` per file took most of the time on a large folder).
    """
    files: list[tuple[bytes, Path]] = []
    total = 0

    def fits(rel: str, path: Path) -> bool:
        nonlocal total
        if _ignored_by_default(rel):
            return True
        try:
            total += path.stat().st_size
        except OSError:
            return False
        files.append((os.fsencode(rel), path))
        return total <= _DIGEST_MAX_BYTES

    for single in _PUBLISHED_FILES:
        path = task_dir / single
        if path.is_file() and not fits(single, path):
            return None
    for folder in _PUBLISHED_DIRS:
        top = task_dir / folder
        if not top.is_dir():
            continue
        start = len(str(top))
        for dirpath, _dirnames, filenames in os.walk(top):
            # The folder below ``top`` that os.walk joined onto its string.
            below = dirpath[start:].lstrip("/")
            prefix = f"{folder}/{below}/" if below else f"{folder}/"
            for filename in filenames:
                path = Path(dirpath, filename)
                if path.is_file() and not fits(prefix + filename, path):
                    return None
    return files


def _is_split_layout(task_dir: Path) -> bool:
    return (task_dir / "task.toml").is_file() and not (task_dir / "task.md").exists()


def for_task(task: Any) -> KeptServices | None:
    """The services a trial of ``task`` keeps, and where they came from.

    The task's own key first; else, for a task.toml folder, BenchFlow's
    Terminal-Bench 2 list on a digest match; else None. The answer is kept on
    the task object, so the folder is hashed once.
    """
    own = getattr(task, "__dict__", None)
    if isinstance(own, dict) and _CACHED in own:
        return own[_CACHED]
    result = _for_task(task)
    if isinstance(own, dict):
        own[_CACHED] = result
    return result


# Kept in the task object's own attributes (never a stand-in's generated one).
_CACHED = "_benchflow_kept_services"
_WAITED = "_benchflow_kept_services_waited"
_TURN_END = "_benchflow_kept_services_turn_end"


def run_identity(task: Any, agent: str | None, sandbox_user: str | None) -> str | None:
    """The account a run's agent processes run as, for service retention.

    The oracle's solve.sh runs as the task's ``[agent].user`` when that is a
    string (su reads it as an account name, digits included), else as root,
    ``""`` here (``_run_oracle``), whatever the image's USER;
    ``--sandbox-user`` is not used by it (DT2-R9-2). Every other agent runs as
    ``sandbox_user`` (None: root, nothing is killed or kept).
    """
    if agent == "oracle":
        config = getattr(task, "config", None)
        declared = getattr(getattr(config, "agent", None), "user", None)
        return declared if isinstance(declared, str) and declared else ""
    return sandbox_user


async def prepare_identities(
    env: Any, task: Any, sandbox_user: str | None, *, inherit: bool = False
) -> None:
    """Resolve service policy before any agent runs, never from its spelling.

    With ``inherit`` (the oracle) ``sandbox_user`` is the account solve.sh runs
    as, an empty one root (:func:`run_identity`). Each
    account is the one its own launcher selects, measured in the sandbox
    before any agent code (DT2-R11-1): the oracle's through the same su that
    runs solve.sh; the verifier's by ``docker exec -u``'s rule on Docker,
    elsewhere through the backend's own exec as that user; an image default
    by its own ``id``. A ``--sandbox-user`` is a name
    (``setup_sandbox_user`` refuses any other spelling) that every launcher
    looks up the same way, so it is read from the image's passwd.
    """
    if inherit:
        sandbox_user = sandbox_user or ""
    elif not sandbox_user:
        return
    if for_task(task) is None:
        return
    from benchflow.sandbox._identities import (
        ID_PROBE,
        launched_identity,
        oracle_command,
        resolve,
        snapshot,
    )
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    config = getattr(task, "config", None)
    if separate_verifier_requested(config):
        return
    table = await snapshot(env)
    if inherit and sandbox_user:
        agent = await launched_identity(
            env,
            oracle_command(sandbox_user, ID_PROBE),
            key=("oracle", sandbox_user),
            what="[agent] user",
            # su from root, as the oracle runs it, whatever the image's USER.
            user="root",
        )
    else:
        # An oracle with no account of its own runs as root (_run_oracle), not
        # as the image's USER: setup gives the workspace and /logs/agent to
        # the sandbox user, so an image user such as nobody could use neither.
        agent = resolve(table, "0" if inherit else sandbox_user)
        if not inherit and agent.uid == 0 and sandbox_user != "root":
            # The kill before grading refuses UID 0, and check counts every
            # sandbox user but root as non-root: refuse it here, first.
            raise RuntimeError(
                f"{KEY}: --sandbox-user {sandbox_user!r} is UID 0 in this "
                "image; choose a non-root account, or --sandbox-user none to "
                "run the agent as root"
            )
    service = getattr(task.config.verifier, "service", "main") or "main"
    user = task.config.verifier.user
    verifier_table = await snapshot(env, service=service)
    if verifier_table.docker_rule or user is None or user == "":
        verifier = resolve(verifier_table, user)
    else:
        verifier = await launched_identity(
            env,
            ID_PROBE,
            key=("verifier", service, user),
            what="[verifier] user",
            user=user,
            service=service,
        )
    vars(task)["_benchflow_service_identities"] = (agent, verifier)
    # Refuse an unsupported combination now, before any agent code runs, not
    # at the kill before grading.
    kept_by_identity(task)


#: Backends whose exec selects a user the way ``docker exec -u`` does (a
#: decimal is a UID, whatever the image's passwd says) on every task. Daytona
#: does only for Docker-in-Docker tasks, and Modal, Apple Container and
#: AgentCore switch users with their own launchers (su selects by name).
DOCKER_EXEC_BACKENDS = frozenset({"docker", "remote-docker"})


def _docker_fixed_uid(selector: Any) -> int | None:
    """The UID ``docker exec -u`` runs ``selector`` as without reading any
    account: an integer, or a decimal before an optional group (``"0"``,
    ``"+0"``, ``"0:0"``). A name (``root`` included), an empty name and no
    selector depend on the image, and any selector on another backend may be
    an account name: None."""
    if type(selector) is int:
        return selector
    if not isinstance(selector, str) or not selector:
        return None
    from benchflow.sandbox._identities import Identities

    return Identities.number(selector.partition(":")[0])


def static_refusal(
    task: Any,
    agent: str | None,
    sandbox_user: str | None,
    sandbox: str | None = None,
) -> str | None:
    """The refusal of :func:`_refusal` when it is decided without a sandbox.

    The shared preflight (``bench tasks check`` and a run's plan gate) uses
    it, so check refuses what a run refuses (DT2-R9-1), and only that: it
    refuses a task only when every account the image could give would make
    setup refuse it, and never guesses an account from its spelling. That
    needs three facts the task and the run decide alone: the task keeps
    services; the agent runs as non-root (a ``--sandbox-user`` other than
    ``root`` is, in every run that gets past setup, since
    :func:`prepare_identities` refuses one that is UID 0); and the verifier
    is UID 0 by ``docker exec -u``'s own rule on a backend that uses it
    (:data:`DOCKER_EXEC_BACKENDS`). Everything else is decided at setup from
    the account the launcher selects: every oracle account (su selects
    ``toor`` or ``"1000"`` by name, DT2-R10-1, DT2-R11-1), a named verifier
    (``root`` included), and every verifier on a backend that may select it
    by name (``"0"`` is an account name under su, DT2-R12-1).
    """
    if for_task(task) is None:
        return None
    config = getattr(task, "config", None)
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    if separate_verifier_requested(config):
        return None
    if agent == "oracle" or sandbox_user in (None, "", "root"):
        return None
    if sandbox not in DOCKER_EXEC_BACKENDS:
        return None
    user = getattr(getattr(config, "verifier", None), "user", None)
    if _docker_fixed_uid(user) != 0:
        return None
    return _refusal(config)


def _refusal(config: Any) -> str | None:
    """Why a trial that would keep services beside a root verifier is refused.

    Kept services are agent code alive while the verifier runs. BenchFlow
    keeps them only beside a verifier in ``main``, the one service whose test
    command gets its env through a file only root can read; a verifier in
    another service gets it on its shell's command line, which a kept process
    reads through /proc whenever that service shares main's process namespace
    (Compose ``pid: "service:main"``, DT2-CONF-10); BenchFlow does not check
    which namespace a service has. And the verifier names root as ``root``,
    ``0`` or not at all: other spellings of UID 0 (``"0"``, ``"0:0"``, an alias
    such as ``toor``) are refused rather than matched (DT2-CONF-6/7).
    """
    verifier = getattr(config, "verifier", None)
    service = getattr(verifier, "service", None) or "main"
    if service != "main":
        return (
            f"{KEY}: the verifier runs as root in service {service!r}, where "
            "the agent processes kept alive for grading could read the "
            "command lines that carry its env; BenchFlow keeps services "
            'only beside a verifier in main: set [verifier] service = "main", '
            f"run the verifier as a non-root user, or remove {KEY}"
        )
    user = getattr(verifier, "user", None)
    if user in (None, "", "root") or (type(user) is int and user == 0):
        return None
    return (
        f"{KEY}: [verifier] user = {user!r} names root (UID 0) by another "
        'spelling; write user = "root" or leave it unset'
    )


def trial_keep(task: Any, sandbox_user: str | None) -> KeptServices | None:
    """The services a trial run as ``sandbox_user`` keeps, or None.

    None when there is nothing to spare them from or nobody to grade them:
    the agent runs as root (no kill), the verifier runs in a sandbox of its
    own (it cannot reach the agent's), or the verifier runs as a user other
    than root (the verifier's files are then closed only to root's).
    :class:`RuntimeError` when it would keep them beside a root verifier
    BenchFlow does not support (:func:`_refusal`).
    """
    if not sandbox_user:
        return None
    return kept_by_identity(task)


def kept_by_identity(task: Any) -> KeptServices | None:
    """:func:`trial_keep` for the identities :func:`prepare_identities` cached,
    whatever account name the caller holds (the oracle's may be the image's
    default user)."""
    if for_task(task) is None:
        return None
    config = getattr(task, "config", None)
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    if separate_verifier_requested(config):
        return None
    identities = vars(task).get("_benchflow_service_identities")
    if identities is None:
        raise RuntimeError(
            "service retention: resolve the agent and verifier UID/GID before "
            "starting the agent (prepare_identities)"
        )
    agent, verifier = identities
    # The measured UIDs decide first: a non-root verifier, however it is
    # spelled ("0" names UID 1001 under su), keeps nothing and is never
    # refused; root-only restrictions apply to a measured root (DT2-R12-1).
    if agent.uid == 0 or verifier.uid != 0:
        return None
    refusal = _refusal(config)
    if refusal is not None:
        raise RuntimeError(refusal)
    return for_task(task)


def mark_waited(task: Any) -> None:
    """Note that a kill of this trial already waited for the kept ports."""
    own = getattr(task, "__dict__", None)
    if isinstance(own, dict):
        own[_WAITED] = True


def note_turn_end(task: Any, outcome: Any) -> None:
    """Keep what the turn-end kill of this trial did (how many it killed, how
    long it waited, the pids it kept) for the record of the kill before
    grading, which otherwise holds only its own pass."""
    own = getattr(task, "__dict__", None)
    kept = getattr(outcome, "kept", None)
    if not isinstance(own, dict) or kept is None:
        return
    own[_TURN_END] = {
        "killed": int(getattr(outcome, "killed", 0)),
        "waited_sec": float(getattr(outcome, "waited_sec", 0.0)),
        "kept": [{"port": k.port, "pid": k.pid} for k in kept],
    }


def turn_end(task: Any) -> dict[str, Any] | None:
    """What the turn-end kill of this trial did (``note_turn_end``), or None."""
    own = getattr(task, "__dict__", None)
    if not isinstance(own, dict):
        return None
    value = own.get(_TURN_END)
    return value if isinstance(value, dict) else None


def waited(task: Any) -> bool:
    """Whether a kill of this trial already waited for the kept ports: only
    the first kill waits (a service that has not started by then never did)."""
    own = getattr(task, "__dict__", None)
    return isinstance(own, dict) and own.get(_WAITED) is True


def _for_task(task: Any) -> KeptServices | None:
    config = getattr(task, "config", None)
    metadata = getattr(config, "metadata", None)
    if isinstance(metadata, dict) and KEY in metadata:
        services, _problems = read(metadata[KEY])
        return KeptServices(tuple(services), SOURCE_TASK) if services else None
    task_dir = getattr(task, "task_dir", None)
    if task_dir is None:
        task_dir = getattr(getattr(task, "paths", None), "task_dir", None)
    if not isinstance(task_dir, (str, os.PathLike)):
        return None
    task_dir = Path(task_dir)
    if not _is_split_layout(task_dir):
        return None
    return listed(task_digest(task_dir))


def may_keep(task: Any, agent: str | None, sandbox_user: str | None) -> bool:
    """Whether a trial of ``task`` may keep services, decided without any
    identity, before the sandbox exists.

    False only where no measured account can change it: the task keeps
    nothing, the verifier runs in a sandbox of its own, nothing runs agent
    code (nop), or an agent other than the oracle runs as root (no
    ``sandbox_user``, no kill). Everything else (the oracle, whatever account
    it runs as; an agent run as ``sandbox_user`` beside a verifier of any
    spelling) is decided from the measured identities once
    :func:`prepare_identities` has them (:func:`kept_by_identity`).
    """
    if for_task(task) is None:
        return False
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    if separate_verifier_requested(getattr(task, "config", None)):
        return False
    if agent in (None, "nop"):
        return False
    return agent == "oracle" or bool(sandbox_user)


def runner_intent(task: Any, agent: str | None, sandbox_user: str | None) -> str | None:
    """What runs agent code of a task that may keep services, for the
    sandbox's start to decide the freezer: "oracle", "agent"
    for an agent that runs as a non-root user, or None (nop, a task that
    keeps nothing). Rollout setup and the SDK's ``Runtime.execute`` both ask
    here, before any identity exists, so it reads only :func:`may_keep`: a
    trial whose measured verifier turns out non-root asks for a freezer it
    does not use, never the other way round."""
    if not may_keep(task, agent, sandbox_user):
        return None
    return "oracle" if agent == "oracle" else "agent"


def assign_runner_intent(
    sandbox: Any,
    task: Any,
    agent: str | None,
    sandbox_user: str | None,
    *,
    verifier_only: bool = False,
) -> None:
    """Set ``sandbox.kept_service_runner`` before it starts (see
    :func:`runner_intent`); None for a verifier-only sandbox (recovery), which
    runs no agent code. A backend without the attribute is left alone."""
    if hasattr(sandbox, "kept_service_runner"):
        sandbox.kept_service_runner = (
            None if verifier_only else runner_intent(task, agent, sandbox_user)
        )


#: The sandboxes whose kills can hold kept services: root in ``main`` gets a
#: private cgroup namespace it may write (docker-compose-kept-freezer.yaml),
#: where the kills keep the services frozen until the release (K1-CONFIRM-1).
#: Daytona only for a Compose task, which runs in Docker inside the sandbox.
FREEZER_SANDBOXES = ("docker", "remote-docker")


def _declared_root(
    task: Any, user: Any, environment_dir: Path, keep: KeptServices
) -> bool:
    """Whether the verifier is surely root before the sandbox exists:
    ``[verifier] user = "root"``; or unset on a task of BenchFlow's
    Terminal-Bench 2 list (each measured root by the check after setup); or
    unset where the task's own Dockerfile is built and its final stage ends
    on ``USER root``. Anything else, a ``user = 0`` or an account that comes
    from a base image, a prebuilt ``docker_image`` or ``--base-image``
    included, is left to the check after setup."""
    if user == "root":
        return True
    if user not in (None, ""):
        return False
    if keep.source == SOURCE_TB2:
        return True
    sandbox = getattr(getattr(task, "config", None), "sandbox", None)
    if getattr(sandbox, "docker_image", None):
        return False
    dockerfile = environment_dir / "Dockerfile"
    if not dockerfile.is_file():
        return False
    from benchflow.sandbox.daytona_layers import _last_user

    last = _last_user(dockerfile.read_text(errors="replace"))
    return last is not None and last.partition(":")[0] in ("root", "0")


def preflight_daytona(
    task: Any,
    task_dir: Path,
    *,
    sandbox: str,
    agent: str | None,
    sandbox_user: str | None,
    base_image: str | None = None,
) -> None:
    """Refuse, before paying for setup, a native Daytona trial that the
    freezer check after setup (``Rollout._prove_kept_freezer`` through
    :func:`provider_refusal`) would refuse: an agent other than the oracle
    or nop, run as a non-root ``--sandbox-user`` beside a root verifier in
    ``main``, on a task that keeps services and has no Compose file. The
    oracle, nop and a root agent are never refused here; a run this cannot
    decide without measuring the sandbox's accounts is left to that check."""
    if sandbox != "daytona" or agent in (None, "nop", "oracle"):
        return
    if sandbox_user in (None, "", "root") or base_image:
        return
    keep = for_task(task)
    if keep is None:
        return
    from benchflow.sandbox._compose import compose_definition_path
    from benchflow.sandbox.protocol import KeptServiceFreezerUnavailable
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    config = getattr(task, "config", None)
    if separate_verifier_requested(config):
        return
    verifier = getattr(config, "verifier", None)
    if (getattr(verifier, "service", None) or "main") != "main":
        return
    environment_dir = task_dir / "environment"
    if not environment_dir.exists():
        environment_dir = task.paths.environment_dir
    if compose_definition_path(environment_dir) is not None:
        return
    if not _declared_root(task, getattr(verifier, "user", None), environment_dir, keep):
        return
    ports = ", ".join(str(p) for p in keep.ports)
    raise KeptServiceFreezerUnavailable(
        f"{task_dir.name} declares kept services on port {ports} "
        f"({KEY}, from {keep.describe_source()}); native Daytona cannot freeze "
        f"them for an agent run as non-root user {sandbox_user!r}. "
        "This trial is not scored: unsupported backend feature.\n"
        "next: run with --sandbox docker or --sandbox remote-docker, "
        "or supply a Compose environment on Daytona"
    )


def provider_refusal(
    task: Any,
    task_dir: Path,
    *,
    sandbox: str,
    keep: KeptServices | None,
    agent: str | None = None,
) -> str | None:
    """Why ``sandbox`` cannot run a trial of ``task`` that keeps ``keep``, or
    None.

    A kept service is held in a frozen cgroup between the kill and the
    release, and a sandbox that cannot give root one would fail the trial
    only after the agent's turn. So setup refuses it first, saying what to
    run instead. ``keep`` is what the trial keeps by its measured identities
    (:func:`kept_by_identity`), so this runs once :func:`prepare_identities`
    has measured them, before any agent code runs: before
    that no spelling tells a root verifier from a non-root one, which keeps
    nothing and is never refused (DT2-R12-1). None when the trial keeps
    nothing, and for the runners that may need no freezer (K1-R11-2):
    ``nop`` runs nothing, and the oracle is judged by the freezer probe
    before solve.sh, which refuses the trial where the sandbox has no
    freezer.
    """
    if keep is None or sandbox in FREEZER_SANDBOXES or agent in ("nop", "oracle"):
        return None
    environment_dir = task_dir / "environment"
    if not environment_dir.exists():
        environment_dir = task.paths.environment_dir
    from benchflow.sandbox._compose import compose_definition_path

    # Any of Compose's file names selects Daytona's DinD, which gives the
    # freezer as Docker does (K1-R12-1).
    if sandbox == "daytona" and compose_definition_path(environment_dir) is not None:
        return None
    ports = ", ".join(str(p) for p in keep.ports)
    return (
        f"{task_dir.name} keeps services on port {ports} ({KEY}, from "
        f"{keep.describe_source()}). BenchFlow holds them frozen in a cgroup of "
        f"root's from the kill after the agent's turn until the verifier "
        f"starts, which a {sandbox} sandbox cannot give; run it with "
        f"--sandbox docker or --sandbox remote-docker (or, on Daytona, as a "
        f"task with a Compose file in environment/)"
    )
