"""Cached image layers BenchFlow builds ``FROM`` a task image (Docker).

Installing an agent in every trial (apt-get, a Node tarball, ``npm
install``) costs tens of seconds per trial; the bytes it writes depend only on
the agent's pinned recipe and on the task image underneath. This module builds
those bytes once as an image layer ``FROM`` the task image and reuses it on a
matching key, as Rust's ``bench`` does (26.9 s the first time, then 2.0 s).

What goes into a layer's key: the layer kind, the agent and
its pinned version, the digest of its install recipe, the shared root install
environment, the uv version where uv installs it, the task image's content
digest, its libc (glibc or musl) and its architecture. The task image's
*content* digest (its filesystem layers and
runtime config, see :func:`image_content_digest`) rather than its image id:
teardown removes a built task image after each trial and the next trial
rebuilds it from BuildKit's cache with the same bytes but a new id (a new
``Created`` time), so an id would miss every trial while the bytes match.

What never goes into a layer: credentials, per-trial files and skills. The
build has an empty context, no build arguments, no secrets and no mounts; the
recipe is the only thing it runs, so whatever is per trial (the agent user,
credential files, skills, lockdown) stays in the trial (``install_agent``).

The model proxy gets a layer of its own (kind ``proxy``): the
LiteLLM venv of the locked, hashed requirement set, which every trial with
the proxy in the sandbox otherwise installs afresh (about 104 packages). Its
key holds the requirement file's digest, the extra requirements, the Python
and uv versions, the install command, and the task image's content digest,
libc and architecture (:func:`proxy_layer_key`). Its build's context is the
requirement file alone, and uv runs with no cache that outlives the build
step (no BuildKit cache mount, which any build on the daemon could reach). The
agent's layer, when the trial has both, is built on top of the proxy's.

Every layer image carries ``benchflow.layer=<kind>`` and its key as labels,
and every BuildKit cache record its build makes starts with :func:`marker`
(BuildKit records carry no labels, only their step's text). :meth:`prune`
lists layers by that label and removes one at a time by its own tag
(``benchflow-layer:<kind>-<key>``), only when the labels match and no other
tag or container holds the image, then the key's records: never a
daemon-wide prune, and never an image that merely inherited the labels.
Kept layers are bounded by count, by bytes, and by the daemon disk's free
space.

Labels and a tag prove nothing about who built an image: a task's Compose
file can build anything ``FROM`` a public task image and tag it
``benchflow-layer:proxy-<key>`` with matching labels, since every input of
the key is public. So a layer is reused only when its image id
is one this machine's BenchFlow built for that key (:func:`trusted_layer_ids`,
written from the build's own ``--iidfile``), and a trial's alias tags that
id, never the cached tag another build could move in between.

One build per key among cooperating processes: :meth:`_ensure`
takes the key's process lock and then its host lock (``_layer_lock``: a
``flock`` per daemon, kind and full key) and holds both across the hit
check, the build and the trial's tag; the prune try-locks each key it would
remove and holds the lock through the removal. A trial that cannot use the
host lock, or waits past its bound, installs fresh without touching the
shared layer (:class:`LayerLockFallback`); ``bench cache`` reads
:meth:`DockerLayers.inventory` and removes only what it selects.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from benchflow.sandbox import _base, _layer_lock, _pull_provenance
from benchflow.sandbox._base import ROOT_EXEC_PATH, with_system_path
from benchflow.sandbox._docker_subnets import DockerDaemon, daemon_key
from benchflow.sandbox._docker_sweep import OwnerState, current_process, owner_state
from benchflow.sandbox._trial_volumes import LABEL as TRIAL_VOLUME_LABEL
from benchflow.task.paths import SandboxPaths

logger = logging.getLogger("benchflow")

LAYER_LABEL = "benchflow.layer"
LAYER_KEY_LABEL = "benchflow.layer.key"
LAYER_BASE_LABEL = "benchflow.layer.base"
LAYER_LIBC_LABEL = "benchflow.layer.libc"
LAYER_AGENT_LABEL = "benchflow.layer.agent"
LAYER_AGENT_VERSION_LABEL = "benchflow.layer.agent-version"
LAYER_PROXY_GROUP_LABEL = "benchflow.layer.proxy-group"
# The task a verifier layer was first built for (``bench cache list``).
LAYER_VERIFIER_LABEL = "benchflow.layer.verifier"
AGENT = "agent"
# The model proxy's layer: the LiteLLM venv of a locked requirement set
# (benchflow.providers.proxy_requirements) on the task image. The agent's
# layer, when there is one, is built on top of it.
PROXY = "proxy"
# The grader's toolchain (uv, managed CPython, the packages test.sh asks
# for) baked under the task image, and in separate mode the verifier's own
# image (benchflow.sandbox.verifier_layers).
VERIFIER = "verifier"
# Cached layers are tagged ``benchflow-layer:<kind>-<key>``; a trial runs
# from its own alias tag (``alias``), which teardown's ``compose down --rmi
# all`` removes instead of the cached tag. The alias is
# ``bf__layer__<compose project>:<owner>``: its tag names the process that
# made it (:func:`alias_owner_tag`), so the prune can drop the alias of a
# trial whose process died before its teardown.
LAYER_REPOSITORY = "benchflow-layer"
ALIAS_PREFIX = "bf__layer__"
_BASE_TAG_REPOSITORY = "benchflow-layer-base"
# Bump when the Dockerfile :func:`agent_layer_dockerfile` or
# :func:`proxy_layer_dockerfile` writes changes: every key (and the proxy
# layer's stamp) changes with it, so no layer of the old shape is reused.
# 2: the proxy layer's install stops at its first failed step (U2b-R1-A-1).
LAYER_FORMAT = 2
# How many agent layers the prune after a build keeps (newest first), and
# the variable that changes it. Layers a trial of this process still uses are
# kept on top of these.
DEFAULT_AGENT_LAYER_KEEP = 64
AGENT_LAYER_KEEP_ENV = "BENCHFLOW_AGENT_LAYER_KEEP"
# A layer holds the whole task image under it, and teardown with ``delete``
# no longer frees those bytes while the layer is kept. So the prune also
# keeps the layers' sizes (``docker image inspect``'s ``Size``, task image
# included) under this many GB together, and removes more, oldest first,
# while the daemon's disk (when it is on this machine) has less free space
# than the second. ``0`` turns either bound off.
DEFAULT_AGENT_LAYER_MAX_GB = 20.0
AGENT_LAYER_MAX_GB_ENV = "BENCHFLOW_AGENT_LAYER_MAX_GB"
DEFAULT_AGENT_LAYER_MIN_FREE_GB = 10.0
AGENT_LAYER_MIN_FREE_GB_ENV = "BENCHFLOW_AGENT_LAYER_MIN_FREE_GB"
_GB = 1_000_000_000
# The same two bounds for proxy layers, counted apart from agent layers (a
# job with one proxy set has one proxy layer per task image). The free-disk
# floor is the daemon's, so both kinds share it.
DEFAULT_PROXY_LAYER_KEEP = 32
PROXY_LAYER_KEEP_ENV = "BENCHFLOW_PROXY_LAYER_KEEP"
DEFAULT_PROXY_LAYER_MAX_GB = 20.0
PROXY_LAYER_MAX_GB_ENV = "BENCHFLOW_PROXY_LAYER_MAX_GB"
# Every label a layer build sets: an image committed from a layer's container
# (a snapshot) blanks them, so it is never taken for a layer.
LAYER_LABELS = (
    LAYER_LABEL,
    LAYER_KEY_LABEL,
    LAYER_BASE_LABEL,
    LAYER_LIBC_LABEL,
    LAYER_AGENT_LABEL,
    LAYER_AGENT_VERSION_LABEL,
    LAYER_PROXY_GROUP_LABEL,
    LAYER_VERIFIER_LABEL,
)

# Where the proxy layer leaves its venv and the stamp of what it installed,
# and where its build reads the requirement file. The trial uses the venv
# only when the stamp names the install it would have run itself.
PROXY_LAYER_VENV = "/opt/runtime/litellm-proxy/venv"
PROXY_LAYER_STAMP = "/opt/runtime/layers/litellm-proxy.stamp"
PROXY_LAYER_REQUIREMENTS_DIR = "/run/runtime-proxy-layer"
# The proxy layer's build keeps no uv cache: uv runs with ``UV_NO_CACHE=1``
# (a temporary cache it removes before the step ends), and the build mounts
# no BuildKit cache (U2b-R1-B-1). A cache mount's id is one
# namespace for every build on the daemon, task Dockerfiles included, and
# uv trusts the hashes its cache records, so an untrusted task image built
# there could plant a changed litellm that a later proxy layer (whose proxy
# holds the provider key) would install. The build runs once per task image,
# so a cache across builds would save little.
_REQUIREMENTS_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.txt")

#: Backends that build no cached layers: the proxy and the agent
#: install in every trial, and config.json says ``unsupported``.
UNSUPPORTED_BACKENDS = frozenset({"modal", "apple-container", "agentcore"})

_KEY_RE = re.compile(r"[0-9a-f]{64}")
_KIND_RE = re.compile(r"[a-z][a-z0-9-]*")
# A Dockerfile ``USER`` value: a name or uid, optionally ``:group``.
_USER_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(?::[A-Za-z0-9_][A-Za-z0-9_.-]*)?")
_DOCKER_TIMEOUT_SEC = 60.0
# Seconds a trial waits for a key's lock beyond the recipe's build budget
# (another process may be building the key) before it installs fresh.
LOCK_WAIT_GRACE_SEC = 60.0
_PULL_TIMEOUT_SEC = 900.0
_LOG_TAIL_LINES = 40

# The task image's libc, read from its loader's file name without running
# anything but ``sh`` and ``[``. ``none`` (a static image) still gets a key.
_LIBC_PROBE = (
    "for f in /lib/ld-musl-*.so.1 /usr/lib/ld-musl-*.so.1; do "
    '[ -e "$f" ] && { echo musl; exit 0; }; done; '
    "for f in /lib*/ld-linux*.so.* /lib/*/ld-linux*.so.* "
    "/usr/lib*/ld-linux*.so.* /usr/lib/*/ld-linux*.so.*; do "
    '[ -e "$f" ] && { echo glibc; exit 0; }; done; '
    "echo none"
)
_LIBCS = ("glibc", "musl", "none")
# Tries, half a second apart, to remove a libc probe volume still in use.
_LIBC_PROBE_VOLUME_TRIES = 20

# Fields of an image's ``Config`` that change what a container of it runs or
# sees. ``Hostname``, ``Image`` and the attach flags describe how the image
# was made, not what it is, and differ between equal rebuilds.
_CONFIG_FIELDS = (
    "User",
    "Env",
    "Cmd",
    "Entrypoint",
    "WorkingDir",
    "Volumes",
    "ExposedPorts",
    "StopSignal",
    "Shell",
    "Healthcheck",
    "OnBuild",
    "ArgsEscaped",
    "Labels",
)
# Compose labels a built image with the project that built it, and every
# trial has its own project: per-trial metadata, not content.
_COMPOSE_LABEL_PREFIX = "com.docker.compose."


class LayerError(RuntimeError):
    """A layer could not be built or used; the trial installs fresh instead."""


class LayerSkipped(LayerError):
    """The image is ineligible for a layer; install fresh without a cache lookup."""


class LayerLockFallback(LayerError):
    """The layer's host lock could not be used or waited for in time.

    The trial installs fresh and never reads, builds or tags the shared
    layer: without the host lock, another process's prune could remove the
    layer under it. ``lock`` is ``process`` (the lock cannot be used
    here) or ``timeout`` (still held past the wait bound).
    """

    def __init__(self, lock: str, wait_sec: float, reason: str) -> None:
        super().__init__(reason)
        self.lock = lock
        self.wait_sec = wait_sec

    def record_fields(self) -> dict[str, Any]:
        return {"lock": self.lock, "lock_wait_sec": round(self.wait_sec, 3)}


@dataclass(frozen=True)
class AgentLayerRecipe:
    """What an agent's cached layer runs: its pinned install, nothing per trial.

    ``install_cmd`` is the shell the layer's one ``RUN`` step runs as root. It
    must not read credentials or write per-trial state: the build gets no
    environment from the host, so a recipe that needs a key fails here rather
    than baking it in. ``uv_version`` is set when the recipe installs with uv.
    """

    agent: str
    version: str
    install_cmd: str
    uv_version: str = ""
    timeout_sec: int = 900

    def __post_init__(self) -> None:
        if not self.agent.strip():
            raise ValueError("an agent layer needs the agent's name")
        version = self.version.strip()
        if not version or version.lower() in {"latest", "*", "next"}:
            raise ValueError(
                f"agent {self.agent!r}: a cached layer needs a pinned version, "
                f"not {self.version!r}"
            )
        if not self.install_cmd.strip():
            raise ValueError(f"agent {self.agent!r}: the install recipe is empty")

    @property
    def recipe_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.install_cmd.encode()).hexdigest()


@dataclass(frozen=True)
class ProxyLayerRecipe:
    """What the model proxy's cached layer installs: its locked set, nothing per trial.

    ``install_cmd`` is the shell the layer's one ``RUN`` step runs as root:
    it reads ``requirements_text`` at
    ``PROXY_LAYER_REQUIREMENTS_DIR/<requirements_name>`` (a bind mount of the
    build's context, so the file is not left in the image) and leaves the
    venv at :data:`PROXY_LAYER_VENV`. The proxy's config, its patches, the
    master key and the provider keys are written in the trial, never here.
    ``python`` and ``uv_version`` are the pins the command installs with.
    """

    group: str
    requirements_name: str
    requirements_text: str
    install_cmd: str
    python: str
    uv_version: str
    extra_requirements: tuple[str, ...] = ()
    timeout_sec: int = 900

    def __post_init__(self) -> None:
        if not _REQUIREMENTS_NAME_RE.fullmatch(self.requirements_name):
            raise ValueError(f"not a requirement file name: {self.requirements_name!r}")
        if not self.group.strip() or not self.requirements_text.strip():
            raise ValueError("a proxy layer needs its group and requirement file")
        if not self.install_cmd.strip():
            raise ValueError("the proxy layer's install command is empty")
        if not self.python.strip() or not self.uv_version.strip():
            raise ValueError("a proxy layer needs pinned Python and uv versions")

    @property
    def requirements_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.requirements_text.encode()).hexdigest()

    @property
    def extras(self) -> tuple[str, ...]:
        """The extra requirements as the key sees them: a set, in one order."""
        return tuple(sorted(set(self.extra_requirements)))

    @property
    def recipe_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.install_cmd.encode()).hexdigest()

    @property
    def stamp(self) -> str:
        """What the layer installed, without the task image: the trial compares it.

        The same requirement file, extras, Python, uv and install command give
        the same stamp on every task image; the key adds the image.
        """
        material = {
            "format": LAYER_FORMAT,
            "group": self.group,
            "requirements": self.requirements_sha256,
            "extras": list(self.extras),
            "python": self.python,
            "uv": self.uv_version,
            "recipe": self.recipe_digest,
        }
        blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class BaseImage:
    """The task image a layer is built ``FROM``, as the key sees it."""

    ref: str
    image_id: str
    digest: str
    arch: str
    libc: str
    user: str = ""
    rootfs: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayerResult:
    """One trial's layer: which, whether it was reused, and how long it took."""

    kind: str
    key: str
    tag: str
    image: str
    status: str  # "hit" or "miss"
    base: BaseImage
    seconds: float
    build_sec: float | None = None
    # The layer image itself, as the base of a layer built on top of it (the
    # agent's on the proxy's), so that build needs no second inspect.
    as_base: BaseImage | None = None
    # How the key was serialised (_layer_lock): ``host`` for a layer used
    # under the host lock, and the seconds spent waiting for it.
    lock: str = _layer_lock.HOST
    lock_wait_sec: float = 0.0
    daemon_id: str | None = None

    def record(self) -> dict[str, Any]:
        """The trial's ``layers`` entry for this layer (``config.json``)."""
        out: dict[str, Any] = {
            "status": self.status,
            "key": self.key,
            "image": self.tag,
            "base": self.base.digest,
            "libc": self.base.libc,
            "arch": self.base.arch,
            "seconds": round(self.seconds, 3),
            "lock": self.lock,
            "lock_wait_sec": round(self.lock_wait_sec, 3),
        }
        if self.build_sec is not None:
            out["build_sec"] = round(self.build_sec, 3)
        return out


def fresh_record(status: str, reason: str) -> dict[str, Any]:
    """A ``layers`` entry for a trial that installs without a layer.

    ``status`` is ``fresh`` (asked for, or the layer failed) or
    ``unsupported`` (this backend or task cannot use one), or ``skipped``
    (the image's environment requires fresh installation).
    """
    return {"status": status, "reason": reason}


def unsupported_records(
    backend: str, kinds: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """``layers`` entries for a backend in :data:`UNSUPPORTED_BACKENDS`.

    Empty for every other backend: Docker and remote Docker record their own
    (hit, miss, fresh); Daytona builds no layers yet and records none.
    """
    if backend not in UNSUPPORTED_BACKENDS:
        return {}
    reason = f"the {backend} sandbox builds no cached layers; installed in the trial"
    return {kind: fresh_record("unsupported", reason) for kind in kinds}


#: Forces every install of a trial to run in the trial, with no cached layer
#:. ``--fresh-install`` sets the run config's ``fresh_install``.
FRESH_INSTALL_ENV = "BENCHFLOW_FRESH_INSTALL"


def fresh_install_requested(flag: bool | None = None) -> bool:
    """Whether this trial installs fresh: the flag, else the environment."""
    if flag:
        return True
    value = os.environ.get(FRESH_INSTALL_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _trust_root() -> Path:
    """Where this machine records the image ids of the layers it built."""
    cache_home = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "benchflow" / "layers" / "built"


def trusted_layer_ids(kind: str, key: str) -> set[str]:
    """Image ids BenchFlow on this machine built as ``kind``'s layer for ``key``."""
    try:
        text = (_trust_root() / f"{_check_kind(kind)}-{key}").read_text()
    except OSError:
        return set()
    return {line.strip() for line in text.splitlines() if line.strip()}


def _record_built(kind: str, key: str, image_id: str) -> None:
    root = _trust_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        # One short line per build, appended: concurrent jobs on this machine
        # each add theirs, and none reads a half-written id.
        with open(root / f"{_check_kind(kind)}-{key}", "a", encoding="utf-8") as fh:
            fh.write(image_id + "\n")
    except OSError as exc:
        # Cache bookkeeping is optional. Keep refusing unrecorded hits,
        # while letting the caller install inside the trial (FF-install-C8).
        raise LayerError(
            f"could not write the {kind} layer trust record; "
            "installing in the trial instead"
        ) from exc


def _check_kind(kind: str) -> str:
    if not _KIND_RE.fullmatch(kind):
        raise ValueError(f"not a layer kind: {kind!r}")
    return kind


def marker(kind: str, key: str) -> str:
    """The no-op every layer build step's shell starts with.

    BuildKit keeps a cache record per step, without labels, described by the
    step's text; this ``:`` command names BenchFlow's label and the key there,
    so :meth:`DockerLayers.prune` can match a layer's records and no one
    else's. The ``;`` ends the key, so one key never matches a longer one.
    """
    _check_kind(kind)
    if not _KEY_RE.fullmatch(key):
        raise ValueError(f"not a layer key: {key!r}")
    return f": {LAYER_LABEL}={kind} key={key};"


def _marker_filter(kind: str, key: str) -> str:
    """The ``docker builder prune`` filter that matches :func:`marker` alone.

    BuildKit reads ``description~=`` as a regular expression over the
    record's description; ``[.]`` keeps the label's dot literal.
    """
    text = marker(kind, key).removeprefix(": ").replace(".", "[.]")
    return f'description~="{text}"'


def layer_tag(kind: str, key: str) -> str:
    _check_kind(kind)
    if not _KEY_RE.fullmatch(key):
        raise ValueError(f"not a layer key: {key!r}")
    return f"{LAYER_REPOSITORY}:{kind}-{key}"


def image_content_digest(info: Mapping[str, Any]) -> str:
    """The digest of what an image is, from ``docker image inspect``.

    Its filesystem layers (``RootFS``: content digests of each layer's
    uncompressed tar), its platform, and the config fields a container of it
    runs with. Two builds of the same Dockerfile from the same cache get the
    same digest; any change to a file, the environment, the user or the
    platform gives another.
    """
    config = dict(info.get("Config") or {})
    picked: dict[str, Any] = {name: config.get(name) for name in _CONFIG_FIELDS}
    labels = picked.get("Labels") or {}
    picked["Labels"] = {
        k: v
        for k, v in sorted(labels.items())
        if not k.startswith(_COMPOSE_LABEL_PREFIX)
    }
    rootfs = info.get("RootFS") or {}
    material = {
        "rootfs": {
            "type": rootfs.get("Type"),
            "layers": list(rootfs.get("Layers") or []),
        },
        "os": info.get("Os"),
        "arch": info.get("Architecture"),
        "variant": info.get("Variant") or "",
        "config": picked,
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def _agent_layer_shell_env_policy() -> dict[str, Any]:
    # RUN starts its dynamically linked shell before our shell-text wrapper
    # can unset anything. Images setting these names must install fresh, even
    # for empty values (some loader switches act on presence alone).
    return {
        "flags": _base.system_shell_env_flags(),
        "unset": _base.SYSTEM_SHELL_UNSET_VARS,
        "loader_prefix": "LD_",
    }


def _check_agent_layer_env(config: Mapping[str, Any]) -> None:
    policy = _agent_layer_shell_env_policy()
    cleared = set(policy["unset"]) | {
        flag.partition("=")[0] for flag in policy["flags"] if flag != "-e"
    }
    names = sorted(
        {
            entry.partition("=")[0]
            for entry in config.get("Env") or ()
            if entry.partition("=")[0] in cleared
            or entry.startswith(policy["loader_prefix"])
        }
    )
    if names:
        raise LayerSkipped(f"image env sets {', '.join(names)}")


def agent_layer_key(recipe: AgentLayerRecipe, base: BaseImage) -> str:
    """The key an agent layer is cached under.

    Changes with the agent, its version, its recipe, the uv version, the task
    image's content, its libc and its architecture, and with
    :data:`LAYER_FORMAT`.
    """
    material = {
        "format": LAYER_FORMAT,
        "kind": AGENT,
        # A layer made before this check must never be reused. Include the
        # check and its destinations so a changed mount layout also misses.
        "mount_check": _agent_layer_mount_check(),
        "agent": recipe.agent,
        "version": recipe.version,
        "recipe": recipe.recipe_digest,
        # Fresh installation precedes user setup/runtime-kit capture. Share
        # its root route's environment, and invalidate when that route changes.
        "install_environment": with_system_path("", ROOT_EXEC_PATH),
        "shell_env_policy": _agent_layer_shell_env_policy(),
        "uv": recipe.uv_version,
        "base": base.digest,
        "libc": base.libc,
        "arch": base.arch,
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _agent_layer_mount_check() -> str:
    """Require the default bind mounts to hide no image contents.

    Check after installation, not just before it: recipes and pre-installed
    dependencies can reach these trees through any number of symlinks. Empty
    real directories (or missing paths) are the only supported destinations;
    symlinks in their ancestors could redirect the mounts themselves.
    """
    script = r"""
for mount in "$@"; do
    parent="$mount"
    while [ "$parent" != / ]; do
        if [ -L "$parent" ] || { [ -e "$parent" ] && [ ! -d "$parent" ]; }; then
            echo "agent layer: runtime log mount is not a real directory: $parent" >&2
            exit 1
        fi
        parent="${parent%/*}"
        [ -n "$parent" ] || parent=/
    done
    for entry in "$mount"/* "$mount"/.[!.]* "$mount"/..?*; do
        if [ -e "$entry" ] || [ -L "$entry" ]; then
            echo "agent layer: runtime log mount would hide image contents: $mount" >&2
            exit 1
        fi
    done
done
"""
    return shlex.join(
        (
            "/bin/sh",
            "-c",
            script,
            "benchflow-layer-mounts",
            str(SandboxPaths.agent_dir),
            str(SandboxPaths.verifier_dir),
            str(SandboxPaths.artifacts_dir),
        )
    )


def agent_layer_dockerfile(
    recipe: AgentLayerRecipe, base_from: str, key: str, *, user: str = ""
) -> str:
    """The Dockerfile of an agent layer: the task image plus one ``RUN``.

    The step runs as root (as ``install_agent`` does) through ``/bin/sh -c``
    in exec form, so neither the task image's ``SHELL`` nor Dockerfile
    variable expansion touches the recipe. Its environment uses the fresh
    install's root route, before user setup and runtime-kit capture: image
    PATH and Python/loader inputs cannot select a different interpreter.
    A task image with a ``USER`` gets it back after the step, so the layer
    runs as the task image does.
    """
    if user and not _USER_RE.fullmatch(user):
        raise LayerError(f"task image user {user!r} cannot be restored in a layer")
    # A separate shell preserves the recipe's `exit` and `set -e` semantics:
    # neither a successful early exit nor an AND-list can skip the check or
    # turn off errexit inside the installer. No image is published on failure.
    install = shlex.join(
        ("/bin/sh", "-c", with_system_path(recipe.install_cmd, ROOT_EXEC_PATH))
    )
    step = json.dumps(
        [
            "/bin/sh",
            "-c",
            f"{marker(AGENT, key)} {install} && {_agent_layer_mount_check()}",
        ]
    )
    lines = [f"FROM {base_from}"]
    if user:
        lines.append("USER root")
    lines.append(f"RUN {step}")
    if user:
        lines.append(f"USER {user}")
    return "\n".join(lines) + "\n"


def proxy_layer_key(recipe: ProxyLayerRecipe, base: BaseImage) -> str:
    """The key a proxy layer is cached under.

    Changes with the requirement file's digest, the extra requirements, the
    Python, the uv version, the install command, the task image's content,
    its libc and its architecture, and with :data:`LAYER_FORMAT`. Two task
    images with the same requirements get two keys, so two layers: a layer
    is never reused on a task image it was not built from.
    """
    material = {
        "format": LAYER_FORMAT,
        "kind": PROXY,
        "group": recipe.group,
        "requirements": recipe.requirements_sha256,
        "extras": list(recipe.extras),
        "python": recipe.python,
        "uv": recipe.uv_version,
        "recipe": recipe.recipe_digest,
        "base": base.digest,
        "libc": base.libc,
        "arch": base.arch,
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def proxy_layer_dockerfile(
    recipe: ProxyLayerRecipe,
    base_from: str,
    key: str,
    *,
    user: str = "",
) -> str:
    """The Dockerfile of a proxy layer: the task image plus one ``RUN``.

    As for an agent layer, the step runs as root through ``/bin/sh -c`` in
    exec form and the task image's ``USER`` comes back after it. The step
    mounts the requirement file from the build's context (nothing else is
    in it) and runs uv with ``UV_NO_CACHE=1``, with no cache mount that
    another build on the daemon could reach, then writes
    :attr:`ProxyLayerRecipe.stamp` to :data:`PROXY_LAYER_STAMP`.
    """
    if user and not _USER_RE.fullmatch(user):
        raise LayerError(f"task image user {user!r} cannot be restored in a layer")
    stamp_dir = PROXY_LAYER_STAMP.rpartition("/")[0]
    # The install runs at the script's top level under ``set -eu``, and the
    # stamp is written on the lines after it: a failed step stops the build
    # before the stamp. (Inside ``( ... ) && ...`` sh ignores ``-e``, and the
    # install's status would be its last command's, ``import litellm``.)
    script = "\n".join(
        [
            f"{marker(PROXY, key)} set -eu",
            "export UV_NO_CACHE=1",
            recipe.install_cmd,
            f"mkdir -p {stamp_dir}",
            f"printf '%s\\n' {recipe.stamp} > {PROXY_LAYER_STAMP}",
            f"chmod 0755 {stamp_dir}",
            f"chmod 0644 {PROXY_LAYER_STAMP}",
        ]
    )
    mounts = (
        f"--mount=type=bind,source={recipe.requirements_name},"
        f"target={PROXY_LAYER_REQUIREMENTS_DIR}/{recipe.requirements_name}"
    )
    lines = [f"FROM {base_from}"]
    if user:
        lines.append("USER root")
    lines.append(f"RUN {mounts} {json.dumps(['/bin/sh', '-c', script])}")
    if user:
        lines.append(f"USER {user}")
    return "\n".join(lines) + "\n"


def _env_count(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not a whole number; keeping %d", name, raw, default)
        return default
    return max(value, 0)


def agent_layer_keep() -> int:
    """How many agent layers to keep (:data:`AGENT_LAYER_KEEP_ENV`)."""
    return _env_count(AGENT_LAYER_KEEP_ENV, DEFAULT_AGENT_LAYER_KEEP)


def proxy_layer_keep() -> int:
    """How many proxy layers to keep (:data:`PROXY_LAYER_KEEP_ENV`)."""
    return _env_count(PROXY_LAYER_KEEP_ENV, DEFAULT_PROXY_LAYER_KEEP)


def _env_gb(name: str, default: float) -> int | None:
    """Bytes from a GB variable (``name``), or None when it is ``0`` (off)."""
    raw = os.environ.get(name, "").strip()
    value = default
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = -1.0
        if not value >= 0 or value == float("inf"):
            logger.warning("%s=%r is not a number of GB; using %g", name, raw, default)
            value = default
    return int(value * _GB) if value > 0 else None


def agent_layer_max_bytes() -> int | None:
    """The bytes agent layers may take together (:data:`AGENT_LAYER_MAX_GB_ENV`)."""
    return _env_gb(AGENT_LAYER_MAX_GB_ENV, DEFAULT_AGENT_LAYER_MAX_GB)


def proxy_layer_max_bytes() -> int | None:
    """The bytes proxy layers may take together (:data:`PROXY_LAYER_MAX_GB_ENV`)."""
    return _env_gb(PROXY_LAYER_MAX_GB_ENV, DEFAULT_PROXY_LAYER_MAX_GB)


def agent_layer_min_free_bytes() -> int | None:
    """The free disk the prune keeps (:data:`AGENT_LAYER_MIN_FREE_GB_ENV`)."""
    return _env_gb(AGENT_LAYER_MIN_FREE_GB_ENV, DEFAULT_AGENT_LAYER_MIN_FREE_GB)


def _parse_created(text: str) -> datetime:
    """``docker image ls``'s ``CreatedAt`` (``2026-10-16 02:35:52 +0000 UTC``)."""
    head = " ".join(text.split()[:3])
    try:
        return datetime.strptime(head, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def _description_keys(du_text: str, kind: str) -> set[str]:
    """Keys of ``kind``'s cache records in ``docker builder du --verbose``."""
    pattern = re.compile(re.escape(f"{LAYER_LABEL}={kind} key=") + r"([0-9a-f]{64});")
    return set(pattern.findall(du_text))


# What `docker image inspect` prints for a reference the daemon does not have.
_NO_SUCH_IMAGE_RE = re.compile(r"No such (image|object)", re.IGNORECASE)
# `docker info`'s DriverStatus entry when the daemon uses the containerd image store.
_CONTAINERD_STORE = ["driver-type", "io.containerd.snapshotter.v1"]


def last_tag_time(info: Mapping[str, Any]) -> str:
    """``docker image inspect``'s Metadata.LastTagTime, or "" when unset.

    The containerd image store sets a time on every pull and tag of a name,
    and inspect shows the newest of all the image's names, so a reference
    removed and pulled again shows a newer time. The classic store keeps one
    time per image: ``docker tag`` sets it (a layer build's pin of the task
    image too), a pull does not, and untagging does not clear it (checked on
    Docker 29.1.3). Only :meth:`DockerLayers.tag_times_per_name` daemons
    have a time worth recording.
    """
    metadata = info.get("Metadata")
    value = metadata.get("LastTagTime") if isinstance(metadata, Mapping) else None
    if not isinstance(value, str) or not value or value.startswith("0001-01-01"):
        return ""
    return value


def _other_tags(info: Mapping[str, Any], tag: str) -> list[str]:
    """The image's tags besides ``tag`` (``docker image inspect``)."""
    return [
        t
        for t in info.get("RepoTags") or []
        if isinstance(t, str) and t and t != tag and t != "<none>:<none>"
    ]


def _host_hash(host: str) -> str:
    return hashlib.sha256(host.encode()).hexdigest()[:12]


# <host hash>-<pid>-<start ticks>-<boot id>: what ``benchflow.process`` holds
# (_docker_sweep.ProcessId), in the characters an image tag allows.
_ALIAS_TAG_RE = re.compile(r"([0-9a-f]{12})-([1-9][0-9]*)-([0-9]*)-([A-Za-z0-9-]*)")


def alias_owner_tag() -> str:
    """The tag of a trial's alias: which process made it.

    The host is hashed (an image tag holds no ``:`` and at most 128
    characters); boot id and start time are empty off Linux, as in the
    ``benchflow.process`` label of containers.
    """
    me = current_process()
    boot = re.sub(r"[^A-Za-z0-9-]", "", me.boot)
    start = re.sub(r"[^0-9]", "", me.start)
    return f"{_host_hash(me.host)}-{me.pid}-{start}-{boot}"


def alias_owner_state(tag: str) -> OwnerState:
    """Whether the process an alias's tag names is this one, alive, gone, or
    unknown (another host, or a tag in no known form, ``latest`` included)."""
    match = _ALIAS_TAG_RE.fullmatch(tag)
    if match is None:
        return "unknown"
    host_hash, pid, start, boot = match.groups()
    me = current_process()
    if host_hash != _host_hash(me.host):
        return "unknown"
    return owner_state(f"{me.host}:{pid}:{boot}:{start}")


def pin_tag(kind: str, key: str) -> str:
    """The temporary tag a layer build pins its task image with.

    ``benchflow-layer-base/<kind>-<key>:<owner>``: the full key and kind in
    the repository, the making process (:func:`alias_owner_tag`) in the tag.
    Pins made before this form (``benchflow-layer-base:<key16>-<random>``)
    name no owner and are never removed by BenchFlow.
    """
    _check_kind(kind)
    if not _KEY_RE.fullmatch(key):
        raise ValueError(f"not a layer key: {key!r}")
    return f"{_BASE_TAG_REPOSITORY}/{kind}-{key}:{alias_owner_tag()}"


_PIN_REPOSITORY_RE = re.compile(
    re.escape(_BASE_TAG_REPOSITORY) + r"/([a-z][a-z0-9-]*?)-([0-9a-f]{64})"
)


def parse_pin(repository: str, tag: str) -> tuple[str, str, str] | None:
    """(kind, key, owner tag) of a new-form pin, or None (legacy or foreign)."""
    match = _PIN_REPOSITORY_RE.fullmatch(repository)
    if match is None or not _ALIAS_TAG_RE.fullmatch(tag):
        return None
    return match.group(1), match.group(2), tag


def _image_size(info: Mapping[str, Any]) -> int:
    try:
        return max(int(info.get("Size") or 0), 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class PruneReport:
    removed: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    in_use: list[str] = field(default_factory=list)
    records_pruned: list[str] = field(default_factory=list)
    # Trial tags (aliases) of processes that are gone, untagged by the prune.
    aliases_removed: list[str] = field(default_factory=list)
    # Keys left alone because their lock (_layer_lock) was held by a trial
    # building or using the key, or could not be used here: key -> why.
    skipped: dict[str, str] = field(default_factory=dict)
    # Sizes (``docker image inspect``) of the layers removed and of those left.
    bytes_removed: int = 0
    bytes_left: int = 0


@dataclass
class _Run:
    stdout: str
    stderr: str
    code: int

    @property
    def text(self) -> str:
        return (self.stderr or self.stdout).strip()


class DockerLayers:
    """Build, reuse and prune cached layers on one Docker daemon.

    ``env`` is the environment of every ``docker`` call (``None`` inherits
    this process's): remote Docker passes one pointed at its host. Providers
    with a transport shim supply their shared ``daemon`` identity and
    ``redact_diagnostic`` before any consumer sees a
    diagnostic. Successful stdout remains unmodified machine data.
    """

    # Keys a trial of this process runs from: never pruned while held.
    _claims: ClassVar[dict[tuple[str, str], int]] = {}
    # Queried daemon and task digest -> libc.
    _libc_cache: ClassVar[dict[tuple[str, str], str]] = {}

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        *,
        redact_diagnostic: Callable[[str], str] | None = None,
        daemon: DockerDaemon | None = None,
    ) -> None:
        self._daemon = daemon if daemon is not None else daemon_key(env)
        self._env = dict(self._daemon.env)
        redactor = redact_diagnostic or self._daemon.redact
        self._redact_diagnostic = redactor or (lambda text: text)
        self._remote_diagnostics = redactor is not None

    async def _docker(
        self,
        *args: str,
        stdin: bytes | None = None,
        timeout: float = _DOCKER_TIMEOUT_SEC,
    ) -> _Run:
        # Exceptions are diagnostics too: spawn, communicate and cleanup errors
        # must cross the same boundary as stderr before any fallback records them.
        try:
            return await self._docker_run(*args, stdin=stdin, timeout=timeout)
        except Exception as exc:
            if not self._remote_diagnostics:
                raise
            raise LayerError(self._redact_diagnostic(str(exc))) from None

    async def _docker_run(
        self,
        *args: str,
        stdin: bytes | None = None,
        timeout: float = _DOCKER_TIMEOUT_SEC,
    ) -> _Run:
        cancelled_during_cleanup = False

        async def settle(task: asyncio.Task[Any]) -> Any:
            nonlocal cancelled_during_cleanup
            # A second cancel must not let the key lock go while the owned
            # Docker/buildx process is still running or holding its pipes.
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled_during_cleanup = True
                    if task.done():
                        return task.result()

        async def stop(proc: Any) -> None:
            with contextlib.suppress(ProcessLookupError):
                if os.name == "posix":
                    # Docker can launch buildx: stop the owned group, not
                    # just its launcher, before releasing the build lock.
                    os.killpg(proc.pid, signal.SIGKILL)
                else:  # pragma: no cover - Windows
                    proc.kill()
            await proc.communicate()

        spawning = asyncio.create_task(
            asyncio.create_subprocess_exec(
                "docker",
                *args,
                stdin=asyncio.subprocess.PIPE
                if stdin is not None
                else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
                start_new_session=os.name == "posix",
            )
        )
        try:
            proc = await asyncio.shield(spawning)
        except asyncio.CancelledError:
            # Cancellation may arrive after fork but before asyncio hands
            # back the process. Finish acquiring that handle, then reap it.
            proc = await settle(spawning)
            await settle(asyncio.create_task(stop(proc)))
            raise
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
        except (asyncio.CancelledError, TimeoutError) as error:
            await settle(asyncio.create_task(stop(proc)))
            if isinstance(error, asyncio.CancelledError):
                raise
            if cancelled_during_cleanup:
                # Timeout started cleanup, but a later caller cancellation
                # still owns the outcome. Do not trigger fresh installation.
                raise asyncio.CancelledError from None
            raise LayerError(
                f"docker {args[0]} timed out after {timeout:.0f}s"
            ) from None
        stdout = (out or b"").decode(errors="replace")
        stderr = (err or b"").decode(errors="replace")
        # Every consumer gets diagnostics already redacted, before any tail
        # slicing, log or layer record. Successful stdout remains machine data.
        return _Run(
            stdout=self._redact_diagnostic(stdout) if proc.returncode else stdout,
            stderr=self._redact_diagnostic(stderr),
            code=proc.returncode or 0,
        )

    # -- the task image -------------------------------------------------

    async def _inspect(self, ref: str) -> dict[str, Any] | None:
        return (await self._inspect_or_absent(ref))[0]

    async def _inspect_or_absent(self, ref: str) -> tuple[dict[str, Any] | None, bool]:
        """(``ref``'s inspect, whether Docker said there is no such image).

        A failed inspect for any other reason (a busy daemon's 500, a lost
        connection) is not "absent": a pull after it may find the user's
        image there, and must not record it as BenchFlow's.
        """
        run = await self._docker("image", "inspect", ref)
        if run.code != 0:
            return None, bool(_NO_SUCH_IMAGE_RE.search(run.text))
        try:
            items = json.loads(run.stdout)
        except json.JSONDecodeError:
            return None, False
        return (items[0] if isinstance(items, list) and items else None), False

    async def absent(self, ref: str) -> bool:
        """Whether Docker says it has no ``ref`` (False when it cannot tell)."""
        return (await self._inspect_or_absent(ref))[1]

    async def image_id(self, ref: str) -> str | None:
        """The id ``ref`` names on the daemon, or None when it is absent."""
        info = await self._inspect(ref)
        found = str((info or {}).get("Id") or "")
        return found or None

    async def tag_times_per_name(self) -> bool:
        """Whether the daemon gives each pulled or tagged name its own time.

        True only for the containerd image store (``docker info``'s
        DriverStatus says ``driver-type io.containerd.snapshotter.v1``). On
        the classic store a pull leaves the image's LastTagTime alone and a
        layer build's ``docker tag`` pin sets it, so a time read there
        cannot tell BenchFlow's pull from a later one of the user's; a daemon
        that cannot say counts as classic.
        """
        try:
            run = await self._docker("info", "--format", "{{json .DriverStatus}}")
        except LayerError:
            return False
        if run.code != 0:
            return False
        try:
            status = json.loads(run.stdout)
        except json.JSONDecodeError:
            return False
        return isinstance(status, list) and _CONTAINERD_STORE in status

    async def note_pulled(
        self,
        ref: str,
        info: Mapping[str, Any] | None = None,
        *,
        unless_recorded_since: datetime | None = None,
    ) -> None:
        """Record that this process pulled ``ref`` (absent before) (_pull_provenance).

        ``info`` is ``ref``'s ``docker image inspect`` just after the pull
        (read here when not given): its id goes in the record, and its tag
        time when the daemon keeps one per name (:meth:`tag_times_per_name`;
        "" otherwise, which keeps the image). With
        ``unless_recorded_since``, a record this process made of the same
        reference and id at or after that time stands: it was taken just
        after the pull, and a later read may carry the time of a layer
        build's pin or of another trial's.
        """
        if info is None:
            info = await self._inspect(ref)
        image_id = str((info or {}).get("Id") or "")
        daemon = await self.daemon_id()
        if daemon is None or not info or not image_id:
            return
        owner = alias_owner_tag()
        if unless_recorded_since is not None and any(
            r.ref == ref
            and r.image_id == image_id
            and r.owner == owner
            and r.pulled is not None
            and r.pulled >= unless_recorded_since
            for r in _pull_provenance.read_pulls(daemon)
        ):
            return
        tag_time = last_tag_time(info) if await self.tag_times_per_name() else ""
        _pull_provenance.record_pull(daemon, ref, image_id, owner, tag_time)

    async def pull_if_absent(self, ref: str) -> str:
        """Pull ``ref`` when Docker says it has no such image.

        ``present`` (nothing pulled), ``pulled`` (and recorded as BenchFlow's
        pull, as a trial's own pull would be), or ``unknown`` when
        the inspect failed for another reason (then nothing is pulled: the
        trial's start decides, as before). Raises :class:`LayerError` when
        the pull fails.
        """
        info, absent = await self._inspect_or_absent(ref)
        if info is not None:
            return "present"
        if not absent:
            return "unknown"
        run = await self._docker("pull", "--quiet", ref, timeout=_PULL_TIMEOUT_SEC)
        if run.code != 0:
            raise LayerError(f"could not pull task image {ref}: {run.text[-500:]}")
        info = await self._inspect(ref)
        if info is not None:
            await self.note_pulled(ref, info)
        return "pulled"

    async def inspect_base(
        self,
        ref: str,
        *,
        pull: bool = False,
        for_agent: bool = False,
        agent_recipe: AgentLayerRecipe | None = None,
    ) -> BaseImage:
        """Read the task image ``ref`` (pulling it first if asked and absent).

        With ``agent_recipe``, the libc is first read from that recipe's own
        cached layer on this image, when there is one (:meth:`_libc`)."""
        info, absent = await self._inspect_or_absent(ref)
        if info is None and pull:
            run = await self._docker("pull", "--quiet", ref, timeout=_PULL_TIMEOUT_SEC)
            if run.code != 0:
                raise LayerError(f"could not pull task image {ref}: {run.text[-500:]}")
            info = await self._inspect(ref)
            if info is not None and absent:
                # Absent before, pulled by BenchFlow: `bench cache clean
                # --only pulled-images` may remove it later.
                await self.note_pulled(ref, info)
        if info is None:
            raise LayerError(f"task image {ref} is not on the Docker daemon")
        # FROM consumes ONBUILD even when the trusted task intended those
        # instructions only for its own downstream builds. No cached install
        # may add work beyond its recipe; both layer kinds use this boundary.
        config = info.get("Config")
        if not isinstance(config, dict):
            raise LayerError("could not inspect the task image's configuration")
        if for_agent:
            # Refuse before even the libc probe starts a shell with image env.
            _check_agent_layer_env(config)
        if config.get("OnBuild"):
            raise LayerError("task images with ONBUILD instructions install live")
        digest = image_content_digest(info)
        arch = str(info.get("Architecture") or "")
        if info.get("Variant"):
            arch = f"{arch}/{info['Variant']}"
        image_id = str(info.get("Id") or "")
        layer_tags = None
        if agent_recipe is not None:
            unknown = BaseImage(
                ref=ref, image_id=image_id, digest=digest, arch=arch, libc=""
            )

            def layer_tags(libc: str) -> str:
                return layer_tag(
                    AGENT, agent_layer_key(agent_recipe, _replace(unknown, libc=libc))
                )

        libc = await self._libc(
            image_id, digest, sorted(config.get("Volumes") or {}), layer_tags=layer_tags
        )
        return BaseImage(
            ref=ref,
            image_id=image_id,
            digest=digest,
            arch=arch,
            libc=libc,
            user=str((info.get("Config") or {}).get("User") or ""),
            rootfs=tuple((info.get("RootFS") or {}).get("Layers") or ()),
        )

    async def _drop_libc_probe(self, container: str, volumes: list[str]) -> None:
        """Remove the libc probe's named volumes. A cancelled or timed-out
        probe kills only the ``docker run`` CLI; its container can run on and
        hold the volumes, so it goes first, by name. Its ``--rm`` may still
        be removing it, so a volume in use is tried again, for a few
        seconds."""
        with contextlib.suppress(OSError, LayerError):
            await self._docker("rm", "--force", container)
        left = list(volumes)
        for attempt in range(_LIBC_PROBE_VOLUME_TRIES):
            if attempt:
                await asyncio.sleep(0.5)
            still = []
            for name in left:
                with contextlib.suppress(OSError, LayerError):
                    removed = await self._docker("volume", "rm", name)
                    if removed.code == 0 or "no such volume" in removed.text.lower():
                        continue
                still.append(name)
            left = still
            if not left:
                return
        logger.warning("libc probe volumes left in use: %s", ", ".join(left))

    async def _libc(
        self,
        image_id: str,
        digest: str,
        volumes: Iterable[str] = (),
        *,
        layer_tags: Callable[[str], str] | None = None,
    ) -> str:
        daemon = await self.daemon_id()
        scope = (daemon, digest) if daemon is not None else None
        cached = self._libc_cache.get(scope) if scope is not None else None
        if cached:
            return cached
        libc = (
            await self._libc_from_layer_tags(digest, layer_tags)
            if layer_tags is not None
            else None
        )
        if libc is None:
            libc = await self._libc_from_layers(digest)
        if libc is None:
            # Each VOLUME of the image gets a named volume with the trial
            # volume label, never an anonymous one (_trial_volumes), and goes
            # by name after the probe, the probe's container first.
            probe = uuid.uuid4().hex[:12]
            container = f"benchflow-libc-probe-{probe}"
            named = [
                (f"{container}-{index}", path) for index, path in enumerate(volumes)
            ]
            try:
                run = await self._docker(
                    "run",
                    "--rm",
                    "--name",
                    container,
                    "--pull",
                    "never",
                    "--network",
                    "none",
                    "--user",
                    "0:0",
                    *(
                        arg
                        for name, path in named
                        for arg in (
                            "--mount",
                            f"type=volume,src={name},dst={path},"
                            f"volume-label={TRIAL_VOLUME_LABEL}=libc-probe-{probe}",
                        )
                    ),
                    "--entrypoint",
                    "/bin/sh",
                    image_id,
                    "-c",
                    _LIBC_PROBE,
                )
            finally:
                if named:
                    await asyncio.shield(
                        self._drop_libc_probe(container, [name for name, _ in named])
                    )
            libc = (
                run.stdout.strip().splitlines()[-1]
                if run.code == 0 and run.stdout.strip()
                else ""
            )
            if libc not in _LIBCS:
                raise LayerError(
                    f"could not tell the task image's libc ({run.text[-300:] or 'no output'})"
                )
        if scope is not None:
            self._libc_cache[scope] = libc
        return libc

    async def _libc_from_layer_tags(
        self, digest: str, layer_tags: Callable[[str], str]
    ) -> str | None:
        """The libc of the one layer on this task image whose tag names it.

        ``layer_tags(libc)`` is the tag a layer for that libc would have (its
        key holds the libc). Read from the same labels as
        :meth:`_libc_from_layers`, by tag: the daemon's filtered image list
        that method needs takes 0.3-0.5 s on a daemon with a few hundred
        images, an inspect 30 ms (SPEED-FIX). None unless exactly one
        candidate's layer exists and its labels name this image and that
        libc; then the listing decides, as before."""
        candidates = list(_LIBCS)
        infos = await asyncio.gather(
            *(self._inspect(layer_tags(libc)) for libc in candidates)
        )
        found = [
            libc
            for libc, info in zip(candidates, infos, strict=True)
            if isinstance(info, dict)
            and ((info.get("Config") or {}).get("Labels") or {}).get(LAYER_BASE_LABEL)
            == digest
            and ((info.get("Config") or {}).get("Labels") or {}).get(LAYER_LIBC_LABEL)
            == libc
        ]
        return found[0] if len(found) == 1 else None

    async def _libc_from_layers(self, digest: str) -> str | None:
        """The libc an earlier layer on the same task image recorded, if any.

        A label on a local image only; anyone who can write it can run any
        container on the daemon anyway.
        """
        run = await self._docker(
            "image",
            "ls",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"label={LAYER_BASE_LABEL}={digest}",
        )
        ids = run.stdout.split() if run.code == 0 else []
        if not ids:
            return None
        run = await self._docker(
            "image",
            "inspect",
            "--format",
            f'{{{{index .Config.Labels "{LAYER_LIBC_LABEL}"}}}}',
            ids[0],
        )
        libc = run.stdout.strip() if run.code == 0 else ""
        return libc if libc in _LIBCS else None

    # -- build and reuse ------------------------------------------------

    async def ensure_agent_layer(
        self,
        recipe: AgentLayerRecipe,
        base_ref: str,
        *,
        alias: str | None = None,
        pull: bool = False,
        prune: bool = True,
        prune_keep: int | None = None,
        base: BaseImage | None = None,
    ) -> LayerResult:
        """The agent layer for ``recipe`` on ``base_ref``: reused, or built now.

        ``alias`` is a tag of the trial's own for the layer (the trial starts
        from it and teardown removes it); the key stays claimed until
        :meth:`release`. After a build (and with ``prune``), layers past
        ``prune_keep`` (default :func:`agent_layer_keep`), past
        :func:`agent_layer_max_bytes` together, or needed to keep
        :func:`agent_layer_min_free_bytes` free are pruned. ``base`` is
        ``base_ref`` already inspected (a proxy layer's ``as_base``).
        """
        started = time.monotonic()
        if base is None:
            base = await self.inspect_base(
                base_ref, pull=pull, for_agent=True, agent_recipe=recipe
            )
        # Inspect the exact image used by the build, including when it is a
        # proxy layer. Runtime mounts or startup code can invalidate installed
        # bytes without touching the recipe stamp; those images install live.
        info = await self._inspect(base.image_id)
        config = info.get("Config") if isinstance(info, dict) else None
        if not isinstance(config, dict):
            raise LayerError("could not inspect the task image's runtime configuration")
        _check_agent_layer_env(config)
        if config.get("OnBuild"):
            # Also covers a caller-supplied, already inspected BaseImage.
            raise LayerError("task images with ONBUILD instructions install live")
        if any(config.get(key) for key in ("Volumes", "Entrypoint", "Healthcheck")):
            raise LayerError(
                "the task image's runtime configuration can change the install"
            )
        key = agent_layer_key(recipe, base)
        tag = layer_tag(AGENT, key)

        async def build() -> str:
            return await self._build_agent_layer(recipe, base, key, tag)

        return await self._ensure(
            AGENT,
            key,
            base,
            build,
            started=started,
            alias=alias,
            prune=prune,
            keep=agent_layer_keep() if prune_keep is None else prune_keep,
            max_bytes=agent_layer_max_bytes(),
            timeout_sec=recipe.timeout_sec,
        )

    async def ensure_proxy_layer(
        self,
        recipe: ProxyLayerRecipe,
        base_ref: str,
        *,
        alias: str | None = None,
        pull: bool = False,
        prune: bool = True,
        prune_keep: int | None = None,
    ) -> LayerResult:
        """The proxy layer for ``recipe`` on ``base_ref``: reused, or built now.

        As :meth:`ensure_agent_layer`, with the proxy's own bounds
        (:func:`proxy_layer_keep`, :func:`proxy_layer_max_bytes`).
        """
        started = time.monotonic()
        base = await self.inspect_base(base_ref, pull=pull)
        key = proxy_layer_key(recipe, base)
        tag = layer_tag(PROXY, key)

        async def build() -> str:
            return await self._build_proxy_layer(recipe, base, key, tag)

        return await self._ensure(
            PROXY,
            key,
            base,
            build,
            started=started,
            alias=alias,
            prune=prune,
            keep=proxy_layer_keep() if prune_keep is None else prune_keep,
            max_bytes=proxy_layer_max_bytes(),
            timeout_sec=recipe.timeout_sec,
        )

    async def daemon_id(self) -> str | None:
        """Query this sandbox's daemon once, then pass its identity to consumers."""
        async with self._daemon._async_lock:
            if not self._daemon._checked:
                try:
                    run = await self._docker("info", "--format", "{{.ID}}")
                    found = run.stdout.strip() if run.code == 0 else ""
                except (LayerError, OSError):
                    found = ""
                await asyncio.to_thread(self._daemon.bind, found)
            identity = self._daemon._identity
            return identity.server_id if identity is not None else None

    async def _lock_key(
        self, kind: str, key: str, timeout_sec: float
    ) -> tuple[asyncio.Lock, _layer_lock.HostLock, float]:
        """Take ``key``'s process lock, then its host lock (_layer_lock).

        Waits at most ``timeout_sec`` (the recipe's build budget) plus
        :data:`LOCK_WAIT_GRACE_SEC` in all: long enough for another
        process's build of the key to finish. Returns both locks, held, and
        the seconds waited. Raises :class:`LayerLockFallback` when the host
        lock cannot be used or is still held past that bound.
        """
        started = time.monotonic()
        deadline = started + max(float(timeout_sec), 0.0) + LOCK_WAIT_GRACE_SEC
        daemon = await self.daemon_id()
        if daemon is None:
            raise LayerLockFallback(
                _layer_lock.PROCESS,
                time.monotonic() - started,
                "the Docker daemon's id could not be read, so its layer lock "
                "cannot be named; installing in the trial",
            )
        plock = _layer_lock.process_lock(daemon, kind, key)
        try:
            await asyncio.wait_for(
                plock.acquire(), timeout=max(deadline - time.monotonic(), 0.0)
            )
        except TimeoutError:
            raise LayerLockFallback(
                _layer_lock.TIMEOUT,
                time.monotonic() - started,
                f"the {kind} layer {key[:16]} was still being built by this "
                f"process after {time.monotonic() - started:.0f}s; installing "
                "in the trial",
            ) from None
        hlock = _layer_lock.HostLock(daemon, kind, key)
        try:
            await hlock.acquire(deadline)
        except _layer_lock.LockUnavailable as e:
            plock.release()
            raise LayerLockFallback(
                _layer_lock.PROCESS,
                time.monotonic() - started,
                f"the layer lock cannot be used ({e}); installing in the trial",
            ) from None
        except _layer_lock.LockTimeout:
            plock.release()
            raise LayerLockFallback(
                _layer_lock.TIMEOUT,
                time.monotonic() - started,
                f"the {kind} layer {key[:16]} was still locked by another "
                f"process after {time.monotonic() - started:.0f}s; installing "
                "in the trial",
            ) from None
        except BaseException:
            plock.release()
            raise
        return plock, hlock, time.monotonic() - started

    async def _ensure(
        self,
        kind: str,
        key: str,
        base: BaseImage,
        build: Callable[[], Awaitable[str]],
        *,
        started: float,
        alias: str | None,
        prune: bool,
        keep: int,
        max_bytes: int | None,
        timeout_sec: float = 900,
    ) -> LayerResult:
        tag = layer_tag(kind, key)
        build_sec: float | None = None
        # Both locks are held across the hit check, the build and the
        # trial's tag: no other process builds the key meanwhile, and no
        # prune (which try-locks the key) removes a layer this trial found.
        plock, hlock, waited = await self._lock_key(kind, key, timeout_sec)
        try:
            info = await self._layer_info(tag, key, base, kind)
            if info is not None and str(info.get("Id") or "") not in (
                trusted_layer_ids(kind, key)
            ):
                # Right labels, but no build of ours made it: another build
                # on the daemon (a task's Compose file) can tag anything.
                logger.warning(
                    "Not reusing %s: no BenchFlow build on this machine made it; "
                    "building the %s layer again",
                    tag,
                    kind,
                )
                info = None
            status = "hit" if info is not None else "miss"
            if info is None:
                # A failed build never reaches the post-build prune. Reclaim
                # idle layers first when the daemon is already low on space.
                floor = agent_layer_min_free_bytes()
                if prune and floor is not None:
                    free = await self._free_bytes()
                    if free is not None and free < floor:
                        await self.prune(
                            kind, keep=keep, max_bytes=max_bytes, min_free=floor
                        )
                build_started = time.monotonic()
                built_id = await build()
                build_sec = time.monotonic() - build_started
                # The image this build made, by its id: the tag may already
                # name another one.
                info = await self._layer_info(built_id, key, base, kind)
                if info is None:
                    raise LayerError(
                        f"the {kind} layer built as {tag} is not one step on top "
                        f"of task image {base.ref}"
                    )
                _record_built(kind, key, str(info.get("Id") or built_id))
            # Prune before acquisition. The key locks keep this layer alive;
            # cancellation here owns neither a trial alias nor a claim.
            if status == "miss" and prune:
                try:
                    await self.prune(
                        kind,
                        keep=keep,
                        max_bytes=max_bytes,
                        min_free=agent_layer_min_free_bytes(),
                    )
                except Exception as e:
                    logger.warning("Pruning old %s layers failed: %s", kind, e)
            daemon = await self.daemon_id()
            assert daemon is not None  # _lock_key already established it.
            claim = (daemon, key)
            image = alias or tag
            result = LayerResult(
                kind=kind,
                key=key,
                tag=tag,
                image=image,
                status=status,
                base=base,
                seconds=time.monotonic() - started,
                build_sec=build_sec,
                lock=_layer_lock.HOST,
                lock_wait_sec=waited,
                daemon_id=daemon,
                as_base=BaseImage(
                    ref=tag,
                    image_id=str(info.get("Id") or ""),
                    digest=image_content_digest(info),
                    arch=base.arch,
                    libc=base.libc,
                    user=base.user,
                    rootfs=tuple((info.get("RootFS") or {}).get("Layers") or ()),
                ),
            )
            if alias:
                try:
                    # By the id this trial checked: the tag may name another
                    # image by now.
                    run = await self._docker("tag", str(info.get("Id") or tag), alias)
                    if run.code != 0:
                        raise LayerError(
                            f"could not tag layer {tag} as {alias}: {run.text[-300:]}"
                        )
                except BaseException:
                    # Docker may have tagged before cancellation was observed.
                    # Finish rollback even if the caller cancels again, before
                    # unlocking the key. Never remove the shared cached tag.
                    cleanup = asyncio.create_task(self._discard_trial_alias(alias))
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            continue
                    cleanup.result()
                    raise
            # No await from acquisition through handoff: only the caller can
            # own a claim, and it receives the result needed to release it.
            self._claims[claim] = self._claims.get(claim, 0) + 1
            return result
        finally:
            hlock.release()
            plock.release()

    async def _discard_trial_alias(self, alias: str) -> None:
        try:
            run = await self._docker("image", "rm", "--no-prune", alias)
            if run.code != 0:
                logger.warning(
                    "Removing unreturned layer alias %s: %s", alias, run.text[-300:]
                )
        except Exception as e:
            logger.warning("Removing unreturned layer alias %s failed: %s", alias, e)

    @classmethod
    def release(cls, key: str, *, daemon: str | None = None) -> None:
        """A trial no longer runs from ``key``'s layer: prune may remove it."""
        if daemon is None:
            return  # Unknown ownership cannot release another daemon's claim.
        claim = (daemon, key)
        held = cls._claims.get(claim, 0) - 1
        if held > 0:
            cls._claims[claim] = held
        else:
            cls._claims.pop(claim, None)

    async def _is_layer(
        self, tag: str, key: str, base: BaseImage, kind: str = AGENT
    ) -> bool:
        return await self._layer_info(tag, key, base, kind) is not None

    async def _layer_info(
        self, tag: str, key: str, base: BaseImage, kind: str
    ) -> dict[str, Any] | None:
        """``tag``'s inspect when it is ``kind``'s layer for ``key`` on ``base``."""
        info = await self._inspect(tag)
        if info is None:
            return None
        labels = (info.get("Config") or {}).get("Labels") or {}
        layers = tuple((info.get("RootFS") or {}).get("Layers") or ())
        if (
            labels.get(LAYER_LABEL) == kind
            and labels.get(LAYER_KEY_LABEL) == key
            and labels.get(LAYER_BASE_LABEL) == base.digest
            and (
                len(layers) > len(base.rootfs)
                if kind != VERIFIER
                else (base.digest.startswith("context:") or layers == base.rootfs)
            )
            and layers[: len(base.rootfs)] == base.rootfs
        ):
            return info
        return None

    async def _pin_base(self, base: BaseImage, key: str, kind: str = AGENT) -> str:
        # FROM a tag of our own on the exact image id inspected: the task
        # image's own tag can move (another trial rebuilds it) between the
        # inspect and the build, and BuildKit cannot build FROM a bare id.
        # The pin names the layer's kind and full key and the process that
        # made it (:func:`pin_tag`), so `bench cache clean` can untag one
        # its process left behind (killed during a build) and nothing else.
        base_tag = pin_tag(kind, key)
        run = await self._docker("tag", base.image_id, base_tag)
        if run.code != 0:
            raise LayerError(f"could not pin task image {base.ref}: {run.text[-300:]}")
        return base_tag

    @staticmethod
    def _common_labels(kind: str, key: str, base: BaseImage) -> dict[str, str]:
        return {
            LAYER_LABEL: kind,
            LAYER_KEY_LABEL: key,
            LAYER_BASE_LABEL: base.digest,
            LAYER_LIBC_LABEL: base.libc,
            # The task image's compose labels name the project that built
            # it; with them `compose down --rmi all` of that project would
            # delete this layer as one of its images.
            f"{_COMPOSE_LABEL_PREFIX}project": "",
            f"{_COMPOSE_LABEL_PREFIX}service": "",
        }

    async def _build(
        self,
        tag: str,
        labels: Mapping[str, str],
        dockerfile: str,
        *,
        timeout: float,
        context: str | None = None,
    ) -> tuple[_Run, str]:
        """Run the build; its result and the id of the image it made.

        The id is read only from a build that succeeded: the legacy builder
        (a Docker without buildx) deletes the iid file before it fails on the
        layer's ``RUN --mount``, and a missing file here was a
        FileNotFoundError that stopped the trial instead of the LayerError
        whose fallback installs the proxy or agent in the trial. A failed
        build, or one that left no id, gives an empty id; the callers name it.
        """
        fd, iidfile = tempfile.mkstemp(prefix="benchflow-layer-", suffix=".iid")
        os.close(fd)
        image_id = ""
        try:
            args = ["build", "--iidfile", iidfile, "--tag", tag]
            for name, value in labels.items():
                args += ["--label", f"{name}={value}"]
            if context is None:
                args.append("-")  # Dockerfile on stdin, no build context
            else:
                args += ["--file", "-", context]
            run = await self._docker(*args, stdin=dockerfile.encode(), timeout=timeout)
            if run.code == 0:
                with (
                    contextlib.suppress(OSError),
                    open(iidfile, encoding="utf-8") as fh,
                ):
                    image_id = fh.read().strip()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(iidfile)
        return run, image_id

    async def _build_agent_layer(
        self, recipe: AgentLayerRecipe, base: BaseImage, key: str, tag: str
    ) -> str:
        base_tag = await self._pin_base(base, key, AGENT)
        try:
            dockerfile = agent_layer_dockerfile(recipe, base_tag, key, user=base.user)
            labels = {
                **self._common_labels(AGENT, key, base),
                LAYER_AGENT_LABEL: recipe.agent,
                LAYER_AGENT_VERSION_LABEL: recipe.version,
            }
            run, image_id = await self._build(
                tag, labels, dockerfile, timeout=float(recipe.timeout_sec)
            )
        finally:
            with contextlib.suppress(Exception):
                await self._docker("image", "rm", "--no-prune", base_tag)
        if run.code != 0:
            tail = "\n".join((run.stderr or run.stdout).splitlines()[-_LOG_TAIL_LINES:])
            raise LayerError(
                f"building the {recipe.agent} {recipe.version} layer failed:\n{tail}"
            )
        if not image_id:
            raise LayerError(f"the {recipe.agent} layer build named no image")
        return image_id

    async def _build_proxy_layer(
        self, recipe: ProxyLayerRecipe, base: BaseImage, key: str, tag: str
    ) -> str:
        base_tag = await self._pin_base(base, key, PROXY)
        context: str | None = None
        try:
            # The build's context holds the requirement file and nothing
            # else. A full or read-only temp folder here is a failed layer
            # (the proxy then installs in the trial), not a failed trial.
            try:
                context = tempfile.mkdtemp(prefix="benchflow-proxy-layer-")
                path = os.path.join(context, recipe.requirements_name)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(recipe.requirements_text)
            except OSError as e:
                raise LayerError(
                    f"could not write the proxy layer's build context: {e}"
                ) from e
            dockerfile = proxy_layer_dockerfile(
                recipe,
                base_tag,
                key,
                user=base.user,
            )
            labels = {
                **self._common_labels(PROXY, key, base),
                LAYER_PROXY_GROUP_LABEL: recipe.group,
            }
            run, image_id = await self._build(
                tag,
                labels,
                dockerfile,
                timeout=float(recipe.timeout_sec),
                context=context,
            )
        finally:
            if context is not None:
                shutil.rmtree(context, ignore_errors=True)
            with contextlib.suppress(Exception):
                await self._docker("image", "rm", "--no-prune", base_tag)
        if run.code != 0:
            tail = "\n".join((run.stderr or run.stdout).splitlines()[-_LOG_TAIL_LINES:])
            raise LayerError(f"building the {recipe.group} proxy layer failed:\n{tail}")
        if not image_id:
            raise LayerError(f"the {recipe.group} proxy layer build named no image")
        return image_id

    # -- prune ----------------------------------------------------------

    async def prune(
        self,
        kind: str = AGENT,
        *,
        keep: int,
        max_bytes: int | None = None,
        min_free: int | None = None,
        hold: Iterable[str] = (),
        only: Iterable[str] | None = None,
        orphans: bool = True,
    ) -> PruneReport:
        """Remove ``kind``'s layers past the newest that fit, and their records.

        The layers kept are the newest ones that fit: at most ``keep`` of
        them and, with ``max_bytes``, at most that many bytes together
        (layers in use, and those ``only`` leaves out, count toward both,
        however old they are). With ``min_free``, more go, oldest
        first, while the daemon's disk (when it runs on this machine) has
        less free space than that.

        Whatever the bounds say, a layer stays while something holds it: a
        claim of this process (:meth:`release` not yet called), any tag on
        its image besides the cached one (a trial's own tag,
        :data:`ALIAS_PREFIX`, of a live process or of one this machine cannot
        check; a verifier-recovery lease; a user's tag), a container, or
        ``hold``. A trial's tag whose process is gone (killed before its
        teardown) holds nothing: it is untagged first, unless a container
        still uses the layer. A layer is removed by its own
        tag once its labels are checked, without ``--force``, so an image
        built or committed from it (a snapshot) keeps its bytes; then its
        BuildKit records go, by a filter on the key's :func:`marker`. With
        ``orphans``, records of keys whose image is gone (removed by hand, or
        by an older prune that failed half way) are pruned too. ``only``
        limits removal to those keys (the rest count as kept).
        """
        _check_kind(kind)
        report = PruneReport()
        run = await self._docker(
            "image",
            "ls",
            "--filter",
            f"label={LAYER_LABEL}={kind}",
            "--format",
            "{{json .}}",
        )
        if run.code != 0:
            raise LayerError(f"could not list layers: {run.text[-300:]}")
        created: dict[str, datetime] = {}
        image_ids: dict[str, str] = {}
        aliased: set[str] = set()
        stale: list[tuple[str, str]] = []
        prefix = f"{kind}-"
        for line in run.stdout.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            repository = str(row.get("Repository") or "")
            if repository.startswith(ALIAS_PREFIX):
                alias_tag = str(row.get("Tag") or "")
                image_id = str(row.get("ID") or "")
                if alias_owner_state(alias_tag) == "gone":
                    stale.append((f"{repository}:{alias_tag}", image_id))
                else:
                    aliased.add(image_id)
                continue
            tag = str(row.get("Tag") or "")
            key = tag.removeprefix(prefix)
            if (
                row.get("Repository") != LAYER_REPOSITORY
                or not tag.startswith(prefix)
                or not _KEY_RE.fullmatch(key)
            ):
                continue
            when = _parse_created(str(row.get("CreatedAt") or ""))
            created[key] = max(created.get(key, when), when)
            image_ids[key] = str(row.get("ID") or "")
        # Before the inspect below: an alias left on the image is another
        # tag there, and would hold the layer.
        for ref, image_id in stale:
            if not await self._remove_stale_alias(ref, report):
                aliased.add(image_id)
        newest = sorted(created, key=lambda k: (created[k], k), reverse=True)
        daemon = await self.daemon_id()
        held = {key for owner, key in self._claims if owner == daemon} | set(hold)
        held |= {key for key, image_id in image_ids.items() if image_id in aliased}
        infos = await self._inspect_many([layer_tag(kind, key) for key in newest])
        sizes: dict[str, int] = {}
        for key in newest:
            info = infos.get(layer_tag(kind, key))
            if info is None:
                continue
            sizes[key] = _image_size(info)
            if _other_tags(info, layer_tag(kind, key)):
                held.add(key)
        chosen = None if only is None else set(only)
        # What stays whatever the bounds say (held layers, and those ``only``
        # leaves out) takes its places and bytes first, the oldest included;
        # idle layers then fit, newest first, in what is left. Charging a held
        # layer only when the walk reached it let a newer idle layer use bytes
        # an older held one already took.
        fixed = [
            k for k in newest if k in held or (chosen is not None and k not in chosen)
        ]
        used = sum(sizes.get(key, 0) for key in fixed)
        count = len(fixed)
        # Kept by the bounds alone, newest first: what a low disk takes back.
        spare: list[str] = []
        doomed: list[str] = []
        for key in newest:
            size = sizes.get(key, 0)
            if key in held:
                report.in_use.append(key)
            elif chosen is not None and key not in chosen:
                report.kept.append(key)
            elif count < keep and (max_bytes is None or used + size <= max_bytes):
                report.kept.append(key)
                spare.append(key)
                used += size
                count += 1
            else:
                doomed.append(key)
        if min_free is not None and spare:
            free = await self._free_bytes()
            if free is not None and free < min_free:
                need = min_free - free
                while spare and need > 0:
                    key = spare.pop()
                    report.kept.remove(key)
                    doomed.append(key)
                    need -= max(sizes.get(key, 0), 1)
        daemon = await self.daemon_id() if doomed or orphans else None
        for key in doomed:
            async with self._removal_lock(daemon, kind, key, report) as free:
                if free:
                    await self._remove_layer(kind, key, report, sizes.get(key, 0))
        gone = set(report.removed)
        report.bytes_left = sum(sizes.get(key, 0) for key in newest if key not in gone)
        if report.removed:
            logger.info(
                "Removed %d old %s layer(s), %.1f GB; %.1f GB of layers left",
                len(report.removed),
                kind,
                report.bytes_removed / _GB,
                report.bytes_left / _GB,
            )
        if orphans:
            du = await self._docker("builder", "du", "--verbose", timeout=120.0)
            if du.code == 0:
                orphaned = _description_keys(du.stdout, kind) - set(created) - held
                if chosen is not None:
                    orphaned &= chosen
                for key in sorted(orphaned):
                    async with self._removal_lock(daemon, kind, key, report) as free:
                        # Built again since the listing: its records are its own.
                        if free and await self._inspect(layer_tag(kind, key)) is None:
                            await self._prune_records(kind, key, report)
        return report

    @contextlib.asynccontextmanager
    async def _removal_lock(
        self, daemon: str | None, kind: str, key: str, report: PruneReport
    ) -> AsyncIterator[bool]:
        """Try-lock ``key`` for a removal and hold it through the removal.

        Yields False (and records why in ``report.skipped``) when a trial of
        any cooperating process holds the key's lock (it is building the
        layer, or found it and is tagging it), when the lock cannot be used
        here, or when this process has claimed the key since the listing. A
        lock that cannot be used never counts as free: the key stays.
        """
        reason: str | None = None
        lock: _layer_lock.HostLock | None = None
        if daemon is None:
            reason = (
                "the Docker daemon's id could not be read, so its lock cannot be taken"
            )
        else:
            lock = _layer_lock.HostLock(daemon, kind, key)
            try:
                if not lock.try_acquire():
                    reason = "a trial holds its lock (building or using the layer)"
                    lock = None
            except _layer_lock.LockUnavailable as e:
                reason = f"its lock cannot be used ({e})"
                lock = None
        if reason is None and (daemon, key) in self._claims:
            reason = "a trial of this process uses it"
        if reason is not None:
            report.skipped[key] = reason
            logger.info("Kept %s layer %s: %s", kind, key[:16], reason)
        try:
            yield reason is None
        finally:
            if lock is not None:
                lock.release()

    async def _remove_stale_alias(self, ref: str, report: PruneReport) -> bool:
        """Untag the alias ``ref`` of a trial whose process is gone.

        False (the alias stays and holds its layer) while a container uses
        the image, or when Docker cannot say or refuses.
        """
        ps = await self._docker("ps", "--all", "--quiet", "--filter", f"ancestor={ref}")
        if ps.code != 0 or ps.stdout.strip():
            return False
        run = await self._docker("image", "rm", "--no-prune", ref)
        if run.code != 0:
            logger.warning(
                "Removing leftover layer tag %s failed: %s", ref, run.text[-300:]
            )
            return False
        report.aliases_removed.append(ref)
        return True

    async def _inspect_many(self, refs: list[str]) -> dict[str, dict[str, Any]]:
        """``docker image inspect`` of ``refs`` in one call, by tag.

        A ref that is gone is left out (Docker still prints the others).
        """
        if not refs:
            return {}
        run = await self._docker("image", "inspect", *refs, timeout=120.0)
        try:
            items = json.loads(run.stdout or "[]")
        except json.JSONDecodeError:
            return {}
        wanted = set(refs)
        out: dict[str, dict[str, Any]] = {}
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            for tag in item.get("RepoTags") or []:
                if tag in wanted:
                    out[tag] = item
        return out

    async def _free_bytes(self) -> int | None:
        """Free bytes on the daemon's disk, when the daemon runs on this machine.

        Only an effective local Unix connection permits reading this host's
        disk. Hostname equality alone cannot establish locality, and a saved
        context can select a remote daemon even without DOCKER_HOST.
        """
        if await self.daemon_id() is None:
            return None
        from benchflow.sandbox._docker_subnets import _uses_local_socket

        if not _uses_local_socket(self._daemon):
            return None
        run = await self._docker("info", "--format", "{{.DockerRootDir}}\t{{.Name}}")
        root, _, name = (
            run.stdout.strip().partition("\t") if run.code == 0 else ("", "", "")
        )
        if not root or name != socket.gethostname() or not os.path.isdir(root):
            return None
        try:
            return shutil.disk_usage(root).free
        except OSError:
            return None

    async def _remove_layer(
        self,
        kind: str,
        key: str,
        report: PruneReport,
        size: int,
        *,
        records: bool = True,
    ) -> None:
        tag = layer_tag(kind, key)
        # A container of the layer (a running or kept trial of any process,
        # through any tag) keeps it.
        ps = await self._docker("ps", "--all", "--quiet", "--filter", f"ancestor={tag}")
        if ps.code != 0 or ps.stdout.strip():
            report.in_use.append(key)
            return
        info = await self._inspect(tag)
        if info is not None:
            labels = (info.get("Config") or {}).get("Labels") or {}
            if labels.get(LAYER_LABEL) != kind or labels.get(LAYER_KEY_LABEL) != key:
                # Not the layer BenchFlow built under this key: left alone.
                report.kept.append(key)
                return
            if _other_tags(info, tag):
                # Tagged since the listing (a trial, a recovery lease): in use.
                report.in_use.append(key)
                return
            # Only the cached tag, never --force: the image goes once nothing
            # else names it, and an image built or committed from the layer
            # keeps its bytes (a label-scoped `image prune --all` would take
            # every tag of the layer and every image that inherited its labels).
            run = await self._docker("image", "rm", "--no-prune", tag, timeout=120.0)
            if run.code != 0:
                logger.warning("Removing layer %s failed: %s", key, run.text[-300:])
                return
            if await self._inspect(tag) is not None:
                # Rebuilt or tagged again in between.
                report.in_use.append(key)
                return
            report.bytes_removed += size
        report.removed.append(key)
        if records:
            await self._prune_records(kind, key, report)

    async def _prune_records(self, kind: str, key: str, report: PruneReport) -> None:
        run = await self._docker(
            "builder",
            "prune",
            "--all",
            "--force",
            "--filter",
            _marker_filter(kind, key),
            timeout=300.0,
        )
        if run.code == 0:
            report.records_pruned.append(key)
        else:
            logger.warning(
                "Pruning build records of layer %s failed: %s", key, run.text[-300:]
            )

    # -- inventory and explicit removal (``bench cache``) ---------------

    async def inventory(self) -> LayerInventory:
        """What BenchFlow keeps on this daemon, read only (``bench cache list``).

        Layer images (``benchflow-layer:<kind>-<key>`` whose labels name that
        kind and key; any other image is left out, an unlabelled
        ``benchflow-layer:mine`` included), what holds each, the BuildKit
        records of each key (:func:`marker`), and the task-image pins layer
        builds left behind (:func:`pin_tag`). Nothing is removed or tagged:
        a dead process's alias is reported, not untagged.
        """
        inv = LayerInventory()
        run = await self._docker("image", "ls", "--format", "{{json .}}", timeout=120.0)
        if run.code != 0:
            raise LayerError(f"could not list images: {run.text[-300:]}")
        candidates: dict[str, tuple[str, str]] = {}
        pins: list[tuple[str, str]] = []
        for line in run.stdout.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            repository = str(row.get("Repository") or "")
            tag = str(row.get("Tag") or "")
            if repository == LAYER_REPOSITORY:
                kind, _, key = tag.partition("-")
                if kind in (AGENT, PROXY, VERIFIER) and _KEY_RE.fullmatch(key):
                    candidates[f"{repository}:{tag}"] = (kind, key)
            elif repository == _BASE_TAG_REPOSITORY or repository.startswith(
                f"{_BASE_TAG_REPOSITORY}/"
            ):
                pins.append((repository, tag))
        infos = await self._inspect_many(sorted(candidates))
        for ref, (kind, key) in sorted(candidates.items()):
            info = infos.get(ref)
            if info is None:
                continue
            labels = (info.get("Config") or {}).get("Labels") or {}
            if labels.get(LAYER_LABEL) != kind or labels.get(LAYER_KEY_LABEL) != key:
                continue
            entry = LayerEntry(
                kind=kind,
                key=key,
                tag=ref,
                image_id=str(info.get("Id") or ""),
                size=_image_size(info),
                created=_parse_iso(str(info.get("Created") or "")),
                base=str(labels.get(LAYER_BASE_LABEL) or ""),
                libc=str(labels.get(LAYER_LIBC_LABEL) or ""),
                agent=str(labels.get(LAYER_AGENT_LABEL) or ""),
                agent_version=str(labels.get(LAYER_AGENT_VERSION_LABEL) or ""),
                proxy_group=str(labels.get(LAYER_PROXY_GROUP_LABEL) or ""),
                verifier=str(labels.get(LAYER_VERIFIER_LABEL) or ""),
            )
            for other in _other_tags(info, ref):
                repository, _, alias_tag = other.rpartition(":")
                if repository.startswith(ALIAS_PREFIX):
                    state = alias_owner_state(alias_tag)
                    if state == "gone":
                        entry.dead_aliases.append(other)
                        continue
                    entry.holders.append(f"alias {other} (process {state})")
                else:
                    entry.holders.append(f"tag {other}")
            ps = await self._docker(
                "ps", "--all", "--quiet", "--filter", f"ancestor={ref}"
            )
            if ps.code != 0:
                entry.holders.append("containers could not be listed")
            else:
                for cid in ps.stdout.split():
                    entry.holders.append(f"container {cid}")
            inv.layers.append(entry)
        du = await self._docker("builder", "du", "--verbose", timeout=120.0)
        if du.code == 0:
            present = {(e.kind, e.key) for e in inv.layers}
            for kind, key, record_id, size, created in _du_records(du.stdout):
                found = inv.record(kind, key)
                if found is None:
                    found = RecordEntry(
                        kind=kind, key=key, image_present=(kind, key) in present
                    )
                    inv.records.append(found)
                found.ids.append(record_id)
                found.size += size
                if created is not None and (
                    found.created is None or created < found.created
                ):
                    found.created = created
        else:
            inv.notes.append(f"build records could not be listed: {du.text[-200:]}")
        for repository, tag in sorted(pins):
            inv.pins.append(await self._pin_entry(repository, tag))
        return inv

    async def _pin_entry(self, repository: str, tag: str) -> PinEntry:
        ref = f"{repository}:{tag}"
        parsed = parse_pin(repository, tag)
        info = await self._inspect(ref)
        entry = PinEntry(
            ref=ref,
            image_id=str((info or {}).get("Id") or ""),
            created=_parse_iso(str((info or {}).get("Created") or "")),
            size=_image_size(info or {}),
        )
        if parsed is None:
            entry.reason = "an older pin that names no owner; kept"
            return entry
        entry.kind, entry.key, owner = parsed
        state = alias_owner_state(owner)
        entry.owner = state
        if state != "gone":
            entry.reason = (
                f"its build may still run (process {state})"
                if state in ("this", "alive")
                else "its process is on another host or cannot be checked; kept"
            )
            return entry
        entry.removable = True
        return entry

    async def remove_selected(
        self,
        *,
        layers: Iterable[tuple[str, str]] = (),
        records: Iterable[tuple[str, str]] = (),
        pins: Iterable[str] = (),
    ) -> PruneReport:
        """Remove what ``bench cache clean --remove`` selected, nothing more.

        ``layers`` are (kind, key) of layer images, removed by their own tag
        once nothing holds them (a dead process's alias is untagged first);
        their records stay unless the same key is in ``records``. ``records``
        are pruned by the key's exact marker filter. ``pins`` are new-form
        pin refs whose process is gone. Every removal holds the key's lock
        (busy or unusable: kept, with the reason in ``skipped``) and looks
        again under it: claims, containers, other tags, labels, owner.
        """
        report = PruneReport()
        daemon = await self.daemon_id()
        record_keys = set(records)
        for kind, key in layers:
            _check_kind(kind)
            async with self._removal_lock(daemon, kind, key, report) as free:
                if not free:
                    continue
                await self._untag_dead_aliases(kind, key, report)
                info = await self._inspect(layer_tag(kind, key))
                await self._remove_layer(
                    kind,
                    key,
                    report,
                    _image_size(info or {}),
                    records=(kind, key) in record_keys,
                )
                if key in report.removed:
                    # Its records went with it (when they were selected).
                    record_keys.discard((kind, key))
        for kind, key in sorted(record_keys):
            _check_kind(kind)
            async with self._removal_lock(daemon, kind, key, report) as free:
                if free:
                    await self._prune_records(kind, key, report)
        for ref in pins:
            await self._remove_pin(daemon, ref, report)
        return report

    async def _untag_dead_aliases(
        self, kind: str, key: str, report: PruneReport
    ) -> None:
        info = await self._inspect(layer_tag(kind, key))
        for other in _other_tags(info or {}, layer_tag(kind, key)):
            repository, _, alias_tag = other.rpartition(":")
            if (
                repository.startswith(ALIAS_PREFIX)
                and alias_owner_state(alias_tag) == "gone"
            ):
                await self._remove_stale_alias(other, report)

    async def _remove_pin(
        self, daemon: str | None, ref: str, report: PruneReport
    ) -> None:
        repository, _, tag = ref.rpartition(":")
        parsed = parse_pin(repository, tag)
        if parsed is None:
            report.skipped[ref] = "not a pin that names its owner"
            return
        kind, key, owner = parsed
        lock_report = PruneReport()
        async with self._removal_lock(daemon, kind, key, lock_report) as free:
            if not free:
                report.skipped[ref] = lock_report.skipped.get(key, "busy")
                return
            if alias_owner_state(owner) != "gone":
                report.skipped[ref] = "its process is not known to be gone"
                return
            ps = await self._docker(
                "ps", "--all", "--quiet", "--filter", f"ancestor={ref}"
            )
            if ps.code != 0 or ps.stdout.strip():
                report.in_use.append(ref)
                return
            run = await self._docker("image", "rm", "--no-prune", ref, timeout=120.0)
            if run.code != 0:
                report.skipped[ref] = f"docker image rm failed: {run.text[-200:]}"
                return
            report.removed.append(ref)


@dataclass
class LayerEntry:
    """One cached layer image, as ``bench cache list`` shows it."""

    kind: str
    key: str
    tag: str
    image_id: str
    size: int
    created: datetime | None
    base: str = ""
    libc: str = ""
    agent: str = ""
    agent_version: str = ""
    proxy_group: str = ""
    verifier: str = ""
    # What keeps it (a live alias, another tag, a container); empty: idle.
    holders: list[str] = field(default_factory=list)
    # Aliases of processes that are gone: they hold nothing.
    dead_aliases: list[str] = field(default_factory=list)


@dataclass
class RecordEntry:
    """The BuildKit records of one layer key (found by :func:`marker`)."""

    kind: str
    key: str
    image_present: bool
    ids: list[str] = field(default_factory=list)
    size: int = 0
    created: datetime | None = None


@dataclass
class PinEntry:
    """A task-image pin a layer build left behind (:func:`pin_tag`)."""

    ref: str
    image_id: str
    created: datetime | None
    size: int
    kind: str = ""
    key: str = ""
    owner: str = ""
    removable: bool = False
    reason: str = ""


@dataclass
class LayerInventory:
    layers: list[LayerEntry] = field(default_factory=list)
    records: list[RecordEntry] = field(default_factory=list)
    pins: list[PinEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def record(self, kind: str, key: str) -> RecordEntry | None:
        for entry in self.records:
            if entry.kind == kind and entry.key == key:
                return entry
        return None


_SIZE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([kKMGTP]?i?B)")
_SIZE_UNITS = {
    "B": 1,
    "kB": 1000,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
}


def _parse_size(text: str) -> int:
    match = _SIZE_RE.search(text)
    if match is None:
        return 0
    return int(float(match.group(1)) * _SIZE_UNITS.get(match.group(2), 1))


def _parse_iso(text: str) -> datetime | None:
    """An RFC 3339 time (Docker's, nanoseconds included) in UTC, or None."""
    text = text.strip()
    if not text:
        return None
    match = re.fullmatch(
        r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)", text
    )
    if match is None:
        return None
    head, frac, zone = match.groups()
    frac = (frac or "0")[:6].ljust(6, "0")
    zone = "+00:00" if zone == "Z" else zone
    try:
        return datetime.fromisoformat(f"{head}.{frac}{zone}").astimezone(UTC)
    except ValueError:
        return None


def _du_records(du_text: str) -> list[tuple[str, str, str, int, datetime | None]]:
    """(kind, key, record id, bytes, created) of every layer record in
    ``docker builder du --verbose``; other records are left out."""
    pattern = re.compile(
        re.escape(f"{LAYER_LABEL}=") + r"([a-z][a-z0-9-]*) key=([0-9a-f]{64});"
    )
    out = []
    for block in re.split(r"\n\s*\n", du_text):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            name, sep, value = line.partition(":")
            if sep and name.strip() and name.strip() not in fields:
                fields[name.strip()] = value.strip()
        match = pattern.search(fields.get("Description", ""))
        if match is None or not fields.get("ID"):
            continue
        created = None
        stamp = fields.get("Created at", "")
        parts = stamp.split()
        if len(parts) >= 3:
            created = _parse_created(
                " ".join([parts[0], parts[1].split(".")[0], parts[2]])
            )
            if created == datetime.min.replace(tzinfo=UTC):
                created = None
        out.append(
            (
                match.group(1),
                match.group(2),
                fields["ID"],
                _parse_size(fields.get("Size", "")),
                created,
            )
        )
    return out


@dataclass
class PulledEntry:
    """An image reference BenchFlow pulled, and whether it may go."""

    record: _pull_provenance.PullRecord
    size: int = 0
    gone: bool = False
    removable: bool = False
    reason: str = ""


async def pulled_inventory(
    layers: DockerLayers,
) -> tuple[list[PulledEntry], str | None]:
    """The references BenchFlow recorded pulling on ``layers``' daemon (read only).

    Each is checked now: still the recorded image id and tag time, no other
    tag, no container, and its pulling process gone. Anything else keeps it,
    with the reason. Returns (entries, None), or ([], why) when the daemon
    cannot be named.
    """
    daemon = await layers.daemon_id()
    if daemon is None:
        return [], "the Docker daemon's id could not be read"
    out = []
    for record in _pull_provenance.read_pulls(daemon):
        out.append(await _check_pulled(layers, record))
    return out, None


async def _check_pulled(
    layers: DockerLayers, record: _pull_provenance.PullRecord
) -> PulledEntry:
    entry = PulledEntry(record=record)
    info, absent = await layers._inspect_or_absent(record.ref)
    if info is None and absent:
        entry.gone = True
        entry.reason = "no longer on the daemon"
        return entry
    if info is None:
        # A busy daemon or a lost connection: the record stays for next time.
        entry.reason = "the daemon could not be asked about it; kept"
        return entry
    entry.size = _image_size(info)
    if str(info.get("Id") or "") != record.image_id:
        entry.reason = "the reference names another image now; kept"
        return entry
    others = _other_tags(info, record.ref)
    if others:
        entry.reason = f"also tagged {', '.join(sorted(others)[:3])}; kept"
        return entry
    if not record.tag_time:
        entry.reason = "the daemon did not say when BenchFlow's pull tagged it; kept"
        return entry
    if last_tag_time(info) != record.tag_time:
        # Removed and pulled again (same id) or tagged since BenchFlow's pull:
        # the reference may be the user's now.
        entry.reason = "pulled or tagged again since BenchFlow pulled it; kept"
        return entry
    state = alias_owner_state(record.owner)
    if state in ("this", "alive"):
        entry.reason = "the run that pulled it is still going"
        return entry
    if state != "gone":
        entry.reason = "the process that pulled it cannot be checked; kept"
        return entry
    ps = await layers._docker(
        "ps", "--all", "--quiet", "--filter", f"ancestor={record.image_id}"
    )
    if ps.code != 0 or ps.stdout.strip():
        entry.reason = "a container uses it"
        return entry
    # Docker has no compare-and-remove for a reference. Even after all of
    # these checks a pull/tag can replace it before `image rm ref`; removing
    # by id can instead remove a new, unowned tag of that same image. A host
    # lock cannot serialize external Docker clients or other hosts. Keep it
    # until the backend offers conditional reference deletion.
    entry.reason = "Docker cannot atomically remove a checked reference; kept"
    return entry


async def remove_pulled(
    layers: DockerLayers, entries: Iterable[PulledEntry]
) -> tuple[list[str], dict[str, str]]:
    """Recheck selected pulled references and retire records proved vanished.

    Docker cannot condition deletion on the inspected reference identity;
    existing references stay. Returns (removed refs, kept ref -> why).
    """
    daemon = await layers.daemon_id()
    removed: list[str] = []
    kept: dict[str, str] = {}
    done: list[_pull_provenance.PullRecord] = []
    if daemon is None:
        return removed, {e.record.ref: "the daemon cannot be named" for e in entries}
    for entry in entries:
        fresh = await _check_pulled(layers, entry.record)
        if fresh.gone:
            done.append(entry.record)
            continue
        kept[entry.record.ref] = fresh.reason
    _pull_provenance.forget_pulls(daemon, done)
    return removed, kept
