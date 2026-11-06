"""The cloud metadata block's rules from a helper container (Docker).

``lockdown.block_cloud_metadata`` refuses the cloud metadata endpoints to
every non-root user of ``main`` with iptables rules in main's network
namespace. Task images rarely carry iptables, so root installed it with the
image's package manager in every trial: 1 to 6 s of apt-get before each agent
start (PERF, 2026-10-19), and no block at all where the image has no package
manager or cannot reach its mirror. On Docker the same root script now runs
in a container of BenchFlow's own that joins main's network namespace
(``docker run --network container:<main>``) and brings its own iptables: the
same rules land in the same namespace, and nothing is installed in main.

- The image (:data:`HELPER_DOCKERFILE`: a pinned Alpine with pinned iptables,
  iptables-legacy and nftables) is built by this process, once per Docker
  daemon, beside the first sandbox start and with no build cache, and run by
  the image ID that build gave: never an image found under a tag or a label,
  which a task's Compose file can give an image of its own, nor a cached step
  a task's build may have imported.
- The helper holds NET_ADMIN and NET_RAW and nothing else (no new
  privileges, a read-only root, tmpfs without exec) and shares only the
  network namespace with main: the agent can neither see nor signal its
  processes, and it runs nothing but BenchFlow's scripts from BenchFlow's
  image. It is exec'd, checked and removed by the container ID its run gave.
- It starts as soon as main exists, before the inert bootstrap releases the
  task's processes, and installs the rules right then. At the block, the
  helper must still run the image this process built and have
  :data:`_BLOCK_MARGIN_SEC` of its lifetime left; otherwise a new one runs
  the install again (it adds nothing to rules already there). The block then
  proves the rules from main as before (the agent user's refused connect, and
  the rule's packet counter read through the helper) and removes the helper.
  What the helper's scripts print comes back as a checked receipt, as the
  block's reads in main do.
- The helper serves the block only when its script added every rule
  (``status=0``). Otherwise (no image and no way to build one, a daemon that
  refuses it, rules it could not add, a main that changed since) the block
  installs iptables in main as before, and its record says why.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import stat
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._exec_receipt import (
    new_receipt_nonce,
    read_receipt,
    receipt_command,
)

logger = logging.getLogger("benchflow")

#: Alpine 3.24's image index (every architecture), pinned by digest.
HELPER_BASE = (
    "alpine:3.24@sha256:"
    "294b683cb724975bec92580e1e685676bd4b50bda910ddb8c51d4cabeaec77e6"
)
#: The firewall tools and the netfilter libraries they load, at the versions
#: the pinned base's repository had when this recipe was checked. A
#: repository that has moved on fails the build, and the block installs in
#: main. iptables-legacy gives the script's tool choice the same fallback
#: root has in a Debian main, for hosts where nf_tables does not work.
HELPER_PACKAGES = (
    "iptables=1.8.13-r0",
    "iptables-legacy=1.8.13-r0",
    "nftables=1.1.6-r1",
    "libxtables=1.8.13-r0",
    "libip4tc=1.8.13-r0",
    "libip6tc=1.8.13-r0",
    "libnftnl=1.3.1-r0",
    "libmnl=1.0.5-r2",
)
#: The root scripts' tool custody walk calls /bin/readlink by path
#: (``lockdown._cloud_metadata_root_prelude``); BusyBox keeps readlink in
#: /usr/bin, so the image links it where the walk looks.
HELPER_DOCKERFILE = (
    f"FROM {HELPER_BASE}\n"
    f"RUN apk add --no-cache {' '.join(HELPER_PACKAGES)}"
    " && ln -s /bin/busybox /bin/readlink\n"
)
HELPER_RECIPE = hashlib.sha256(HELPER_DOCKERFILE.encode()).hexdigest()[:16]
HELPER_TAG = f"benchflow-helper:metadata-{HELPER_RECIPE}"
HELPER_LABEL = "benchflow.helper"
HELPER_RECIPE_LABEL = "benchflow.helper.recipe"
#: ``0`` keeps the block's install in main (the helper never runs). A root
#: agent's main then keeps NET_RAW, so every root agent on Docker is refused
#: unless the run opts out of the block.
HELPER_ENV = "BENCHFLOW_METADATA_HELPER"
# A helper ends itself after this long, whatever happens to BenchFlow.
_HELPER_LIFETIME_SEC = 1800
# The block uses a helper only with this long left: more than its reads
# through the helper can take (count, agent check, connect, count: 190 s at
# their timeouts).
_BLOCK_MARGIN_SEC = 300
_BUILD_SEC = 300
_RUN_SEC = 60
# A failed build is tried again after this long (a mirror that was down).
_RETRY_BUILD_SEC = 600
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")

DockerRun = Callable[..., Awaitable[ExecResult]]


class HelperUnavailable(RuntimeError):
    """The helper cannot serve the block; it installs in main instead."""


def helper_enabled() -> bool:
    return os.environ.get(HELPER_ENV, "").strip() != "0"


#: ``1`` keeps main's NET_ADMIN for the block where the helper adds its
#: rules, as it was before the helper's own step. A root agent's main then keeps NET_RAW
#: too, so every root agent on Docker is refused unless the run opts out.
MAIN_NET_ADMIN_ENV = "BENCHFLOW_METADATA_MAIN_NET_ADMIN"


def main_keeps_net_admin() -> bool:
    return os.environ.get(MAIN_NET_ADMIN_ENV, "").strip() == "1"


def _clock() -> float:
    return time.monotonic()


# Per daemon: the image ID this process's build gave, or the time and
# reason a build failed.
_READY: dict[str, str] = {}
_FAILED: dict[str, tuple[float, str]] = {}
_BUILD_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}


#: Where the IDs of helper images BenchFlow built are kept, one file per
#: daemon and recipe. The folder is 0700 and each file 0600, both
#: this user's; anything else is not read.
STATE_DIR_ENV = "BENCHFLOW_METADATA_HELPER_STATE_DIR"


def state_dir() -> Path:
    configured = os.environ.get(STATE_DIR_ENV, "").strip()
    if configured:
        return Path(configured)
    state = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(state) if state else Path.home() / ".local" / "state"
    return base / "benchflow" / "metadata-helper"


def _state_path(daemon_id: str) -> Path:
    key = hashlib.sha256(f"{daemon_id}\0{HELPER_RECIPE}".encode()).hexdigest()
    return state_dir() / f"{key[:32]}.json"


def _private(st: os.stat_result, kind: int) -> bool:
    """This user's, of ``kind`` (S_IFDIR or S_IFREG), and closed to others."""
    return (
        stat.S_IFMT(st.st_mode) == kind
        and st.st_uid == os.getuid()
        and not st.st_mode & 0o077
    )


def recorded_image(daemon_id: str) -> str | None:
    """The image ID a build recorded for ``daemon_id`` and this recipe, or
    None: no record, or one in a folder or file that is not this user's
    alone (a link included), or one that names another daemon or recipe."""
    folder = state_dir()
    try:
        if not _private(os.lstat(folder), stat.S_IFDIR):
            return None
        fd = os.open(
            _state_path(daemon_id),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError:
        return None
    try:
        if not _private(os.fstat(fd), stat.S_IFREG):
            return None
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = json.loads(f.read(4096) or b"{}")
    except (OSError, ValueError):
        return None
    finally:
        os.close(fd)
    if not isinstance(data, dict):
        return None
    image = data.get("image")
    if (
        data.get("daemon") != daemon_id
        or data.get("recipe") != HELPER_RECIPE
        or not isinstance(image, str)
        or not _IMAGE_ID.fullmatch(image)
    ):
        return None
    return image


def record_image(daemon_id: str, image: str) -> None:
    """Keep ``image``, built here with no cache, for later processes: a new
    0600 file swapped in whole, in a 0700 folder of this user's."""
    folder = state_dir()
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = os.lstat(folder)
    if stat.S_IFMT(st.st_mode) != stat.S_IFDIR or st.st_uid != os.getuid():
        return
    if st.st_mode & 0o077:
        os.chmod(folder, 0o700)
    path = _state_path(daemon_id)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    fd = os.open(
        tmp,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"daemon": daemon_id, "recipe": HELPER_RECIPE, "image": image}, f)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


async def _daemon_id(docker: DockerRun) -> str:
    """The daemon's own ID (``docker info``), "" when it cannot be read."""
    with contextlib.suppress(Exception):
        result = await docker(["info", "--format", "{{.ID}}"], timeout=_RUN_SEC)
        found = (result.stdout or "").strip()
        if result.return_code == 0 and re.fullmatch(r"[A-Za-z0-9:_-]{8,200}", found):
            return found
    return ""


async def _still_ours(docker: DockerRun, image: str) -> bool:
    """Whether the daemon holds ``image`` with the helper's labels for this
    recipe. An image ID is its content's digest: no task build can give
    another image this ID."""
    with contextlib.suppress(Exception):
        result = await docker(
            [
                "image",
                "inspect",
                "--format",
                "{{.Id}}\t{{index .Config.Labels "
                f'"{HELPER_LABEL}"'
                "}}\t{{index .Config.Labels "
                f'"{HELPER_RECIPE_LABEL}"'
                "}}",
                image,
            ],
            timeout=_RUN_SEC,
        )
        return result.return_code == 0 and (result.stdout or "").strip() == (
            f"{image}\tmetadata\t{HELPER_RECIPE}"
        )
    return False


async def ensure_image(docker: DockerRun, daemon: str) -> str:
    """The helper image's ID on this daemon, as BenchFlow built it.

    Built from the pinned recipe with no build cache (``docker build
    --no-cache --quiet`` prints the ID): the daemon's cache is shared with
    task builds, which may import cache from outside (``build.cache_from``),
    so a cached step is not one BenchFlow ran. The ID is recorded per daemon
    (:func:`record_image`), so later processes on the daemon reuse that build
    once ``docker image inspect`` shows the ID with this recipe's labels; a
    missing, unreadable or mismatched record builds again. Never
    an image found by tag or label. :func:`prebuild` starts it beside the
    sandbox's start. Raises :class:`HelperUnavailable` with the reason when
    it cannot be built.
    """
    built = _READY.get(daemon)
    if built:
        return built
    lock = _BUILD_LOCKS.setdefault(
        (id(asyncio.get_running_loop()), daemon), asyncio.Lock()
    )
    async with lock:
        built = _READY.get(daemon)
        if built:
            return built
        failed = _FAILED.get(daemon)
        if failed is not None and _clock() - failed[0] < _RETRY_BUILD_SEC:
            raise HelperUnavailable(failed[1])
        daemon_id = await _daemon_id(docker)
        recorded = recorded_image(daemon_id) if daemon_id else None
        if recorded is not None and await _still_ours(docker, recorded):
            _READY[daemon] = recorded
            return recorded
        # Build locks are process-local. Two controllers can export this
        # recipe at once, and containerd rejects concurrent exports to the
        # same tag. Keep each export named until its immutable ID has its
        # own retained tag; never resolve an image through this name.
        build_tag = f"{HELPER_TAG}-build-{secrets.token_hex(12)}"
        result = await docker(
            [
                "build",
                "--no-cache",
                "--quiet",
                "--tag",
                build_tag,
                "--label",
                f"{HELPER_LABEL}=metadata",
                "--label",
                f"{HELPER_RECIPE_LABEL}={HELPER_RECIPE}",
                "-",
            ],
            stdin=HELPER_DOCKERFILE.encode(),
            timeout=_BUILD_SEC,
        )
        ids = _IMAGE_ID.findall(result.stdout or "")
        if result.return_code != 0 or not ids:
            why = "building the metadata firewall helper image failed: " + (
                _last_line(result.stderr or result.stdout or "") or "no image ID"
            )
            _FAILED[daemon] = (_clock(), why)
            raise HelperUnavailable(why)
        _FAILED.pop(daemon, None)
        _READY[daemon] = ids[-1]
        if daemon_id and await _keep(docker, ids[-1]):
            # Remove only this build's temporary name, after keeping the
            # image by ID. A failed keep leaves the export tag intact.
            with contextlib.suppress(Exception):
                await docker(["image", "rm", build_tag], timeout=_RUN_SEC)
            try:
                record_image(daemon_id, ids[-1])
            except OSError as exc:
                logger.debug("Metadata helper image ID not recorded: %s", exc)
        return ids[-1]


def kept_tag(image: str) -> str:
    """The build's own tag: after its temporary export tag is removed,
    containerd still needs a tag retaining the recorded image ID."""
    return f"{HELPER_TAG}-{image.removeprefix('sha256:')[:12]}"


async def _keep(docker: DockerRun, image: str) -> bool:
    """Tag ``image`` with :func:`kept_tag`, so a later build (another
    process's, or another BenchFlow's) cannot remove it. A task that tags
    another image so names an image with another ID, which is never used."""
    with contextlib.suppress(Exception):
        result = await docker(["tag", image, kept_tag(image)], timeout=_RUN_SEC)
        return result.return_code == 0
    return False


# Per daemon: this process's build started by prebuild, kept until it ends.
_PREBUILDS: dict[str, asyncio.Task[None]] = {}


def prebuild(docker: DockerRun, daemon: str) -> None:
    """Start this process's build of the helper image for ``daemon`` in the
    background, once, so it runs beside the sandbox's start instead of after
    main exists; the helper's start then takes the built ID (or the build's
    failure, and the block installs in main)."""
    if daemon in _READY or daemon in _PREBUILDS:
        return

    async def build() -> None:
        try:
            await ensure_image(docker, daemon)
        except Exception:
            # Kept in _FAILED with its reason, for the helper's start.
            logger.debug("Metadata helper image build failed", exc_info=True)
        finally:
            _PREBUILDS.pop(daemon, None)

    _PREBUILDS[daemon] = asyncio.get_running_loop().create_task(build())


def image_ready(daemon: str) -> bool:
    """Whether this process has built the helper image for ``daemon``
    already. A build still running is never waited for: a sandbox's start
    does not wait on it."""
    return daemon in _READY


def build_failure(daemon: str) -> str:
    """Why this process's last build of the helper image for ``daemon``
    failed ("" when none did)."""
    failed = _FAILED.get(daemon)
    return failed[1] if failed is not None else ""


def _last_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""


def _status(stdout: str) -> str | None:
    """The ``status=`` of the script's ``BF metadata`` line; None without one."""
    for line in (stdout or "").splitlines():
        if line.startswith("BF metadata "):
            found = re.search(r"\bstatus=(\S*)", line)
            return found[1] if found else ""
    return None


class MetadataHelper:
    """One sandbox's helper container in its main's network namespace.

    ``docker`` runs a docker CLI command for this sandbox's daemon and never
    touches main; ``name`` starts every helper container's name (each start
    adds a random part); ``labels`` are the owner labels a leftover sweep
    reads.
    """

    def __init__(
        self,
        docker: DockerRun,
        *,
        daemon: str,
        name: str,
        labels: Mapping[str, str],
    ) -> None:
        self._docker = docker
        self._daemon = daemon
        self.name = name
        self._labels = dict(labels)
        self._main: str | None = None
        # The install script the running install runs.
        self._command: str | None = None
        self._install: asyncio.Task[tuple[Any, str]] | None = None
        #: The running helper: its container ID, the image ID it runs, and
        #: when its run began (its lifetime counts from then at the latest).
        self.container: str | None = None
        self._image: str | None = None
        self._started_at: float | None = None
        # A run with no ID back yet (cancelled midway): the name it was given,
        # too random for any task container to hold, is all that finds it.
        self._starting: str | None = None
        self._closing: asyncio.Task[None] | None = None
        #: Why the helper did not serve the block, for its record.
        self.unavailable = ""

    @property
    def target(self) -> str | None:
        """The main container whose namespace the helper joins."""
        return self._main

    def install_early(self, main: str, command: str) -> None:
        """Start the helper in ``main``'s namespace and run root's install
        ``command`` there, in the background."""
        self._main = main
        self._command = command
        self._install = asyncio.get_running_loop().create_task(
            self._start_and_run(main, command)
        )

    async def installed(
        self, main: str | None, command: str, *, retry: bool = False
    ) -> tuple[Any, str] | None:
        """The install's ``(result, "")`` for ``main``, from a helper that
        added every rule, still runs the image this process built and has
        :data:`_BLOCK_MARGIN_SEC` of its lifetime left: the early one, or a
        new one that runs the install again (a changed main or script, a
        helper gone or near its end). None when no helper can serve the block
        (``unavailable`` says why). ``retry``, for a main without NET_ADMIN
        of its own, gives a new helper one more try first."""
        if not main:
            self.unavailable = "the sandbox's main container is not known"
            return None
        outcome = await self._served(main, command)
        if outcome is None and retry:
            await self._restart(main, command)
            outcome = await self._served(main, command)
        return outcome

    async def _served(self, main: str, command: str) -> tuple[Any, str] | None:
        # Another script (the root agent's refuses root too) runs anew.
        if self._install is None or self._main != main or self._command != command:
            await self._restart(main, command)
        outcome = await self._outcome()
        if outcome is None:
            return None
        unfit, again = await self._unfit()
        if unfit and again:
            await self._restart(main, command)
            outcome = await self._outcome()
            if outcome is None:
                return None
            unfit, _ = await self._unfit()
        if unfit:
            self.unavailable = unfit
            return None
        return outcome

    async def _restart(self, main: str, command: str) -> None:
        # An earlier helper (an earlier block's, one gone or near its end)
        # goes first.
        closing, self._closing = self._closing, None
        if closing is not None:
            with contextlib.suppress(Exception):
                await closing
        await self._remove()
        self.install_early(main, command)

    async def _outcome(self) -> tuple[Any, str] | None:
        task = self._install
        assert task is not None
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
            self.unavailable = "the metadata firewall helper was stopped"
        except Exception as exc:
            self.unavailable = str(exc) or type(exc).__name__
        return None

    async def _unfit(self) -> tuple[str, bool]:
        """Why the helper cannot carry the block's reads ("" when it can),
        and whether a new helper may: not for a container that runs another
        image than the one built."""
        if self.container is None or self._started_at is None:
            return "the metadata firewall helper has no container", True
        left = _HELPER_LIFETIME_SEC - (_clock() - self._started_at)
        if left < _BLOCK_MARGIN_SEC:
            return (
                f"the metadata firewall helper had {max(left, 0):.0f} s of its "
                "lifetime left",
                True,
            )
        try:
            state = await self._docker(
                [
                    "inspect",
                    "--format",
                    "{{.State.Running}} {{.Image}}",
                    self.container,
                ],
                timeout=_RUN_SEC,
            )
        except Exception:
            return "the metadata firewall helper could not be inspected", True
        running, _, image = (state.stdout or "").strip().partition(" ")
        if state.return_code != 0 or running != "true":
            return "the metadata firewall helper is no longer running", True
        if image != self._image:
            return (
                "the metadata firewall helper's container does not run the "
                "image this process built",
                False,
            )
        return "", False

    async def _start_and_run(self, main: str, command: str) -> tuple[Any, str]:
        image = await ensure_image(self._docker, self._daemon)
        labels = [f"--label={k}={v}" for k, v in sorted(self._labels.items())]
        name = f"{self.name}-{secrets.token_hex(6)}"
        self._starting = name
        # Its lifetime runs from no later than this.
        started_at = _clock()
        started = await self._docker(
            [
                "run",
                "--detach",
                "--rm",
                "--name",
                name,
                "--network",
                f"container:{main}",
                "--user",
                "0:0",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "NET_ADMIN",
                "--cap-add",
                "NET_RAW",
                "--security-opt",
                "no-new-privileges",
                "--read-only",
                "--tmpfs",
                "/run:rw,noexec,nosuid,nodev,size=1m",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=8m",
                "--ipc",
                "none",
                "--pids-limit",
                "32",
                "--memory",
                "64m",
                f"--label={HELPER_LABEL}=metadata",
                *labels,
                "--entrypoint",
                "/bin/sleep",
                image,
                str(_HELPER_LIFETIME_SEC),
            ],
            timeout=_RUN_SEC,
        )
        container = _last_line(started.stdout or "")
        if started.return_code != 0 or not _CONTAINER_ID.fullmatch(container):
            if started.return_code != 0:
                # The image may be gone since (a prune): build it again.
                _READY.pop(self._daemon, None)
            raise HelperUnavailable(
                "the metadata firewall helper did not start: "
                + (_last_line(started.stderr or started.stdout or "") or "no output")
            )
        if container == main:
            # Never main: the helper's execs reach only the container its own
            # run created.
            raise HelperUnavailable(
                "the metadata firewall helper's run reported main's id"
            )
        self.container, self._image, self._started_at = container, image, started_at
        self._starting = None
        from benchflow.sandbox.lockdown import _CLOUD_METADATA_EXEC_SEC

        result, failed = await self.run(command, timeout_sec=_CLOUD_METADATA_EXEC_SEC)
        status = _status(getattr(result, "stdout", "") or "")
        if failed or status is None:
            # Root's script in main still can: the block installs there.
            why = failed or _last_line(getattr(result, "stderr", "") or "")
            raise HelperUnavailable(
                "the metadata firewall helper's script did not report"
                + (f" ({why})" if why else "")
            )
        if status != "0":
            # Never a helper's failed install: root's script in main may
            # still add the rules with the image's own tools.
            raise HelperUnavailable(
                "the metadata firewall helper could not add the rules "
                f"(status {status or 'missing'})"
            )
        return result, failed

    async def run(
        self, command: str, *, timeout_sec: int, user: str | None = None
    ) -> tuple[Any, str]:
        """Root's ``command`` in the helper: ``(result, "")``, or
        ``(None, why)`` when the exec itself failed, as the block's own
        execs report. Its output comes back as a checked receipt, like every
        read the block makes in main (``_exec_receipt``): an answer without
        one reads as a failed exec with no output.

        ``user`` (``<uid>:<gid>``, numeric) runs it as that user instead:
        the block's test connect, as the agent's uid, for an image without
        the tools to make it (the rules match the uid in main's namespace,
        which the helper shares)."""
        if self.container is None:
            return None, "the metadata firewall helper has no container"
        if user is not None and not re.fullmatch(r"[0-9]{1,10}:[0-9]{1,10}", user):
            return None, f"not a numeric user for the helper: {user!r}"
        nonce = new_receipt_nonce()
        try:
            result = await self._docker(
                [
                    "exec",
                    *(["--user", user] if user is not None else []),
                    self.container,
                    "/bin/sh",
                    "-c",
                    receipt_command(command, nonce),
                ],
                timeout=timeout_sec,
            )
        except Exception as exc:
            detail = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else ""
            return None, f"{type(exc).__name__}{': ' + detail if detail else ''}"
        return read_receipt(result, nonce), ""

    def close(self) -> None:
        """Remove the helper in the background (the block has its proof)."""
        if self._closing is None:
            self._closing = asyncio.get_running_loop().create_task(self._remove())

    async def aclose(self) -> None:
        """Remove the helper and wait for it (teardown)."""
        self.close()
        if self._closing is not None:
            with contextlib.suppress(Exception):
                await self._closing

    async def _remove(self) -> None:
        install = self._install
        if install is not None and not install.done():
            install.cancel()
            await asyncio.gather(install, return_exceptions=True)
        self._install = None
        target = self.container or self._starting
        self.container = self._image = self._started_at = self._starting = None
        if target is None:
            return
        with contextlib.suppress(Exception):
            await self._docker(["rm", "--force", target], timeout=_RUN_SEC)
