"""Putting bfx where the agent cannot write (TRUSTED-RUNTIME.md §6).

Every backend puts the pinned binary at :data:`SANDBOX_PATH` (``/.bfx/bfx``)
before the agent's first byte, and says which guarantee that gives
(:data:`GUARANTEE`, a backend's ``bfx_delivery``):

- ``bind`` (Docker on this machine): a read-only bind of a host folder that
  holds only the verified binary, set when the container is created. No
  process in the container can change, unmount or cover it: that takes
  CAP_SYS_ADMIN, which no container BenchFlow starts has, and Docker's
  seccomp profile refuses mount calls. Holds against any agent.
- ``volume`` (a remote Docker daemon, or local by choice): a harness-owned
  named volume ``benchflow-bfx-<sha12>``, mounted read-only with
  ``volume-nocopy`` (so it never takes an image's own ``/.bfx``). It is
  filled once per daemon by bfx itself: a created container of a helper
  image made from the pinned bytes alone (``docker import``; no task image,
  no pull) runs ``bfx _populate``, which writes a new private file and
  renames it into place, so concurrent fillers never leave a partial file. Before any sandbox uses it, it must be
  BenchFlow's (:func:`_check_owned`: our label with the pin, the ``local``
  driver with no options, no Compose project's label), hold nothing but
  ``bfx`` (and fillers' temporary files), and its ``bfx`` must read back,
  through a never-started container, as the pinned SHA-256. A volume whose
  ``bfx`` is anything else is refused, never refilled. Holds against any
  agent.
- ``upload`` (providers without read-only mounts): root uploads the binary
  to ``/.bfx/`` (root:root, 0555) before the agent starts. Holds only
  against a non-root agent.

A backend whose delivery has not passed its tests delivers nothing
(``None``) and keeps today's kit path.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from benchflow._bfx import pins
from benchflow._bfx.locate import (
    SANDBOX_DIR,
    SANDBOX_PATH,
    BfxUnavailable,
    normalize_arch,
    sha256_of,
    verified_binary,
)

BIND = "bind"
VOLUME = "volume"
UPLOAD = "upload"

#: Who each delivery holds against.
GUARANTEE = {
    BIND: "any-agent",
    VOLUME: "any-agent",
    UPLOAD: "non-root-agent",
}

#: Operators and tests choose the Docker delivery: ``auto`` (bind on a local
#: daemon, volume on a remote one), ``bind``, ``volume`` or ``off`` (no bfx;
#: the kit path).
DELIVERY_ENV = "BENCHFLOW_BFX_DELIVERY"

VOLUME_LABEL = "benchflow.bfx.sha256"


def docker_mode(
    *, local_daemon: bool, environ: Mapping[str, str] | None = None
) -> str | None:
    """The delivery a Docker-family sandbox uses, or None for none."""
    choice = (environ if environ is not None else os.environ).get(DELIVERY_ENV, "auto")
    choice = choice.strip().lower() or "auto"
    if choice == "off":
        return None
    if choice == "auto":
        return BIND if local_daemon else VOLUME
    if choice == BIND:
        # A remote daemon cannot see this machine's folders.
        return BIND if local_daemon else VOLUME
    if choice == VOLUME:
        return VOLUME
    raise ValueError(
        f"{DELIVERY_ENV} must be auto, bind, volume or off, not {choice!r}"
    )


def _cache_root() -> Path:
    cache_home = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "benchflow" / "bfx"


def pinned_sha(arch: str) -> str:
    return pins.SHA256[normalize_arch(arch)]


def volume_name(arch: str) -> str:
    return f"benchflow-bfx-{pinned_sha(arch)[:12]}"


def host_dir(arch: str) -> Path:
    """A host folder that holds only the verified binary as ``bfx`` (0555),
    for a read-only bind at :data:`SANDBOX_DIR`. Made once per binary, by
    copy and rename, and checked again on every use."""
    name = normalize_arch(arch)
    source = verified_binary(name)
    want = pins.SHA256[name]
    root = _cache_root()
    target = root / f"{want[:16]}-{name}"
    if _holds(target, want):
        return target
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    try:
        shutil.copyfile(source, staging / "bfx")
        os.chmod(staging / "bfx", 0o555)
        if sha256_of(staging / "bfx") != want:
            raise BfxUnavailable("the copy of bfx does not match its pin")
        os.chmod(staging, 0o555)
        # Another process may have made it first; theirs is checked below.
        with contextlib.suppress(OSError):
            os.rename(staging, target)
    finally:
        if staging.exists():
            os.chmod(staging, 0o755)
            shutil.rmtree(staging, ignore_errors=True)
    if not _holds(target, want):
        raise BfxUnavailable(f"{target} does not hold the pinned bfx")
    return target


def _holds(folder: Path, want: str) -> bool:
    try:
        st = os.lstat(folder)
        names = os.listdir(folder)
    except OSError:
        return False
    if not stat.S_ISDIR(st.st_mode) or names != ["bfx"]:
        return False
    try:
        fst = os.lstat(folder / "bfx")
    except OSError:
        return False
    return stat.S_ISREG(fst.st_mode) and sha256_of(folder / "bfx") == want


def compose_overlay(
    mode: str,
    *,
    source: str,
    services: Sequence[str] = ("main",),
) -> str:
    """A Compose overlay that mounts bfx read-only at :data:`SANDBOX_DIR` in
    *services*: *source* is the host folder (``bind``) or the volume name
    (``volume``)."""
    if mode == BIND:
        mount = {
            "type": "bind",
            "source": source,
            "target": SANDBOX_DIR,
            "read_only": True,
        }
        document: dict = {"services": {s: {"volumes": [mount]} for s in services}}
    elif mode == VOLUME:
        mount = {
            "type": "volume",
            "source": source,
            "target": SANDBOX_DIR,
            "read_only": True,
            "volume": {"nocopy": True},
        }
        document = {
            "services": {s: {"volumes": [mount]} for s in services},
            "volumes": {source: {"external": True, "name": source}},
        }
    else:
        raise ValueError(f"no Compose overlay for delivery {mode!r}")
    return json.dumps(document, indent=2)


def run_args(mode: str, *, source: str) -> list[str]:
    """``docker run``/``create`` arguments for the same mount."""
    if mode == BIND:
        return ["--mount", f"type=bind,src={source},dst={SANDBOX_DIR},readonly"]
    if mode == VOLUME:
        return [
            "--mount",
            f"type=volume,src={source},dst={SANDBOX_DIR},readonly,volume-nocopy",
        ]
    raise ValueError(f"no mount for delivery {mode!r}")


def at_or_beneath(path: object) -> bool:
    if not isinstance(path, str) or not path.startswith("/"):
        return False
    clean = "/" + os.path.normpath(path).lstrip("/")
    return clean == SANDBOX_DIR or clean.startswith(SANDBOX_DIR + "/")


def mount_problems(
    inspected: Mapping[str, Any], *, mode: str | None, source: str | None
) -> list[str]:
    """What is wrong with the mounts at or beneath :data:`SANDBOX_DIR` of a
    created container (its ``docker inspect``): with a delivery, exactly one
    mount sits there, BenchFlow's own (*mode* from *source*, at
    :data:`SANDBOX_DIR`, read-only); without one, none does. This covers
    every way a mount gets there: a task's or operator's Compose entry, an
    image's ``VOLUME``, a ``--tmpfs``, a replayed snapshot."""
    problems: list[str] = []
    ours = 0
    mounts = inspected.get("Mounts")
    if not isinstance(mounts, list):
        return ["the container's mounts are unknown"]
    for mount in mounts:
        if not isinstance(mount, dict):
            problems.append("a mount the daemon did not describe")
            continue
        destination = mount.get("Destination")
        if not at_or_beneath(destination):
            continue
        kind = mount.get("Type")
        named = mount.get("Name") if kind == VOLUME else mount.get("Source")
        own = (
            mode is not None
            and source is not None
            and kind == mode
            and destination == SANDBOX_DIR
            and mount.get("RW") is False
            and (
                named == source
                or (
                    kind == BIND
                    and isinstance(named, str)
                    and os.path.realpath(named) == os.path.realpath(source)
                )
            )
        )
        if own:
            ours += 1
        else:
            problems.append(
                f"a {kind} mount of {named!r} at {destination} (RW {mount.get('RW')!r})"
            )
    host = inspected.get("HostConfig")
    tmpfs = host.get("Tmpfs") if isinstance(host, dict) else None
    for destination in tmpfs or {}:
        if at_or_beneath(destination):
            problems.append(f"a tmpfs mount at {destination}")
    if mode is not None and ours != 1:
        problems.append(f"{ours} mounts of BenchFlow's bfx at {SANDBOX_DIR}, not 1")
    return problems


# ------------------------------------------------------------------ volume

#: Every volume step (making the helper image, a check, a fill) finishes
#: within this many seconds, start to end. A step that runs out is a
#: :class:`BfxUnavailable`, and its container or image is removed by the name
#: or label it was given before it was asked for.
STEP_TIMEOUT = 60.0

#: The label that names one helper image (:func:`_helper_image`).
HELPER_LABEL = "benchflow.bfx.helper"


async def _docker(
    docker: Sequence[str],
    args: Sequence[str],
    env: Mapping[str, str] | None,
    *,
    timeout: float = 120.0,
) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *docker,
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env) if env is not None else None,
        start_new_session=os.name == "posix",
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        # A timeout here or a cancelled step: the CLI never outlives it.
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        raise
    return process.returncode or 0, out, err


def _member_bytes(archive: bytes, name: str) -> bytes | None:
    """The bytes of the one regular file *name* in a ``docker cp`` archive."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        members = tar.getmembers()
        if len(members) != 1 or members[0].name != name or not members[0].isreg():
            return None
        f = tar.extractfile(members[0])
        return None if f is None else f.read()


class VolumeTransport(Protocol):
    """How the volume steps reach a Docker daemon."""

    async def run(
        self, args: list[str], *, timeout: float = 120.0
    ) -> tuple[int, bytes, bytes]: ...

    async def file_digest(self, container: str, path: str) -> str | None: ...

    async def import_image(
        self, archive: Path, changes: Sequence[str], *, timeout: float
    ) -> tuple[int, bytes, bytes]: ...

    async def folder_names(self, container: str, path: str) -> list[str] | None: ...


#: The temporary file ``bfx _populate`` writes before its rename
#: (``ops::temp_file``): a filler at work, or one that died before renaming.
_FILL_TEMP = re.compile(r"\.bfx-[0-9]+-[0-9]+-[0-9a-f]+")


class DockerTransport:
    """How the volume steps reach a Docker daemon. This one runs the
    ``docker`` CLI on this machine; Daytona DinD runs it on the outer VM
    (:class:`benchflow.sandbox.daytona_dind.DinDBfxTransport`)."""

    def __init__(self, docker: Sequence[str], env: Mapping[str, str] | None) -> None:
        self.docker = list(docker)
        self.env = env

    async def run(
        self, args: list[str], *, timeout: float = 120.0
    ) -> tuple[int, bytes, bytes]:
        return await _docker(self.docker, args, self.env, timeout=timeout)

    async def file_digest(self, container: str, path: str) -> str | None:
        """The SHA-256 of the regular file *path* in *container* (created,
        not necessarily started); None when it is not one."""
        code, out, _ = await self.run(["cp", f"{container}:{path}", "-"])
        if code != 0:
            return None
        data = _member_bytes(out, PurePosixPath(path).name)
        return None if data is None else hashlib.sha256(data).hexdigest()

    async def import_image(
        self, archive: Path, changes: Sequence[str], *, timeout: float
    ) -> tuple[int, bytes, bytes]:
        """``docker import`` of *archive*, a file on this machine (the CLI
        sends it to the daemon)."""
        flags = [part for change in changes for part in ("--change", change)]
        return await self.run(["import", *flags, str(archive)], timeout=timeout)

    async def folder_names(self, container: str, path: str) -> list[str] | None:
        """The names directly in the folder *path* of *container* (created,
        not necessarily started); None when it cannot be read."""
        code, out, _ = await self.run(["cp", f"{container}:{path}", "-"])
        if code != 0:
            return None
        return _top_names(out, PurePosixPath(path).name)


def _top_names(archive: bytes, top: str) -> list[str] | None:
    """The names directly beneath *top* in a ``docker cp`` archive of a
    folder; None when the archive is not one folder."""
    names: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != top:
                return None
            if len(parts) == 1:
                if not member.isdir():
                    return None
            else:
                names.add(parts[1])
    return sorted(names)


def _helper_archive(host_folder: Path, want: str) -> bytes:
    """A tar whose one member is ``bfx`` (root, 0555): the verified bytes of
    *host_folder*'s ``bfx``, read once and checked against *want*."""
    data = (host_folder / "bfx").read_bytes()
    if hashlib.sha256(data).hexdigest() != want:
        raise BfxUnavailable("the copy of bfx does not match its pin")
    member = tarfile.TarInfo("bfx")
    member.size = len(data)
    member.mode = 0o555
    member.uid = member.gid = 0
    member.mtime = 0
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        tar.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


async def _bounded(t: VolumeTransport, step: str, coroutine, cleanup: list[str]):
    """*coroutine* within :data:`STEP_TIMEOUT`; afterwards, however it ended,
    ``docker`` runs *cleanup* (an ``rm -f`` of the step's recorded name), so
    a container the daemon made after the client gave up is still removed."""
    try:
        return await asyncio.wait_for(coroutine, STEP_TIMEOUT)
    except TimeoutError:
        raise BfxUnavailable(
            f"the bfx volume {step} did not finish within {STEP_TIMEOUT:g} s"
        ) from None
    finally:
        if cleanup:
            with contextlib.suppress(Exception):
                await t.run(cleanup, timeout=30.0)


async def _helper_image(
    t: VolumeTransport, host_folder: Path, want: str, nonce: str
) -> str:
    """An image whose whole filesystem is the pinned ``bfx`` at ``/bfx`` (its
    entrypoint), made by ``docker import`` from our bytes alone: no task
    image, no registry, no pull. Its own *nonce* label makes its ID unique
    to this use, so removing it never touches another's."""
    with tempfile.TemporaryDirectory(prefix="benchflow-bfx-image-") as folder:
        archive = Path(folder) / "bfx.tar"
        archive.write_bytes(_helper_archive(host_folder, want))
        code, out, err = await t.import_image(
            archive,
            ['ENTRYPOINT ["/bfx"]', f"LABEL {HELPER_LABEL}={nonce}"],
            timeout=STEP_TIMEOUT,
        )
    image = out.decode(errors="replace").strip().splitlines()[-1:] if code == 0 else []
    if not image or not _IMAGE_ID.fullmatch(image[0]):
        raise BfxUnavailable(
            f"cannot make bfx's helper image: {(out + err).decode(errors='replace')[-300:]}"
        )
    return image[0]


async def _remove_helper_image(t: VolumeTransport, nonce: str) -> None:
    """Remove the helper image(s) carrying *nonce*, found by that label.
    The list needs ``--all``: ``docker import`` gives the image no tag, and
    Docker 29's containerd image store lists no untagged image without it."""
    with contextlib.suppress(Exception):
        code, out, _ = await t.run(
            [
                "image",
                "ls",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"label={HELPER_LABEL}={nonce}",
            ],
            timeout=30.0,
        )
        images = sorted(set(out.decode(errors="replace").split())) if code == 0 else []
        if images:
            await t.run(["rmi", "--force", *images], timeout=30.0)


async def volume_contents(
    t: VolumeTransport, *, helper: str, volume: str
) -> tuple[list[str] | None, str | None]:
    """The names in *volume* and the SHA-256 of its ``bfx`` (None when it
    holds no such regular file), read through a created and never started
    container of the *helper* image (mounted read-only, no copy from the
    image), within :data:`STEP_TIMEOUT`."""
    name = f"benchflow-bfx-check-{uuid.uuid4().hex[:12]}"

    async def read() -> tuple[list[str] | None, str | None]:
        code, _, err = await t.run(
            [
                "create",
                "--name",
                name,
                "--network",
                "none",
                "--pull",
                "never",
                "--label",
                "benchflow.bfx=check",
                "--mount",
                f"type=volume,src={volume},dst=/v,readonly,volume-nocopy",
                helper,
            ],
            timeout=STEP_TIMEOUT,
        )
        if code != 0:
            raise BfxUnavailable(
                f"cannot read the bfx volume: {err.decode(errors='replace')[-300:]}"
            )
        names = await t.folder_names(name, "/v")
        digest = (
            await t.file_digest(name, "/v/bfx") if names and "bfx" in names else None
        )
        return names, digest

    return await _bounded(t, "check", read(), ["rm", "-f", name])


def _check_names(volume: str, names: list[str] | None) -> list[str]:
    """*volume* holds only ``bfx`` and fillers' temporary files: nothing an
    image or a task put there."""
    if names is None:
        raise BfxUnavailable(f"cannot list {volume}")
    foreign = [n for n in names if n != "bfx" and not _FILL_TEMP.fullmatch(n)]
    if foreign:
        raise BfxUnavailable(
            f"{volume} holds files that are not bfx's ({', '.join(foreign[:5])}); "
            "it is never used or refilled: remove it by name"
        )
    return names


async def _check_owned(t: VolumeTransport, volume: str, want: str) -> None:
    """*volume* is BenchFlow's helper volume for the pin *want*: our label
    with that pin, the ``local`` driver with no options (so no bind of a
    host folder), and no Compose project's label (a task's own volume of
    that name). ``docker volume create`` on an existing name changes
    nothing, so a volume someone else made first is caught here."""
    code, out, err = await t.run(
        ["volume", "inspect", "--format", "{{json .}}", volume], timeout=STEP_TIMEOUT
    )
    if code != 0:
        raise BfxUnavailable(
            f"cannot inspect {volume}: {err.decode(errors='replace')[-300:]}"
        )
    try:
        info = json.loads(out.decode(errors="replace").strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise BfxUnavailable(f"cannot read {volume}'s inspect output") from None
    labels = info.get("Labels") or {}
    problems = []
    if labels.get(VOLUME_LABEL) != want:
        problems.append(f"its {VOLUME_LABEL} label is {labels.get(VOLUME_LABEL)!r}")
    if info.get("Driver") != "local":
        problems.append(f"its driver is {info.get('Driver')!r}")
    if info.get("Options"):
        problems.append("it has driver options")
    if any(k.startswith("com.docker.compose.") for k in labels):
        problems.append("it belongs to a Compose project")
    if problems:
        raise BfxUnavailable(
            f"{volume} is not BenchFlow's bfx volume ({'; '.join(problems)}); "
            "it is never used or refilled: remove it by name"
        )


async def _populate(t: VolumeTransport, *, helper: str, volume: str, want: str) -> None:
    """Fill *volume* with bfx: a created container of the *helper* image
    (nothing in it but the pinned ``/bfx``, checked again before it starts)
    runs ``bfx _populate /v`` (no network, no capabilities, no new
    privileges), which renames a complete file into place. Within
    :data:`STEP_TIMEOUT`."""
    name = f"benchflow-bfx-fill-{uuid.uuid4().hex[:12]}"

    async def fill() -> None:
        code, _, err = await t.run(
            [
                "create",
                "--name",
                name,
                "--network",
                "none",
                "--pull",
                "never",
                "--user",
                "0:0",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--label",
                "benchflow.bfx=fill",
                "--mount",
                f"type=volume,src={volume},dst=/v,volume-nocopy",
                helper,
                "_populate",
                "/v",
            ],
            timeout=STEP_TIMEOUT,
        )
        if code != 0:
            raise BfxUnavailable(
                f"cannot fill the bfx volume: {err.decode(errors='replace')[-300:]}"
            )
        got = await t.file_digest(name, "/bfx")
        if got != want:
            raise BfxUnavailable(
                f"the bfx filler holds {got}, not the pinned {want}; it was not started"
            )
        code, out, err = await t.run(["start", "-a", name], timeout=STEP_TIMEOUT)
        if code != 0:
            raise BfxUnavailable(
                f"bfx _populate failed ({code}): {(out + err).decode(errors='replace')[-300:]}"
            )

    await _bounded(t, "fill", fill(), ["rm", "-f", name])


_verified_volumes: set[tuple[str, str]] = set()


async def ensure_volume(
    docker: Sequence[str],
    env: Mapping[str, str] | None,
    *,
    arch: str,
    daemon_key: str,
    transport: VolumeTransport | None = None,
) -> str:
    """The name of a volume on this daemon that holds the pinned bfx for
    *arch*, filling it only when it holds no ``bfx`` yet. A volume that is
    not BenchFlow's, holds other files, or whose ``bfx`` is not the pin is
    refused. Checked once per daemon per process; fillers on this machine
    take turns (a lock file), and fillers elsewhere race safely (each
    renames a complete file into place).

    No step uses a task image: the check and the fill run in containers of
    a helper image made here from the pinned bytes alone
    (:func:`_helper_image`), removed again at the end. Every step is
    bounded (:data:`STEP_TIMEOUT`), and any failure is a
    :class:`BfxUnavailable`."""
    t = transport or DockerTransport(docker, env)
    want = pinned_sha(arch)
    volume = volume_name(arch)
    if (daemon_key, volume) in _verified_volumes:
        return volume
    folder = host_dir(arch)
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = (
        root / f".{hashlib.sha256(daemon_key.encode()).hexdigest()[:16]}-{volume}.lock"
    )
    nonce = uuid.uuid4().hex
    with open(lock_path, "a") as lock:
        await asyncio.to_thread(fcntl.flock, lock.fileno(), fcntl.LOCK_EX)
        try:
            code, _, err = await t.run(
                ["volume", "create", "--label", f"{VOLUME_LABEL}={want}", volume],
                timeout=STEP_TIMEOUT,
            )
            if code != 0:
                raise BfxUnavailable(
                    f"cannot create {volume}: {err.decode(errors='replace')[-300:]}"
                )
            await _check_owned(t, volume, want)
            helper = await _bounded(
                t, "helper image", _helper_image(t, folder, want, nonce), []
            )
            names, got = await volume_contents(t, helper=helper, volume=volume)
            if got is None and "bfx" not in _check_names(volume, names):
                # Empty (or only fillers' temporary files): fill it. Another
                # filler on another machine may be at work; both rename
                # complete files into place.
                await _populate(t, helper=helper, volume=volume, want=want)
                names, got = await volume_contents(t, helper=helper, volume=volume)
            _check_names(volume, names)
            if got != want:
                raise BfxUnavailable(
                    f"{volume} holds {got}, not the pinned {want}; "
                    "it is never used or refilled: remove it by name"
                )
        except TimeoutError:
            raise BfxUnavailable(
                f"a step on {volume} did not finish within {STEP_TIMEOUT:g} s"
            ) from None
        finally:
            await _remove_helper_image(t, nonce)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    _verified_volumes.add((daemon_key, volume))
    return volume


# ------------------------------------------------------------------ upload


def upload_prepare_command() -> str:
    """Root's shell command before the upload: ``/`` must be root's and
    writable by nobody else, and :data:`SANDBOX_DIR` must not exist yet (not
    even as a link), so the upload lands in a folder root has just made.
    Runs before the agent's first byte (pristine)."""
    return (
        f"set -e; d={SANDBOX_DIR}; "
        '[ "$(stat -c %u /)" = 0 ]; '
        'case "$(stat -c %A /)" in ?????w????|????????w?) exit 3;; esac; '
        '[ ! -e "$d" ] && [ ! -L "$d" ]; '
        'mkdir -m 0700 "$d"'
    )


def upload_finish_command() -> str:
    """Root's shell command after the binary is uploaded to
    :data:`SANDBOX_PATH`: root's own folder and file, 0555, and nothing
    else in the folder. Runs before the agent's first byte (pristine)."""
    return (
        f"set -e; d={SANDBOX_DIR}; "
        '[ -d "$d" ] && [ ! -L "$d" ] && [ -f "$d/bfx" ] && [ ! -L "$d/bfx" ]; '
        'chown 0:0 "$d" "$d/bfx"; chmod 0555 "$d/bfx"; chmod 0555 "$d"; '
        '[ "$(ls -A "$d")" = bfx ]; '
        f"{SANDBOX_PATH} version"
    )


def check_version(answer: Mapping[str, object], arch: str) -> None:
    """The delivered binary's own report must name the pinned hash."""
    if answer.get("sha256") != pinned_sha(arch):
        raise BfxUnavailable(
            f"the delivered bfx reports {answer.get('sha256')!r}, not the pinned {pinned_sha(arch)}"
        )


def version_answer(output: str) -> dict:
    """The JSON object ``bfx version`` printed (its last line)."""
    lines = [line for line in (output or "").splitlines() if line.strip()]
    if not lines:
        raise BfxUnavailable("bfx version printed nothing")
    try:
        answer = json.loads(lines[-1])
    except ValueError:
        raise BfxUnavailable(f"bfx version printed {lines[-1][:200]!r}") from None
    if not isinstance(answer, dict) or answer.get("name") != "bfx":
        raise BfxUnavailable(f"not a bfx version answer: {lines[-1][:200]!r}")
    return answer


# ---------------------------------------------------------------- providers

#: Providers whose delivery has passed a live run. The others deliver only
#: when :data:`PROVIDER_ENV` is ``on`` (their tests use fakes until then),
#: and otherwise keep the kit path (TRUSTED-RUNTIME.md §6, fallback).
PROVEN_LIVE: frozenset[str] = frozenset()

#: ``on`` lets a provider not in :data:`PROVEN_LIVE` deliver bfx.
PROVIDER_ENV = "BENCHFLOW_BFX_PROVIDERS"

#: Each provider's delivery.
PROVIDER_DELIVERY = {
    "modal": VOLUME,
    "daytona-dind": VOLUME,
    "daytona": UPLOAD,
    "agentcore": UPLOAD,
}


def provider_enabled(backend: str, environ: Mapping[str, str] | None = None) -> bool:
    """Whether *backend* (a :data:`PROVIDER_DELIVERY` key) delivers bfx."""
    if backend not in PROVIDER_DELIVERY:
        raise ValueError(f"no bfx delivery for {backend!r}")
    values = environ if environ is not None else os.environ
    if (values.get(DELIVERY_ENV) or "").strip().lower() == "off":
        return False
    if backend in PROVEN_LIVE:
        return True
    return (values.get(PROVIDER_ENV) or "").strip().lower() in ("1", "on", "true")


async def deliver_by_upload(upload, run_root) -> str:
    """Upload delivery (native Daytona, AgentCore): before the agent's
    first byte, root makes :data:`SANDBOX_DIR`, the verified binary is
    uploaded into it, and root makes both 0555 and its own. Returns the
    architecture; raises :class:`BfxUnavailable` when any step fails.

    *upload(source: Path, target: str)* is the backend's file upload, and
    *run_root(command: str) -> (exit code, stdout)* runs a shell command as
    root.
    """
    code, out = await run_root("uname -m")
    if code != 0:
        raise BfxUnavailable("cannot read the sandbox's architecture")
    arch = normalize_arch(out.strip())
    source = verified_binary(arch)
    code, out = await run_root(upload_prepare_command())
    if code != 0:
        raise BfxUnavailable(
            f"{SANDBOX_DIR} cannot be made root's own here ({code}): {out[-300:]}"
        )
    try:
        await upload(source, SANDBOX_PATH)
    except Exception as exc:
        raise BfxUnavailable(f"the upload of bfx failed: {exc}") from exc
    code, out = await run_root(upload_finish_command())
    if code != 0:
        raise BfxUnavailable(f"the uploaded bfx was not sealed ({code}): {out[-300:]}")
    check_version(version_answer(out), arch)
    return arch


async def deliver_upload(sandbox, backend: str, upload, run_root) -> None:
    """Set ``sandbox.bfx_delivery`` to ``upload`` when *backend* delivers
    bfx (:func:`provider_enabled`) and :func:`deliver_by_upload` succeeds;
    otherwise leave it None, so the sandbox keeps the kit path."""
    sandbox.bfx_delivery = None
    if not provider_enabled(backend):
        return
    try:
        await deliver_by_upload(upload, run_root)
    except BfxUnavailable as exc:
        sandbox.logger.debug("bfx: not delivered (%s)", exc)
        return
    sandbox.bfx_delivery = UPLOAD


#: Modal's ``Volume.read_only()`` (a read-only mount) came in 1.0.5.
MODAL_MIN_SDK = (1, 0, 5)


def modal_sdk_supported(version: str) -> bool:
    found = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", version.strip())
    if found is None:
        return False
    return tuple(int(n or 0) for n in found.groups()) >= MODAL_MIN_SDK


def _modal_digest(volume) -> str | None:
    digest = hashlib.sha256()

    class _Sink:
        def write(self, data: bytes) -> int:
            digest.update(data)
            return len(data)

    try:
        volume.read_file_into_fileobj("bfx", _Sink())
    except Exception:
        return None
    return digest.hexdigest()


def modal_volume(modal, arch: str = "x86_64"):
    """A Modal Volume that holds the pinned bfx as ``bfx`` (0555), mounted
    read-only. The rule is Docker's (:func:`ensure_volume`): a volume with
    no ``bfx`` is filled (never overwriting one that appeared meanwhile) and
    read back against the pin; one whose ``bfx`` is anything else is
    refused, never refilled. Blocking (Modal's synchronous API): call in a
    thread. Raises :class:`BfxUnavailable` on an SDK older than
    :data:`MODAL_MIN_SDK` or any failed step."""
    if not modal_sdk_supported(getattr(modal, "__version__", "0")):
        raise BfxUnavailable(
            f"modal {getattr(modal, '__version__', '?')} has no read-only volumes "
            f"(needs {'.'.join(map(str, MODAL_MIN_SDK))})"
        )
    want = pinned_sha(arch)
    volume = modal.Volume.from_name(volume_name(arch), create_if_missing=True)
    got = _modal_digest(volume)
    if got is None:
        source = host_dir(arch) / "bfx"
        # Another filler may have put one there first; theirs is read back.
        with (
            contextlib.suppress(Exception),
            volume.batch_upload(force=False) as batch,
        ):
            batch.put_file(str(source), "/bfx", mode=0o555)
        got = _modal_digest(volume)
    if got != want:
        raise BfxUnavailable(
            f"the Modal bfx volume holds {got}, not the pinned {want}; "
            "it is never used or refilled: remove it by name"
        )
    return volume.read_only()
