"""Content-addressed Apptainer images built from task Dockerfiles."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from benchflow.sandbox.protocol import ImageConfig, ImageRef

_BUILD_LOCKS: dict[str, asyncio.Lock] = {}


@dataclass(frozen=True)
class ApptainerImage:
    """A local SIF and the image working directory used by ``exec``."""

    path: Path
    digest: str
    workdir: str | None = None


def _cache_root() -> Path:
    configured = os.environ.get("BENCHFLOW_APPTAINER_CACHE_DIR")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "benchflow" / "apptainer"
    )


def _context_digest(config: ImageConfig) -> str:
    """Hash the staged build context without following symlinks."""

    digest = hashlib.sha256()
    dockerfile = config.dockerfile.resolve()
    context = config.context_dir.resolve()
    digest.update(dockerfile.relative_to(context).as_posix().encode())
    digest.update(json.dumps(config.build_args or {}, sort_keys=True).encode())
    for path in sorted(context.rglob("*")):
        if path.is_symlink():
            continue
        relative = path.relative_to(context).as_posix()
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            digest.update(f"d:{relative}\0".encode())
        elif stat.S_ISREG(mode):
            digest.update(f"f:{relative}:{stat.S_IMODE(mode):o}\0".encode())
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


_VARIABLE_RE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)


def dockerfile_workdir(
    dockerfile: Path, build_args: dict[str, str] | None = None
) -> str | None:
    """Return the final stage's WORKDIR for ordinary Dockerfiles."""

    values = dict(build_args or {})
    workdir: str | None = None
    for raw in dockerfile.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        instruction, _, value = line.partition(" ")
        instruction = instruction.upper()
        value = value.strip()
        if instruction == "FROM":
            workdir = None
        elif instruction in {"ARG", "ENV"}:
            assignments = value.split() if instruction == "ENV" else [value]
            for assignment in assignments:
                key, separator, item = assignment.partition("=")
                if separator:
                    values.setdefault(key, item)
        elif instruction == "WORKDIR" and value:
            expanded = _VARIABLE_RE.sub(
                lambda match: values.get(
                    match.group("braced") or match.group("plain"), match.group(0)
                ),
                value,
            )
            if expanded.startswith("/"):
                workdir = expanded
            elif workdir:
                workdir = str(Path(workdir) / expanded)
    return workdir


async def _run(*args: str, timeout_sec: float) -> tuple[str, str]:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(f"Apptainer command timed out: {args[1]}") from None
    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace")
    if process.returncode != 0:
        detail = (err or out).strip()[-2000:]
        raise RuntimeError(f"Apptainer command failed ({process.returncode}): {detail}")
    return out, err


class ApptainerImageBuilder:
    """Build Dockerfile contexts into immutable, content-addressed SIFs."""

    def __init__(self, cache_dir: Path | None = None, timeout_sec: float = 600) -> None:
        self.cache_dir = (cache_dir or _cache_root()).resolve()
        self.timeout_sec = timeout_sec

    def _image_ref(self, config: ImageConfig) -> ImageRef:
        digest = config.cache_key or _context_digest(config)
        return ImageRef(str(self.cache_dir / f"{digest}.sif"), digest)

    async def cached(self, config: ImageConfig) -> ImageRef | None:
        image = self._image_ref(config)
        return image if Path(image.tag).is_file() else None

    async def build(self, config: ImageConfig) -> ImageRef:
        if config.build_args:
            raise ValueError(
                "Apptainer BuildKit image builds do not accept Docker build arguments"
            )
        image = self._image_ref(config)
        target = Path(image.tag)
        if target.is_file():
            return image
        lock = _BUILD_LOCKS.setdefault(str(target), asyncio.Lock())
        async with lock:
            if target.is_file():
                return image
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.stem}-",
                suffix=".sif",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            temporary.unlink()
            args = ["apptainer", "build"]
            buildkit_host = (
                os.environ.get("BENCHFLOW_APPTAINER_BUILDKIT_HOST")
                or os.environ.get("APPTAINER_BUILDKIT_HOST")
                or os.environ.get("BUILDKIT_HOST")
            )
            if buildkit_host:
                args.extend(["--buildkit-host", buildkit_host])
            args.extend([str(temporary), f"buildkit:{config.context_dir.resolve()}"])
            try:
                await _run(*args, timeout_sec=self.timeout_sec)
                await _run("apptainer", "inspect", str(temporary), timeout_sec=30)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return image

    async def resolve(
        self,
        *,
        dockerfile: Path,
        context_dir: Path,
        prebuilt: str | None,
        build_args: dict[str, str] | None = None,
        force_build: bool = False,
    ) -> ApptainerImage:
        if prebuilt and not force_build:
            path = Path(prebuilt).expanduser().resolve()
            if path.suffix != ".sif" or not path.is_file():
                raise ValueError(
                    "Apptainer sandbox image must be an existing local .sif file"
                )
            metadata = path.stat()
            digest = f"{metadata.st_size:x}-{metadata.st_mtime_ns:x}"
            workdir = (
                dockerfile_workdir(dockerfile, build_args)
                if dockerfile.is_file()
                else None
            )
            return ApptainerImage(path, digest, workdir)

        config = ImageConfig(
            dockerfile=dockerfile,
            context_dir=context_dir,
            build_args=build_args,
        )
        if force_build:
            cached = self._image_ref(config)
            Path(cached.tag).unlink(missing_ok=True)
        image = await self.build(config)
        return ApptainerImage(
            Path(image.tag),
            image.digest or "",
            dockerfile_workdir(dockerfile, build_args),
        )


def require_apptainer() -> str:
    """Return the Apptainer executable or raise an actionable error."""

    executable = shutil.which("apptainer")
    if not executable:
        raise RuntimeError("Apptainer executable not found on PATH")
    return executable
