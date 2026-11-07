"""Opt-in reuse of a separate verifier's image, without installing tools.

Invariant: the verifier layer changes how fast a verify runs, never what it
computes. The cache adds Docker labels only: no filesystem layer, RUN,
interpreter, package, environment override, or state copied into grading.
``test.sh`` is hashed as opaque bytes and is never analysed or executed here.
Shared verifiers use their ordinary image and tool state.

A separate verifier's immutable image can be wrapped in labels for U13's
locking, inventory and pruning. A context can also be reused when it is a
pure image recipe: digest-pinned FROMs without inherited ONBUILD, local
COPY, and declarative image settings. RUN, ADD, ARG and unresolved inputs
use the ordinary build path. In particular, network installers are never
cached by predicting what they would resolve. Docker's own build cache
still applies on the ordinary path.

Format 9 keys a certified plain Compose invocation for cache builds only.
Ordinary and fallback builds retain their original Docker launcher.
The cache decides whether to reuse that result; it has no context builder.
There is no seed hook or hardening exemption: every verifier starts with the same tool state as it
would with this option off. Daytona remains unsupported.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.sandbox import env_layers
from benchflow.sandbox.env_layers import (
    LAYER_VERIFIER_LABEL,
    VERIFIER,
    BaseImage,
    DockerLayers,
    LayerError,
    LayerResult,
    fresh_record,
    layer_tag,
    marker,
)

VERIFIER_LAYER_FORMAT = 9
DEFAULT_VERIFIER_LAYER_KEEP = 64
VERIFIER_LAYER_KEEP_ENV = "BENCHFLOW_VERIFIER_LAYER_KEEP"
DEFAULT_VERIFIER_LAYER_MAX_GB = 50.0
VERIFIER_LAYER_MAX_GB_ENV = "BENCHFLOW_VERIFIER_LAYER_MAX_GB"
_CONTEXT_BUILD_TIMEOUT_SEC = 1800
VERIFIER_LAYER_ENV = "BENCHFLOW_VERIFIER_LAYER"
_CACHE_PLATFORM = "linux/amd64"
# Go's strconv.ParseBool spellings, which Docker and Compose both use.
_GO_TRUE = frozenset({"1", "t", "T", "TRUE", "true", "True"})
_GO_FALSE = frozenset({"0", "f", "F", "FALSE", "false", "False"})


def compose_builder_identity(env: Mapping[str, str]) -> str:
    """Resolve the supported frontend without reading credential-bearing config.

    Compose can choose Bake from Docker's saved plugin configuration when
    COMPOSE_BAKE is absent (and newer versions default to Bake). Only an
    explicit Go false spelling overrides those defaults. Refuse uncertainty
    before lookup, including warm entries created with an unresolved default.
    Saved buildx/context selections still pass the fresh platform/daemon probes.
    """
    if env.get("COMPOSE_BAKE") not in _GO_FALSE:
        raise LayerError(
            "verifier context caching needs an explicit internal Compose builder "
            "(COMPOSE_BAKE=false); use the ordinary build path otherwise"
        )
    # Older Compose/CLI versions consult the daemon's saved BuildKit default
    # when this variable is unset or empty; tool versions do not identify it.
    if env.get("DOCKER_BUILDKIT") not in _GO_TRUE:
        raise LayerError(
            "verifier context caching needs an explicit BuildKit builder "
            "(DOCKER_BUILDKIT=1); use the ordinary build path otherwise"
        )
    return "compose-internal-buildkit"


async def _require_cache_platform(
    layers: DockerLayers,
    compose_env: Mapping[str, str] | None,
    compose_dir: Path | None,
) -> None:
    """Admit only the default Linux amd64 builder, never infer it from an image.

    Image inspection ignores DOCKER_DEFAULT_PLATFORM. Even a digest can name
    an index whose ordinary build selects different bytes. Keep this cache's
    supported platform fixed; every override (including aliases and lists)
    uses ordinary grading instead of implementing Docker's platform parser.
    Check Compose's effective environment too: task settings can override it.
    """
    client_env = os.environ if layers._env is None else layers._env
    for env in (client_env, compose_env if compose_env is not None else client_env):
        # Policies can CONVERT even a digest-pinned FROM before resolution.
        # Their contents and dependencies are not part of our image identity;
        # never read them or reuse an entry when either build route supplies one.
        if "EXPERIMENTAL_BUILDKIT_SOURCE_POLICY" in env:
            raise LayerError(
                "verifier image caching cannot establish inputs with a source policy; "
                "use the ordinary build path"
            )
        if (
            env.get("DOCKER_DEFAULT_PLATFORM")
            or env.get("BUILDX_BUILDER", "") not in {"", "default"}
            or env.get("BUILDKIT_HOST")
            or env.get("COMPOSE_ENV_FILES")
            or env.get("COMPOSE_BAKE", "0") not in _GO_FALSE
        ):
            raise LayerError(
                "verifier image caching needs the default linux/amd64 platform "
                "and builder; use the ordinary build path with overrides"
            )
        # Retain the conservative boundary for explicit legacy/invalid
        # settings. Unset/empty defaults need no cross-frontend equivalence:
        # the context is built by the very same Compose call in both modes.
        # Exact Go ParseBool spellings, no normalising.
        if env.get("DOCKER_BUILDKIT", "") not in _GO_TRUE | {""}:
            raise LayerError(
                "verifier image caching needs BuildKit for every build; "
                "use the ordinary build path when DOCKER_BUILDKIT disables it"
            )
    # Compose also loads defaults from dotenv files, including the working
    # directory before the project directory. Never read their contents
    # (they may contain credentials), or try to duplicate its precedence.
    for directory in {Path.cwd(), compose_dir} - {None}:
        assert directory is not None
        dotenv = directory / ".env"
        try:
            has_dotenv = dotenv.exists() or dotenv.is_symlink()
        except OSError as exc:
            raise LayerError(
                "the verifier Compose dotenv inputs could not be checked"
            ) from exc
        if has_dotenv:
            raise LayerError(
                "verifier image caching cannot establish the platform with "
                "Compose dotenv inputs; use the ordinary build path"
            )
    # Compose/buildx can use a persistently selected builder even when the
    # environment has no override. docker build alone defaults to Engine's
    # builder, so require those two selection routes to agree as well.
    # Probe the complete effective environments, not a list of selector names:
    # HOME, DOCKER_CONFIG and BUILDX_CONFIG can all relocate saved selections.
    # Two native default builders are equivalent only on the same daemon. Do
    # not use daemon_id()'s cache: its scope does not include every config root.
    clients = [layers]
    if compose_env is not None:
        clients.append(DockerLayers(env=compose_env))
    daemon_ids: set[str] = set()
    for client in clients:
        builder = await client._docker("buildx", "inspect")
        summary = builder.stdout.split("\n\n", 1)[0]
        if (
            builder.code != 0
            or re.findall(r"^Name:[ \t]+([^\r\n]+)$", summary, re.M) != ["default"]
            or re.findall(r"^Driver:[ \t]+([^\r\n]+)$", summary, re.M) != ["docker"]
        ):
            raise LayerError("verifier image caching needs the default Docker builder")
        run = await client._docker("info", "--format", "{{.OSType}}/{{.Architecture}}")
        if run.code != 0 or run.stdout.strip() not in {"linux/x86_64", "linux/amd64"}:
            raise LayerError(
                "verifier image caching could not establish the default linux/amd64 platform"
            )
        identity = await client._docker("info", "--format", "{{.ID}}")
        daemon_id = identity.stdout.strip()
        if identity.code != 0 or not daemon_id or any(c.isspace() for c in daemon_id):
            raise LayerError(
                "verifier image caching could not establish the Docker daemon"
            )
        daemon_ids.add(daemon_id)
    if len(daemon_ids) != 1:
        raise LayerError(
            "verifier image caching needs the same Docker daemon as Compose"
        )


def _require_image_platform(info: Mapping[str, Any] | None) -> None:
    if (
        info is None
        or info.get("Os") != "linux"
        or info.get("Architecture") != "amd64"
        or info.get("Variant")
    ):
        raise LayerError(
            "the verifier image does not resolve to the default linux/amd64 platform"
        )


def verifier_layer_requested() -> bool:
    """Whether verifier layers are asked for (:data:`VERIFIER_LAYER_ENV`)."""
    value = os.environ.get(VERIFIER_LAYER_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def verifier_layer_keep() -> int:
    return env_layers._env_count(VERIFIER_LAYER_KEEP_ENV, DEFAULT_VERIFIER_LAYER_KEEP)


def verifier_layer_max_bytes() -> int | None:
    return env_layers._env_gb(VERIFIER_LAYER_MAX_GB_ENV, DEFAULT_VERIFIER_LAYER_MAX_GB)


def _tree_entries(root: Path) -> Iterable[tuple[str, os.stat_result, Path]]:
    """Every entry under ``root`` (sorted, not following links)."""
    stack = [Path()]
    while stack:
        relative = stack.pop()
        directory = root / relative
        names = sorted(os.listdir(directory))
        for name in names:
            path = directory / name
            st = os.lstat(path)
            rel = (relative / name).as_posix()
            yield rel, st, path
            if stat.S_ISDIR(st.st_mode):
                stack.append(relative / name)


def _file_sha256(path: Path, st: os.stat_result) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path) -> str:
    """``sha256:`` of every entry under ``root``: path, type, mode, bytes, link.

    Raises OSError when an entry cannot be read; the caller then caches
    nothing (an unreadable test file is a task the verifier reports anyway).
    """
    entries: list[tuple[str, str]] = []
    for rel, st, path in _tree_entries(root):
        mode = stat.S_IMODE(st.st_mode)
        if stat.S_ISLNK(st.st_mode):
            entries.append((rel, f"l {mode:o} {os.readlink(path)}"))
        elif stat.S_ISREG(st.st_mode):
            entries.append((rel, f"f {mode:o} {_file_sha256(path, st)}"))
        elif stat.S_ISDIR(st.st_mode):
            entries.append((rel, f"d {mode:o}"))
        else:
            entries.append((rel, f"o {stat.S_IFMT(st.st_mode):o} {mode:o}"))
    blob = json.dumps(sorted(entries), separators=(",", ":"))
    return (
        "sha256:" + hashlib.sha256(blob.encode("utf-8", "surrogateescape")).hexdigest()
    )


def verifier_inputs_digest(tests_dir: Path, verifier_config: Mapping[str, Any]) -> str:
    """``sha256:`` of the verifier's inputs: ``tests/`` (test.sh in it) and its config."""
    material = {
        "tests": tree_digest(tests_dir) if tests_dir.is_dir() else None,
        "config": verifier_config,
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class VerifierLayerRecipe:
    """An image-cache identity; it cannot express any tool installation."""

    task: str
    inputs: str
    separate: bool = True
    timeout_sec: int = 900

    def summary(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "cache": "image-only"}


def verifier_layer_key(
    recipe: VerifierLayerRecipe, base: BaseImage, *, build_identity: str = ""
) -> str:
    material = {
        "format": VERIFIER_LAYER_FORMAT,
        "kind": VERIFIER,
        "base": base.digest,
        "libc": base.libc,
        "arch": base.arch,
        "inputs": recipe.inputs,
        "build": build_identity,
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def plan_verifier_layer(
    task: Any,
    *,
    separate: bool,
    fresh_install: bool = False,
    allow_internet: bool = True,
    sandbox_env: Mapping[str, str] | None = None,
) -> tuple[VerifierLayerRecipe | None, dict[str, Any] | None]:
    """Plan image reuse, independent of shell syntax or tool resolution.

    ``allow_internet`` and ``sandbox_env`` remain accepted by both rollout
    hooks, but do not affect eligibility: no tools are installed or seeded.
    """
    if not verifier_layer_requested():
        return None, None
    if fresh_install:
        return None, fresh_record("fresh", "fresh install asked for")
    if not separate:
        return None, fresh_record(
            "unsupported",
            "verifier caching reuses separate images only; shared grading uses "
            "the ordinary image and installs its tools normally",
        )
    tests_dir = Path(task.paths.tests_dir)
    try:
        inputs = verifier_inputs_digest(
            tests_dir, task.config.verifier.model_dump(mode="json")
        )
    except OSError as exc:
        return None, fresh_record(
            "fresh", f"the verifier's files could not be read: {exc}"
        )
    name = str(getattr(task, "name", "") or tests_dir.parent.name)
    return VerifierLayerRecipe(task=name, inputs=inputs), None


def verifier_layer_dockerfile(
    recipe: VerifierLayerRecipe, base_from: str, key: str, *, base: BaseImage
) -> str:
    """Labels only: preserve every filesystem byte and execution setting.

    The caller has inspected this exact pinned image ID and refused ONBUILD,
    so FROM cannot execute an inherited instruction. Never add RUN here.
    """
    return (
        f"FROM {base_from}\n"
        f"LABEL ai.benchflow.verifier-image-cache={json.dumps(marker(VERIFIER, key))}\n"
    )


@dataclass(frozen=True)
class VerifierLayer:
    result: LayerResult
    recipe: VerifierLayerRecipe

    def record(self) -> dict[str, Any]:
        return {**self.result.record(), **self.recipe.summary()}


def _dockerfile_lines(dockerfile: str) -> list[str]:
    """Join Docker continuations after removing whole-line comments.

    Shell continuations have different comment rules. Docker discards a
    comment even in the middle of an instruction and removes the escaped
    newline without adding whitespace, including inside an option name.
    Refuse directives and incomplete instructions we cannot interpret.
    """
    # An explicit alphabet makes Python's whitespace operations agree with
    # Docker on every admitted character. In particular, Python alone treats
    # U+001C..U+001F as whitespace. Refuse other controls and non-ASCII source
    # before a blank/comment/continuation check can erase a real instruction.
    if re.fullmatch(r"[\t\n\r\x20-\x7e]*", dockerfile) is None:
        raise LayerError("the verifier context uses unsupported Dockerfile characters")
    lines: list[str] = []
    pending = ""
    continued = False
    # Docker splits only on LF and strips trailing CR. Python's splitlines
    # and universal-newline reads also split other characters that can be
    # part of a RUN, exposing a continuation Docker never saw.
    for raw in dockerfile.split("\n"):
        raw = raw.rstrip("\r")
        if re.match(r"\s*#\s*(?:syntax|escape)\s*=", raw, re.I):
            raise LayerError(
                "the verifier context uses an unsupported Dockerfile directive"
            )
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        text = raw.rstrip(" \t")
        # Docker's rule excludes a backslash preceded by another backslash,
        # even for an odd-length run. This differs from shell continuations.
        if text.endswith("\\") and not text.endswith("\\\\"):
            pending += text[:-1]
            continued = True
            continue
        lines.append(pending + text)
        pending = ""
        continued = False
    if continued:
        raise LayerError(
            "the verifier context has an incomplete Dockerfile instruction"
        )
    return lines


def _copy_source(rest: str) -> str | None:
    """Read only canonical, allowlisted COPY flags before its source paths.

    BuildKit removes quotes and escapes even inside flag names. Rather than
    imitate that grammar, admit only literal flags whose meaning is fixed;
    unknown flags and noncanonical spellings use the ordinary build path.
    Once an argument starts, subsequent text is data, not builder flags.
    """
    flags: dict[str, str | None] = {}
    for word in rest.split():
        if not word.startswith("--"):
            break
        match = re.fullmatch(
            r"--(?:(from)=([A-Za-z0-9_.-]+)|(chown)=([A-Za-z0-9_.:-]+)"
            r"|(chmod)=([0-7]{3,4})|(link)(?:=(true|false))?)",
            word,
        )
        if match is None:
            raise LayerError("the verifier context has unsupported COPY flags")
        groups = match.groups()
        name, value = next(
            (groups[i], groups[i + 1]) for i in range(0, len(groups), 2) if groups[i]
        )
        if name in flags:
            raise LayerError("the verifier context has repeated COPY flags")
        flags[name] = value
    return flags.get("from")


async def _context_base(
    layers: DockerLayers, context: Path, *, pull: bool
) -> BaseImage:
    """Resolve a context's immutable image inputs before looking up its layer.

    A context hash alone does not identify mutable FROM tags or external
    COPY sources. Only digest-pinned FROMs, prior stages, and local COPYs
    are reused. Other Dockerfiles fall back to Docker's ordinary build path
    rather than risk grading on a stale image. Pin the build's platform to
    the inspected inputs as well (a digest may name a multi-platform index).
    """
    try:
        identity = await asyncio.to_thread(tree_digest, context)
        with (context / "Dockerfile").open(newline="") as source:
            dockerfile = source.read()
    except OSError as e:
        raise LayerError("the verifier image's build context could not be read") from e
    stage_names: set[str] = set()
    copy_indices: set[str] = set()
    stage_count = 0
    images: list[BaseImage] = []
    for text in _dockerfile_lines(dockerfile):
        text = text.strip()
        words = text.split(None, 1)
        instruction, rest = words[0], words[1] if len(words) > 1 else ""
        instruction = instruction.upper()
        if (
            instruction
            not in {
                "FROM",
                "COPY",
                "ADD",
                "ONBUILD",
                "ENV",
                "WORKDIR",
                "USER",
                "LABEL",
                "EXPOSE",
                "VOLUME",
                "CMD",
                "ENTRYPOINT",
                "STOPSIGNAL",
                "HEALTHCHECK",
                "SHELL",
                "MAINTAINER",
            }
            or not rest.strip()
            or "<<" in rest
        ):
            raise LayerError(
                "the verifier context needs a declarative Dockerfile (no RUN or ARG); use the ordinary build path"
            )
        if instruction == "FROM":
            match = re.fullmatch(
                r"([^\s]+)(?:\s+AS\s+([A-Za-z0-9_.-]+))?", rest.strip(), re.I
            )
            if not match:
                raise LayerError(
                    "the verifier context needs digest-pinned FROM images without platform overrides"
                )
            ref, stage = match.groups()
            if ref.lower() not in stage_names and ref.lower() != "scratch":
                if not re.fullmatch(r"[A-Za-z0-9._/:-]+@sha256:[0-9a-f]{64}", ref):
                    raise LayerError(
                        "the verifier context needs digest-pinned external FROM images"
                    )
                info = await layers._inspect(ref)
                _require_image_platform(info)
                if info is None or (info.get("Config") or {}).get("OnBuild"):
                    raise LayerError(
                        "the verifier context has unresolved inherited build inputs"
                    )
                image = await layers.inspect_base(str(info["Id"]))
                images.append(image)
            # FROM resolves names only; numeric stage indices are COPY's
            # syntax and must never exempt an external FROM from validation.
            copy_indices.add(str(stage_count))
            stage_count += 1
            if stage:
                stage_names.add(stage.lower())
        elif instruction == "COPY":
            source = _copy_source(rest)
            if source is not None and source.lower() not in stage_names | copy_indices:
                raise LayerError(
                    "the verifier context has an unresolved external COPY image"
                )
        elif instruction == "ADD" or (
            instruction == "RUN" and rest.lstrip().startswith("--")
        ):
            raise LayerError(
                "the verifier context has unsupported external build inputs"
            )
        elif instruction == "ONBUILD":
            raise LayerError(
                "the verifier context has unsupported deferred build inputs"
            )
    if not images or len({image.arch for image in images}) != 1:
        raise LayerError(
            "the verifier context needs pinned images on one resolved platform"
        )
    dependencies = [(image.digest, image.arch, image.libc) for image in images]
    material = json.dumps([identity, dependencies], separators=(",", ":"))
    return BaseImage(
        ref=str(context),
        image_id="",
        digest="context:" + hashlib.sha256(material.encode()).hexdigest(),
        arch=images[-1].arch,
        libc=images[-1].libc,
    )


async def ensure_verifier_layer(
    layers: DockerLayers,
    recipe: VerifierLayerRecipe,
    base_ref: str | None,
    *,
    alias: str | None,
    pull: bool = False,
    context: Path | None = None,
    base: BaseImage | None = None,
    compose_env: Mapping[str, str] | None = None,
    compose_dir: Path | None = None,
    build_context: Callable[[], Awaitable[BaseImage]] | None = None,
    build_identity: Callable[[], Awaitable[str]] | None = None,
) -> VerifierLayer:
    """``recipe``'s layer on ``base_ref`` (or built from ``context``): reused, or built now.

    With ``context`` (a separate verifier's image from its Dockerfile) the
    key holds the context's digest and its resolved, pinned image inputs,
    so a hit builds nothing at all; a miss calls the sandbox's ordinary
    build function, then adds cache labels, both under the key's lock.
    Without that shared function and its input identity, context reuse is off.
    ``base`` is ``base_ref`` already inspected (the agent's or the proxy's
    layer under this one).
    """
    started = time.monotonic()
    await _require_cache_platform(layers, compose_env, compose_dir or context)
    identity = ""
    if context is not None:
        if build_context is None or build_identity is None:
            raise LayerError(
                "verifier context caching needs the backend's shared Compose build "
                "and build identity; use the ordinary build path"
            )
        identity = await build_identity()
        if not identity:
            raise LayerError("the shared verifier build identity is unavailable")
        base = await _context_base(layers, context, pull=pull)
    else:
        if base_ref is None:
            raise LayerError("a verifier layer needs a base image or a build context")
        info = await layers._inspect(base.image_id if base else base_ref)
        _require_image_platform(info)
        if base is None:
            assert info is not None
            base = await layers.inspect_base(str(info["Id"]))
    key = verifier_layer_key(recipe, base, build_identity=identity)
    tag = layer_tag(VERIFIER, key)

    async def build() -> str:
        if context is None:
            return await _build_layer(layers, recipe, base, key, tag, label_base=base)
        assert build_context is not None
        built = await build_context()
        return await _build_layer(layers, recipe, built, key, tag, label_base=base)

    result = await layers._ensure(
        VERIFIER,
        key,
        base,
        build,
        started=started,
        alias=alias,
        prune=True,
        keep=verifier_layer_keep(),
        max_bytes=verifier_layer_max_bytes(),
        timeout_sec=recipe.timeout_sec
        + (_CONTEXT_BUILD_TIMEOUT_SEC if context is not None else 0),
    )
    return VerifierLayer(result=result, recipe=recipe)


async def _build_layer(
    layers: DockerLayers,
    recipe: VerifierLayerRecipe,
    base: BaseImage,
    key: str,
    tag: str,
    *,
    label_base: BaseImage,
    base_from: str | None = None,
) -> str:
    """Publish labels on the exact image; never run image code to fill a cache.

    Returns the id of the image the build made (its ``--iidfile``): only an
    id a BenchFlow build recorded is reused (``env_layers.trusted_layer_ids``).
    """
    info = await layers._inspect(base.image_id)
    _require_image_platform(info)
    if info is None or (info.get("Config") or {}).get("OnBuild"):
        raise LayerError(
            "the verifier image cache needs an image without ONBUILD; "
            "use the ordinary image"
        )
    pinned = base_from is None
    base_tag = base_from or await layers._pin_base(base, key, VERIFIER)
    try:
        dockerfile = verifier_layer_dockerfile(recipe, base_tag, key, base=base)
        labels = {
            **DockerLayers._common_labels(VERIFIER, key, label_base),
            LAYER_VERIFIER_LABEL: recipe.task,
        }
        fd, iidfile = tempfile.mkstemp(prefix="benchflow-layer-", suffix=".iid")
        os.close(fd)
        image_id = ""
        try:
            args = ["build", "--platform", _CACHE_PLATFORM, "--iidfile", iidfile]
            args += ["--tag", tag]
            for name, value in labels.items():
                args += ["--label", f"{name}={value}"]
            run = await layers._docker(
                *args, "-", stdin=dockerfile.encode(), timeout=float(recipe.timeout_sec)
            )
            # Only a build that succeeded names an image; a failed one may
            # have deleted the file (env_layers.DockerLayers._build).
            if run.code == 0:
                with (
                    contextlib.suppress(OSError),
                    open(iidfile, encoding="utf-8") as fh,
                ):
                    image_id = fh.read().strip()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(iidfile)
    finally:
        if pinned:
            removed = await layers._docker("image", "rm", "--no-prune", base_tag)
            del removed
    if run.code != 0:
        tail = "\n".join(run.text.splitlines()[-env_layers._LOG_TAIL_LINES :])
        raise LayerError(
            f"building the verifier layer of {recipe.task} failed:\n{tail}"
        )
    _require_image_platform(await layers._inspect(tag))
    return image_id or tag
