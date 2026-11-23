"""Final Compose overrides that task topology cannot replace."""

from __future__ import annotations

import json
import os
import posixpath
import re
from base64 import b32encode
from collections.abc import Awaitable, Callable
from hashlib import sha256
from typing import Any

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._compose_teardown import image_ref_identity

#: The one spelling BenchFlow gives every service that can run agent code.
NO_NEW_PRIVILEGES = "no-new-privileges:true"


# The oldest Docker Engine and Compose BenchFlow supports in full (
#: the lead's ruling R7). Below them a trial under a network policy is refused
#: before anything starts (service_cover.start_sandbox), `bench doctor` warns,
#: and build-only services and ``models:`` do not resolve; open-network trials
#: run.
DOCKER_MINIMUM_ENGINE = (29, 0, 0)
DOCKER_MINIMUM_COMPOSE = (2, 40, 0)
DOCKER_MINIMUM_TEXT = "Docker Engine 29 and Docker Compose 2.40"
_VERSION = re.compile(r"\s*v?(\d+)\.(\d+)(?:\.(\d+))?")
#: Each daemon's versions, asked once per process (by the CLI flags and
#: DOCKER_HOST/DOCKER_CONTEXT that address it).
_DOCKER_VERSIONS: dict[tuple[str, ...], tuple[str | None, str | None]] = {}


def _version_tuple(text: str | None) -> tuple[int, int, int] | None:
    found = _VERSION.match(text or "")
    if found is None:
        return None
    major, minor, patch = found.groups()
    return int(major), int(minor), int(patch or 0)


def docker_below_minimum(engine: str | None, compose: str | None) -> str | None:
    """Why *engine* and *compose* are below :data:`DOCKER_MINIMUM_TEXT`, or
    None when neither is (a version that cannot be read is not taken as
    older)."""
    old_engine = (v := _version_tuple(engine)) is not None and v < DOCKER_MINIMUM_ENGINE
    old_compose = (
        v := _version_tuple(compose)
    ) is not None and v < DOCKER_MINIMUM_COMPOSE
    if not (old_engine or old_compose):
        return None
    return (
        f"Docker Engine {engine or 'unknown'} with Compose {compose or 'unknown'} "
        f"is older than {DOCKER_MINIMUM_TEXT}, the oldest BenchFlow supports in full"
    )


async def docker_versions(sandbox: Any) -> tuple[str | None, str | None] | None:
    """The Engine and Compose versions of the daemon *sandbox* starts on, once
    per process and daemon; None for a sandbox that does not run on Docker."""
    run = getattr(sandbox, "_run_docker_process", None)
    flags = getattr(sandbox, "_docker_compose_global_args", None)
    if not callable(run) or not callable(flags):
        return None
    args = [str(a) for a in flags()]
    key = (
        *args,
        os.environ.get("DOCKER_HOST", ""),
        os.environ.get("DOCKER_CONTEXT", ""),
    )
    if key in _DOCKER_VERSIONS:
        return _DOCKER_VERSIONS[key]

    async def ask(argv: list[str], what: str) -> str | None:
        try:
            result = await run(
                argv, check=False, timeout_sec=30, what=what, separate_stderr=True
            )
        except Exception:
            return None
        out = (getattr(result, "stdout", "") or "").strip()
        if getattr(result, "return_code", 1) != 0 or not out:
            return None
        return out.splitlines()[0].strip()

    engine = await ask(
        ["docker", *args, "version", "--format", "{{.Server.Version}}"],
        "Docker daemon version",
    )
    compose = await ask(
        ["docker", *args, "compose", "version", "--short"], "Docker Compose version"
    )
    _DOCKER_VERSIONS[key] = (engine, compose)
    return engine, compose


async def old_docker_policy_refusal(sandbox: Any, policy: str) -> str | None:
    """Why a trial under the network *policy* is refused before anything
    starts on *sandbox*'s daemon, or None. On Docker Engine 28 / Compose 2.38
    main's image ENTRYPOINT ran before the policy held, and a connection a
    process of it opened as root stayed open for the whole trial (
    gap 3): BenchFlow refuses rather than run the trial outside its policy."""
    versions = await docker_versions(sandbox)
    if versions is None:
        return None
    old = docker_below_minimum(*versions)
    if old is None:
        return None
    return (
        f"sandbox startup refused: {old}. On it the task's network policy "
        f"({policy}) does not hold main's image ENTRYPOINT from its first "
        "instruction: a process it starts as root can open a connection before "
        "the policy is in force and keep it for the whole trial. Upgrade Docker "
        "(Docker's apt repository: docker-ce and docker-compose-plugin; on "
        "GitHub Actions' ubuntu-24.04 runners, install them before running "
        "BenchFlow); a trial with an open network runs on this Docker"
    )


def with_no_new_privileges(options: Any) -> list[Any]:
    """A service's resolved ``security_opt`` with no-new-privileges set once.

    Every spelling of the option (``no-new-privileges``, ``:true``, ``=true``,
    ``:false``...) is replaced by :data:`NO_NEW_PRIVILEGES`; every other
    option is kept, in order. BenchFlow's base files no longer add the option
    before the task's own files (Compose 2.40 refuses the merged list when a
    task sets the same spelling: "security_opt items at 0 and 1 are equal"),
    so it is set here, on the project Compose resolved.
    """
    return [
        option
        for option in (options if isinstance(options, list) else [])
        if not (
            isinstance(option, str)
            and option.split("=", 1)[0].split(":", 1)[0] == "no-new-privileges"
        )
    ] + [NO_NEW_PRIVILEGES]


async def image_mount_daemon_version(
    config: dict[str, Any], query: Callable[[], Awaitable[ExecResult]]
) -> str | None:
    """Read the executing server, not the client or a configured image tag.

    Only image mounts need this probe. A failed or unknown version uses the
    stricter rule in the planner; diagnostics must never become version data.
    The backend bounds the query and uses the same transport as Compose.
    """
    if not any(
        isinstance(mount, dict) and mount.get("type") == "image"
        for spec in config["services"].values()
        for mount in spec.get("volumes", [])
    ):
        return None
    try:
        result = await query()
    except Exception:
        # SDK transports have their own exception types. A failed probe is
        # unknown, not evidence for the looser rule; cancellation propagates.
        return None
    return (result.stdout or "").strip() if result.return_code == 0 else None


def _image_mount_layer_name(source: str, target: str, version: str | None) -> str:
    """Match Moby's versioned filesystem component construction.

    Docker 28 uses container ID plus source, without hex encoding or target.
    Docker 29 hex-encodes ID, source and the parser's cleaned destination.
    Unknown versions conservatively use that stricter representation.
    """
    if version and re.fullmatch(r"28\.\d+\.\d+(?:[-+][A-Za-z0-9.+-]+)?", version):
        return "0" * 64 + "-" + source
    # Go filepath.Clean also collapses a leading double slash on Linux.
    target = posixpath.normpath(target)
    if target.startswith("//"):
        target = "/" + target.lstrip("/")
    return ("0" * 64 + ",src=" + source + ",dst=" + target).encode().hex()


def private_image_repository(identity: str) -> str:
    """Bound private names without truncating trial or service identity.

    128 digest bits in base32 leave room for image-mount destinations on
    daemons that hex-encode container ID, image reference and destination
    into a single filesystem component. The prefix makes this a repository,
    never an all-hex image ID. Exact references still govern cleanup.
    """
    digest = sha256(identity.encode()).digest()[:16]
    return "b" + b32encode(digest).decode().lower().rstrip("=")


def prebuilt_main_override(image: str) -> str:
    """Keep a selected prebuilt main out of all explicit/implicit builds.

    A null in an ordinary Compose merge retains the task's build mapping.
    The reset tag removes it, including additional output tags, by construction.
    """
    return (
        "services:\n  main:\n"
        f"    image: {json.dumps(image)}\n"
        "    build: !reset null\n"
        "    pull_policy: missing\n"
    )


async def implicit_build_aliases(
    config: dict[str, Any],
    *,
    prebuilt: bool,
    compose: Callable[..., Awaitable[ExecResult]],
) -> dict[str, list[str]]:
    """Obtain implicit output names from Compose, under its execution env.

    A build-only service's name depends on Compose's compatibility settings.
    The printed bake plan supplies exact per-service tags without building
    or consulting stale local images. Older Compose versions that cannot
    describe this plan must refuse these builds rather than guess a name.
    """
    implicit = {
        name
        for name, spec in config["services"].items()
        if spec.get("build")
        and not spec.get("image")
        and not (name == "main" and prebuilt)
    }
    if not implicit:
        return {}
    # Explicit selection also describes services in inactive profiles.
    result = await compose(
        ["build", "--print", "--", *sorted(implicit)], timeout_sec=30
    )
    if result.return_code != 0:
        raise ValueError(
            f"Compose could not describe implicit build outputs{_BUILD_ONLY_NOTE}"
        )
    targets = json.loads(result.stdout or "{}").get("target")
    if not isinstance(targets, dict):
        raise ValueError(f"Compose did not describe build targets{_BUILD_ONLY_NOTE}")
    aliases: dict[str, list[str]] = {}
    for target in targets.values():
        service = target.get("labels", {}).get("com.docker.compose.service")
        if service not in implicit:
            continue
        tags = target.get("tags")
        if (
            service in aliases
            or not isinstance(tags, list)
            or not tags
            or not all(isinstance(tag, str) and tag for tag in tags)
        ):
            raise ValueError("Compose did not describe unique build outputs")
        aliases[service] = tags
    if aliases.keys() != implicit:
        raise ValueError(f"Compose omitted implicit build outputs{_BUILD_ONLY_NOTE}")
    return aliases


# Compose 2.38 does not describe a build-only service's output (gap 1).
_BUILD_ONLY_NOTE = (
    f" (a service with a build and no image needs {DOCKER_MINIMUM_TEXT} or newer:"
    " `docker compose version`)"
)


def private_build_override(
    config: dict[str, Any],
    main_image: str,
    *,
    prebuilt: bool,
    implicit_aliases: dict[str, list[str]] | None = None,
    daemon_version: str | None = None,
    agent_security: bool = False,
) -> tuple[str, set[str]]:
    """Keep producers and their image consumers on the same private output.

    Reset additional tags instead of merging their spelling variants, and
    forbid up from pulling a replacement over the build we just completed.
    Normalize before looking up consumers, including aliases declared in
    build.tags. Multiple producers can build independently, but an image-only
    consumer of their shared alias has no deterministic producer: refuse it.
    The selected prebuilt main is never a consumer of a sidecar's build.
    """
    services = config["services"]
    outputs: dict[str, str] = {}
    aliases: dict[str, set[str]] = {}
    for service, spec in services.items():
        if not spec.get("build") or (service == "main" and prebuilt):
            continue
        repository = (
            main_image
            if service == "main"
            else private_image_repository(json.dumps([main_image, service]))
        )
        ref = f"{repository}:latest"
        outputs[service] = ref
        original = spec.get("image")
        originals = [original] if original else (implicit_aliases or {}).get(service)
        if not originals:
            raise ValueError("Compose build output has not been resolved")
        build = spec["build"]
        tags = build.get("tags", []) if isinstance(build, dict) else []
        for alias in originals + tags:
            aliases.setdefault(image_ref_identity(alias), set()).add(ref)

    def consumer_ref(original: str, consumer: str) -> str | None:
        candidates = aliases.get(image_ref_identity(original), set())
        if len(candidates) > 1:
            raise ValueError(
                f"Compose image consumer {consumer!r} has multiple build producers"
            )
        return next(iter(candidates), None)

    lines = ["services:"]
    for service, spec in services.items():
        ref = outputs.get(service)
        if ref is None and not (service == "main" and prebuilt):
            original = spec.get("image")
            ref = consumer_ref(original, service) if original else None
        mounts = []
        changed_mount = False
        for mount in spec.get("volumes", []):
            source = (
                consumer_ref(mount["source"], service)
                if isinstance(mount, dict) and mount.get("type") == "image"
                else None
            )
            if source:
                layer_name = _image_mount_layer_name(
                    source, mount["target"], daemon_version
                )
                if len(layer_name.encode()) > 255:
                    known = daemon_version and re.fullmatch(
                        r"(?:28|29)\.\d+\.\d+(?:[-+][A-Za-z0-9.+-]+)?", daemon_version
                    )
                    fallback = (
                        ""
                        if known
                        else "; daemon version unavailable or unrecognized, "
                        "applying the stricter Docker 29 bound"
                    )
                    raise ValueError(
                        f"Compose image mount on {service!r} exceeds Docker's "
                        "255-byte layer name limit with a private image; "
                        f"shorten the mount target{fallback}"
                    )
            mounts.append({**mount, "source": source} if source else mount)
            changed_mount |= source is not None
        if ref is None and not changed_mount and not agent_security:
            continue
        lines.append(f"  {json.dumps(service)}:")
        if agent_security:
            # The final resolved inventory is authoritative, including
            # includes, extends and profiles. Any service can run agent code.
            # Replace this option by identity, retaining all other settings.
            # The backends stack this file after every other BenchFlow overlay
            # that sets security_opt (the kept-service freezer's), so no later
            # file appends an option this list already holds.
            security = with_no_new_privileges(spec.get("security_opt", []))
            lines.append(f"    security_opt: !override {json.dumps(security)}")
        if ref is not None:
            lines.append(f"    image: {json.dumps(ref)}")
            if service in outputs:
                lines.extend(["    build:", "      tags: !reset []"])
            lines.append("    pull_policy: never")
        if changed_mount:
            # The resolved sequence is authoritative: do not retain an old
            # image source through Compose's target-based mount merging.
            # Values were already interpolated; preserve literal dollars.
            literal = json.dumps(mounts).replace("$", "$$")
            lines.append(f"    volumes: !override {literal}")
    return "\n".join(lines) + "\n", set(outputs.values())
