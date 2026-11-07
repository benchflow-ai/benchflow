"""Final Compose overrides that task topology cannot replace."""

from __future__ import annotations

import json
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
        raise ValueError("Compose could not describe implicit build outputs")
    targets = json.loads(result.stdout or "{}").get("target")
    if not isinstance(targets, dict):
        raise ValueError("Compose did not describe build targets")
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
        raise ValueError("Compose omitted implicit build outputs")
    return aliases


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
