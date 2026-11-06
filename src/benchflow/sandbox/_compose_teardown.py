"""How every Compose sandbox (Docker, remote Docker, Daytona DinD) tears down.

Teardown runs after verification and copy-out, so nothing inside a trial's
containers needs a graceful stop. ``compose down -t 5`` sent SIGTERM to a
``sleep infinity`` PID 1, which ignores it, waited the 5 s out, then sent
SIGKILL, one service after another: 5 s per container on every trial, 10 s
for a task with a sidecar (engine bench, 2026-10-16). Instead:

1. ``compose kill`` sends SIGKILL to every container of the project at once,
   in parallel;
2. ``compose down -t 0`` removes the containers and the network. Named
   volumes are removed only by their project ownership label, never by
   ``down --volumes`` (Compose adopts pre-existing named volumes without
   taking ownership of them). Docker makes no anonymous volume for a trial:
   each one is named and carries the project's label from its creation
   (:mod:`benchflow.sandbox._trial_volumes`), so it goes by that label too.
   Never ``rm --volumes``, which also removes a volume a container inherited
   through ``volumes_from``.

``down`` is never given ``--rmi``: ``--rmi all`` also deletes images the task
only pulls (the operator's own ``ubuntu:24.04``), and with
``--remove-orphans`` either mode removes the *first* tag of each image the
project built, which can be a verifier-recovery lease instead of the build's
own tag. The Docker backends remove the images they built themselves
(:meth:`benchflow.sandbox.docker.DockerSandbox._remove_built_images`).
"""

from __future__ import annotations

import json

# SIGKILL to every container of the project, all at once.
COMPOSE_KILL_COMMAND: tuple[str, ...] = ("kill",)

# The verifier-recovery lease's repository (_recovery_baseline.capture_baseline
# tags the main container's image with it). The rollout releases the lease
# after scoring; teardown never takes it, even when it is the only tag left on
# a build (another trial of the same environment built ``bf__<env>`` later and
# took that tag).
RECOVERY_LEASE_REPOSITORY = "benchflow-recovery-lease"


def compose_down_command() -> list[str]:
    """The ``compose down`` arguments teardown runs after :data:`COMPOSE_KILL_COMMAND`.

    ``-t 0``: a container ``kill`` missed (one started in between) is killed
    at once, without the SIGTERM grace. Never remove named volumes here:
    their declared names may resolve to pre-existing operator data.
    """
    return ["down", "--remove-orphans", "-t", "0"]


def image_repository(ref: str) -> str:
    """``ref`` without its ``:tag`` (``host:5000/a/b:1`` -> ``host:5000/a/b``)."""
    head, sep, last = ref.rpartition("/")
    name = last.split("@", 1)[0].split(":", 1)[0]
    return f"{head}{sep}{name}"


def image_ref_identity(ref: str) -> str:
    """Docker's familiar identity for a reference accepted by Compose/the daemon.

    Follow distribution/reference's splitDockerDomain and ParseDockerRef:
    default the registry, namespace and tag, map the legacy Hub domain, and
    keep other registries/namespaces distinct. A digest supersedes a tag.
    This is an identity comparison, not validation of arbitrary user input.
    """
    name, at, digest = ref.partition("@")
    if at:
        name = image_repository(name)
    elif ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    first, slash, rest = name.partition("/")
    if not slash:
        domain, path = "docker.io", f"library/{name}"
    elif first == "index.docker.io":
        domain, path = "docker.io", rest
    elif first == "localhost" or "." in first or ":" in first or first.lower() != first:
        domain, path = first, rest
    else:
        domain, path = "docker.io", name
    if domain == "docker.io":
        if "/" not in path:
            path = f"library/{path}"
        # Only the single-component official namespace has a short spelling.
        if path.startswith("library/") and "/" not in path[len("library/") :]:
            path = path[len("library/") :]
        identity = path
    else:
        identity = f"{domain}/{path}"
    return f"{identity}@{digest}" if at else identity


def built_image_refs(listing: str, *, build_tags: set[str]) -> list[str]:
    """IDs eligible for cleanup through a private alias, never deletion by ID.

    ``listing`` is the JSON from inspecting the project's full image IDs.
    RepoDigests retain images too: ``image ls``'s repo:<none> row does not
    mean a build is unreferenced. Unknown inventories conservatively stay.
    Keep foreign tags and leases, including an image's sole remaining tag.
    Even an eligible image can gain a reference after this inventory; the
    caller pins its ID with a random alias and removes only that alias with
    ``--no-prune``. Shared build names stay while they still name the image.
    """
    try:
        images = json.loads(listing)
    except (ValueError, TypeError):
        return []
    if not isinstance(images, list):
        return []
    build_tags = {image_ref_identity(tag) for tag in build_tags}
    refs: list[str] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        image_id = image.get("Id")
        tags, digests = image.get("RepoTags"), image.get("RepoDigests")
        if (
            not isinstance(image_id, str)
            or not image_id
            or "RepoTags" not in image
            or "RepoDigests" not in image
            or (tags is not None and not isinstance(tags, list))
            or (digests is not None and not isinstance(digests, list))
            or digests
        ):
            continue
        tags = tags or []
        if not all(isinstance(tag, str) for tag in tags):
            continue
        identities = [image_ref_identity(tag) for tag in tags]
        leased = any(
            image_repository(tag) == RECOVERY_LEASE_REPOSITORY for tag in identities
        )
        if not tags or (len(tags) == 1 and not leased and identities[0] in build_tags):
            refs.append(image_id)
    return refs
