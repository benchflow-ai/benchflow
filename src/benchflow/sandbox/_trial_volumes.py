"""Every volume a trial makes is named and labelled at creation (DOCKER-ANON-VOLUME-LEAK).

Docker gives a container an anonymous volume for each path its image
declares ``VOLUME`` and for each Compose ``volumes: [/path]`` entry. Such a
volume has no owner a later inspect can prove (round 1 removed an
operator's volume inherited through ``volumes_from``; round 2 kept the
trial's own). So BenchFlow never lets Docker make one for a trial: before
Compose creates a container, :func:`overlay` gives each of those paths a
named volume carrying :data:`LABEL` (and Compose's project label), at the
same target. Docker fills an empty named volume from the image path as it
fills an anonymous one, so the container sees the same files, owner and
mode. Teardown removes the volumes with the project's label, and nothing
else.

A path the service already mounts (any ``volumes`` entry, a ``tmpfs``, a
file secret or config) gets no volume from Docker, and neither does a path a
``volumes_from`` source passes on. So those paths are left as they are, and
the sharing stays the same.

What a ``volumes_from`` source passes on is its mount points, and only those
(Docker copies the source's ``MountPoints``): its volume, bind and image
mounts, the binds Compose makes for its file secrets and configs, the
volumes Docker gives its image's ``VOLUME`` paths, and what its own sources
pass on. A ``tmpfs`` is not a mount point (``tmpfs:`` is
``HostConfig.Tmpfs``), so it is not passed on, and an image ``VOLUME`` path
under the source's own ``tmpfs`` gets no volume to pass on either. (A source
with a ``type: tmpfs`` entry makes Docker refuse the inheritor's create.)
"""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from typing import Any

#: On every volume BenchFlow names for a trial; its value is the project.
LABEL = "io.benchflow.trial-volume"


class Unreadable(Exception):
    """A path's source cannot be read, so the overlay cannot be made."""


def _target(path: Any) -> str | None:
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    return posixpath.normpath(path)


def _tmpfs_targets(spec: Mapping[str, Any]) -> set[str]:
    value = spec.get("tmpfs") or ()
    if isinstance(value, str):
        value = [value]
    return {t for entry in value if (t := _target(str(entry).split(":")[0]))}


def _secret_targets(
    spec: Mapping[str, Any], defined: Mapping[str, Any], key: str, base: str
) -> set[str]:
    """Where Compose bind-mounts the service's file-backed ``secrets`` or
    ``configs`` (*key*): an absolute target as given, a relative one or the
    source's name under *base*. Content and environment ones are copied in,
    not mounted."""
    targets = set()
    for entry in spec.get(key) or ():
        if isinstance(entry, str):
            entry = {"source": entry}
        if not isinstance(entry, Mapping):
            continue
        source = str(entry.get("source") or "")
        if not (defined.get(source) or {}).get("file"):
            continue
        target = str(entry.get("target") or source)
        if t := _target(target if target.startswith("/") else base + target):
            targets.add(t)
    return targets


def volumes_from(spec: Mapping[str, Any]) -> list[tuple[str, bool]]:
    """A service's ``volumes_from`` sources: (name, outside the project)."""
    sources = []
    for entry in spec.get("volumes_from") or ():
        text = str(entry)
        if text.startswith("container:"):
            sources.append((text.removeprefix("container:").split(":")[0], True))
        else:
            sources.append((text.split(":")[0], False))
    return sources


def image_of(project: Mapping[str, Any], service: str) -> str:
    """The image Compose runs *service* from (its default name for a build
    without ``image``)."""
    spec = project["services"][service]
    return str(spec.get("image") or f"{project.get('name')}-{service}")


def overlay(
    project: Mapping[str, Any],
    image_volumes: Mapping[str, list[str]],
    outside: Mapping[str, list[str]],
    generation: int,
) -> dict[str, Any]:
    """The Compose overlay that names every volume Docker would make
    anonymous for *project*'s services (an empty dict when there is none).

    *image_volumes* maps each service's image (:func:`image_of`) to the paths
    it declares ``VOLUME``; *outside* maps each external ``volumes_from``
    container to its mount destinations, read before start. A missing entry
    raises :class:`Unreadable`. *generation* makes the names new for each
    start, as anonymous volumes were."""
    name = str(project.get("name") or "")
    services: Mapping[str, Any] = project.get("services") or {}
    secrets: Mapping[str, Any] = project.get("secrets") or {}
    configs: Mapping[str, Any] = project.get("configs") or {}
    gives: dict[str, set[str]] = {}

    def given(service: str, seen: frozenset[str] = frozenset()) -> set[str]:
        """The mount points of *service*'s containers: what a peer copies
        through ``volumes_from``. Never a ``tmpfs``."""
        if service in gives:
            return gives[service]
        if service in seen:
            raise Unreadable(f"volumes_from of {service!r} is a cycle")
        spec = services[service]
        passed = inherited(spec, seen | {service})
        paths = mounted(spec) | passed
        paths |= declared(service) - (paths | tmpfs(spec))
        gives[service] = paths
        return paths

    def declared(service: str) -> set[str]:
        image = image_of(project, service)
        if image not in image_volumes:
            raise Unreadable(f"the VOLUMEs of {image!r} were not read")
        return {t for p in image_volumes[image] if (t := _target(p))}

    def mounted(spec: Mapping[str, Any]) -> set[str]:
        """The service's own mount points: every ``volumes`` entry but a
        ``tmpfs`` one, and its file secrets and configs."""
        return (
            {
                t
                for mount in spec.get("volumes") or ()
                if isinstance(mount, dict)
                and mount.get("type") != "tmpfs"
                and (t := _target(mount.get("target")))
            }
            | _secret_targets(spec, secrets, "secrets", "/run/secrets/")
            | _secret_targets(spec, configs, "configs", "/")
        )

    def tmpfs(spec: Mapping[str, Any]) -> set[str]:
        return {
            t
            for mount in spec.get("volumes") or ()
            if isinstance(mount, dict)
            and mount.get("type") == "tmpfs"
            and (t := _target(mount.get("target")))
        } | _tmpfs_targets(spec)

    def inherited(spec: Mapping[str, Any], seen: frozenset[str]) -> set[str]:
        paths: set[str] = set()
        for source, external in volumes_from(spec):
            if external:
                if source not in outside:
                    raise Unreadable(f"volumes_from container {source!r} was not read")
                paths |= {t for p in outside[source] if (t := _target(p))}
            elif source in services:
                paths |= given(source, seen)
            else:
                raise Unreadable(f"volumes_from names no service {source!r}")
        return paths

    out_services: dict[str, Any] = {}
    out_volumes: dict[str, Any] = {}
    for service in sorted(services):
        spec = services[service]
        mounts: list[dict[str, Any]] = []
        for mount in spec.get("volumes") or ():
            if (
                isinstance(mount, dict)
                and mount.get("type") == "volume"
                and not mount.get("source")
                and _target(mount.get("target"))
            ):
                mounts.append({**mount})
        covered = mounted(spec) | tmpfs(spec) | inherited(spec, frozenset({service}))
        for path in sorted(declared(service)):
            if path not in covered:
                mounts.append({"type": "volume", "target": path})
        if not mounts:
            continue
        for index, mount in enumerate(mounts):
            key = f"bfvol{generation}_{service}_{index}"
            if key in (project.get("volumes") or {}):
                raise Unreadable(f"the task declares a volume {key!r} itself")
            mount["source"] = key
            out_volumes[key] = {"labels": {LABEL: name}}
        out_services[service] = {"volumes": mounts}
    if not out_services:
        return {}
    return {"services": out_services, "volumes": out_volumes}
