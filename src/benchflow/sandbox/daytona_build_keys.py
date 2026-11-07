"""Daytona's build keys, and how BenchFlow changes one.

A sandbox created from a recipe (``Image.from_dockerfile`` or ``Image.base``,
plus the install layers of ``daytona_layers``) sends Daytona the recipe's
Dockerfile text and one hash per COPY source, which the SDK uploads as a
context. Daytona builds the recipe once and keeps the build under
``'daytona-' + sha256(dockerfile + ''.join(sorted(context_hashes)))``; a
later create of the same key reuses it.

The SDK hashes a directory source in ``os.walk`` order, which is the
filesystem's own listing order (ext4 and APFS differ), so one task got one
key on one machine and another elsewhere, and was built again on each.
:func:`context_hash` is the SDK's hash with the walk sorted
(:func:`sorted_walk`); ``_sdk_ops`` installs it over the SDK's own.

A key can go bad for good: a build cancelled by a delete stays failed under
its key, and a runner that holds a built image and never starts it gets
every create of that key (``daytona_start.rebuild_reason``).
:func:`salted_params` gives the same image a new key: the Dockerfile with
one trailing comment line, which builds nothing. Salt *n* is the same text
on every run, so a later run reuses the salted build.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator
from typing import Any

#: Salts tried in order after a create of the plain recipe fails for good.
BUILD_SALTS = 3
SALT_LINE = "# benchflow: Daytona rebuild {salt}"
_READ_SIZE = 8192


def sorted_walk(top: str) -> Iterator[tuple[str, list[str], list[str]]]:
    """``os.walk(top)`` with each folder's subfolders and files in sorted
    order, the same on every filesystem.

    The walk is top-down and descends into ``dirs`` after each yield, so
    sorting the list in place orders the descent too, also for a caller
    that collects the whole walk first.
    """
    for root, dirs, files in os.walk(top):
        dirs.sort()
        files.sort()
        yield root, dirs, files


def _update_from_file(hasher: Any, path: str) -> None:
    with open(path, "rb") as source:
        while chunk := source.read(_READ_SIZE):
            hasher.update(chunk)


def context_hash(
    path_str: str,
    archive_base_path: str | None,
    base_path: Callable[[str], str],
) -> str:
    """The SDK's ``ObjectStorage._compute_hash_for_path_md5`` with the walk
    sorted.

    Line for line the SDK's function in daytona 0.207 (*base_path* is its
    ``compute_archive_base_path``): md5 of the archive path, then a file's
    bytes, or for a folder the relative path of each empty folder and the
    relative path and bytes of each file, in walk order. Only the order is
    :func:`sorted_walk`'s: a file's hash is the SDK's, and a folder's is the
    SDK's on a filesystem that lists sorted. This is Daytona's cache key,
    not a secret.
    """
    md5_hasher = hashlib.md5(usedforsecurity=False)
    abs_path_str = os.path.abspath(path_str)

    if archive_base_path is None:
        archive_base_path = base_path(path_str)
    md5_hasher.update(archive_base_path.encode("utf-8"))

    if os.path.isfile(abs_path_str):
        _update_from_file(md5_hasher, abs_path_str)
    else:
        for root, dirs, files in sorted_walk(abs_path_str):
            if not dirs and not files:
                rel_dir = os.path.relpath(root, path_str)
                md5_hasher.update(rel_dir.encode("utf-8"))
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, abs_path_str)
                md5_hasher.update(rel_path.encode("utf-8"))
                _update_from_file(md5_hasher, file_path)

    return md5_hasher.hexdigest()


def salted_dockerfile(dockerfile: str, salt: int) -> str:
    """*dockerfile* with one trailing comment line naming *salt*."""
    if dockerfile and not dockerfile.endswith("\n"):
        dockerfile += "\n"
    return dockerfile + SALT_LINE.format(salt=salt) + "\n"


def salted_params(params: Any, salt: int) -> Any | None:
    """A copy of the create *params* whose image's Dockerfile ends with
    *salt*, or None when they carry no image to salt (a snapshot create).

    The salt goes after everything else, install layers included, since the
    text the SDK sends is what Daytona keys. Neither *params* nor its image
    changes: the plain recipe stays the start's fallback and its retries'.
    """
    image = getattr(params, "image", None)
    dockerfile = getattr(image, "dockerfile", None)
    if (
        not callable(dockerfile)
        or not hasattr(image, "model_copy")
        or not hasattr(params, "model_copy")
    ):
        return None
    salted = image.model_copy(deep=True)
    salted._dockerfile = salted_dockerfile(dockerfile(), salt)
    return params.model_copy(update={"image": salted})
