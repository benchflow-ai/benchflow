"""Bound every expanded dimension before extracting remote artifacts."""

from __future__ import annotations

import gzip
import io
import tarfile
import tempfile
from pathlib import Path

from benchflow._utils.archive_paths import ArchivePaths


def extract_download(
    blob: bytes, target: Path, *, tar_bytes: int, content_bytes: int, members: int
) -> None:
    """Validate a bounded plain tar before publishing regular files/directories.

    Limiting the uncompressed stream first also bounds PAX/GNU metadata before
    tarfile parses it. Links, sparse files and special entries are outside this
    transfer contract; neither aliases nor repeated paths can expand the write
    set beyond the validated budget.
    """
    with tempfile.TemporaryFile() as plain:
        with gzip.GzipFile(fileobj=io.BytesIO(blob)) as compressed:
            size = 0
            while chunk := compressed.read(min(1024 * 1024, tar_bytes - size + 1)):
                size += len(chunk)
                if size > tar_bytes:
                    raise RuntimeError("AgentCore directory tar byte cap exceeded")
                plain.write(chunk)
        plain.seek(0)
        with tarfile.open(fileobj=plain, mode="r:") as archive:
            accepted: list[tarfile.TarInfo] = []
            inventory = ArchivePaths()
            expanded = 0
            for member in archive:
                if len(accepted) >= members:
                    raise RuntimeError("AgentCore directory member limit exceeded")
                try:
                    inventory.add(member.name, directory=member.isdir())
                except ValueError as exc:
                    raise RuntimeError(
                        "AgentCore directory has an unsafe tar member"
                    ) from exc
                if (
                    not (member.isfile() or member.isdir())
                    or member.issparse()
                    or member.size < 0
                ):
                    raise RuntimeError("AgentCore directory has an unsafe tar member")
                expanded += member.size
                if expanded > content_bytes:
                    raise RuntimeError("AgentCore directory expanded byte cap exceeded")
                accepted.append(member)
            archive.extractall(target, members=accepted, filter="data")
