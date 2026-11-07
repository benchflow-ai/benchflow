"""Guards the evidence-transfer defects reproduced on ac3bedaa1."""

import io
import os
import tarfile

import pytest

from benchflow.review.evidence import (
    EvidenceError,
    EvidenceLimitError,
    _bundle_from_tar,
    capture_workspace,
)
from tests.test_capture_bfx import BfxSandbox
from tests.test_evidence_without_python import LocalSandbox, _entries


async def test_excluded_cache_is_pruned_before_transfer(tmp_path):
    """Guards ac3bedaa1: exclusion applies before bytes leave the sandbox."""
    source = tmp_path / "source"
    cache = source / ".cache" / "pip"
    cache.mkdir(parents=True)
    (cache / "large").write_bytes(b"x" * (1024 * 1024))
    (source / "answer").write_text("42\n")
    transferred = []

    class Recorded(BfxSandbox):
        async def bfx_tar_out(self, args, destination, *, timeout_sec):
            report = await super().bfx_tar_out(
                args, destination, timeout_sec=timeout_sec
            )
            transferred.append(destination.stat().st_size)
            return report

    manifest = await capture_workspace(
        Recorded(), str(source), tmp_path / "bundle", max_bytes=3
    )
    assert transferred[0] < 20_000
    assert {e.path for e in manifest.entries} == {".cache", "answer"}
    assert [e.reason for e in manifest.exclusions] == ["package_cache"]


@pytest.mark.parametrize("limit", [{"max_bytes": 2}, {"max_entries": 1}])
async def test_limits_fail_before_transfer(tmp_path, limit):
    """Guards ac3bedaa1: files and special exclusions share the same budget."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer").write_text("42\n")
    os.mkfifo(source / "pipe")
    for index, box in enumerate([LocalSandbox(python=True), BfxSandbox()]):
        with pytest.raises(EvidenceLimitError):
            await capture_workspace(box, str(source), tmp_path / str(index), **limit)
        assert not (tmp_path / str(index)).exists()
        if isinstance(box, BfxSandbox):
            assert not box.tar_outs


async def test_exclusions_links_and_empty_directories_match_python(tmp_path):
    """Guards ac3bedaa1: selection retains the original evidence contract."""
    source = tmp_path / "source"
    (source / "empty").mkdir(parents=True)
    (source / "answer").write_text("42\n")
    (source / "discard.tmp").write_text("excluded\n")
    (source / "link").symlink_to("answer")
    (source / "escape").symlink_to("/etc/hostname")
    os.mkfifo(source / "pipe")
    manifests = [
        await capture_workspace(
            box, str(source), tmp_path / str(index), exclude=["*.tmp"]
        )
        for index, box in enumerate([LocalSandbox(python=True), BfxSandbox()])
    ]
    assert _entries(manifests[0]) == _entries(manifests[1])
    assert sorted(e.model_dump_json() for e in manifests[0].exclusions) == sorted(
        e.model_dump_json() for e in manifests[1].exclusions
    )


def test_combined_manifest_limit_rejects_file_plus_special(tmp_path):
    """Guards ac3bedaa1 even if a transport omits special files from tar."""
    raw = tmp_path / "raw.tar"
    with tarfile.open(raw, "w") as archive:
        member = tarfile.TarInfo("answer")
        member.size = 3
        archive.addfile(member, io.BytesIO(b"42\n"))
    with pytest.raises(EvidenceLimitError):
        _bundle_from_tar(
            raw,
            tmp_path,
            tmp_path / "published",
            "/app",
            {"credentials": [], "sandbox_runtime": [], "exclude": [], "paths": []},
            digest="a" * 64,
            max_bytes=3,
            max_entries=1,
            specials=["pipe"],
        )
    assert not (tmp_path / "published").exists()


async def test_changed_inventory_is_not_published(tmp_path):
    """Guards ac3bedaa1: a disappearing selected file cannot look complete."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer").write_text("42\n")

    class Changed(BfxSandbox):
        async def bfx_tar_out(self, args, destination, *, timeout_sec):
            return await super().bfx_tar_out(
                {**args, "paths": []}, destination, timeout_sec=timeout_sec
            )

    with pytest.raises(EvidenceError, match="changed between metadata"):
        await capture_workspace(Changed(), str(source), tmp_path / "published")
    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize(
    "corruption", ["outside", "duplicate", "unlisted", "child_error"]
)
async def test_invalid_metadata_fails_before_archive(tmp_path, corruption):
    """Guards ac3bedaa1: incomplete or unsafe inventories never authorize a tar."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer").write_text("42\n")

    class Invalid(BfxSandbox):
        async def bfx_call(self, command, args, *, timeout_sec):
            report = await super().bfx_call(command, args, timeout_sec=timeout_sec)
            if corruption == "outside":
                report["entries"][-1]["path"] = "/etc/hostname"
            elif corruption == "duplicate":
                report["entries"].append(dict(report["entries"][-1]))
            elif corruption == "child_error":
                report["errors"] = [{"path": str(source / "answer"), "error": "ENOENT"}]
            else:
                report["entries"][0]["listing"] = False
            return report

    box = Invalid()
    with pytest.raises(EvidenceError):
        await capture_workspace(box, str(source), tmp_path / "published")
    assert not box.tar_outs
    assert not (tmp_path / "published").exists()
