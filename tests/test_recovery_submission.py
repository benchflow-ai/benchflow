"""GH #948: recover only the files a task declares in ``submission_files``.

The synthetic task declares three submission files under /root; one of them is
never written, so every test also covers a declared-but-missing output.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from benchflow.sandbox import _recovery_submission as submission

PATHS = ("/root/result.csv", "/root/summary.json", "/root/report.md")


def source_tree(tmp_path):
    root = tmp_path / "source"
    home = root / "root"
    home.mkdir(parents=True)
    (home / "result.csv").write_bytes(b"a,b\n1,2\n")
    (home / "summary.json").write_bytes(b'{"valid": true}')
    (home / ".credentials").write_text("must-never-copy")
    return root


def captured(tmp_path):
    root = source_tree(tmp_path)
    bundle = tmp_path / "bundle"
    digest = submission.capture_submission(bundle, paths=PATHS, filesystem_root=root)
    return root, bundle, digest


def target_tree(tmp_path):
    root = tmp_path / "target"
    home = root / "root"
    home.mkdir(parents=True)
    for path in PATHS:
        (home / Path(path).name).write_text("old")
    (home / "notes.txt").write_bytes(b"original notes")
    (home / "report_template").mkdir()
    (home / "report_template/main.md").write_text("original template")
    return root


def test_capture_restore_preserves_missing_and_unrelated_root_files(
    tmp_path, monkeypatch
):
    """GH #948: root outputs return without copying auth or replacing /root."""
    _, bundle, digest = captured(tmp_path)
    manifest = submission.validate_submission(
        bundle, paths=PATHS, expected_manifest_sha256=digest
    )
    assert [entry["status"] for entry in manifest["files"]] == [
        "present",
        "present",
        "missing",
    ]
    assert {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    } == {"manifest.json", "files/0", "files/1"}
    target = target_tree(tmp_path)
    original_replace = os.replace
    replacements = []

    def atomic_replace(source, destination, **kwargs):
        assert kwargs["src_dir_fd"] == kwargs["dst_dir_fd"]
        replacements.append(destination)
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(os, "replace", atomic_replace)
    submission.restore_submission(
        bundle, paths=PATHS, expected_manifest_sha256=digest, filesystem_root=target
    )
    assert replacements == ["result.csv", "summary.json"]
    assert (target / "root/result.csv").read_bytes() == b"a,b\n1,2\n"
    assert not (target / "root/report.md").exists()
    assert not (target / "root/.credentials").exists()
    assert (target / "root/notes.txt").read_bytes() == b"original notes"
    assert (target / "root/report_template/main.md").read_text() == "original template"
    assert not list((target / "root").glob(".benchflow-submission-*"))


@pytest.mark.parametrize("shape", ["symlink", "directory", "fifo", "parent_symlink"])
def test_capture_rejects_nonregular_and_symlink_components(tmp_path, shape):
    """GH #948: capture never follows output or parent links or opens a FIFO."""
    root = source_tree(tmp_path)
    file = root / "root/result.csv"
    file.unlink()
    if shape == "symlink":
        file.symlink_to(root / "root/.credentials")
    elif shape == "directory":
        file.mkdir()
    elif shape == "fifo":
        os.mkfifo(file)
    else:
        (root / "root").rename(root / "elsewhere")
        (root / "root").symlink_to(root / "elsewhere", target_is_directory=True)
    bundle = tmp_path / "bundle"
    with pytest.raises((OSError, submission.SubmissionError)):
        submission.capture_submission(bundle, paths=PATHS, filesystem_root=root)
    assert not (bundle / "manifest.json").exists()


@pytest.mark.parametrize("shape", ["symlink", "directory", "fifo", "parent_symlink"])
def test_restore_checks_every_target_before_any_changes(tmp_path, shape):
    """GH #948: hostile final/missing target cannot partially replace earlier files."""
    _, bundle, digest = captured(tmp_path)
    target = target_tree(tmp_path)
    file = target / "root/report.md"
    file.unlink()
    if shape == "symlink":
        file.symlink_to(target / "root/notes.txt")
    elif shape == "directory":
        file.mkdir()
    elif shape == "fifo":
        os.mkfifo(file)
    else:
        (target / "root").rename(target / "elsewhere")
        (target / "root").symlink_to(target / "elsewhere", target_is_directory=True)
    with pytest.raises((OSError, submission.SubmissionError)):
        submission.restore_submission(
            bundle, paths=PATHS, expected_manifest_sha256=digest, filesystem_root=target
        )
    assert (target / "root/result.csv").read_text() == "old"
    assert (target / "root/summary.json").read_text() == "old"
    assert (target / "root/notes.txt").read_bytes() == b"original notes"


@pytest.mark.parametrize(
    "tampering", ["blob", "blob_link", "manifest", "destination", "extra", "files_link"]
)
def test_corrupt_or_extra_bundle_data_cannot_modify_destination(tmp_path, tampering):
    """GH #948: externally pinned manifest and blob hashes fence admission."""
    _, bundle, digest = captured(tmp_path)
    target = target_tree(tmp_path)
    if tampering == "blob":
        (bundle / "files/1").write_bytes(b"bad")
    elif tampering == "blob_link":
        (bundle / "files/1").unlink()
        (bundle / "files/1").symlink_to(target / "root/summary.json")
    elif tampering == "manifest":
        (bundle / "manifest.json").write_bytes(b"{}")
    elif tampering == "destination":
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["files"][0]["path"] = "/root/.credentials"
        raw = json.dumps(manifest).encode()
        (bundle / "manifest.json").write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
    elif tampering == "extra":
        (bundle / "secret.txt").write_text("not a declared output")
    else:
        (bundle / "files").rename(bundle / "old-files")
        (bundle / "files").symlink_to(bundle / "old-files", target_is_directory=True)
    with pytest.raises((OSError, submission.SubmissionError)):
        submission.restore_submission(
            bundle, paths=PATHS, expected_manifest_sha256=digest, filesystem_root=target
        )
    assert (target / "root/result.csv").read_text() == "old"
    assert (target / "root/summary.json").read_text() == "old"
    assert (target / "root/report.md").read_text() == "old"


def test_source_mutation_during_capture_is_rejected(tmp_path, monkeypatch):
    """GH #948: a completed bundle cannot silently combine moving source bytes."""
    root = source_tree(tmp_path)
    file = root / "root/result.csv"
    inode = file.stat().st_ino
    original_read = os.read
    mutated = False

    def mutating_read(fd, count):
        nonlocal mutated
        data = original_read(fd, count)
        if not mutated and os.fstat(fd).st_ino == inode:
            mutated = True
            file.write_bytes(b"a,b\n3,4\n")
        return data

    monkeypatch.setattr(os, "read", mutating_read)
    bundle = tmp_path / "bundle"
    with pytest.raises(submission.SubmissionError, match="changed"):
        submission.capture_submission(bundle, paths=PATHS, filesystem_root=root)
    assert mutated
    assert not (bundle / "manifest.json").exists()


@pytest.mark.parametrize("limit", ["MAX_FILE_BYTES", "MAX_TOTAL_BYTES"])
def test_capture_enforces_size_limits(tmp_path, monkeypatch, limit):
    """GH #948: declared contracts cannot make recovery copy unlimited data."""
    root = source_tree(tmp_path)
    monkeypatch.setattr(submission, limit, 3 if limit == "MAX_FILE_BYTES" else 5)
    with pytest.raises(submission.SubmissionError, match="size limit"):
        submission.capture_submission(
            tmp_path / "bundle", paths=PATHS, filesystem_root=root
        )


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["/root"],
        ["root/result.csv"],
        ["/root/*.csv"],
        ["/root/out[0-9].csv"],
        ["/root/../etc/passwd"],
        ["/root/./result.csv"],
        ["/root//result.csv"],
        ["/root/result.csv/"],
        ["/tests/test.sh"],
        ["/logs/verifier/reward.txt"],
        ["/root/result.csv", "/root/result.csv"],
        [
            f"/root/out{index}.csv"
            for index in range(submission.MAX_SUBMISSION_FILES + 1)
        ],
        ["/root/" + "x" * submission.MAX_PATH_CHARS],
        ["/root/line\nbreak"],
    ],
)
def test_contract_cannot_broaden_to_home_globs_or_verifier_paths(tmp_path, paths):
    """GH #948: a contract is a short list of literal, canonical file paths."""
    with pytest.raises(submission.SubmissionError):
        submission.capture_submission(
            tmp_path / "bundle", paths=paths, filesystem_root=tmp_path
        )
    assert not (tmp_path / "bundle").exists()


def test_contract_keeps_declared_order_and_any_file_names():
    """The SDK hard-codes no file names: any valid declaration is its own contract."""
    declared = ["/srv/out/b.txt", "/root/a.json"]
    assert submission.validate_contract(declared) == tuple(declared)
    assert submission.validate_contract(list(PATHS)) == PATHS


def test_restore_rejects_a_different_declaration(tmp_path):
    """A bundle captured for one declaration cannot restore under another."""
    _, bundle, digest = captured(tmp_path)
    target = target_tree(tmp_path)
    with pytest.raises(submission.SubmissionError):
        submission.restore_submission(
            bundle,
            paths=list(reversed(PATHS)),
            expected_manifest_sha256=digest,
            filesystem_root=target,
        )
    with pytest.raises(submission.SubmissionError):
        submission.validate_submission(
            bundle, paths=PATHS[:2], expected_manifest_sha256=digest
        )
    assert (target / "root/result.csv").read_text() == "old"


def test_capture_detects_parent_replacement(tmp_path, monkeypatch):
    """GH #948: a renamed parent cannot make pinned-descriptor bytes look current."""
    root = source_tree(tmp_path)
    original_copy = submission._copy
    replaced = False

    def replace_parent(*args, **kwargs):
        nonlocal replaced
        result = original_copy(*args, **kwargs)
        if not replaced:
            replaced = True
            (root / "root").rename(root / "former_root")
            (root / "root").mkdir()
        return result

    monkeypatch.setattr(submission, "_copy", replace_parent)
    with pytest.raises(submission.SubmissionError, match="parent changed"):
        submission.capture_submission(
            tmp_path / "bundle", paths=PATHS, filesystem_root=root
        )
    assert not (tmp_path / "bundle/manifest.json").exists()


def test_restore_detects_target_change_while_staging_and_cleans_temps(
    tmp_path, monkeypatch
):
    """GH #948: mutation after preflight cannot overwrite newer destination state."""
    _, bundle, digest = captured(tmp_path)
    target = target_tree(tmp_path)
    original_copy = submission._copy

    def mutate_target(parent, name, output, **kwargs):
        result = original_copy(parent, name, output, **kwargs)
        if output is not None:
            (target / "root/report.md").write_text("newer")
        return result

    monkeypatch.setattr(submission, "_copy", mutate_target)
    with pytest.raises(submission.SubmissionError, match="destination changed"):
        submission.restore_submission(
            bundle, paths=PATHS, expected_manifest_sha256=digest, filesystem_root=target
        )
    assert (target / "root/result.csv").read_text() == "old"
    assert (target / "root/report.md").read_text() == "newer"
    assert not list((target / "root").glob(".benchflow-submission-*"))
