"""Guards FF-sw-trajectories against unsafe upload paths at 363d815b."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.publish import huggingface as hf


@pytest.mark.parametrize("backend", ["hf", "bucket"])
@pytest.mark.parametrize("kind", ["file-link", "directory-link", "fifo", "dangling"])
def test_upload_clients_only_receive_regular_snapshot(
    tmp_path, monkeypatch, backend, kind
):
    """Guards the fix across upload backends."""
    source = tmp_path / "job"
    source.mkdir()
    (source / "normal.txt").write_text("public result")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "dummy.txt").write_text("DUMMY_PRIVATE_MARKER")
    bad = source / "unsafe"
    if kind == "file-link":
        bad.symlink_to(outside / "dummy.txt")
    elif kind == "directory-link":
        bad.symlink_to(outside, target_is_directory=True)
    elif kind == "fifo":
        os.mkfifo(bad)
    else:
        bad.symlink_to(tmp_path / "missing")

    def inspect(folder):
        files = list(Path(folder).rglob("*"))
        assert {p.relative_to(folder).as_posix() for p in files} == {"normal.txt"}
        assert files[0].read_text() == "public result"
        assert Path(folder) != source

    monkeypatch.setattr(
        hf,
        "_require_hf_api",
        lambda: SimpleNamespace(
            create_repo=lambda *a, **kw: None,
            upload_folder=lambda **kw: inspect(kw["folder_path"]),
        ),
    )
    monkeypatch.setattr("huggingface_hub.create_bucket", lambda *a, **kw: None)
    monkeypatch.setattr(
        "huggingface_hub.sync_bucket", lambda folder, remote: inspect(folder)
    )
    if backend == "hf":
        hf.publish_folder_to_hf(source, repo_id="dummy/results", path_in_repo="run")
    else:
        hf.publish_folder_to_bucket(source, bucket_id="dummy/results")


@pytest.mark.parametrize("link_at", ["root", "ancestor"])
def test_upload_refuses_linked_source_root(tmp_path, monkeypatch, link_at):
    """Guards 363d815b's root/ancestor aliases, before any upload."""
    real = tmp_path / "real"
    (real / "job").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(
        real if link_at == "ancestor" else real / "job", target_is_directory=True
    )
    source = alias / "job" if link_at == "ancestor" else alias
    monkeypatch.setattr(
        hf,
        "_require_hf_api",
        lambda: SimpleNamespace(
            create_repo=lambda *a, **kw: None,
            upload_folder=lambda **kw: pytest.fail(
                "unsafe source reached upload client"
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"source|symbolic|link"):
        hf.publish_folder_to_hf(source, repo_id="dummy/results", path_in_repo="run")


@pytest.mark.parametrize("swap", ["leaf", "directory"])
def test_snapshot_does_not_follow_a_replaced_entry(tmp_path, monkeypatch, swap):
    """Guards FF-sw-trajectories: enumeration never authorizes a later path."""
    from benchflow.publish import _snapshot

    source = tmp_path / "job"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "result").write_text("public")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result").write_text("DUMMY_PRIVATE_MARKER")
    original = os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        match = "result" if swap == "leaf" else "sub"
        if path == match and not swapped and "dir_fd" in kwargs:
            swapped = True
            if swap == "leaf":
                target = source / "sub" / "result"
                target.unlink()
                target.symlink_to(outside / "result")
            else:
                (source / "sub").rename(source / "old")
                (source / "sub").symlink_to(outside, target_is_directory=True)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(_snapshot.os, "open", racing_open)
    with _snapshot.staged_folder(source) as staged:
        assert all(
            b"DUMMY_PRIVATE_MARKER" not in p.read_bytes()
            for p in staged.rglob("*")
            if p.is_file()
        )
    assert swapped


@pytest.mark.parametrize("flag", ["O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"])
def test_missing_no_follow_support_refuses_with_an_action(tmp_path, monkeypatch, flag):
    """Guards FF-sw-trajectories: unavailable safe traversal never falls back."""
    from benchflow.publish import _snapshot

    with monkeypatch.context() as patch:
        patch.delattr(_snapshot.os, flag)
        with pytest.raises(ValueError, match="Linux, macOS or WSL"):
            _snapshot._open_directory(tmp_path)
