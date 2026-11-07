"""Guards CONFIRM-trajectories-1 against the bypass at fe70610b."""

import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.eval_artifacts import materialize_canonical_job
from benchflow.publish import huggingface as hf


def selection_for(job, rollout):
    selection = job / "selection.json"
    selection.write_text(
        json.dumps({"job_dir": str(job), "selected": [{"rollout_dir": str(rollout)}]})
    )
    return selection


@pytest.mark.parametrize("backend", ["hf", "bucket"])
@pytest.mark.parametrize("nested", [False, True], ids=["standalone", "in-job"])
@pytest.mark.parametrize(
    "kind",
    [
        "absolute",
        "relative",
        "directory",
        "dangling",
        "cycle",
        "hardlink",
        "fifo",
        "socket",
    ],
)
def test_canonical_copy_cannot_launder_unsafe_artifacts(
    tmp_path, monkeypatch, backend, nested, kind
):
    """Guards fe70610b: canonicalization must keep the publishing invariant."""
    job = tmp_path / "job"
    rollout = job / "task__1"
    artifacts = rollout / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "normal.txt").write_text("public result")
    (rollout / "result.json").write_text('{"rewards": {"reward": 1}}')
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "dummy.txt"
    marker.write_text("DUMMY_PRIVATE_MARKER")
    bad = artifacts / "unsafe"
    listener = None
    if kind == "absolute":
        bad.symlink_to(marker)
    elif kind == "relative":
        bad.symlink_to(os.path.relpath(marker, artifacts))
    elif kind == "directory":
        bad.symlink_to(outside, target_is_directory=True)
    elif kind == "dangling":
        bad.symlink_to(outside / "missing")
    elif kind == "cycle":
        bad.symlink_to(artifacts, target_is_directory=True)
    elif kind == "hardlink":
        os.link(marker, bad)
    elif kind == "fifo":
        os.mkfifo(bad)
    else:
        listener = socket.socket(socket.AF_UNIX)
        # AF_UNIX has a short path limit; xdist's temporary roots may exceed it.
        with monkeypatch.context() as patch:
            patch.chdir(artifacts)
            listener.bind(bad.name)
    output = (job if nested else tmp_path) / "canonical"
    uploaded = []

    def inspect(folder):
        files = {
            p.relative_to(folder): p.read_bytes()
            for p in Path(folder).rglob("*")
            if p.is_file()
        }
        assert all(b"DUMMY_PRIVATE_MARKER" not in data for data in files.values())
        prefix = Path("canonical") if nested else Path()
        assert files[prefix / rollout.name / "artifacts/normal.txt"] == b"public result"
        assert (
            files[prefix / rollout.name / "result.json"]
            == (rollout / "result.json").read_bytes()
        )
        assert not any("unsafe" in path.parts for path in files)
        uploaded.append(True)

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
    try:
        materialize_canonical_job(selection_for(job, rollout), output)
        source = job if nested else output
        if backend == "hf":
            hf.publish_folder_to_hf(source, repo_id="dummy/results", path_in_repo="run")
        else:
            hf.publish_folder_to_bucket(source, bucket_id="dummy/results")
        assert uploaded == [True]
        assert {
            p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
        } == {"task__1/artifacts/normal.txt", "task__1/result.json"}
    finally:
        if listener is not None:
            listener.close()


@pytest.mark.parametrize("link_at", ["root", "ancestor"])
def test_canonical_copy_refuses_linked_source_root(tmp_path, link_at):
    """Guards fe70610b: selected source ancestors cannot redirect reads."""
    job = tmp_path / "job"
    rollout = job / "task__1"
    rollout.mkdir(parents=True)
    (rollout / "dummy.txt").write_text("DUMMY_PRIVATE_MARKER")
    alias = tmp_path / "alias"
    alias.symlink_to(
        job if link_at == "ancestor" else rollout, target_is_directory=True
    )
    source = alias / rollout.name if link_at == "ancestor" else alias
    with pytest.raises(ValueError, match="source folder"):
        materialize_canonical_job(selection_for(job, source), tmp_path / "canonical")


@pytest.mark.parametrize("swap", ["leaf", "directory"])
def test_canonical_copy_keeps_descriptor_custody(tmp_path, monkeypatch, swap):
    """Guards fe70610b: a swap after discovery cannot enter canonical copies."""
    job = tmp_path / "job"
    rollout = job / "task__1"
    artifacts = rollout / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "normal.txt").write_text("public result")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "normal.txt").write_text("DUMMY_PRIVATE_MARKER")
    original = os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if (
            path == ("normal.txt" if swap == "leaf" else "artifacts")
            and not swapped
            and "dir_fd" in kwargs
        ):
            swapped = True
            if swap == "leaf":
                (artifacts / "normal.txt").unlink()
                (artifacts / "normal.txt").symlink_to(outside / "normal.txt")
            else:
                artifacts.rename(rollout / "old")
                artifacts.symlink_to(outside, target_is_directory=True)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    output = tmp_path / "canonical"
    materialize_canonical_job(selection_for(job, rollout), output)
    assert swapped
    assert all(
        b"DUMMY_PRIVATE_MARKER" not in p.read_bytes()
        for p in output.rglob("*")
        if p.is_file()
    )
