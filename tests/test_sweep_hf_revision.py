"""Guards HF source identity at PY8 base 363d815b."""

import sys
from types import SimpleNamespace

import pytest

from benchflow._utils import hf_datasets as hf


@pytest.mark.parametrize("requested", [None, "main", "release-tag", "a" * 40])
@pytest.mark.parametrize("selected", [(), ("task-a",)])
def test_download_and_metadata_use_the_same_resolved_revision(
    tmp_path, monkeypatch, requested, selected
):
    """Guards 363d815b: a moving ref cannot stamp one commit onto another."""
    source = tmp_path / "remote"
    (source / "task-a").mkdir(parents=True)
    resolved = "b" * 40
    monkeypatch.setattr(hf, "_read_hf_revision", lambda *args: resolved)
    received = []

    def download(*args, revision, **kwargs):
        received.append(revision)
        return source

    monkeypatch.setattr(hf, "_download_snapshot", download)
    result = hf.snapshot_hf_dataset(
        "org/tasks",
        output_dir=tmp_path / "output",
        revision=requested,
        include_tasks=selected,
    )
    assert received == [resolved]
    assert result.provenance["resolved_revision"] == resolved
    assert result.provenance["requested_revision"] == requested


def test_missing_resolved_identity_is_refused(monkeypatch):
    """Guards 363d815b: a failed lookup cannot fall back to a mutable ref."""
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: SimpleNamespace(
                repo_info=lambda *a, **kw: SimpleNamespace(sha=None)
            )
        ),
    )
    with pytest.raises(ValueError, match="Could not resolve"):
        hf._read_hf_revision("org/tasks", "main")
