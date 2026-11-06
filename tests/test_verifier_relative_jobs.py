"""Guards PR #77 publication with relative jobs directories."""

from pathlib import Path

import pytest

from benchflow.rollout._separate_verifier import _publish_verifier_outputs
from benchflow.task.paths import RolloutPaths


@pytest.mark.parametrize("name", ["jobs/trial", "../jobs/trial"])
def test_relative_jobs_publish(tmp_path, monkeypatch, name):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    trial = Path(name)
    paths = RolloutPaths(trial)
    paths.mkdir()
    (paths.verifier_dir / "planted").write_text("old")
    source = tmp_path / "outputs"
    source.mkdir()
    (source / "reward.txt").write_text("1")
    record = {}
    _publish_verifier_outputs(source, trial, record)
    assert (trial / "verifier/reward.txt").read_text() == "1"
    assert not (trial / "verifier/planted").exists()
    assert (
        trial / record["planted_in_verifier_dir"]["moved_to"] / "planted"
    ).read_text() == "old"
