"""Guards the attempt cleanup against PY8 base commit 363d815b."""

import subprocess

import pytest

from benchflow.taskmd import family


@pytest.mark.parametrize("failure", ["timeout", "error", "success"])
def test_every_generator_attempt_removes_its_exact_container(
    tmp_path, monkeypatch, failure
):
    """Client exit does not imply container exit, on any result path."""
    (tmp_path / "generate.sh").write_text("#!/bin/sh\nexit 0\n")
    calls = []

    def docker(args, **kwargs):
        calls.append(args)
        if args[0] == "run" and failure == "timeout":
            raise family.FamilyError("generator timed out")
        return subprocess.CompletedProcess(
            args, 1 if args[0] == "run" and failure == "error" else 0, "", ""
        )

    monkeypatch.setattr(family, "_docker", docker)
    monkeypatch.setattr(family, "check_instance", lambda *a, **kw: "valid")
    config = {
        "family": {"generator": "generate.sh"},
        "sandbox": {"image": "dummy:trusted"},
    }
    if failure == "success":
        assert family.generate(tmp_path, config, 1, tmp_path / "out") == "valid"
    else:
        with pytest.raises(family.FamilyError):
            family.generate(tmp_path, config, 1, tmp_path / "out")
    run = next(c for c in calls if c[0] == "run")
    removals = [c for c in calls if c[:2] == ["rm", "-f"]]
    assert len(removals) == 1
    assert "--name" in run and removals[0][2] == run[run.index("--name") + 1]
