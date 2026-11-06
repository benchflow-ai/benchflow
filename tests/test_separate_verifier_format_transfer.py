"""A task.toml task's separate verifier gets its declared artifacts over its own image.

(One that declares no artifacts still gets its whole workspace, as before:
test_separate_verifier.py's first transfer test.)

That format bundles the tests in the verifier image built from tests/ and
gives the verifier each declared artifact at its path, over the files the
image put there. Terminal-Bench 4.1's live-database-cutover bakes its
pristine /app/api git baseline into the verifier image and grades the
agent's /tmp/agent.patch against it; distributed-dedup builds the agent's
one declared source file with the sbt harness its image put in /app.
BenchFlow emptied the verifier's workspace and unpacked the agent's whole
workspace there, so git found the baseline owned by another user and sbt
read the agent's build file: both reference solutions scored 0.
"""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
from pathlib import Path

import pytest

from benchflow.rollout._separate_verifier import (
    _destination_paths,
    _install_script,
    build_transfer_payload,
)
from tests.test_separate_verifier import _agent_trial, _members


def _names(archive: Path) -> list[str]:
    return list(_members(archive))


@pytest.mark.asyncio
async def test_only_the_declared_artifacts_cross_from_the_workspace(tmp_path):
    """Unchanged image paths stay trusted."""
    answer = tmp_path / "box" / "app" / "src" / "answer.txt"
    _, _, trial = await _agent_trial(tmp_path, artifacts=[str(answer)])
    summary = build_transfer_payload(trial, tmp_path / "p.tar", artifacts_only=True)
    names = _names(tmp_path / "p.tar")
    assert summary["artifacts_only"] is True and summary["emptied"] == []
    assert summary["agent_paths"] == [str(answer), "/logs/artifacts"]
    assert any(n.endswith("app/src/answer.txt") for n in names)
    # The agent's other workspace files stay behind, the planted conftest.py
    # among them; /logs/artifacts still crosses.
    assert not any(n.endswith("app/conftest.py") for n in names)
    assert any(n.endswith("logs/artifacts/mode.txt") for n in names)


@pytest.mark.asyncio
async def test_any_other_verifier_still_gets_the_whole_workspace(tmp_path):
    answer = tmp_path / "box" / "app" / "src" / "answer.txt"
    _, _, trial = await _agent_trial(tmp_path, artifacts=[str(answer)])
    summary = build_transfer_payload(trial, tmp_path / "p.tar")
    assert summary["artifacts_only"] is False
    assert any(n.endswith("app/conftest.py") for n in _names(tmp_path / "p.tar"))


@pytest.mark.asyncio
async def test_a_declared_workspace_crosses_whole(tmp_path):
    _, _, trial = await _agent_trial(
        tmp_path, artifacts=[str(tmp_path / "box" / "app")]
    )
    summary = build_transfer_payload(trial, tmp_path / "p.tar", artifacts_only=True)
    assert summary["artifacts_only"] is False
    assert any(n.endswith("app/conftest.py") for n in _names(tmp_path / "p.tar"))


def _archive(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    (source / "app" / "out").mkdir(parents=True)
    (source / "app" / "out" / "new.txt").write_text("the agent's\n")
    archive = tmp_path / "transfer.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(source / "app" / "out" / "new.txt", arcname="app/out/new.txt")
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def _verifier_root(tmp_path: Path) -> Path:
    root = tmp_path / "verifier"
    (root / "app" / "api").mkdir(parents=True)
    (root / "app" / "api" / "baseline.txt").write_text("pristine\n")
    (root / "app" / "out").mkdir()
    (root / "app" / "out" / "stale.txt").write_text("the image's\n")
    return root


def test_the_install_keeps_the_images_workspace_and_empties_declared_folders(
    tmp_path,
):
    root = _verifier_root(tmp_path)
    archive, sha = _archive(tmp_path)
    script = _install_script(
        str(archive),
        "/app",
        sha,
        root=str(root),
        keep_workspace=True,
        emptied=["/app/out"],
    )
    subprocess.run(["sh", "-c", script], check=True, capture_output=True)
    assert (root / "app" / "api" / "baseline.txt").read_text() == "pristine\n"
    assert not (root / "app" / "out" / "stale.txt").exists()
    assert (root / "app" / "out" / "new.txt").read_text() == "the agent's\n"


def test_the_whole_workspace_install_still_replaces_it(tmp_path):
    root = _verifier_root(tmp_path)
    archive, sha = _archive(tmp_path)
    script = _install_script(str(archive), "/app", sha, root=str(root))
    subprocess.run(["sh", "-c", script], check=True, capture_output=True)
    assert not (root / "app" / "api").exists()
    assert (root / "app" / "out" / "new.txt").is_file()


def test_the_probe_walks_a_workspace_that_is_kept():
    walked = ["/app/src", "/app/out"]
    # Emptied first, a dedicated workspace's folders are not probed ...
    assert "/app/src" not in _destination_paths(walked, ["/app"], "/app")
    # ... kept, they are, except the declared folders the install empties.
    kept = _destination_paths(walked, ["/app"], "/app", emptied=["/app/out"])
    assert "/app" in kept and "/app/src" in kept and "/app/out" not in kept
