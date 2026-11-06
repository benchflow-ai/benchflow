"""Guards the incomplete host-copy fix at 04d3d5f9."""

import os
from contextlib import suppress
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier_core import Verifier
from tests.test_llm_judge_verifier import _judge_toml, _make_sandbox, _make_task


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("swap", ["ancestor", "destination", "chmod", "hardlink"])
def test_copy_destination_cannot_redirect_writes(
    tmp_path, monkeypatch, directory, swap
):
    from benchflow.sandbox._host_copy import copy_mounted

    root = tmp_path / "source"
    root.mkdir()
    (root / "dummy").write_text("submission")
    (root / "dummy").chmod(0o644)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o700)
    marker = outside / "dummy"
    marker.write_text("DUMMY_HOST_MARKER")
    marker.chmod(0o600)
    parent = tmp_path / "parent"
    parent.mkdir()
    target = parent / "out" if directory else parent / "dummy"
    if swap == "hardlink":
        target.parent.mkdir(exist_ok=True)
        if directory:
            target.mkdir()
            os.link(marker, target / "dummy")
        else:
            os.link(marker, target)
    original_open, original_listdir = os.open, os.listdir
    swapped = False

    def exchange(path):
        nonlocal swapped
        if swapped:
            return
        path.rename(path.with_name(path.name + "-saved"))
        path.symlink_to(outside, target_is_directory=True)
        swapped = True

    def opening(path, flags, *args, **kwargs):
        if flags & os.O_WRONLY and swap in {"ancestor", "destination"}:
            exchange(target if directory and swap == "destination" else parent)
        return original_open(path, flags, *args, **kwargs)

    def listing(fd):
        entries = original_listdir(fd)
        if swap == "chmod" and directory:
            exchange(target)
        return entries

    monkeypatch.setattr(os, "open", opening)
    monkeypatch.setattr(os, "listdir", listing)
    # Refusal is safe; an outside mutation is not.
    with suppress(OSError, ValueError):
        copy_mounted(root, () if directory else ("dummy",), target, directory=directory)
    assert marker.read_text() == "DUMMY_HOST_MARKER"
    assert marker.stat().st_mode & 0o777 == 0o600
    assert outside.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.asyncio
async def test_judge_consumes_private_snapshot_until_score_finishes(tmp_path, failure):
    task = _make_task(tmp_path, _judge_toml())
    tests = task.task_dir / "tests"
    tests.mkdir()
    (tests / "rubric.json").write_text('{"criteria": []}')
    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    public = paths.verifier_dir / "deliverables"
    public.mkdir()
    (public / "dummy").write_text("stale")
    observed = []

    async def score(directory):
        directory = Path(directory)
        observed.append(directory)
        assert not directory.is_relative_to(paths.rollout_dir.resolve())
        assert directory.is_dir()
        assert (directory / "dummy").read_text() == "submission"
        (public / "dummy").write_text("tampered after download")
        assert (directory / "dummy").read_text() == "submission"
        if failure:
            raise RuntimeError("judge unavailable")
        return 1.0

    with patch("benchflow.rewards.builtins.LLMJudgeRewardFunc") as reward:
        reward.return_value.score = AsyncMock(side_effect=score)
        reward.return_value.events = []
        verifier = Verifier(task, paths, _make_sandbox({"dummy": "submission"}))
        if failure:
            with pytest.raises(RuntimeError, match="judge unavailable"):
                await verifier._verify_llm_judge()
        else:
            assert (await verifier._verify_llm_judge()).rewards == {"reward": 1.0}
    assert observed
    assert all(not directory.exists() for directory in observed)


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
@pytest.mark.asyncio
async def test_judge_publication_never_truncates_linked_host_files(tmp_path, link):
    """Guards the fix at 04d3d5f9 through final judge publication."""
    task = _make_task(tmp_path, _judge_toml())
    tests = task.task_dir / "tests"
    tests.mkdir()
    (tests / "rubric.json").write_text('{"criteria": []}')
    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    markers = [tmp_path / "host-reward", tmp_path / "host-details"]
    for target, marker in zip(
        [paths.reward_json_path, paths.reward_details_json_path], markers, strict=True
    ):
        from tests.reachable_fixture import on_disk

        target = on_disk(target)
        marker.write_text("DUMMY_HOST_MARKER")
        if link == "symlink":
            target.symlink_to(marker)
        else:
            os.link(marker, target)
    with patch("benchflow.rewards.builtins.LLMJudgeRewardFunc") as reward:
        reward.return_value.score = AsyncMock(return_value=1.0)
        reward.return_value.events = []
        result = await Verifier(task, paths, _make_sandbox({}))._verify_llm_judge()
    assert result.rewards == {"reward": 1.0}
    assert all(marker.read_text() == "DUMMY_HOST_MARKER" for marker in markers)
    assert not paths.reward_json_path.is_symlink()


@pytest.mark.asyncio
async def test_unsupported_host_refuses_before_download_or_judge(tmp_path, monkeypatch):
    """Guards C1's portable refusal before side effects (04d3d5f9 follow-up)."""
    task = _make_task(tmp_path, _judge_toml())
    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    sandbox = _make_sandbox({})
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with (
        patch("benchflow.rewards.builtins.LLMJudgeRewardFunc") as reward,
        pytest.raises(RuntimeError, match="Linux, macOS or WSL"),
    ):
        await Verifier(task, paths, sandbox)._verify_llm_judge()
    reward.assert_not_called()
    sandbox.download_dir.assert_not_called()
