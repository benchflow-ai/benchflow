"""Independently guard 5d38379f/7ee9daee's R3-1 fix at its role and cleanup boundaries."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.taskmd import verify
from tests.reachable_fixture import directory
from tests.test_taskmd_agent_judge import FIXTURE, _Box, _grade


@pytest.mark.parametrize("isolation", ["shared", "separate"])
@pytest.mark.parametrize("case,expected", [("oracle", 1.0), ("nop", 0.0)])
@pytest.mark.parametrize("attack", ["session", "evidence"])
async def test_llm_publication_keeps_all_writes_private(
    tmp_path, monkeypatch, isolation, case, expected, attack
):
    """Guards 5d38379f against R3-1's shared-LLM route and separate-LLM sibling."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-not-a-key")
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))
    package = tmp_path / "package"
    shutil.copytree(FIXTURE, package)
    task = package / "task.md"
    task.write_text(
        task.read_text()
        .replace('isolation = "separate"', f'isolation = "{isolation}"')
        .replace("verifier.judges.agent", "verifier.judges.llm")
        .replace(
            'budget = { tool_calls = 8, tokens = "400K" }',
            'budget = { tokens = "400K" }',
        )
    )
    rubric_file = package / "verifier/rubric.json"
    rubric = json.loads(rubric_file.read_text())
    rubric["criteria"][0].update(
        judge="llm",
        text="The saved answer is 42.",
        evidence=["out/z.txt"],
        guidance="Read and cite the saved answer.",
    )
    rubric_file.write_text(json.dumps(rubric))
    root = tmp_path / "container"
    (root / "app/out").mkdir(parents=True)
    answer = "42" if case == "oracle" else "missing"
    (root / "app/out/z.txt").write_text(answer)
    public = tmp_path / f"trial-{case}/verifier" / verify.JUDGE_DIR
    victim = tmp_path / "victim-a"
    victim.mkdir()
    record_name = "llm-criterion-prints-42-s1-a1.json"
    marker = victim / (record_name if attack == "session" else "z.txt")
    marker.write_text("controller marker")
    attacked = []
    original_copy = shutil.copyfile

    def copy(source, destination, *args, **kwargs):
        destination = Path(destination)
        if destination.name == "z.txt" and public in destination.parents:
            parent = destination.parent
            parent.rename(parent.with_name("old-out"))
            parent.symlink_to(victim, target_is_directory=True)
            attacked.append(True)
        return original_copy(source, destination, *args, **kwargs)

    if attack == "evidence":
        monkeypatch.setattr(shutil, "copyfile", copy)

    class Judge:
        calls = 0

        def client(self, credentials):
            return self.call

        async def call(self, body, timeout):
            self.calls += 1
            assert answer in body["messages"][0]["content"]
            if attack == "session" and not attacked:
                if public.exists():
                    public.rename(public.with_name("old-judge"))
                public.symlink_to(victim, target_is_directory=True)
                attacked.append(True)
            return {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "submit",
                        "name": "submit_review",
                        "input": {
                            "verdicts": [
                                {
                                    "id": "prints-42",
                                    "verdict": "pass" if expected else "fail",
                                    "citations": [
                                        {
                                            "source": "file",
                                            "path": "/app/out/z.txt",
                                            "quote": answer,
                                        }
                                    ],
                                    "rationale": "The saved answer determines the verdict.",
                                }
                            ],
                            "tags": [],
                        },
                    }
                ],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }

    judge = Judge()
    result, _, box = await _grade(
        tmp_path, case, judge, monkeypatch, package=package, box=_Box(root)
    )
    assert result.rewards["reward"] == expected
    assert judge.calls == 1
    assert not box.runs  # Both LLM isolation paths grade the saved evidence.
    assert marker.read_text() == "controller marker"
    assert (public / "fs/app/out/z.txt").read_text() == answer
    assert json.loads((public / record_name).read_text())["accepted"] is not None


@pytest.mark.parametrize("ending", ["success", "error", "cancel"])
async def test_retired_tree_cleanup_cannot_follow_a_retained_writer(
    tmp_path, monkeypatch, ending
):
    """Guards 5d38379f's retirement against live descendants on every exit path."""
    import asyncio

    verifier_dir = tmp_path / "trial/verifier"
    public = verifier_dir / verify.JUDGE_DIR
    (public / "child").mkdir(parents=True)
    (public / "child/stale").write_text("old record")
    victim = tmp_path / "victim-a"
    victim.mkdir()
    marker = victim / "marker"
    marker.write_text("controller marker")
    verifier = SimpleNamespace(
        _rollout_paths=SimpleNamespace(verifier_dir=directory(verifier_dir))
    )
    held = []
    attacked = []
    from benchflow._utils import reachable_write

    original_open = reachable_write._open_beneath
    # A sandbox writer can retain this handle even after its directory is retired.
    writer = os.open(public, os.O_RDONLY | os.O_DIRECTORY)

    def swap(parent, path, flags, *args, **kwargs):
        if path == "child" and not attacked:
            attacked.append(True)
            os.rename("child", "old-child", src_dir_fd=writer, dst_dir_fd=writer)
            os.symlink(victim, "child", dir_fd=writer)
        return original_open(parent, path, flags, *args, **kwargs)

    async def grade(verifier, strategy, fs):
        held.append(fs.parent)
        fs.mkdir()
        (fs / "answer").write_text("evidence")
        if ending == "error":
            raise RuntimeError("dummy judge failure")
        if ending == "cancel":
            raise asyncio.CancelledError()
        return "result"

    monkeypatch.setattr(verify, "_verify_taskmd", grade)
    monkeypatch.setattr(reachable_write, "_open_beneath", swap)
    try:
        if ending == "success":
            assert await verify.verify_taskmd(verifier, None) == "result"
        else:
            with pytest.raises(
                RuntimeError if ending == "error" else asyncio.CancelledError
            ):
                await verify.verify_taskmd(verifier, None)
        assert attacked
        assert marker.read_text() == "controller marker"
        assert (public / "fs/answer").read_text() == "evidence"
        # A detected concurrent mutation may leave an unreachable private remnant;
        # no descendant of the victim can be removed by that best-effort cleanup.
        assert held
    finally:
        os.close(writer)
