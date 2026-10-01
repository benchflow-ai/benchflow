"""The ``taskmd`` verifier strategy grades a materialized package: tests, lazy, judges, and the reward files.

A stand-in sandbox and verifier replace Docker: the "container" is a folder on
the host, ``test.sh`` is replaced by writing its report, and the Messages API
by a script of ``submit_review`` calls.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from benchflow.task import Task
from benchflow.task.formats import materialize_task_dir
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier_document import load_verifier_document
from benchflow.task.verifier_errors import VerifierResult
from benchflow.taskmd import judging, verify
from tests._taskmd_helpers import EXAMPLES


@pytest.fixture(autouse=True)
def format_cache(tmp_path, monkeypatch) -> Path:
    cache = tmp_path / "format-cache"
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(cache))
    for name in (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_OAUTH_TOKEN",
        judging.MODEL_ENV,
        judging.SESSION_LIMIT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    return cache


@dataclass
class _Result:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _Sandbox:
    """A container that is a folder on the host."""

    root: Path
    online: bool = False
    commands: list[str] = field(default_factory=list)

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        if "then echo link" in command:
            path = re.search(r"\[ -L '?([^' ]+)'? \]", command).group(1)
            target = self.root / path.lstrip("/")
            kind = (
                "link"
                if target.is_symlink()
                else "dir"
                if target.is_dir()
                else "file"
                if target.is_file()
                else "none"
            )
            return _Result(stdout=kind + "\n")
        if "/proc/net/dev" in command:
            return (
                _Result(1, "online: the container reaches the internet")
                if self.online
                else _Result(0, "offline: no network interface but lo\n")
            )
        if "kept.tar --files-from" in command:
            return _Result(stdout="setpriv\n")
        if command.startswith("cat /run/taskmd-judge/path"):
            return _Result(stdout="/usr/local/bin:/usr/bin:/bin\n")
        if "base64 -d)" in command:
            output = b"alpha=2.55\n"
            return _Result(
                stdout=f"0 {len(output)}\n{base64.b64encode(output).decode()}\n"
            )
        return _Result()

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(
            self.root / source_dir.lstrip("/"),
            target_dir,
            dirs_exist_ok=True,
            symlinks=True,
        )

    async def download_file(self, source, target):
        shutil.copy2(self.root / source.lstrip("/"), target)


class _Verifier:
    """The parts of benchflow.task.verifier_core.Verifier the strategy reads."""

    def __init__(
        self,
        native: Path,
        trial: Path,
        sandbox: _Sandbox,
        *,
        report: dict | None,
        script_reward: str | None = None,
    ) -> None:
        self._task = Task(native)
        self._rollout_paths = RolloutPaths(rollout_dir=trial)
        self._rollout_paths.mkdir()
        self._sandbox = sandbox
        self.report = report
        self.script_reward = script_reward
        self.test_return_code = None
        self.calls: list[dict] = []

    async def _verify_test_script(self, **kwargs):
        self.calls.append(kwargs)
        verifier_dir = self._rollout_paths.verifier_dir
        if self.report is not None:
            (verifier_dir / "ctrf.json").write_text(json.dumps(self.report))
        if self.script_reward is not None:
            (verifier_dir / "reward.txt").write_text(self.script_reward)
        self.test_return_code = 0
        if kwargs.get("parse_rewards"):
            return VerifierResult(rewards={"reward": float(self.script_reward or 0)})
        return VerifierResult(rewards=None)


def _strategy(native: Path):
    tests = native / "verifier" if (native / "verifier").is_dir() else native / "tests"
    return load_verifier_document(tests).selected_strategy


def _ctrf(*tests: tuple[str, str]) -> dict:
    return {
        "results": {
            "tool": {"name": "test.sh"},
            "tests": [{"name": n, "status": s} for n, s in tests],
        }
    }


async def test_a_test_gate_passes_and_fails(tmp_path) -> None:
    native = materialize_task_dir(EXAMPLES / "hello-world")
    sandbox = _Sandbox(tmp_path / "container")
    for status, reward in (("passed", 1.0), ("failed", 0.0)):
        trial = tmp_path / f"trial-{status}"
        verifier = _Verifier(
            native,
            trial,
            sandbox,
            report=_ctrf(("test_hello", status)),
            script_reward="0.5",
        )
        result = await verify.verify_taskmd(verifier, _strategy(native))
        assert result.rewards == {"reward": reward, "strict": reward, "partial": reward}
        assert (
            verifier.calls[0]["parse_rewards"] is False
            and verifier.calls[0]["cwd"] is None
        )
        review = json.loads((trial / "verifier" / "review.json").read_text())
        assert review["verdicts"][0]["gate"] is True
        assert review["verdicts"][0]["verdict"] == ("pass" if reward else "fail")
        # A reward file the script wrote is set aside: the rubric decides.
        assert (trial / "verifier" / "script-reward.txt").read_text() == "0.5"
        assert (
            json.loads((trial / "verifier" / "reward.json").read_text())["reward"]
            == reward
        )
        assert (trial / "verifier" / "reward.txt").read_text() == f"{reward}\n"


async def test_no_report_fails_every_test_criterion(tmp_path) -> None:
    native = materialize_task_dir(EXAMPLES / "hello-world")
    verifier = _Verifier(
        native, tmp_path / "trial", _Sandbox(tmp_path / "c"), report=None
    )
    result = await verify.verify_taskmd(verifier, _strategy(native))
    assert result.rewards["reward"] == 0.0
    review = json.loads((tmp_path / "trial" / "verifier" / "review.json").read_text())
    assert (
        review["x-benchflow"]["tests"]["problem"] == "the verifier wrote no ctrf.json"
    )


async def test_a_harbor_import_keeps_the_scripts_reward(tmp_path) -> None:
    native = materialize_task_dir(EXAMPLES / "regex-log")
    verifier = _Verifier(
        native,
        tmp_path / "trial",
        _Sandbox(tmp_path / "c"),
        report=None,
        script_reward="1",
    )
    result = await verify.verify_taskmd(verifier, _strategy(native))
    assert result.rewards == {"reward": 1.0}
    assert verifier.calls[0]["parse_rewards"] is True


async def test_an_offline_verifier_that_is_online_is_not_scored(tmp_path) -> None:
    native = materialize_task_dir(EXAMPLES / "analysis-judge")
    verifier = _Verifier(
        native, tmp_path / "trial", _Sandbox(tmp_path / "c", online=True), report=None
    )
    with pytest.raises(
        verify.TaskMdVerifierError, match="could not take the verifier offline"
    ):
        await verify.verify_taskmd(verifier, _strategy(native))


class _Judge:
    """A Messages API stand-in: one scripted reply list per session, in order."""

    def __init__(self, replies: list[list[dict]]) -> None:
        self.replies = replies
        self.sessions = 0
        self.bodies: list[dict] = []

    def client(self, credentials):
        judge = self

        async def call(body, timeout):
            judge.bodies.append(body)
            if len(body["messages"]) == 1:
                judge.sessions += 1
            script = judge.replies[(judge.sessions - 1) % len(judge.replies)]
            step = (len(body["messages"]) - 1) // 2
            return {
                "content": script[min(step, len(script) - 1)],
                "usage": {"input_tokens": 1000, "output_tokens": 50},
            }

        return call


def _analysis_container(root: Path) -> None:
    (root / "work").mkdir(parents=True)
    (root / "work" / "fit.json").write_text(
        '{"alpha": 2.596, "alpha_err": 0.05, "xmin": 15.13, "n_tail": 860}\n'
    )
    (root / "work" / "report.md").write_text(
        "# Fit\n\nMaximum likelihood above a KS-chosen x_min.\n\nCaveat: sensitive to x_min.\n"
    )
    (root / "work" / "fit.png").write_bytes(b"\x89PNG\r\n\x1a\nplot")


def _submit(levels: dict[str, str], passes: dict[str, str]) -> dict:
    verdicts = [
        {
            "id": cid,
            "verdict": "level",
            "level": level,
            "citations": [
                {
                    "source": "file",
                    "path": "/work/report.md",
                    "quote": "Maximum likelihood",
                }
            ],
            "rationale": "ok",
        }
        for cid, level in levels.items()
    ] + [
        {
            "id": cid,
            "verdict": verdict,
            "citations": [{"source": "judge", "step": 1, "quote": "alpha=2.55"}],
            "rationale": "ok",
        }
        for cid, verdict in passes.items()
    ]
    return {
        "type": "tool_use",
        "id": "submit",
        "name": "submit_review",
        "input": {"verdicts": verdicts, "tags": []},
    }


FULL = _submit(
    {"mle-method": "4", "xmin-justified": "2", "uncertainty-justified": "2"},
    {"figure-shows-fit": "pass", "caveats": "pass", "unsupported-claim": "pass"},
)
RUN = {
    "type": "tool_use",
    "id": "run",
    "name": "run",
    "input": {"command": "python3 -c 'print(2.55)'"},
}


async def test_the_agent_judge_grades_the_oracle(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.setenv(judging.MODEL_ENV, "claude-haiku-4-5-20251001")
    judge = _Judge([[[RUN], [FULL]]])
    monkeypatch.setattr(judging, "MessagesClient", judge.client)
    native = materialize_task_dir(EXAMPLES / "analysis-judge")
    container = tmp_path / "container"
    _analysis_container(container)
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "acp_trajectory.jsonl").write_text(
        json.dumps(
            {"type": "oracle", "return_code": 0, "stdout": "KS scan: xmin=15.1\n"}
        )
        + "\n"
    )
    report = _ctrf(
        ("test_outputs_present", "passed"),
        ("test_alpha_accurate", "passed"),
        ("test_uncertainty_covers", "passed"),
    )
    verifier = _Verifier(native, trial, _Sandbox(container), report=report)
    result = await verify.verify_taskmd(verifier, _strategy(native))
    assert result.rewards == {"reward": 1.0, "strict": 1.0, "partial": 1.0}
    assert judge.sessions == 3  # samples = 3, per rubric
    first = judge.bodies[0]
    assert first["model"] == "claude-haiku-4-5-20251001"
    assert first["messages"][0]["content"].startswith(
        "You are grading a power-law analysis"
    )
    assert (
        "60 tool calls, 2000000 tokens, 1200 seconds" in first["messages"][0]["content"]
    )
    review = json.loads((trial / "verifier" / "review.json").read_text())
    by_id = {v["id"]: v for v in review["verdicts"]}
    assert (
        by_id["mle-method"]["verdict"] == "level"
        and by_id["mle-method"]["level"] == "4"
        and by_id["mle-method"]["score"] == 4.0
    )
    assert (
        len(by_id["mle-method"]["samples"]) == 3
        and by_id["mle-method"]["spread"] == 0.0
    )
    assert by_id["mle-method"]["judge"]["model"] == "claude-haiku-4-5-20251001"
    assert (
        by_id["mle-method"]["judge"]["harness"] == "judge-loop"
        and by_id["mle-method"]["judge"]["prompt_format"] == "judge-prompt@1"
    )
    assert by_id["mle-method"]["citations"][0]["verified"] is True
    assert by_id["caveats"]["citations"][0]["label"] == "solver-executed"
    assert by_id["unsupported-claim"]["score"] == 0.0
    assert review["submission_tree"].startswith("sha256:")
    trajectory = (
        trial / "verifier" / "taskmd-judge" / "fs" / "judge" / "trajectory.jsonl"
    ).read_text()
    assert (
        '"arguments":"{\\"command\\": \\"bash /opt/seat/run.sh\\"}"' in trajectory
        and "KS scan" in trajectory
    )
    sessions = sorted(
        (trial / "verifier" / "taskmd-judge").glob("agent-rubric-s*-a1.json")
    )
    assert (
        len(sessions) == 3 and json.loads(sessions[0].read_text())["end"] == "accepted"
    )


async def test_a_failed_gate_skips_the_judges_but_not_a_penalty(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    judge = _Judge([[[FULL]]])
    monkeypatch.setattr(judging, "MessagesClient", judge.client)
    native = materialize_task_dir(EXAMPLES / "analysis-judge")
    container = tmp_path / "container"
    (container / "work").mkdir(parents=True)
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "nop"}) + "\n"
    )
    report = _ctrf(
        ("test_outputs_present", "failed"),
        ("test_alpha_accurate", "failed"),
        ("test_uncertainty_covers", "failed"),
    )
    verifier = _Verifier(native, trial, _Sandbox(container), report=report)
    result = await verify.verify_taskmd(verifier, _strategy(native))
    assert result.rewards["reward"] == 0.0
    review = json.loads((trial / "verifier" / "review.json").read_text())
    by_id = {v["id"]: v for v in review["verdicts"]}
    assert by_id["mle-method"]["verdict"] == "skip" and by_id["mle-method"][
        "flags"
    ] == ["lazy"]
    assert by_id["caveats"]["verdict"] == "skip" and by_id["caveats"]["flags"] == [
        "lazy"
    ]
    # unsupported-claim has negative points, so lazy never skips it; the session still judges it.
    assert by_id["unsupported-claim"]["verdict"] == "pass"
    assert judge.sessions == 3


async def test_a_criterion_whose_files_were_never_saved_fails_without_a_session(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    judge = _Judge([[[FULL]]])
    monkeypatch.setattr(judging, "MessagesClient", judge.client)
    native = materialize_task_dir(EXAMPLES / "analysis-judge")
    container = tmp_path / "container"
    _analysis_container(container)
    (container / "work" / "report.md").unlink()
    report = _ctrf(
        ("test_outputs_present", "passed"),
        ("test_alpha_accurate", "passed"),
        ("test_uncertainty_covers", "passed"),
    )
    trial = tmp_path / "trial"
    result = await verify.verify_taskmd(
        _Verifier(native, trial, _Sandbox(container), report=report), _strategy(native)
    )
    by_id = {
        v["id"]: v
        for v in json.loads((trial / "verifier" / "review.json").read_text())[
            "verdicts"
        ]
    }
    assert by_id["caveats"]["verdict"] == "fail" and by_id["caveats"]["flags"] == [
        "evidence-missing"
    ]
    assert "samples" not in by_id["caveats"]
    assert (
        by_id["mle-method"]["verdict"] == "level"
    )  # its other items (the trajectory) remain
    assert result.rewards["partial"] == 14 / 15


async def test_a_judge_that_never_submits_leaves_the_trial_unscored(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    judge = _Judge([[[{"type": "text", "text": "thinking"}]]])
    monkeypatch.setattr(judging, "MessagesClient", judge.client)
    monkeypatch.setattr(judging, "SESSIONS", judging.SessionLimit())
    native = materialize_task_dir(EXAMPLES / "analysis-judge")
    container = tmp_path / "container"
    _analysis_container(container)
    report = _ctrf(
        ("test_outputs_present", "passed"),
        ("test_alpha_accurate", "passed"),
        ("test_uncertainty_covers", "passed"),
    )
    verifier = _Verifier(native, tmp_path / "trial", _Sandbox(container), report=report)
    with pytest.raises(verify.TaskMdVerifierError, match="valid samples"):
        await verify.verify_taskmd(verifier, _strategy(native))
    assert judge.sessions == 6  # 3 samples, each rejudged once


async def test_the_session_limit_stops_the_judges(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.setenv(judging.SESSION_LIMIT_ENV, "2")
    monkeypatch.setattr(judging, "SESSIONS", judging.SessionLimit())
    judge = _Judge([[[FULL]]])
    monkeypatch.setattr(judging, "MessagesClient", judge.client)
    native = materialize_task_dir(EXAMPLES / "analysis-judge")
    container = tmp_path / "container"
    _analysis_container(container)
    report = _ctrf(
        ("test_outputs_present", "passed"),
        ("test_alpha_accurate", "passed"),
        ("test_uncertainty_covers", "passed"),
    )
    verifier = _Verifier(native, tmp_path / "trial", _Sandbox(container), report=report)
    with pytest.raises(judging.JudgeError, match="spent"):
        await verify.verify_taskmd(verifier, _strategy(native))
    assert judge.sessions == 2


def test_kept_paths() -> None:
    assert verify.kept_paths({"sandbox": {"workdir": "/work/"}}) == ["/work"]
    assert verify.kept_paths(
        {
            "sandbox": {
                "workdir": "/work",
                "outputs": ["/work/a.md", {"path": "/logs/x"}],
            }
        }
    ) == ["/work/a.md", "/logs/x"]
    assert verify.kept_paths({}) == []


async def test_a_symbolic_link_in_an_output_is_never_kept(tmp_path) -> None:
    container = tmp_path / "container"
    (container / "work").mkdir(parents=True)
    (container / "work" / "ok.md").write_text("ok")
    (container / "work" / "link").symlink_to("/etc/passwd")
    refused = await verify.copy_kept(_Sandbox(container), ["/work"], tmp_path / "fs")
    assert refused == {"/work/link": "a symbolic link or special file is never saved"}
    assert (tmp_path / "fs" / "work" / "ok.md").read_text() == "ok"
    assert not (tmp_path / "fs" / "work" / "link").exists()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_the_runner_hides_mounted_logs_and_never_deletes_through_a_mount(
    tmp_path,
) -> None:
    """Before the judge's first ``run`` call, the verifier's files leave the sandbox.

    BenchFlow bind-mounts /logs/verifier and /logs/agent from the host, so
    deleting them would delete the trial's records (it once did: ``rm -rf``
    emptied the mounted folder, then failed on the mount point). The setup
    script runs here against a folder, with a mount table naming which paths
    are mount points.
    """
    root = tmp_path / "root"
    for rel, text in {
        "verifier/rubric.json": "{}",
        "tests/test_outputs.py": "x",
        "solution/data/answer.txt": "42",
        "logs/verifier/ctrf.json": "{}",
        "logs/agent/acp_trajectory.jsonl": "",
        "logs/artifacts/fit.png": "png",
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "solve.sh").write_text("echo")
    (root / "oracle").symlink_to(elsewhere)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "22 1 0:21 / / rw - overlay overlay rw\n"
        + "".join(
            f"{n} 22 8:1 /src{n} {root}/{rel} rw - ext4 /dev/sda1 rw\n"
            for n, rel in enumerate(
                ["tests", "solution/data", "logs/verifier", "logs/agent"], start=30
            )
        )
    )
    watched = [
        root / "tests",
        root / "solution",
        root / "logs/verifier",
        root / "logs/agent",
    ]
    before = {p: _mode(p) for p in watched}
    logs_mode = _mode(root / "logs")
    subprocess.run(
        ["bash", "-c", verify._hide_script(str(root), str(mountinfo))], check=True
    )
    # Removed: files that are not mounted, and a link (never what it points to).
    assert not (root / "verifier").exists()
    assert not (root / "oracle").is_symlink() and (elsewhere / "solve.sh").is_file()
    # Hidden, not deleted: a mount point, a folder holding one, and the logs.
    for rel in (
        "tests/test_outputs.py",
        "solution/data/answer.txt",
        "logs/verifier/ctrf.json",
        "logs/agent/acp_trajectory.jsonl",
    ):
        assert (root / rel).is_file(), rel
    assert all(_mode(p) == 0o700 for p in watched)
    # /logs itself and the solver's artifacts stay as they were.
    assert _mode(root / "logs") == logs_mode
    assert (root / "logs/artifacts/fit.png").is_file()
    subprocess.run(["bash", "-c", verify._restore_script(str(root))], check=True)
    assert {p: _mode(p) for p in watched} == before


def test_the_runner_leaves_a_log_folder_holding_a_kept_path_visible(tmp_path) -> None:
    root = tmp_path / "root"
    (root / "logs" / "agent").mkdir(parents=True)
    (root / "logs" / "verifier").mkdir(parents=True)
    before = _mode(root / "logs" / "agent")
    script = verify._hide_script(
        str(root), str(tmp_path / "no-mountinfo"), hidden=("/logs/verifier",)
    )
    subprocess.run(["bash", "-c", script], check=True)
    assert _mode(root / "logs" / "verifier") == 0o700
    assert _mode(root / "logs" / "agent") == before


async def test_the_runner_does_not_hide_a_log_folder_holding_a_kept_path_or_view() -> (
    None
):
    sandbox = _Sandbox(Path("/nonexistent"))
    runner = verify.SandboxRunner(
        sandbox=sandbox, kept=["/work", "/logs/agent/report.md"], views={}
    )
    await runner.setup()
    setup = next(c for c in sandbox.commands if "kept.tar --files-from" in c)
    assert "for p in /logs/verifier; do" in setup
    assert "/logs/agent;" not in setup
    sandbox = _Sandbox(Path("/nonexistent"))
    runner = verify.SandboxRunner(
        sandbox=sandbox, kept=["/work"], views={"/logs/verifier/trajectory.json": b""}
    )
    await runner.setup()
    setup = next(c for c in sandbox.commands if "kept.tar --files-from" in c)
    assert "for p in /logs/agent; do" in setup
