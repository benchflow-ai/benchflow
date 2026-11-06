"""The agent judge against Rust 0.8.2's list (P1.4a, P1.4b), on the ``judged-agent`` fixture.

``tests/fixtures/taskmd/judged-agent`` has one criterion an agent judge
decides by running ``python3 /app/main.py``. A stand-in sandbox replaces
Docker: the "container" is a folder, and a ``run`` call executes the judge's
command against that folder with this interpreter (the fixture's own
programs only). A scripted Messages API stands in for the model: it reads
the evidence, runs the submission, sends a review naming a criterion that is
not in its assignment (rejected), then submits the verdict the run's output
supports. The live run of the same fixture on Docker is in round PY8's J1
report; with a real model as the judge it waits for a key.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from benchflow.sandbox._base import (
    ROOT_EXEC_PATH,
    SYSTEM_SHELL_ROUTE,
    SYSTEM_SHELL_UNSET_VARS,
    ExecResult,
    system_shell_env_flags,
    system_shell_prefix,
    with_system_path,
)
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task import Task
from benchflow.task.formats import materialize_task_dir
from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier_document import load_verifier_document
from benchflow.task.verifier_errors import VerifierResult
from benchflow.taskmd import judging, verify
from benchflow.taskmd import trajectory as traj
from benchflow.taskmd.check import check_package
from benchflow.taskmd.materialize import TaskMdError, TaskMdFormat
from tests._docker_cp_stream import fake_docker_cp
from tests._taskmd_helpers import emulate_kept_tar

FIXTURE = Path(__file__).parent / "fixtures" / "taskmd" / "judged-agent"
JUDGE_KEY = "judge-key-placeholder-0123456789abcdef"
AGENT_TOKEN = "sk-ant-api03-" + "Q" * 40
# Where a shell BenchFlow starts for root looks for programs (benchflow.sandbox._base.SYSTEM_PATH).
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "format-cache"))
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_OAUTH_TOKEN",
        "BENCHFLOW_TASKMD_JUDGE_BASE_URL",
        "BENCHFLOW_TASKMD_JUDGE_AUTH_TOKEN",
        "BENCHFLOW_TASKMD_JUDGE_TRUST_BASE_URL",
        judging.MODEL_ENV,
        judging.SESSION_LIMIT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", JUDGE_KEY)
    monkeypatch.setattr(judging, "SESSIONS", judging.SessionLimit())


@dataclass
class _Result:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


# The judge's command, base64-encoded in the runner's script (0.8.0 decoded it into a variable).
_COMMAND = re.compile(r"printf %s '?([A-Za-z0-9+/=]*)'? \| base64 -d")


@dataclass
class _Box:
    """A container that is a folder; ``run`` executes the judge's command against it."""

    root: Path
    drop: str = "BF drop setpriv /usr/bin/setpriv"
    commands: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    uploads: list[str] = field(default_factory=list)
    supports_system_shell = True  # as Docker and Daytona

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        if copied := emulate_kept_tar(self.root, command):
            return _Result(stdout=copied)
        if command.endswith("pwd -P"):
            return _Result(stdout="/app\n")
        if "then echo link" in command:
            path = re.search(r"\[ -L '?([^' ]+)'? \]", command).group(1)
            target = self.root / path.lstrip("/")
            kind = "dir" if target.is_dir() else "file" if target.is_file() else "none"
            return _Result(stdout=kind + "\n")
        if "/proc/net/dev" in command:
            return _Result(0, "offline: no network interface but lo\n")
        if 'echo "BF drop' in command:
            return _Result(stdout=self.drop + "\n")
        if "cat /run/taskmd-judge/path" in command:
            return _Result(stdout="/usr/local/bin:/usr/bin:/bin\n")
        found = _COMMAND.search(command)
        if found:
            self.runs.append(command)
            code, output = self.run(base64.b64decode(found.group(1)).decode(), command)
            return _Result(
                stdout=f"{code} {len(output)}\n{base64.b64encode(output).decode()}\n"
            )
        return _Result()

    def run(self, command: str, script: str) -> tuple[int, bytes]:
        """``python3 <path>`` against the folder, with the environment the script gives the command."""
        words = command.split()
        assert words[0] == "python3" and len(words) == 2, command
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"}
        safepath = re.search(r" PYTHONSAFEPATH=(\S+)", script)
        if safepath:
            env["PYTHONSAFEPATH"] = safepath.group(1).strip("'")
        program = self.root / words[1].lstrip("/")
        done = subprocess.run(
            [sys.executable, str(program)],
            env=env,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = (done.stdout + done.stderr).replace(str(self.root), "")
        return done.returncode, output.encode()

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(
            self.root / source_dir.lstrip("/"), target_dir, dirs_exist_ok=True
        )

    async def download_file(self, source, target):
        shutil.copy2(self.root / source.lstrip("/"), target)

    async def upload_file(self, source, target, **kwargs):
        self.uploads.append(target)


class _Verifier:
    """The parts of benchflow.task.verifier_core.Verifier the strategy reads."""

    def __init__(self, native: Path, trial: Path, sandbox: _Box) -> None:
        self._task = Task(native)
        self._rollout_paths = RolloutPaths(rollout_dir=trial)
        self._rollout_paths.mkdir()
        self._sandbox = sandbox
        self.test_return_code = None

    async def _verify_test_script(self, **kwargs):  # the fixture has no test.sh
        raise AssertionError("judged-agent has no test.sh")
        return VerifierResult(rewards=None)


def _use(name: str, args: dict, ident: str) -> dict:
    return {"type": "tool_use", "id": ident, "name": name, "input": args}


def _review(criterion: str, verdict: str, quote: str) -> dict:
    return {
        "verdicts": [
            {
                "id": criterion,
                "verdict": verdict,
                "citations": [{"source": "judge", "step": 2, "quote": quote}],
                "rationale": f"python3 /app/main.py printed {quote!r}",
            }
        ],
        "tags": [],
    }


def _data(message: dict) -> str:
    """The text inside a tool result's DATA block."""
    text = message["content"][0]["content"]
    found = re.search(r"<<<DATA \w+\n(.*?)END DATA \w+>>>", text, re.S)
    return found.group(1) if found else ""


class _ScriptedJudge:
    """judge-loop@1's model, scripted: read the evidence, run the submission, a review for
    another criterion (rejected), then the verdict the run's output supports."""

    def __init__(self, replay: dict | None = None) -> None:
        self.bodies: list[dict] = []
        self.replay = replay
        self.accepted: dict | None = None

    def client(self, credentials):
        return self.call

    async def call(self, body, timeout):
        self.bodies.append(json.loads(json.dumps(body)))
        n = len(body["messages"])
        if n == 1:
            content = [_use("read", {"path": "/app/main.py"}, "read-1")]
        elif n == 3:
            content = [_use("run", {"command": "python3 /app/main.py"}, "run-1")]
        elif n == 5:
            content = [_use("submit_review", _review("prints-41", "pass", "42"), "s-1")]
        else:
            output = _data(body["messages"][4]).strip()
            quote = output.splitlines()[-1] if output else "exit"
            review = self.replay or _review(
                "prints-42", "pass" if output == "42" else "fail", quote
            )
            self.accepted = review
            content = [_use("submit_review", review, "s-2")]
        return {"content": content, "usage": {"input_tokens": 900, "output_tokens": 40}}


def _solve(script: Path, root: Path) -> None:
    """Run one of the fixture's solver scripts against a container folder."""
    text = script.read_text().replace("/app/", f"{root}/app/")
    subprocess.run(["bash", "-c", text], check=True)


def _container(tmp_path: Path, case: str) -> Path:
    root = tmp_path / f"container-{case}"
    (root / "app").mkdir(parents=True)
    if case == "oracle":
        _solve(FIXTURE / "oracle" / "solve.sh", root)
    elif case == "decoy":
        _solve(FIXTURE / "controls" / "decoy.sh", root)
    elif case == "split":  # an honest program in two files (the score change)
        (root / "app" / "main.py").write_text(
            "from answer import ANSWER\n\nprint(ANSWER)\n"
        )
        (root / "app" / "answer.py").write_text("ANSWER = 6 * 7\n")
    return root


async def _grade(
    tmp_path: Path,
    case: str,
    judge: _ScriptedJudge,
    monkeypatch,
    *,
    package=FIXTURE,
    box=None,
):
    monkeypatch.setattr(judging, "MessagesClient", judge.client)
    native = materialize_task_dir(package)
    box = box or _Box(_container(tmp_path, case))
    trial = tmp_path / f"trial-{case}"
    (trial / "agent").mkdir(parents=True)
    event = (
        {"type": "nop"}
        if case == "nop"
        else {"type": "oracle", "return_code": 0, "stdout": ""}
    )
    (trial / "agent" / "acp_trajectory.jsonl").write_text(json.dumps(event) + "\n")
    tests = native / "verifier" if (native / "verifier").is_dir() else native / "tests"
    strategy = load_verifier_document(tests).selected_strategy
    result = await verify.verify_taskmd(_Verifier(native, trial, box), strategy)
    return result, trial, box


def _judgments(trial: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (trial / "verifier" / "judgments.jsonl").read_text().splitlines()
    ]


async def test_the_scripted_judge_scores_oracle_nop_and_decoy_1_0_0(
    tmp_path, monkeypatch
) -> None:
    runs: dict[str, str] = {}
    for case, expected in (("oracle", 1.0), ("nop", 0.0), ("decoy", 0.0)):
        judge = _ScriptedJudge()
        result, trial, _ = await _grade(tmp_path, case, judge, monkeypatch)
        assert result.rewards["reward"] == expected, case
        lines = _judgments(trial)
        sessions = [x for x in lines if x["record"] == "session"]
        assert len(sessions) == 1 and sessions[0]["end"] == "accepted", case
        calls = sessions[0]["calls"]
        # judgments.jsonl shows each call: read, run, the rejected review, the accepted one.
        assert [[t["name"] for t in c["tools"]] for c in calls] == [
            ["read"],
            ["run"],
            ["submit_review"],
            ["submit_review"],
        ]
        assert (
            "'prints-41' is not a criterion in the assignment"
            in calls[2]["tools"][0]["result"]
        )
        assert "no verdict for criterion prints-42" in calls[2]["tools"][0]["result"]
        assert calls[3]["tools"][0]["result"] == "Review accepted."
        runs[case] = calls[1]["tools"][0]["result"]
        criterion = next(x for x in lines if x["record"] == "criterion")
        assert criterion["id"] == "prints-42"
        assert criterion["verdict"] == ("pass" if expected else "fail")
        assert criterion["judge"]["provider"] == "anthropic"
        review = json.loads((trial / "verifier" / "review.json").read_text())
        assert review["verdicts"][0]["citations"][0]["verified"] is True
    # The decoy's source reads like the oracle's; only running it shows the difference.
    assert _data({"content": [{"content": runs["oracle"]}]}).strip() == "42"
    assert _data({"content": [{"content": runs["decoy"]}]}).strip() == "41"
    assert runs["decoy"] != runs["oracle"]
    assert "can't open file '/app/main.py'" in runs["nop"]


async def test_a_review_for_another_trial_or_task_is_checked_against_this_assignment(
    tmp_path, monkeypatch
) -> None:
    """judge-loop@1 checks submit_review against the session's assignment.

    A review naming criteria the assignment does not hold (another task's, or
    another unit's) is rejected; three such end the session and leave the trial
    unscored. A review copied from another trial of the same task names the
    same criteria, so its schema passes; its citations are resolved against
    this session's own judge steps, so the oracle's "42" does not verify in the
    decoy's session and the verdict carries ``no-verified-citation``.
    """
    oracle = _ScriptedJudge()
    await _grade(tmp_path, "oracle", oracle, monkeypatch)
    replayed = _ScriptedJudge(replay=oracle.accepted)
    _, trial, _ = await _grade(tmp_path, "decoy", replayed, monkeypatch)
    review = json.loads((trial / "verifier" / "review.json").read_text())
    verdict = review["verdicts"][0]
    assert verdict["citations"][0]["verified"] is False
    assert "no-verified-citation" in verdict["flags"]

    class _Stubborn(_ScriptedJudge):
        async def call(self, body, timeout):
            self.bodies.append(body)
            other_task = _review("mle-method", "pass", "42")
            return {"content": [_use("submit_review", other_task, "s")], "usage": {}}

    monkeypatch.setattr(judging, "SESSIONS", judging.SessionLimit())
    with pytest.raises(verify.TaskMdVerifierError, match="valid samples"):
        await _grade(tmp_path / "again", "oracle", _Stubborn(), monkeypatch)
    lines = _judgments(tmp_path / "again" / "trial-oracle")
    assert [x["end"] for x in lines] == ["rejected", "rejected"]
    assert all(len(x["calls"]) == 3 for x in lines)


async def test_secrets_never_reach_the_judge_or_its_records(
    tmp_path, monkeypatch
) -> None:
    root = _container(tmp_path, "oracle")
    (root / "app" / "notes.txt").write_text(f"key={JUDGE_KEY}\n")
    (root / "app" / "main.py").write_text(
        f"# {JUDGE_KEY}\nprint('{AGENT_TOKEN}')\nprint(6 * 7)\n"
    )
    judge = _ScriptedJudge()
    _, trial, _ = await _grade(tmp_path, "oracle", judge, monkeypatch, box=_Box(root))
    sent = json.dumps(judge.bodies)
    assert "print(6 * 7)" in sent  # the rest of the file is sent
    for secret in (JUDGE_KEY, AGENT_TOKEN):
        assert secret not in sent
        for path in (trial / "verifier").rglob("*.json*"):
            if path.is_file() and "fs" not in path.relative_to(trial).parts:
                assert secret not in path.read_text(), path


def test_a_bearer_token_for_an_anthropic_compatible_endpoint() -> None:
    env = {
        "BENCHFLOW_TASKMD_JUDGE_AUTH_TOKEN": "tp-0123456789",
        "BENCHFLOW_TASKMD_JUDGE_BASE_URL": "https://x.example/anthropic",
    }
    credentials = judging.credentials_from_env(env)
    assert credentials is not None and credentials.kind == "auth-token"
    assert credentials.base_url == "https://x.example/anthropic"
    headers = credentials.headers()
    assert headers["authorization"] == "Bearer tp-0123456789"
    assert "anthropic-beta" not in headers and "x-api-key" not in headers
    assert credentials.system() == judging.jp.SYSTEM_PROMPT_1
    assert credentials.system_note is None
    # The endpoint gets its own token, even with an Anthropic key set.
    assert (
        judging.credentials_from_env({**env, "ANTHROPIC_API_KEY": "k" * 10}).kind
        == "auth-token"
    )
    # Every solver agent is given ANTHROPIC_AUTH_TOKEN, so it is never the judges' credential.
    for other in (
        {"ANTHROPIC_AUTH_TOKEN": "tp-0123456789"},
        {
            "ANTHROPIC_AUTH_TOKEN": "tp-0123456789",
            "ANTHROPIC_BASE_URL": "https://x.example/anthropic",
        },
    ):
        assert judging.credentials_from_env(other) is None
    # One without the other is refused by name.
    for half in env:
        credentials, why = judging.resolve_credentials({half: env[half]})
        assert credentials is None and why is not None and half in why


def test_the_scrub_replaces_the_judges_credential_and_secret_looking_values() -> None:
    credentials = judging.Credentials("api-key", JUDGE_KEY)
    scrub = judging.Scrub.for_run(
        credentials,
        {"AUTH_MODE": "anonymous", "MY_TOKEN": "tok-0123456789abcdef", "HOME": "/h"},
    )
    text = f"{JUDGE_KEY} tok-0123456789abcdef anonymous {AGENT_TOKEN} /h"
    assert scrub.text(text) == (
        "***REDACTED*** ***REDACTED*** anonymous ***REDACTED*** /h"
    )
    assert scrub.obj({"a": [text]}) == {"a": [scrub.text(text)]}


def test_the_judge_sees_the_submissions_own_text_except_keys() -> None:
    """read, run and the llm evidence use Scrub.evidence: exact values and prefixed key
    families. The trajectory patterns' carriers and their boundary-less ``sk-`` rule would
    rewrite what the program did (``password=hunter2secret``) and ordinary words."""
    scrub = judging.Scrub.for_run(judging.Credentials("api-key", JUDGE_KEY), {})
    github = "ghp_" + "a1" * 12
    text = (
        "password=hunter2secret task-specification desk-reservations "
        f"{JUDGE_KEY} key={AGENT_TOKEN} {github}"
    )
    assert scrub.evidence(text) == (
        "password=hunter2secret task-specification desk-reservations "
        "***REDACTED*** key=***REDACTED*** ***REDACTED***"
    )
    # The trajectory depth still masks the carriers.
    assert "hunter2secret" not in scrub.text(text)


async def test_a_program_that_prints_a_password_is_shown_printing_it(
    tmp_path, monkeypatch
) -> None:
    root = _container(tmp_path, "oracle")
    (root / "app" / "main.py").write_text(
        f"print('password=hunter2secret')\nprint('{AGENT_TOKEN}')\nprint(6 * 7)\n"
    )
    judge = _ScriptedJudge()
    _, trial, _ = await _grade(tmp_path, "oracle", judge, monkeypatch, box=_Box(root))
    calls = _judgments(trial)[0]["calls"]
    read, run = calls[0]["tools"][0]["result"], calls[1]["tools"][0]["result"]
    assert "password=hunter2secret" in run and AGENT_TOKEN not in run
    assert "run: exit 0; 1 secret value replaced by ***REDACTED***\n" in run
    assert "password=hunter2secret" in read and AGENT_TOKEN not in read
    assert "1 secret value replaced by ***REDACTED***" in read


def test_a_file_citation_resolves_against_the_text_read_showed(tmp_path) -> None:
    root = tmp_path / "fs"
    (root / "app").mkdir(parents=True)
    (root / "app" / "config.py").write_text(f'TOKEN = "{AGENT_TOKEN}"\nDEBUG = 0\n')
    files = judging.JudgeFiles(root=root, kept=["/app"])
    scrub = judging.Scrub.for_run(None, {})
    text, _, _ = judging.read_result(
        files, {}, 1, "f00d", {"path": "/app/config.py"}, scrub.evidence
    )
    quote = 'TOKEN = "***REDACTED***"'
    assert quote in text and AGENT_TOKEN not in text
    ctx = judging.CitationContext(
        files=files,
        record={},
        tests={},
        instruction=b"",
        workdir=None,
        scrub=scrub.evidence,
    )
    cite = {"source": "file", "path": "/app/config.py", "lines": [1, 1]}
    assert judging.check_citation({**cite, "quote": quote}, ctx, {})["verified"]
    assert judging.check_citation({**cite, "quote": "TOKEN ="}, ctx, {})["verified"]


async def test_pythonsafepath_comes_from_the_task_only(tmp_path, monkeypatch) -> None:
    """An honest program split across two files passes; 0.8.0 set PYTHONSAFEPATH=1, which broke its import."""
    result, trial, box = await _grade(tmp_path, "split", _ScriptedJudge(), monkeypatch)
    assert result.rewards["reward"] == 1.0
    assert all("PYTHONSAFEPATH" not in c for c in box.runs) and box.runs
    package = tmp_path / "with-env"
    shutil.copytree(FIXTURE, package)
    task = package / "task.md"
    task.write_text(
        task.read_text().replace(
            'network = "none"\n\n[verifier.judges.agent]',
            'network = "none"\nenv = { PYTHONSAFEPATH = "1" }\n\n[verifier.judges.agent]',
        )
    )
    assert "PYTHONSAFEPATH" in task.read_text()
    result, trial, box = await _grade(
        tmp_path / "env", "split", _ScriptedJudge(), monkeypatch, package=package
    )
    assert result.rewards["reward"] == 0.0
    assert all("PYTHONSAFEPATH=1 " in c for c in box.runs) and box.runs
    run = _judgments(trial)[0]["calls"][1]["tools"][0]["result"]
    assert "ModuleNotFoundError" in run


async def test_an_image_without_a_tool_to_drop_root_runs_no_command(
    tmp_path, monkeypatch
) -> None:
    box = _Box(_container(tmp_path, "oracle"), drop="BF drop none")
    with pytest.raises(
        verify.TaskMdVerifierError, match="never runs a judge's command as root"
    ):
        await _grade(tmp_path, "oracle", _ScriptedJudge(), monkeypatch, box=box)
    assert box.runs == []
    assert not any("timeout -k" in c for c in box.commands)


async def _script_for(drop_line: str, monkeypatch) -> str:
    box = _Box(Path("/nonexistent"), drop=drop_line)
    runner = verify.SandboxRunner(sandbox=box, kept=["/app"], views={})
    await runner.setup()
    commands: list[str] = []

    async def record(command, **kwargs):
        commands.append(command)
        return _Result(stdout="0 3\nNDIK\n")

    monkeypatch.setattr(box, "exec", record)
    code, timed_out, output = await runner("python3 /app/main.py", 30)
    assert (code, timed_out, output) == (0, False, b"42\n")
    return commands[-1]


async def test_a_command_runs_as_uid_65534_through_the_system_tool(monkeypatch) -> None:
    script = await _script_for("BF drop setpriv /usr/bin/setpriv", monkeypatch)
    # The tool is named by its system path, and env, timeout and bash are looked up after root is dropped.
    assert (
        "/usr/bin/setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs env -i PATH="
        in script
    )
    assert f"bash --noprofile --norc {verify.COMMAND_FILE} </dev/null" in script
    # The command is a file only root can change, written afresh for each call.
    assert (
        f"chmod 444 {verify.COMMAND_FILE}" in script
        and "chmod 755 /run/taskmd-judge-cmd" in script
    )
    script = await _script_for("BF drop runuser /usr/sbin/runuser nobody", monkeypatch)
    assert "/usr/sbin/runuser -u nobody -- env -i PATH=" in script
    script = await _script_for("BF drop su /bin/su nobody", monkeypatch)
    assert "/bin/su -s /bin/sh nobody -c 'exec env -i PATH=" in script
    for line in (
        "BF drop setpriv /usr/bin/setpriv",
        "BF drop runuser /usr/sbin/runuser nobody",
        "BF drop su /bin/su nobody",
    ):
        script = await _script_for(line, monkeypatch)
        subprocess.run(["sh", "-n", "-c", script], check=True)
        subprocess.run(["bash", "-n", "-c", script], check=True)


async def test_a_huge_output_stays_bounded_on_the_host(monkeypatch) -> None:
    """A judged program can print gigabytes within its timeout, or ``ftruncate`` its
    stdout to a sparse file of any size at once; the sandbox sends back its first and
    last 8 KiB and its size. The host keeps only those, so the memory ``run`` takes
    does not grow with the size, and the tool text is shell@1's, as for the same bytes.
    Guards J1-codex-2 (f130d382 rebuilt the middle as NUL bytes: 2 GiB for 1 GiB)."""
    size = 64 * 1024 * 1024
    head, tail = b"a" * 8192, b"z" * 8192
    box = _Box(Path("/nonexistent"))
    runner = verify.SandboxRunner(sandbox=box, kept=["/app"], views={})
    await runner.setup()

    async def huge(command, **kwargs):
        encoded = (base64.b64encode(part).decode() for part in (head, tail))
        return _Result(stdout=f"0 {size}\n" + "\n".join(encoded) + "\n")

    monkeypatch.setattr(box, "exec", huge)
    tracemalloc.start()
    try:
        text, _ = await judging.run_result(runner, 2, "F0", {"command": "yes"})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024, peak
    assert f"\n[... {size - 16384} bytes omitted ...]\n" in text
    assert text.startswith("[judge step 2] run: exit 0\n") and len(text) < 17_000
    # The same text as for the bytes themselves.
    middle = b"m" * 4000
    capped = traj.CappedOutput(head=head, tail=tail, size=16384 + len(middle))
    assert traj.shell_output(capped) == traj.shell_output(head + middle + tail)


async def test_leftovers_are_swept_before_the_restore_and_after_the_command(
    monkeypatch,
) -> None:
    script = await _script_for("BF drop setpriv /usr/bin/setpriv", monkeypatch)
    sweeps = [m.start() for m in re.finditer(r"^bf_sweep \|\| exit 0$", script, re.M)]
    assert len(sweeps) == 2
    restore = script.index("tar -C / -xpf /run/taskmd-judge/kept.tar")
    command = script.index("/run/taskmd-judge-cmd/command </dev/null")
    assert sweeps[0] < restore < command < sweeps[1]
    assert (
        "/usr/bin/setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs /bin/sh -c 'kill -9 -1'"
        in script
    )


def _docker_box(answers) -> tuple[DockerSandbox, list[list[str]]]:
    """A DockerSandbox whose ``docker compose`` calls are recorded and answered by ``answers(script)``."""
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._persistent_env = {"TASK_SETTING": "1"}  # the env-file wrapper runs too
    sandbox.default_user = None
    argvs: list[list[str]] = []

    async def compose(
        command,
        check=False,
        timeout_sec=None,
        *,
        merge_stderr=True,
        raw_output=False,
        stdin_data=None,
    ):
        argvs.append(list(command))
        if command == ["ps", "-q", "main"]:  # a download's container lookup
            return ExecResult(stdout="c" * 64 + "\n", stderr="", return_code=0)
        return ExecResult(stdout=answers(command[-1]), stderr="", return_code=0)

    sandbox._run_docker_compose_command = compose  # type: ignore[method-assign]
    # Downloads stream ``docker cp MAIN:SRC -`` (tests._docker_cp_stream).
    sandbox._docker_compose_global_args = lambda: []  # type: ignore[method-assign]
    sandbox._docker_compose_env = lambda: {}  # type: ignore[method-assign]
    sandbox._compose_transport_env = lambda: {}  # type: ignore[method-assign]
    return sandbox, argvs


async def test_root_never_runs_a_program_from_the_images_path(
    tmp_path, monkeypatch
) -> None:
    """The image's PATH can lead into a kept path (``ENV PATH=/app/.venv/bin:$PATH``
    with /app kept). The shell ``docker compose exec -u root`` starts is looked up
    before any line of the script runs, so every root call the verifier makes (the
    offline check, copy_kept's probe and its tar of a kept folder, the runner's setup,
    its PATH read, a view, each run call, restore_hidden) starts ``/bin/sh`` by absolute
    path, with PATH set
    to the system folders before the env-file wrapper and the script; a ``sh``,
    ``chown`` or ``tar`` the submission leaves there never runs as root. So does the
    Docker sandbox's own chown before it copies a kept folder out. The image's loader
    variables can lead into a kept path too (``ENV LD_LIBRARY_PATH=/app/lib``), and
    ``/bin/sh`` is dynamically linked: each root call starts it with them set empty,
    so a library the submission leaves there is never loaded as root.

    Guards J1-R2-A-1 (py8/agent-judge 7a6f6906 prepended PATH to the script only),
    J1-R3-A-1 (f130d382: the download's chown was a plain ``sh -c``) and J1-R3-B-1
    (f130d382: the image's LD_LIBRARY_PATH chose root's libc)."""
    planted = tmp_path / "app" / ".venv" / "bin"
    planted.mkdir(parents=True)
    tools = (
        "sh",
        "chown",
        "tar",
        "rm",
        "mkdir",
        "chmod",
        "base64",
        "awk",
        "env",
        "head",
        "wc",
    )
    for tool in tools:
        (planted / tool).write_text("#!/bin/sh\nexit 0\n")
        (planted / tool).chmod(0o755)
    image_path = f"{planted}:/usr/bin:/bin"
    # The container's environment, as the image's ENV sets it.
    image_env = {"PATH": image_path}
    if sys.platform.startswith("linux"):  # glibc's loader: a libc that is not one
        lib = tmp_path / "app" / "lib"
        lib.mkdir()
        (lib / "libc.so.6").write_bytes(b"planted")
        image_env |= {"LD_LIBRARY_PATH": str(lib), "BASH_ENV": str(lib / "env.sh")}

    def answers(script: str) -> str:
        if "/proc/net/dev" in script:
            return "offline: no network interface but lo\n"
        if "then echo link" in script:
            return "dir\n"
        if 'echo "BF copy' in script:
            # Model a bounded archive rejected at its cap. The assertions below
            # still exercise the root shell of capture and its cleanup.
            return f"BF copy {verify.COPY_TAR_LIMIT} 0\n"
        if 'echo "BF drop' in script:
            return "BF drop setpriv /usr/bin/setpriv\n"
        if "cat /run/taskmd-judge/path" in script:  # the image's PATH, in the container
            return "/app/.venv/bin:/usr/bin:/bin\n"
        if verify.COMMAND_FILE in script:
            return "0 0\n\n"
        return ""

    box, argvs = _docker_box(answers)
    await verify.take_offline(box)
    await verify.copy_kept(box, ["/app"], tmp_path / "fs")
    # Direct backend downloads still need the same safe root shell, even
    # though judge capture now uses the bounded archive instead.
    copies = fake_docker_cp(monkeypatch, {})
    await box.download_dir(source_dir="/app", target_dir=tmp_path / "fs" / "app")
    assert (tmp_path / "fs" / "app").is_dir()
    (tmp_path / "fs" / "app" / "main.py").write_text("print(6 * 7)\n")
    runner = verify.SandboxRunner(
        sandbox=box,
        kept=["/app"],
        views={"/srv/view/trajectory.jsonl": b"{}\n"},
        custody=tmp_path / "fs",
    )
    await runner("python3 /app/main.py", 30)
    await verify.restore_hidden(box)
    # The download streams the folder as a tar and publishes it through held
    # descriptors: no root shell runs for it at all, and nothing
    # is staged beside the custody tree.
    assert copies == [("docker", "cp", f"{'c' * 64}:/app", "-")]
    assert not any(a[:2] == ["cp", "main:/app/."] for a in argvs)
    assert sorted(p.name for p in (tmp_path / "fs").iterdir()) == ["app"]
    assert any(
        a[0] == "cp" and a[-1] == "main:/run/taskmd-judge/kept.tar" for a in argvs
    )
    execs = [a for a in argvs if a[0] == "exec"]
    # offline, probe, root's bounded tar and its cleanup, setup, the upload's mkdir, the stash's check,
    # the image PATH, the view, one run call, restore_hidden. The streamed
    # download needs no chown: the controller creates every file it receives.
    assert len(execs) == 11, execs
    assert "tar -C / -cf - ./app" in execs[2][-1]
    assert f"rm -rf {verify.COPY_DIR}" in execs[3][-1]
    assert not any("chown -R" in a[-1] and "/app" in a[-1].split()[-1] for a in execs)
    assert "mkdir -p /run/taskmd-judge" in execs[5][-1]
    lookup = "for t in " + " ".join(tools) + "; do command -v $t; done"
    for argv in execs:
        service = argv.index("main")
        assert argv[:4] == ["exec", "-T", "-u", "root"], argv
        options = argv[4:service]
        shell, flag, script = argv[service + 1 :]
        assert flag == "-c"
        # docker exec -e NAME=VALUE sets NAME in the container's environment.
        assert options[::2] == ["-e"] * (len(options) // 2), argv
        env = image_env | dict(o.split("=", 1) for o in options[1::2])
        # Docker resolves the shell on the container's PATH (the image's) when it has no slash.
        resolved = shutil.which(shell, path=image_path)
        assert resolved is not None and not resolved.startswith(str(planted)), argv
        assert script.startswith(with_system_path("")), script[:200]
        assert verify.ROOT_PATH in script
        done = subprocess.run(
            [resolved, "-c", f"{script.splitlines()[0]}\n{lookup}\nenv"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, (argv[:service], done.stderr)
        found = done.stdout.split()[: len(tools)]
        assert len(found) == len(tools), (script[:200], found)
        assert not any(f.startswith(str(planted)) for f in found), (script[:200], found)
        left = [
            v for v in done.stdout.split()[len(tools) :] if v.split("=")[0] in image_env
        ]
        assert left == [f"PATH={SYSTEM_PATH}"], left
    # The judge's own command still gets the image's PATH, without the kept path.
    run = next(a for a in execs if verify.COMMAND_FILE in a[-1])
    assert "env -i PATH=/usr/bin:/bin HOME=" in run[-1]


async def test_the_runner_refuses_a_sandbox_that_cannot_start_roots_shell_by_path() -> (
    None
):
    """A backend whose ``exec`` always lets the image's PATH pick the shell (Modal, for
    one) is refused by name before the runner runs anything as root, and restore_hidden
    then runs nothing. Guards J1-R2-A-1."""
    calls: list[str] = []

    class _PathShell:
        async def exec(self, command, **kwargs):
            calls.append(command)
            return _Result()

    box = _PathShell()
    runner = verify.SandboxRunner(sandbox=box, kept=["/app"], views={})
    with pytest.raises(
        verify.TaskMdVerifierError, match="start root's shell by absolute path"
    ):
        await runner.setup()
    await verify.restore_hidden(box)
    assert calls == []
    assert f"PATH={SYSTEM_PATH}; export PATH" == verify.ROOT_PATH


async def test_docker_and_daytona_start_a_system_shell_only_when_asked(
    monkeypatch,
) -> None:
    """``exec(..., system_shell=True)``: Docker and Daytona DinD start ``/bin/sh`` by
    absolute path with the loader variables set empty and PATH set first; Daytona's
    direct sandbox unsets the loader and start-up variables and sets PATH before ``su``
    and ``timeout`` are looked up. Without the option the commands are as before.
    Guards J1-R2-A-1 and J1-R3-B-1."""
    box, argvs = _docker_box(lambda script: "")
    box._persistent_env = {}
    await box.exec("echo hi", user="root")
    await box.exec("echo hi", user="root", system_shell=True)
    # Without the option root is still plumbing (C8): /bin/sh, the loader
    # values empty, PATH pinned (with /usr/local, where python3 can be).
    assert argvs[0] == [
        "exec", "-T", "-u", "root", *system_shell_env_flags(), "main", "/bin/sh", "-c",
        with_system_path("echo hi", ROOT_EXEC_PATH),
    ]  # fmt: skip
    assert argvs[1] == [
        "exec", "-T", "-u", "root", *system_shell_env_flags(), "main", "/bin/sh", "-c",
        with_system_path("echo hi"),
    ]  # fmt: skip
    for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "BASH_ENV", "ENV"):
        assert f"{name}=" in argvs[1] and name in SYSTEM_SHELL_UNSET_VARS
    assert with_system_path("echo hi").startswith(system_shell_prefix())
    assert "; unset LD_PRELOAD LD_LIBRARY_PATH " in system_shell_prefix()

    pytest.importorskip("daytona")  # sandbox-daytona optional dependency
    from benchflow.sandbox.daytona import (
        DaytonaSandbox,
        _DaytonaDinD,
        _DaytonaDirect,
        _load_daytona_sdk,
    )

    _load_daytona_sdk()
    sent: list[str] = []

    class _Process:
        async def create_session(self, session_id):
            pass

        async def execute_session_command(self, session_id, request, timeout=None):
            sent.append(request.command)
            return SimpleNamespace(cmd_id="cmd-1")

    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._sandbox = SimpleNamespace(process=_Process())
    sandbox._strategy = _DaytonaDirect(sandbox)

    async def poll(*args, **kwargs):
        return ExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(sandbox, "_poll_response", poll)
    assert sandbox.supports_system_shell
    await sandbox.exec("echo hi", user="root", timeout_sec=30)
    await sandbox.exec("echo hi", user="root", timeout_sec=30, system_shell=True)
    assert sent[0].startswith(f"{system_shell_prefix(ROOT_EXEC_PATH)}; /bin/sh -c ")
    assert "/bin/su root -s /bin/sh -c" in sent[0]
    assert "/usr/bin/timeout 30 /bin/sh -c" in sent[0]
    assert sent[1].startswith(f"{system_shell_prefix()}; /bin/sh -c "), sent[1]
    assert "/usr/bin/timeout 30 /bin/sh -c" in sent[1]
    assert "timeout 30 /bin/bash" not in sent[1]

    strategy = _DaytonaDinD.__new__(_DaytonaDinD)
    parts: list[list[str]] = []

    async def compose_exec(command, timeout_sec=None, cleanup_session=False):
        parts.append(list(command))
        return ExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(strategy, "_compose_exec", compose_exec)
    await strategy.exec("echo hi", user="root", route=SYSTEM_SHELL_ROUTE)
    flags = system_shell_env_flags()
    assert parts[0][: 6 + len(flags)] == [
        "exec", "-T", "-u", "root", *flags, "main", "/bin/sh",
    ]  # fmt: skip
    assert parts[0][-1].startswith(with_system_path(""))


@dataclass
class _Planting(_Box):
    """A container that, while the judge's command runs, leaves a link (or a FIFO) in the
    bind-mounted verifier folder where BenchFlow writes next."""

    verifier_dir: Path | None = None
    name: str = ""
    target: Path | None = None
    readers: list[int] = field(default_factory=list)

    def run(self, command: str, script: str) -> tuple[int, bytes]:
        assert self.verifier_dir is not None
        planted = self.verifier_dir / self.name
        planted.unlink(missing_ok=True)
        if self.target is None:
            os.mkfifo(planted)
            # A reader holds it open, so a plain open() would not block, it would write into it.
            self.readers.append(os.open(planted, os.O_RDONLY | os.O_NONBLOCK))
        else:
            planted.symlink_to(self.target)
        return super().run(command, script)


@pytest.mark.parametrize(
    "name", ["judgments.jsonl", "review.json", "reward.json", "reward.txt"]
)
async def test_the_host_never_writes_through_a_link_in_the_verifier_folder(
    tmp_path, monkeypatch, name
) -> None:
    victim = tmp_path / "victim-a"
    victim.write_text("before\n")
    box = _Planting(
        _container(tmp_path, "oracle"),
        verifier_dir=tmp_path / "trial-oracle" / "verifier",
        name=name,
        target=victim,
    )
    with pytest.raises(verify.TaskMdVerifierError, match=f"did not write {name}"):
        await _grade(tmp_path, "oracle", _ScriptedJudge(), monkeypatch, box=box)
    assert victim.read_text() == "before\n"
    assert (box.verifier_dir / name).is_symlink()


async def test_the_host_never_writes_into_a_fifo_in_the_verifier_folder(
    tmp_path, monkeypatch
) -> None:
    box = _Planting(
        _container(tmp_path, "oracle"),
        verifier_dir=tmp_path / "trial-oracle" / "verifier",
        name="judgments.jsonl",
    )
    try:
        with pytest.raises(verify.TaskMdVerifierError, match="did not write judgments"):
            await _grade(tmp_path, "oracle", _ScriptedJudge(), monkeypatch, box=box)
        assert box.readers
        assert os.read(box.readers[0], 4096) == b""
    finally:
        for fd in box.readers:
            os.close(fd)


@dataclass
class _Swapping(_Box):
    """Root code in the verifier sandbox (test.sh, or something it left running) that, once
    the judges' copy is made, swaps ``taskmd-judge/fs/app/main.py`` in the bind-mounted
    verifier folder for a link to a host file."""

    verifier_dir: Path | None = None
    target: Path | None = None

    async def exec(self, command, **kwargs):
        if 'echo "BF drop' in command:
            assert self.verifier_dir is not None and self.target is not None
            planted = self.verifier_dir / "taskmd-judge" / "fs" / "app" / "main.py"
            planted.parent.mkdir(parents=True, exist_ok=True)
            planted.unlink(missing_ok=True)
            planted.symlink_to(self.target)
        return await super().exec(command, **kwargs)


async def test_the_judges_never_read_a_host_file_through_a_link_in_the_verifier_folder(
    tmp_path, monkeypatch
) -> None:
    """``read``, file citations and the llm evidence read the judges' copy on the host. It
    lives outside the verifier folder the sandbox mounts read-write, so a link planted there
    after the copy never leads them to a host file (a key that Scrub.evidence would not mask).
    Guards J1-R2-A-3."""
    victim = tmp_path / "victim-a"
    victim.write_text(
        "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----\n"
    )
    root = _container(tmp_path, "oracle")
    saved = (root / "app" / "main.py").read_text()
    box = _Swapping(
        root, verifier_dir=tmp_path / "trial-oracle" / "verifier", target=victim
    )
    judge = _ScriptedJudge()
    result, trial, _ = await _grade(tmp_path, "oracle", judge, monkeypatch, box=box)
    assert "PRIVATE KEY" not in json.dumps(judge.bodies)
    read = _judgments(trial)[0]["calls"][0]["tools"][0]["result"]
    assert saved.splitlines()[0] in read
    assert result.rewards["reward"] == 1.0
    for path in trial.rglob("*"):
        if path.is_file() and not path.is_symlink():
            assert "PRIVATE KEY" not in path.read_text(errors="replace"), path


@dataclass
class _Restoring(_Box):
    """A container whose ``run`` sees only what the runner restores before each call: its
    stash (a tar the sandbox made of its own kept paths at setup, or the one BenchFlow
    uploaded), unpacked into a fresh folder, as the runner's script does."""

    stash: bytes | None = None

    async def exec(self, command, **kwargs):
        if "tar -cf /run/taskmd-judge/kept.tar $present" in command:  # the live sandbox
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as tar:
                tar.add(self.root / "app", arcname="app")
            self.stash = buffer.getvalue()
        return await super().exec(command, **kwargs)

    async def upload_file(self, source, target, **kwargs):
        await super().upload_file(source, target, **kwargs)
        if target == "/run/taskmd-judge/kept.tar":
            self.stash = Path(source).read_bytes()

    def run(self, command: str, script: str) -> tuple[int, bytes]:
        live = self.root
        restored = live.with_name(live.name + "-restored")
        shutil.rmtree(restored, ignore_errors=True)
        restored.mkdir()
        if self.stash:
            with tarfile.open(fileobj=io.BytesIO(self.stash)) as tar:
                tar.extractall(restored, filter="data")
        self.root = restored
        try:
            return super().run(command, script)
        finally:
            self.root = live


async def test_run_executes_the_submission_read_shows_whatever_test_sh_changed(
    tmp_path, monkeypatch
) -> None:
    """``read`` serves the judges' host copy, made before test.sh runs; every ``run`` call
    restores the saved outputs from that same copy, not from the sandbox test.sh ran in. A
    test.sh (or the submission code it runs) that rewrites a kept file cannot change what
    the judge executes: the decoy still prints 41 though test.sh wrote ``ANSWER = 42``.
    Guards J1-codex-1 (f130d382 stashed the outputs from the sandbox after test.sh)."""
    root = _container(tmp_path, "decoy")
    copy = verify.copy_kept

    async def then_test_sh(sandbox, paths, fs_root, *rest):
        refused = await copy(sandbox, paths, fs_root, *rest)
        (root / "app" / "answer.py").write_text("ANSWER = 42\n")  # what test.sh writes
        return refused

    monkeypatch.setattr(verify, "copy_kept", then_test_sh)
    box = _Restoring(root)
    judge = _ScriptedJudge()
    result, trial, _ = await _grade(tmp_path, "decoy", judge, monkeypatch, box=box)
    calls = _judgments(trial)[0]["calls"]
    read, run = (calls[i]["tools"][0]["result"] for i in (0, 1))
    assert "from answer import ANSWER" in read
    assert _data({"content": [{"content": run}]}).strip() == "41"
    assert result.rewards["reward"] == 0.0
    assert box.uploads == ["/run/taskmd-judge/kept.tar"]
    with tarfile.open(fileobj=io.BytesIO(box.stash or b"")) as tar:
        assert sorted(tar.getnames()) == ["app", "app/answer.py", "app/main.py"]
        assert all(m.uid == 0 and m.gid == 0 for m in tar.getmembers())


def test_the_judges_folders_must_not_be_links(tmp_path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "taskmd-judge"
    linked.symlink_to(real)
    verify._real_dirs(real)
    with pytest.raises(verify.TaskMdVerifierError, match="taskmd-judge"):
        verify._real_dirs(real, linked)


async def test_a_failed_restore_stops_the_judging(monkeypatch) -> None:
    script = await _script_for("BF drop setpriv /usr/bin/setpriv", monkeypatch)
    assert (
        "if [ -s /run/taskmd-judge/kept.tar ] && ! tar -C / -xpf /run/taskmd-judge/kept.tar"
        in script
    )
    box = _Box(Path("/nonexistent"))
    runner = verify.SandboxRunner(sandbox=box, kept=["/app"], views={})
    await runner.setup()
    for line, why in (
        ("BF restore-failed tar: short read", "could not be restored"),
        ("BF survivors", "would not die"),
        ("BF prepare-failed", "could not be prepared"),
    ):

        async def answer(command, line=line, **kwargs):
            return _Result(stdout=line + "\n")

        monkeypatch.setattr(box, "exec", answer)
        with pytest.raises(verify.TaskMdVerifierError, match=why):
            await runner("python3 /app/main.py", 30)


# The runner's shell, run for real against folders ----------------------------------------------


def _fake_stat(tmp_path: Path) -> Path:
    """A ``stat`` that says root owns everything but paths naming ``notroot``, with real modes."""
    bin_dir = tmp_path / "standin-bin"
    bin_dir.mkdir()
    stat_tool = bin_dir / "stat"
    stat_tool.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "path = sys.argv[-1]\n"
        "try:\n"
        "    st = os.stat(path)\n"
        "except OSError:\n"
        "    sys.exit(1)\n"
        "uid = 1000 if 'notroot' in path else 0\n"
        "print(uid, format(st.st_mode & 0o7777, 'o'))\n"
    )
    stat_tool.chmod(0o755)
    return bin_dir


def _tool(path: Path, help_text: str = "", mode: int = 0o755) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho '{help_text}'\n")
    path.chmod(mode)


def _probe(root: Path, bin_dir: Path) -> str:
    for d in verify.DROP_DIRS:
        (root / d.lstrip("/")).mkdir(parents=True, exist_ok=True)
        (root / d.lstrip("/")).chmod(0o755)
    passwd = root / "passwd"
    if not passwd.exists():
        passwd.write_text(
            "root:x:0:0::/root:/bin/sh\nnobody:x:65534:65534::/:/sbin/nologin\n"
        )
    done = subprocess.run(
        ["sh", "-c", verify.drop_probe(str(root), str(passwd))],
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip().splitlines()[-1]


def test_the_drop_probe_trusts_only_root_owned_system_tools(tmp_path) -> None:
    bin_dir = _fake_stat(tmp_path)
    base = tmp_path.resolve()
    # setpriv with --reuid, in a system folder: taken.
    root = base / "a"
    _tool(root / "usr/bin/setpriv", "--reuid=<uid>")
    assert _probe(root, bin_dir) == f"BF drop setpriv {root}/usr/bin/setpriv"
    # busybox's setpriv (no --reuid) is skipped for runuser.
    root = base / "b"
    _tool(root / "usr/bin/setpriv", "--nnp")
    _tool(root / "usr/sbin/runuser")
    assert _probe(root, bin_dir) == f"BF drop runuser {root}/usr/sbin/runuser nobody"
    # A tool others can write is not trusted, nor one root does not own.
    root = base / "c"
    _tool(root / "usr/bin/setpriv", "--reuid", mode=0o757)
    _tool(root / "sbin/runuser", mode=0o775)
    assert _probe(root, bin_dir) == "BF drop none"
    root = base / "c-notroot"
    _tool(root / "usr/bin/setpriv", "--reuid")
    assert _probe(root, bin_dir) == "BF drop none"
    # Alpine's su: a link to busybox in a system folder is named by its own path.
    root = base / "d"
    _tool(root / "bin/busybox")
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "bin/su").symlink_to("busybox")
    assert _probe(root, bin_dir) == f"BF drop su {root}/bin/su nobody"
    # A link out of the system folders is not trusted; without uid 65534 there is no runuser or su.
    root = base / "e"
    _tool(root / "opt/su")
    (root / "bin").mkdir(parents=True)
    (root / "bin/su").symlink_to(root / "opt/su")
    assert _probe(root, bin_dir) == "BF drop none"
    root = base / "f"
    _tool(root / "usr/sbin/runuser")
    (root / "passwd").parent.mkdir(parents=True, exist_ok=True)
    (root / "passwd").write_text("root:x:0:0::/root:/bin/sh\n")
    assert _probe(root, bin_dir) == "BF drop none"
    # What the probe prints parses; anything else is no tool.
    assert verify.parse_drop("BF drop su /bin/su nobody\n") == (
        "su",
        "/bin/su",
        "nobody",
    )
    assert verify.parse_drop("BF drop none\n") is None
    assert verify.parse_drop("BF drop su /bin/su no;body\n") is None
    assert verify.parse_drop("root\n") is None


def _fake_proc(tmp_path: Path, processes: dict[int, tuple[int, str]]) -> Path:
    """A /proc of made-up pids above Linux's pid_max, so no real process is ever named."""
    proc = tmp_path / "proc"
    for pid, (uid, state) in processes.items():
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "status").write_text(
            f"Name:\tx\nState:\t{state} (x)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
        )
    return proc


def _sweep(tmp_path: Path, proc: Path, rounds: int, dies: bool) -> tuple[int, str, str]:
    """Run the sweep with ``kill`` a shell function that logs (and, if ``dies``, removes the
    process's status file, except PID 1's: a container's main process ignores SIGKILL from
    inside); the drop tool is a stand-in that only records its arguments."""
    standin = tmp_path / "drop-standin"
    log = tmp_path / "log"
    standin.write_text(f"#!/bin/sh\necho \"drop $*\" >> '{log}'\n")
    standin.chmod(0o755)
    kill = (
        f"kill() {{ echo \"kill $*\" >> '{log}'; "
        + (
            f'case "$2" in 1) ;; *) rm -f \'{proc}\'/"$2"/status;; esac; '
            if dies
            else ""
        )
        + "return 0; }\n"
    )
    script = (
        kill
        + verify.sweep_script(
            ("setpriv", str(standin), None), proc=str(proc), rounds=rounds
        )
        + "\nbf_sweep\n"
    )
    done = subprocess.run(
        ["sh", "-c", script], capture_output=True, text=True, timeout=60
    )
    return done.returncode, done.stdout, log.read_text() if log.exists() else ""


def test_the_sweep_kills_every_live_process_of_the_runner_and_names_survivors(
    tmp_path,
) -> None:
    processes = {
        99_000_101: (65534, "S"),
        99_000_102: (0, "S"),  # root's: never touched
        99_000_103: (65534, "Z"),  # a zombie is dead already
        99_000_104: (65534, "D"),
        99_000_105: (1000, "R"),
    }
    proc = _fake_proc(tmp_path / "one", processes)
    code, out, log = _sweep(tmp_path / "one", proc, rounds=5, dies=True)
    assert code == 0 and "BF survivors" not in out
    kills = [ln for ln in log.splitlines() if ln.startswith("kill ")]
    assert kills == ["kill -9 99000101", "kill -9 99000104"]
    assert (
        log.count(
            "--reuid=65534 --regid=65534 --clear-groups --no-new-privs /bin/sh -c kill -9 -1"
        )
        == 2
    )
    proc = _fake_proc(tmp_path / "two", {99_000_201: (65534, "R")})
    code, out, log = _sweep(tmp_path / "two", proc, rounds=3, dies=False)
    assert code == 1 and out.strip() == "BF survivors"
    assert log.count("kill -9 99000201") == 3


def test_the_sweep_never_counts_pid_1(tmp_path) -> None:
    """An image whose USER is 65534 runs the container's main process as that uid; PID 1
    ignores SIGKILL from inside, so counting it made the sweep spin and fail every run."""
    proc = _fake_proc(tmp_path, {1: (65534, "S"), 99_000_301: (65534, "S")})
    code, out, log = _sweep(tmp_path, proc, rounds=5, dies=True)
    assert code == 0 and "BF survivors" not in out
    kills = [ln for ln in log.splitlines() if ln.startswith("kill ")]
    assert kills == ["kill -9 99000301"]


def _busy(proc: Path) -> str:
    done = subprocess.run(
        ["sh", "-c", verify.busy_script(str(proc))],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return done.stdout.strip()


def test_setup_finds_processes_already_running_as_the_runners_uid(tmp_path) -> None:
    proc = _fake_proc(
        tmp_path / "busy",
        {1: (65534, "S"), 99_000_401: (0, "S"), 99_000_402: (65534, "Z")},
    )
    out = _busy(proc)
    assert out == "BF busy 1:x"
    assert verify.parse_busy(out) == ["pid 1 (x)"]
    proc = _fake_proc(tmp_path / "free", {1: (0, "S"), 99_000_403: (65534, "Z")})
    assert _busy(proc) == ""
    assert verify.parse_busy("") == []
    assert verify.parse_busy("BF busy 1:a;b") == ["processes it could not name"]


async def test_an_image_whose_user_is_65534_is_refused_by_name_before_any_run(
    tmp_path, monkeypatch
) -> None:
    box = _Box(
        _container(tmp_path, "oracle"),
        drop="BF busy 1:sleep\nBF drop setpriv /usr/bin/setpriv",
    )
    with pytest.raises(
        verify.TaskMdVerifierError,
        match=r"needs uid 65534 to itself.*pid 1 \(sleep\).*USER",
    ):
        await _grade(tmp_path, "oracle", _ScriptedJudge(), monkeypatch, box=box)
    assert box.runs == []
    setup = next(c for c in box.commands if 'echo "BF drop' in c)
    assert verify.busy_script() in setup


# Refusals, and what bench tasks check plans ---------------------------------------------------------


def _variant(
    tmp_path: Path, name: str, old: str, new: str, *, dockerfile=False
) -> Path:
    package = tmp_path / name
    shutil.copytree(FIXTURE, package)
    task = package / "task.md"
    text = task.read_text()
    assert old in text
    task.write_text(text.replace(old, new))
    if dockerfile:
        (package / "verifier" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    return package


REFUSALS = [
    (
        "shared",
        'isolation = "separate"',
        'isolation = "shared"',
        False,
        'BenchFlow provides that only with [verifier] isolation = "separate"',
    ),
    (
        "network",
        'isolation = "separate"\nnetwork = "none"',
        'isolation = "separate"\nnetwork = "open"',
        False,
        "the verifier sandbox BenchFlow runs them in has network 'open'. Set [verifier] network = \"none\"",
    ),
    (
        "image",
        'network = "none"\n\n[verifier.judges.agent]',
        'network = "none"\n\n[verifier.sandbox]\nimage = "python:3.12-slim"\n\n[verifier.judges.agent]',
        False,
        "an agent judge's runner starts from the solver's image, and this verifier runs another image",
    ),
    (
        "dockerfile",
        "x",
        "x",
        True,
        "an agent judge's runner starts from the solver's image, and this verifier builds its own from verifier/Dockerfile",
    ),
]


@pytest.mark.parametrize(("name", "old", "new", "dockerfile", "why"), REFUSALS)
def test_an_agent_judge_bench_cannot_serve_is_refused_by_name_before_the_trial(
    tmp_path, name, old, new, dockerfile, why
) -> None:
    package = _variant(tmp_path, name, old, new, dockerfile=dockerfile)
    # Run: materializing the package (before any sandbox starts) refuses it by name.
    with pytest.raises(TaskMdError) as refused:
        TaskMdFormat().materialize(package, tmp_path / "out")
    assert any(
        r.startswith("[verifier.judges.agent]") and why in r
        for r in refused.value.reasons
    ), refused.value.reasons
    # bench tasks check makes the same plan and refuses the same field, with the same words.
    report = check_package(package)
    assert not report.ok and report.native_dir is None
    assert any(
        line.startswith("refused: [verifier.judges.agent]: ") and why in line
        for line in report.benchflow
    ), report.benchflow
    from benchflow.cli.main import app

    out = CliRunner().invoke(
        app, ["tasks", "check", str(package)], env={"COLUMNS": "400"}
    )
    assert out.exit_code == 1
    assert "refused: [verifier.judges.agent]:" in out.output


USER_REFUSALS = [
    ("nobody", "FROM python:3.12-slim\nWORKDIR /app\nUSER nobody\n", "USER nobody"),
    (
        "uid",
        "FROM python:3.12-slim AS base\nUSER 65534:65534\n\nFROM base\nWORKDIR /app\n",
        "USER 65534:65534",
    ),
]


@pytest.mark.parametrize(("name", "dockerfile", "user"), USER_REFUSALS)
def test_an_image_whose_user_is_the_runners_uid_is_refused_before_the_trial(
    tmp_path, name, dockerfile, user
) -> None:
    """The runner needs uid 65534 to itself, so the verifier sandbox refuses an image whose
    USER is 65534 at grading, after the solver ran. The plan and bench tasks check refuse
    it first, from sandbox/Dockerfile's final USER. Guards J1-R2-A-2."""
    package = tmp_path / name
    shutil.copytree(FIXTURE, package)
    (package / "sandbox" / "Dockerfile").write_text(dockerfile)
    why = f"needs uid 65534 to itself, and sandbox/Dockerfile ends with {user}"
    with pytest.raises(TaskMdError) as refused:
        TaskMdFormat().materialize(package, tmp_path / "out")
    assert any(
        r.startswith("[verifier.judges.agent]") and why in r
        for r in refused.value.reasons
    ), refused.value.reasons
    report = check_package(package)
    assert not report.ok
    assert any(
        line.startswith("refused: [verifier.judges.agent]: ") and why in line
        for line in report.benchflow
    ), report.benchflow


def test_the_final_user_of_a_dockerfile(tmp_path) -> None:
    from benchflow.taskmd.plan import dockerfile_user

    cases = {
        "FROM alpine\n": None,
        "FROM alpine\nUSER nobody\nUSER root\n": "root",
        "from alpine\n# USER nobody\nuser \\\n  nobody\n": "nobody",
        'FROM alpine\nUSER "65534"\n': "65534",
        # The last stage starts from an image, whose USER is not known here.
        "FROM alpine AS a\nUSER nobody\nFROM debian\nRUN true\n": None,
        "FROM --platform=linux/amd64 alpine AS A\nUSER nobody\nFROM a AS b\nFROM b\n": "nobody",
    }
    for text, user in cases.items():
        (tmp_path / "Dockerfile").write_text(text)
        assert dockerfile_user(tmp_path / "Dockerfile") == user, text
    assert dockerfile_user(tmp_path / "missing") is None


@pytest.mark.parametrize("role", ["vlm", "panel"])
def test_vlm_and_panel_criteria_stay_refused_in_0_8s_words(tmp_path, role) -> None:
    package = tmp_path / role
    shutil.copytree(FIXTURE, package)
    rubric = package / "verifier" / "rubric.json"
    rubric.write_text(
        rubric.read_text().replace('"judge": "agent"', f'"judge": "{role}"')
    )
    words = f"the {role} role is not implemented" + (
        " (task.md leaves the panel role undefined)" if role == "panel" else ""
    )
    report = check_package(package)
    assert f"refused: verifier/rubric.json prints-42: {words}" in report.benchflow


def test_check_prints_the_judges_plan() -> None:
    report = check_package(FIXTURE)
    assert report.ok
    line = next(
        x for x in report.benchflow if x.startswith("honored: [verifier.judges.agent]")
    )
    assert (
        "1 criterion in 1 session per sample (per = rubric), 1 sample, each tried at most twice, "
        "180 s per session; tools read, run and submit_review" in line
    )


async def test_another_providers_model_runs_through_an_anthropic_compatible_endpoint(
    tmp_path, monkeypatch
) -> None:
    from benchflow.taskmd.launch import TaskMdLaunchRefused, check_launch

    native = materialize_task_dir(FIXTURE)
    monkeypatch.setenv(judging.MODEL_ENV, "mimo-v2.6-flash")
    monkeypatch.setenv("BENCHFLOW_TASKMD_JUDGE_AUTH_TOKEN", "tp-0123456789")
    monkeypatch.setenv(
        "BENCHFLOW_TASKMD_JUDGE_BASE_URL",
        "https://user:pw@gateway.example:8443/anthropic",
    )
    check_launch(native, primary_agent="oracle", sandbox_user=None)
    monkeypatch.delenv("BENCHFLOW_TASKMD_JUDGE_BASE_URL")
    with pytest.raises(TaskMdLaunchRefused, match="BENCHFLOW_TASKMD_JUDGE_BASE_URL"):
        check_launch(native, primary_agent="oracle", sandbox_user=None)
    monkeypatch.setenv(
        "BENCHFLOW_TASKMD_JUDGE_BASE_URL",
        "https://user:pw@gateway.example:8443/anthropic",
    )
    judge = _ScriptedJudge()
    result, trial, _ = await _grade(tmp_path, "oracle", judge, monkeypatch)
    assert (
        result.rewards["reward"] == 1.0
        and judge.bodies[0]["model"] == "mimo-v2.6-flash"
    )
    review = json.loads((trial / "verifier" / "review.json").read_text())
    assert (
        review["verdicts"][0]["judge"]["provider"]
        == "anthropic-compatible gateway.example:8443"
    )
    assert "pw" not in (trial / "verifier" / "judgments.jsonl").read_text()


class _SdkDownloads(_Restoring):
    """Daytona's downloads: the SDK writes each file with ``open(dst, "wb")``, so it has
    the host's mode (0644), and it never makes a folder that holds no file. The probe
    copy_kept runs is a real shell on the container folder, so the mode record is what
    ``find`` and ``stat`` print there."""

    async def exec(self, command, **kwargs):
        if "then echo link" in command:
            self.commands.append(command)
            done = subprocess.run(
                ["sh", "-c", command.replace(" /app", f" {self.root}/app")],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return _Result(stdout=done.stdout.replace(str(self.root), ""))
        return await super().exec(command, **kwargs)

    async def download_dir(self, source_dir, target_dir):
        source = self.root / source_dir.lstrip("/")
        for found in sorted(source.rglob("*")):
            if found.is_file():
                dest = Path(target_dir) / found.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(found.read_bytes())
                dest.chmod(0o644)


async def test_run_gets_the_modes_and_folders_the_sandbox_had_not_the_downloads(
    tmp_path, monkeypatch
) -> None:
    """On Daytona the judges' copy of the saved outputs has every file at 0644 and no
    empty folder, and ``run`` restores from that copy. A compiled program or a script
    the submission made executable must still run, and a folder it left empty must
    still be there: copy_kept records each mode inside the sandbox before test.sh, and
    the stash takes the modes and folders from that record (permission bits only).
    Guards J1-R1-A-1 (df1b0347: ``run /app/bin/tool`` got Permission denied, so a
    correct submission scored 0)."""
    root = _container(tmp_path, "split")
    (root / "app" / "main.py").write_text(
        "import pathlib, subprocess\n"
        "tool = pathlib.Path(__file__).parent / 'bin' / 'tool'\n"
        "print(subprocess.run([str(tool)], capture_output=True, text=True).stdout.strip())\n"
    )
    tool = root / "app" / "bin" / "tool"
    tool.parent.mkdir()
    tool.write_text(
        '#!/bin/sh\n[ -d "$(dirname "$0")/../out" ] || { echo no out folder; exit 1; }\necho 42\n'
    )
    tool.chmod(0o4755)  # setuid in the sandbox: the stash keeps only 0755
    (root / "app" / "out").mkdir()
    box = _SdkDownloads(root)
    judge = _ScriptedJudge()
    result, trial, _ = await _grade(tmp_path, "split", judge, monkeypatch, box=box)
    run = _judgments(trial)[0]["calls"][1]["tools"][0]["result"]
    assert _data({"content": [{"content": run}]}).strip() == "42", run
    assert result.rewards["reward"] == 1.0
    # Bounded tar capture preserves modes on the host and in the runner stash.
    fs = trial / "verifier" / "taskmd-judge" / "fs" / "app"
    assert (fs / "bin" / "tool").stat().st_mode & 0o777 == 0o755
    with tarfile.open(fileobj=io.BytesIO(box.stash or b"")) as tar:
        members = {m.name: m for m in tar.getmembers()}
    assert members["app/bin/tool"].mode == 0o755
    assert members["app/out"].isdir()


def test_the_mode_record_names_only_what_is_under_the_kept_path() -> None:
    lines = [
        "d 755 /app",
        "f 4755 /app/bin/tool",
        "f 644 /app/a b.txt",
        "d 777 /etc",  # a file name with a newline in it can print this
        "f 644 /app/../etc/passwd",
        "f 644 /application/x",
        "f 9 /app/x",
        "garbage",
    ]
    assert verify.parse_mode_record("/app", lines) == {
        "/app": ("d", 0o755),
        "/app/bin/tool": ("f", 0o755),
        "/app/a b.txt": ("f", 0o644),
    }


class _DockerCp(_Restoring):
    """``docker cp`` keeps hard links: two names of one file stay one inode on the host."""

    async def download_dir(self, source_dir, target_dir):
        source = self.root / source_dir.lstrip("/")
        first: dict[int, Path] = {}
        for found in sorted(source.rglob("*")):
            dest = Path(target_dir) / found.relative_to(source)
            if found.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            inode = found.stat().st_ino
            if inode in first:
                os.link(first[inode], dest)
            else:
                shutil.copy2(found, dest)
                first[inode] = dest


async def test_run_sees_both_names_of_a_hard_linked_file(tmp_path, monkeypatch) -> None:
    """Cargo hard-links ``target/debug/<bin>`` to ``deps/<bin>-<hash>``, and ``docker cp``
    keeps the link in the judges' copy, so ``read`` shows both names. tarfile writes the
    second name it meets as a link to the first, and the stash must keep it, or the
    judge's ``run`` of that name finds no file. Here ``aaa.py`` sorts first, so
    ``main.py`` is the second name. Guards J1-R1-A-2 (df1b0347 dropped every member
    that was not a regular file or a folder)."""
    root = _container(tmp_path, "oracle")
    os.link(root / "app" / "main.py", root / "app" / "aaa.py")
    box = _DockerCp(root)
    judge = _ScriptedJudge()
    result, trial, _ = await _grade(tmp_path, "oracle", judge, monkeypatch, box=box)
    run = _judgments(trial)[0]["calls"][1]["tools"][0]["result"]
    assert _data({"content": [{"content": run}]}).strip() == "42", run
    assert result.rewards["reward"] == 1.0
    with tarfile.open(fileobj=io.BytesIO(box.stash or b"")) as tar:
        main = tar.getmember("app/main.py")
    assert main.islnk() and main.linkname == "app/aaa.py"


class _DockerCpModes(_SdkDownloads, _DockerCp):
    """``docker cp`` after Docker's chown to the host user: the host copy is the host
    user's, at each mode the sandbox had (``sandbox_modes``), which the probe records.
    The container folder stays readable, so the box can copy it."""

    sandbox_modes: dict[str, int] | None = None

    async def exec(self, command, **kwargs):
        result = await super().exec(command, **kwargs)
        if "then echo link" not in command:
            return result
        lines = []
        modes = self.sandbox_modes or {}
        for line in result.stdout.splitlines():
            kind, _, rest = line.partition(" ")
            name = rest.partition(" ")[2]
            if kind in ("d", "f") and name in modes:
                line = f"{kind} {modes[name]:o} {name}"
            lines.append(line)
        return _Result(stdout="\n".join(lines) + "\n")

    async def download_dir(self, source_dir, target_dir):
        await _DockerCp.download_dir(self, source_dir, target_dir)
        # Deepest first, so a folder loses its own access after what is under it is set.
        for name, mode in sorted((self.sandbox_modes or {}).items(), reverse=True):
            os.chmod(Path(target_dir) / os.path.relpath(name, source_dir), mode)


async def test_a_kept_file_its_owner_cannot_read_is_still_judged_on_docker(
    tmp_path, monkeypatch
) -> None:
    """On Docker the judges' copy is the host user's at the modes the sandbox had, so a
    kept file at 000 or a folder at 0300 is one BenchFlow (not root on the host) cannot
    open; root in the sandbox reads both. The copy gets owner access, the stash keeps the
    sandbox's modes, and the trial is judged. Run as root this cannot fail (root reads
    any file); the suite runs as an ordinary user. Guards J1-R2-A-1 (5a80966a built the
    stash on the host, and ``tarfile.add`` raised PermissionError, so the trial errored
    unscored where integ/py-0.8's in-sandbox tar judged it)."""
    root = _container(tmp_path, "oracle")
    (root / "app" / "secret.txt").write_text("unreadable\n")
    (root / "app" / "locked").mkdir()
    (root / "app" / "locked" / "inner.txt").write_text("inside\n")
    box = _DockerCpModes(root)
    box.sandbox_modes = {"/app/secret.txt": 0o000, "/app/locked": 0o300}
    judge = _ScriptedJudge()
    result, trial, _ = await _grade(tmp_path, "oracle", judge, monkeypatch, box=box)
    run = _judgments(trial)[0]["calls"][1]["tools"][0]["result"]
    assert _data({"content": [{"content": run}]}).strip() == "42", run
    assert result.rewards["reward"] == 1.0
    with tarfile.open(fileobj=io.BytesIO(box.stash or b"")) as tar:
        members = {m.name: m for m in tar.getmembers()}
    assert members["app/secret.txt"].mode == 0o000
    assert members["app/locked"].mode == 0o300
    assert members["app/locked/inner.txt"].isfile()


async def test_a_stash_that_cannot_be_archived_stops_the_judging_by_name(
    tmp_path,
) -> None:
    """Should a file of the judges' copy still be one BenchFlow cannot open, the stash
    names it as a verifier error rather than raising a bare PermissionError. Run as root
    this cannot fail. Guards J1-R2-A-1 (5a80966a let tarfile's PermissionError out of
    ``runner.setup()``)."""
    custody = tmp_path / "custody"
    (custody / "app").mkdir(parents=True)
    locked = custody / "app" / "secret.txt"
    locked.write_text("unreadable\n")
    locked.chmod(0o000)
    runner = verify.SandboxRunner(
        sandbox=_Box(tmp_path / "box"), kept=["/app"], views={}, custody=custody
    )
    try:
        if os.access(locked, os.R_OK):
            await runner._stash()
        else:
            with pytest.raises(
                verify.TaskMdVerifierError, match=r"could not archive.*secret\.txt"
            ):
                await runner._stash()
    finally:
        locked.chmod(0o600)


class _GoArchiveCp(_DockerCpModes):
    """Docker as BenchFlow's own (non-root) host user sees it. ``docker compose cp``
    extracts a folder the way go-archive does: it makes each folder and sets its mode at
    once, then writes what is in it, so a folder at 0555 or 0000 that holds a file fails
    the copy. Root's tar in the sandbox (``tar -C / -cf - ./app``) has each mode the
    sandbox has (``sandbox_modes``)."""

    copied: bytes | None = None

    async def exec(self, command, **kwargs):
        made = re.search(r"tar -C / -cf - '?\./([^' ]+)'? ", command)
        if made is None:
            return await super().exec(command, **kwargs)
        self.commands.append(command)
        modes = self.sandbox_modes or {}

        def as_in_sandbox(info: tarfile.TarInfo) -> tarfile.TarInfo:
            mode = modes.get("/" + info.name.removeprefix("./"))
            if mode is not None:
                info.mode = mode
            return info

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as tar:
            tar.add(
                self.root / made.group(1),
                arcname="./" + made.group(1),
                filter=as_in_sandbox,
            )
        self.copied = buffer.getvalue()
        return _Result(stdout=f"BF copy {len(self.copied)} 0\n")

    async def download_file(self, source, target):
        if source == verify.COPY_TAR:
            Path(target).write_bytes(self.copied or b"")
            return
        await super().download_file(source, target)

    async def download_dir(self, source_dir, target_dir):
        source = self.root / source_dir.lstrip("/")
        modes = self.sandbox_modes or {}
        for found in sorted(source.rglob("*")):
            parent = "/" + found.parent.relative_to(self.root).as_posix()
            if modes.get(parent, 0o700) & 0o300 != 0o300:
                raise RuntimeError(
                    f"docker compose cp failed: open {found.relative_to(self.root)}: permission denied"
                )
        await super().download_dir(source_dir, target_dir)


async def test_a_kept_folder_its_owner_cannot_write_or_enter_is_still_judged_on_docker(
    tmp_path, monkeypatch
) -> None:
    """A read-only folder (0555: ``chmod -R a-w``, a Go module cache) or one at 0000 that
    holds a file is one ``docker compose cp`` cannot write as BenchFlow's own user: it sets
    the folder's mode before it writes the file. Root in the sandbox reads both, so the
    copy is root's tar from the sandbox, unpacked on the host, and the trial is judged; the
    stash keeps the sandbox's modes. Guards J1-R3-A-1 (fa161e96 and integ/py-0.8 copied
    with ``docker compose cp``, which failed, so the trial errored unscored)."""
    root = _container(tmp_path, "oracle")
    for name in ("ro", "locked"):
        (root / "app" / name).mkdir()
        (root / "app" / name / "inner.txt").write_text(f"{name}\n")
    (root / "app" / "secret.txt").write_text("unreadable\n")
    box = _GoArchiveCp(root)
    box.sandbox_modes = {
        "/app/ro": 0o555,
        "/app/locked": 0o000,
        "/app/secret.txt": 0o000,
    }
    judge = _ScriptedJudge()
    result, trial, _ = await _grade(tmp_path, "oracle", judge, monkeypatch, box=box)
    run = _judgments(trial)[0]["calls"][1]["tools"][0]["result"]
    assert _data({"content": [{"content": run}]}).strip() == "42", run
    assert result.rewards["reward"] == 1.0
    with tarfile.open(fileobj=io.BytesIO(box.stash or b"")) as tar:
        members = {m.name: m for m in tar.getmembers()}
    assert members["app/ro"].mode == 0o555
    assert members["app/locked"].mode == 0o000
    assert members["app/secret.txt"].mode == 0o000
    assert members["app/locked/inner.txt"].isfile()
    fs = trial / "verifier" / "taskmd-judge" / "fs" / "app"
    assert (fs / "locked" / "inner.txt").read_text() == "locked\n"
    assert any(c.strip().endswith(f"rm -rf {verify.COPY_DIR}") for c in box.commands)


def test_the_sandboxs_tar_of_a_kept_folder_is_read_as_untrusted(tmp_path) -> None:
    """The sandbox's ``tar`` is the image's, which a root solver can replace, so the host
    unpacks only regular files, folders and hard links to a file kept before them, inside
    the kept path. A link or special file is refused by name; a name outside the kept
    path stops the judging before anything is written."""

    def entry(name, kind=tarfile.REGTYPE, data=b"", mode=0o644, linkname=""):
        info = tarfile.TarInfo(name)
        info.type, info.mode, info.linkname = kind, mode, linkname
        if kind == tarfile.REGTYPE:
            info.size = len(data)
            return info, io.BytesIO(data)
        return info, None

    def archive(name, *members):
        path = tmp_path / f"{name}.tar"
        with tarfile.open(path, "w", format=tarfile.GNU_FORMAT) as tar:
            for info, data in members:
                tar.addfile(info, data)
        return path

    good = archive(
        "good",
        entry("./app", tarfile.DIRTYPE, mode=0o000),
        entry("./app/main.py", data=b"print(42)\n", mode=0o000),
        entry("./app/also.py", tarfile.LNKTYPE, linkname="./app/main.py"),
        entry("./app/passwd", tarfile.SYMTYPE, linkname="/etc/passwd"),
        entry("./app/pipe", tarfile.FIFOTYPE),
        entry("./app/via", tarfile.LNKTYPE, linkname="./app/passwd"),
    )
    fs = tmp_path / "fs"
    fs.mkdir()
    refused: dict[str, str] = {}
    assert verify.unpack_kept(good, "/app", fs, refused) is None
    assert (fs / "app" / "main.py").read_bytes() == b"print(42)\n"
    assert (fs / "app" / "also.py").samefile(fs / "app" / "main.py")
    assert sorted(p.name for p in (fs / "app").iterdir()) == ["also.py", "main.py"]
    assert (fs / "app").stat().st_mode & 0o777 == 0o700
    assert sorted(refused) == ["/app/passwd", "/app/pipe", "/app/via"]
    for i, name in enumerate(("./etc/passwd", "/app/x", "./app/../etc/x", "../x")):
        bad = archive(f"bad{i}", entry("./app", tarfile.DIRTYPE), entry(name))
        with pytest.raises(verify.TaskMdVerifierError, match="outside it"):
            verify.unpack_kept(bad, "/app", tmp_path / "fs2", {})
    assert not (tmp_path / "fs2").exists()


def _entry(name, kind=tarfile.REGTYPE, data=b"", mode=0o644, linkname=""):
    info = tarfile.TarInfo(name)
    info.type, info.mode, info.linkname = kind, mode, linkname
    if kind == tarfile.REGTYPE:
        info.size = len(data)
        return info, data
    return info, None


def _tar_bytes(*members, mode="w", format=tarfile.GNU_FORMAT) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode, format=format) as tar:
        for info, data in members:
            tar.addfile(info, None if data is None else io.BytesIO(data))
    return buffer.getvalue()


# What an earlier kept path, /result, holds before /app is copied.
_RESULT_TAR = (
    _entry("./result", tarfile.DIRTYPE, mode=0o755),
    _entry("./result/value.txt", data=b"earlier\n"),
)


@dataclass
class _HandsOverTar(_Box):
    """A sandbox whose root ``tar`` of each kept folder is the archive the test made
    (``tars``, by the folder's path without its leading slash)."""

    tars: dict[str, bytes] = field(default_factory=dict)
    handed: bytes = b""

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        if "then echo link" in command:
            return _Result(stdout="dir\n")
        made = re.search(r"tar -C / -cf - '?\./([^' ]+)'? ", command)
        if made is not None:
            self.handed = self.tars[made.group(1)]
            return _Result(stdout=f"BF copy {len(self.handed)} 0\n")
        return _Result()

    async def download_file(self, source, target):
        assert source == verify.COPY_TAR, source
        Path(target).write_bytes(self.handed)


@pytest.mark.parametrize(
    "members",
    [
        pytest.param(
            (_entry("./app/a", data=b"first\n"), _entry("./app/a", data=b"second\n")),
            id="a-file-twice",
        ),
        pytest.param(
            (
                _entry("./app/a", data=b"first\n"),
                _entry("./app/a", tarfile.SYMTYPE, linkname="../result"),
            ),
            id="a-file-then-a-symlink",
        ),
        pytest.param(
            (_entry("./app/a", tarfile.DIRTYPE), _entry("./app/a", data=b"second\n")),
            id="a-folder-then-a-file",
        ),
        pytest.param(
            (_entry("./app/a", data=b"first\n"), _entry("app//a", data=b"second\n")),
            id="one-name-spelled-two-ways",
        ),
        pytest.param(
            (
                _entry("./app/a", data=b"first\n"),
                _entry("./app/b", data=b"second\n"),
                _entry("./app/a", tarfile.LNKTYPE, linkname="./app/b"),
            ),
            id="a-file-then-a-hard-link",
        ),
    ],
)
async def test_a_name_twice_in_the_sandboxs_tar_stops_the_judging_before_anything_is_written(
    tmp_path, members
) -> None:
    """The sandbox's ``tar`` can be the submission's, and a tar may name one path twice;
    tarfile writes the second over the first, and its hard-link fallback looks names up
    in the whole tar, refused members included. Any name that comes twice (after
    normalising) stops the judging before anything of the folder is written, and an
    earlier kept path's copy is left as it was. Guards J1-codex-r4-1 (fbc260a5 unpacked
    the duplicate)."""
    box = _HandsOverTar(tmp_path / "unused")
    box.tars = {
        "result": _tar_bytes(*_RESULT_TAR),
        "app": _tar_bytes(_entry("./app", tarfile.DIRTYPE, mode=0o755), *members),
    }
    fs = tmp_path / "fs"
    with pytest.raises(verify.TaskMdVerifierError, match="more than once"):
        await verify.copy_kept(box, ["/result", "/app"], fs)
    assert (fs / "result" / "value.txt").read_bytes() == b"earlier\n"
    assert sorted(p.name for p in fs.iterdir()) == ["result"]


@pytest.mark.parametrize(
    ("mode", "opener", "kind"),
    [
        ("w:gz", "gzopen", "gzip"),
        ("w:bz2", "bz2open", "bzip2"),
        ("w:xz", "xzopen", "xz"),
    ],
)
async def test_a_compressed_tar_from_the_sandbox_is_refused_before_it_is_decompressed(
    tmp_path, monkeypatch, mode, opener, kind
) -> None:
    """Root's ``tar -cf`` writes an uncompressed tar, and its size is what BenchFlow
    caps; a compressed one could hold far more. It is refused by its first bytes, and
    tarfile never decompresses it. Guards J1-codex-r4-2 (fbc260a5 opened it with
    ``tarfile.open``, which decompresses whatever it is given)."""

    def decompresses(*args, **kwargs):
        raise AssertionError(f"tarfile's {opener} ran")

    box = _HandsOverTar(tmp_path / "unused")
    box.tars = {
        "app": _tar_bytes(
            _entry("./app", tarfile.DIRTYPE, mode=0o755),
            _entry("./app/a", data=b"x" * 4096),
            mode=mode,
        )
    }
    monkeypatch.setattr(tarfile.TarFile, opener, decompresses)
    fs = tmp_path / "fs"
    with pytest.raises(verify.TaskMdVerifierError, match=rf"compressed \({kind}\)"):
        await verify.copy_kept(box, ["/app"], fs)
    assert not (fs / "app").exists()


def _declared(kind: bytes, size: int, data: bytes) -> bytes:
    """An extended header (``kind``) that declares ``size`` bytes, followed by only ``data``."""
    info = tarfile.TarInfo("././@LongLink")
    info.type, info.size = kind, size
    return info.tobuf(format=tarfile.GNU_FORMAT) + data.ljust(512, b"\0")


def _file_evidence_package(tmp_path: Path) -> Path:
    """One positive criterion whose missing main.py would otherwise score zero."""
    package = tmp_path / "file-evidence"
    shutil.copytree(FIXTURE, package)
    path = package / "verifier" / "rubric.json"
    rubric = json.loads(path.read_text())
    rubric["criteria"][0]["evidence"] = ["main.py"]
    path.write_text(json.dumps(rubric))
    return package


async def test_a_global_pax_header_stops_the_judging_before_anything_is_written(
    tmp_path, monkeypatch
) -> None:
    """Global PAX keys accumulate and are copied into every later member, so encoded
    metadata caps alone do not bound memory. Refuse even one small global header
    before tarfile parses it. Guards J1-codex-r5-1 at 3ee8aee9."""
    box = _HandsOverTar(tmp_path / "unused")
    box.tars = {
        "app": _declared(tarfile.XGLTYPE, 9, b"9 BF.k=x\n")
        + _tar_bytes(
            _entry("./app", tarfile.DIRTYPE),
            _entry("./app/a", data=b"small\n"),
            _entry("./app/extra", data=b"small\n"),
        )
    }
    parsed = []
    proc_pax = tarfile.TarInfo._proc_pax

    def reads(self, tar):
        parsed.append(self.type)
        return proc_pax(self, tar)

    monkeypatch.setattr(tarfile.TarInfo, "_proc_pax", reads)
    judge = _ScriptedJudge()
    with pytest.raises(verify.TaskMdVerifierError, match="global PAX header"):
        await _grade(
            tmp_path,
            "oracle",
            judge,
            monkeypatch,
            box=box,
            package=_file_evidence_package(tmp_path),
        )
    assert parsed == []
    assert judge.bodies == []
    verifier = tmp_path / "trial-oracle" / "verifier"
    assert list((verifier / "taskmd-judge" / "fs").iterdir()) == []
    assert not (verifier / "reward.txt").exists()
    assert not (verifier / "reward.json").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "pax",
        "mode",
        "uid",
        "gid",
        "mtime",
        "devmajor",
        "devminor",
        "partial",
        "missing",
    ],
)
async def test_a_malformed_later_header_stops_the_judging_before_anything_is_written(
    tmp_path, monkeypatch, damage
) -> None:
    """tarfile silently stops at malformed headers after a valid member. That must
    be a verifier error, not missing evidence scored as a failed criterion; no member
    may be extracted and no reward or judge call made. Small checksum-correct PAX
    and numeric headers, and a cut-short header, guard J1-codex-r5-2 at 3ee8aee9."""
    prefix = _tar_bytes(
        _entry("./app", tarfile.DIRTYPE),
        _entry("./app/early.txt", data=b"must not be written\n"),
    )[:1536]
    suffix = _tar_bytes(_entry("./app/main.py", data=b"print(42)\n"))
    if damage == "pax":
        broken = _declared(tarfile.XHDTYPE, 8, b"not-pax\n") + suffix
    elif damage in ("partial", "missing"):
        broken = suffix[:200] if damage == "partial" else b""
    else:
        start, end = {
            "mode": (100, 108),
            "uid": (108, 116),
            "gid": (116, 124),
            "mtime": (136, 148),
            "devmajor": (329, 337),
            "devminor": (337, 345),
        }[damage]
        header = bytearray(suffix[:512])
        header[start:end] = b"invalid".ljust(end - start, b"\0")
        checksum = tarfile.calc_chksums(header)[0]
        header[148:156] = f"{checksum:06o}\0 ".encode("ascii")
        broken = bytes(header) + suffix[512:]
    box = _HandsOverTar(tmp_path / "unused", tars={"app": prefix + broken})
    judge = _ScriptedJudge()
    with pytest.raises(verify.TaskMdVerifierError, match=r"archive.*saved output /app"):
        await _grade(
            tmp_path,
            "oracle",
            judge,
            monkeypatch,
            box=box,
            package=_file_evidence_package(tmp_path),
        )
    assert judge.bodies == []
    verifier = tmp_path / "trial-oracle" / "verifier"
    assert list((verifier / "taskmd-judge" / "fs").iterdir()) == []
    assert not (verifier / "reward.txt").exists()
    assert not (verifier / "reward.json").exists()


def test_an_extended_header_over_the_caps_is_refused_before_tarfile_reads_it(
    tmp_path, monkeypatch
) -> None:
    """tarfile reads a GNU long name or a PAX header whole, before it returns the member
    or any name is checked. BenchFlow reads each header first, and one that declares
    more than the cap (here a gigabyte, with half a kilobyte behind it), extended headers
    over the total, or more headers than the cap, stop the judging before tarfile reads
    them. Long names and PAX headers under the caps unpack as before. Guards
    J1-codex-r4-2 (fbc260a5 let tarfile read each extended header first)."""

    def reads(*args, **kwargs):
        raise AssertionError("tarfile read an extended header's payload")

    folder = _tar_bytes(_entry("./app", tarfile.DIRTYPE))[:512]
    for kind in (tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK, tarfile.XHDTYPE):
        archive = tmp_path / "declared.tar"
        archive.write_bytes(
            folder + _declared(kind, 1 << 30, b"./app/x") + b"\0" * 1024
        )
        with monkeypatch.context() as patch:
            patch.setattr(tarfile.TarInfo, "_proc_gnulong", reads)
            patch.setattr(tarfile.TarInfo, "_proc_pax", reads)
            with pytest.raises(
                verify.TaskMdVerifierError,
                match=f"extended header of {1 << 30} bytes, over the cap of 65536",
            ):
                verify.unpack_kept(archive, "/app", tmp_path / "fs", {})
    assert not (tmp_path / "fs").exists()

    long_name = "./app/" + "n" * 200
    many = _tar_bytes(
        _entry("./app", tarfile.DIRTYPE),
        _entry(long_name, data=b"long\n"),
        _entry("./app/" + "m" * 200, data=b"long\n"),
    )
    pax = _tar_bytes(
        _entry("./app", tarfile.DIRTYPE),
        _entry("./app/café.txt", data=b"pax\n"),
        format=tarfile.PAX_FORMAT,
    )
    for name, data in (("many", many), ("pax", pax)):
        archive = tmp_path / f"{name}.tar"
        archive.write_bytes(data)
        assert verify.unpack_kept(archive, "/app", tmp_path / name, {}) is None
    assert (tmp_path / "many" / long_name).read_bytes() == b"long\n"
    assert (tmp_path / "pax" / "app" / "café.txt").read_bytes() == b"pax\n"

    # Headers whose data tarfile finds elsewhere than a plain walk of the tar does.
    sized, sized_data = _entry("./app/a", data=b"abc\n")
    sized.pax_headers = {"size": "4"}
    for why, data in (
        ("old-style folder", _tar_bytes(_entry("./app/", tarfile.AREGTYPE))),
        (
            "PAX size",
            _tar_bytes(
                _entry("./app", tarfile.DIRTYPE),
                (sized, sized_data),
                format=tarfile.PAX_FORMAT,
            ),
        ),
    ):
        archive = tmp_path / "odd.tar"
        archive.write_bytes(data)
        with pytest.raises(verify.TaskMdVerifierError, match=why):
            verify.unpack_kept(archive, "/app", tmp_path / "fs", {})

    archive = tmp_path / "many.tar"
    monkeypatch.setattr(verify, "TAR_EXTENDED_TOTAL", 300)
    with pytest.raises(verify.TaskMdVerifierError, match="more than 300 bytes"):
        verify.unpack_kept(archive, "/app", tmp_path / "fs", {})
    monkeypatch.setattr(verify, "TAR_EXTENDED_TOTAL", 1 << 20)
    monkeypatch.setattr(verify, "TAR_HEADERS", 4)
    with pytest.raises(verify.TaskMdVerifierError, match="more than 4 headers"):
        verify.unpack_kept(archive, "/app", tmp_path / "fs", {})
    assert not (tmp_path / "fs").exists()


def test_a_hard_link_is_made_only_to_a_file_written_from_the_same_tar(
    tmp_path, monkeypatch
) -> None:
    """A hard link in the sandbox's tar is made with ``os.link`` to a regular file
    BenchFlow wrote from that tar before it. One to a file that comes later, to a file
    an earlier kept path left on the host, or to a folder is refused by name, and
    tarfile's fallback (which copies whatever member of that name it finds anywhere in
    the tar when the link cannot be made) never runs, even where a kept path inside
    this one already left a file at the link's name. Guards J1-codex-r4-1 (fbc260a5
    extracted links with tarfile, whose fallback ran there)."""

    def looks_up(*args, **kwargs):
        raise AssertionError("tarfile looked a hard link's target up in the tar")

    monkeypatch.setattr(tarfile.TarFile, "_find_link_target", looks_up)
    fs = tmp_path / "fs"
    (fs / "result").mkdir(parents=True)
    (fs / "result" / "value.txt").write_bytes(b"earlier\n")
    (fs / "app").mkdir()
    (fs / "app" / "b").write_bytes(b"a kept path inside /app\n")
    archive = tmp_path / "links.tar"
    archive.write_bytes(
        _tar_bytes(
            _entry("./app", tarfile.DIRTYPE, mode=0o755),
            _entry("./app/early", tarfile.LNKTYPE, linkname="./app/a"),
            _entry("./app/a", data=b"from the tar\n"),
            _entry("./app/b", tarfile.LNKTYPE, linkname="./app/a"),
            _entry("./app/out", tarfile.LNKTYPE, linkname="./result/value.txt"),
            _entry("./app/dir", tarfile.DIRTYPE),
            _entry("./app/to-dir", tarfile.LNKTYPE, linkname="./app/dir"),
        )
    )
    refused: dict[str, str] = {}
    assert verify.unpack_kept(archive, "/app", fs, refused) is None
    assert sorted(refused) == ["/app/early", "/app/out", "/app/to-dir"]
    assert all("hard link" in why for why in refused.values())
    assert (fs / "app" / "b").samefile(fs / "app" / "a")
    assert (fs / "app" / "b").read_bytes() == b"from the tar\n"
    assert sorted(p.name for p in (fs / "app").iterdir()) == ["a", "b", "dir"]
    assert (fs / "result" / "value.txt").read_bytes() == b"earlier\n"
    assert (fs / "result" / "value.txt").stat().st_nlink == 1


def test_no_solver_agent_is_given_the_judges_token(tmp_path, monkeypatch) -> None:
    """The judge's key for another provider's endpoint must not reach the sandbox of the
    model being judged: every solver agent inherits ANTHROPIC_AUTH_TOKEN (without the
    base URL), so the judges read their own variables. Whichever route gives the judges
    a bearer token for that endpoint, a solver's environment must not hold it.
    Guards J1-R1-B-2 (df1b0347 read the judge's token from ANTHROPIC_AUTH_TOKEN)."""
    from benchflow.agents.env import resolve_agent_env

    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "solver-oauth-placeholder-0123456789")
    token = "judge-endpoint-token-0123456789"
    endpoint = "https://judge.example/anthropic"
    routes = (
        {"ANTHROPIC_AUTH_TOKEN": token, "ANTHROPIC_BASE_URL": endpoint},
        {
            "BENCHFLOW_TASKMD_JUDGE_AUTH_TOKEN": token,
            "BENCHFLOW_TASKMD_JUDGE_BASE_URL": endpoint,
        },
    )
    served = 0
    for route in routes:
        with monkeypatch.context() as m:
            for name, value in route.items():
                m.setenv(name, value)
            credentials = judging.credentials_from_env()
            if credentials is None or credentials.token != token:
                continue
            served += 1
            env = resolve_agent_env(
                "claude-agent-acp", "claude-haiku-4-5-20251001", None
            )
            assert token not in env.values(), [k for k, v in env.items() if v == token]
    assert served == 1


async def test_an_anthropic_key_never_goes_to_another_providers_endpoint(
    monkeypatch,
) -> None:
    """With ANTHROPIC_BASE_URL naming another provider (a solver's route) or the judges'
    own endpoint set, an Anthropic API key or a Claude OAuth token in the environment
    is never sent there: the judges' endpoint gets only its own token, and an Anthropic
    credential goes to a server ANTHROPIC_BASE_URL names only when the user opts in.
    The preflight refuses the run by name. Guards J1-R1-B-3 (df1b0347 sent
    ANTHROPIC_API_KEY as x-api-key to whatever ANTHROPIC_BASE_URL named)."""
    import httpx

    from benchflow.taskmd.launch import TaskMdLaunchRefused, check_launch

    anthropic_key = "sk-ant-api03-" + "K" * 40
    oauth = "sk-ant-oat01-" + "O" * 40
    endpoint = "https://judge.example/anthropic"
    sent: list[tuple[str, dict]] = []

    async def post(self, url, *, headers=None, json=None, **kwargs):
        sent.append((str(url), dict(headers or {})))
        return httpx.Response(200, json={"content": [], "usage": {}})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    for env in (
        {
            "ANTHROPIC_API_KEY": anthropic_key,
            "CLAUDE_CODE_OAUTH_TOKEN": oauth,
            "ANTHROPIC_BASE_URL": endpoint,
            "ANTHROPIC_AUTH_TOKEN": "tp-0123456789",
        },
        {"CLAUDE_CODE_OAUTH_TOKEN": oauth, "ANTHROPIC_BASE_URL": endpoint},
        {
            "ANTHROPIC_API_KEY": anthropic_key,
            "CLAUDE_CODE_OAUTH_TOKEN": oauth,
            "ANTHROPIC_BASE_URL": endpoint,
            "BENCHFLOW_TASKMD_JUDGE_BASE_URL": endpoint,
            "BENCHFLOW_TASKMD_JUDGE_AUTH_TOKEN": "tp-0123456789",
        },
    ):
        credentials = judging.credentials_from_env(env)
        if credentials is not None:
            await judging.MessagesClient(credentials)({"model": "m"}, 30)
    for url, headers in sent:
        assert url.startswith(endpoint)
        assert not any(
            anthropic_key in value or oauth in value for value in headers.values()
        ), headers
    assert [headers.get("authorization") for _, headers in sent] == [
        "Bearer tp-0123456789"
    ]
    # Opting in sends the key to the server ANTHROPIC_BASE_URL names (a gateway of one's own).
    trusted = judging.credentials_from_env(
        {
            "ANTHROPIC_API_KEY": anthropic_key,
            "ANTHROPIC_BASE_URL": endpoint,
            "BENCHFLOW_TASKMD_JUDGE_TRUST_BASE_URL": "1",
        }
    )
    assert trusted is not None and trusted.kind == "api-key"
    # Anthropic's own API, named or not, needs no opt-in.
    for base in ("https://api.anthropic.com", "https://api.anthropic.com/"):
        named = judging.credentials_from_env(
            {"ANTHROPIC_API_KEY": anthropic_key, "ANTHROPIC_BASE_URL": base}
        )
        assert named is not None and named.kind == "api-key"
    # The preflight refuses by name before the solver starts.
    native = materialize_task_dir(FIXTURE)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", endpoint)
    with pytest.raises(
        TaskMdLaunchRefused, match="BENCHFLOW_TASKMD_JUDGE_TRUST_BASE_URL=1"
    ):
        check_launch(native, primary_agent="oracle", sandbox_user=None)
    monkeypatch.setenv("BENCHFLOW_TASKMD_JUDGE_TRUST_BASE_URL", "1")
    check_launch(native, primary_agent="oracle", sandbox_user=None)
