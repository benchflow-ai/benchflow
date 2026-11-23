"""No credential on a command line in a whole Docker launch.

One ``bench eval run`` of the deterministic task (claude-agent-acp on the
scripted fake provider) per route, with a placeholder credential: the proxy
on the host (``ANTHROPIC_API_KEY``) and Claude Code's own route
(``ANTHROPIC_AUTH_TOKEN``). A recorder loaded into the ``bench`` process
(``sitecustomize`` on its ``PYTHONPATH``) writes down every process it starts
(``subprocess.Popen``: the docker client, the host proxy), everything a
backend hands its provider (``sandbox._transport.sent``: each exec's argv) and
each file the Docker backend uploads (whether it holds the placeholder).
The placeholder appears in none of them, as is or encoded (the Docker
backend gives an exec its env on stdin; a credential file goes up as a file).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h
from tests.test_operator_credentials import leaks

pytestmark = pytest.mark.deterministic

RECORDER = r"""
import json
import os
import subprocess

_LOG = os.environ.get("BF_TEST_ARGV_LOG")


def _record(kind, what):
    if not _LOG:
        return
    if isinstance(what, bytes):
        what = what.decode("latin-1")
    elif not isinstance(what, str):
        what = [str(part) for part in what]
    with open(_LOG, "a") as out:
        out.write(json.dumps({"kind": kind, "what": what}) + "\n")


_init = subprocess.Popen.__init__


def _popen(self, args, *rest, **kwargs):
    _record("popen", args)
    return _init(self, args, *rest, **kwargs)


subprocess.Popen.__init__ = _popen
try:
    from benchflow.sandbox import _transport

    _sent = _transport.sent

    def _sent_recorded(backend, what, *rest, **kwargs):
        _record("sent:" + backend, what)
        return _sent(backend, what, *rest, **kwargs)

    _transport.sent = _sent_recorded
except Exception as exc:  # the row fails on this record
    _record("recorder-error", repr(exc))
try:
    from benchflow.sandbox import docker as _docker

    _mark = os.environ.get("BF_TEST_ARGV_MARK", "").encode()
    _upload = _docker.DockerSandbox.upload_file

    async def _upload_recorded(self, source_path, target_path, *rest, **kwargs):
        try:
            with open(source_path, "rb") as source:
                held = bool(_mark) and _mark in source.read()
        except OSError:
            held = False
        _record("upload:mark" if held else "upload", str(target_path))
        return await _upload(self, source_path, target_path, *rest, **kwargs)

    _docker.DockerSandbox.upload_file = _upload_recorded
except Exception as exc:  # the row fails on this record
    _record("recorder-error", repr(exc))
"""


@pytest.fixture(scope="module")
def recorder(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("argv-recorder")
    (folder / "sitecustomize.py").write_text(RECORDER)
    return folder


@pytest.fixture(scope="module")
def fake_url(tmp_path_factory):
    sandbox, _ = h.select_sandbox()
    if sandbox != "docker":
        pytest.skip("Docker only")
    log = tmp_path_factory.mktemp("fake") / "host-fake-llm.jsonl"
    with h.host_fake_llm(log, host=h.docker_bridge_host()) as url:
        yield url


@pytest.mark.parametrize("route", ["proxy", "native"])
def test_a_docker_launch_puts_no_credential_on_a_command_line(
    tmp_path, recorder, fake_url, route
):
    task = h.materialize_task(
        h.TaskVariant(f"argv-{route}", "hello-pass"), tmp_path / "tasks"
    )
    log = tmp_path / "argv.jsonl"
    run = h.run_bench(
        ["eval", "run"],
        tasks_dir=task,
        jobs_dir=tmp_path / "jobs",
        sandbox="docker",
        host_fake_url=fake_url,
        route=route,
        extra=["--retry-attempts", "0"],
        env_extra={"PYTHONPATH": str(recorder), "BF_TEST_ARGV_LOG": str(log)},
    )
    assert run.returncode == 0, run.output[-3000:]
    (trial,) = run.trial_dirs()
    result = json.loads((trial / "result.json").read_text())
    assert result["rewards"]["reward"] == 1.0
    # The trial's record: Claude Code's own route holds its token in the
    # sandbox; the proxy on the host holds nothing of the operator's there.
    if route == "native":
        [entry] = result["operator_credentials"]["in_sandbox"]
        assert (entry["kind"], entry["names"]) == ("api-key", ["ANTHROPIC_AUTH_TOKEN"])
        # The harness accepts the task's service beside the token; the trial
        # and its reward's provenance say so.
        [accepted] = result["operator_credentials"]["granted"]
        assert accepted["run"] == "--allow-task-processes-with-credentials"
        provenance = json.loads(
            (trial / "integrity" / "verifier-inputs.json").read_text()
        )["provenance"]
        assert provenance["operator_credentials_lifted"] == [accepted]
    else:
        assert "operator_credentials" not in result
    records = [json.loads(line) for line in log.read_text().splitlines()]
    kinds = {r["kind"] for r in records}
    assert "recorder-error" not in kinds and "sent:Docker" in kinds, kinds
    texts = [
        r["what"] if isinstance(r["what"], str) else " ".join(r["what"])
        for r in records
    ]
    assert leaks(texts, [h.DUMMY_KEY]) == []


def test_a_task_service_beside_the_agents_token_is_refused_without_the_flag(
    tmp_path, fake_url
):
    """Docker, `public`: Claude Code's own token goes into the sandbox, and the
    task's environment-plane service (the fake provider) runs in main. Without
    --allow-task-processes-with-credentials the trial is refused before the
    agent starts, naming the service, the flag and the proxy on the host."""
    import subprocess

    task = h.materialize_task(
        h.TaskVariant("quiet-native", "hello-pass"), tmp_path / "tasks"
    )
    args, env = h.bench_command(
        ["eval", "run"],
        tasks_dir=task,
        jobs_dir=tmp_path / "jobs",
        sandbox="docker",
        host_fake_url=fake_url,
        route="native",
        extra=["--retry-attempts", "0"],
    )
    args = [a for a in args if a != "--allow-task-processes-with-credentials"]
    subprocess.run(
        args, cwd=h.REPO_ROOT, env=env, capture_output=True, text=True, timeout=1500
    )
    [result_path] = list((tmp_path / "jobs").rglob("result.json"))
    result = json.loads(result_path.read_text())
    error = str(result.get("error") or "")
    assert "with an operator credential in the sandbox" in error, error[-1500:]
    assert "fake_llm" in error
    assert "--allow-task-processes-with-credentials" in error
    assert "the proxy on the host (Docker, `public`)" in error
    assert result.get("rewards") in (None, {})
    assert not list(result_path.parent.rglob("agent/launched"))


TP_FLAG = "--allow-task-processes-with-credentials"


def _run(args, env, **extra_env):
    import subprocess

    return subprocess.run(
        args,
        cwd=h.REPO_ROOT,
        env={**env, **extra_env},
        capture_output=True,
        text=True,
        timeout=1500,
    )


def _with_env(args: list[str], name: str, value: str) -> list[str]:
    """*args* with ``--agent-env NAME=...`` set to *value*."""
    out = list(args)
    for i, arg in enumerate(out):
        if arg.startswith(f"{name}="):
            out[i] = f"{name}={value}"
            return out
    raise AssertionError(f"no --agent-env {name}")


def test_main_is_listed_before_the_proxy_in_the_sandbox_gets_its_key(
    tmp_path, recorder, fake_url
):
    """Docker under an allowlist: the model proxy runs in the sandbox with
    the operator's API key. With --allow-task-processes-with-credentials only
    the policy asks for quiet main, and main is still listed before any file
    holding the key goes up; the trial records the key where the proxy holds
    it, and no command line holds it."""
    task = h.materialize_task(
        h.TaskVariant("order-allowlist", "hello-pass", allowed_hosts=("example.com",)),
        tmp_path / "tasks",
    )
    args, env = h.bench_command(
        ["eval", "run"],
        tasks_dir=task,
        jobs_dir=tmp_path / "jobs",
        sandbox="docker",
        host_fake_url=fake_url,
        route="proxy",
        extra=["--retry-attempts", "0", TP_FLAG],
    )
    args = _with_env(args, "BENCHFLOW_PROVIDER_BASE_URL", h.SIDECAR_FAKE_URL)
    log = tmp_path / "argv.jsonl"
    done = _run(
        args,
        env,
        PYTHONPATH=str(recorder),
        BF_TEST_ARGV_LOG=str(log),
        BF_TEST_ARGV_MARK=h.DUMMY_KEY,
    )
    [result_path] = list((tmp_path / "jobs").rglob("result.json"))
    result = json.loads(result_path.read_text())
    assert (result.get("rewards") or {}).get("reward") == 1.0, (
        str(result.get("error"))[-1500:],
        done.stdout[-1500:],
    )
    record = result["operator_credentials"]
    [entry] = record["in_sandbox"]
    assert entry["kind"] == "api-key" and entry["route"] == "proxy", entry
    assert "ANTHROPIC_API_KEY" in entry["names"]
    assert entry["staged"] == "the sandbox proxy's process (root)"
    assert [g["run"] for g in record["granted"]] == [TP_FLAG]
    records = [json.loads(line) for line in log.read_text().splitlines()]
    texts = [
        r["what"] if isinstance(r["what"], str) else " ".join(r["what"])
        for r in records
    ]
    listed = next(i for i, t in enumerate(texts) if "BF task-procs" in t)
    key_up = next(i for i, r in enumerate(records) if r["kind"] == "upload:mark")
    assert listed < key_up, (listed, key_up)
    assert leaks(texts, [h.DUMMY_KEY]) == []


#: As root, a process that outlives step one's test.sh.
STEP_DAEMON = (
    "setsid bash -c 'exec -a bf-root-daemon sleep 900' </dev/null >/dev/null 2>&1 &\n"
)


#: The template task as task.toml (a multi-step task runs its steps in that
#: layout), its fake in a compose service of its own.
TWO_STEPS_TOML = """version = "1.0"
multi_step_reward_strategy = "final"

[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "sanity"
tags = ["deterministic-integration"]

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
allow_internet = true

[[steps]]
name = "one"
min_reward = 1.0

[[steps]]
name = "two"
"""


def _two_steps(root: Path, name: str, *, daemon: bool) -> Path:
    """The template task in two steps, its fake in its own service so main
    starts quiet; step one's test.sh may leave a process as root."""
    task = h.materialize_task(
        h.TaskVariant(name, "hello-pass", fake_in_sidecar=True), root
    )
    (task / "task.md").unlink()
    (task / "task.toml").write_text(TWO_STEPS_TOML)
    test_sh = (task / "tests" / "test.sh").read_text()
    for step, extra in (("one", STEP_DAEMON if daemon else ""), ("two", "")):
        folder = task / "steps" / step
        (folder / "tests").mkdir(parents=True)
        (folder / "instruction.md").write_text(
            "Create a file called `hello.txt` in `/app` containing exactly "
            "`Hello, world!`.\n\n[[fake-llm:hello-pass]]\n"
        )
        (folder / "tests" / "test.sh").write_text(test_sh + extra)
        (folder / "tests" / "test.sh").chmod(0o755)
    return task


@pytest.mark.parametrize("daemon", [True, False], ids=["test-sh-daemon", "quiet"])
def test_a_steps_test_sh_process_is_refused_beside_the_next_steps_token(
    tmp_path, fake_url, daemon
):
    """Docker, `public`, Claude Code's own token in the sandbox, two steps:
    step one's test.sh, run as root between the steps, leaves a process.
    Step two's connection lists main again and is refused, naming it; with
    nothing left, both steps run."""
    task = _two_steps(tmp_path / "tasks", f"steps-{daemon}", daemon=daemon)
    args, env = h.bench_command(
        ["eval", "run"],
        tasks_dir=task,
        jobs_dir=tmp_path / "jobs",
        sandbox="docker",
        host_fake_url=fake_url,
        route="native",
        extra=["--retry-attempts", "0"],
    )
    args = [a for a in args if a != TP_FLAG]
    args = _with_env(args, "ANTHROPIC_BASE_URL", h.SIDECAR_FAKE_URL)
    done = _run(args, env)
    [result_path] = list((tmp_path / "jobs").rglob("result.json"))
    result = json.loads(result_path.read_text())
    error = str(result.get("error") or "")
    if not daemon:
        assert (result.get("rewards") or {}).get("reward") == 1.0, (
            error[-1500:],
            done.stdout[-1500:],
        )
        return
    assert "with an operator credential in the sandbox" in error, (
        error[-1500:],
        done.stdout[-2000:],
    )
    assert "(uid 0): bf-root-daemon" in error and TP_FLAG in error, error[-1500:]
    # Step one ran and was graded; step two's agent never started.
    assert "Step 1/2 one: reward 1.00" in done.stdout + done.stderr


def test_a_scoped_token_beside_a_task_process_runs_and_is_recorded(tmp_path, fake_url):
    """Hosted's shape on Docker, `public`: Claude Code's own token
    (ANTHROPIC_AUTH_TOKEN with ANTHROPIC_BASE_URL at a key proxy, here the
    fake) beside a task process in main (the fake provider, the task's
    environment-plane service). Without a flag it is refused (the row
    above); named with --scoped-credential it runs, its record names the
    token as scoped, and its value is in no run card, result.json or log."""
    task = h.materialize_task(
        h.TaskVariant("scoped-native", "hello-pass"), tmp_path / "tasks"
    )
    args, env = h.bench_command(
        ["eval", "run"],
        tasks_dir=task,
        jobs_dir=tmp_path / "jobs",
        sandbox="docker",
        host_fake_url=fake_url,
        route="native",
        extra=["--retry-attempts", "0", "--scoped-credential", "ANTHROPIC_AUTH_TOKEN"],
    )
    args = [a for a in args if a != TP_FLAG]
    done = _run(args, env)
    [result_path] = list((tmp_path / "jobs").rglob("result.json"))
    result = json.loads(result_path.read_text())
    assert (result.get("rewards") or {}).get("reward") == 1.0, (
        str(result.get("error"))[-1500:],
        done.stdout[-1500:],
    )
    record = result["operator_credentials"]
    assert record["scoped"] == ["ANTHROPIC_AUTH_TOKEN"]
    [entry] = record["in_sandbox"]
    assert entry["names"] == ["ANTHROPIC_AUTH_TOKEN"] and entry["scoped"] is True
    assert record["granted"] == []
    texts = [result_path.read_text(), done.stdout, done.stderr]
    texts += [
        p.read_text(errors="replace")
        for p in (tmp_path / "jobs").rglob("run-card.json")
    ]
    texts += [p.read_text(errors="replace") for p in (tmp_path / "jobs").rglob("*.log")]
    assert len(texts) > 3
    assert leaks(texts, [h.DUMMY_KEY]) == []
