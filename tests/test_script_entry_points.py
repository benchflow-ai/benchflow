"""EN-632: a ``script:`` agent on a backend without its runner is refused before
any provider call, the same way on every entry point.

RCI (#266 9953b1634): the plan refusal covered ``bench run``'s flags only. A
``--config`` file (whose dry run exited 0), ``bench branch`` and the SDK's
``Rollout.create`` -> ``setup`` -> ``start`` still created a Daytona sandbox, or
connected to Modal, before the runner refused the script. Then (RCI, round 2)
``--matrix``, the SDK's ``Evaluation.run``, ``bf.branch``, ``bench eval resume``
and ``bench skills eval`` still listed Daytona sandboxes first: the key check
(``eval_preflight``, ``check_host``) and the job-start reaper, which lists the
owner's sandboxes and can delete stale ones.

Here a fake Daytona API and a fake Modal endpoint count every request (any
method) and connection, with the key check on, and every test asserts none,
and that the reaper was never reached.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from benchflow.cli.main import app

NO_RUNNER = ("daytona", "modal")
HINT = (
    "use --sandbox docker, or run the logic as an ACP agent "
    "(docs/agents/custom-agent.md)"
)


def refusal(sandbox: str) -> str:
    return (
        "script: runs only on docker, remote-docker: its runner needs a trusted "
        f"root transport the {sandbox} backend does not have"
    )


class FakeProviders:
    """A fake Daytona API (HTTP) and a fake Modal endpoint (TCP) on free local
    ports, counting every request and connection. A Daytona GET answers an
    empty page and anything else 503; a Modal connection is closed at once."""

    def __init__(self) -> None:
        self.daytona: list[str] = []
        self.modal = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _any(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                if length:
                    self.rfile.read(length)
                with fake._lock:
                    fake.daytona.append(f"{self.command} {self.path}")
                if self.command == "GET":
                    body = json.dumps(
                        {"items": [], "total": 0, "page": 1, "totalPages": 0}
                    ).encode()
                    self.send_response(200)
                else:
                    body = b'{"message":"fake provider: refused"}'
                    self.send_response(503)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _any

            def log_message(self, *args: object) -> None:
                pass

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._tcp = socket.socket()
        self._tcp.bind(("127.0.0.1", 0))
        self._tcp.listen(16)
        self._tcp.settimeout(0.2)
        self.daytona_url = f"http://127.0.0.1:{self._http.server_address[1]}/api"
        self.modal_url = f"http://127.0.0.1:{self._tcp.getsockname()[1]}"
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._tcp.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with self._lock:
                self.modal += 1
            conn.close()

    def requests(self) -> dict[str, object]:
        with self._lock:
            return {"daytona": list(self.daytona), "modal": self.modal}

    def close(self) -> None:
        self._stop.set()
        self._http.shutdown()
        self._http.server_close()
        self._tcp.close()


NONE = {"daytona": [], "modal": 0}
REAPER = "daytona-auto-reap"


def _join_reapers(timeout: float = 60.0) -> None:
    """Wait out a Daytona reaper an earlier test in this process started: a
    daemon thread that reads DAYTONA_API_URL when it runs, so it would count
    here (RCI: the flags dry run's count flaked under -n)."""
    deadline = time.monotonic() + timeout
    for thread in threading.enumerate():
        if thread.name == REAPER:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                raise RuntimeError(f"a {REAPER} thread is still running")


@pytest.fixture
def providers(monkeypatch):
    from benchflow.evaluation import Evaluation

    _join_reapers()
    fake = FakeProviders()
    # Every reach of the job-start reaper, whatever it then does.
    fake.reaper_reached = []
    reap = Evaluation._maybe_start_daytona_reap

    def counted(self) -> None:
        fake.reaper_reached.append(self._config.environment)
        reap(self)

    monkeypatch.setattr(Evaluation, "_maybe_start_daytona_reap", counted)
    monkeypatch.setenv("DAYTONA_API_URL", fake.daytona_url)
    monkeypatch.setenv("DAYTONA_API_KEY", "fake-not-a-key")
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    monkeypatch.setenv("MODAL_SERVER_URL", fake.modal_url)
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-fake-not-a-token")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-fake-not-a-secret")
    # The run's Daytona key check stays on: a request it makes counts too.
    monkeypatch.delenv("BENCHFLOW_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setenv("COLUMNS", "400")
    yield fake
    _join_reapers()
    fake.close()


def _nothing_reached(providers: FakeProviders) -> None:
    """No request of any method to Daytona, no Modal connection, and the
    job-start reaper never reached."""
    assert providers.requests() == NONE
    assert providers.reaper_reached == []


@pytest.fixture
def files(tmp_path, monkeypatch):
    """A plain task and an operator script."""
    monkeypatch.chdir(tmp_path)
    task = tmp_path / "tasks" / "s-open"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /app\n"
    )
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/sh\ntrue\n")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[agent]\ntimeout_sec = 300\n\n'
        "[verifier]\ntimeout_sec = 120\n\n[environment]\n"
    )
    (task / "instruction.md").write_text("Do nothing.\n")
    script = tmp_path / "op.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    return task.parent, script


def _bench(*args: object):
    return CliRunner().invoke(app, [str(a) for a in args], prog_name="bench")


def _refused_alike(result, sandbox: str) -> None:
    """bench run's flags' refusal: its text, its next step, exit 2."""
    text = " ".join(result.output.split())
    assert result.exit_code == 2, result.output
    assert refusal(sandbox) in text, result.output
    assert HINT in text, result.output


def _run_config(tmp_path: Path, tasks: Path, script: Path, sandbox: str) -> Path:
    path = tmp_path / f"run-{sandbox}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tasks_dir": str(tasks),
                "agent": f"script:{script}",
                "environment": sandbox,
                "sandbox_user": "agent",
                "jobs_dir": str(tmp_path / "jobs"),
            }
        )
    )
    return path


RUN_MODES = pytest.mark.parametrize("dry", [True, False], ids=["dry-run", "run"])


@RUN_MODES
@pytest.mark.parametrize("sandbox", NO_RUNNER)
def test_bench_run_flags_refuse_before_any_provider_call(
    tmp_path, providers, files, sandbox, dry
):
    """The path already refused (build_eval_plan), kept: the text the other
    entry points must give."""
    tasks, script = files
    flags = ["--dry-run"] if dry else []
    result = _bench(
        "run",
        "--tasks-dir",
        tasks,
        "--agent",
        f"script:{script}",
        "--sandbox",
        sandbox,
        "--sandbox-user",
        "agent",
        "--jobs-dir",
        tmp_path / "jobs",
        *flags,
    )
    _refused_alike(result, sandbox)
    _nothing_reached(providers)


@RUN_MODES
@pytest.mark.parametrize("sandbox", NO_RUNNER)
def test_a_config_file_refuses_before_any_provider_call(
    tmp_path, providers, files, sandbox, dry
):
    """``bench run --config`` with ``agent: script:...``: its dry run exited
    0, and its run reached the provider."""
    tasks, script = files
    flags = ["--dry-run"] if dry else []
    result = _bench(
        "run", "--config", _run_config(tmp_path, tasks, script, sandbox), *flags
    )
    _refused_alike(result, sandbox)
    _nothing_reached(providers)


@pytest.mark.parametrize("sandbox", NO_RUNNER)
def test_bench_branch_refuses_before_any_provider_call(
    tmp_path, providers, files, sandbox
):
    tasks, script = files
    result = _bench(
        "branch",
        "--tasks-dir",
        tasks,
        "--agent",
        f"script:{script}",
        "--sandbox",
        sandbox,
        "--child",
        "label=a",
        "--child",
        "label=b",
        "--jobs-dir",
        tmp_path / "jobs",
    )
    _refused_alike(result, sandbox)
    _nothing_reached(providers)


def _sdk_config(tasks: Path, script: Path, sandbox: str, tmp_path: Path):
    from benchflow.rollout import RolloutConfig

    return RolloutConfig(
        task_path=tasks / "s-open",
        environment=sandbox,
        agent=f"script:{script}",
        sandbox_user="agent",
        jobs_dir=tmp_path / "jobs",
    )


def _create_setup_start(cfg) -> BaseException | None:
    """The SDK's lifecycle: ``Rollout.create``, ``setup()``, ``start()``; what
    it raised, or None."""
    from benchflow.rollout import Rollout

    async def drive() -> BaseException | None:
        rollout = await Rollout.create(cfg)
        try:
            await rollout.setup()
            await rollout.start()
        except Exception as exc:
            return exc
        return None

    return asyncio.run(drive())


def _refused_in_python(raised: BaseException | None, sandbox: str) -> None:
    assert isinstance(raised, ValueError), repr(raised)
    assert str(raised) == refusal(sandbox)
    assert getattr(raised, "hint", None) == HINT


@pytest.mark.parametrize("sandbox", NO_RUNNER)
def test_the_sdk_rollout_is_refused_by_setup_before_start(
    tmp_path, providers, files, sandbox
):
    tasks, script = files
    raised = _create_setup_start(_sdk_config(tasks, script, sandbox, tmp_path))
    _refused_in_python(raised, sandbox)
    _nothing_reached(providers)


@pytest.mark.parametrize("sandbox", NO_RUNNER)
def test_a_config_changed_after_it_was_built_is_refused_too(
    tmp_path, providers, files, sandbox
):
    """Why setup holds the check, not RolloutConfig's own validation: a
    config's fields can change after it is built (it is a plain dataclass),
    and setup reads them just before start creates the sandbox."""
    tasks, script = files
    cfg = _sdk_config(tasks, script, "docker", tmp_path)
    cfg.environment = sandbox
    _refused_in_python(_create_setup_start(cfg), sandbox)
    _nothing_reached(providers)


# --- Round 2: the paths that still listed Daytona sandboxes first -------------------


@RUN_MODES
@pytest.mark.parametrize("sandbox", NO_RUNNER)
def test_a_matrix_with_a_script_entry_refuses_before_any_provider_call(
    tmp_path, providers, files, sandbox, dry
):
    tasks, script = files
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump({"models": {"ops": {"agent": f"script:{script}"}}})
    )
    flags = ["--dry-run"] if dry else []
    result = _bench(
        "run",
        "--tasks-dir",
        tasks,
        "--matrix",
        matrix,
        "--sandbox",
        sandbox,
        "--sandbox-user",
        "agent",
        "--jobs-dir",
        tmp_path / "jobs",
        *flags,
    )
    _refused_alike(result, sandbox)
    _nothing_reached(providers)


@pytest.mark.parametrize("sandbox", NO_RUNNER)
def test_the_sdk_evaluation_is_refused_before_any_provider_call(
    tmp_path, providers, files, sandbox
):
    from benchflow.evaluation import Evaluation, EvaluationConfig

    tasks, script = files
    evaluation = Evaluation(
        tasks_dir=str(tasks),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(
            agent=f"script:{script}", environment=sandbox, sandbox_user="agent"
        ),
    )
    with pytest.raises(ValueError) as refused:
        asyncio.run(evaluation.run())
    _refused_in_python(refused.value, sandbox)
    _nothing_reached(providers)
    assert not (tmp_path / "jobs").exists() or not any((tmp_path / "jobs").iterdir())


@pytest.mark.parametrize("sandbox", NO_RUNNER)
def test_bf_branch_is_refused_before_any_provider_call(
    tmp_path, providers, files, sandbox
):
    import benchflow as bf

    tasks, script = files
    with pytest.raises(ValueError) as refused:
        bf.branch(
            tasks / "s-open",
            agent=f"script:{script}",
            sandbox=sandbox,
            children={"a": None, "b": None},
            jobs_dir=tmp_path / "jobs",
        )
    _refused_in_python(refused.value, sandbox)
    _nothing_reached(providers)


@pytest.mark.parametrize("sandbox", NO_RUNNER)
def test_resuming_a_script_job_is_refused_before_any_provider_call(
    tmp_path, providers, files, sandbox
):
    """A job folder a script: run on this backend left (an earlier release
    started it)."""
    from benchflow.evaluation import (
        EVALUATION_RECORD,
        EvaluationConfig,
        _config_to_record,
    )

    tasks, script = files
    job = tmp_path / "jobs" / "script-job"
    job.mkdir(parents=True)
    config = EvaluationConfig(
        agent=f"script:{script}", environment=sandbox, sandbox_user="agent"
    )
    (job / EVALUATION_RECORD).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tasks_dir": str(tasks),
                "job_name": job.name,
                "config": _config_to_record(config),
            }
        )
    )
    _refused_alike(_bench("eval", "resume", job), sandbox)
    _nothing_reached(providers)


@pytest.mark.parametrize("sandbox", NO_RUNNER)
def test_skills_eval_with_a_script_agent_is_refused_before_any_provider_call(
    tmp_path, providers, files, sandbox
):
    _tasks, script = files
    skill = tmp_path / "simple"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / "evals.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"id": "s-001", "question": "Say hello", "ground_truth": "hello"}
                ]
            }
        )
    )
    result = _bench(
        "skills",
        "eval",
        skill,
        "--agent",
        f"script:{script}",
        "--sandbox",
        sandbox,
        "--jobs-dir",
        tmp_path / "jobs",
    )
    _refused_alike(result, sandbox)
    _nothing_reached(providers)


def test_the_in_trial_refusal_is_its_own_category_never_retried():
    """Where a trial records the refusal (its setup refused it), it is
    ``script_backend``: unscored, never retried, with its own cause and
    explanation; it was "other error (infrastructure problem)", which retries."""
    from benchflow._utils import scoring
    from benchflow._utils.scoring import (
        ERROR_CATEGORIES,
        SCRIPT_BACKEND,
        classify_error,
    )
    from benchflow.cli.explain import find
    from benchflow.evaluation import RetryConfig
    from benchflow.failures import describe

    recorded = f"ScriptBackendError: {refusal('daytona')}"
    assert classify_error(recorded) == SCRIPT_BACKEND == "script_backend"
    assert SCRIPT_BACKEND in ERROR_CATEGORIES
    # unscored (the 0.10 line keeps a table of the endings that keep a reward)
    keeps = getattr(scoring, "ENDING_KEEPS_REWARD", {})
    assert keeps.get(SCRIPT_BACKEND, False) is False
    retry = RetryConfig()
    assert retry.should_retry(recorded) is False
    assert retry.reruns_on_resume(recorded) is False
    assert describe(SCRIPT_BACKEND).next_step
    assert find(SCRIPT_BACKEND).body
    # only the class that opens it: an agent's error quoting it is not it
    assert classify_error(f"ACP error -32603: {recorded}") != SCRIPT_BACKEND


@pytest.mark.parametrize("sandbox", NO_RUNNER)
def test_a_trial_its_setup_refuses_records_the_category(
    tmp_path, providers, files, sandbox
):
    """A rollout run through the SDK whose setup refuses the script records
    the refusal, which reads as its own category and is never retried, with
    no provider call."""
    from benchflow.rollout import Rollout

    tasks, script = files

    async def drive():
        rollout = await Rollout.create(_sdk_config(tasks, script, sandbox, tmp_path))
        return await rollout.run()

    from benchflow._utils.scoring import SCRIPT_BACKEND, classify_error
    from benchflow.evaluation import RetryConfig

    result = asyncio.run(drive())
    assert result.error and result.error.startswith("ScriptBackendError: "), result
    # No category is recorded for a setup refusal; every reader (the summary,
    # explain, retry, resume) classifies the error, as here.
    assert result.error_category in (None, SCRIPT_BACKEND), result.error_category
    assert classify_error(result.error) == SCRIPT_BACKEND
    assert (
        RetryConfig().should_retry(result.error, category=result.error_category)
        is False
    )
    _nothing_reached(providers)


# --- What is not refused --------------------------------------------------------


def test_a_script_still_plans_on_docker(tmp_path, files):
    tasks, script = files
    for args in (
        ["--config", _run_config(tmp_path, tasks, script, "docker")],
        [
            "--tasks-dir",
            tasks,
            "--agent",
            f"script:{script}",
            "--sandbox",
            "docker",
            "--sandbox-user",
            "agent",
        ],
    ):
        result = _bench("run", *args, "--dry-run")
        assert result.exit_code == 0, result.output
        assert "script: runs only on" not in result.output


def test_a_script_on_docker_reaches_its_sandbox(tmp_path, files, monkeypatch):
    """The SDK lifecycle on Docker passes setup and goes on to create its
    sandbox (stopped there: nothing is built)."""
    from benchflow.rollout_planes import DefaultRolloutPlanes

    def reached(self, *args, **kwargs):
        raise RuntimeError("SandboxCreationReached")

    monkeypatch.setattr(DefaultRolloutPlanes, "create_environment", reached)
    tasks, script = files
    raised = _create_setup_start(_sdk_config(tasks, script, "docker", tmp_path))
    assert isinstance(raised, RuntimeError), repr(raised)
    assert str(raised) == "SandboxCreationReached"


@pytest.mark.parametrize("sandbox", NO_RUNNER)
@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_oracle_and_nop_pass_the_transport_check(sandbox, agent):
    from benchflow.task.plan_validation import validate_agent_transport

    validate_agent_transport(sandbox=sandbox, agent=agent, harness="acp")


def test_a_backend_of_the_callers_own_planes_is_left_to_its_sandbox():
    """A name BenchFlow does not know is a backend the caller's planes build:
    whether it has a runner is its sandbox's own check (script_runner_exec)."""
    from benchflow.task.plan_validation import validate_agent_transport

    validate_agent_transport(sandbox="my-backend", agent="script:/x.sh", harness="acp")
