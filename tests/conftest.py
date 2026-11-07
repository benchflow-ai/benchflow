"""Test fixtures."""

import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Unless LITELLM_MODE is PRODUCTION, importing litellm runs load_dotenv(),
# which walks ancestor directories and injects any developer `.env` (real API
# keys included) into os.environ for the rest of the pytest process — breaking
# env-sensitive tests that run after a litellm-importing module. Same intent
# as isolate_local_dotenv below, but for litellm's import-time side effect.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

# GitHub Actions sets GITHUB_ACTIONS=true, and Typer then forces a terminal on
# every console it makes (typer.rich_utils.FORCE_TERMINAL, read once when it
# is imported), so the help and errors the tests read as plain text come with
# colour codes, in this process and in every `bench` they start. Typer's own
# switch keeps them plain, as on a developer's machine. Set before anything
# imports Typer; the patch below covers a Typer imported earlier.
os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")
if "typer.rich_utils" in sys.modules:
    sys.modules["typer.rich_utils"].FORCE_TERMINAL = False

# Vendored upstream tasks ship their own pytest files (task verifiers, run
# inside task sandboxes) — they are fixtures, not suite tests.
collect_ignore = ["fixtures/skillsbench_slice", "fixtures/taskmd"]

REF_TASKS = REPO_ROOT / ".cache" / "datasets-v2" / "benchflow" / "examples" / "tasks"


# ---------------------------------------------------------------------------
# pytest-xdist scheduling (`pytest -n auto`; pyproject's addopts set
# `--dist loadgroup`). Two things make the deterministic tier slow under
# xdist's defaults. Its session fixtures each run one real `bench` job that
# several tests read; a session fixture runs once per worker, so tests of one
# job spread over N workers ran that job up to N times. And its tests are a
# few minutes each and contiguous in collection order, so they landed on one
# or two workers at the end. In each worker (collection runs there), tests
# that share such a job get one xdist_group, and deterministic tests move to
# the front so the long ones start first, one per worker. A serial run is
# untouched.
# ---------------------------------------------------------------------------


def _module_job_fixtures(items: list[pytest.Item]) -> dict[str, set[str]]:
    """Per module: the session fixtures it defines that no fixture uses.

    Those are the jobs (``eval_job``, ``codex_acp_job``). A session fixture
    other fixtures depend on (``det_root``, ``fake_url``) is cheap shared
    set-up; grouping by it would put the whole module on one worker.
    """
    defined: dict[str, set[str]] = {}
    used: dict[str, set[str]] = {}
    for item in items:
        module = item.nodeid.split("::", 1)[0]
        info = getattr(item, "_fixtureinfo", None)
        for name, defs in (getattr(info, "name2fixturedefs", None) or {}).items():
            for fixturedef in defs:
                if fixturedef.baseid != module or fixturedef.scope != "session":
                    continue
                defined.setdefault(module, set()).add(name)
                used.setdefault(module, set()).update(fixturedef.argnames)
    return {m: names - used.get(m, set()) for m, names in defined.items()}


def _group_tests_by_session_job(items: list[pytest.Item]) -> None:
    jobs = _module_job_fixtures(items)
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        while parent.setdefault(key, key) != key:
            key = parent[key]
        return key

    item_jobs: list[tuple[pytest.Item, list[str]]] = []
    for item in items:
        if item.get_closest_marker("xdist_group") is not None:
            continue
        module = item.nodeid.split("::", 1)[0]
        # The group is part of the test id xdist reports (test@group), so
        # keep it short: module stem and fixture names.
        mine = sorted(
            f"{Path(module).stem}.{name}"
            for name in getattr(item, "fixturenames", ())
            if name in jobs.get(module, ())
        )
        if not mine:
            continue
        item_jobs.append((item, mine))
        for other in mine[1:]:
            parent[find(other)] = find(mine[0])
    members: dict[str, set[str]] = {}
    for _, mine in item_jobs:
        members.setdefault(find(mine[0]), set()).update(mine)
    for item, mine in item_jobs:
        name = "+".join(sorted(members[find(mine[0])]))
        item.add_marker(pytest.mark.xdist_group(name=name))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    # Only in an xdist worker under loadgroup (xdist reads the marks after
    # this hook, as it collects; a worker sees `dist` as "no" and the mode
    # as `loadgroup`); `-n 0` and a serial run keep the file order.
    if not hasattr(config, "workerinput") or not config.getoption("loadgroup"):
        return
    _group_tests_by_session_job(items)
    items.sort(key=lambda item: item.get_closest_marker("deterministic") is None)


def _process_umask() -> int:
    """This process's umask, read without changing it where Linux allows."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError):
        pass
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


@pytest.fixture(autouse=True)
def keep_process_umask() -> Iterator[None]:
    """Fail a test that leaves this worker's umask changed, and put it back.

    Every later test in the worker would write files with it: a disarm test
    left 077, so test_verifier_confirm7's non-root verifier could not read
    its test.sh and test_local_daemon_docker's runtime could not remove a
    sandbox-owned folder (#1220).
    """
    before = _process_umask()
    yield
    after = _process_umask()
    if after != before:
        os.umask(before)
        pytest.fail(
            f"the test left the process umask at {after:04o} (it was {before:04o})",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def isolate_local_dotenv(monkeypatch, tmp_path) -> None:
    """Keep developer-machine `.env` secrets out of unit tests."""
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / ".env"))


@pytest.fixture(autouse=True)
def isolate_failed_daytona_layers(monkeypatch) -> None:
    """Independent tests represent independent process build histories."""
    from benchflow.sandbox import daytona_layers

    monkeypatch.setattr(daytona_layers, "_FAILED", set())


@pytest.fixture(autouse=True)
def isolate_daytona_ledger(monkeypatch, tmp_path) -> Iterator[None]:
    """Keep Daytona leftovers out of the developer's state file, and give
    each test an empty in-process ledger (sandbox/daytona_ledger.py)."""
    monkeypatch.setenv(
        "BENCHFLOW_DAYTONA_LEFTOVERS", str(tmp_path / "daytona-leftovers.jsonl")
    )
    from benchflow.sandbox import daytona_ledger

    daytona_ledger.LEDGER.entries.clear()
    yield
    daytona_ledger.LEDGER.entries.clear()


@pytest.fixture(autouse=True)
def fresh_compose_check_memo() -> None:
    """Each test starts with no kept Compose checks (compose_env._CHECK_MEMO)."""
    from benchflow.task import compose_env

    compose_env._CHECK_MEMO.clear()


@pytest.fixture(autouse=True)
def isolate_pull_records(monkeypatch, tmp_path) -> None:
    """Keep the images tests "pull" out of the developer's own pull records.

    A record under ~/.local/state/benchflow would let a later `bench cache
    clean --only pulled-images --remove` act on images the suite pulled.
    """
    monkeypatch.setenv("BENCHFLOW_PULL_RECORD_DIR", str(tmp_path / "pulled-images"))
    # Likewise the metadata helper images BenchFlow built: never
    # the developer's own record, and never another test's.
    monkeypatch.setenv(
        "BENCHFLOW_METADATA_HELPER_STATE_DIR", str(tmp_path / "metadata-helper")
    )


@pytest.fixture(autouse=True)
def no_exec_channel(request, monkeypatch) -> None:
    """Keep root's execs on one ``docker exec`` each in tests.

    Most Docker tests script the docker CLI's answers call by call; the
    command channel (sandbox/_exec_channel.py) would add its own calls to
    them. tests/test_exec_channel.py turns it back on; a ``bench`` the
    deterministic tier starts uses it as any run does. A module that runs
    whole trials on real Docker in this process sets
    ``DOCKER_RUN_DEFAULTS = True`` and keeps both as a run has them, so the
    side-by-side steps are tested on real Docker too.
    """
    from benchflow.sandbox.docker import DockerSandbox

    if getattr(request.module, "DOCKER_RUN_DEFAULTS", False) is True:
        return
    monkeypatch.setattr(DockerSandbox, "exec_channel", False)
    # Their answers come in call order, so the independent steps before the
    # verifier run one after another too (sandbox/_in_order.py);
    # tests/test_preverify_order.py turns it back on.
    monkeypatch.setattr(DockerSandbox, "concurrent_steps", False)


@pytest.fixture(autouse=True)
def no_direct_docker(monkeypatch) -> None:
    """Keep Docker sandboxes on Compose in tests that script its answers.

    The direct Docker path (sandbox/_direct_docker.py) talks to the daemon's
    socket instead of running `docker compose`, which those scripts could
    not see. Its own tests turn it back on; a `bench` the deterministic tier
    starts uses it as any run does.
    """
    from benchflow.sandbox.docker import DockerSandbox

    monkeypatch.setattr(DockerSandbox, "direct_docker", False)


@pytest.fixture(autouse=True)
def exec_transport_tripwire(request) -> Iterator[None]:
    """Fail any test in which a backend handed its provider a command over
    the transport bound (benchflow.sandbox._transport), even when the code
    under test caught the error: the bound is a contract of every backend,
    never a call site's to work around. A test that sends an oversized
    command on purpose marks itself ``transport_over_bound``."""
    from benchflow.sandbox import _transport

    over: list[tuple[str, int]] = []

    def observe(backend: str, size: int) -> None:
        if size > _transport.COMMAND_BOUND:
            over.append((backend, size))

    _transport._observers.append(observe)
    try:
        yield
    finally:
        _transport._observers.remove(observe)
    if over and request.node.get_closest_marker("transport_over_bound") is None:
        pytest.fail(
            "a backend sent a command over the "
            f"{_transport.COMMAND_BOUND}-byte transport bound: {over[:5]}",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def skip_update_check(monkeypatch) -> None:
    """Keep the `bench traj` PyPI latest-version check out of unit tests.

    Tests that exercise the hint delete this variable and monkeypatch the
    fetch instead of touching the network.
    """
    monkeypatch.setenv("BENCHFLOW_SKIP_UPDATE_CHECK", "1")


@pytest.fixture(autouse=True)
def skip_eval_preflight(monkeypatch) -> None:
    """Keep `bench eval run`'s Docker preflight (a real `docker info`) out of unit tests.

    Tests that exercise the preflight delete this variable and fake
    `benchflow.doctor.check_docker` instead (tests/test_eval_preflight.py).
    """
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "verifier_runtime: run hardening's real protected-verifier-runtime check",
    )
    config.addinivalue_line(
        "markers",
        "claude_connector_policy: run Claude Code's real managed connector policy",
    )


@pytest.fixture(autouse=True)
def no_claude_connector_policy(monkeypatch, request) -> None:
    """Admit Claude Code's managed connector policy in unit tests (#79).

    The real check (agents/claude_connector_policy.py) runs one root command
    in the sandbox: BenchFlow's Node and the pinned CLI, then the root-owned
    managed settings, for the managed launcher only. A unit test's fake
    sandbox launches Claude Code by name and cannot answer it. Tests of the
    policy, and of where a connection runs it, mark themselves
    ``claude_connector_policy``; a ``bench`` the deterministic tier starts
    runs it as any run does.
    """
    if request.node.get_closest_marker("claude_connector_policy"):
        return
    from benchflow.agents import claude_connector_policy

    async def admitted(env, *, agent_env, **kwargs):
        return agent_env

    monkeypatch.setattr(
        claude_connector_policy, "enforce_claude_connector_policy", admitted
    )


@pytest.fixture(autouse=True)
def no_verifier_runtime(monkeypatch, request) -> None:
    """Hardening without BenchFlow's protected verifier runtime.

    A unit test's fake sandbox has none, and a task that infers --ctrf would
    stop hardening for want of it. Tests of the runtime mark themselves
    ``verifier_runtime``; the guard's scoring harness models its own.
    """
    if request.node.get_closest_marker("verifier_runtime"):
        return
    from benchflow.sandbox import lockdown

    async def none(*args, **kwargs):
        return None

    monkeypatch.setattr(lockdown, "_verifier_runtime_manifest", none)


_VERIFIER_RUNTIME_CACHE_FILLED = False
# The cache is under the home the suite started with, which a test may patch.
_SUITE_HOME = str(Path.home())


def _fill_verifier_runtime_cache() -> None:
    """Fetch the releases hardening's default protected runtime needs into
    this host's cache (~/.cache/benchflow/verifier-runtime), once per test
    process, with PyPI allowed.

    Tests marked ``verifier_runtime`` build that runtime offline (below), so
    they need the cache filled; only a few tests (``test_job_matrix_docker``,
    ``test_guard_evidence_container``) fetch it online, and on a fresh runner
    an offline test could run before them. The releases are BenchFlow's own
    pytest and the ctrf plugin, resolved for a Linux CPython 3.12, as those
    tests' images ask. Writes are atomic, so xdist workers may race. A failed
    fetch (no network) leaves the tests to fail with their own message.
    """
    global _VERIFIER_RUNTIME_CACHE_FILLED
    if _VERIFIER_RUNTIME_CACHE_FILLED:
        return
    _VERIFIER_RUNTIME_CACHE_FILLED = True
    import warnings

    from benchflow.sandbox import _verifier_runtime as runtime

    linux_312 = runtime.Interpreter(
        exe="/usr/local/bin/python3",
        version=(3, 12),
        machine="x86_64",
        glibc=(2, 36),
        musl=None,
        markers=runtime._marker_environment(None),
    )
    with pytest.MonkeyPatch.context() as online:
        online.delenv("BENCHFLOW_PLUGIN_RELEASES", raising=False)
        online.setenv("HOME", _SUITE_HOME)
        try:
            runtime.resolve([], ctrf=True, interpreter=linux_312)
        except Exception as exc:  # the tests report what is missing
            warnings.warn(
                f"could not fill the verifier runtime cache: {exc}", stacklevel=1
            )


@pytest.fixture(autouse=True)
def plugin_releases_offline(monkeypatch, request) -> None:
    """Keep hardening's PyPI refresh of vouched plugin releases out of unit tests.

    Hardening then uses the built-in table and the cache; tests of the refresh
    fake the fetch (tests/test_verifier_plugin_published.py). A test of the
    real protected runtime (``verifier_runtime``) gets the cache filled first,
    so it does not depend on which test ran before it.
    """
    if request.node.get_closest_marker("verifier_runtime"):
        _fill_verifier_runtime_cache()
    monkeypatch.setenv("BENCHFLOW_PLUGIN_RELEASES", "offline")


@pytest.fixture(autouse=True)
def skip_provider_key_check(monkeypatch) -> None:
    """Keep the preflight's provider key request (benchflow.key_check) out of
    unit tests, including those that turn the other preflight checks on.

    tests/test_key_check_preflight.py deletes this variable and fakes the request.
    """
    monkeypatch.setenv("BENCHFLOW_KEY_CHECK", "0")


@pytest.fixture(autouse=True)
def skip_traj_wait(monkeypatch) -> None:
    """Keep the post-upload storage-verification poll out of unit tests.

    Tests that exercise the wait loop set their own budget and monkeypatch
    the status fetch instead of touching the network.
    """
    monkeypatch.setenv("BENCHFLOW_TRAJ_WAIT_SECONDS", "0")


@pytest.fixture
def hello_world_task_dir() -> Path:
    path = REF_TASKS / "hello-world"
    if not path.exists():
        pytest.skip("Reference tasks not available")
    return path


@pytest.fixture
def build_result_json(tmp_path):
    """Factory: call SDK._build_result with 14-field defaults and return parsed result.json.

    Accepts `trajectory_source` override so callers that care about that field can
    set it; otherwise defaults to None. The `rollout_dir` used is `tmp_path / "trial"`
    (created if missing). Any override via kwargs is merged on top of the defaults.
    """

    def _build(**overrides) -> dict:
        from benchflow.sdk import SDK

        rollout_dir = tmp_path / "trial"
        rollout_dir.mkdir(exist_ok=True)
        defaults = dict(
            task_name="t1",
            rollout_name="trial-1",
            agent="test",
            agent_name="",
            model="",
            n_tool_calls=0,
            prompts=["x"],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            trajectory_source=None,
            rewards={"reward": 1.0},
            started_at=datetime.now(),
            timing={},
        )
        defaults.update(overrides)
        SDK._build_result(rollout_dir, **defaults)
        return json.loads((rollout_dir / "result.json").read_text())

    return _build


@pytest.fixture
def job_factory(tmp_path):
    """Create a Evaluation with n task directories and a mocked SDK."""
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

    def _make(n_tasks=1, max_retries=0):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir(exist_ok=True)
        for i in range(n_tasks):
            td = tasks_dir / f"task-{i}"
            td.mkdir(exist_ok=True)
            (td / "task.toml").write_text(
                'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
                "[agent]\ntimeout_sec = 60\n[environment]\n"
            )
        cfg = EvaluationConfig(retry=RetryConfig(max_retries=max_retries))
        job = Evaluation(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)
        return job, tasks_dir

    return _make
