"""The warm pool: the next trial's sandbox is made while the current one runs.

Each job here is a real ``Evaluation`` on Docker running real ``Rollout``
lifecycles, whose sandbox steps are fakes writing one ordered event log:
``setup`` (the trial folder and the sandbox object), ``start`` (compose up),
``install`` (the sandbox user, lockdown, the freezer proof, the verifier
baseline: the hardening), ``verify`` and ``cleanup`` (the sandbox's stop and
delete). The log names the trial and the rollout object, so a sandbox made
for one trial and run by another, made twice, or never removed shows up.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import pytest

import benchflow.evaluation as evaluation_mod
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.rollout import Rollout
from benchflow.rollout._setup import _init_rollout

IMAGE = "example.invalid/warm-pool@sha256:" + "a" * 64


def _tasks(root: Path, names: list[str], image: str = IMAGE) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        task = root / name
        (task / "tests").mkdir(parents=True)
        (task / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n[agent]\n'
            f'timeout_sec = 60\n[environment]\ndocker_image = "{image}"\n'
        )
        (task / "instruction.md").write_text("Do nothing.\n")
        (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    return root


class Steps:
    """The fake sandbox steps and the log they write."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, int]] = []
        self.rollouts: dict[int, Rollout] = {}
        # Per trial name: an event a step waits for, and an error a step raises.
        self.hold: dict[tuple[str, str], asyncio.Event] = {}
        self.fail: dict[tuple[str, str], BaseException] = {}
        self.seen = asyncio.Event()
        self.watch: tuple[str, str] | None = None
        # Sandboxes alive (started, not yet cleaned up), and the most at once.
        self.live: set[int] = set()
        self.peak = 0

    def log(self, step: str, rollout: Rollout) -> None:
        name = rollout._config.task_path.name
        self.rollouts[id(rollout)] = rollout
        self.events.append((step, name, id(rollout)))
        if self.watch == (step, name):
            self.seen.set()

    async def gate(self, step: str, rollout: Rollout) -> None:
        name = rollout._config.task_path.name
        self.log(step, rollout)
        held = self.hold.get((step, name))
        if held is not None:
            await held.wait()
        error = self.fail.get((step, name))
        if error is not None:
            raise error

    def steps(self, name: str) -> list[str]:
        return [step for step, task, _ in self.events if task == name]

    def objects(self, name: str, step: str) -> list[int]:
        return [obj for s, task, obj in self.events if task == name and s == step]

    def index(self, step: str, name: str) -> int:
        return self.events.index(
            next(e for e in self.events if e[0] == step and e[1] == name)
        )


@pytest.fixture
def steps(monkeypatch) -> Steps:
    s = Steps()

    async def setup(self, *, verifier_only: bool = False) -> None:
        cfg = self._config
        (
            self._task,
            self._rollout_dir,
            self._rollout_paths,
            self._started_at,
            self._job_name,
            self._rollout_name,
        ) = _init_rollout(cfg.task_path, cfg.job_name, cfg.rollout_name, cfg.jobs_dir)
        self._phase = "setup"
        await s.gate("setup", self)

    async def start(self) -> None:
        s.live.add(id(self))
        s.peak = max(s.peak, len(s.live))
        await s.gate("start", self)
        self._phase = "started"

    async def install_agent(self) -> None:
        await s.gate("install", self)
        self._phase = "installed"

    async def verify(self) -> dict | None:
        await s.gate("verify", self)
        self._rewards = {"reward": 1.0}
        self._phase = "verified"
        return self._rewards

    async def cleanup(self) -> None:
        s.log("cleanup", self)
        s.live.discard(id(self))
        self._phase = "cleaned"

    monkeypatch.setattr(Rollout, "setup", setup)
    monkeypatch.setattr(Rollout, "start", start)
    monkeypatch.setattr(Rollout, "install_agent", install_agent)
    monkeypatch.setattr(Rollout, "verify", verify)
    monkeypatch.setattr(Rollout, "cleanup", cleanup)
    # No Docker daemon: the job's own sweeps, subnet pins and pulls are off.
    monkeypatch.setattr(Evaluation, "_sweep_docker", _no_sweep)
    monkeypatch.setattr(evaluation_mod, "_pin_evaluation_subnets", lambda *a: [])
    monkeypatch.setenv("BENCHFLOW_PREPULL", "off")
    monkeypatch.delenv("BENCHFLOW_WARM_POOL", raising=False)
    return s


async def _no_sweep(self: Evaluation) -> None:
    return None


def _job(
    tmp_path: Path,
    names: list[str],
    concurrency: int = 1,
    warm_pool: int | None = 1,
) -> Evaluation:
    """A nop job on Docker. One prepared sandbox by default (at
    --concurrency 1 that is an explicit --warm-pool 1); None: the default."""
    tasks = _tasks(tmp_path / "tasks", names)
    extra: dict[str, Any] = {} if warm_pool is None else {"warm_pool": warm_pool}
    config = EvaluationConfig(
        agent="nop",
        environment="docker",
        concurrency=concurrency,
        retry=RetryConfig(max_retries=0),
        integrity="off",
        **extra,
    )
    return Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="pool",
        config=config,
        preflight=False,
    )


def _trial_folders(tmp_path: Path) -> list[Path]:
    job = tmp_path / "jobs" / "pool"
    return sorted(p for p in job.iterdir() if p.is_dir() and "__" in p.name)


async def _run_with_trial_a_held(
    job: Evaluation, steps: Steps, *, until: tuple[str, str]
) -> Any:
    """Run the job; trial a stays in its verifier until ``until`` is logged.

    Without a pool nothing of trial b happens while a holds its only slot,
    so the wait times out and a goes on; the assertions then fail.
    """
    release = steps.hold[("verify", "a")] = asyncio.Event()
    steps.watch = until

    async def releaser() -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(steps.seen.wait(), timeout=5)
        release.set()

    helper = asyncio.ensure_future(releaser())
    try:
        return await job.run()
    finally:
        await helper


@pytest.mark.asyncio
async def test_next_trial_is_made_and_hardened_while_the_current_one_runs(
    tmp_path, steps
):
    job = _job(tmp_path, ["a", "b", "c"])
    result = await _run_with_trial_a_held(job, steps, until=("install", "b"))

    # b's sandbox was made and hardened while a still held the only slot.
    assert steps.index("install", "b") < steps.index("cleanup", "a")
    # Every trial ran its own steps once, in the cold start's order.
    for name in ("a", "b", "c"):
        assert steps.steps(name) == ["setup", "start", "install", "verify", "cleanup"]
    assert result.passed == 3


@pytest.mark.asyncio
async def test_a_prepared_sandbox_is_used_once_by_its_own_trial(tmp_path, steps):
    job = _job(tmp_path, ["a", "b", "c", "d"])
    result = await _run_with_trial_a_held(job, steps, until=("install", "b"))

    assert steps.index("setup", "b") < steps.index("cleanup", "a")
    for name in ("a", "b", "c", "d"):
        made = steps.objects(name, "setup")
        # One rollout per trial: made once, and every later step on it.
        assert len(made) == 1
        for step in ("start", "install", "verify", "cleanup"):
            assert steps.objects(name, step) == made
    # No rollout object served two trials.
    owners: dict[int, set[str]] = {}
    for _, name, obj in steps.events:
        owners.setdefault(obj, set()).add(name)
    assert all(len(names) == 1 for names in owners.values())
    assert len(_trial_folders(tmp_path)) == 4
    assert result.passed == 4


@pytest.mark.asyncio
async def test_an_entry_whose_image_changed_is_removed_and_the_trial_starts_cold(
    tmp_path, steps
):
    job = _job(tmp_path, ["a", "b"])
    release = steps.hold[("verify", "a")] = asyncio.Event()
    steps.watch = ("install", "b")

    async def change_b_image() -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(steps.seen.wait(), timeout=5)
        toml = tmp_path / "tasks" / "b" / "task.toml"
        toml.write_text(
            toml.read_text().replace(IMAGE, "example.invalid/other@sha256:" + "b" * 64)
        )
        release.set()

    helper = asyncio.ensure_future(change_b_image())
    result = await job.run()
    await helper

    made = steps.objects("b", "setup")
    # The prepared rollout was removed, and b ran on a new one.
    assert len(made) == 2
    stale, fresh = made
    # Each removed once (the stale one in the background, in any order).
    assert sorted(steps.objects("b", "cleanup")) == sorted([stale, fresh])
    assert steps.objects("b", "verify") == [fresh]
    stale_rollout = steps.rollouts[stale]
    assert stale_rollout._rollout_dir is not None
    assert not stale_rollout._rollout_dir.exists()
    assert len(_trial_folders(tmp_path)) == 2
    assert result.passed == 2


@pytest.mark.asyncio
async def test_a_hardening_failure_in_a_prepared_sandbox_fails_its_trial_as_cold(
    tmp_path, steps, monkeypatch
):
    steps.fail[("install", "b")] = RuntimeError("lockdown refused /app")
    job = _job(tmp_path, ["a", "b"])
    result = await _run_with_trial_a_held(job, steps, until=("install", "b"))

    # The hardening ran ahead (while a held the slot) and failed there.
    assert steps.index("install", "b") < steps.index("cleanup", "a")
    # b never ran an agent or a verifier on that sandbox; it was removed once.
    assert steps.steps("b") == ["setup", "start", "install", "cleanup"]
    b = result.results["b"]
    assert b.error is not None and "lockdown refused /app" in b.error

    # The same failure without the pool gives the same trial error.
    cold_steps = Steps()
    cold_steps.fail[("install", "b")] = RuntimeError("lockdown refused /app")
    steps.events.clear()
    steps.fail = cold_steps.fail
    steps.hold.clear()
    cold = _job(tmp_path / "cold", ["a", "b"])
    monkeypatch.setenv("BENCHFLOW_WARM_POOL", "off")
    cold_result = await cold.run()
    assert cold_result.results["b"].error == b.error
    assert steps.steps("b") == ["setup", "start", "install", "cleanup"]


@pytest.mark.asyncio
async def test_a_trial_cancelled_while_its_prepared_sandbox_starts_removes_it_once(
    tmp_path, steps
):
    # b's sandbox start never finishes; c's entry is never taken.
    steps.hold[("start", "b")] = asyncio.Event()
    release_a = steps.hold[("verify", "a")] = asyncio.Event()
    job = _job(tmp_path, ["a", "b", "c"])
    steps.watch = ("start", "b")
    runner = asyncio.ensure_future(job.run())
    await asyncio.wait_for(steps.seen.wait(), timeout=5)
    release_a.set()
    # Let a finish and b's trial take its slot and wait on the start.
    for _ in range(200):
        await asyncio.sleep(0.01)
        if "cleanup" in steps.steps("a"):
            break
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner

    # b was being made ahead of its turn, and is removed exactly once.
    assert steps.index("start", "b") < steps.index("cleanup", "a")
    assert steps.steps("b") == ["setup", "start", "cleanup"]
    # Every rollout that made a sandbox removed it once; none ran a verifier
    # after the cancel.
    for obj in {obj for _, _, obj in steps.events}:
        made = [s for s, _, o in steps.events if o == obj and s == "setup"]
        removed = [s for s, _, o in steps.events if o == obj and s == "cleanup"]
        assert len(removed) == len(made) == 1
    assert "verify" not in steps.steps("b")


@pytest.mark.asyncio
async def test_ctrl_c_drains_the_pool_and_removes_unused_trial_folders(tmp_path, steps):
    job = _job(tmp_path, ["a", "b", "c"])
    steps.hold[("verify", "a")] = asyncio.Event()  # a never finishes
    steps.watch = ("install", "b")
    runner = asyncio.ensure_future(job.run())
    await asyncio.wait_for(steps.seen.wait(), timeout=5)
    runner.cancel()  # what asyncio.run does on Ctrl-C
    with pytest.raises(asyncio.CancelledError):
        await runner

    # b was ready in the pool; the cancel removed it, and the folder it made.
    assert steps.steps("b") == ["setup", "start", "install", "cleanup"]
    b_rollout = steps.rollouts[steps.objects("b", "setup")[0]]
    assert b_rollout._rollout_dir is not None
    assert not b_rollout._rollout_dir.exists()
    assert steps.steps("a")[-1] == "cleanup"
    # c was never made: the pool prepares one ahead per running trial.
    assert steps.steps("c") == []
    remaining = [p.name for p in _trial_folders(tmp_path)]
    assert all(name.startswith("a__") for name in remaining)


@pytest.mark.asyncio
async def test_a_budget_style_stop_removes_entries_whose_trial_never_starts(
    tmp_path, steps, monkeypatch
):
    job = _job(tmp_path, ["a", "b", "c"])
    stop = {"now": False}
    original = getattr(Evaluation, "_make_warm_pool", None)

    def make_pool(self, remaining, stopped):
        return original(self, remaining, lambda: stopped() or stop["now"])

    monkeypatch.setattr(Evaluation, "_make_warm_pool", make_pool, raising=False)
    # The usage limit's stop: every trial not started yet is skipped.
    monkeypatch.setattr(
        evaluation_mod.UsageLimitStop, "skip", lambda self, name: stop["now"]
    )
    release = steps.hold[("verify", "a")] = asyncio.Event()
    steps.watch = ("install", "b")

    async def stop_after_b_is_ready() -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(steps.seen.wait(), timeout=5)
        stop["now"] = True
        release.set()

    helper = asyncio.ensure_future(stop_after_b_is_ready())
    await job.run()
    await helper
    assert steps.steps("b") == ["setup", "start", "install", "cleanup"]
    assert "verify" not in steps.steps("b")
    assert steps.steps("c") == []
    b_rollout = steps.rollouts[steps.objects("b", "setup")[0]]
    assert b_rollout._rollout_dir is not None
    assert not b_rollout._rollout_dir.exists()


def test_warm_pool_skips_tasks_whose_sandboxes_cannot_coexist(tmp_path):
    from benchflow._warm_pool import task_shares_nothing

    plain = _tasks(tmp_path / "t1", ["plain"]) / "plain"
    assert task_shares_nothing(plain)
    for body in (
        "services:\n  main:\n    container_name: fixed\n",
        "services:\n  main:\n    ports: ['8080:80']\n",
        "services:\n  main:\n    network_mode: host\n",
        "services:\n  main: {}\nnetworks:\n  n:\n    ipam:\n      config: []\n",
        "services:\n  main: {}\nnetworks:\n  n:\n    external: true\n",
        "services:\n  main: {}\nvolumes:\n  v:\n    name: shared\n",
        "name: fixed-project\nservices:\n  main: {}\n",
        "services: [not a mapping",
    ):
        (plain / "environment").mkdir(exist_ok=True)
        (plain / "environment" / "docker-compose.yaml").write_text(body)
        assert not task_shares_nothing(plain), body
    (plain / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    environment: {A: '1'}\n  db:\n    image: x\n"
    )
    assert task_shares_nothing(plain)


@pytest.mark.asyncio
async def test_no_pool_off_docker_or_when_every_trial_has_a_slot(tmp_path, steps):
    job = _job(tmp_path, ["a", "b"], concurrency=2)
    await job.run()
    # Both start at once: nothing is made ahead, every step runs in-trial.
    assert "warm_pool_wait" not in json.dumps(
        [
            json.loads((p / "result.json").read_text()).get("timing")
            for p in _trial_folders(tmp_path)
        ]
    )


@pytest.mark.asyncio
async def test_job_start_pulls_each_missing_task_image_once_in_parallel(
    tmp_path, steps, monkeypatch
):
    """Pre-pull: every distinct prebuilt image the job lacks, at its start."""
    from benchflow.sandbox import env_layers

    monkeypatch.setenv("BENCHFLOW_PREPULL", "on")
    monkeypatch.setenv("BENCHFLOW_WARM_POOL", "off")
    monkeypatch.setenv("BENCHFLOW_PULL_RECORD_DIR", str(tmp_path / "pulls"))
    present: set[str] = {"example.invalid/have:1"}
    pulls: list[str] = []
    flight = {"now": 0, "peak": 0}
    all_pulled = asyncio.Event()
    wanted = {"example.invalid/x:1", "example.invalid/y:1", "example.invalid/z:1"}

    async def docker(self, *args, stdin=None, timeout=None):
        if args[:2] == ("image", "inspect"):
            ref = args[2]
            if ref in present:
                info = [{"Id": f"sha256:{abs(hash(ref)):064x}"[:71]}]
                return env_layers._Run(stdout=json.dumps(info), stderr="", code=0)
            return env_layers._Run(stdout="", stderr=f"No such image: {ref}", code=1)
        if args[0] == "pull":
            flight["now"] += 1
            flight["peak"] = max(flight["peak"], flight["now"])
            await asyncio.sleep(0.05)
            flight["now"] -= 1
            pulls.append(args[-1])
            present.add(args[-1])
            if wanted <= present:
                all_pulled.set()
            return env_layers._Run(stdout="", stderr="", code=0)
        if args[0] == "info":
            text = "daemon-1\n" if "{{.ID}}" in args else "[]\n"
            return env_layers._Run(stdout=text, stderr="", code=0)
        return env_layers._Run(stdout="", stderr="", code=0)

    monkeypatch.setattr(env_layers.DockerLayers, "_docker", docker)
    tasks = tmp_path / "tasks"
    _tasks(tasks, ["a", "b"], image="example.invalid/x:1")
    _tasks(tasks, ["c"], image="example.invalid/y:1")
    _tasks(tasks, ["d"], image="example.invalid/z:1")
    _tasks(tasks, ["e"], image="example.invalid/have:1")
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="pool",
        config=EvaluationConfig(
            agent="nop",
            environment="docker",
            concurrency=1,
            retry=RetryConfig(max_retries=0),
            integrity="off",
        ),
        preflight=False,
    )
    release = steps.hold[("verify", "a")] = asyncio.Event()

    async def releaser() -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(all_pulled.wait(), timeout=5)
        release.set()

    helper = asyncio.ensure_future(releaser())
    await job.run()
    await helper

    # Each missing image once, several at a time, while the first trial ran;
    # the image the daemon had is not pulled.
    assert sorted(pulls) == sorted(wanted)
    assert flight["peak"] >= 2
    assert steps.steps("a")[-1] == "cleanup"
    # Recorded as BenchFlow's pulls, as a trial's own pull is.
    records = "".join(p.read_text() for p in (tmp_path / "pulls").glob("*.jsonl"))
    for ref in wanted:
        assert ref in records
    assert "example.invalid/have:1" not in records


# --- round 2: the pool's size, budget, platform -------------


async def _hold_first_wave(steps: Steps, names: list[str], seconds: float) -> None:
    """Keep the running trials in their verifier for a while, so the pool
    fills as far as it may."""
    events = [steps.hold.setdefault(("verify", n), asyncio.Event()) for n in names]
    await asyncio.sleep(seconds)
    for event in events:
        event.set()


@pytest.mark.asyncio
async def test_live_sandboxes_stay_within_concurrency_plus_the_default_pool(
    tmp_path, steps
):
    job = _job(tmp_path, ["a", "b", "c", "d", "e", "f"], concurrency=2, warm_pool=None)
    helper = asyncio.ensure_future(_hold_first_wave(steps, ["a", "b"], 0.5))
    result = await job.run()
    await helper
    # Two running and one prepared: never the 2 x --concurrency of 9f783126.
    assert steps.peak == 3
    assert result.passed == 6


@pytest.mark.asyncio
async def test_concurrency_one_keeps_one_sandbox(tmp_path, steps):
    job = _job(tmp_path, ["a", "b", "c"], concurrency=1, warm_pool=None)
    helper = asyncio.ensure_future(_hold_first_wave(steps, ["a"], 0.5))
    result = await job.run()
    await helper
    assert steps.peak == 1
    for name in ("a", "b", "c"):
        assert steps.steps(name) == ["setup", "start", "install", "verify", "cleanup"]
    assert result.passed == 3


@pytest.mark.asyncio
async def test_warm_pool_n_bounds_prepared_sandboxes(tmp_path, steps):
    job = _job(tmp_path, ["a", "b", "c", "d", "e", "f"], concurrency=2, warm_pool=2)
    helper = asyncio.ensure_future(_hold_first_wave(steps, ["a", "b"], 0.5))
    await job.run()
    await helper
    assert steps.peak == 4


@pytest.mark.asyncio
async def test_the_start_line_names_the_pool(tmp_path, steps, caplog):
    job = _job(tmp_path, ["a", "b", "c"], concurrency=2, warm_pool=None)
    with caplog.at_level("INFO", logger="benchflow.evaluation"):
        await job.run()
    assert any(
        "(concurrency=2, warm pool: up to 1 prepared sandbox)" in r.getMessage()
        for r in caplog.records
    )


def test_warm_pool_size_defaults_and_bounds(monkeypatch):
    from benchflow._warm_pool import warm_pool_size

    monkeypatch.delenv("BENCHFLOW_WARM_POOL", raising=False)
    assert [warm_pool_size(c) for c in (1, 2, 8)] == [0, 1, 1]
    assert warm_pool_size(4, 0) == 0 and warm_pool_size(4, 3) == 3
    assert warm_pool_size(4, 9) == 4
    monkeypatch.setenv("BENCHFLOW_WARM_POOL", "off")
    assert warm_pool_size(4, 3) == 0


@pytest.mark.parametrize(
    "service",
    [
        "volumes: ['./state:/state']",
        "volumes: ['/srv/data:/data:ro']",
        "volumes: ['~/cache:/cache']",
        "volumes: [{type: bind, source: ./state, target: /state}]",
        "volumes: [{source: /srv/x, target: /x}]",
        "devices: ['/dev/fuse:/dev/fuse']",
        "deploy: {resources: {reservations: {devices: [{capabilities: [gpu]}]}}}",
        "pid: host",
        "ipc: host",
        "cgroup: host",
        "privileged: true",
        "gpus: all",
        "runtime: nvidia",
        "volumes_from: ['container:shared-data']",
    ],
)
def test_tasks_sharing_host_state_are_never_prepared_ahead(tmp_path, service):
    from benchflow._warm_pool import task_shares_nothing

    task = _tasks(tmp_path / "t", ["host"]) / "host"
    (task / "environment").mkdir(exist_ok=True)
    (task / "environment" / "docker-compose.yaml").write_text(
        f"services:\n  main:\n    {service}\n"
    )
    assert not task_shares_nothing(task)


def test_named_volumes_and_a_shareable_ipc_still_coexist(tmp_path):
    from benchflow._warm_pool import task_shares_nothing

    task = _tasks(tmp_path / "t", ["ok"]) / "ok"
    (task / "environment").mkdir(exist_ok=True)
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    volumes: ['data:/data']\n    ipc: shareable\n"
        "volumes:\n  data: {}\n"
    )
    assert task_shares_nothing(task)


@pytest.mark.asyncio
async def test_a_prepared_trial_is_charged_for_its_sandbox_before_its_turn(
    tmp_path, monkeypatch
):
    """L1: the prestart ran before the job admitted the trial; its seconds go
    to the trial at adoption, as a cold trial's setup counts."""
    from benchflow.budget import Budget, BudgetGuard
    from benchflow.rollout import RolloutConfig

    async def prestart(self) -> None:
        await asyncio.sleep(0.3)
        self._phase = "installed"

    monkeypatch.setattr(Rollout, "_prestart_steps", prestart)
    task = _tasks(tmp_path / "tasks", ["a"]) / "a"
    rollout = Rollout(
        RolloutConfig.from_legacy(
            task_path=task,
            agent="nop",
            environment="docker",
            jobs_dir=str(tmp_path / "jobs"),
            job_name="pool",
        )
    )
    guard = BudgetGuard(Budget(max_sandbox_seconds=10_000))
    rollout.begin_prestart()
    await asyncio.sleep(0.5)  # prepared, waiting for its slot
    with guard.track_trial("a"):
        await rollout._setup_and_start()
    spent = guard.spent()["sandbox_seconds"]
    assert 0.25 <= spent <= 0.45


@pytest.mark.asyncio
async def test_pre_pull_leaves_a_platform_pinned_task_to_its_own_start(
    tmp_path, steps, monkeypatch
):
    """L2: a task whose compose file pins a service's platform is not
    pre-pulled (the trial's own start pulls that variant)."""
    from benchflow.sandbox import _prepull

    monkeypatch.setenv("BENCHFLOW_PREPULL", "on")
    asked: list[list[str]] = []

    async def prepull_images(refs, **_):
        asked.append(list(refs))
        return {}

    monkeypatch.setattr(_prepull, "prepull_images", prepull_images)
    tasks = tmp_path / "tasks"
    _tasks(tasks, ["a"], image="example.invalid/x:1")
    _tasks(tasks, ["b"], image="example.invalid/arm:1")
    (tasks / "b" / "environment").mkdir(exist_ok=True)
    (tasks / "b" / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    platform: linux/amd64\n"
    )
    job = _job(tmp_path, [], concurrency=1, warm_pool=None)
    pulling = job._start_prepull([tasks / "a", tasks / "b"])
    assert pulling is not None
    await pulling
    assert asked == [["example.invalid/x:1"]]
    assert _prepull.task_sets_platform(tasks / "b")
    assert not _prepull.task_sets_platform(tasks / "a")


# --- round 3: the pool is job-wide across worker shards (R2-M1) ---------------


@pytest.mark.parametrize(
    ("concurrency", "worker_concurrency", "pool", "bound"),
    [
        (8, 2, None, 9),
        (8, 2, 1, 9),
        (8, 1, 1, 9),
        (8, 2, 3, 11),
        (4, 1, 0, 4),
    ],
)
def test_sharded_jobs_keep_live_sandboxes_within_concurrency_plus_n(
    monkeypatch, concurrency, worker_concurrency, pool, bound
):
    """With --worker-concurrency every worker runs its own Evaluation; the
    job's N prepared sandboxes are shared out, so all workers together have
    at most --concurrency + N sandboxes alive."""
    from benchflow._warm_pool import warm_pool_size
    from benchflow.eval_sharding import _config_payload, plan_eval_shards
    from benchflow.eval_worker import _evaluation_config

    monkeypatch.delenv("BENCHFLOW_WARM_POOL", raising=False)
    config = EvaluationConfig(
        agent="nop", environment="docker", concurrency=concurrency, warm_pool=pool
    )
    plan = plan_eval_shards(
        [f"t{i}" for i in range(40)],
        total_concurrency=concurrency,
        worker_concurrency=worker_concurrency,
    )
    live = 0
    for shard in plan.shards:
        worker = _evaluation_config(_config_payload(config, shard=shard))
        assert worker.concurrency == shard.concurrency
        live += shard.concurrency + warm_pool_size(worker.concurrency, worker.warm_pool)
    assert live <= bound


def test_a_worker_payload_without_a_pool_prepares_none():
    from benchflow.eval_sharding import _config_payload, plan_eval_shards
    from benchflow.eval_worker import _evaluation_config

    config = EvaluationConfig(agent="nop", environment="docker", concurrency=4)
    (shard,) = plan_eval_shards(
        ["t0"], total_concurrency=4, worker_concurrency=4
    ).shards
    payload = _config_payload(config, shard=shard)
    payload.pop("warm_pool")
    assert _evaluation_config(payload).warm_pool == 0


def test_a_volume_through_driver_options_is_host_state(tmp_path):
    """R2-L1: the local driver's bind (driver_opts) or another driver."""
    from benchflow._warm_pool import task_shares_nothing

    task = _tasks(tmp_path / "t", ["vol"]) / "vol"
    (task / "environment").mkdir(exist_ok=True)
    compose = task / "environment" / "docker-compose.yaml"
    compose.write_text(
        "services:\n  main:\n    volumes: ['data:/data']\nvolumes:\n  data:\n"
        "    driver_opts: {type: none, o: bind, device: /srv/shared}\n"
    )
    assert not task_shares_nothing(task)
    compose.write_text(
        "services:\n  main:\n    volumes: ['data:/data']\nvolumes:\n  data:\n"
        "    driver: some-nfs-plugin\n"
    )
    assert not task_shares_nothing(task)
    # A project network with a driver stays fine; a sibling's volumes too.
    compose.write_text(
        "services:\n  main:\n    networks: [n]\n    volumes_from: [db]\n  db: {}\n"
        "networks:\n  n: {driver: bridge}\n"
    )
    assert task_shares_nothing(task)
