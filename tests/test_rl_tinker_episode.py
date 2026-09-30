"""The Tinker cookbook's episode logic (docs/examples/rl/tinker/tinker_episode.py).

No network and no Tinker packages: the task runtime is a fake with the
`bf.TaskRuntime` surface the episode uses (bash, verify, close, rollout_dir).
Covers the training rule (what each ending is worth), that the sandbox is
closed and its slot returned on every path, the sandbox cap, the streak
breaker for drops, and the held-out statistics.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import benchflow as bf

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "examples" / "rl" / "tinker"
sys.path.insert(0, str(EXAMPLE))
import tinker_episode as ep  # noqa: E402
import tinker_stats as stats  # noqa: E402


@dataclass
class Exec:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeRuntime:
    """The TaskRuntime surface an Episode uses, scripted per test."""

    def __init__(self, config: Any, script: dict[str, Any]) -> None:
        self.config = config
        self.script = script
        self.commands: list[str] = []
        self.verified = 0
        self.closed = 0
        self.rollout_dir = Path(config.jobs_dir) / config.job_name / "task__1"
        self.rollout_dir.mkdir(parents=True, exist_ok=True)

    async def bash(
        self, command: str, *, timeout_sec: int = 30, user: Any = None
    ) -> Exec:
        self.commands.append(command)
        outcome = self.script.get("bash", Exec(stdout="ok\n"))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def verify(self) -> Any:
        self.verified += 1
        outcome = self.script.get("verify", SimpleNamespace(reward=1.0))
        if outcome == "hang":
            await asyncio.sleep(3600)
        if isinstance(outcome, BaseException):
            raise outcome
        outcome.rollout_dir = self.rollout_dir
        return outcome

    async def close(self) -> None:
        self.closed += 1
        if "close" in self.script:
            raise self.script["close"]


def make_episode(tmp_path: Path, slots: ep.SandboxSlots | None = None, **script: Any):
    runtimes: list[FakeRuntime] = []

    async def factory(config: Any) -> FakeRuntime:
        if "start" in script:
            raise script["start"]
        runtime = FakeRuntime(config, script)
        runtimes.append(runtime)
        return runtime

    settings = ep.EpisodeSettings(
        sandbox="docker",
        jobs_dir=str(tmp_path / "trials"),
        verify_timeout_sec=script.pop("verify_timeout", 900.0),
        episode_timeout_sec=script.pop("episode_timeout", 900.0),
    )
    episode = ep.Episode(
        tmp_path / "task-a",
        settings,
        slots=slots or ep.SandboxSlots(4),
        job_name="train-0000",
        runtime_factory=factory,
        clock=script.pop("clock", None) or time.monotonic,
    )
    return episode, runtimes


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# -- the training rule -----------------------------------------------------------


def result(**fields: Any) -> SimpleNamespace:
    base = {"reward": None, "rewards": None, "error": None, "verifier_error": None}
    return SimpleNamespace(**{**base, **fields})


@pytest.mark.parametrize(
    ("fields", "acted", "reward", "reason"),
    [
        ({"reward": 1.0}, True, 1.0, "scored"),
        ({"reward": 0.0}, True, 0.0, "scored"),
        ({"reward": 0.5, "verifier_error": "warning"}, True, 0.5, "scored"),
        ({"rewards": {"reward": 1}}, True, 1.0, "scored"),
        # A clean run: the policy never acted, so a missing reward is a drop.
        (
            {"verifier_error": "verifier crashed: boom"},
            False,
            None,
            "verifier_crash_clean_run",
        ),
        ({}, False, None, "verifier_crash_clean_run"),
        ({"error": "Sandbox startup failed: quota"}, True, None, "sandbox_start"),
        # After the policy acted, a missing reward scores 0 by what went wrong.
        ({"verifier_error": "verifier timed out after 60s"}, True, 0.0, "timeout"),
        ({"verifier_error": "verifier crashed: boom"}, True, 0.0, "verifier_error"),
        ({"error": "connection lost"}, True, 0.0, "run_error"),
        ({}, True, 0.0, "no_reward"),
    ],
)
def test_the_training_rule_maps_each_verify_result(fields, acted, reward, reason):
    decision = ep.decide(result(**fields), policy_acted=acted)
    assert decision.reward == reward
    assert decision.reason == reason
    assert decision.dropped is (reward is None)


@pytest.mark.parametrize("bad", [True, math.nan, math.inf, "1.0"])
def test_a_reward_must_be_a_finite_number(bad):
    decision = ep.decide(result(reward=bad), policy_acted=True)
    assert decision.reason == "no_reward"
    assert decision.reward == 0.0


def test_decisions_keep_drops_and_zeros_apart():
    with pytest.raises(ValueError):
        ep.Decision(None, "timeout")  # a drop needs an infrastructure reason
    with pytest.raises(ValueError):
        ep.Decision(0.0, "sandbox_start")  # an infrastructure reason is a drop
    with pytest.raises(ValueError):
        ep.zero("sandbox_start")
    assert ep.scored(1).solved and not ep.scored(0.99).solved


# -- the episode's lifecycle ----------------------------------------------------------


def test_a_submitted_episode_is_verified_once_and_closed(tmp_path):
    slots = ep.SandboxSlots(2)
    episode, runtimes = make_episode(tmp_path, slots)

    async def go():
        await episode.start()
        assert slots.in_use == 1 and episode in ep.LIVE
        assert await episode.run_bash("ls") == "ok\n"
        assert await episode.submit("42 it's") is None
        first = await episode.finish()
        again = await episode.finish()
        return first, again

    first, again = run(go())
    rt = runtimes[0]
    assert first == again == ep.Decision(1.0, "scored")
    assert rt.verified == 1 and rt.closed == 1
    assert slots.in_use == 0 and episode not in ep.LIVE
    assert episode.policy_acted and episode.submitted
    # submit writes the answer file the way the TRL adapter does, quoted
    assert rt.commands[1] == (
        "mkdir -p $(dirname /workdir/answer.txt) && "
        "printf %s '42 it'\"'\"'s' > /workdir/answer.txt"
    )


def test_a_sandbox_that_never_started_is_dropped_and_frees_its_slot(tmp_path):
    slots = ep.SandboxSlots(1)
    episode, runtimes = make_episode(tmp_path, slots, start=RuntimeError("no capacity"))
    with pytest.raises(ep.InfrastructureError) as info:
        run(episode.start())
    assert info.value.reason == "sandbox_start"
    assert episode.decision == ep.dropped("sandbox_start", RuntimeError("no capacity"))
    assert slots.in_use == 0 and not runtimes and episode.closed
    run(episode.close())  # idempotent
    assert slots.in_use == 0


@pytest.mark.parametrize(
    ("acted", "reason"), [(True, "verifier_error"), (False, "verifier_crash_clean_run")]
)
def test_a_verifier_that_raises_still_closes_the_sandbox(tmp_path, acted, reason):
    episode, runtimes = make_episode(tmp_path, verify=RuntimeError("tests missing"))

    async def go():
        await episode.start()
        if acted:
            await episode.run_bash("rm -rf /verifier")
        return await episode.finish()

    decision = run(go())
    assert decision.reason == reason
    assert decision.reward == (0.0 if acted else None)
    assert runtimes[0].closed == 1 and episode.slots.in_use == 0


def test_a_hung_verifier_is_cut_off_and_the_sandbox_closed(tmp_path):
    episode, runtimes = make_episode(tmp_path, verify="hang", verify_timeout=0.05)

    async def go():
        await episode.start()
        await episode.run_bash("sleep 1")
        return await episode.finish()

    decision = run(go())
    assert decision.reason == "verifier_error" and "TimeoutError" in decision.detail
    assert runtimes[0].closed == 1


def test_a_timed_out_episode_scores_zero_without_a_verifier_run(tmp_path):
    now = [0.0]
    episode, runtimes = make_episode(
        tmp_path, episode_timeout=10.0, clock=lambda: now[0]
    )

    async def go():
        await episode.start()
        assert not episode.past_deadline()
        now[0] = 11.0
        assert episode.past_deadline()
        return await episode.time_out()

    decision = run(go())
    assert decision == ep.Decision(0.0, "timeout", "the episode ran past 10 s")
    assert runtimes[0].verified == 0 and runtimes[0].closed == 1


def test_a_cancelled_finish_still_closes_the_sandbox(tmp_path):
    episode, runtimes = make_episode(tmp_path, verify="hang")

    async def go():
        await episode.start()
        task = asyncio.create_task(episode.finish())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(go())
    assert runtimes[0].closed == 1 and episode.slots.in_use == 0
    assert episode not in ep.LIVE


def test_a_failing_close_still_frees_the_slot(tmp_path):
    episode, runtimes = make_episode(tmp_path, close=RuntimeError("delete failed"))

    async def go():
        await episode.start()
        return await episode.finish()

    assert run(go()).reason == "scored"
    assert runtimes[0].closed == 1
    assert episode.slots.in_use == 0 and episode not in ep.LIVE


def test_command_output_is_what_the_shared_harness_shows(tmp_path):
    long = "x" * 5000
    episode, _ = make_episode(tmp_path, bash=Exec(1, stdout=long, stderr="err\n"))

    async def go():
        await episode.start()
        try:
            return await episode.run_bash("cat big")
        finally:
            await episode.close()

    text = run(go())
    assert len(text) == ep.MAX_OUTPUT_CHARS
    assert text.endswith(ep.TRUNCATION_MARKER) and text.startswith("x")
    assert ep.truncate("out\nerr\n", 2000) == "out\nerr\n"


def test_a_command_that_fails_to_run_is_reported_and_the_episode_goes_on(tmp_path):
    episode, _ = make_episode(
        tmp_path, bash=RuntimeError("Command timed out after 30 seconds")
    )

    async def go():
        await episode.start()
        text = await episode.run_bash("sleep 99")
        return text, await episode.finish()

    text, decision = run(go())
    assert json.loads(text) == {"error": "Command timed out after 30 seconds"}
    assert decision.reason == "scored"  # the verifier still ran


def test_close_all_live_closes_every_open_sandbox(tmp_path):
    slots = ep.SandboxSlots(3)
    pairs = [make_episode(tmp_path / str(i), slots) for i in range(3)]

    async def go():
        for episode, _ in pairs:
            await episode.start()
        assert {e for e, _ in pairs} <= ep.LIVE
        return await ep.close_all_live()

    assert run(go()) >= 3
    assert not ep.LIVE
    assert slots.in_use == 0 and all(rts[0].closed == 1 for _, rts in pairs)


def test_the_sandbox_cap_holds_across_concurrent_episodes(tmp_path):
    slots = ep.SandboxSlots(2)
    pairs = [make_episode(tmp_path / str(i), slots) for i in range(6)]

    async def one(episode):
        await episode.start()
        await asyncio.sleep(0.01)
        await episode.finish()

    async def go():
        await asyncio.gather(*(one(e) for e, _ in pairs))

    run(go())
    assert slots.peak == 2 and slots.in_use == 0


def test_the_rollout_record_follows_the_shared_harness(tmp_path):
    episode, runtimes = make_episode(tmp_path)

    async def go():
        await episode.start()
        await episode.run_bash("ls")
        await episode.finish()

    run(go())
    episode.ended = "submitted"
    episode.write_record(episode.record([{"role": "user", "content": "hi"}], model="m"))
    record = json.loads(
        (runtimes[0].rollout_dir / "policy" / "messages.json").read_text()
    )
    assert record["reward"] == 1.0 and record["reason"] == "scored"
    assert record["dropped"] is False and record["policy_acted"] is True
    assert record["ended"] == "submitted" and record["messages"][0]["content"] == "hi"
    lines = (tmp_path / "trials" / "rollouts.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["task_id"] == "task-a"


# -- drops ---------------------------------------------------------------------------------


def test_drops_are_counted_and_a_streak_stops_the_run(tmp_path):
    log = ep.DropLog(path=tmp_path / "drops.jsonl", max_consecutive=3)
    error = ep.InfrastructureError(ep.dropped("sandbox_start", "quota"), task="t")
    log.add(error, task="t", where="train-0000")
    log.add(RuntimeError("sampler 503"), task="t", where="train-0000")
    log.completed()  # an episode finished: the streak resets
    log.add(error, task="t", where="train-0001")
    log.add(error, task="t", where="train-0001")
    with pytest.raises(ep.TooManyInfrastructureFailures):
        log.add(error, task="t", where="train-0001")
    assert log.counts == {"sandbox_start": 4, "model_endpoint": 1}
    rows = [
        json.loads(line) for line in (tmp_path / "drops.jsonl").read_text().splitlines()
    ]
    assert [r["reason"] for r in rows][:2] == ["sandbox_start", "model_endpoint"]


# -- statistics ------------------------------------------------------------------------------


def test_solve_rate_is_the_mean_of_per_task_rates():
    results = {"a": [1.0, 1.0, 0.0, 0.0], "b": [1.0], "c": []}
    assert stats.solve_rate(results) == pytest.approx(0.75)
    summary = stats.summarize(results)
    assert summary["tasks"] == 2 and summary["episodes"] == 5 and summary["solved"] == 3
    low, high = stats.bootstrap_interval({"a": [1.0] * 3, "b": [1.0] * 2})
    assert low == high == 1.0


def test_the_paired_difference_is_zero_for_identical_policies():
    results = {f"t{i}": [1.0, 0.0] for i in range(10)}
    diff = stats.paired_difference(results, results, samples=2000)
    assert diff["delta"] == 0.0 and diff["low"] <= 0.0 <= diff["high"]
    better = {t: [1.0, 1.0] for t in results}
    up = stats.paired_difference(results, better, samples=2000)
    assert up["delta"] == pytest.approx(0.5) and up["low"] > 0.0


def test_wilson_interval_bounds():
    low, high = stats.wilson_interval(0, 10)
    assert low == 0.0 and 0.2 < high < 0.35
    assert all(math.isnan(x) for x in stats.wilson_interval(5, 0))


# -- the example keeps to BenchFlow's public surface -------------------------------------------


def test_the_cookbook_uses_only_public_benchflow_names():
    for path in EXAMPLE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "benchflow"
            ):
                pytest.fail(
                    f"{path.name} imports {node.module}; use `import benchflow as bf`"
                )
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "bf"
            ):
                assert node.attr in bf.__all__, f"{path.name} uses bf.{node.attr}"


# -- groups ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rewards", "kind"),
    [
        ([1.0, 0.0, 1.0], "mixed"),
        ([0.5, 0.25], "mixed"),
        ([1.0, 1.0], "all_solved"),
        ([0.0, 0.0, 0.0], "all_failed"),
        ([0.5, 0.5], "constant_partial"),
    ],
)
def test_groups_are_classified_by_reward_variance(rewards, kind):
    assert ep.group_kind(rewards) == kind


def test_the_group_log_counts_and_records_every_group(tmp_path):
    log = ep.GroupLog(path=tmp_path / "groups.jsonl")
    log.add([1.0, 0.0], task="a", where="train-0000", dropped=False)
    log.add([0.0, 0.0], task="b", where="train-0000", dropped=True)
    assert log.counts == {"mixed": 1, "all_failed": 1}
    rows = [json.loads(x) for x in (tmp_path / "groups.jsonl").read_text().splitlines()]
    assert [(r["task"], r["kind"], r["dropped"]) for r in rows] == [
        ("a", "mixed", False),
        ("b", "all_failed", True),
    ]
