"""The Tinker cookbook's environment for tinker-cookbook (docs/examples/rl/tinker/tinker_env.py).

No network: the task runtime, the renderer and the policy are fakes. Skipped
when tinker-cookbook is not installed (it is not a BenchFlow dependency; the
cookbook's README installs it). Covers the tools and how an episode ends
(each ending is verified except a wall-clock timeout), that every path closes
the sandbox, the group cleanup, and the rollout strategy that drops only
infrastructure failures.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("tinker_cookbook")

import tinker
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.rl import types

from benchflow.integrations.rewards import dropped

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "examples" / "rl" / "tinker"
sys.path.insert(0, str(EXAMPLE))
import tinker_env as te  # noqa: E402
import tinker_episode as ep  # noqa: E402


@dataclass
class Exec:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeRuntime:
    def __init__(self, config: Any, verify: Any) -> None:
        self.commands: list[str] = []
        self.verified = 0
        self.closed = 0
        self._verify = verify
        self.rollout_dir = Path(config.jobs_dir) / config.job_name / f"t__{id(self)}"
        self.rollout_dir.mkdir(parents=True, exist_ok=True)

    async def bash(
        self, command: str, *, timeout_sec: int = 30, user: Any = None
    ) -> Exec:
        self.commands.append(command)
        return Exec(stdout=f"ran: {command}\n")

    async def verify(self) -> Any:
        self.verified += 1
        if isinstance(self._verify, BaseException):
            raise self._verify
        return SimpleNamespace(reward=self._verify, rollout_dir=self.rollout_dir)

    async def close(self) -> None:
        self.closed += 1


class FakeRenderer:
    """Enough of a Renderer for message-level episodes."""

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        self.tools = tools
        return [{"role": "system", "content": json.dumps(tools)}]

    def get_stop_sequences(self):
        return []

    def build_generation_prompt(self, messages, **_):
        return tinker.ModelInput.from_ints([1] * len(messages))


def make_env(
    tmp_path: Path, *, verify: Any = 1.0, renderer_name: str = "qwen3_5", **settings
):
    runtimes: list[FakeRuntime] = []

    async def factory(config: Any) -> FakeRuntime:
        runtimes.append(FakeRuntime(config, verify))
        return runtimes[-1]

    task = te.TaskSpec("t", tmp_path / "t", "Write 4 to the answer file.")
    config = te.EnvConfig(
        model_name="Qwen/Qwen3.6-35B-A3B",
        renderer_name=renderer_name,
        episode=ep.EpisodeSettings(jobs_dir=str(tmp_path / "trials"), **settings),
    )
    env = te.BenchFlowEnv(
        task,
        config,
        FakeRenderer(),
        job_name="train-0000",
        slots=ep.SandboxSlots(4),
        runtime_factory=factory,
    )
    return env, runtimes


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(
        function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)),
        id=f"call_{name}",
    )


def assistant(*calls: ToolCall) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# -- the harness ---------------------------------------------------------------------


def test_the_tools_are_the_shared_harness_tools():
    from benchflow.integrations.trl import bash_tool_schemas

    assert [s["function"] for s in bash_tool_schemas()] == [
        te.RUN_BASH_SPEC,
        te.SUBMIT_SPEC,
    ]


def test_qwen3_5_renderers_get_the_openai_tool_wrapper(tmp_path):
    env, _ = make_env(tmp_path)
    assert env.messages.initial_messages[0]["content"] == json.dumps(
        [
            {"type": "function", "function": s}
            for s in (te.RUN_BASH_SPEC, te.SUBMIT_SPEC)
        ]
    )
    assert te.tool_specs("gpt_oss_medium_reasoning") == [
        te.RUN_BASH_SPEC,
        te.SUBMIT_SPEC,
    ]
    user = env.messages.initial_messages[-1]
    assert user == {
        "role": "user",
        "content": "Write 4 to the answer file." + ep.HARNESS_MESSAGE,
    }
    assert len(env.messages.initial_messages) == 2  # no system prompt besides the tools


# -- how an episode ends ----------------------------------------------------------------


def test_submit_writes_the_answer_and_the_verifier_scores_the_episode(tmp_path):
    env, runtimes = make_env(tmp_path, verify=1.0)

    async def go():
        await env.episode.start()
        first = await env.messages.step(assistant(call("run_bash", command="ls")))
        last = await env.messages.step(
            assistant(
                call("submit", answer="4"), call("run_bash", command="rm -rf /workdir")
            )
        )
        return first, last

    first, last = run(go())
    rt = runtimes[0]
    assert not first.episode_done and first.reward == 0.0
    assert last.episode_done and last.reward == 1.0
    assert (
        rt.commands[0] == "ls" and "printf %s 4 > /workdir/answer.txt" in rt.commands[1]
    )
    assert len(rt.commands) == 2  # nothing runs after submit
    assert rt.verified == 1 and rt.closed == 1
    tool_texts = [m["content"] for m in env.messages.history if m["role"] == "tool"]
    assert tool_texts == [
        "ran: ls\n",
        "submission recorded",
        "not run: the task was already submitted",
    ]


def test_bad_tool_arguments_are_reported_like_the_shared_harness(tmp_path):
    env, runtimes = make_env(tmp_path)

    async def go():
        await env.episode.start()
        await env.messages.step(assistant(call("run_bash", cmd="ls"), call("submit")))
        await env.episode.close()

    run(go())
    tool_texts = [m["content"] for m in env.messages.history if m["role"] == "tool"]
    assert tool_texts == [
        json.dumps({"error": "'command'"}),
        json.dumps({"error": "'answer'"}),
    ]
    assert runtimes[0].commands == [] and not env.episode.policy_acted


def test_the_turn_limit_ends_with_a_verifier_run(tmp_path):
    env, runtimes = make_env(tmp_path, verify=0.0)

    async def go():
        await env.episode.start()
        results = []
        for _ in range(env.config.max_turns):
            results.append(
                await env.messages.step(assistant(call("run_bash", command="true")))
            )
        return results

    results = run(go())
    turns = env.config.max_turns
    assert [r.episode_done for r in results] == [False] * (turns - 1) + [True]
    assert results[-1].reward == 0.0 and runtimes[0].verified == 1
    assert (env.episode.decision.reward, env.episode.decision.reason) == (0.0, "scored")


def test_a_verifier_crash_on_a_clean_run_is_dropped(tmp_path):
    env, runtimes = make_env(tmp_path, verify=RuntimeError("verifier image broken"))

    async def go():
        await env.episode.start()
        # No tool call: the episode ends and the untouched sandbox is verified.
        await env.messages.step({"role": "assistant", "content": "I am done."})

    with pytest.raises(ep.InfrastructureError) as info:
        run(go())
    assert info.value.reason == "verifier_crash_clean_run"
    assert runtimes[0].closed == 1


@pytest.mark.parametrize("stop", ["parse_error", "max_tokens", "context_overflow"])
def test_a_protocol_break_scores_what_the_policy_left(tmp_path, stop):
    env, runtimes = make_env(tmp_path, verify=1.0)

    async def inner_step(action, *, extra=None):
        return types.StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={f"stop/{stop}": 1.0},
        )

    env.inner.step = inner_step

    async def go():
        await env.episode.start()
        await env.episode.run_bash("echo 4 > /workdir/answer.txt")
        return await env.step([1, 2, 3])

    result = run(go())
    assert (
        result.reward == 1.0 and runtimes[0].verified == 1 and runtimes[0].closed == 1
    )
    assert result.metrics["bf/solved"] == 1.0
    assert result.metrics[f"bf/ended/{stop}"] == 1.0
    assert env.episode.ended == stop
    record = json.loads(
        (runtimes[0].rollout_dir / "policy" / "messages.json").read_text()
    )
    assert record["reason"] == "scored" and record["ended"] == stop


def test_past_the_wall_clock_budget_an_episode_scores_zero_unverified(tmp_path):
    env, runtimes = make_env(tmp_path, verify=1.0, episode_timeout_sec=0.01)

    async def inner_step(action, *, extra=None):
        return types.StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=tinker.ModelInput.from_ints([1]),
            next_stop_condition=[],
        )

    env.inner.step = inner_step

    async def go():
        await env.episode.start()
        await asyncio.sleep(0.02)
        return await env.step([1])

    result = run(go())
    assert result.episode_done and result.reward == 0.0
    assert (
        result.metrics["bf/reason/timeout"] == 1.0
        and result.metrics["stop/rollout_timeout"] == 1.0
    )
    assert runtimes[0].verified == 0 and runtimes[0].closed == 1


def test_decision_metrics_carry_every_key():
    metrics = te.decision_metrics(ep.Decision(0.0, "verifier_error"), "submitted")
    reasons = {k: v for k, v in metrics.items() if k.startswith("bf/reason/")}
    ended = {k: v for k, v in metrics.items() if k.startswith("bf/ended/")}
    assert len(reasons) == len(ep.KEPT_REASONS) and sum(reasons.values()) == 1.0
    assert metrics["bf/reason/verifier_error"] == 1.0
    assert sum(ended.values()) == 1.0 and metrics["bf/ended/submitted"] == 1.0
    assert "bf/ended/timeout" in ended


# -- tokens -------------------------------------------------------------------------------


def test_the_token_tally_builds_the_datums_the_cookbook_trains_on():
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.data_processing import trajectory_to_data

    turns = [
        ([1, 2, 3], [4, 5]),
        ([1, 2, 3, 4, 5, 6], [7]),  # extends the previous prompt and reply
        ([1, 2, 9, 9], [8]),  # rewrites the history: a new datum
    ]
    tally = te.TokenTally()
    for ob, ac in turns:
        tally.prompt(ob)
        tally.reply(ac)
    tally.finish()
    tally.finish()  # idempotent
    assert tally.as_dict() == {
        "calls": 3,
        "prefill": 13,
        "reusable": 5 + 2,
        "sampled": 4,
        "train": 7 + 5,
        "datums": 2,
    }
    trajectory = types.Trajectory(
        transitions=[
            types.Transition(
                ob=tinker.ModelInput.from_ints(ob),
                ac=TokensWithLogprobs(tokens=ac, maybe_logprobs=[0.0] * len(ac)),
                reward=0.0,
                episode_done=i == len(turns) - 1,
            )
            for i, (ob, ac) in enumerate(turns)
        ],
        final_ob=tinker.ModelInput.empty(),
    )
    data = trajectory_to_data(trajectory, 1.0)
    assert len(data) == tally.datums
    # A datum's input drops its last token (the targets are shifted by one).
    assert sum(d.model_input.length + 1 for d in data) == tally.train


def test_an_episode_records_its_token_tally(tmp_path):
    env, runtimes = make_env(tmp_path, verify=1.0)
    results = iter(
        [
            types.StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=tinker.ModelInput.from_ints([1, 1, 7, 8, 9]),
                next_stop_condition=[],
            ),
            types.StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=[],
                metrics={"stop/max_tokens": 1.0},
            ),
        ]
    )

    async def inner_step(action, *, extra=None):
        return next(results)

    env.inner.step = inner_step

    async def go():
        await env.initial_observation()  # the fake renders [1, 1]
        await env.step([7, 8])
        return await env.step([4])

    run(go())
    record = json.loads(
        (runtimes[0].rollout_dir / "policy" / "messages.json").read_text()
    )
    assert record["tokens"] == {
        "calls": 2,
        "prefill": 2 + 5,
        "reusable": 4,
        "sampled": 3,
        "train": 6,
        "datums": 1,
    }


def test_qwen3_8_renderers_get_the_wrapper_and_their_template_arguments():
    for name in (
        "qwen3_8_xhigh_reasoning",
        "qwen3_8_medium_reasoning",
        "qwen3_8_low_reasoning",
        "qwen3_8_disable_thinking",
    ):
        assert name in te.WRAPPED_TOOL_SPEC_RENDERERS
        assert name in te.TEMPLATE_KWARGS
    assert te.TEMPLATE_KWARGS["qwen3_8_disable_thinking"] == {"enable_thinking": False}
    assert te.ENDPOINT_BODY["qwen3_8_disable_thinking"] == {"reasoning_effort": False}


# -- groups and the rollout strategy ---------------------------------------------------------


def test_group_cleanup_closes_a_cut_off_episode_and_records_it(tmp_path):
    runtimes: list[FakeRuntime] = []

    async def factory(config):
        runtimes.append(FakeRuntime(config, 1.0))
        return runtimes[-1]

    config = te.EnvConfig(
        model_name="m",
        renderer_name="qwen3_5",
        episode=ep.EpisodeSettings(jobs_dir=str(tmp_path / "trials")),
    )
    builder = te.BenchFlowEnvGroupBuilder(
        te.TaskSpec("t", tmp_path / "t", "p"),
        group_size=2,
        config=config,
        job_name="train-0003",
    )

    async def go():
        envs = [
            te.BenchFlowEnv(
                builder.task,
                config,
                FakeRenderer(),
                job_name="train-0003",
                slots=ep.SandboxSlots(2),
                runtime_factory=factory,
            )
            for _ in range(2)
        ]
        builder._envs.extend(envs)
        await envs[0].episode.start()  # started, then cut off from outside
        await builder.cleanup()
        return envs

    envs = run(go())
    assert runtimes[0].closed == 1 and len(runtimes) == 1  # the spare never started
    assert envs[0].episode.ended == "cut_off"
    lines = (tmp_path / "trials" / "rollouts.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["ended"] == "cut_off"


class ScriptedEnv(types.Env):
    """One-step episode; fails at start when told to."""

    def __init__(self, fail: BaseException | None = None, reward: float = 1.0) -> None:
        self.fail = fail
        self.reward = reward

    async def initial_observation(self):
        if self.fail is not None:
            raise self.fail
        return tinker.ModelInput.from_ints([1, 2]), []

    async def step(self, action, *, extra=None):
        return types.StepResult(
            reward=self.reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
        )


class ScriptedBuilder(types.EnvGroupBuilder):
    def __init__(
        self,
        failures: list[BaseException | None],
        group_size: int,
        rewards: list[float] | None = None,
    ) -> None:
        self.failures = failures
        self.group_size = group_size
        self.rewards = rewards or []
        self.task = SimpleNamespace(name="t")
        self.job_name = "train-0000"
        self.made = 0

    async def make_envs(self):
        envs = []
        for _ in range(self.group_size):
            n = self.made
            fail = self.failures[n] if n < len(self.failures) else None
            reward = self.rewards[n] if n < len(self.rewards) else 1.0
            envs.append(ScriptedEnv(fail, reward))
            self.made += 1
        return envs


async def policy(ob, stop, *, max_tokens=None):
    return TokensWithLogprobs(tokens=[7], maybe_logprobs=[-0.1])


def start_failure() -> ep.InfrastructureError:
    return ep.InfrastructureError(dropped("sandbox_start", "no capacity"))


def test_infrastructure_failures_are_dropped_counted_and_replaced(tmp_path):
    te.configure(max_sandboxes=4, drops_path=tmp_path / "drops.jsonl")
    builder = ScriptedBuilder(
        [start_failure(), tinker.APIConnectionError(request=None)], 4
    )
    strategy = te.DropInfrastructureFailures(max_retries=2)
    result = run(strategy.execute(builder, policy))
    assert len(result.trajectories) == 4 and len(result.errors) == 2
    assert te.DROPS.counts == {"sandbox_start": 1, "model_endpoint": 1}


def test_a_group_left_too_small_is_skipped(tmp_path):
    te.configure(max_sandboxes=4)
    failures = [start_failure()] * 4
    builder = ScriptedBuilder(failures, 3)
    with pytest.raises(AllTrajectoriesFailedError):
        run(
            te.DropInfrastructureFailures(max_retries=1, min_group_size=2).execute(
                builder, policy
            )
        )


def test_a_harness_bug_stops_the_run_instead_of_dropping(tmp_path):
    te.configure(max_sandboxes=4)
    builder = ScriptedBuilder([ValueError("bug in the harness")], 3)
    with pytest.raises(ValueError, match="bug in the harness"):
        run(te.DropInfrastructureFailures().execute(builder, policy))
    assert te.DROPS.counts == {}


def test_the_dataset_walks_shuffled_epochs(tmp_path):
    tasks = [te.TaskSpec(f"t{i}", tmp_path, "p") for i in range(5)]
    config = te.EnvConfig(model_name="m", renderer_name="r")
    data = te.BenchFlowRLDataset(
        tasks,
        groups_per_batch=2,
        group_size=4,
        config=config,
        split="train",
        n_batches=5,
        seed=3,
    )
    names = [b.task.name for i in range(len(data)) for b in data.get_batch(i)]
    assert len(names) == 10 and sorted(names[:5]) == sorted(t.name for t in tasks)
    assert {b.job_name for b in data.get_batch(4)} == {"train-0004"}
    again = te.BenchFlowRLDataset(
        tasks,
        groups_per_batch=2,
        group_size=4,
        config=config,
        split="train",
        n_batches=5,
        seed=3,
    )
    assert [b.task.name for b in again.get_batch(0)] == names[:2]


def test_openai_message_keeps_reasoning_and_tool_calls():
    message = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "plan"},
            {"type": "text", "text": "ok"},
        ],
        "tool_calls": [call("run_bash", command="ls")],
    }
    out = te.openai_message(message)
    assert out["content"] == "ok" and out["reasoning_content"] == "plan"
    assert out["tool_calls"][0]["function"] == {
        "name": "run_bash",
        "arguments": json.dumps({"command": "ls"}),
    }


def test_training_drops_and_counts_groups_without_reward_variance(tmp_path):
    te.configure(max_sandboxes=4, groups_path=tmp_path / "groups.jsonl")
    train = te.DropInfrastructureFailures(drop_constant_groups=True)
    with pytest.raises(te.NoRewardVariance) as info:
        run(train.execute(ScriptedBuilder([], 3, rewards=[1.0, 1.0, 1.0]), policy))
    assert isinstance(info.value, AllTrajectoriesFailedError)  # the cookbook skips it
    mixed = run(train.execute(ScriptedBuilder([], 3, rewards=[1.0, 0.0, 1.0]), policy))
    assert len(mixed.trajectories) == 3
    # Evaluation keeps constant groups; they are still counted.
    kept = run(te.DropInfrastructureFailures().execute(ScriptedBuilder([], 2), policy))
    assert len(kept.trajectories) == 2
    assert te.GROUPS.counts == {"all_solved": 2, "mixed": 1}
    rows = [json.loads(x) for x in (tmp_path / "groups.jsonl").read_text().splitlines()]
    assert [r["dropped"] for r in rows] == [True, False, False]


def test_a_malformed_tool_call_is_answered_and_the_episode_goes_on(tmp_path):
    from tinker_cookbook.renderers.base import UnparsedToolCall

    env, runtimes = make_env(tmp_path)

    async def go():
        await env.episode.start()
        result = await env.messages.step(
            {
                "role": "assistant",
                "content": "",
                "unparsed_tool_calls": [
                    UnparsedToolCall(raw_text="<tool_call>{bad", error="Invalid JSON")
                ],
            }
        )
        await env.episode.close()
        return result

    result = run(go())
    assert not result.episode_done and result.reward == 0.0
    assert env.messages.history[-1]["role"] == "user"
    assert "Invalid JSON" in env.messages.history[-1]["content"]
    assert runtimes[0].verified == 0


def test_a_resumed_run_sets_aside_the_steps_it_runs_again(tmp_path):
    import tinker_train

    (tmp_path / "trials").mkdir()
    (tmp_path / "checkpoints.jsonl").write_text(
        json.dumps(
            {"name": "000002", "batch": 2, "state_path": "tinker://run/weights/000002"}
        )
        + "\n"
    )
    rows = [{"step": f"train-{i:04d}", "reward": 1.0} for i in range(4)]
    (tmp_path / "trials" / "rollouts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    groups = [{"where": f"train-{i:04d}", "kind": "mixed"} for i in range(4)]
    (tmp_path / "groups.jsonl").write_text(
        "".join(json.dumps(g) + "\n" for g in groups)
    )
    drops = [{"where": "train-0001", "reason": "sandbox_start"}]
    drops.append({"where": "train-0003", "reason": "sandbox_start"})
    (tmp_path / "infrastructure_drops.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in drops)
    )
    assert tinker_train.prune_for_resume(tmp_path) == 5
    kept_drops = (tmp_path / "infrastructure_drops.jsonl").read_text().splitlines()
    assert [json.loads(x)["where"] for x in kept_drops] == ["train-0001"]
    kept = [
        json.loads(x)["step"]
        for x in (tmp_path / "trials" / "rollouts.jsonl").read_text().splitlines()
    ]
    assert kept == ["train-0000", "train-0001"]
    aside = (tmp_path / "trials" / "rollouts.discarded.jsonl").read_text().splitlines()
    assert [json.loads(x)["step"] for x in aside] == ["train-0002", "train-0003"]
