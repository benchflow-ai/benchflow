"""A BenchFlow environment for tinker-cookbook's RL framework.

Each `BenchFlowEnv` is one episode on one BenchFlow task (`tinker_episode.Episode`),
with the shared RL harness: the task prompt plus the harness message, and the
`run_bash` and `submit` tools, rendered and parsed in the model's native
tool-calling format by the cookbook's renderer. For Qwen3.5/3.6 the first
prompt is token-identical to what Tinker's OpenAI-compatible endpoint renders
for the held-out evaluator (see `chat_template_parity`).

An episode ends when the model submits, replies without a tool call, runs out
of turns, or breaks the protocol (a malformed tool call, a turn cut off at
`max_tokens`, a conversation past `max_trajectory_tokens`); the verifier then
scores the sandbox as the policy left it. Past its wall-clock budget an
episode scores 0 without a verifier run. The sandbox is closed on every path,
and `BenchFlowEnvGroupBuilder.cleanup()` closes whatever an interrupted
episode left open.

A group is `group_size` episodes of one task, so the cookbook's advantages
(reward minus the group mean) are GRPO-style. `DropInfrastructureFailures`
drops and counts the episodes the policy could not have caused to fail,
replaces them while its retry budget lasts, and lets any other exception stop
the run: a harness bug must not quietly drop the episodes it touches.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import chz
import tinker
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl import types
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, RolloutLimits
from tinker_cookbook.rl.rollout_runner import SamplingTurnTimeoutError
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy
from tinker_cookbook.tool_use import (
    AgentToolMessageEnv,
    ToolInput,
    ToolResult,
    simple_tool_result,
)
from tinker_episode import (
    HARNESS_MESSAGE,
    KEPT_REASONS,
    MAX_TURNS,
    Decision,
    DropLog,
    Episode,
    EpisodeSettings,
    GroupLog,
    InfrastructureError,
    SandboxSlots,
    solved,
)

import benchflow as bf
from benchflow.integrations.trl import bash_tool_schemas

log = logging.getLogger(__name__)

# The run_bash and submit tools exactly as TRL and the held-out evaluator see them.
RUN_BASH_SPEC, SUBMIT_SPEC = (schema["function"] for schema in bash_tool_schemas())

# Renderers that print each tool spec as they get it, whose Hugging Face chat
# template (the one Tinker's OpenAI-compatible endpoint applies) prints the
# OpenAI wrapper {"type": "function", "function": ...}: they get the wrapper.
WRAPPED_TOOL_SPEC_RENDERERS = frozenset({"qwen3_5", "qwen3_5_disable_thinking"})

# What the model sees after a tool call that does not parse ({details}: why).
PARSE_ERROR_MESSAGE = '{{"error": "the tool call could not be parsed: {details}"}}'

# How an episode ended, from the cookbook's stop reason.
ENDED = {
    types.StopReason.TOOL_STOPPED: "submitted",
    types.StopReason.COMPLETED: "no_tool_call",
    types.StopReason.MAX_TURNS: "turn_limit",
    types.StopReason.MAX_TOOL_CALLS: "turn_limit",
    types.StopReason.PARSE_ERROR: "parse_error",
    types.StopReason.MAX_TOKENS: "max_tokens",
    types.StopReason.CONTEXT_OVERFLOW: "context_overflow",
    types.StopReason.ROLLOUT_TIMEOUT: "timeout",
}


# -- configuration -------------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    """Everything an episode needs besides its task. Picklable."""

    model_name: str
    renderer_name: str
    episode: EpisodeSettings = field(default_factory=EpisodeSettings)
    max_turns: int = (
        MAX_TURNS  # tool-calling turns; the verifier runs when they run out
    )
    max_tokens: int = 4096  # sampled tokens per turn
    max_trajectory_tokens: int = 32768  # prompt plus generation, per episode
    sampling_timeout_sec: float = (
        300.0  # one sampling call; past it: dropped (endpoint)
    )
    prompt_suffix: str = HARNESS_MESSAGE  # appended to the task prompt


@dataclass(frozen=True)
class TaskSpec:
    """A BenchFlow task: its folder and the prompt the model sees."""

    name: str
    task_dir: Path
    instruction: str


def load_tasks(
    tasks_dir: Path | str,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> list[TaskSpec]:
    """Tasks under `tasks_dir` (or the task at `tasks_dir`), sorted by name."""
    root = Path(tasks_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    if _is_task_dir(root):
        dirs = [root]
    else:
        dirs = sorted(d for d in root.iterdir() if d.is_dir() and _is_task_dir(d))
    wanted, unwanted = set(include), set(exclude)
    specs = [
        TaskSpec(name=d.name, task_dir=d, instruction=bf.Task(d).instruction)
        for d in dirs
        if (not wanted or d.name in wanted) and d.name not in unwanted
    ]
    missing = wanted - {s.name for s in specs}
    if missing:
        raise ValueError(f"tasks not found under {root}: {', '.join(sorted(missing))}")
    if not specs:
        raise ValueError(f"no BenchFlow tasks under {root}")
    return specs


def _is_task_dir(path: Path) -> bool:
    return (path / "task.md").is_file() or (path / "task.toml").is_file()


# -- process-wide state ----------------------------------------------------------

_SLOTS = SandboxSlots(16)
DROPS = DropLog()
GROUPS = GroupLog()


def configure(
    *,
    max_sandboxes: int,
    drops_path: Path | None = None,
    groups_path: Path | None = None,
) -> None:
    """Set this process's sandbox cap and where drops and groups are logged."""
    global _SLOTS
    _SLOTS = SandboxSlots(max_sandboxes)
    DROPS.path = drops_path
    DROPS.counts.clear()
    DROPS.consecutive = 0
    GROUPS.path = groups_path
    GROUPS.counts.clear()


def sandbox_slots() -> SandboxSlots:
    return _SLOTS


@functools.lru_cache(maxsize=8)
def cached_renderer(model_name: str, renderer_name: str) -> Renderer:
    return get_renderer(renderer_name, tokenizer_utils.get_tokenizer(model_name))


def tool_specs(renderer_name: str) -> list[dict[str, Any]]:
    specs = [RUN_BASH_SPEC, SUBMIT_SPEC]
    if renderer_name in WRAPPED_TOOL_SPEC_RENDERERS:
        return [{"type": "function", "function": spec} for spec in specs]
    return specs


def initial_messages(
    task: TaskSpec, config: EnvConfig, renderer: Renderer
) -> list[Message]:
    """The tools prefix (no system prompt) and the task prompt plus the harness message."""
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_specs(config.renderer_name),  # type: ignore[arg-type]
        system_prompt="",
    )
    return [
        *prefix,
        {"role": "user", "content": task.instruction + config.prompt_suffix},
    ]


def chat_template_parity(task: TaskSpec, config: EnvConfig) -> str:
    """Compare the first prompt with the model's Hugging Face chat template.

    Tinker's OpenAI-compatible endpoint (which the held-out evaluator calls)
    renders with that template; 'identical' means training and evaluation
    start from the same tokens.
    """
    renderer = cached_renderer(config.model_name, config.renderer_name)
    tokenizer = tokenizer_utils.get_tokenizer(config.model_name)
    ours = [
        t
        for chunk in renderer.build_generation_prompt(
            initial_messages(task, config, renderer)
        ).chunks
        for t in getattr(chunk, "tokens", [])
    ]
    try:
        theirs = tokenizer.apply_chat_template(
            [{"role": "user", "content": task.instruction + config.prompt_suffix}],
            tools=[
                {"type": "function", "function": s}
                for s in (RUN_BASH_SPEC, SUBMIT_SPEC)
            ],
            add_generation_prompt=True,
            tokenize=True,
        )
    except Exception as exc:  # no chat template, or one without tools
        return f"not checked ({type(exc).__name__}: {exc})"
    if hasattr(theirs, "keys"):
        theirs = theirs["input_ids"]
    theirs = list(theirs)
    if ours == theirs:
        return f"identical ({len(ours)} tokens)"
    at = next(
        (i for i, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b), None
    )
    at = min(len(ours), len(theirs)) if at is None else at
    return f"differs at token {at} ({len(ours)} vs {len(theirs)} tokens)"


# -- tools -------------------------------------------------------------------------


def _argument(input: ToolInput, name: str) -> str:
    # The shared harness reads arguments[name]; a missing one is a KeyError
    # whose text becomes {"error": "'name'"}.
    return str(input.arguments[name])


class RunBash:
    """The `run_bash` tool (tinker-cookbook's Tool protocol)."""

    name = "run_bash"
    description: ClassVar[str] = RUN_BASH_SPEC["description"]
    parameters_schema: ClassVar[dict[str, Any]] = RUN_BASH_SPEC["parameters"]

    def __init__(self, episode: Episode) -> None:
        self.episode = episode

    def to_spec(self) -> dict[str, Any]:
        return RUN_BASH_SPEC

    async def run(self, input: ToolInput) -> ToolResult:
        if self.episode.submitted:
            # The evaluator runs nothing after submit; neither does training.
            text = "not run: the task was already submitted"
        else:
            try:
                command = _argument(input, "command")
            except KeyError as exc:
                text = json.dumps({"error": str(exc)})
            else:
                text = await self.episode.run_bash(command)
        return simple_tool_result(text, call_id=input.call_id or "", name=self.name)


class Submit:
    """The `submit` tool: writes the answer file and ends the episode."""

    name = "submit"
    description: ClassVar[str] = SUBMIT_SPEC["description"]
    parameters_schema: ClassVar[dict[str, Any]] = SUBMIT_SPEC["parameters"]

    def __init__(self, episode: Episode) -> None:
        self.episode = episode

    def to_spec(self) -> dict[str, Any]:
        return SUBMIT_SPEC

    async def run(self, input: ToolInput) -> ToolResult:
        call_id = input.call_id or ""
        if self.episode.submitted:
            return simple_tool_result(
                "submission recorded", call_id=call_id, name=self.name, should_stop=True
            )
        try:
            answer = _argument(input, "answer")
        except KeyError as exc:
            return simple_tool_result(
                json.dumps({"error": str(exc)}), call_id=call_id, name=self.name
            )
        error = await self.episode.submit(answer)
        if error is not None:  # the answer was not written: the episode goes on
            return simple_tool_result(error, call_id=call_id, name=self.name)
        return simple_tool_result(
            "submission recorded", call_id=call_id, name=self.name, should_stop=True
        )


# -- the environment ---------------------------------------------------------------


class BenchFlowEnv(types.Env):
    """One episode on one BenchFlow task, in tinker-cookbook's Env protocol."""

    def __init__(
        self,
        task: TaskSpec,
        config: EnvConfig,
        renderer: Renderer,
        *,
        job_name: str,
        slots: SandboxSlots,
        runtime_factory: Any = None,
    ) -> None:
        self.task = task
        self.config = config
        self.episode = Episode(
            task.task_dir,
            config.episode,
            slots=slots,
            job_name=job_name,
            runtime_factory=runtime_factory,
        )
        tools = [RunBash(self.episode), Submit(self.episode)]
        # A tool call that does not parse is answered with an error and the
        # episode goes on, within its turns, as in the evaluator (which reports
        # unparsable arguments the same way). Broken framing still ends it.
        parse_errors = ParseErrorPolicy(
            max_consecutive=config.max_turns,
            retry_message_template=PARSE_ERROR_MESSAGE,
        )
        self.messages = AgentToolMessageEnv(
            tools=tools,
            initial_messages=initial_messages(task, config, renderer),
            max_turns=config.max_turns,
            reward_fn=self._grade,
            failed_parse_reward=0.0,
            parse_error_policy=parse_errors,
            tool_execution="sequential",
        )
        self.inner = EnvFromMessageEnv(
            renderer=renderer,
            message_env=self.messages,
            failed_parse_reward=0.0,
            max_trajectory_tokens=config.max_trajectory_tokens,
            max_generation_tokens=config.max_tokens,
            context_overflow_reward=0.0,
            parse_error_policy=parse_errors,
        )
        # Read by the cookbook's rollout runner: a sampling call that hangs
        # raises, and DropInfrastructureFailures drops the episode.
        self.rollout_limits = RolloutLimits(
            sampling_turn_timeout_seconds=config.sampling_timeout_sec
        )
        self._recorded = False

    async def initial_observation(
        self,
    ) -> (
        tuple[tinker.ModelInput, types.StopCondition] | types.InitialObservationOverflow
    ):
        first = await self.inner.initial_observation()
        if isinstance(first, types.InitialObservationOverflow):
            # A configuration error, not something to train on.
            raise ValueError(
                f"{self.task.name}: the prompt does not fit max_trajectory_tokens="
                f"{self.config.max_trajectory_tokens} with max_tokens={self.config.max_tokens}"
            )
        await self.episode.start()  # InfrastructureError: dropped by the strategy
        return first

    async def step(
        self, action: types.Action, *, extra: types.ActionExtra | None = None
    ) -> types.StepResult:
        result = await self.inner.step(action, extra=extra)
        if result.episode_done:
            reason = stop_reason_of(result.metrics)
            self.episode.ended = ENDED.get(reason or "", reason or "ended")
            if self.episode.decision is None:
                # Ended before grading: a malformed tool call, a turn cut off at
                # max_tokens, a context overflow. Score what the policy left.
                await self._decide()
            result.reward = self._reward()
            result.metrics.update(
                decision_metrics(self.episode.decision, self.episode.ended)
            )
            self._record()
            return result
        if self.episode.past_deadline():
            await self.episode.time_out()
            self.episode.ended = "timeout"
            self._record()
            return types.StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=result.next_stop_condition,
                metrics={
                    **result.metrics,
                    f"{types.STOP_METRIC_PREFIX}{types.StopReason.ROLLOUT_TIMEOUT}": 1.0,
                    **decision_metrics(self.episode.decision, "timeout"),
                },
                logs=result.logs,
            )
        return result

    async def _grade(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """The cookbook's reward_fn: on submit, no tool call, or the turn limit."""
        await self._decide()
        return self._reward(), {}

    async def _decide(self) -> Decision:
        decision = await self.episode.finish()
        if decision.dropped:
            if self.episode.ended is None:
                self.episode.ended = "submitted" if self.episode.submitted else "graded"
            self._record()
            raise InfrastructureError(decision, task=self.task.name)
        return decision

    def _reward(self) -> float:
        decision = self.episode.decision
        return (
            decision.reward
            if decision is not None and decision.reward is not None
            else 0.0
        )

    async def close(self) -> None:
        """Close the sandbox if the rollout was cut off before the episode ended."""
        if not self.episode.closed:
            self.episode.ended = self.episode.ended or "cut_off"
            await self.episode.close()
        self._record()

    def _record(self) -> None:
        if self._recorded or not self.episode.started:
            return
        self._recorded = True
        self.episode.write_record(
            self.episode.record(
                [openai_message(m) for m in self.messages.history],
                model=self.config.model_name,
                renderer=self.config.renderer_name,
            )
        )


def stop_reason_of(metrics: dict[str, Any]) -> str | None:
    prefix = types.STOP_METRIC_PREFIX
    return next((k.removeprefix(prefix) for k in metrics if k.startswith(prefix)), None)


def decision_metrics(decision: Decision | None, ended: str | None) -> dict[str, float]:
    """Per-episode numbers the cookbook averages into env/all/... metrics.

    Every key is present (0 or 1): the cookbook averages a key only over the
    episodes that report it, so a one-hot key alone would always read 1.
    """
    if decision is None:
        return {}
    metrics = {
        "bf/solved": float(solved(decision)),
        "bf/scored": float(decision.reason == KEPT_REASONS[0]),
    }
    for reason in KEPT_REASONS:
        metrics[f"bf/reason/{reason}"] = float(decision.reason == reason)
    for how in dict.fromkeys(ENDED.values()):
        metrics[f"bf/ended/{how}"] = float(ended == how)
    return metrics


def openai_message(message: Message) -> dict[str, Any]:
    """A renderer message as an OpenAI-style chat message, for the audit record."""
    out: dict[str, Any] = {"role": message["role"]}
    content = message.get("content")
    if isinstance(content, str):
        out["content"] = content
    else:
        parts = content or []
        out["content"] = "".join(
            p.get("text", "") for p in parts if p.get("type") == "text"
        )
        thinking = "".join(
            p.get("thinking", "") for p in parts if p.get("type") == "thinking"
        )
        if thinking:
            out["reasoning_content"] = thinking
    calls = message.get("tool_calls") or []
    if calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in calls
        ]
    for key in ("tool_call_id", "name"):
        if message.get(key):
            out[key] = message[key]
    if message.get("unparsed_tool_calls"):
        out["unparsed_tool_calls"] = [str(c) for c in message["unparsed_tool_calls"]]
    return out


# -- groups, datasets ----------------------------------------------------------------


class BenchFlowEnvGroupBuilder(types.EnvGroupBuilder):
    """`group_size` episodes of one task; their rewards are centered together."""

    def __init__(
        self, task: TaskSpec, *, group_size: int, config: EnvConfig, job_name: str
    ) -> None:
        self.task = task
        self.group_size = group_size
        self.config = config
        self.job_name = job_name
        self._envs: list[BenchFlowEnv] = []

    async def make_envs(self) -> Sequence[types.Env]:
        # Sandboxes start lazily in initial_observation, so a strategy that
        # calls make_envs() again for one replacement starts one sandbox.
        renderer = cached_renderer(self.config.model_name, self.config.renderer_name)
        envs = [
            BenchFlowEnv(
                self.task,
                self.config,
                renderer,
                job_name=self.job_name,
                slots=sandbox_slots(),
            )
            for _ in range(self.group_size)
        ]
        self._envs.extend(envs)
        return envs

    @property
    def envs(self) -> list[BenchFlowEnv]:
        return list(self._envs)

    async def cleanup(self) -> None:
        results = await asyncio.gather(
            *(env.close() for env in self._envs), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                log.warning("cleanup of %s: %s", self.task.name, result)

    def logging_tags(self) -> list[str]:
        return ["benchflow"]


class BenchFlowRLDataset(types.RLDataset):
    """Batches of `groups_per_batch` task groups, drawn epoch by shuffled epoch."""

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        *,
        groups_per_batch: int,
        group_size: int,
        config: EnvConfig,
        split: str,
        n_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        if not tasks:
            raise ValueError("no tasks")
        self.groups_per_batch = groups_per_batch
        self.group_size = group_size
        self.config = config
        self.split = split
        per_epoch = max(len(tasks) // groups_per_batch, 1)
        self.n_batches = n_batches if n_batches is not None else per_epoch
        rng = random.Random(seed)
        order: list[TaskSpec] = []
        while len(order) < self.n_batches * groups_per_batch:
            epoch = list(tasks)
            rng.shuffle(epoch)
            order.extend(epoch)
        self.order = order[: self.n_batches * groups_per_batch]

    def get_batch(self, index: int) -> Sequence[types.EnvGroupBuilder]:
        start = index * self.groups_per_batch
        return [
            BenchFlowEnvGroupBuilder(
                task,
                group_size=self.group_size,
                config=self.config,
                job_name=f"{self.split}-{index:04d}",
            )
            for task in self.order[start : start + self.groups_per_batch]
        ]

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class BenchFlowDatasetBuilder(types.RLDatasetBuilder):
    """The train split for tinker-cookbook's `rl.train.Config.dataset_builder`."""

    train_tasks: list[TaskSpec]
    config: EnvConfig
    groups_per_batch: int
    group_size: int
    n_batches: int
    seed: int = 0

    async def __call__(self) -> tuple[types.RLDataset, types.RLDataset | None]:
        train = BenchFlowRLDataset(
            self.train_tasks,
            groups_per_batch=self.groups_per_batch,
            group_size=self.group_size,
            config=self.config,
            split="train",
            n_batches=self.n_batches,
            seed=self.seed,
        )
        return train, None


# -- rollout strategy ------------------------------------------------------------------

# Sampler failures the policy cannot cause (the "model endpoint" drop). Auth,
# billing and bad-request errors are not here: a retry cannot fix them, so
# they stop the run.
_SAMPLER_INFRASTRUCTURE: tuple[type[BaseException], ...] = (
    SamplingTurnTimeoutError,
    tinker.APIConnectionError,
    tinker.InternalServerError,
    tinker.RateLimitError,
    tinker.RequestFailedError,
)


def is_infrastructure(exc: BaseException) -> bool:
    return isinstance(exc, (InfrastructureError, *_SAMPLER_INFRASTRUCTURE))


@dataclass(frozen=True)
class DropInfrastructureFailures(RolloutStrategy):
    """Run a group; drop, count and replace only infrastructure failures.

    Any other exception cancels the group and propagates, which stops
    training (`catches_group_errors` is False): dropping episodes on a harness
    bug could select them by what the policy did. A group left with fewer
    than `min_group_size` episodes is skipped (AllTrajectoriesFailedError).
    With `drop_constant_groups` (training), a group whose episodes all got the
    same reward is dropped too (NoRewardVariance); every group is counted in
    GROUPS by kind either way.
    """

    max_retries: int = 2
    min_group_size: int = 2
    drop_constant_groups: bool = False

    @property
    def catches_group_errors(self) -> bool:
        return False

    async def execute(
        self, env_group_builder: types.EnvGroupBuilder, policy: TokenCompleter
    ) -> RolloutResult:
        from tinker_cookbook.rl.rollouts import do_single_rollout

        envs = list(await env_group_builder.make_envs())
        floor = min(self.min_group_size, len(envs))
        task_name = getattr(getattr(env_group_builder, "task", None), "name", "?")
        where = getattr(env_group_builder, "job_name", "")
        running = {
            asyncio.create_task(do_single_rollout(policy, env)): env for env in envs
        }
        pending = set(running)
        trajectories: list[types.Trajectory] = []
        survivors: list[types.Env] = []
        errors: list[types.RolloutError] = []
        retries = self.max_retries
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if task.cancelled():
                        raise asyncio.CancelledError()
                    exc = task.exception()
                    if exc is None:
                        DROPS.completed()
                        trajectories.append(task.result())
                        survivors.append(running[task])
                        continue
                    if not is_infrastructure(exc):
                        raise exc
                    # Raises TooManyInfrastructureFailures past the streak limit.
                    DROPS.add(exc, task=task_name, where=where)
                    errors.append(types.RolloutError(type(exc).__name__, str(exc)))
                    if retries > 0:
                        retries -= 1
                        env = (await env_group_builder.make_envs())[0]
                        replacement = asyncio.create_task(
                            do_single_rollout(policy, env)
                        )
                        running[replacement] = env
                        pending.add(replacement)
        except BaseException:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in running:  # mark every outcome retrieved
                if task.done() and not task.cancelled():
                    task.exception()
            raise
        if len(trajectories) < floor:
            raise AllTrajectoriesFailedError(
                f"{task_name}: {len(trajectories)} of {len(envs)} episodes left after "
                f"{len(errors)} infrastructure failures; the group is skipped"
            )
        rewards = [sum(t.reward for t in traj.transitions) for traj in trajectories]
        drop = self.drop_constant_groups and len(set(rewards)) == 1
        kind = GROUPS.add(rewards, task=task_name, where=where, dropped=drop)
        if drop:
            # Every advantage would be 0. The cookbook skips a group whose
            # strategy raises AllTrajectoriesFailedError.
            raise NoRewardVariance(
                f"{task_name}: all {len(rewards)} episodes scored {rewards[0]:g} "
                f"({kind}); the group is dropped"
            )
        return RolloutResult(trajectories=trajectories, envs=survivors, errors=errors)


class NoRewardVariance(AllTrajectoriesFailedError):
    """A group whose episodes all got the same reward: nothing to learn from it."""
