"""A BenchFlow environment for tinker-cookbook's RL framework.

Each `BenchFlowEnv` is one episode on one BenchFlow task: `initial_observation`
starts the task sandbox (`tinker_episode.Episode`), the model acts through two
tools in its native tool-calling format (rendered and parsed by the cookbook's
renderer), and the episode ends when the model submits, stops calling tools,
runs out of turns, breaks the protocol, or runs past its wall-clock budget.
The verifier runs on the first three; the others score 0 under the training
rule (see tinker_episode.py). The sandbox is closed on every path, and
`BenchFlowEnvGroupBuilder.cleanup()` closes whatever an interrupted episode
left open.

A group is `group_size` episodes of one task, so the cookbook's advantages
(reward minus the group mean) are GRPO-style. `DropInfrastructureFailures`
drops and counts the episodes the policy could not have caused to fail,
replaces them while its retry budget lasts, and lets any other exception
stop the run: a harness bug must not quietly drop the episodes it touches.
"""

from __future__ import annotations

import asyncio
import functools
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
from tinker_cookbook.renderers.base import Message, Renderer, ToolSpec
from tinker_cookbook.rl import types
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, RolloutLimits
from tinker_cookbook.rl.rollout_runner import SamplingTurnTimeoutError
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy
from tinker_cookbook.tool_use import (
    AgentToolMessageEnv,
    ToolInput,
    ToolResult,
    error_tool_result,
    simple_tool_result,
)
from tinker_episode import (
    EXITS,
    INFRASTRUCTURE,
    POLICY_FAILURE,
    SCORED,
    DropLog,
    Episode,
    EpisodeSettings,
    InfrastructureError,
    Outcome,
    SandboxSlots,
    VerifierCrashOnCleanRun,
)

import benchflow as bf

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an agent working in a Linux sandbox on the task the user gives you. "
    "Use the run_bash tool to run shell commands: each call starts a fresh bash "
    "shell in the task's working directory, so chain related commands with && "
    "or write scripts to files. When the task is complete, call the submit tool "
    "once; the task's checks then run on the sandbox as you left it."
)

RUN_BASH_DESCRIPTION = (
    "Run a bash command in the task sandbox and return its exit code, stdout and "
    "stderr. Each call starts a fresh shell in the task's working directory; "
    "commands time out after {timeout} seconds and long output is cut."
)
SUBMIT_DESCRIPTION = (
    "Submit your work: the task's checks run on the sandbox as it is now and the "
    "episode ends. Call it once, when the task is complete."
)


# -- configuration -----------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    """Everything an episode needs besides its task. Picklable."""

    model_name: str
    renderer_name: str
    episode: EpisodeSettings = field(default_factory=EpisodeSettings)
    max_turns: int = 12  # model turns; the verifier runs when they run out
    max_tokens: int = 2048  # sampled tokens per turn
    max_trajectory_tokens: int = 32768  # prompt plus generation, per episode
    parse_retries: int = 2  # malformed tool calls answered with a correction
    sampling_timeout_sec: float = 300.0  # one sampling call; beyond it: infrastructure
    system_prompt: str = SYSTEM_PROMPT


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


# -- process-wide state ------------------------------------------------------

_SLOTS = SandboxSlots(16)
DROPS = DropLog()


def configure(*, max_sandboxes: int, drops_path: Path | None = None) -> None:
    """Set this process's sandbox cap and where infrastructure drops are logged."""
    global _SLOTS
    _SLOTS = SandboxSlots(max_sandboxes)
    DROPS.path = drops_path
    DROPS.counts.clear()
    DROPS.consecutive = 0


def sandbox_slots() -> SandboxSlots:
    return _SLOTS


@functools.lru_cache(maxsize=8)
def cached_renderer(model_name: str, renderer_name: str) -> Renderer:
    return get_renderer(renderer_name, tokenizer_utils.get_tokenizer(model_name))


# -- tools -------------------------------------------------------------------


class RunBash:
    """The `run_bash` tool (tinker-cookbook's Tool protocol)."""

    name = "run_bash"

    def __init__(self, episode: Episode) -> None:
        self.episode = episode
        self.description = RUN_BASH_DESCRIPTION.format(
            timeout=episode.settings.command_timeout_sec
        )
        self.parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."}
            },
            "required": ["command"],
        }

    def to_spec(self) -> ToolSpec:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, input: ToolInput) -> ToolResult:
        call_id = input.call_id or ""
        command = input.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return error_tool_result(
                "run_bash needs a non-empty string argument `command`",
                call_id=call_id,
                name=self.name,
                error_type="validation_failed",
            )
        text = await self.episode.run_bash(command)
        # A lost sandbox ends the episode (it scores 0: see tinker_episode.py).
        return simple_tool_result(
            text, call_id=call_id, name=self.name, should_stop=self.episode.lost
        )


class Submit:
    """The `submit` tool: ends the episode; the verifier runs next."""

    name = "submit"
    description = SUBMIT_DESCRIPTION
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, episode: Episode) -> None:
        self.episode = episode

    def to_spec(self) -> ToolSpec:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, input: ToolInput) -> ToolResult:
        first = not self.episode.submitted
        self.episode.submitted = True
        text = (
            "Submitted; the task's checks run now." if first else "Already submitted."
        )
        return simple_tool_result(
            text, call_id=input.call_id or "", name=self.name, should_stop=True
        )


# -- the environment ---------------------------------------------------------


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
        prefix = renderer.create_conversation_prefix_with_tools(
            tools=[t.to_spec() for t in tools], system_prompt=config.system_prompt
        )
        # Parse errors cost nothing extra: the training rule scores them 0.
        parse_policy = ParseErrorPolicy(max_consecutive=config.parse_retries)
        self.messages = AgentToolMessageEnv(
            tools=tools,
            initial_messages=[*prefix, {"role": "user", "content": task.instruction}],
            max_turns=config.max_turns,
            reward_fn=self._grade,
            failed_parse_reward=0.0,
            parse_error_policy=parse_policy,
            tool_execution="sequential",
        )
        self.inner = EnvFromMessageEnv(
            renderer=renderer,
            message_env=self.messages,
            failed_parse_reward=0.0,
            max_trajectory_tokens=config.max_trajectory_tokens,
            max_generation_tokens=config.max_tokens,
            context_overflow_reward=0.0,
            parse_error_policy=parse_policy,
        )
        # Read by the cookbook's rollout runner: a hung sampling call raises,
        # and DropInfrastructureFailures drops that episode.
        self.rollout_limits = RolloutLimits(
            sampling_turn_timeout_seconds=config.sampling_timeout_sec
        )
        self.stop_reason: str | None = None
        self._recorded = False

    async def initial_observation(
        self,
    ) -> (
        tuple[tinker.ModelInput, types.StopCondition] | types.InitialObservationOverflow
    ):
        first = await self.inner.initial_observation()
        if isinstance(first, types.InitialObservationOverflow):
            # The prompt alone overflows the budget: no sandbox is started.
            await self.episode.abandon("prompt_too_long", str(first.logs))
            self.stop_reason = types.StopReason.MAX_TOKENS
            self._record()
            return types.InitialObservationOverflow(
                reward=0.0,
                metrics={**first.metrics, **outcome_metrics(self.episode.outcome)},
                logs=first.logs,
            )
        await self.episode.start()  # SandboxStartError: dropped by the strategy
        return first

    async def step(
        self, action: types.Action, *, extra: types.ActionExtra | None = None
    ) -> types.StepResult:
        result = await self.inner.step(action, extra=extra)
        if result.episode_done:
            reason = stop_reason_of(result.metrics)
            if self.episode.outcome is None:
                # Ended before any grading: a parse error, a turn cut at
                # max_tokens, or a context overflow. The policy did it: 0.
                await self.episode.abandon(reason or "ended")
            self.stop_reason = reason
            result.reward = _reward_of(self.episode.outcome)
            result.metrics.update(outcome_metrics(self.episode.outcome))
            self._record()
            return result
        if self.episode.past_deadline():
            timeout = self.config.episode.episode_timeout_sec
            await self.episode.abandon("timeout", f"still running after {timeout:g} s")
            self.stop_reason = types.StopReason.ROLLOUT_TIMEOUT
            self._record()
            return types.StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=result.next_stop_condition,
                metrics={
                    **result.metrics,
                    f"{types.STOP_METRIC_PREFIX}{types.StopReason.ROLLOUT_TIMEOUT}": 1.0,
                    **outcome_metrics(self.episode.outcome),
                },
                logs=result.logs,
            )
        return result

    async def _grade(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        """The cookbook's reward_fn: submit, no tool call, or out of turns."""
        outcome = await self.episode.finish()
        if outcome.status == INFRASTRUCTURE:
            self._record()
            raise VerifierCrashOnCleanRun(outcome.detail, task=self.task.name)
        return _reward_of(outcome), {}

    async def close(self) -> None:
        """Close the sandbox if the episode was cut off before it ended."""
        if not self.episode.closed:
            await self.episode.abandon(
                "cut_off", "the rollout ended from outside the episode"
            )
        self._record()

    def _record(self) -> None:
        if self._recorded or self.episode.rollout_dir is None:
            return
        self._recorded = True
        self.episode.write_record(
            self.episode.record(
                stop_reason=self.stop_reason,
                model=self.config.model_name,
                renderer=self.config.renderer_name,
                messages=[_jsonable_message(m) for m in self.messages.history],
            )
        )


def _reward_of(outcome: Outcome | None) -> float:
    if outcome is None or outcome.reward is None:
        return 0.0
    return outcome.reward


def stop_reason_of(metrics: dict[str, Any]) -> str | None:
    prefix = types.STOP_METRIC_PREFIX
    return next((k.removeprefix(prefix) for k in metrics if k.startswith(prefix)), None)


def outcome_metrics(outcome: Outcome | None) -> dict[str, float]:
    """Per-episode numbers the cookbook averages into env/all/... metrics.

    Every exit key is present (0 or 1): the cookbook averages a key only over
    the episodes that report it, so a one-hot key alone would always read 1.
    """
    if outcome is None:
        return {}
    metrics = {
        "bf/solved": float(outcome.solved),
        "bf/scored": float(outcome.status == SCORED),
        "bf/policy_failure": float(outcome.status == POLICY_FAILURE),
    }
    for exit in (*EXITS, outcome.exit):
        metrics[f"bf/exit/{exit}"] = float(exit == outcome.exit)
    return metrics


def _jsonable_message(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in message.items():
        if key == "tool_calls" and value:
            out[key] = [
                call.model_dump() if hasattr(call, "model_dump") else str(call)
                for call in value
            ]
        elif isinstance(value, str | int | float | bool) or value is None:
            out[key] = value
        else:
            out[key] = value if isinstance(value, list | dict) else str(value)
    return out


# -- groups, datasets --------------------------------------------------------


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


# -- rollout strategy --------------------------------------------------------

# Sampler failures the policy cannot cause. Auth, billing and bad-request
# errors are not here: retrying cannot fix them, so they stop the run.
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
    """

    max_retries: int = 2
    min_group_size: int = 2

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
                    DROPS.add(
                        exc,
                        task=task_name,
                        where=getattr(env_group_builder, "job_name", ""),
                    )
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
        return RolloutResult(trajectories=trajectories, envs=survivors, errors=errors)
