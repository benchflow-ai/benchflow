"""User-loop contracts for progressive-disclosure rollouts."""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast


@dataclass
class RoundResult:
    """Outcome of one agent round, passed to ``BaseUser.run()``."""

    round: int
    trajectory: list[dict] = field(default_factory=list)
    rewards: dict[str, Any] | None = None
    verifier_output: str | None = None
    verifier_error: str | None = None
    n_tool_calls: int = 0
    scene: str | None = None
    role: str | None = None
    handoff_from: str | None = None
    handoff_to: str | None = None


class BaseUser:
    """Abstract user that drives a progressive-disclosure rollout loop."""

    async def setup(self, instruction: str, solution: str | None = None) -> None:
        """Called once before the first round."""

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        """Produce the next prompt for the agent, or ``None`` to stop."""
        raise NotImplementedError


class PassthroughUser(BaseUser):
    """Sends the original instruction unchanged for one round."""

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        if round == 0:
            return instruction
        return None


class FunctionUser(BaseUser):
    """Wrap a sync or async function as a ``BaseUser``."""

    def __init__(
        self,
        fn: Callable[
            [int, str, RoundResult | None],
            str | None | Awaitable[str | None],
        ],
    ) -> None:
        self._fn = fn

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        result = self._fn(round, instruction, round_result)
        if inspect.isawaitable(result):
            return cast(str | None, await result)
        return cast(str | None, result)


class DocumentNudgeUser(BaseUser):
    """Deterministic user compiled from a task document.

    The first round sends the public task instruction. Private facts are revealed
    only after the previous agent trajectory explicitly asks for a matching fact
    key, so package metadata can carry hidden user context without leaking it into
    the solver's initial prompt.
    """

    def __init__(
        self,
        *,
        persona: str | None = None,
        private_facts: dict[str, str] | None = None,
        branchable: bool = False,
        branch_execution: str = "none",
        confirmation_policy: str | None = None,
        handoff_kind: str = "none",
        handoff_team: str | None = None,
    ) -> None:
        self.persona = persona
        self.private_facts = dict(private_facts or {})
        self.branchable = branchable
        self.branch_execution = branch_execution
        self.confirmation_policy = confirmation_policy
        self.handoff_kind = handoff_kind
        self.handoff_team = handoff_team
        self._revealed: set[str] = set()

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        if round == 0:
            return instruction
        if round_result is None or not self.private_facts:
            return None

        previous_text = _trajectory_text(round_result.trajectory)
        for key, value in self.private_facts.items():
            if key in self._revealed:
                continue
            if _asks_for_fact(previous_text, key, value):
                self._revealed.add(key)
                label = key.replace("_", " ")
                return f"Additional user detail for {label}: {value}"
        return None


UserModelCaller = Callable[[str, str], Awaitable[str] | str]


class ModelDocumentNudgeUser(BaseUser):
    """Model-backed user compiled from a task document.

    The solver only sees prompts returned from ``run()``. Hidden private facts
    stay inside the model-user prompt until that simulated user chooses to reveal
    them, which keeps the task's initial prompt and package metadata redacted
    while still making richer NudgeBench-style users executable.
    """

    def __init__(
        self,
        *,
        model: str,
        persona: str | None = None,
        private_facts: dict[str, str] | None = None,
        branchable: bool = False,
        branch_execution: str = "none",
        confirmation_policy: str | None = None,
        handoff_kind: str = "none",
        handoff_team: str | None = None,
        call_model: UserModelCaller | None = None,
    ) -> None:
        self.model = model
        self.persona = persona
        self.private_facts = dict(private_facts or {})
        self.branchable = branchable
        self.branch_execution = branch_execution
        self.confirmation_policy = confirmation_policy
        self.handoff_kind = handoff_kind
        self.handoff_team = handoff_team
        self._call_model = call_model or _default_user_model_call

    async def run(
        self,
        round: int,
        instruction: str,
        round_result: RoundResult | None = None,
    ) -> str | None:
        if round == 0:
            return instruction
        if round_result is None:
            return None
        prompt = _model_user_prompt(
            persona=self.persona,
            private_facts=self.private_facts,
            instruction=instruction,
            round_result=round_result,
        )
        response = self._call_model(self.model, prompt)
        if inspect.isawaitable(response):
            response = await response
        text = str(response).strip()
        if not text or text.lower() in {"none", "null", "stop", "done"}:
            return None
        return text


def _trajectory_text(events: list[dict]) -> str:
    chunks: list[str] = []
    for event in events:
        _collect_strings(event, chunks)
    return "\n".join(chunks).lower()


def _collect_strings(value: Any, chunks: list[str]) -> None:
    if isinstance(value, str):
        chunks.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, chunks)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, chunks)


def _asks_for_fact(text: str, key: str, value: str) -> bool:
    if not text:
        return False
    asks = "?" in text or any(
        token in text for token in ("clarify", "need", "tell me", "what", "which")
    )
    if not asks:
        return False

    key_words = _signal_words(key.replace("_", " "))
    if key_words and all(word in text for word in key_words):
        return True

    value_words = _signal_words(value)
    return any(word in text for word in value_words)


def _signal_words(value: str) -> list[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "use",
        "need",
        "this",
        "that",
        "from",
        "only",
        "after",
    }
    return [
        word
        for word in _split_words(value.lower())
        if len(word) > 2 and word not in stopwords
    ]


def _split_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value)


async def _default_user_model_call(model: str, prompt: str) -> str:
    from benchflow.rewards.llm import call_judge

    return await call_judge(model, prompt, max_tokens=512)


def _model_user_prompt(
    *,
    persona: str | None,
    private_facts: dict[str, str],
    instruction: str,
    round_result: RoundResult,
) -> str:
    rewards = round_result.rewards or {}
    hidden = "\n".join(
        f"- {key}: {value}" for key, value in sorted(private_facts.items())
    )
    if not hidden:
        hidden = "- none"
    trajectory = _trajectory_text(round_result.trajectory)[-6000:] or "(no text)"
    return (
        "You are a simulated benchmark user. Decide whether to send one short "
        "follow-up message to the solver.\n"
        "Return only the message text. Return STOP if no more user message is "
        "needed.\n\n"
        f"Persona:\n{persona or '(none)'}\n\n"
        f"Original task instruction:\n{instruction}\n\n"
        "Private facts visible only to you. Reveal a fact only when the solver "
        "asks a targeted clarification question or the verifier result shows "
        "the solver needs that fact:\n"
        f"{hidden}\n\n"
        f"Previous round: {round_result.round}\n"
        f"Previous scene: {round_result.scene or '(none)'}\n"
        f"Previous role: {round_result.role or '(none)'}\n"
        "Previous handoff: "
        f"{round_result.handoff_from or '(none)'} -> "
        f"{round_result.handoff_to or '(none)'}\n"
        f"Rewards: {rewards}\n"
        f"Verifier output: {round_result.verifier_output or ''}\n"
        f"Verifier error: {round_result.verifier_error or ''}\n"
        f"Solver trajectory text:\n{trajectory}\n"
    )
