"""The reasoning effort a run asks for, however it asks.

A run asks for an effort in any of these ways, and each is the request:

- ``--effort`` (``reasoning_effort`` in a config or the SDK);
- an effort in the model's bracket suffix, ``-m 'gpt-6-astra[ultra]'``;
- the agent's own settings: Codex's ``model_reasoning_effort`` in
  ``CODEX_CONFIG``, OpenCode's ``reasoningEffort`` in
  ``OPENCODE_CONFIG_CONTENT``, Claude Code's ``CLAUDE_CODE_EFFORT_LEVEL``;
- the model proxy's: ``BENCHFLOW_REASONING_EFFORT`` (or
  ``LLM_REASONING_EFFORT``) and, for a Bedrock Claude route,
  ``BENCHFLOW_BEDROCK_THINKING_EFFORT``.

0.10.0rc3 read only the first. A suffix or a CODEX_CONFIG effort left
``reasoning_effort`` unset, so codex-acp's model selection
(``acp/runtime.py`` ``_codex_session_model_id``) kept the session's current
variant, or took the first of medium, high, low, and the trial ran at an effort
nobody asked for, with ``effort.requested`` null and nothing to compare.

A model's bracket suffixes are read by kind (:func:`split_model_effort`): a
level of :data:`~benchflow._utils.config.EFFORT_LEVELS` is an effort, taken
off the model id; a context size such as ``[1m]`` stays part of the model id;
anything else is refused.

:func:`settle_effort` folds every request into the run's ``reasoning_effort``
and takes an effort suffix off the model, so the session, the native CLI and
the model proxy all get the bare model and the one effort. Before anything
launches it refuses two ways of asking that differ, a level BenchFlow does
not know, a level the pinned Codex's catalog does not list for the model
(:mod:`benchflow.agents.codex_models`), and a level Claude Code does not take.
A level the run's path is known to run as another or not to take (the
Antigravity CLI's three tiers, Muse's ``none``, a level a Claude model's
documentation does not list for it, any level for an agent whose session
takes none) is refused before launch, naming the levels it takes
(:func:`refuse_known_downgrade`). This module is the one place that
decides: the model proxy sends the level asked for, and a level a
provider lowers where BenchFlow cannot know it is measured from the calls
and recorded as a mismatch, never assumed.

Every check reads the environment the trial reads (:func:`run_env`): the
run's ``agent_env`` with what BenchFlow copies from ``.env`` and the host's
environment. A route that sets an effort of its own on each call (a Bedrock
Claude route) is sent the level the run asked for (:func:`with_route_effort`):
a request wins over the route's default.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchflow.agents.registry import AgentConfig

#: Where a request came from, as records and messages name it.
RUN = "--effort"
MODEL_SUFFIX = "model[effort]"
MATRIX = "--matrix entry"
CODEX_CONFIG = "CODEX_CONFIG model_reasoning_effort"
OPENCODE_CONFIG = "OPENCODE_CONFIG_CONTENT reasoningEffort"
CLAUDE_EFFORT_ENV = "CLAUDE_CODE_EFFORT_LEVEL"
PROVIDER_EFFORT_ENVS = ("BENCHFLOW_REASONING_EFFORT", "LLM_REASONING_EFFORT")
BEDROCK_EFFORT_ENV = "BENCHFLOW_BEDROCK_THINKING_EFFORT"

_CODEX_AGENTS = frozenset({"codex-acp"})
_CLAUDE_AGENTS = frozenset({"claude-agent-acp"})
_OPENCODE_AGENTS = frozenset({"opencode"})
_SUFFIX = re.compile(r"\[([^\[\]]*)\]$")
#: Providers whose gateway route turns an Anthropic Messages agent's effort
#: into a thinking budget: LiteLLM's DeepSeek provider sends DeepSeek's
#: Anthropic endpoint thinking.budget_tokens and no effort (max as 16384).
_BUDGET_ROUTES = frozenset({"deepseek"})
#: Providers whose API applies no reasoning effort, with the control they take
#: instead: any level the run asks of such a route is refused before launch
#: MiMo controls its reasoning with ``thinking.type``. Measured
#: 2026-11-15 on mimo-v2.6-pro at the xiaomi/ route's endpoint, one prompt and
#: three calls per level: reasoning tokens 93-109 at low, 69-88 at high, 89-114
#: at max and 82-95 with no level; xhigh refused with HTTP 400
#: ("reasoning_effort must be one of 'none', 'low', 'medium' or 'high'");
#: ``thinking.type: disabled``, 0. X4's claude-agent-acp runs on
#: xiaomi/mimo-v2.6-flash at xhigh and max went out as high before the
#: gateway's passthrough, and would meet that 400 after it.
_NO_EFFORT_ROUTES = {
    "xiaomi": "MiMo controls its reasoning with thinking.type (enabled or "
    "disabled), not reasoning_effort",
}
#: The thinking budget the gateway's LiteLLM (1.103.3) sends such a route in
#: place of each level: its DEFAULT_REASONING_EFFORT_<LEVEL>_THINKING_BUDGET.
_THINKING_BUDGETS = {
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "xhigh": 8192,
    "max": 16384,
}
#: How to ask an agent that declares no ACP effort option for an effort it
#: applies itself, and the levels that way carries, which its refusal names.
#: OpenHands writes LLM_REASONING_EFFORT into its LLM settings and every call
#: carries it (low, high and xhigh reached Bedrock Converse as asked);
#: for max its launch config sends a ``reasoning`` field the gateway refuses
#:, so max has no working path there.
_OWN_EFFORT_SETTINGS = {
    "openhands": (
        "--agent-env LLM_REASONING_EFFORT={effort}",
        ("none", "low", "medium", "high", "xhigh"),
    )
}
#: The Bedrock Runtime id (a US inference profile, as the gateway's LiteLLM
#: 1.103.3 model map lists it) of each Claude model a refusal can point to,
#: by its dashed id.
_BEDROCK_CLAUDE_IDS = {
    "claude-opus-4-5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-opus-4-7": "us.anthropic.claude-opus-4-7",
    "claude-opus-4-8": "us.anthropic.claude-opus-4-8",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
}
#: A context-window size, as Claude Code's ``[1m]``: part of the model id.
_CONTEXT_SIZE = re.compile(r"^\d+(?:\.\d+)?[km]$", re.IGNORECASE)


#: The Claude Code CLI the table's ``claude_code`` levels were checked on:
#: the one claude-agent-acp runs (agents/registry.py ``_CLAUDE_CODE_PACKAGE``).
#: A test fails when that pin moves, so a bump re-checks the table.
CLAUDE_CODE_CHECKED = "2.1.316"
_LOW_TO_MAX = ("low", "medium", "high", "xhigh", "max")
_NO_XHIGH = ("low", "medium", "high", "max")
_LOW_TO_HIGH = ("low", "medium", "high")


@dataclass(frozen=True)
class ClaudeEfforts:
    """A Claude model's efforts. ``docs``: the levels Anthropic's model
    reference lists for it (2026-11-13), which any agent's route takes.
    ``claude_code``: the levels Claude Code :data:`CLAUDE_CODE_CHECKED` sends
    upstream as asked for that id, from real ``bench run`` trials in Docker
    against a recording upstream."""

    docs: tuple[str, ...]
    claude_code: tuple[str, ...]


_AS_DOCUMENTED = {
    # No effort before Opus 4.5.
    "claude-3-haiku": (),
    "claude-3-sonnet": (),
    "claude-3-opus": (),
    "claude-3-5-haiku": (),
    "claude-3-5-sonnet": (),
    "claude-3-7-sonnet": (),
    "claude-opus-4": (),
    "claude-sonnet-4": (),
    "claude-opus-4-1": (),
    "claude-opus-4-5": _LOW_TO_HIGH,
    "claude-sonnet-4-5": (),
    "claude-haiku-4-5": (),
    "claude-opus-4-6": _NO_XHIGH,
    "claude-sonnet-4-6": _NO_XHIGH,
    "claude-opus-4-7": _LOW_TO_MAX,
    "claude-opus-4-8": _LOW_TO_MAX,
    "claude-opus-5": _LOW_TO_MAX,
    "claude-opus-5-5": _LOW_TO_MAX,
    "claude-sonnet-5": _LOW_TO_MAX,
    "claude-sonnet-5-5": _LOW_TO_MAX,
    "claude-haiku-5-5": _LOW_TO_MAX,
    "claude-fable-5": _LOW_TO_MAX,
    "claude-fable-5-1": _LOW_TO_MAX,
    "claude-fable-5-2": _LOW_TO_MAX,
}
#: Each Claude model's efforts on 0.10, by its id as the route writes it:
#: dashed on the anthropic/ and Bedrock routes (``claude-opus-4-6``), dotted
#: on OpenRouter (``claude-opus-4.6``). A dotted id has a row of its own only
#: where Claude Code treats it apart from the dashed one.
CLAUDE_EFFORTS: dict[str, ClaudeEfforts] = {
    **{
        model: ClaudeEfforts(levels, levels) for model, levels in _AS_DOCUMENTED.items()
    },
    # Claude Code applies high on OpenRouter's dotted Opus 4.5 and sends no
    # effort upstream, whatever -e asks.
    "claude-opus-4.5": ClaudeEfforts(_LOW_TO_HIGH, ()),
}
_CLAUDE_ID = re.compile(
    r"claude-(?:(?:opus|sonnet|haiku|fable)-\d+(?:[.-]\d{1,2})?(?!\d)"
    r"|3(?:[.-][57])?-(?:opus|sonnet|haiku))",
    re.IGNORECASE,
)


class EffortRequestError(ValueError):
    """An effort request refused before launch: the message names the model,
    the level asked for and the levels it takes, or the ways of asking that
    differ. ``hint`` is the next step where that isn't asking for another
    level: a route no level reaches as asked, an agent that takes none."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True)
class EffortRequest:
    """``effort`` is what the run asked for (None: nothing asked), ``source``
    the ways it was asked, joined with ``, `` (``--effort`` only when nothing
    else asked: a config folds the others into it); ``launch`` the effort
    BenchFlow hands the agent's session (the request, except an effort the
    agent applies from its own settings, which it is left to)."""

    effort: str | None
    source: str | None
    launch: str | None


@dataclass(frozen=True)
class SettledEffort:
    """What a config keeps once the request is settled: the model without an
    effort suffix, the run's effort, and where it was asked."""

    model: str | None
    effort: str | None
    source: str | None


def split_model_effort(model: str | None) -> tuple[str | None, str | None]:
    """A model id's bracket suffixes, read by kind: ``("gpt-6-astra",
    "ultra")`` for ``gpt-6-astra[ultra]``, ``("opus[1m]", None)`` for
    ``opus[1m]``, ``("claude-opus-4-6[1m]", "max")`` for
    ``claude-opus-4-6[1m][max]``. An effort level is taken off; a context
    size stays. Raises :class:`EffortRequestError` for any other suffix, or
    for two efforts."""
    if not isinstance(model, str):
        return model, None
    base = model.strip()
    groups: list[str] = []
    while (match := _SUFFIX.search(base)) is not None and match.start() > 0:
        groups.insert(0, match.group(1))
        base = base[: match.start()]
    from benchflow._utils.config import EFFORT_LEVELS

    effort: str | None = None
    kept = ""
    for group in groups:
        value = group.strip().lower()
        if value in EFFORT_LEVELS:
            if effort is not None:
                raise EffortRequestError(
                    f"model {model!r} names two reasoning efforts; give one"
                )
            effort = value
        elif _CONTEXT_SIZE.match(value):
            kept += f"[{group}]"
        else:
            raise EffortRequestError(
                f"model {model!r}: [{group}] is neither a reasoning effort "
                f"({', '.join(EFFORT_LEVELS)}) nor a context size such as [1m]"
            )
    return base + kept, effort


def _normalize(value: object, where: str) -> str | None:
    from benchflow._utils.config import normalize_reasoning_effort

    try:
        return normalize_reasoning_effort(value)
    except ValueError as exc:
        raise EffortRequestError(f"{where} asks for effort {value!r}: {exc}") from None


def _json_object(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _values_of(node: Any, keys: frozenset[str]) -> Iterator[Any]:
    """Every value under one of *keys*, anywhere in a JSON document."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in keys:
                yield value
            else:
                yield from _values_of(value, keys)
    elif isinstance(node, list):
        for item in node:
            yield from _values_of(item, keys)


def _agent_setting_efforts(
    agent: str | None, agent_env: Mapping[str, str]
) -> list[tuple[str, str]]:
    asked: list[tuple[str, str]] = []
    if agent in _CODEX_AGENTS:
        config = _json_object(agent_env.get("CODEX_CONFIG")) or {}
        effort = _normalize(config.get("model_reasoning_effort"), CODEX_CONFIG)
        if effort:
            asked.append((CODEX_CONFIG, effort))
    if agent in _OPENCODE_AGENTS:
        config = _json_object(agent_env.get("OPENCODE_CONFIG_CONTENT"))
        for value in _values_of(config, frozenset({"reasoningEffort"})):
            effort = _normalize(value, OPENCODE_CONFIG)
            if effort:
                asked.append((OPENCODE_CONFIG, effort))
    for name in PROVIDER_EFFORT_ENVS:
        effort = _normalize(agent_env.get(name), name)
        if effort:
            asked.append((name, effort))
    raw = agent_env.get(BEDROCK_EFFORT_ENV)
    if raw is not None and str(raw).strip():
        effort = str(raw).strip().lower()
        _refuse_bad_bedrock_effort(raw, effort)
        asked.append((BEDROCK_EFFORT_ENV, effort))
    return asked


#: The ways of asking BenchFlow hands to the agent's session (its effort
#: option, codex-acp's model[effort] variant). The others act elsewhere: the
#: agent applies its own settings, the model proxy's route its own.
_SESSION_SOURCES = frozenset({RUN, MATRIX, MODEL_SUFFIX, CODEX_CONFIG})
_PROXY_SOURCES = frozenset({*PROVIDER_EFFORT_ENVS, BEDROCK_EFFORT_ENV})
#: The ways of asking that are a variable's value, which .env or the host's
#: environment can give as well as the run.
_ENV_SOURCES = frozenset({CLAUDE_EFFORT_ENV, *_PROXY_SOURCES})


def _given(agent_env: Mapping[str, str] | None) -> dict[str, str]:
    """The run's own ``agent_env``, its neutral spellings (``RUNTIME_*``)
    read as the trial reads them."""
    from benchflow.agents.visible_runtime import host_env_aliases

    return host_env_aliases(dict(agent_env or {}))


def run_env(
    agent: str | None,
    model: str | None,
    agent_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment a trial reads its effort and its route from:
    *agent_env* with what BenchFlow copies from ``.env`` and the host's
    environment, as each trial resolves it (the first steps of
    ``agents.env.resolve_agent_env``). Read alone, *agent_env* missed a
    ``BENCHFLOW_BEDROCK_THINKING_EFFORT`` or an ``AZURE_API_ENDPOINT`` set in
    the shell, which the trial then used."""
    from benchflow.agents.env import inherit_agent_auth_env

    bare, _ = split_model_effort(model)
    return inherit_agent_auth_env(agent or "", bare, _given(agent_env))


def _from_elsewhere(source: str, given: Mapping[str, str]) -> str:
    """Where a variable the run did not set came from, for a refusal to
    name: `` (from .env)`` or `` (from the host's environment)``."""
    if source not in _ENV_SOURCES or source in given:
        return ""
    from benchflow._dotenv import load_dotenv_env

    if load_dotenv_env().get(source):
        return " (from .env)"
    return " (from the host's environment)"


def _without_unread_bedrock_effort(
    model: str | None, env: Mapping[str, str], given: Mapping[str, str]
) -> Mapping[str, str]:
    """*env* without a ``BENCHFLOW_BEDROCK_THINKING_EFFORT`` from .env or the
    shell that no route reads: only a Bedrock Claude route takes it, so off
    one it changes nothing and asks for nothing. The run's own (--agent-env)
    is still refused there; on a Bedrock Claude route either is the request."""
    if BEDROCK_EFFORT_ENV not in env or BEDROCK_EFFORT_ENV in given:
        return env
    rest = {key: value for key, value in env.items() if key != BEDROCK_EFFORT_ENV}
    bare, _ = split_model_effort(model)
    return env if _route_sets_effort(bare, rest) else rest


def effort_request(
    agent: str | None,
    model: str | None,
    reasoning_effort: str | None,
    agent_env: Mapping[str, str] | None = None,
    *,
    run_source: str = RUN,
) -> EffortRequest:
    """The effort the run asks of *agent* on *model*, from every way of
    asking, read from the environment the trial reads (:func:`run_env`).

    Raises :class:`EffortRequestError` when a level is none BenchFlow knows,
    a model suffix is no effort or context size, or two ways ask for
    different levels.
    """
    return _request(
        agent,
        model,
        reasoning_effort,
        run_env(agent, model, agent_env),
        _given(agent_env),
        run_source=run_source,
    )


def _request(
    agent: str | None,
    model: str | None,
    reasoning_effort: str | None,
    env: Mapping[str, str],
    given: Mapping[str, str],
    *,
    run_source: str,
) -> EffortRequest:
    env = _without_unread_bedrock_effort(model, env, given)
    asked: list[tuple[str, str]] = []
    run = _normalize(reasoning_effort, run_source)
    if run:
        asked.append((run_source, run))
    _, suffix = split_model_effort(model)
    if suffix:
        asked.append((MODEL_SUFFIX, suffix))
    asked += _agent_setting_efforts(agent, env)
    claude = env.get(CLAUDE_EFFORT_ENV) if agent in _CLAUDE_AGENTS else None
    if isinstance(claude, str) and claude.strip().lower() not in ("", "auto", "unset"):
        effort = _normalize(claude, CLAUDE_EFFORT_ENV)
        if effort:
            asked.append((CLAUDE_EFFORT_ENV, effort))
    if len({effort for _, effort in asked}) > 1:
        ways = ", ".join(
            f"{source} {effort}{_from_elsewhere(source, given)}"
            for source, effort in asked
        )
        raise EffortRequestError(
            f"the run asks for more than one reasoning effort ({ways}); ask for one"
        )
    if not asked:
        return EffortRequest(None, None, None)
    effort = asked[0][1]
    # A config folds a suffix or a setting's effort into its own
    # reasoning_effort (settle_effort), so the same level beside one of them
    # is that one's, not a second ask.
    named = [source for source, _ in asked if source != RUN] or [RUN]
    # The session gets the level only when the run asked in a way the session
    # applies: Claude Code applies its own variable, OpenHands its
    # LLM_REASONING_EFFORT, the model proxy's route its own settings; an
    # agent with no effort option handed one fails after its sandbox starts.
    session = any(source in _SESSION_SOURCES for source, _ in asked)
    return EffortRequest(effort, ", ".join(named), effort if session else None)


def refuse_unsupported_effort(
    agent: str | None, model: str | None, effort: str | None
) -> None:
    """Refuse, before launch, a level *agent* cannot take on *model*:
    for Codex, one the pinned Codex's catalog does not list for the model
    (a model it does not hold is left to the session, which refuses a level
    it does not offer: ``AgentEffortNotOfferedError``); for Claude Code, one
    outside :data:`~benchflow._utils.config.CLAUDE_EFFORT_LEVELS`. The
    message names the levels it does take."""
    if not effort:
        return
    bare, _ = split_model_effort(model)
    if agent in _CLAUDE_AGENTS:
        from benchflow._utils.config import CLAUDE_EFFORT_LEVELS

        if effort not in CLAUDE_EFFORT_LEVELS:
            raise EffortRequestError(
                f"Claude Code ({agent}) does not take reasoning effort {effort}; "
                f"it takes {', '.join(CLAUDE_EFFORT_LEVELS)}"
            )
        return
    if agent not in _CODEX_AGENTS:
        return
    from benchflow.agents.codex_models import CODEX_CATALOG_VERSION, codex_model_efforts

    levels = codex_model_efforts(bare)
    if levels is None or effort in levels:
        return
    raise EffortRequestError(
        f"{bare} does not take reasoning effort {effort} in the pinned Codex "
        f"{CODEX_CATALOG_VERSION} ({agent}); it takes {', '.join(levels)}"
    )


def _refuse_bad_bedrock_effort(raw: object, value: str) -> None:
    """A Bedrock thinking effort the route cannot take fails here, never
    becomes another level (``extreme`` ran at high)."""
    from benchflow.providers.litellm_config import _BEDROCK_EFFORT_LADDER

    if value not in _BEDROCK_EFFORT_LADDER:
        raise EffortRequestError(
            f"{BEDROCK_EFFORT_ENV}={raw!r} is no Bedrock thinking effort; it "
            f"takes {', '.join(_BEDROCK_EFFORT_LADDER)}"
        )


def _route(model: str | None, agent_env: Mapping[str, str]) -> Any:
    """The model proxy's route for *model*; None when it has none here (no
    model, or a route its own checks refuse when the run starts)."""
    if not model:
        return None
    from benchflow.providers.litellm_config import resolve_litellm_route

    try:
        return resolve_litellm_route(model, dict(agent_env))
    except ValueError:
        return None


def _refuse_requests_nothing_applies(
    agent: str | None,
    model: str | None,
    sources: list[str],
    effort: str,
    agent_env: Mapping[str, str],
) -> None:
    """A model proxy's setting acts only where its route puts it on each call
    (BENCHFLOW_REASONING_EFFORT on an Azure OpenAI route, the Bedrock
    thinking effort on a Bedrock Claude route), and LLM_REASONING_EFFORT also
    where OpenHands reads it. Anywhere else nothing applies it: refused here,
    never after the sandbox starts."""
    proxy = [source for source in sources if source in _PROXY_SOURCES]
    if not proxy:
        return
    route = _route(model, agent_env)
    pinned = (getattr(route, "litellm_params", None) or {}).get("reasoning_effort")
    for source in proxy:
        if source == "LLM_REASONING_EFFORT" and agent == "openhands":
            continue
        if source == BEDROCK_EFFORT_ENV and pinned is not None:
            # The Bedrock route takes it, and sends it as asked.
            continue
        if source != BEDROCK_EFFORT_ENV and pinned == effort:
            continue
        where = (
            "a Bedrock Claude route"
            if source == BEDROCK_EFFORT_ENV
            else "an Azure OpenAI route"
        )
        raise EffortRequestError(
            f"{source}={effort}: nothing applies it to {agent} on {model}: the "
            f"model proxy sets it only on {where}; ask with --effort where the "
            "agent takes one, or drop it"
        )


def _route_sets_effort(model: str | None, env: Mapping[str, str]) -> bool:
    """Whether the model proxy's route for *model* sets an effort of its own
    on each call, whatever the agent sends: a Bedrock Claude route that takes
    adaptive thinking (``BENCHFLOW_BEDROCK_THINKING_EFFORT``, high when
    unset)."""
    route = _route(model, env)
    return getattr(route, "provider_name", None) == "aws-bedrock" and (
        "reasoning_effort" in (getattr(route, "litellm_params", None) or {})
    )


def _claude_ids(model: str) -> tuple[str, str] | None:
    """*model*'s Claude id as the route writes it (``claude-opus-4.6``) and
    dashed (``claude-opus-4-6``); None for a model that is no Claude id."""
    found = _CLAUDE_ID.search(model)
    if not found:
        return None
    own = found.group(0).lower()
    return own, own.replace(".", "-")


def _claude_entry(model: str) -> ClaudeEfforts | None:
    """*model*'s row of :data:`CLAUDE_EFFORTS`: its own id's, else its
    dashed id's; None for a Claude model the table does not name."""
    ids = _claude_ids(model)
    if ids is None:
        return None
    own, dashed = ids
    return CLAUDE_EFFORTS.get(own) or CLAUDE_EFFORTS.get(dashed)


def _claude_levels(model: str) -> tuple[str, ...] | None:
    """The efforts Anthropic's model reference lists for a Claude model
    (:data:`CLAUDE_EFFORTS` ``docs``); None for a model it does not name."""
    entry = _claude_entry(model)
    return entry.docs if entry is not None else None


def _known_levels(
    agent: str | None, model: str | None
) -> tuple[str, tuple[str, ...]] | None:
    """``(who, levels)`` where BenchFlow knows the path runs only some levels
    as asked: the Antigravity CLI's three tiers (its shim maps the others onto
    them), Muse without ``none`` (its shim runs it at minimal), a Claude
    model's documented levels (:func:`_claude_levels`; Claude Code refuses
    the others itself, but only after its sandbox starts). None where
    BenchFlow does not know."""
    from benchflow._utils.config import EFFORT_LEVELS

    if agent == "antigravity":
        from benchflow.agents.antigravity_acp_shim import _EFFORT_ALIASES

        return "the Antigravity CLI (agy)", tuple(
            level for level in EFFORT_LEVELS if _EFFORT_ALIASES.get(level) == level
        )
    if agent == "muse-code":
        from benchflow.agents.muse_acp_shim import _EFFORT_ALIASES as muse

        return "Muse", tuple(
            level for level in EFFORT_LEVELS if muse.get(level, level) == level
        )
    bare, _ = split_model_effort(model)
    if bare:
        levels = _claude_levels(bare)
        if levels is not None:
            return bare, levels
        # An OpenAI model the pinned Codex lists takes those levels whichever
        # agent asks: the provider refuses the others after launch. A
        # provider catalog BenchFlow ships for Codex lists only the levels
        # Codex offers (DeepSeek's API takes every level), so it binds Codex
        # alone (refuse_unsupported_effort), not another agent.
        from benchflow.agents.codex_models import codex_model_efforts

        catalog = codex_model_efforts(bare, provider_catalogs=False)
        if catalog:
            return bare, catalog
    return None


def refuse_known_downgrade(
    agent: str | None, model: str | None, effort: str | None
) -> None:
    """Refuse, before launch, a level the run's path is known to run as
    another or not to take (:func:`_known_levels`), naming the levels it does
    take and the highest, so the run can ask for one of them (a
    warned downgrade still runs lower than asked)."""
    if not effort:
        return
    known = _known_levels(agent, model)
    if known is None or effort in known[1]:
        return
    who, levels = known
    if not levels:
        raise EffortRequestError(
            f"{who} takes no reasoning effort ({agent}): drop {effort}",
            hint="drop -e: the model takes no reasoning effort",
        )
    raise EffortRequestError(
        f"{who} does not take reasoning effort {effort} ({agent}); it takes "
        f"{', '.join(levels)} (at most {levels[-1]}): ask for one of them"
    )


def _refuse_what_the_route_cannot_carry(
    agent: str | None, model: str | None, effort: str | None, launch: str | None
) -> None:
    """Refuse, before launch, an effort Claude Code's route cannot carry as
    asked: one the gateway turns into a thinking budget (a DeepSeek route,
    :data:`_BUDGET_ROUTES`; a run known to mismatch every call is a failed
    run), and a level Claude Code sends no effort for on that id
    (:data:`CLAUDE_EFFORTS` ``claude_code``: OpenRouter's dotted Opus 4.5)."""
    from benchflow.agents.registry import AGENT_ALIASES, AGENTS

    name = AGENT_ALIASES.get(agent or "", agent)
    agent_cfg = AGENTS.get(name or "")
    bare, _ = split_model_effort(model)
    if not bare or agent_cfg is None or agent_cfg.api_protocol != "anthropic-messages":
        return
    if effort:
        from benchflow.agents.providers import (
            find_provider,
            find_provider_for_bare_model,
        )

        provider = find_provider(bare) or find_provider_for_bare_model(bare)
        if provider is not None and provider[0] in _BUDGET_ROUTES:
            budgets = ", ".join(
                f"{level} {tokens}" for level, tokens in _THINKING_BUDGETS.items()
            )
            budget = _THINKING_BUDGETS.get(effort)
            runs_as = f"budget_tokens {budget}" if budget else "a thinking budget"
            raise EffortRequestError(
                f"{bare} turns reasoning effort into thinking.budget_tokens "
                f"({name}): the gateway's {provider[0]} route sends a thinking "
                f"budget and no effort ({budgets}), so -e {effort} would run as "
                f"{runs_as}, not as {effort}",
                hint="drop -e: no level reaches this route as asked, and the "
                "thinking budget it sends is recorded",
            )
    ids = _claude_ids(bare) if launch else None
    entry = _claude_entry(bare) if ids else None
    if ids and entry and launch in entry.docs and launch not in entry.claude_code:
        dashed = CLAUDE_EFFORTS.get(ids[1])
        if dashed is not None and launch in dashed.claude_code:
            where, hint = _routes_that_take(ids[1], launch)
        else:
            where = f"no route BenchFlow knows carries -e {launch} for it"
            hint = "drop -e"
        raise EffortRequestError(
            f"Claude Code {CLAUDE_CODE_CHECKED} ({name}) sends no effort for "
            f"{bare}: it runs at its own level with none upstream; {where}",
            hint=hint,
        )


def _routes_that_take(dashed: str, launch: str) -> tuple[str, str]:
    """Where a refused Claude Code effort can run instead, for a Claude
    model's dashed id: the anthropic/ route, and the Bedrock route where
    :data:`_BEDROCK_CLAUDE_IDS` names its id (the refusal's words, its hint)."""
    bedrock = _BEDROCK_CLAUDE_IDS.get(dashed)
    routes = f"the anthropic/ route (anthropic/{dashed})"
    run_as = f"anthropic/{dashed}"
    if bedrock:
        routes += f" or the Bedrock route (aws-bedrock/{bedrock})"
        run_as += f" or aws-bedrock/{bedrock}"
    return f"it takes -e {launch} on {routes}", f"run it as {run_as}, or drop -e"


def session_takes_effort(
    config: AgentConfig, upstream_model: str | None = None
) -> bool:
    """Whether a registered agent's session takes an effort, as its registry
    entry (core or manifest) declares: an ACP effort option
    (``acp_effort_config_id``), or for codex-acp the ``model[effort]`` ids its
    adapter lists. These are the only ways the ACP runtime hands a session an
    effort; any other agent took ``-e`` through the dry run and failed after
    its sandbox and install ("does not declare or advertise an ACP effort
    config option"). An entry whose option exists only for a model
    with variants (``acp_effort_needs_variants``: OpenCode) takes one only
    for an *upstream_model* the agent is handed with variants
    (:func:`~benchflow.agents.registry.registers_model_variants`)."""
    if config.acp_effort_needs_variants and upstream_model is not None:
        from benchflow.agents.registry import registers_model_variants

        return registers_model_variants(config.name, upstream_model)
    return bool(config.acp_effort_config_id) or config.name in _CODEX_AGENTS


def _refuse_agent_without_effort(
    agent: str | None,
    model: str | None,
    launch: str | None,
    agent_env: Mapping[str, str],
    named: str | None = None,
) -> None:
    """Refuse, before launch, an effort the session would be handed where the
    agent's registry entry says it takes none (:func:`session_takes_effort`),
    naming the agent as the run named it, and its own setting for the levels
    that carries. A model proxy's own setting (``BENCHFLOW_REASONING_EFFORT``
    on an Azure OpenAI route) and OpenHands' ``LLM_REASONING_EFFORT`` never
    reach the session and are still taken."""
    from benchflow.agents.registry import find_registered_agent

    config = find_registered_agent(agent) if launch and agent else None
    if config is None:
        return
    bare, _ = split_model_effort(model)
    upstream = None
    if config.acp_effort_needs_variants and bare:
        upstream = getattr(_route(bare, agent_env), "upstream_model", bare)
    named = named or agent or config.name
    if session_takes_effort(config, upstream):
        if upstream is not None and launch:
            _refuse_a_level_the_model_lacks(named, bare or "", launch, upstream)
        return
    from benchflow.agents.registry import opencode_google_provider

    if config.acp_effort_needs_variants and opencode_google_provider(upstream or ""):
        why = f"it offers no effort option for {bare}"
    elif config.acp_effort_needs_variants:
        why = (
            "it offers an effort option only for a model with variants, and "
            f"the gateway registers {bare} for it with none (a gemini/ model, "
            "on its own google provider, has them)"
        )
    else:
        why = (
            "its registry entry declares no ACP effort option, so BenchFlow "
            "can't hand it one"
        )
    own = _OWN_EFFORT_SETTINGS.get(config.name)
    if own and launch in own[1]:
        ask = (
            f"ask through its own setting, {own[0].format(effort=launch)}, or "
            "drop the effort"
        )
    elif own:
        ask = (
            f"drop the effort (its own setting, {own[0].format(effort='<level>')}, "
            f"carries {', '.join(own[1])}, not {launch})"
        )
    else:
        ask = "drop the effort"
    raise EffortRequestError(
        f"reasoning effort {launch} can't reach the {named} session: {why}; "
        f"{ask}.\nNothing ran, so nothing was recorded.",
        hint=ask,
    )


def _refuse_a_level_the_model_lacks(
    named: str, model: str, launch: str, upstream_model: str
) -> None:
    """Refuse, before launch, a level the pinned OpenCode doesn't send the
    model it is handed
    (:data:`~benchflow.agents.registry.OPENCODE_GEMINI_EFFORTS`): its session
    refuses it after the install ("effort not found: max"), or its
    google provider at the first call ("invalid google provider options",
    :data:`~benchflow.agents.registry.OPENCODE_GEMINI_UNSENT`). Levels it
    sends as a thinking budget (Gemini 2.5) are refused too: none reaches the
    model as asked (as for DeepSeek)."""
    from benchflow.agents.registry import (
        OPENCODE_CATALOG_DATE,
        OPENCODE_GEMINI_EFFORTS_VERSION,
        opencode_model_variants,
        opencode_unsent_levels,
    )

    variants = opencode_model_variants(upstream_model)
    if variants is None:
        return
    levels, budget = variants
    version = OPENCODE_GEMINI_EFFORTS_VERSION
    if budget:
        raise EffortRequestError(
            f"{model} turns reasoning effort into a thinking budget ({named}): "
            f"OpenCode {version} offers it {', '.join(levels)}, each sent as "
            "generationConfig.thinkingConfig.thinkingBudget and no level, so "
            f"-e {launch} can't run as asked",
            hint="drop -e: no level reaches this model as asked",
        )
    if launch not in levels:
        sends = f"it sends {', '.join(levels)} for this model"
        if launch in opencode_unsent_levels(upstream_model):
            sends = (
                f"BenchFlow's pinned OpenCode catalog {OPENCODE_CATALOG_DATE} lists "
                f"{launch} for this model, but its google provider "
                'refuses that level at the first call ("invalid google provider '
                'options"); '
                f"it sends {', '.join(levels)}"
            )
        raise EffortRequestError(
            f"{model} takes no reasoning effort {launch} from OpenCode {version} "
            f"({named}): {sends}; ask for one of them"
        )


def _refuse_a_pin_from_benchflow_s_environment(agent_env: Mapping[str, str]) -> None:
    """Refuse a run that does not ask for the effort pin while
    ``BENCHFLOW_EFFORT_PIN`` is set in BenchFlow's own environment: a host
    proxy starts from that environment with the run's agent env over it, so
    the pin would hold the run's calls while its config and run card say
    none (benchflow.effort_pin)."""
    import os

    from benchflow.effort_pin import EFFORT_PIN_ENV, effort_pin_setting

    try:
        host = effort_pin_setting(os.environ)
    except ValueError as exc:
        raise EffortRequestError(str(exc)) from None
    if host is not None and EFFORT_PIN_ENV not in agent_env:
        raise EffortRequestError(
            f"{EFFORT_PIN_ENV} is set in BenchFlow's own environment (the shell "
            "or .env), and this run does not ask for the effort pin: a model "
            "proxy started here would pin its calls anyway",
            hint=f"unset {EFFORT_PIN_ENV}, or pin the run with --effort-pin "
            "(effort_pin: true in a run config)",
        )


def _refuse_a_pin_nothing_can_hold(
    agent: str | None,
    model: str | None,
    effort: str | None,
    agent_env: Mapping[str, str],
) -> str | None:
    """The level ``--effort-pin`` holds the run's model calls to (the run's
    effort), None without the pin; refused before launch where it could pin
    nothing: the run asks for no effort, or its model calls go around
    BenchFlow's model proxy (benchflow.effort_pin)."""
    from benchflow.effort_pin import SWITCH_ON, effort_pin_setting

    _refuse_a_pin_from_benchflow_s_environment(agent_env)
    try:
        pin = effort_pin_setting(agent_env)
    except ValueError as exc:
        raise EffortRequestError(str(exc)) from None
    if pin is None:
        return None
    if not effort:
        raise EffortRequestError(
            "--effort-pin holds every model call's reasoning effort to the run's, "
            "and this run asks for none",
            hint="give --effort (or a model[effort] suffix), or drop --effort-pin",
        )
    if pin not in (SWITCH_ON, effort):
        raise EffortRequestError(
            f"BENCHFLOW_EFFORT_PIN={pin} is not the run's effort {effort}",
            hint="use --effort-pin, which pins to the run's --effort",
        )
    from benchflow.trial_budget import unenforceable_reason

    why = unenforceable_reason(agent or "", model, agent_env)
    if why is not None:
        raise EffortRequestError(
            f"--effort-pin needs BenchFlow's model proxy, and {agent}'s model "
            f"calls would go around it: {why}",
            hint="run it on a route the proxy serves (an API key), or drop "
            "--effort-pin",
        )
    return effort


def _pin_to_the_run_s_level(
    agent: str | None,
    model: str | None,
    reasoning_effort: str | None,
    env: dict[str, str],
) -> None:
    """Give the proxy the run's level in place of the pin's switch."""
    from benchflow.effort_pin import EFFORT_PIN_ENV, effort_pin_on

    # Every rollout passes here before its proxy starts, the SDK's too.
    _refuse_a_pin_from_benchflow_s_environment(env)
    try:
        on = effort_pin_on(env)
    except ValueError as exc:
        raise EffortRequestError(str(exc)) from None
    if not on:
        return
    asked = effort_request(agent, model, reasoning_effort, env).effort
    level = _refuse_a_pin_nothing_can_hold(agent, model, asked, env)
    if level is not None:
        env[EFFORT_PIN_ENV] = level


def with_route_effort(
    agent: str | None,
    model: str | None,
    reasoning_effort: str | None,
    agent_env: Mapping[str, str],
) -> dict[str, str]:
    """A trial's resolved *agent_env*, with the level the run asked for where
    the model proxy's route sets an effort of its own on each call. A Bedrock
    Claude route sets ``BENCHFLOW_BEDROCK_THINKING_EFFORT``, high when unset,
    over what the agent sends, so ``-e low`` ran at high and every trial
    recorded a mismatch. A request wins over a route's default (lead ruling,
    2026-11-13): the route is sent the request. With nothing asked, the
    route keeps its default. With ``--effort-pin`` (benchflow.effort_pin) the
    proxy is given the run's level, which it holds every call to."""
    env = dict(agent_env)
    _pin_to_the_run_s_level(agent, model, reasoning_effort, env)
    if str(env.get(BEDROCK_EFFORT_ENV) or "").strip():
        # The request itself: settle_effort refused it beside another level.
        return env
    asked = effort_request(agent, model, reasoning_effort, env).effort
    bare, _ = split_model_effort(model)
    if asked and _route_sets_effort(bare, env):
        env[BEDROCK_EFFORT_ENV] = asked
    return env


def _refuse_effort_the_route_ignores(
    agent: str | None, model: str | None, effort: str | None, source: str | None
) -> None:
    """Refuse, before launch, any effort on a route whose provider applies
    none (:data:`_NO_EFFORT_ROUTES`), however it was asked: MiMo refuses xhigh
    at the first call and runs low, high and max alike."""
    bare, _ = split_model_effort(model)
    if not effort or not bare:
        return
    from benchflow.agents.providers import find_provider, find_provider_for_bare_model

    provider = find_provider(bare) or find_provider_for_bare_model(bare)
    why = _NO_EFFORT_ROUTES.get(provider[0]) if provider is not None else None
    if why is None:
        return
    raise EffortRequestError(
        f"{bare} applies no reasoning effort ({agent}): {why}. On "
        "mimo-v2.6-pro the same prompt used as many reasoning tokens at low, "
        "high and max, and xhigh was refused with HTTP 400, so "
        f"{source or RUN} {effort} would not run as {effort}",
        hint="drop the effort: MiMo applies none, and thinking.type is its own control",
    )


def _refuse_a_model_the_pinned_catalog_lacks(
    agent: str | None,
    model: str | None,
    agent_env: Mapping[str, str],
    named: str | None,
) -> None:
    """Refuse, before launch, OpenCode on a Gemini model its pinned catalog
    (:mod:`benchflow.agents.opencode_catalog`, which every OpenCode launch
    reads through ``OPENCODE_MODELS_PATH``) lacks, or holds but OpenCode
    drops: its session refuses the model after the install ("model not
    found"), and no live catalog fetch stands in for it."""
    from benchflow.agents.registry import (
        ACPX_KEY_PREFIX,
        find_registered_agent,
        opencode_catalog_refusal,
    )

    config = find_registered_agent(agent) if agent else None
    if config is None or config.name.removeprefix(ACPX_KEY_PREFIX) != "opencode":
        return
    bare, _ = split_model_effort(model)
    if not bare:
        return
    upstream = getattr(_route(bare, agent_env), "upstream_model", bare)
    why = opencode_catalog_refusal(upstream, named or agent or config.name)
    if why:
        raise EffortRequestError(
            f"{bare} {why}",
            hint=(
                "pick a model the catalog holds (benchflow/agents/opencode_catalog/"
                "models-dev.json), or refresh it (tools/opencode_catalog.py)"
            ),
        )


def settle_effort(
    agent: str | None,
    model: str | None,
    reasoning_effort: str | None,
    agent_env: Mapping[str, str] | None = None,
    *,
    run_source: str = RUN,
    named_as: str | None = None,
) -> SettledEffort:
    """The run's model and ``reasoning_effort`` once every way of asking is
    counted (a suffix or a setting's effort folded in, an effort suffix taken
    off the model), after refusing what cannot run
    (:func:`effort_request`, :func:`refuse_unsupported_effort`, a provider
    setting nothing applies, a level the path is known not to run as asked:
    :func:`refuse_known_downgrade`), all read from the environment the trial
    reads (:func:`run_env`)."""
    given = _given(agent_env)
    env = _without_unread_bedrock_effort(model, run_env(agent, model, agent_env), given)
    _refuse_a_model_the_pinned_catalog_lacks(agent, model, env, named_as)
    request = _request(
        agent, model, reasoning_effort, env, given, run_source=run_source
    )
    # A route that applies no effort first: every level an agent's own
    # check would suggest instead is refused there too (R643 L1).
    _refuse_effort_the_route_ignores(agent, model, request.effort, request.source)
    refuse_unsupported_effort(agent, model, request.effort)
    refuse_known_downgrade(agent, model, request.effort)
    _refuse_agent_without_effort(agent, model, request.launch, env, named_as)
    _refuse_what_the_route_cannot_carry(agent, model, request.effort, request.launch)
    _refuse_a_pin_nothing_can_hold(agent, model, request.effort, given)
    bare, _ = split_model_effort(model)
    if request.effort and request.source:
        _refuse_requests_nothing_applies(
            agent, bare, request.source.split(", "), request.effort, env
        )
    return SettledEffort(bare, request.launch, request.source)


def effort_source(kept: str | None, settled: str | None) -> str | None:
    """The source a config records for its effort: the one settled now,
    unless that is ``--effort`` for an effort an earlier settling folded in
    from elsewhere (a ``dataclasses.replace()`` sees the folded effort and
    the bare model, not the suffix)."""
    if settled is None:
        return None
    return kept if kept is not None and settled == RUN else settled
