"""The run card: what a run is set to measure, shown before it spends anything.

Every job writes ``run-card.json`` in its folder and logs a short card before
its first trial: the tasks, the agent and what it installs, the model with its
effort and sampling, the trials per task and in all, the concurrency, the time
limits, the network policy, how grading runs, the controls already run on these
tasks, the longest the run can take, and any budget.

A task set can name the setup its published numbers used, in
``run-reference.toml`` beside its tasks. The card then lists every setting where
this run differs. A reference with ``strict = true`` refuses such a run before
its job folder exists, unless the run says why (``accept_deviations``, or
``--accept-deviations`` on the command line); the note is kept on the card.

Some settings BenchFlow does not set for an agent trial: temperature, top_p,
thinking, a step limit, the agent's own version when it is not a BenchFlow pin.
It does set one output cap, Claude Code's CLAUDE_CODE_MAX_OUTPUT_TOKENS, unless
the operator does; the card names its value and where it came from, beside
every other variable BenchFlow knows that changes what a trial measures
(:data:`ENV_SETTINGS`): from ``--agent-env``, ``.env``, the host's
environment or BenchFlow's default. A reference that names them gets them
listed for the operator to check in the agent's own settings, because the card
cannot see what the agent sends.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import re
import shlex
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow.agents.env import inherit_agent_auth_env, uses_direct_anthropic_endpoint
from benchflow.errors import UserError
from benchflow.model_settings import vendor_setting

if TYPE_CHECKING:
    from benchflow.evaluation import Evaluation

logger = logging.getLogger(__name__)

RUN_CARD = "run-card.json"
REFERENCE_FILE = "run-reference.toml"
SCHEMA_VERSION = 1
CONTROL_AGENTS = ("oracle", "nop")
_CONTROL_ROLES = {
    "oracle": "the task's reference solution",
    "nop": "an agent that does nothing",
}
# Folders read for control jobs (see _control_roots): a large tree is read
# only this far.
_MAX_SCANNED = 5000
_TRIAL_DIR = re.compile(r"trial-\d+")
# Tasks matched to control trials by content digest; a larger selection is
# matched by name, and the card says so.
_MAX_DIGEST_TASKS = 200

_NUMBER = (int, float)
# Settings the run resolves, compared with the reference.
_CHECKED: dict[str, type | tuple[type, ...]] = {
    "agent": str,
    "harness": str,
    "model": str,
    "effort": str,
    "trials": int,
    "agent_timeout_sec": _NUMBER,
    "network": str,
    "verifier_sandbox": str,
}
# Settings BenchFlow does not set for an agent trial: listed, never compared.
_UNCHECKED: dict[str, type | tuple[type, ...]] = {
    "agent_version": str,
    "temperature": _NUMBER,
    "top_p": _NUMBER,
    "thinking": bool,
    "max_steps": int,
    "max_output_tokens": int,
}
_META: dict[str, type | tuple[type, ...]] = {
    "source": str,
    "strict": bool,
    "notes": str,
}
_PIN_RE = re.compile(
    r"(?:npm install -g --prefix \S+ |\s)(\S+@\d[\w.\-]*|\S+==\d[\w.\-]*)"
)


@dataclass(frozen=True)
class EnvSetting:
    """A variable BenchFlow knows that changes what a trial measures."""

    name: str
    reads: str  # who reads it
    sets: str  # what it sets, in a few words
    agents: tuple[str, ...] = ()  # the agents that read it; () for any


_CLAUDE_CODE = ("claude-agent-acp",)
#: Read by the agent or by BenchFlow's model proxy, from the agent's
#: environment: ``--agent-env``, or copied from .env or the host's environment
#: for the names agents.env.inheritable_env_names lists (and Muse's settings,
#: for muse-code), or BenchFlow's default.
ENV_SETTINGS: tuple[EnvSetting, ...] = (
    EnvSetting(
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "Claude Code",
        "its output cap per response",
        _CLAUDE_CODE,
    ),
    EnvSetting(
        "MAX_THINKING_TOKENS",
        "Claude Code",
        "its thinking budget per response",
        _CLAUDE_CODE,
    ),
    EnvSetting(
        "CLAUDE_CODE_EFFORT_LEVEL",
        "Claude Code",
        "its effort, over the run's",
        _CLAUDE_CODE,
    ),
    EnvSetting(
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
        "Claude Code",
        "the context window it assumes for the model",
        _CLAUDE_CODE,
    ),
    EnvSetting(
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
        "Claude Code",
        "how full its context gets before it compacts",
        _CLAUDE_CODE,
    ),
    EnvSetting(
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
        "Claude Code",
        "the share of that window at which it compacts",
        _CLAUDE_CODE,
    ),
    EnvSetting("DISABLE_COMPACT", "Claude Code", "no compaction", _CLAUDE_CODE),
    EnvSetting(
        "CLAUDE_CODE_DISABLE_1M_CONTEXT",
        "Claude Code",
        "a 200K window for a model that has 1M",
        _CLAUDE_CODE,
    ),
    EnvSetting(
        "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT",
        "Claude Code",
        "no compaction for a model it does not know until the context is too long",
        _CLAUDE_CODE,
    ),
    EnvSetting(
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "Claude Code",
        "its subagents' model",
        _CLAUDE_CODE,
    ),
    EnvSetting(
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "Claude Code",
        "the model of its background calls",
        _CLAUDE_CODE,
    ),
    EnvSetting(
        "ANTHROPIC_SMALL_FAST_MODEL",
        "Claude Code",
        "the model of its background calls (the older name)",
        _CLAUDE_CODE,
    ),
    EnvSetting(
        "BENCHFLOW_MODEL_TEMPERATURE",
        "the deepagents agent",
        "the temperature it asks for",
        ("deepagents",),
    ),
    EnvSetting(
        "BENCHFLOW_MODEL_TOP_P",
        "the deepagents agent",
        "the top_p it asks for",
        ("deepagents",),
    ),
    EnvSetting(
        "BENCHFLOW_MODEL_MAX_TOKENS",
        "the deepagents agent",
        "the output cap it asks for",
        ("deepagents",),
    ),
    EnvSetting(
        "BENCHFLOW_MODEL_SEED",
        "the deepagents agent",
        "the sampling seed it asks for",
        ("deepagents",),
    ),
    EnvSetting(
        "BENCHFLOW_BEDROCK_THINKING_EFFORT",
        "BenchFlow's model proxy",
        "the thinking effort of a Bedrock Claude model",
    ),
    EnvSetting(
        "BENCHFLOW_REASONING_EFFORT",
        "BenchFlow's model proxy",
        "the reasoning effort of an Azure OpenAI model",
    ),
    EnvSetting(
        "LLM_REASONING_EFFORT",
        "BenchFlow's model proxy",
        "the reasoning effort of an Azure OpenAI model (the older name)",
    ),
    EnvSetting(
        "BENCHFLOW_MUSE_QUOTA_WAIT_SEC",
        "the Muse agent",
        "how long a turn waits for a spent quota to reset",
        ("muse-code",),
    ),
    EnvSetting(
        "BENCHFLOW_MUSE_QUOTA_SLACK_SEC",
        "the Muse agent",
        "the margin it keeps before the prompt's time runs out",
        ("muse-code",),
    ),
)
#: Read by BenchFlow on the machine that runs the job, never by the agent.
HOST_SETTINGS: tuple[EnvSetting, ...] = (
    EnvSetting(
        "BENCHFLOW_AGENT_PENDING_TOOL_GRACE_SEC",
        "BenchFlow's idle watchdog",
        "how long a silent tool call holds off the idle limit",
    ),
)
#: How the card prints each source (``settings[].source``).
SOURCE_WORDS = {
    "agent_env": "from --agent-env",
    "dotenv": "from .env",
    "host_env": "from the host's environment",
    "benchflow_default": "BenchFlow's default",
}
# A setting's value as the card shows it: a number, a word or a model id.
_PLAIN_SETTING = re.compile(r"[\w.:/@\[\]+-]{1,120}")


def _shown_setting(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "(blank)"
    return text if _PLAIN_SETTING.fullmatch(text) else "(set)"


def env_settings(agent: str, model: str | None, agent_env: dict[str, str]) -> list:
    """Every :data:`ENV_SETTINGS` and :data:`HOST_SETTINGS` variable in
    effect for this run: its value and where it came from.

    The sources, highest first, as agents.env resolves them: the run's
    ``agent_env`` (``--agent-env``, a config's ``agent_env``), then ``.env``,
    then the host's environment (only for the names BenchFlow copies from
    them), then BenchFlow's own default. A setting none of them sets is left
    out."""
    import os

    from benchflow._dotenv import load_dotenv_env
    from benchflow.agents.env import (
        _MUSE_AGENT,
        _MUSE_SETTINGS_ENV,
        CLAUDE_CODE_MAX_OUTPUT_TOKENS_DEFAULT,
        inheritable_env_names,
    )

    dotenv = load_dotenv_env()
    copied = set(inheritable_env_names())
    if agent == _MUSE_AGENT:
        copied |= set(_MUSE_SETTINGS_ENV)
    defaults: dict[str, str] = {}
    if agent in _CLAUDE_CODE:
        defaults["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = (
            CLAUDE_CODE_MAX_OUTPUT_TOKENS_DEFAULT
        )
    bedrock = _bedrock_default_effort(model)
    if bedrock is not None:
        defaults["BENCHFLOW_BEDROCK_THINKING_EFFORT"] = bedrock

    def found(name: str, *, from_host: bool) -> tuple[str, str] | None:
        if from_host:
            # BenchFlow's own environment, to which the CLI adds .env where
            # it is unset.
            value = os.environ.get(name)
            if value and value.strip():
                return value, "dotenv" if dotenv.get(name) == value else "host_env"
        elif name in agent_env:
            return str(agent_env[name]), "agent_env"
        elif name in copied:
            # Resolution copies .env first, then the host: .env wins.
            for source, values in (("dotenv", dotenv), ("host_env", os.environ)):
                value = values.get(name)
                if value and value.strip():
                    return value, source
        if name in defaults:
            return defaults[name], "benchflow_default"
        return None

    rows: list[dict[str, str]] = []
    for setting, from_host in (
        *((s, False) for s in ENV_SETTINGS),
        *((s, True) for s in HOST_SETTINGS),
    ):
        if setting.agents and agent not in setting.agents:
            continue
        hit = found(setting.name, from_host=from_host)
        if hit is None:
            continue
        rows.append(
            {
                "name": setting.name,
                "value": _shown_setting(hit[0]),
                "source": hit[1],
                "reads": setting.reads,
                "sets": setting.sets,
            }
        )
    return rows


# A Claude model name Claude Code recognizes in a model id
# (``us.anthropic.claude-opus-4-8``, ``runtime-anthropic-claude-sonnet-5``),
# and those that have a 1M window with no ``[1m]`` suffix: Fable, Sonnet 5 and
# later, Haiku 5.5, Opus 4.7 and later (Claude Code's model configuration
# page, read 2026-11-05, for its 2.1.316 that BenchFlow pins).
_CLAUDE_MODEL = re.compile(r"claude-(?:opus|sonnet|haiku|fable)-\d", re.IGNORECASE)
_NATIVE_1M = re.compile(
    r"claude-(?:fable-\d|sonnet-(?:[5-9]|\d\d)|haiku-(?:5-5|[6-9])"
    r"|opus-(?:4-(?:[7-9]|\d\d)|[5-9]))",
    re.IGNORECASE,
)
#: Words for each source of a context window (``context.*_source``).
WINDOW_WORDS = {
    "unrecognized": "Claude Code's assumed window for a model id it does not recognize",
    "1m_suffix": "the model id's [1m] suffix",
    "claude_1m": "a Claude model with a 1M window",
    "claude_200k": "a Claude model with a 200K window",
    "disabled_1m": "CLAUDE_CODE_DISABLE_1M_CONTEXT",
    "max_context_tokens": "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
    "litellm_model_map": "LiteLLM's model map",
    "provider_registry": "BenchFlow's provider registry",
}


def claude_code_window(model_id: str, env: dict[str, str]) -> tuple[int, str]:
    """The context window Claude Code assumes for *model_id*, and why, by
    the rules of its model configuration page: a Claude model it knows has
    that model's window; any other id 200K, or 1M with ``[1m]``, or
    ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` when set."""
    lower = model_id.lower()
    no_1m = str(env.get("CLAUDE_CODE_DISABLE_1M_CONTEXT", "")).strip() in (
        "1",
        "true",
    )
    if _CLAUDE_MODEL.search(lower):
        if no_1m:
            return 200_000, "disabled_1m"
        if _NATIVE_1M.search(lower) or "[1m]" in lower:
            return 1_000_000, "claude_1m"
        return 200_000, "claude_200k"
    if "[1m]" in lower and not no_1m:
        return 1_000_000, "1m_suffix"
    declared = str(env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "")).strip()
    if declared.isdecimal() and int(declared) > 0:
        return int(declared), "max_context_tokens"
    return 200_000, "unrecognized"


def served_window(model: str | None) -> tuple[int, str] | None:
    """The model's context window where BenchFlow knows it: LiteLLM's
    model map (``max_input_tokens``), else BenchFlow's provider registry."""
    if not model:
        return None
    from benchflow._utils.session_cost import _price_table
    from benchflow.agents.providers import find_provider, strip_provider_prefix

    bare = strip_provider_prefix(model)
    table = _price_table()
    for key in dict.fromkeys((model, bare, model.split("/", 1)[-1])):
        entry = table.get(key)
        tokens = entry.get("max_input_tokens") if isinstance(entry, dict) else None
        if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0:
            return tokens, "litellm_model_map"
    provider = find_provider(model)
    for entry in getattr(provider[1], "models", None) or [] if provider else []:
        tokens = entry.get("contextWindow") if isinstance(entry, dict) else None
        if entry.get("id") == bare and isinstance(tokens, int) and tokens > 0:
            return tokens, "provider_registry"
    return None


def _context(agent: str, model: str | None, env: dict[str, str]) -> dict:
    """run-card.json's ``context``: the window the harness assumes and the
    one the model serves, where BenchFlow knows them."""
    served = served_window(model)
    context: dict[str, Any] = {
        "harness": None,
        "model_id": None,
        "window": None,
        "window_source": None,
        "compact_window": None,
        "served_window": served[0] if served else None,
        "served_source": served[1] if served else None,
        "under_half": False,
    }
    if agent not in _CLAUDE_CODE or not model:
        return context
    from benchflow.agents.env import (
        uses_direct_anthropic_endpoint,
        uses_native_subscription_auth,
    )
    from benchflow.agents.providers import strip_provider_prefix
    from benchflow.providers.litellm_config import safe_model_alias

    if uses_direct_anthropic_endpoint(
        agent, model, env
    ) or uses_native_subscription_auth(agent, model, env):
        model_id = env.get("ANTHROPIC_MODEL") or strip_provider_prefix(model)
    else:
        # BenchFlow's proxy: Claude Code asks for the route's alias.
        model_id = safe_model_alias(model)
    window, source = claude_code_window(model_id, env)
    if source == "claude_200k" and served and served[0] > window:
        # LiteLLM's map gives some Claude models (Sonnet 4 and 4.5, Opus 4.6)
        # the 1M window Anthropic serves only with its long-context beta,
        # which Claude Code asks for with the [1m] suffix alone and the proxy
        # never adds: without it the model serves the run its 200K.
        served = (window, source)
    compact = str(env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "")).strip()
    context.update(
        harness="Claude Code",
        model_id=model_id,
        window=window,
        window_source=source,
        compact_window=min(window, int(compact))
        if compact.isdecimal() and int(compact) > 0
        else None,
        served_window=served[0] if served else None,
        served_source=served[1] if served else None,
        under_half=bool(served and window < served[0] / 2),
    )
    return context


def _bedrock_default_effort(model: str | None) -> str | None:
    """The thinking effort BenchFlow's proxy gives a Bedrock Claude model
    that takes adaptive thinking, when nothing sets one; else None."""
    if not model or not model.startswith("aws-bedrock/"):
        return None
    from benchflow.agents.providers import strip_provider_prefix
    from benchflow.providers.litellm_config import _bedrock_thinking_effort

    return _bedrock_thinking_effort(strip_provider_prefix(model), {})


class RunCardError(ValueError, UserError):
    """A run-reference.toml that cannot be read, or a strict reference the run
    differs from without saying why."""


@dataclass(frozen=True)
class Reference:
    """A task set's ``run-reference.toml``: the setup its numbers used."""

    path: Path
    source: str
    strict: bool
    notes: str
    values: dict[str, Any]


def find_reference(tasks_root: Path) -> Path | None:
    """The ``run-reference.toml`` beside the run's tasks.

    It sits in the folder that holds the tasks, or one level up when that
    folder is a dataset snapshot's ``tasks/`` or the run's target is one task
    of a set.
    """
    from benchflow.task.discovery import is_task_dir

    folders = [tasks_root]
    if tasks_root.name == "tasks" or is_task_dir(tasks_root):
        folders.append(tasks_root.parent)
    for folder in folders:
        path = folder / REFERENCE_FILE
        if path.is_file():
            return path
    return None


def _kind_name(kinds: type | tuple[type, ...]) -> str:
    if kinds == _NUMBER:
        return "a number"
    if kinds is bool:
        return "true or false"
    if kinds is int:
        return "a whole number"
    return "a string"


def load_reference(path: Path) -> Reference:
    """Read and check ``path``; an unknown key or a wrong type is refused."""
    hint = f"fix {path.name}; docs/evaluate/before-you-run.md#the-run-card-010 lists its keys"
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        error = RunCardError(f"{path}: {exc}")
        error.hint = hint
        raise error from None
    table = raw.get("reference")
    if set(raw) != {"reference"} or not isinstance(table, dict):
        error = RunCardError(f"{path}: put every setting in one [reference] table")
        error.hint = hint
        raise error
    known = {**_CHECKED, **_UNCHECKED, **_META}
    unknown = sorted(set(table) - set(known))
    if unknown:
        error = RunCardError(
            f"{path}: unknown key{'s' if len(unknown) > 1 else ''} "
            f"{', '.join(unknown)} in [reference]; known keys: {', '.join(sorted(known))}"
        )
        error.hint = hint
        raise error
    for key, value in table.items():
        kinds = known[key]
        allowed = kinds if isinstance(kinds, tuple) else (kinds,)
        if (isinstance(value, bool) and bool not in allowed) or not isinstance(
            value, allowed
        ):
            error = RunCardError(
                f"{path}: [reference] {key} must be {_kind_name(kinds)}, not {value!r}"
            )
            error.hint = hint
            raise error
    return Reference(
        path=path,
        source=str(table.get("source", "")),
        strict=bool(table.get("strict", False)),
        notes=str(table.get("notes", "")),
        values={k: v for k, v in table.items() if k not in _META},
    )


def _installs(agent: str) -> str | None:
    """The pinned packages the agent's install command names, if any."""
    from benchflow.agents.registry import AGENTS

    config = AGENTS.get(agent)
    if config is None:
        return None
    pins = []
    for found in _PIN_RE.findall(config.install_cmd):
        # A requirements string written with printf holds several pins and
        # its own quotes ('a==1\nb==2').
        for part in found.replace("\\n", " ").split():
            part = part.strip("'\"")
            if "==" in part or re.search(r"@\d|@[0-9a-f]{7,}", part):
                pins.append(part)
    return " + ".join(dict.fromkeys(pins)) or None


def _selection(cfg: Any) -> dict[str, Any] | None:
    """The run's task selection (-i, -x, --limit, --sample-seed), or None."""
    selection = {
        "include": sorted(cfg.include_tasks or ()),
        "exclude": sorted(cfg.exclude_tasks or ()),
        "limit": cfg.n_tasks,
        "sample_seed": cfg.sample_seed,
    }
    return selection if any(v not in (None, []) for v in selection.values()) else None


def _selection_text(selection: dict[str, Any] | None) -> str:
    """``, selected with -i a -x b --limit 3`` for the Tasks line."""
    if not selection:
        return ""

    def names(flag: str, values: list[str]) -> list[str]:
        shown = [f"{flag} {shlex.quote(v)}" for v in values[:5]]
        return shown + ([f"and {len(values) - 5} more"] if len(values) > 5 else [])

    parts = names("-i", selection.get("include") or []) + names(
        "-x", selection.get("exclude") or []
    )
    if selection.get("limit") is not None:
        parts.append(f"--limit {selection['limit']}")
    if selection.get("sample_seed") is not None:
        parts.append(f"--sample-seed {selection['sample_seed']}")
    return ", selected with " + " ".join(parts)


def describe_vendor_record(record: dict[str, Any]) -> str:
    """The Vendor line from run-card.json's ``model.vendor_setting``."""

    def values(entries: dict[str, Any]) -> str:
        return ", ".join(f"{key} {value:g}" for key, value in entries.items())

    mode = (
        "thinking on, its default" if record.get("thinking_default") else "thinking on"
    )
    return (
        f"{record['vendor']} recommends for {record['covers']}: {mode}, "
        f"{values(record['thinking'])}; with thinking off, "
        f"{values(record['non_thinking'])} ({record['source']}, read {record['read_on']})"
    )


def _thinking_sent(seen: dict[str, Any]) -> bool | None:
    """Whether the requests turned thinking on or off; None when they do not
    say or disagree. ``chat_template_kwargs.enable_thinking`` is vLLM's switch
    for a Qwen model, ``thinking`` Anthropic's, ``reasoning_effort`` OpenAI's."""
    modes: set[bool] = set()
    for encoded in seen.get("chat_template_kwargs", {}):
        value = json.loads(encoded)
        if isinstance(value, dict) and isinstance(value.get("enable_thinking"), bool):
            modes.add(value["enable_thinking"])
    for encoded in seen.get("thinking", {}):
        value = json.loads(encoded)
        kind = value.get("type") if isinstance(value, dict) else value
        if kind in ("enabled", "adaptive", True):
            modes.add(True)
        elif kind in ("disabled", False):
            modes.add(False)
    for encoded in seen.get("reasoning_effort", {}):
        modes.add(json.loads(encoded) not in (None, "none"))
    return modes.pop() if len(modes) == 1 else None


def sampling_check(card: dict[str, Any], seen: dict[str, Any]) -> tuple[str, bool]:
    """One line on the sampling the model requests carried so far (``seen``:
    each key's values and how many requests sent each, from
    ``benchflow.report._sampling``), against the card's Vendor line and its
    reference; and whether anything differs or was left to the server."""
    from benchflow.report import _value

    requests = max((sum(values.values()) for values in seen.values()), default=0)
    sent = {key: [json.loads(v) for v in values] for key, values in seen.items()}
    thinking = _thinking_sent(seen)
    shown = "; ".join(
        f"{key} " + ", ".join(_value(v) for v in values) for key, values in seen.items()
    )
    if thinking is not None:
        shown += f"; thinking {'on' if thinking else 'off'}"
    first = "the first request" if requests == 1 else f"the first {requests} requests"
    parts = [f"Sampling sent so far ({first}): {shown}"]
    differs = False

    def compare(key: str, want: Any) -> str | None:
        got = [v for v in sent.get(key, []) if isinstance(v, int | float)]
        if not got:
            return f"{key} {_shown(want)} (not sent: the server's default applies)"
        if any(float(v) != float(want) for v in got):
            return f"{key} {_shown(want)} (sent {', '.join(_shown(v) for v in got)})"
        return None

    vendor = (card.get("model") or {}).get("vendor_setting")
    if vendor:
        mode = vendor["thinking_default"] if thinking is None else thinking
        notes = [
            note
            for key, want in (
                vendor["thinking"] if mode else vendor["non_thinking"]
            ).items()
            if (note := compare(key, want))
        ]
        differs |= bool(notes)
        parts.append(
            f"Against {vendor['vendor']}'s setting for {vendor['covers']} with "
            f"thinking {'on' if mode else 'off'}: "
            + ("; ".join(notes) if notes else "it matches")
        )
    checks = {
        d["key"]: d["reference"]
        for d in ((card.get("reference") or {}).get("deviations") or [])
        if d.get("kind") == "check"
        and d.get("key") in ("temperature", "top_p", "thinking")
    }
    if checks:
        notes = []
        for key, want in checks.items():
            if key != "thinking":
                note = compare(key, want)
            elif thinking is None:
                note = f"thinking {_shown(want)} (the requests do not say)"
            elif thinking != bool(want):
                note = f"thinking {_shown(want)} (sent {'on' if thinking else 'off'})"
            else:
                note = None
            if note:
                notes.append(note)
        differs |= bool(notes)
        parts.append(
            "Against the reference: " + ("; ".join(notes) if notes else "it matches")
        )
    return ". ".join(parts) + ".", differs


def _vendor_against_reference(
    vendor: dict[str, Any], reference: dict[str, Any] | None
) -> str:
    """A clause when the reference's sampling differs from the vendor's."""
    values = (reference or {}).get("values") or {}
    thinking = values.get("thinking", vendor.get("thinking_default"))
    setting = vendor["thinking"] if thinking else vendor["non_thinking"]
    differ = [
        f"{key} {_shown(values[key])}"
        for key in ("temperature", "top_p")
        if key in values
        and key in setting
        and float(values[key]) != float(setting[key])
    ]
    if not differ:
        return ""
    return (
        f" (the reference asks for {' and '.join(differ)}: follow it to reproduce "
        "the reference's numbers, the vendor's to measure the model as its maker "
        "recommends)"
    )


def _same_model(reference: str, run: str | None) -> bool:
    """Whether a reference's model and the run's are the same: the same full id, or
    the same model part when one of them names no provider. Two providers' ids with
    the same model part (``openrouter/x``, ``together/x``) are two deployments."""
    if run is None:
        return False
    ref, got = str(reference).strip().lower(), str(run).strip().lower()
    if ref == got:
        return True
    if "/" in ref and "/" in got:
        return False
    return ref.rsplit("/", 1)[-1] == got.rsplit("/", 1)[-1]


def _task_facts(
    task_dirs: list[Path],
    multiplier: float | None,
    override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Per task: the agent's time limit after the multiplier, the verifier's,
    the network the agent gets, and the sandbox size. A task that does not
    load is left to the run, which reports it."""
    from benchflow.task.runtime_view import TaskRuntimeView

    facts = []
    for task_dir in task_dirs:
        try:
            config = TaskRuntimeView.from_task_dir(task_dir).config
            if override:
                # The run config's overlay, as each trial applies it.
                from benchflow._utils.config_override import apply_config_override

                config = apply_config_override(config, override)
        except Exception as exc:  # the run itself names a task that does not load
            logger.debug("run card: could not read %s: %s", task_dir, exc)
            continue
        # A step's own agent settings override the task's only where it sets
        # them (task.steps.step_task): each step otherwise has the task's limit.
        limits = (
            [
                step.agent.timeout_sec
                if "timeout_sec" in step.agent.model_fields_set
                else config.agent.timeout_sec
                for step in config.steps
            ]
            if config.steps
            else [config.agent.timeout_sec]
        )
        agent_sec = (
            None
            if any(limit is None for limit in limits)
            else sum(float(limit) for limit in limits if limit is not None)
            * (multiplier or 1.0)
        )
        network = config.agent.network_mode or config.sandbox.network_mode
        facts.append(
            {
                "task": task_dir.name,
                # The task's own trial budget ([agent] token_budget,
                # turn_budget), which a run's measure replaces.
                "token_budget": config.agent.token_budget,
                "turn_budget": config.agent.turn_budget,
                "agent_sec": agent_sec,
                "verifier_sec": float(config.verifier.timeout_sec),
                # A multi-step trial runs a soft check after each step, with
                # that step's verifier limit, then the final check of the last.
                "verifier_total_sec": _verifier_total(config),
                "network": str(network.value if hasattr(network, "value") else network),
                "cpus": config.sandbox.cpus,
                "memory_mb": config.sandbox.memory_mb,
                "steps": len(config.steps or ()),
                # Under mean, the steps before the last score from soft checks,
                # which run before the workspace is frozen (task/steps.py).
                "mean_steps": bool(config.steps)
                and str(getattr(config, "multi_step_reward_strategy", None) or "final")
                .split(".")[-1]
                .lower()
                == "mean",
            }
        )
    return facts


def _verifier_total(config: Any) -> float:
    if not config.steps:
        return float(config.verifier.timeout_sec)
    limits = [
        float(
            step.verifier.timeout_sec
            if "timeout_sec" in step.verifier.model_fields_set
            else config.verifier.timeout_sec
        )
        for step in config.steps
    ]
    return sum(limits) + limits[-1]


def _spread(values: list[float]) -> dict[str, float] | None:
    return {"min": min(values), "max": max(values)} if values else None


def _trial_budget(
    cfg: Any, model: str | None, facts: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """run-card.json's ``trial_budget``: each trial's token and turn budget
    (benchflow.trial_budget) over the tasks, where it came from, and why it
    could not be enforced; None when no trial has one."""
    from benchflow.trial_budget import (
        TrialBudget,
        effective_budget,
        unenforceable_reason,
    )

    run = getattr(cfg, "trial_budget", None)
    budgets = []
    sources: set[str] = set()
    for fact in facts:
        task = TrialBudget.coerce(
            {"tokens": fact.get("token_budget"), "turns": fact.get("turn_budget")}
        )
        budget, where = effective_budget(run, task)
        budgets.append(budget)
        sources.update(where.values())
    if not any(budgets):
        return None
    return {
        "run": run.to_dict() if run is not None else None,
        "tokens": _spread([b.tokens for b in budgets if b and b.tokens is not None]),
        "turns": _spread([b.turns for b in budgets if b and b.turns is not None]),
        "tasks_without": sum(1 for b in budgets if b is None),
        "sources": sorted(sources),
        "unenforceable": None
        if cfg.agent in CONTROL_AGENTS
        else unenforceable_reason(cfg.agent, model, dict(cfg.agent_env or {})),
    }


def _subfolders(folder: Path) -> list[Path]:
    try:
        return sorted(p for p in folder.iterdir() if p.is_dir())
    except OSError:
        return []


def _sibling_folders(folder: Path) -> list[Path]:
    return [p for p in _subfolders(folder.parent) if p != folder]


def _control_roots(jobs_dir: Path) -> list[Path]:
    """The folders whose job folders may hold this run's controls.

    The jobs folder and the folders beside it (``jobs/oracle`` beside
    ``jobs/claude``). A trial job of ``--trials`` sits in
    ``<jobs dir>/<alias>/trial-NN``: from the run's own jobs folder, the
    folders beside it, the alias folders in it, and their trial folders (a
    matrix's ``{agent: oracle}`` alias)."""
    roots = [jobs_dir, *_sibling_folders(jobs_dir)]
    if _TRIAL_DIR.fullmatch(jobs_dir.name):
        base = jobs_dir.parent.parent
        roots += [base, *_sibling_folders(base)]
        for alias in _subfolders(base):
            roots += [alias, *_subfolders(alias)]
    return list(dict.fromkeys(roots))


def _control_jobs(jobs_dir: Path) -> list[tuple[Path, str, str | None]]:
    """Oracle and nop jobs under :func:`_control_roots`: folders whose job
    record names one of them, with the BenchFlow version that wrote it. At most
    :data:`_MAX_SCANNED` folders are read."""
    from benchflow.evaluation import EVALUATION_RECORD

    found: list[tuple[Path, str, str | None]] = []
    scanned = 0
    for root in _control_roots(jobs_dir):
        for job in _subfolders(root):
            scanned += 1
            if scanned > _MAX_SCANNED:
                return found
            record_path = job / EVALUATION_RECORD
            if not record_path.is_file():
                continue
            try:
                record = json.loads(record_path.read_text())
                agent = record.get("config", {}).get("agent")
                version = record.get("benchflow_version")
            except (OSError, ValueError, AttributeError):
                continue
            if agent in CONTROL_AGENTS:
                found.append(
                    (job, str(agent), version if isinstance(version, str) else None)
                )
    return found


def _controls(evaluation: Evaluation, task_dirs: list[Path]) -> dict[str, Any]:
    """What the oracle and nop scored on these tasks in earlier jobs.

    A control trial counts when its task has this run's name and, for up to
    :data:`_MAX_DIGEST_TASKS` tasks, the same content digest: a control run on
    an older version of a task says nothing about this one.
    """
    from benchflow.evaluation import (
        EVALUATION_RECORD,
        _benchflow_version,
        _task_digest_for,
    )
    from benchflow.jobs import load_job

    cfg = evaluation._config
    if cfg.agent in CONTROL_AGENTS:
        return {"this_run": cfg.agent}
    by_name = {d.name: d for d in task_dirs}
    check_digest = len(task_dirs) <= _MAX_DIGEST_TASKS
    digests: dict[str, str | None] = {}
    best: dict[str, dict[str, Any]] = {}
    # A control from this BenchFlow first (another release may grade the same
    # task content otherwise), then the most matching tasks, then the newest:
    # an older control must not hide a later one that disagrees.
    best_rank: dict[str, tuple[bool, int, float]] = {}
    this_version = _benchflow_version()
    for job_dir, agent, version in _control_jobs(evaluation._jobs_dir):
        try:
            job = load_job(job_dir)
        except Exception as exc:  # a broken control job is no evidence
            logger.debug("run card: could not read control job %s: %s", job_dir, exc)
            continue
        rewards: dict[str, float | None] = {}
        for trial in job.trials:
            name = trial.result.task_name
            if name not in by_name:
                continue
            if check_digest:
                if name not in digests:
                    digests[name] = _task_digest_for(by_name[name], cfg)
                recorded = trial.config.get("task_digest")
                if digests[name] is None or recorded != digests[name]:
                    continue
            rewards[name] = trial.reward
        if not rewards:
            continue
        target = 1.0 if agent == "oracle" else 0.0
        entry = {
            "job": str(job_dir),
            "tasks": len(rewards),
            "of": len(task_dirs),
            "as_expected": sum(1 for r in rewards.values() if r == target),
            "expected": target,
            "matched_by": "digest" if check_digest else "name",
            "benchflow_version": version,
        }
        try:
            recorded = (job_dir / EVALUATION_RECORD).stat().st_mtime
        except OSError:
            recorded = 0.0
        rank = (version == this_version, entry["tasks"], recorded)
        if agent not in best or rank > best_rank[agent]:
            best[agent], best_rank[agent] = entry, rank
    return {agent: best.get(agent) for agent in CONTROL_AGENTS}


def _compare(
    reference: Reference, run: dict[str, Any], facts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Where the run differs from the reference, and what it cannot check."""
    deviations: list[dict[str, Any]] = []

    def differs(key: str, ref: Any, got: Any) -> None:
        deviations.append({"key": key, "reference": ref, "run": got, "kind": "differs"})

    values = reference.values
    for key in ("agent", "harness"):
        if key in values and str(values[key]).lower() != str(run[key]).lower():
            differs(key, values[key], run[key])
    if "model" in values and not _same_model(values["model"], run["model"]):
        differs("model", values["model"], run["model"] or "none")
    if (
        "effort" in values
        and str(values["effort"]).lower() != str(run["effort"] or "").lower()
    ):
        differs("effort", values["effort"], run["effort"] or "the agent's default")
    if "trials" in values and values["trials"] != run["trials"]:
        differs("trials", values["trials"], run["trials"])
    if "agent_timeout_sec" in values:
        ref = float(values["agent_timeout_sec"])
        got = sorted({f["agent_sec"] for f in facts}, key=lambda v: (v is None, v or 0))
        if got != [ref]:
            differs(
                "agent_timeout_sec",
                ref,
                ", ".join("no limit" if v is None else f"{v:g}" for v in got),
            )
    if "network" in values:
        got = sorted({f["network"] for f in facts})
        if got != [str(values["network"])]:
            differs("network", values["network"], ", ".join(got))
    if "verifier_sandbox" in values and str(values["verifier_sandbox"]) != str(
        run["verifier_sandbox"]
    ):
        differs("verifier_sandbox", values["verifier_sandbox"], run["verifier_sandbox"])
    for key in _UNCHECKED:
        if key in values:
            deviations.append(
                {"key": key, "reference": values[key], "run": None, "kind": "check"}
            )
    if run.get("selection"):
        # The reference's numbers are over a set this run does not run whole.
        deviations.append(
            {
                "key": "tasks",
                "reference": "the whole set",
                "run": _selection_text(run["selection"]).removeprefix(
                    ", selected with "
                ),
                "kind": "check",
            }
        )
    return deviations


def build_run_card(evaluation: Evaluation, task_dirs: list[Path]) -> dict[str, Any]:
    """The card for ``evaluation`` over ``task_dirs`` (the selected tasks)."""
    from benchflow.agents.registry import agent_routes_own_model, is_operator_script
    from benchflow.evaluation import (
        _benchflow_version,
        effective_model,
        model_from_agent_env,
    )
    from benchflow.task.verifier_sandbox import effective_verifier_sandbox

    cfg = evaluation._config
    facts = _task_facts(task_dirs, cfg.timeout_multiplier, cfg.config_override)
    model = effective_model(cfg.agent, cfg.model, cfg.agent_env)
    # "model": the run named it (--model, the config's model, a matrix
    # entry), even when agent_env names the same one; else the agent_env
    # variable it came from, else the agent's default. The plan resolves the
    # model before the job starts, so cfg.model alone cannot tell.
    given = bool(cfg.model_given)
    env_model, env_key = model_from_agent_env(cfg.agent, cfg.agent_env)
    if model is not None and given:
        model_source = "model"
    elif env_key is not None and env_model == model:
        model_source = env_key
    else:
        model_source = "default"
    per_task = max(1, int(cfg.planned_trials or 1))
    total = len(task_dirs) * per_task
    agent_limits = [f["agent_sec"] for f in facts if f["agent_sec"] is not None]
    verifier_limits = [f["verifier_sec"] for f in facts]
    trial_limits = [
        f["agent_sec"] + f["verifier_total_sec"]
        for f in facts
        if f["agent_sec"] is not None
    ]
    untimed = sum(1 for f in facts if f["agent_sec"] is None)
    # With --trials N each trial is a job of its own, and the jobs run one
    # after another: concurrency applies within a job.
    rounds = math.ceil(len(task_dirs) / max(1, cfg.concurrency)) * per_task
    provenance = cfg.source_provenance or {}
    settings = (
        []
        if cfg.agent in CONTROL_AGENTS or is_operator_script(cfg.agent)
        else env_settings(cfg.agent, model, dict(cfg.agent_env or {}))
    )
    # The output cap BenchFlow gives Claude Code (agents/env.py), or the
    # run's own; no other agent reads it.
    cap = next(
        (s for s in settings if s["name"] == "CLAUDE_CODE_MAX_OUTPUT_TOKENS"), None
    )
    run = {
        "agent": cfg.agent,
        "harness": cfg.harness,
        "model": model,
        "effort": cfg.reasoning_effort,
        "trials": per_task,
        "verifier_sandbox": effective_verifier_sandbox(
            cfg.verifier_sandbox, cfg.separate_verifier
        ),
        "selection": _selection(cfg),
    }
    card: dict[str, Any] = {
        "run_card": SCHEMA_VERSION,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "benchflow_version": _benchflow_version(),
        "tasks": {
            "count": len(task_dirs),
            "root": str(evaluation._tasks_dir),
            "repo": provenance.get("repo"),
            "path": provenance.get("path"),
            "commit": provenance.get("resolved_sha"),
            "dataset": (
                f"{cfg.dataset_name}@{cfg.dataset_version}"
                if cfg.dataset_name
                else None
            ),
            "first": [d.name for d in task_dirs[:5]],
            # How the tasks were picked from the folder, when not all of it.
            "selection": _selection(cfg),
            "multi_step": sum(1 for f in facts if f["steps"]),
            "multi_step_mean": sum(1 for f in facts if f.get("mean_steps")),
        },
        "agent": {
            "name": cfg.agent,
            "harness": cfg.harness,
            "installs": _installs(cfg.agent),
            "env_keys": sorted(cfg.agent_env or {}),
            "direct_endpoint": uses_direct_anthropic_endpoint(
                cfg.agent,
                model,
                inherit_agent_auth_env(cfg.agent, model, cfg.agent_env),
            ),
            "output_cap": cap["value"] if cap else None,
            "output_cap_source": cap["source"] if cap else None,
        },
        # Every variable BenchFlow knows that changes what a trial measures,
        # with its value and where it came from (env_settings).
        "settings": settings,
        # The context window the harness assumes, beside the model's.
        "context": (
            None
            if cfg.agent in CONTROL_AGENTS or is_operator_script(cfg.agent)
            else _context(
                cfg.agent,
                model,
                inherit_agent_auth_env(cfg.agent, model, cfg.agent_env),
            )
        ),
        "model": {
            "id": model,
            "given": model_source == "model",
            "source": model_source,
            "effort": cfg.reasoning_effort,
            "sampling": None,
            # model_route = "agent": the agent calls its own model, if any, so no
            # model id is BenchFlow's to give.
            "routes_own": agent_routes_own_model(cfg.agent),
            # The vendor's published sampling, where BenchFlow has it on file.
            "vendor_setting": (
                setting.as_record() if (setting := vendor_setting(model)) else None
            ),
        },
        "trials": {
            "per_task": per_task,
            "this_trial": cfg.trial_number,
            "total": total,
            "seeds": cfg.seeds,
            # Attempts after a retryable failure (RetryConfig.should_retry); some can follow the agent's actions.
            "retries": int(getattr(getattr(cfg, "retry", None), "max_retries", 0) or 0),
        },
        "sandbox": {
            "backend": cfg.environment,
            "concurrency": cfg.concurrency,
            "cpus": _spread([f["cpus"] for f in facts]),
            "memory_mb": _spread([f["memory_mb"] for f in facts]),
        },
        "limits": {
            "agent_sec": _spread(agent_limits),
            "untimed_tasks": untimed,
            "multiplier": cfg.timeout_multiplier,
            "idle_sec": cfg.agent_idle_timeout,
            # given, or BenchFlow's default: "default", "effort_max", ...
            "idle_source": cfg.idle_timeout_source(),
            "verifier_sec": _spread([f["verifier_sec"] for f in facts]),
        },
        "network": dict(sorted(Counter(f["network"] for f in facts).items())),
        "grading": {
            "verifier_sandbox": run["verifier_sandbox"],
            "integrity": cfg.integrity,
        },
        "controls": _controls(evaluation, task_dirs),
        "estimate": {
            "rounds": rounds,
            "longest_agent_sec": max(agent_limits) if agent_limits else None,
            "longest_verifier_sec": max(verifier_limits) if verifier_limits else None,
            # Setup and image builds are not counted.
            "longest_trial_sec": max(trial_limits) if trial_limits else None,
            "max_wall_sec": (
                rounds * max(trial_limits) if trial_limits and not untimed else None
            ),
        },
        "budget": (
            {k: v for k, v in dataclasses.asdict(cfg.budget).items() if v is not None}
            if cfg.budget is not None
            else None
        ),
        # Each trial's own token and turn budget (benchflow.trial_budget).
        "trial_budget": _trial_budget(cfg, model, facts),
        "extra_instruction": cfg.extra_instruction is not None,
        # --accept-deviations, kept whether or not the tasks name a reference.
        "accepted_deviations": cfg.accept_deviations,
        "reference": None,
    }
    ref_path = find_reference(evaluation._tasks_dir)
    if ref_path is not None:
        reference = load_reference(ref_path)
        card["reference"] = {
            "file": str(ref_path),
            "source": reference.source,
            "strict": reference.strict,
            "notes": reference.notes,
            "values": reference.values,
            # A control checks the tasks, not an agent: never compared.
            "compared": cfg.agent not in CONTROL_AGENTS,
            "deviations": (
                _compare(reference, run, facts)
                if cfg.agent not in CONTROL_AGENTS
                else []
            ),
            "accepted": cfg.accept_deviations,
        }
    return card


def check_card(card: dict[str, Any]) -> None:
    """Refuse a run whose trial budget its agent's route cannot enforce, and
    one that differs from a strict reference and says nothing."""
    budget = card.get("trial_budget") or {}
    if budget.get("unenforceable"):
        agent = (card.get("agent") or {}).get("name")
        error = RunCardError(
            f"The trial budget cannot be enforced on {agent}: "
            f"{budget['unenforceable']}, and BenchFlow's model proxy is what counts "
            "a trial's tokens and turns"
        )
        error.hint = (
            "run it with an API key, so its calls go through the proxy, or drop "
            "--token-budget/--turn-budget (and the tasks' [agent] budget)"
        )
        raise error
    reference = card.get("reference")
    if not reference or not reference["strict"] or reference["accepted"]:
        return
    differs = [d for d in reference["deviations"] if d["kind"] == "differs"]
    if not differs:
        return
    where = reference["source"] or Path(reference["file"]).name
    error = RunCardError(
        f"This run differs from the setup in {reference['file']} ({where}), which "
        "asks every run to match it or say why: "
        + "; ".join(
            f"{d['key']} {d['run']} (reference {d['reference']})" for d in differs
        )
    )
    error.hint = (
        "match the reference, or pass --accept-deviations '<why>' "
        "(accept_deviations: in a run config); the note is kept in run-card.json"
    )
    raise error


def _count(n: int, noun: str, plural: str | None = None) -> str:
    return f"{n} {noun if n == 1 else (plural or noun + 's')}"


def _duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:g} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _limit_text(spread: dict[str, float] | None) -> str:
    if spread is None:
        return "none"
    low, high = _duration(spread["min"]), _duration(spread["max"])
    return low if low == high else f"{low} to {high}"


def _control_jobs_flag(job_dir: Path, agent: str) -> str:
    """``--jobs-dir`` for a control run that the next card finds.

    The card searches the run's jobs folder and the folders beside it, so a
    control goes beside them (``rc-jobs/oracle`` beside ``rc-jobs/claude``).
    In the default ``jobs`` folder a control run needs no flag: it starts a
    job of its own there."""
    jobs_dir = job_dir.parent
    if _TRIAL_DIR.fullmatch(jobs_dir.name):
        jobs_dir = jobs_dir.parent.parent
    with contextlib.suppress(OSError):
        if jobs_dir.resolve() == Path("jobs").resolve():
            return ""
    folder = jobs_dir.parent / agent
    with contextlib.suppress(OSError, ValueError):
        folder = folder.resolve().relative_to(Path.cwd().resolve())
    return f" --jobs-dir {shlex.quote(str(folder))}"


def _control_text(
    agent: str,
    entry: dict[str, Any] | None,
    target: str,
    jobs_flag: str = "",
    version: str | None = None,
) -> str:
    if entry is None:
        return f"{agent} none found: bench run {target} --agent {agent}{jobs_flag}"
    expected = "1" if entry["expected"] == 1.0 else "0"
    # A control job of another dataset can count when it ran these tasks' bytes, which the digest shows.
    unchecked = (
        f", matched by name: content not checked above {_MAX_DIGEST_TASKS} tasks"
        if entry.get("matched_by") == "name"
        else ", same task digests"
    )
    other = entry.get("benchflow_version")
    if other and version and other != version:
        # Another release can grade the same task content otherwise (0.10
        # keeps a repository's conftest.py that 0.9 removed).
        unchecked += f", run with BenchFlow {other}: rerun it on {version}"
    missed = entry["tasks"] - entry["as_expected"]
    shortfall = (
        f"; {missed} did not: those tasks measure nothing as they are" if missed else ""
    )
    return (
        f"{agent} on {entry['tasks']} of {entry['of']} tasks, {entry['as_expected']} "
        f"scored {expected} ({Path(entry['job']).parent.name}/{Path(entry['job']).name}"
        f"{unchecked}){shortfall}"
    )


def card_lines(card: dict[str, Any], job_dir: Path) -> list[str]:
    """The card as printed: one line per part, then the reference's findings."""
    from benchflow.agents.registry import is_operator_script

    tasks, agent, model = card["tasks"], card["agent"], card["model"]
    trials, sandbox, limits = card["trials"], card["sandbox"], card["limits"]
    commit = str(tasks["commit"] or "")
    if tasks["repo"] and commit.startswith("sha256:"):
        # A BenchFlow Hub dataset: the "commit" is the version's dataset digest.
        where = f"{tasks['dataset'] or tasks['repo']} on the Hub at {tasks['repo']}, digest {commit[:19]}"
    elif tasks["repo"]:
        where = "/".join(p for p in (tasks["repo"], tasks["path"]) if p)
        if commit:
            where += f" at {commit[:8]}"
    elif tasks["dataset"]:
        where = tasks["dataset"]
    else:
        where = tasks["root"]
    if tasks["repo"] and commit.startswith("sha256:") and tasks["dataset"]:
        target = f"-d {tasks['dataset']} --hub {tasks['repo']}"
    elif tasks["repo"]:
        target = "/".join(p for p in (tasks["repo"], tasks["path"]) if p)
    else:
        target = f"-d {tasks['dataset']}" if tasks["dataset"] else tasks["root"]
    multi = (
        f" ({tasks['multi_step']} multi-step: each step's min_reward gate is a soft "
        "check run before the workspace is frozen"
        + (
            f"; {tasks['multi_step_mean']} scored by mean, from those soft checks"
            if tasks.get("multi_step_mean")
            else ""
        )
        + ". Use final where a reward must resist tampering, and a workspace other "
        "than /root, where a root soft check can leave a process the final check "
        "does not stop)"
        if tasks["multi_step"]
        else ""
    )
    lines = [
        f"Run card ({job_dir / RUN_CARD}):",
        f"  Tasks:     {tasks['count']} from {where}{_selection_text(tasks.get('selection'))}"
        f"{multi}",
    ]
    script = is_operator_script(agent["name"])
    if agent["name"] in _CONTROL_ROLES:
        lines.append(f"  Agent:     {agent['name']}, {_CONTROL_ROLES[agent['name']]}")
    elif script:
        lines.append(
            f"  Agent:     {agent['name']}, an operator script, run as the sandbox user with no harness"
        )
    else:
        lines.append(
            f"  Agent:     {agent['name']}, harness {agent['harness']}"
            + (f", installs {agent['installs']}" if agent["installs"] else "")
        )
    if card["controls"].get("this_run"):
        lines.append("  Model:     none (a control run)")
    elif script:
        lines.append("  Model:     none (an operator script runs no model)")
    else:
        effort = model["effort"] or "the agent's default"
        source = model.get("source", "default")
        default = (
            ""
            if model["given"]
            else (
                " (the agent's default)" if source == "default" else f" (from {source})"
            )
        )
        if not model["id"] and model.get("routes_own"):
            default = " set (the agent picks its own, if it calls one)"
        lines.append(f"  Model:     {model['id'] or 'none'}{default}; effort: {effort}")
        cap = agent.get("output_cap")
        cap_source = SOURCE_WORDS.get(str(agent.get("output_cap_source")))
        lines.append(
            "  Sampling:  the agent's own; BenchFlow sets no temperature or top_p for "
            "agent trials"
            + (
                f" and caps Claude Code's output at {cap} tokens "
                "(CLAUDE_CODE_MAX_OUTPUT_TOKENS)"
                + (f", {cap_source}" if cap_source else "")
                if cap
                else " and no output cap"
            )
            + "; each proxied request's sampling is in the trial's "
            "trajectory/llm_trajectory.jsonl"
        )
        context = card.get("context") or {}
        if context:
            lines.append("  Context:   " + context_text(context))
        settings = card.get("settings") or []
        lines.append(
            "  Settings:  "
            + (
                "; ".join(
                    f"{s['name']} {s['value']} ({SOURCE_WORDS.get(s['source'], s['source'])})"
                    for s in settings
                )
                if settings
                else "none of the variables BenchFlow knows to change what a trial "
                "measures is set"
            )
        )
        if agent.get("direct_endpoint"):
            lines.append(
                "  Capture:   direct endpoint; no proxy call capture or cost accounting"
            )
        vendor = model.get("vendor_setting")
        lines.append(
            "  Vendor:    "
            + (
                f"{describe_vendor_record(vendor)}: set the agent to match, or say why"
                + _vendor_against_reference(vendor, card.get("reference"))
                if vendor
                else f"no recommended setting on file for {model['id'] or 'this model'}: "
                "read its model card before you compare"
            )
        )
    repeats = trials["per_task"]
    repeat_note = (
        f" (--trials {repeats} runs the task set {repeats} times; each repeat is a "
        "job of its own"
        + (
            f", and this is repeat {trials['this_trial']} of {repeats}, folder "
            f"trial-{trials['this_trial']:02d}"
            if trials["this_trial"]
            else ""
        )
        + ")"
        if repeats > 1
        else ""
    )
    lines.append(
        f"  Trials:    {trials['total']} in all, {_count(repeats, 'per task', 'per task')}"
        f"{repeat_note}; {sandbox['concurrency']} at a time on {sandbox['backend']}"
    )
    retries = int(trials.get("retries") or 0)
    lines.append(
        "  Retries:   none (--retry-attempts 0)"
        if not retries
        else f"  Retries:   up to {retries} per trial on a retryable failure, which can come after the agent "
        "acted (an idle timeout, an ACP error, a verifier that never started); --retry-attempts 0 for none"
    )
    untimed = (
        f", {_count(limits['untimed_tasks'], 'task')} with no agent limit"
        if limits["untimed_tasks"]
        else ""
    )
    idle = "off" if not limits["idle_sec"] else _duration(float(limits["idle_sec"]))
    idle_source = str(limits.get("idle_source") or "")
    if idle_source == "given":
        idle += " (the run's)"
    elif idle_source == "default":
        idle += " (BenchFlow's default)"
    elif idle_source.startswith("effort_"):
        effort = idle_source.removeprefix("effort_")
        idle += f" (BenchFlow's default at effort {effort})"
    summed = (
        " (a multi-step task's is the sum of its steps')" if tasks["multi_step"] else ""
    )
    lines.append(
        f"  Limits:    agent {_limit_text(limits['agent_sec'])} per task{summed}"
        f"{untimed}; idle {idle}; verifier {_limit_text(limits['verifier_sec'])}"
    )
    network = ", ".join(
        f"{mode} ({n} task{'' if n == 1 else 's'})"
        for mode, n in card["network"].items()
    )
    lines.append(f"  Network:   {network or 'unknown'}")
    grading = card["grading"]
    lines.append(
        "  Grading:   "
        + (
            "every verifier in a separate sandbox"
            if grading["verifier_sandbox"] == "separate"
            else "each task's own verifier setting"
        )
        + "; the final check hardened (the agent stopped, its start-up hooks removed)"
        + (
            f"; integrity {grading['integrity']} (--integrity)"
            if grading["integrity"] not in (None, "off")
            else "; no integrity audit (--integrity audit records and checks what "
            "the agent did)"
        )
    )
    controls = card["controls"]
    if controls.get("this_run"):
        lines.append(f"  Controls:  this run is the {controls['this_run']} control")
    else:
        # Read from job folders on this machine; a Hub dataset's own controls are what the Hub reports.
        on_hub = bool(tasks["repo"]) and commit.startswith("sha256:")
        lines.append(
            "  Controls:  "
            + ("local jobs, not the Hub: " if on_hub else "local jobs: ")
            + "; ".join(
                _control_text(
                    agent_name,
                    controls.get(agent_name),
                    target,
                    _control_jobs_flag(job_dir, agent_name),
                    card.get("benchflow_version"),
                )
                for agent_name in CONTROL_AGENTS
            )
        )
    estimate = card["estimate"]
    if estimate["max_wall_sec"] is not None:
        bound = _duration(estimate["max_wall_sec"])
        if trials["per_task"] > 1:
            lines.append(
                f"  Time:      about {bound} if each trial runs once to its limits: "
                f"{trials['per_task']} repeats one after another, each "
                f"{trials['total'] // trials['per_task']} trials "
                f"{sandbox['concurrency']} at a time (setup, image builds, retries and "
                "rubric reviews not counted)"
            )
        else:
            lines.append(
                f"  Time:      about {bound} if each trial runs once to its limits: "
                f"{trials['total']} trial{'' if trials['total'] == 1 else 's'}, "
                f"{sandbox['concurrency']} at a time (setup, image builds, retries and "
                "rubric reviews not counted)"
            )
    else:
        lines.append("  Time:      no estimate: a task has no agent limit")
    budget = card["budget"]
    per_job = (
        f" per repeat ({trials['per_task']} jobs)" if trials["per_task"] > 1 else ""
    )
    lines.append(
        "  Budget:    "
        + (
            ", ".join(f"{k} {v:g}" for k, v in budget.items()) + per_job
            if budget
            else "none (--max-tokens caps a job's tokens; --max-cost-usd its cost, for "
            "a model BenchFlow has a price for"
            + ("; each repeat is a job of its own)" if trials["per_task"] > 1 else ")")
        )
    )
    lines.append("  Per trial: " + trial_budget_text(card.get("trial_budget")))
    reference = card["reference"]
    if reference:
        differs = [d for d in reference["deviations"] if d["kind"] == "differs"]
        checks = [d for d in reference["deviations"] if d["kind"] == "check"]
        name = reference["source"] or Path(reference["file"]).name
        if not reference["compared"]:
            lines.append(f"  Reference: {name}: a control run is not compared")
        else:
            lines.append(
                f"  Reference: {name}: "
                + (
                    f"{_count(len(differs), 'setting')} "
                    + ("differs" if len(differs) == 1 else "differ")
                    if differs
                    else "no setting differs"
                )
                + f", {len(checks)} to check yourself"
                + (" (strict)" if reference["strict"] else "")
            )
    elif card.get("accepted_deviations"):
        # The note --accept-deviations gave, kept with no reference to hold it.
        lines.append(f"  Note:      {card['accepted_deviations']}")
    return lines


def _amount(spread: dict[str, Any] | None, unit: str) -> str:
    if spread is None:
        return ""
    low, high = int(spread["min"]), int(spread["max"])
    shown = f"{low:,}" if low == high else f"{low:,} to {high:,}"
    return f"{shown} {unit}" + ("" if (low == high == 1) else "s")


def trial_budget_text(block: dict[str, Any] | None) -> str:
    """The card's Per trial line, after its label, from run-card.json's
    ``trial_budget``."""
    if not block:
        return (
            "no token or turn budget (--token-budget and --turn-budget cap each "
            "trial's tokens and model calls, subagents' included)"
        )
    amounts = " and ".join(
        part
        for part in (
            _amount(block.get("tokens"), "token"),
            _amount(block.get("turns"), "turn"),
        )
        if part
    )
    where = {
        ("run",): " (the run's)",
        ("task",): " (the tasks' own)",
    }.get(tuple(block.get("sources") or ()), " (the run's and the tasks' own)")
    without = int(block.get("tasks_without") or 0)
    return (
        f"budget {amounts}{where}, a turn being one model call; the model proxy "
        "counts every call of the trial, subagents' and side calls included, "
        "refuses the next once a budget is spent, and the agent is stopped and "
        "graded" + (f"; {_count(without, 'task')} with no budget" if without else "")
    )


def context_text(context: dict[str, Any]) -> str:
    """The card's Context line, after its label."""
    served = context.get("served_window")
    served_text = (
        f"the model serves {served} ({WINDOW_WORDS[context['served_source']]})"
        if served
        else "BenchFlow does not know the window the model serves"
    )
    window = context.get("window")
    if not window:
        return served_text + "; BenchFlow does not know the agent's own window"
    text = (
        f"{context['harness']} assumes {window} tokens "
        f"({WINDOW_WORDS[context['window_source']]})"
    )
    if context.get("compact_window"):
        text += f" and compacts at {context['compact_window']}"
    text += "; " + served_text
    if context.get("under_half"):
        text += ": under half, so it compacts early"
        if context.get("window_source") == "unrecognized":
            text += (
                f"; --agent-env CLAUDE_CODE_MAX_CONTEXT_TOKENS={served} gives it "
                "the model's window"
            )
    return text


def _shown(value: Any) -> str:
    """A reference value as the TOML file spells it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{value:g}" if isinstance(value, float) else str(value)


def reference_findings(card: dict[str, Any]) -> list[str]:
    """One line per deviation, for the warning the run logs."""
    reference = card.get("reference")
    if not reference:
        return []
    found = []
    for d in reference["deviations"]:
        if d["kind"] == "differs":
            found.append(
                f"differs: {d['key']} {_shown(d['run'])} (reference {_shown(d['reference'])})"
            )
        else:
            installs = (card.get("agent") or {}).get("installs")
            if d["key"] == "tasks":
                found.append(
                    f"check: this run selects tasks ({d['run']}); the reference's "
                    "numbers may be over the whole set"
                )
                continue
            if d["key"] != "agent_version":
                how = (
                    "BenchFlow does not set it for agent trials: compare it with the "
                    "agent's own settings or its trajectory"
                )
            elif installs:
                how = f"this run installs {installs}: compare the two"
            else:
                how = "the agent's install has no pinned version: check its trial logs"
            found.append(
                f"check: {d['key']} {_shown(d['reference'])} in the reference; {how}"
            )
    if reference["accepted"]:
        found.append(f"accepted: {reference['accepted']}")
    return found


def write_card(card: dict[str, Any], job_dir: Path) -> Path:
    path = job_dir / RUN_CARD
    path.write_text(json.dumps(card, indent=2, default=str) + "\n")
    return path
