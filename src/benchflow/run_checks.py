"""What a run's model calls carried and how its trials ended.

The run card (:mod:`benchflow.run_card`) says what a run asks for before it
starts. This module checks what happened against it.

Per trial, result.json's ``agent_outcome`` gets how each prompt turn ended
(:func:`turn_endings`): every turn's ACP stop reason, the finish reasons of the
trial's model calls, and whether the trial was cut at the output cap. A trial
is truncated when its agent's last call ended at the cap, its last prompt
ended ``max_tokens``, or the agent's error says the cap was exceeded; it is
not when its ending is known otherwise, and unknown when nothing says.

Per job, :class:`RunChecks` reads each trial's result.json as the trial
finishes, and the result of each attempt the retry policy ran again. The
truncation rate is over the attempts with a known ending: at
:data:`WARN_TRUNCATION` it warns, and at :data:`REFUSE_TRUNCATION`, once at
least :data:`MIN_TRIALS` such attempts have finished, the job starts no more
trials (the running ones finish, as at a budget's rollout cap), and the run
fails. ``summary.json`` and ``run-card.json`` record the figures and why
(:meth:`RunChecks.summary`, :func:`after_run_lines`).

A run of several jobs, the repeats of ``--trials`` or the workers of a sharded
run, is judged on the figures of all of them (:func:`pool`): one job's own
refusal waits for five of its own attempts, as any job's does, and the run's
for five over its jobs.

Per job, the trials' token and turn budgets (result.json's ``trial_budget``,
:mod:`benchflow.trial_budget`) are totalled: the trials that had one, those it
stopped and by which measure, the most tokens and turns a trial used, and the
subagents' share of the tokens where the agents' own logs say
(:meth:`RunChecks.trial_budget`).

Per trial, result.json's ``settings_sent`` counts what the agent's main calls
carried (:func:`settings_sent`): the calls to the run's own model, apart from
side calls to another model (Claude Code's background calls) and subagents'
calls, which are counted, not compared. Per job, each setting the run asked
for (:func:`requested_settings`: the model, the effort, an output cap, a
sampling value) is compared with them. A trial in which more than half the
calls that carried the setting carried another value (:data:`CLEAR_SHARE`), or
whose largest output cap sent is not the one asked for, shows a clear
mismatch: the job stops after that trial and the run fails. Other calls with
another value fail the run when they are more than 1% of the job's calls that
carried the setting (:data:`MISMATCH_SHARE`), judged once its trials have
finished, so a few odd calls never stop a job. A setting no main call carried
is a warning: the provider's default applied. BenchFlow's own default output
cap is compared only for a warning: Claude Code lowers it to the model's own
cap. So is the model of a route that names a deployment or an alias
(:data:`ALIAS_ROUTES`), which the provider answers with the model's own id.
An effort the model proxy's route sets on each request (result.json's
``effort.route``) is the effort that ran: the calls are not compared with
the run's, and a route effort other than the run's is a warning.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchflow._utils.scoring import is_output_cap_error
from benchflow.trajectories.model_calls import (
    _TRUNCATED,
    CALLS_FILE,
    CLAUDE_CODE_OTEL,
    PROXY,
)

logger = logging.getLogger(__name__)

#: The truncation rates that warn and that refuse a run: where a widely used
#: evaluation library's dashboard turns a rate yellow and red.
WARN_TRUNCATION = 0.10
REFUSE_TRUNCATION = 0.50
#: Attempts with a known ending a refusal waits for, in a job or over a run's
#: jobs: a job of fewer is judged with the run's other jobs, or not at all.
MIN_TRIALS = 5

#: The share of the job's main calls that may carry another value before a
#: requested setting counts as not applied.
MISMATCH_SHARE = 0.01
#: A trial in which more than this share of the calls that carried a setting
#: carried another value shows that the setting was not applied: the job
#: stops after it.
CLEAR_SHARE = 0.5
#: The settings a main call carries, by its record's field (model_calls).
SENT_FIELDS = ("effort", "max_output_tokens", "temperature", "top_p")

#: A refused job's trials ran under the setting it was refused for, and a
#: resume with that setting is refused again: with another, they run in a new
#: job, or again in this one with --restart-changed.
_AGAIN = "then run the tasks again in a new job (or in this one with --restart-changed)"
#: What to do about each check that failed the run.
NEXT_STEPS = {
    "truncation": (
        "raise the output cap if the model takes more (Claude Code: --agent-env "
        f"CLAUDE_CODE_MAX_OUTPUT_TOKENS=<n>), {_AGAIN}; at the model's own cap, "
        "report the truncation rate beside the score"
    ),
    "model": (
        "check that the model id names the model you mean on its route (the "
        f"provider answered as another), {_AGAIN}"
    ),
    "effort": (
        "set the effort the agent sends (--reasoning-effort; an agent's own "
        f"setting, such as CLAUDE_CODE_EFFORT_LEVEL, overrides it), {_AGAIN}"
    ),
    "max_output_tokens": (
        "set the agent's cap to what you asked for, at most the model's own "
        f"(Claude Code lowers a larger one to it), {_AGAIN}"
    ),
    "temperature": f"set the temperature where the agent sends it, {_AGAIN}",
    "top_p": f"set top_p where the agent sends it, {_AGAIN}",
}

#: Model ids that name a deployment or an alias, not a model: Azure routes by
#: deployment name (LiteLLM's ``azure/`` and ``azure_ai/``, BenchFlow's
#: ``azure-foundry-openai/`` and ``azure-foundry-anthropic/``), a LiteLLM proxy
#: by the model names its own config gives (BenchFlow's ``litellm/``,
#: LiteLLM's ``litellm_proxy/``). The provider answers with the model's id.
ALIAS_ROUTES = (
    "azure/",
    "azure_ai/",
    "azure-foundry-openai/",
    "azure-foundry-anthropic/",
    "litellm/",
    "litellm_proxy/",
)

#: The causes (failures.py) of a trial whose prompt BenchFlow ended, not the
#: agent: its idle watchdog stopped a silent agent, its time limit ran out, or
#: its token or turn budget was spent (benchflow.trial_budget).
HARNESS_ENDINGS = frozenset({"idle_timeout", "timeout", "trial_budget"})


def harness_ending(
    error_category: str | None, stop_reasons: Sequence[str | None] | None
) -> dict[str, Any]:
    """agent_outcome's ``ended_by`` (``harness``, ``agent`` or None when
    nothing says) and ``harness_cause`` (the failures.py cause)."""
    if error_category in HARNESS_ENDINGS:
        return {"ended_by": "harness", "harness_cause": error_category}
    turns = list(stop_reasons or [])
    # A prompt BenchFlow cancelled did not end on the agent's own.
    if turns and turns[-1] not in (None, "cancelled"):
        return {"ended_by": "agent", "harness_cause": None}
    return {"ended_by": None, "harness_cause": None}


def turn_endings(
    summary: Mapping[str, Any] | None,
    *,
    stop_reasons: Sequence[str | None] | None,
    error: str | None,
) -> dict[str, Any]:
    """agent_outcome's ``stop_reasons``, ``finish_reasons``, ``truncated``
    and ``truncated_by``.

    *summary* is the trial's model-call totals (``model-calls.json``), None
    when no call was recorded; *stop_reasons* each prompt turn's ACP stop
    reason, None for a turn that did not end; *error* the trial's error.
    """
    summary = summary or {}
    last = summary.get("last_agent_call") or {}
    reason = last.get("finish_reason") if isinstance(last, Mapping) else None
    turns = list(stop_reasons or [])
    last_turn = turns[-1] if turns else None
    truncated: bool | None
    by: str | None = None
    if isinstance(reason, str) and reason in _TRUNCATED:
        truncated, by = True, "last_call"
    elif last_turn == "max_tokens":
        truncated, by = True, "stop_reason"
    elif is_output_cap_error(error):
        truncated, by = True, "error"
    elif isinstance(reason, str) or last_turn is not None:
        truncated = False
    else:
        truncated = None
    finish = summary.get("finish_reasons")
    return {
        "stop_reasons": turns,
        "finish_reasons": dict(finish) if isinstance(finish, Mapping) else None,
        "truncated": truncated,
        "truncated_by": by,
    }


# A snapshot or alias suffix on a served model id: claude-sonnet-5-20260301,
# gpt-5-2025-08-07, kimi-k3-0905-preview, gemini-3-pro-preview-06-05.
_SNAPSHOT = re.compile(
    r"(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{2}|\d{3,8})(?:-(?:preview|latest|exp|beta))?"
    r"|preview|latest|exp"
)


def _model_key(model: str) -> str:
    """A model id without what routes it: the provider prefix, Bedrock's
    region, vendor and version, Vertex's date, Claude Code's ``[1m]``, and a
    trailing ``-latest``."""
    key = re.sub(r"\[[^\]]*\]$", "", model.strip().lower())
    key = key.rsplit("/", 1)[-1]
    key = re.sub(r"^(?:us|eu|apac|global|jp|au)\.", "", key)
    key = re.sub(r"^(?:anthropic|amazon|meta|mistral|deepseek|qwen|openai)\.", "", key)
    key = re.sub(r"-v\d+(?::\d+)?$", "", key)
    key = re.sub(r"@\d{8}$", "", key)
    return key.removesuffix("-latest")


def same_model(requested: str | None, served: str | None) -> bool:
    """Whether *served* is the model *requested* names: the same id once its
    routing is set aside (:func:`_model_key`), or a dated or aliased snapshot
    of it (``claude-sonnet-5`` served as ``claude-sonnet-5-20260301``)."""
    if not requested or not served:
        return False
    from benchflow.providers.litellm_config import safe_model_alias

    alias = safe_model_alias(requested)
    # BenchFlow's own proxy alias for the model, which an upstream that echoes
    # the request's model answers with (``benchflow-`` before the neutral
    # names, in records of earlier builds).
    if served in (alias, "benchflow-" + alias.removeprefix("runtime-")):
        return True
    a, b = _model_key(requested), _model_key(served)
    if a == b:
        return True
    return any(
        long.startswith(short + "-") and _SNAPSHOT.fullmatch(long[len(short) + 1 :])
        for short, long in ((a, b), (b, a))
    )


def read_calls(trial_dir: Path) -> list[dict[str, Any]]:
    """The trial's model-call records (``trajectory/model-calls.jsonl``)."""
    try:
        lines = (Path(trial_dir) / "trajectory" / CALLS_FILE).read_text(
            errors="replace"
        )
    except OSError:
        return []
    calls = []
    for line in lines.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            calls.append(row)
    return calls


def _call_kind(call: Mapping[str, Any], model: str | None) -> str | None:
    """``main`` for a call of the agent to the run's model, ``side`` for
    the agent's call to another model or a subagent's, None for a call made
    for someone else (a reviewer, a judge)."""
    if call.get("purpose") not in (None, "agent"):
        return None
    if call.get("model_served") == "<synthetic>":
        # A message Claude Code wrote itself (an API error it reports), no call.
        return None
    if call.get("subagent") is True:
        return "side"
    if not model:
        return "main"
    source = call.get("source")
    if source == PROXY:
        from benchflow.providers.litellm_config import safe_model_alias

        named = call.get("model_requested")
        if not named or named == safe_model_alias(model):
            return "main"
        return (
            "main"
            if same_model(model, named) or same_model(model, call.get("route"))
            else "side"
        )
    if source == CLAUDE_CODE_OTEL:
        # The model Claude Code asked for, its background calls among them.
        named = call.get("model_served")
        return "main" if not named or same_model(model, named) else "side"
    # A session log holds the agent's own conversation.
    return "main"


def _key(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def settings_sent(
    calls: Sequence[Mapping[str, Any]], model: str | None
) -> dict[str, Any] | None:
    """result.json's ``settings_sent``: per setting, each value the agent's
    main calls carried and how many carried it, and how many carried none
    (``without``); the served models and routes; and the side calls. None
    when no call was recorded."""
    if not calls:
        return None
    from benchflow.trajectories.model_calls import offers_no_tools

    main: list[Mapping[str, Any]] = []
    side: list[Mapping[str, Any]] = []
    for call in calls:
        kind = _call_kind(call, model)
        if kind == "main" and offers_no_tools(call, calls):
            # A session title or a summary the agent asked its own model for.
            kind = "side"
        if kind == "main":
            main.append(call)
        elif kind == "side":
            side.append(call)

    def counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
        found: dict[str, int] = {}
        for row in rows:
            value = row.get(field)
            if value is not None:
                found[_key(value)] = found.get(_key(value), 0) + 1
        return found

    sources = sorted({str(c.get("source")) for c in calls if c.get("source")})
    return {
        "source": sources[0] if len(sources) == 1 else sources,
        "main_calls": len(main),
        "side_calls": len(side),
        **{field: counts(main, field) for field in SENT_FIELDS},
        # The sampling seed each main call carried (a run asks for one with
        # BENCHFLOW_MODEL_SEED; a route may drop it).
        "seed": counts(main, "seed"),
        "without": {
            field: sum(1 for row in main if row.get(field) is None)
            for field in (*SENT_FIELDS, "seed", "model_served")
        },
        "models_served": counts(main, "model_served"),
        "routes": counts(main, "route"),
        "side_models": counts(
            side, "model_requested" if PROXY in sources else "model_served"
        ),
    }


def _number(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def requested_settings(card: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What the run asked its model calls to carry, from its run card: the
    model, the effort, the output cap and any sampling it set. ``fails``
    says whether a mismatch fails the run: BenchFlow's own default output
    cap only warns."""
    from benchflow.run_card import CONTROL_AGENTS

    model = card.get("model") or {}
    agent = card.get("agent") or {}
    if not model or agent.get("name") in CONTROL_AGENTS:
        return []
    settings = {s.get("name"): s for s in card.get("settings") or []}
    wanted: list[dict[str, Any]] = []
    if model.get("id") and not model.get("routes_own"):
        # A deployment's or an alias's name says nothing of the model it
        # serves: the model that answered is recorded, and another one warns.
        alias = str(model["id"]).strip().lower().startswith(ALIAS_ROUTES)
        wanted.append(
            {
                "setting": "model",
                "value": model["id"],
                "source": model.get("source") or "default",
                "fails": not alias,
                **({"alias": True} if alias else {}),
            }
        )
    effort = model.get("effort")
    claude_effort = settings.get("CLAUDE_CODE_EFFORT_LEVEL")
    if claude_effort and str(claude_effort.get("value")).lower() not in (
        "auto",
        "unset",
    ):
        # Claude Code applies it over the run's effort.
        wanted.append(
            {
                "setting": "effort",
                "value": str(claude_effort["value"]).lower(),
                "source": claude_effort["source"],
                "fails": True,
            }
        )
    elif effort:
        wanted.append(
            {"setting": "effort", "value": effort, "source": "run", "fails": True}
        )
    for name, setting in (
        ("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "max_output_tokens"),
        ("BENCHFLOW_MODEL_MAX_TOKENS", "max_output_tokens"),
        ("BENCHFLOW_MODEL_TEMPERATURE", "temperature"),
        ("BENCHFLOW_MODEL_TOP_P", "top_p"),
        ("BENCHFLOW_MODEL_SEED", "seed"),
    ):
        row = settings.get(name)
        value = _number(row.get("value")) if row else None
        if row is None or value is None:
            continue
        wanted.append(
            {
                "setting": setting,
                "value": int(value)
                if setting in ("max_output_tokens", "seed")
                else value,
                "source": row["source"],
                # A seed the calls did not carry, or carried another of, warns.
                "fails": row["source"] != "benchflow_default" and setting != "seed",
            }
        )
    return wanted


def compare_sent(
    wanted: Mapping[str, Any], sent: Mapping[str, Any]
) -> dict[str, Any] | None:
    """How one trial's main calls (*sent*, its ``settings_sent``) differ
    from one requested setting (*wanted*), or None when they match or carry
    nothing to compare. ``kind`` is ``mismatch`` (``calls`` of the ``of``
    that carried one carried another value) or ``not_sent``. The tolerance
    is the job's (:class:`RunChecks`), not the trial's."""
    setting = wanted["setting"]
    field = "models_served" if setting == "model" else setting
    carried: dict[str, int] = dict(sent.get(field) or {})
    without = int(
        (sent.get("without") or {}).get(
            "model_served" if setting == "model" else setting, 0
        )
        or 0
    )
    if not carried:
        if setting == "model" or not without or sent.get("source") != PROXY:
            return None
        return {"kind": "not_sent", "sent": {}, "calls": without, "of": without}
    value = wanted["value"]
    if setting == "model":
        differs = {v: n for v, n in carried.items() if not same_model(str(value), v)}
    elif setting == "effort":
        differs = {v: n for v, n in carried.items() if v.lower() != str(value).lower()}
    elif setting == "max_output_tokens":
        # An agent may ask for less on one call (a compaction's summary): the
        # cap applied is the largest one sent.
        largest = max(carried, key=lambda v: _number(v) or 0)
        if _number(largest) == float(value):
            return None
        return {
            "kind": "mismatch",
            "sent": carried,
            "calls": sum(n for v, n in carried.items() if _number(v) != float(value)),
            "of": sum(carried.values()),
        }
    else:
        differs = {v: n for v, n in carried.items() if _number(v) != float(value)}
    if not differs:
        return None
    return {
        "kind": "mismatch",
        "sent": carried,
        "calls": sum(differs.values()),
        "of": sum(carried.values()),
    }


def _tally() -> dict[str, Any]:
    """One requested setting against a job's main calls: the calls that
    carried another value and the trials they were in (``clear_trials``: the
    trials in which the setting was clearly not applied), the calls that
    carried none in the trials where none did, and an effort the model
    proxy's route set in place of the one asked for."""
    return {
        "differing": 0,
        "trials": 0,
        "clear_trials": 0,
        "first_trial": None,
        "not_sent": 0,
        "not_sent_trials": 0,
        "first_not_sent": None,
        # An effort the route set on each request instead: its main calls by
        # the value it set, and the trials.
        "route": {},
        "route_trials": 0,
        "first_route": None,
    }


def _requested_text(wanted: Mapping[str, Any]) -> str:
    from benchflow.run_card import SOURCE_WORDS

    raw = wanted.get("source")
    source = SOURCE_WORDS.get(str(raw))
    if raw == "default":
        source = "the agent's default"
    elif source is None and raw not in (None, "model", "run"):
        source = f"from {raw}"
    return f"{wanted['value']}" + (f" ({source})" if source else "")


def _sent_text(sent: Mapping[str, int]) -> str:
    return ", ".join(
        f"{value} ({n})" for value, n in sorted(sent.items(), key=lambda kv: -kv[1])
    )


def _read_result(trial_dir: Path | None) -> dict[str, Any] | None:
    if trial_dir is None:
        return None
    try:
        data = json.loads((Path(trial_dir) / "result.json").read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _percent(rate: float) -> str:
    return f"{rate:.1%}"


class RunChecks:
    """One job's checks of its finished trials (see the module).

    The Evaluation seeds it with the attempts an earlier run of the job left,
    consults :attr:`stopped` before each start (:meth:`skip` records a trial
    it keeps from starting) and calls :meth:`record` as each trial ends and
    as each attempt the retry policy runs again ends.
    """

    def __init__(self, card: Mapping[str, Any] | None, planned: int) -> None:
        self.card: Mapping[str, Any] = card or {}
        self.planned = max(1, int(planned))
        self.trials = 0
        self.known = 0
        # Of the attempts with a known ending, those the retry policy ran again.
        self.retried = 0
        self.truncated: list[str] = []
        # The trial folders counted, so no attempt counts twice.
        self._counted: set[Path] = set()
        self.reason: str | None = None
        # The check that stopped the job (``truncation`` or the setting), and
        # the truncation figures when the rate refused it.
        self.stopped_by: str | None = None
        self.refused_at: dict[str, int] | None = None
        self._refusal = ""
        self.not_started: list[str] = []
        self._warned = False
        self.requested = requested_settings(self.card)
        # Over the trials' main calls: each setting's values and the calls
        # that carried none, the served models, routes and side calls.
        self.sent: dict[str, dict[str, int]] = {}
        self.main_calls = 0
        self.side_calls = 0
        # The sources that recorded the calls (model_calls): only the proxy's
        # record says a setting was not sent.
        self.sources: set[str] = set()
        # Per requested setting, what the main calls carried against it.
        self.compared: dict[str, dict[str, Any]] = {
            wanted["setting"]: _tally() for wanted in self.requested
        }
        # Claude Code's compactions, over the trials that counted them.
        self.counted = 0
        self.compacted: list[str] = []
        self.compactions = 0
        self.pre_tokens: list[int] = []
        # The trials' token and turn budgets (result.json's trial_budget).
        self.budgeted = 0
        self.budget_limits: dict[str, set[int]] = {"tokens": set(), "turns": set()}
        self.budget_stops: dict[str, list[str]] = {"tokens": [], "turns": []}
        self.budget_used: dict[str, list[int]] = {"tokens": [], "turns": []}
        # Over the trials whose agent's own log says: their subagents' tokens
        # and all their tokens.
        self.subagent_tokens: list[float] = [0, 0]
        self._budget_pooled = False
        # A pool of jobs' checks (pool): the name its own lines start with,
        # the jobs that stopped, and the lines of jobs that recorded only
        # their verdict.
        self.label: str | None = None
        self.stops: list[dict[str, Any]] = []
        self.carried: dict[str, list[str]] = {"failures": [], "warnings": []}

    @property
    def stopped(self) -> bool:
        """Whether no new trial may start: a check refused the run."""
        return self.reason is not None

    def seed(
        self,
        trials: Iterable[tuple[str, Path | None]],
        retried: Iterable[tuple[str, Path | None]] = (),
    ) -> None:
        """Count what an earlier run of this job left: the trials it
        finished, and the other attempts of its tasks (*retried*)."""
        for name, trial_dir in trials:
            self._add(name, trial_dir)
        for name, trial_dir in retried:
            self._add(name, trial_dir, retried=True)
        self._check()

    def record(
        self, name: str, trial_dir: Path | None, *, retried: bool = False
    ) -> None:
        """Count one finished trial, or with *retried* an attempt the retry
        policy runs again, and stop the job if a check refuses."""
        self._add(name, trial_dir, retried=retried)
        self._check()

    def skip(self, name: str) -> bool:
        """True (and recorded) when trial ``name`` must not start."""
        if not self.stopped:
            return False
        self.not_started.append(name)
        return True

    def _add(self, name: str, trial_dir: Path | None, *, retried: bool = False) -> None:
        result = _read_result(trial_dir)
        if result is None or trial_dir is None:
            return
        folder = Path(trial_dir).absolute()
        if folder in self._counted:
            return
        self._counted.add(folder)
        self.trials += 1
        outcome = result.get("agent_outcome")
        truncated = outcome.get("truncated") if isinstance(outcome, dict) else None
        if isinstance(truncated, bool):
            self.known += 1
            if retried:
                self.retried += 1
            if truncated:
                self.truncated.append(name)
        self._add_budget(name, result.get("trial_budget"))
        compactions = result.get("compactions")
        if isinstance(compactions, dict):
            self.counted += 1
            count = int(compactions.get("count") or 0)
            self.compactions += count
            if count:
                self.compacted.append(name)
            self.pre_tokens += [
                t for t in compactions.get("pre_tokens") or [] if isinstance(t, int)
            ]
        sent = result.get("settings_sent")
        if not isinstance(sent, dict):
            return
        self.main_calls += int(sent.get("main_calls") or 0)
        self.side_calls += int(sent.get("side_calls") or 0)
        source = sent.get("source")
        self.sources.update(source if isinstance(source, list) else [str(source)])
        for field in (*SENT_FIELDS, "seed", "without", "models_served", "routes"):
            self._count(field, sent.get(field))
        self._count("side_models", sent.get("side_models"))
        trial = str(result.get("rollout_name") or name)
        effort = result.get("effort")
        route = effort.get("route") if isinstance(effort, dict) else None
        for wanted in self.requested:
            tally = self.compared[wanted["setting"]]
            if wanted["setting"] == "effort" and isinstance(route, str) and route:
                # The route sets the effort on each request (Bedrock pins and
                # caps it for some Claude models): that is the effort that
                # ran, whatever the calls' own record says.
                if route.lower() != str(wanted["value"]).lower():
                    calls = int(sent.get("main_calls") or 0)
                    tally["route"][route] = tally["route"].get(route, 0) + calls
                    tally["route_trials"] += 1
                    tally["first_route"] = tally["first_route"] or trial
                continue
            found = compare_sent(wanted, sent)
            if found is None:
                continue
            if found["kind"] == "not_sent":
                tally["not_sent"] += found["calls"]
                tally["not_sent_trials"] += 1
                tally["first_not_sent"] = tally["first_not_sent"] or trial
                continue
            tally["differing"] += found["calls"]
            tally["trials"] += 1
            tally["first_trial"] = tally["first_trial"] or trial
            # The largest cap sent is the cap applied: one that differs
            # differs for the whole trial.
            if (
                wanted["setting"] == "max_output_tokens"
                or found["calls"] > CLEAR_SHARE * found["of"]
            ):
                tally["clear_trials"] += 1

    def _add_budget(self, name: str, block: Any) -> None:
        """Count one trial's ``trial_budget``: its limits, what it used, the
        measure that stopped it and its subagents' tokens."""
        if not isinstance(block, Mapping):
            return
        self.budgeted += 1
        limits = block.get("budget") if isinstance(block.get("budget"), Mapping) else {}
        used = block.get("used") if isinstance(block.get("used"), Mapping) else {}
        for measure in ("tokens", "turns"):
            if isinstance(limits.get(measure), int):
                self.budget_limits[measure].add(int(limits[measure]))
            if isinstance(used.get(measure), int):
                self.budget_used[measure].append(int(used[measure]))
        stopped = block.get("stopped_by")
        if stopped in self.budget_stops:
            self.budget_stops[stopped].append(name)
        subagents = block.get("subagents")
        if isinstance(subagents, Mapping) and isinstance(
            subagents.get("of_tokens"), int
        ):
            self.subagent_tokens[0] += int(subagents.get("tokens") or 0)
            self.subagent_tokens[1] += int(subagents["of_tokens"])

    def trial_budget(self) -> dict[str, Any] | None:
        """summary.json's ``run_checks.trial_budget``: the trials with a
        budget, the budgets, the trials each measure stopped, the most and
        the median a trial used, and the subagents' share of the tokens
        where the agents' logs say; None when no trial had a budget."""
        if not self.budgeted:
            return None

        def median(values: list[int]) -> int | None:
            if not values:
                return None
            ordered = sorted(values)
            return ordered[len(ordered) // 2]

        sub, of = self.subagent_tokens
        return {
            "trials": self.budgeted,
            "budget": {
                measure: sorted(values)
                for measure, values in self.budget_limits.items()
            },
            "stopped": sum(len(names) for names in self.budget_stops.values()),
            "stopped_by": {
                measure: len(names) for measure, names in self.budget_stops.items()
            },
            "stopped_trials": {
                measure: sorted(names) for measure, names in self.budget_stops.items()
            },
            "used_max": {
                measure: max(values) if values else None
                for measure, values in self.budget_used.items()
            },
            # A pool of jobs keeps each job's most, not every trial's use.
            "used_median": {
                measure: None if self._budget_pooled else median(values)
                for measure, values in self.budget_used.items()
            },
            "subagent_token_share": round(sub / of, 4) if of else None,
        }

    def _merge_budget(self, name: str, block: Any) -> None:
        if not isinstance(block, Mapping) or not block.get("trials"):
            return
        self._budget_pooled = True
        self.budgeted += int(block["trials"])
        stopped = block.get("stopped_trials") or {}
        for measure in ("tokens", "turns"):
            self.budget_limits[measure].update(
                int(v) for v in (block.get("budget") or {}).get(measure) or []
            )
            most = (block.get("used_max") or {}).get(measure)
            if isinstance(most, int):
                self.budget_used[measure].append(most)
            self.budget_stops[measure] += [
                f"{name}/{trial}" for trial in stopped.get(measure) or []
            ]
        share = block.get("subagent_token_share")
        if isinstance(share, float):
            # Weighted by the job's trials: the pool's share is a mean.
            self.subagent_tokens[0] += share * int(block["trials"])
            self.subagent_tokens[1] += int(block["trials"])

    def _count(self, field: str, values: Any) -> None:
        totals = self.sent.setdefault(field, {})
        for value, n in (values if isinstance(values, Mapping) else {}).items():
            totals[value] = totals.get(value, 0) + int(n or 0)

    def merge(self, name: str, block: Mapping[str, Any] | None) -> None:
        """Add another job's figures (its summary.json ``run_checks``), its
        trials under *name*. A block holding only its verdict, as an
        earlier build's worker wrote, keeps its lines as they are."""
        if not isinstance(block, Mapping) or not block:
            return
        truncation = block.get("truncation")
        settings = block.get("settings")
        if not (
            isinstance(truncation, Mapping)
            and isinstance(settings, Mapping)
            and isinstance(settings.get("compared"), list)
        ):
            for kind in ("failures", "warnings"):
                self.carried[kind] += [
                    f"{name}: {line}" for line in block.get(kind) or []
                ]
            return
        if not self.requested:
            self.requested = [
                dict(w)
                for w in settings.get("requested") or []
                if isinstance(w, Mapping)
            ]
            self.compared = {w["setting"]: _tally() for w in self.requested}
        self.trials += int(truncation.get("trials") or 0)
        self.known += int(truncation.get("known") or 0)
        self.retried += int(truncation.get("retried") or 0)
        self.truncated += [
            f"{name}/{t}" for t in truncation.get("truncated_trials") or []
        ]
        self._merge_budget(name, block.get("trial_budget"))
        compactions = block.get("compactions") or {}
        self.counted += int(compactions.get("trials") or 0)
        self.compactions += int(compactions.get("compactions") or 0)
        self.compacted += [
            f"{name}/{t}" for t in compactions.get("compacted_trials") or []
        ]
        if isinstance(compactions.get("max_pre_tokens"), int):
            self.pre_tokens.append(compactions["max_pre_tokens"])
        self.main_calls += int(settings.get("main_calls") or 0)
        self.side_calls += int(settings.get("side_calls") or 0)
        self.sources.update(str(source) for source in settings.get("sources") or [])
        sent = settings.get("sent") or {}
        for field in (*SENT_FIELDS, "seed", "without"):
            self._count(field, sent.get(field))
        for field in ("models_served", "routes", "side_models"):
            self._count(field, settings.get(field))
        for row in settings.get("compared") or []:
            tally = (
                self.compared.get(row.get("setting"))
                if isinstance(row, Mapping)
                else None
            )
            if tally is None:
                continue
            for key, value in row.items():
                if key == "route":
                    for effort, n in (value or {}).items():
                        tally["route"][effort] = tally["route"].get(effort, 0) + int(n)
                elif key.startswith("first_"):
                    if not tally.get(key) and value:
                        tally[key] = f"{name}/{value}"
                elif key in tally:
                    tally[key] += int(value or 0)
        if block.get("stopped"):
            self.stops.append(
                {
                    "name": name,
                    "by": block.get("stopped_by"),
                    "reason": block.get("reason"),
                }
            )
            self.not_started += [f"{name}/{t}" for t in block.get("not_started") or []]

    def mismatches(self, *, final: bool = True) -> list[dict[str, Any]]:
        """The requested settings the main calls did not carry: setting,
        requested, its source, what the job's main calls sent, and how many
        carried another value (``calls``) of those that carried one (``of``).

        A trial that clearly did not apply the setting (``clear_trials``) is a
        mismatch at once; calls with another value in more than
        :data:`MISMATCH_SHARE` of the job's are one when *final*, once the
        job's trials have finished.
        """
        found = []
        for wanted in self.requested:
            setting = wanted["setting"]
            tally = self.compared[setting]
            carried = dict(
                self.sent.get("models_served" if setting == "model" else setting) or {}
            )
            of = sum(carried.values())
            row = {
                "setting": setting,
                "requested": wanted["value"],
                "source": wanted["source"],
                **({"alias": True} if wanted.get("alias") else {}),
            }
            if tally["differing"] and (
                tally["clear_trials"]
                or (final and tally["differing"] > MISMATCH_SHARE * of)
            ):
                found.append(
                    {
                        **row,
                        "kind": "mismatch",
                        "fails": bool(wanted["fails"]),
                        "sent": carried,
                        "calls": tally["differing"],
                        "of": of,
                        "trials": tally["trials"],
                        "first_trial": tally["first_trial"],
                    }
                )
            elif tally["route_trials"]:
                route = dict(tally["route"])
                found.append(
                    {
                        **row,
                        "kind": "route",
                        "fails": False,
                        "sent": route,
                        "calls": sum(route.values()),
                        "of": sum(route.values()),
                        "trials": tally["route_trials"],
                        "first_trial": tally["first_route"],
                    }
                )
            elif tally["not_sent_trials"]:
                found.append(
                    {
                        **row,
                        "kind": "not_sent",
                        "fails": False,
                        "sent": {},
                        "calls": tally["not_sent"],
                        "of": tally["not_sent"],
                        "trials": tally["not_sent_trials"],
                        "first_trial": tally["first_not_sent"],
                    }
                )
        return found

    @staticmethod
    def _mismatch_text(m: Mapping[str, Any]) -> str:
        requested = _requested_text({"value": m["requested"], "source": m["source"]})
        if m["kind"] == "mismatch" and m["source"] == "benchflow_default":
            return (
                f"{m['setting']}: BenchFlow's default {m['requested']}; the main calls "
                f"carried {_sent_text(m['sent'])}, in {m['trials']} trial(s): the agent "
                "lowers it to the model's own cap; set the cap you mean to measure "
                "with --agent-env"
            )
        if m["kind"] == "not_sent":
            return (
                f"{m['setting']}: requested {requested}; no main call carried it "
                f"({m['calls']} calls in {m['trials']} trial(s)), so the provider's "
                "default applied"
            )
        if m["kind"] == "route":
            return (
                f"{m['setting']}: requested {requested}; the model proxy's route "
                f"set it to {', '.join(sorted(m['sent']))} on each request, in "
                f"{m['trials']} trial(s)"
            )
        if m.get("alias"):
            return (
                f"{m['setting']}: requested {requested}, a deployment's or an "
                f"alias's name; the calls were served as {_sent_text(m['sent'])}, "
                "recorded, not compared"
            )
        return (
            f"{m['setting']}: requested {requested}; the main calls carried "
            f"{_sent_text(m['sent'])}: {m['calls']} of {m['of']} calls differ, in "
            f"{m['trials']} trial(s), first {m['first_trial']}"
        )

    def rate(self) -> float | None:
        """The truncation rate over the trials with a known ending."""
        return len(self.truncated) / self.known if self.known else None

    def _level(self) -> str | None:
        rate = self.rate()
        if rate is None:
            return None
        if self.known < MIN_TRIALS:
            return "too_few"
        if rate >= REFUSE_TRUNCATION:
            return "refuse"
        return "warn" if rate >= WARN_TRUNCATION else "ok"

    def _truncation_text(self) -> str:
        return truncation_text(len(self.truncated), self.known, self.retried)

    def _check(self) -> None:
        # Only a clear mismatch stops the job; the share of odd calls is the
        # job's, judged at its end (summary).
        failing = [m for m in self.mismatches(final=False) if m["fails"]]
        if failing and self.reason is None:
            self.reason = self._mismatch_text(failing[0])
            self.stopped_by = failing[0]["setting"]
            logger.error(
                "Stopping the job: %s; running trials finish, no new ones start",
                self.reason,
            )
            return
        level = self._level()
        if level == "refuse" and self.reason is None:
            self._refusal = self._truncation_text()
            self.refused_at = {"known": self.known, "truncated": len(self.truncated)}
            self.stopped_by = "truncation"
            self.reason = (
                f"{self._refusal}, at or above the "
                f"{REFUSE_TRUNCATION:.0%} that refuses a run"
            )
            logger.error(
                "Stopping the job: %s; running trials finish, no new ones start",
                self.reason,
            )
        elif level == "warn" and not self._warned:
            self._warned = True
            logger.warning(
                "%s, at or above %s: raise the output cap, or report the rate "
                "beside the score",
                self._truncation_text(),
                f"{WARN_TRUNCATION:.0%}",
            )

    def truncation(self) -> dict[str, Any]:
        """summary.json's ``run_checks.truncation``."""
        rate = self.rate()
        return {
            "trials": self.trials,
            "known": self.known,
            "retried": self.retried,
            "truncated": len(self.truncated),
            "rate": round(rate, 4) if rate is not None else None,
            "level": self._level(),
            "warn_at": WARN_TRUNCATION,
            "refuse_at": REFUSE_TRUNCATION,
            "min_trials": MIN_TRIALS,
            "truncated_trials": sorted(self.truncated),
            # The figures when the rate stopped the job, else None.
            "refused_at": self.refused_at,
        }

    def summary(self) -> dict[str, Any]:
        """summary.json's ``run_checks`` block: the figures, what failed the
        run and what only warns, and the trials a stop kept from starting.
        ``refused`` says the checks start no more trials: a job stopped, or
        a pool of jobs' figures refuse the run."""
        truncation = self.truncation()
        mismatches = self.mismatches()
        level = truncation["level"]
        failures = [self._mismatch_text(m) for m in mismatches if m["fails"]]
        warnings = [self._mismatch_text(m) for m in mismatches if not m["fails"]]
        failed = [m["setting"] for m in mismatches if m["fails"]]
        if self.refused_at is not None:
            # A refused job stays failed: the trials it kept from starting
            # never ran, whatever the trials still running brought the rate to.
            line = f"truncation: {self._refusal}"
            if self.refused_at != {
                "known": self.known,
                "truncated": len(self.truncated),
            }:
                line += (
                    f" when the job stopped, at or above {REFUSE_TRUNCATION:.0%}; "
                    f"{len(self.truncated)} of {self.known} "
                    f"({_percent(self.rate() or 0.0)}) by its end"
                )
            else:
                line += f", at or above {REFUSE_TRUNCATION:.0%}"
            failures.append(line)
            failed.append("truncation")
        elif level == "refuse":
            failures.append(
                f"truncation: {self._truncation_text()}, at or above "
                f"{REFUSE_TRUNCATION:.0%}"
            )
            failed.append("truncation")
        elif level == "warn":
            warnings.append(
                f"truncation: {self._truncation_text()}, at or above "
                f"{WARN_TRUNCATION:.0%}"
            )
        elif level == "too_few" and (self.rate() or 0.0) >= WARN_TRUNCATION:
            # Too few to refuse the run on, not to say so.
            warnings.append(
                f"truncation: {self._truncation_text()}, at or above "
                f"{WARN_TRUNCATION:.0%}; too few to judge (a refusal waits for "
                f"{MIN_TRIALS})"
            )
        if self.label:
            failures = [f"{self.label}: {line}" for line in failures]
            warnings = [f"{self.label}: {line}" for line in warnings]
        if level != "refuse":
            # A job of the pool its own truncation rate stopped, on five or
            # more of its own attempts: the run stays failed.
            stops = [stop for stop in self.stops if stop["by"] == "truncation"]
            failures += [
                f"{stop['name']}: truncation: {stop['reason']}" for stop in stops
            ]
            if stops:
                failed.append("truncation")
        failures += self.carried["failures"]
        warnings += self.carried["warnings"]
        first = self.stops[0] if self.stops else None
        return {
            "truncation": truncation,
            "trial_budget": self.trial_budget(),
            "compactions": {
                "trials": self.counted,
                "compacted": len(self.compacted),
                "compactions": self.compactions,
                "max_pre_tokens": max(self.pre_tokens) if self.pre_tokens else None,
                "compacted_trials": sorted(self.compacted),
            },
            "settings": {
                "requested": self.requested,
                "mismatches": mismatches,
                "main_calls": self.main_calls,
                "side_calls": self.side_calls,
                "sources": sorted(self.sources),
                "sent": {
                    field: self.sent.get(field, {})
                    for field in (*SENT_FIELDS, "seed", "without")
                },
                # Per requested setting, the calls that carried another value
                # or none and the trials they were in (pool adds these up).
                "compared": [
                    {"setting": setting, **tally}
                    for setting, tally in self.compared.items()
                ],
                "models_served": self.sent.get("models_served", {}),
                "routes": self.sent.get("routes", {}),
                "side_models": self.sent.get("side_models", {}),
            },
            "failed": bool(failures),
            "failures": failures,
            "warnings": warnings,
            # What to do about each check that failed the run.
            "next_steps": [
                NEXT_STEPS[check]
                for check in dict.fromkeys(failed)
                if check in NEXT_STEPS
            ],
            "refused": self.stopped
            or bool(self.stops)
            or level == "refuse"
            or bool(self.carried["failures"]),
            "stopped": self.stopped or bool(self.stops),
            "stopped_by": self.stopped_by or (first["by"] if first else None),
            "reason": self.reason
            or (f"{first['name']}: {first['reason']}" if first else None),
            "not_started": sorted(self.not_started),
        }


def pool(
    blocks: Iterable[tuple[str, Mapping[str, Any] | None]], *, label: str | None = None
) -> dict[str, Any]:
    """The run checks of a run of several jobs (the repeats of ``--trials``,
    the workers of a sharded run) from each job's ``run_checks`` block, by
    name: the same block, judged on their pooled figures. A job's own stop
    keeps the run failed. *label* starts the lines the pooled figures give;
    a job's own lines start with its name."""
    checks = RunChecks(None, planned=1)
    checks.label = label
    for name, block in blocks:
        checks.merge(name, block)
    return checks.summary()


def truncation_text(truncated: int, known: int, retried: int = 0) -> str:
    """``N of M trials with a known ending were cut at the output cap (X%)``;
    attempts when *retried* of them were attempts the retry policy ran again."""
    rate = truncated / known if known else 0.0
    if not retried:
        return (
            f"{truncated} of {known} trials with a known ending were cut at the "
            f"output cap ({_percent(rate)})"
        )
    return (
        f"{truncated} of {known} attempts with a known ending ({retried} of them "
        f"retried) were cut at the output cap ({_percent(rate)})"
    )


def after_run_lines(summary: Mapping[str, Any]) -> list[str]:
    """The run card's lines after the run, from summary.json's ``run_checks``."""
    lines = ["Run card, after the run:"]
    truncation = summary.get("truncation") or {}
    level = truncation.get("level")
    if level is None:
        lines.append(
            "  Truncated: not known: no finished trial says how its model ended"
        )
    else:
        lines.append(
            "  Truncated: "
            + truncation_text(
                int(truncation["truncated"]),
                int(truncation["known"]),
                int(truncation.get("retried") or 0),
            )
            + {
                "ok": f"; under the {WARN_TRUNCATION:.0%} warning",
                "warn": f": at or above {WARN_TRUNCATION:.0%}, a warning; "
                f"{REFUSE_TRUNCATION:.0%} refuses the run",
                "refuse": f": at or above {REFUSE_TRUNCATION:.0%}, the run is refused",
                "too_few": f"; too few to judge (a refusal waits for "
                f"{truncation.get('min_trials')})",
            }[level]
        )
    compactions = summary.get("compactions") or {}
    if compactions.get("trials"):
        largest = compactions.get("max_pre_tokens")
        lines.append(
            f"  Compacted: {compactions['compacted']} of {compactions['trials']} "
            "trials compacted their context"
            + (
                f" ({compactions['compactions']} compactions"
                + (f", the largest at {largest} tokens" if largest else "")
                + ")"
                if compactions.get("compactions")
                else ""
            )
        )
    budget = summary.get("trial_budget")
    if isinstance(budget, Mapping) and budget.get("trials"):
        lines.append("  Per trial: " + trial_budget_used_text(budget))
    settings = summary.get("settings") or {}
    if settings.get("main_calls") or settings.get("side_calls"):
        served = settings.get("models_served") or {}
        routes = settings.get("routes") or {}
        lines.append(
            f"  Calls:     {settings.get('main_calls', 0)} to the run's model"
            + (f", served as {_sent_text(served)}" if served else "")
            + (f", by route {_sent_text(routes)}" if routes else "")
            + (
                f"; {settings['side_calls']} side calls to "
                f"{_sent_text(settings.get('side_models') or {}) or 'other models'}, "
                "recorded, not compared"
                if settings.get("side_calls")
                else ""
            )
        )
        mismatched = {m["setting"]: m for m in settings.get("mismatches") or []}
        sent = settings.get("sent") or {}
        # Only the proxy sees each request whole: what it saw no request
        # carry was not sent; what a session log lacks was not recorded.
        unseen = "none sent" if settings.get("sources") == ["proxy"] else "not recorded"
        for wanted in settings.get("requested") or []:
            m = mismatched.get(wanted["setting"])
            carried = (
                served if wanted["setting"] == "model" else sent.get(wanted["setting"])
            )
            if m is not None:
                text = RunChecks._mismatch_text(m)
                verdict = "the run fails" if m["fails"] else "a warning"
                lines.append(f"  Sent:      {text}: {verdict}")
            elif carried:
                lines.append(
                    f"  Sent:      {wanted['setting']}: requested "
                    f"{_requested_text(wanted)}; the main calls carried it"
                )
            else:
                lines.append(
                    f"  Sent:      {wanted['setting']}: requested "
                    f"{_requested_text(wanted)}; {unseen} in the calls, so not compared"
                )
        requested = {w["setting"] for w in settings.get("requested") or []}
        recorded = [
            f"{field} {_sent_text(sent.get(field) or {}) or unseen}"
            for field in SENT_FIELDS
            if field not in requested
        ]
        if sent.get("seed") and "seed" not in requested:
            # A seed the agent sent of its own accord.
            recorded.append(f"seed {_sent_text(sent['seed'])}")
        if recorded:
            lines.append("  Recorded:  " + "; ".join(recorded) + " (not requested)")
    if summary.get("stopped"):
        lines.append(
            f"  Stopped:   {summary.get('reason')}; "
            f"{len(summary.get('not_started') or [])} trial(s) not started"
        )
    scoring = summary.get("scoring")
    if isinstance(scoring, Mapping):
        # The scoring rule's figures (summary.json's scoring block).
        from benchflow.scoring import card_lines

        lines += card_lines(scoring)
    return lines


def trial_budget_used_text(block: Mapping[str, Any]) -> str:
    """The after-run Per trial line, after its label, from
    ``run_checks.trial_budget``: the budget, the trials it stopped, and what
    the trials used against it."""
    limits = block.get("budget") or {}
    most = block.get("used_max") or {}
    middle = block.get("used_median") or {}
    stopped = int(block.get("stopped") or 0)
    by = block.get("stopped_by") or {}

    def amount(values: list[int], unit: str) -> str:
        if not values:
            return ""
        low, high = min(values), max(values)
        return (f"{low:,}" if low == high else f"{low:,} to {high:,}") + f" {unit}s"

    budget = " and ".join(
        part
        for part in (
            amount(limits.get("tokens") or [], "token"),
            amount(limits.get("turns") or [], "turn"),
        )
        if part
    )
    used = "; ".join(
        f"{measure} used: median {middle[measure]:,}, most {most[measure]:,}"
        for measure in ("tokens", "turns")
        if isinstance(most.get(measure), int) and isinstance(middle.get(measure), int)
    )
    share = block.get("subagent_token_share")
    return (
        f"budget {budget}; {stopped} of {block['trials']} trials stopped at it"
        + (
            " (" + ", ".join(f"{n} by {m}" for m, n in sorted(by.items()) if n) + ")"
            if stopped
            else ""
        )
        + (f"; {used}" if used else "")
        + (
            f"; subagents used {share:.0%} of the tokens, where the agent's own "
            "log says"
            if isinstance(share, float)
            else ""
        )
    )


def seeding(
    *,
    n_tasks: int | None,
    sample_seed: int | None,
    rollout_seeds: Sequence[int] | None,
    checks: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The seeds a job's numbers depend on (summary.json's
    ``scoring.seeding``).

    The task order has no seed: a run starts its tasks in name order (a
    ``--n-tasks`` sample is drawn with ``--sample-seed`` and kept in name
    order), and nothing shuffles them; with concurrency they finish in any
    order. ``sampling_seed`` is the seed the run asked the model for
    (``BENCHFLOW_MODEL_SEED``) and what the agent's main calls carried, from
    the run checks' count (*checks*, summary.json's ``run_checks``); only the
    model proxy sees each request whole, so a seed is ``not recorded`` without
    it. A requested seed that no main call carried (a route dropped it), or
    that some calls carried another of, warns.
    """
    settings = (checks or {}).get("settings") or {}
    requested = next(
        (
            w["value"]
            for w in settings.get("requested") or []
            if isinstance(w, Mapping) and w.get("setting") == "seed"
        ),
        None,
    )
    sent = settings.get("sent") or {}
    carried = {str(k): int(n) for k, n in (sent.get("seed") or {}).items()}
    without = int((sent.get("without") or {}).get("seed") or 0)
    calls = sum(carried.values()) + without
    proxy = settings.get("sources") == [PROXY] and calls > 0
    warnings: list[str] = []
    if not proxy:
        status = "not recorded"
    elif requested is None:
        status = "carried" if carried else "none"
    elif set(carried) == {str(requested)} and not without:
        status = "carried"
    elif not carried:
        status = "dropped"
        warnings.append(
            f"sampling seed: {requested} was asked for, but none of the {calls} main "
            "calls carried it: the route dropped it, so the provider sampled freely"
        )
    else:
        status = "mismatch"
        warnings.append(
            f"sampling seed: {requested} was asked for, but the main calls carried "
            + _sent_text(carried)
            + (f" and {without} carried none" if without else "")
        )
    if n_tasks:
        sample = (
            f"a random sample of {n_tasks} tasks drawn with seed {sample_seed}"
            if sample_seed is not None
            else f"the first {n_tasks} tasks in name order"
        )
    else:
        sample = "every selected task"
    order = (
        f"tasks: {sample}, started in name order (no task-order seed; nothing "
        "shuffles them)"
    )
    rollouts = (
        "rollout seeds " + ", ".join(str(seed) for seed in rollout_seeds)
        if rollout_seeds
        else "no rollout seeds"
    )
    sampling = {
        "not recorded": "the sampling seed is not recorded (no call went through "
        "BenchFlow's model proxy)",
        "none": f"no sampling seed asked for or sent ({calls} main calls)",
        "carried": (
            f"sampling seed {requested} asked for and carried by all {calls} main calls"
            if requested is not None
            else f"no sampling seed asked for; the calls carried {_sent_text(carried)}"
        ),
        "dropped": f"sampling seed {requested} asked for, carried by none of the "
        f"{calls} main calls",
        "mismatch": f"sampling seed {requested} asked for; the calls carried "
        f"{_sent_text(carried)}",
    }[status]
    return {
        "task_order": "name",
        "task_order_seed": None,
        "task_sample": {"size": n_tasks, "seed": sample_seed} if n_tasks else None,
        "rollout_seeds": list(rollout_seeds) if rollout_seeds else None,
        "sampling_seed": {
            "requested": requested,
            "status": status,
            "carried": carried,
            "calls": calls,
            "without": without,
        },
        "warnings": warnings,
        "text": f"{order}; {rollouts}; {sampling}",
    }
