"""YAML job configs: planned jobs and unsupported settings.

A YAML job config runs every requested cell or is refused:

* every entry of ``agents:`` with every entry of ``datasets:`` (or the
  ``tasks:`` list), each a cell with its own job records;
* each cell ``n_attempts`` times: ``n_attempts: 3`` is three trials of every
  task, scored per trial and as pass@k, never one trial and two retries;
* a setting BenchFlow cannot honour exactly (a per-phase timeout multiplier,
  a metric other than the mean, an agent's ``kwargs``, ...) is refused when
  the file loads, by file, line and key, with what to do instead.
  Default-valued blocks are accepted, including fully expanded configs.

The loader is ``Evaluation._from_legacy_yaml``; ``bench run --config`` runs a
config of several cells or trials with :func:`job_plan`, one job per
cell and trial in ``<jobs dir>/<cell>/trial-NN/`` (the ``--matrix`` layout).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# Accepted top-level job keys. The loader reads supported settings;
# unsupported settings must hold their documented default.
JOB_KEYS: frozenset[str] = frozenset(
    {
        "job_name", "jobs_dir", "n_attempts", "install_only",
        "timeout_multiplier", "agent_timeout_multiplier",
        "verifier_timeout_multiplier", "agent_setup_timeout_multiplier",
        "environment_build_timeout_multiplier", "debug", "n_concurrent_trials",
        "quiet", "retry", "environment", "verifier", "metrics", "agents",
        "user_agent", "datasets", "tasks", "artifacts", "extra_instruction_paths",
        "extra_instructions", "source_jobs",
    }
)  # fmt: skip

# Unsupported job settings and their accepted defaults.
# A default is accepted; any other value is refused.
_RETRY_DEFAULT: dict[str, Any] = {
    "max_retries": 0,
    "include_exceptions": None,
    "exclude_exceptions": [
        "RewardFileEmptyError", "ApiUsageLimitError", "AgentAuthenticationError",
        "AgentSafetyRefusalError", "ModelNotFoundError", "RewardFileNotFoundError",
        "VerifierOutputParseError", "AgentTimeoutError", "VerifierTimeoutError",
    ],
    "wait_multiplier": 1.0,
    "min_wait_sec": 1.0,
    "max_wait_sec": 60.0,
}  # fmt: skip
JOB_DEFAULTS: dict[str, Any] = {
    "install_only": False,
    "debug": False,
    "retry": _RETRY_DEFAULT,
    "verifier": {
        "override_timeout_sec": None, "max_timeout_sec": None, "disable": False,
        "include_logs": [], "exclude_logs": [], "env": {}, "import_path": None,
        "kwargs": {},
    },
    "user_agent": None,
    "artifacts": [],
    "source_jobs": [],
}  # fmt: skip

# What to do instead, per refused top-level setting.
_JOB_ADVICE: dict[str, str] = {
    "install_only": "BenchFlow has no install-only run; drop it",
    "debug": "drop it; the job folder keeps each trial's logs",
    "retry": (
        "BenchFlow never retries trials from a YAML job config; "
        "drop it, or run the folder with bench run --retry-attempts N"
    ),
    "verifier": (
        "BenchFlow runs each task's verifier as task.md (or task.toml) sets it; "
        "drop it, or change the task"
    ),
    "user_agent": "BenchFlow has no simulated user for a YAML job config; drop it",
    "artifacts": "drop it; the job folder keeps the agent's logs and trajectory",
    "source_jobs": "re-grade a job with bench regrade <job folder>; drop it",
}

# Each agent entry key with its default. name, model_name, env,
# n_concurrent and concurrency_group are honoured, and BenchFlow's own
# reasoning_effort, harness and skip_install.
_AGENT_HONOURED = frozenset(
    {"name", "model_name", "env", "n_concurrent", "concurrency_group",
     "reasoning_effort", "harness", "skip_install"}
)  # fmt: skip
_AGENT_DEFAULTS: dict[str, Any] = {
    "import_path": None,
    "skills": [], "override_timeout_sec": None, "override_setup_timeout_sec": None,
    "max_timeout_sec": None, "resume_trajectory": False, "load_trajectory": None,
    "extra_allowed_hosts": [], "include_logs": [], "exclude_logs": [],
    "kwargs": {}, "mcp_servers": [],
}  # fmt: skip
_AGENT_ADVICE: dict[str, str] = {
    "import_path": "BenchFlow runs only its own agents (bench agents); use name:",
    "skills": "use skills_dir: with skill_mode: with-skill",
    "override_timeout_sec": "set the task's agent timeout_sec, or use agent_timeout_multiplier",
    "override_setup_timeout_sec": "drop it",
    "max_timeout_sec": "set the task's agent timeout_sec, or use agent_timeout_multiplier",
    "resume_trajectory": "BenchFlow does not run task.toml multi-step tasks; drop it",
    "load_trajectory": "drop it",
    "extra_allowed_hosts": "set the task's network allowlist instead",
    "include_logs": "drop it; the job folder keeps the agent's logs",
    "exclude_logs": "drop it; the job folder keeps the agent's logs",
    "kwargs": "BenchFlow passes no agent kwargs; drop them, or use env:",
    "mcp_servers": "drop it",
}  # fmt: skip

# Environment keys; type is honoured.
_ENVIRONMENT_DEFAULTS: dict[str, Any] = {
    "import_path": None, "force_build": False, "delete": True, "stream": False,
    "cpu_enforcement_policy": "auto", "memory_enforcement_policy": "auto",
    "override_cpus": None, "override_memory_mb": None, "override_storage_mb": None,
    "override_gpus": None, "override_tpu": None, "suppress_override_warnings": False,
    "mounts": None, "mounts_json": None, "extra_docker_compose": [], "env": {},
    "kwargs": {}, "extra_allowed_hosts": [],
}  # fmt: skip
_ENVIRONMENT_ADVICE: dict[str, str] = {
    "env": (
        "This setting applies to every command in the sandbox, including verification; "
        "move these variables to each agent's env: (the agent "
        "only), or pass --agent-env"
    ),
    "force_build": "drop it",
    "delete": "drop it; BenchFlow removes each trial's sandbox",
    "override_cpus": "set the task's cpus instead",
    "override_memory_mb": "set the task's memory instead",
    "override_storage_mb": "set the task's storage instead",
    "override_gpus": "set the task's gpus instead",
}

# Dataset keys besides path; task_names, exclude_task_names and n_tasks
# are honoured, the rest name a remote dataset.
_DATASET_HONOURED = frozenset({"path", "task_names", "exclude_task_names", "n_tasks"})
DATASET_REMOTE_KEYS = (
    "name",
    "version",
    "ref",
    "repo",
    "registry_url",
    "registry_path",
)
_DATASET_DEFAULTS: dict[str, Any] = {
    **dict.fromkeys(DATASET_REMOTE_KEYS),
    "registry": None,
    "overwrite": False,
    "download_dir": None,
}
_TASK_KEYS = frozenset(
    {"path", "git_url", "git_commit_id", "name", "ref", "overwrite", "download_dir",
     "source"}
)  # fmt: skip
_TASK_DEFAULTS: dict[str, Any] = {
    "overwrite": False,
    "download_dir": None,
    "source": None,
}
_ORCHESTRATOR_KEYS = frozenset(
    {"type", "n_concurrent_trials", "quiet", "retry", "kwargs"}
)

# Job phase multipliers that require validation before loading.
_PHASE_MULTIPLIERS = {
    "verifier_timeout_multiplier": "verifier",
    "agent_setup_timeout_multiplier": "agent setup",
    "environment_build_timeout_multiplier": "environment build",
}
# The phases scaled through the run's config overlay (config_override.py).
_OVERLAY_PHASES = {
    "verifier_timeout_multiplier": "verifier",
    "environment_build_timeout_multiplier": "build",
}

# Shell-style ${VAR} / ${VAR:-default} environment templates.
_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def is_job_config(raw: Mapping[str, Any]) -> bool:
    """A job config declares ``agents``, ``datasets``, or a ``tasks`` list."""
    return "agents" in raw or "datasets" in raw or isinstance(raw.get("tasks"), list)


def differs_from_default(value: Any, default: Any) -> bool:
    """A mapping counts as the default when every key it sets has the
    default's value; partial blocks and lists in any order are accepted."""
    if isinstance(value, dict) and isinstance(default, dict):
        return any(
            k not in default or differs_from_default(v, default[k])
            for k, v in value.items()
        )
    if isinstance(value, list) and isinstance(default, list):
        return sorted(map(repr, value)) != sorted(map(repr, default))
    return value != default


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


class _Problems:
    """The refusals of one file, each ``file:line: key: what is wrong; what to do``."""

    def __init__(self, path: Path | str | None, lines: Mapping[tuple[str, ...], int]):
        self.path = path
        self.lines = lines
        self.items: list[str] = []

    def add(self, keys: tuple[str, ...], text: str) -> None:
        from benchflow._utils.config_keys import located

        dotted = keys[0] + "".join(
            f"[{k}]" if k.isdigit() else f".{k}" for k in keys[1:]
        )
        where = located(self.path, self.lines.get(keys))
        self.items.append(
            f"{where}: {dotted}: {text}" if where else f"{dotted}: {text}"
        )

    def unknown(self, keys: tuple[str, ...], allowed: frozenset[str]) -> None:
        from benchflow._utils.config_keys import closest_key

        guess = closest_key(keys[-1], allowed)
        self.add(keys, "unknown key" + (f" (did you mean {guess!r}?)" if guess else ""))

    def mapping(self, keys: tuple[str, ...], value: Any) -> bool:
        if isinstance(value, dict):
            return True
        self.add(keys, f"must be a mapping, got {type(value).__name__}")
        return False


def _check_defaults(
    problems: _Problems,
    prefix: tuple[str, ...],
    block: Mapping[str, Any],
    defaults: Mapping[str, Any],
    advice: Mapping[str, str],
    honoured: frozenset[str],
) -> None:
    allowed = frozenset(defaults) | honoured
    for key, value in block.items():
        key = str(key)
        if key in honoured:
            continue
        if key not in defaults:
            problems.unknown((*prefix, key), allowed)
        elif differs_from_default(value, defaults[key]):
            problems.add(
                (*prefix, key),
                f"{value!r}: BenchFlow cannot honour this job setting; "
                + advice.get(key, "drop it"),
            )


def job_config_problems(
    raw: Mapping[str, Any],
    *,
    path: Path | str | None,
    lines: Mapping[tuple[str, ...], int],
) -> list[str]:
    """One line per setting of a YAML job config BenchFlow cannot honour
    exactly. Unknown top-level keys are the caller's (``check_keys``)."""
    problems = _Problems(path, lines)
    for key, default in JOB_DEFAULTS.items():
        if key in raw and differs_from_default(raw[key], default):
            problems.add(
                (key,),
                f"{raw[key]!r}: BenchFlow cannot honour this job setting; "
                + _JOB_ADVICE[key],
            )

    attempts = raw.get("n_attempts", 1)
    if not _count(attempts):
        problems.add(("n_attempts",), f"{attempts!r}: must be a whole number >= 1")
    for key in ("n_concurrent_trials",):
        if key in raw and not _count(raw[key]):
            problems.add((key,), f"{raw[key]!r}: must be a whole number >= 1")
    if "quiet" in raw and not isinstance(raw["quiet"], bool):
        problems.add(("quiet",), f"{raw['quiet']!r}: must be true or false")

    # Timeouts: each phase uses its own multiplier, else
    # timeout_multiplier. BenchFlow scales the agent, verifier and environment
    # build timeouts the same way (the task's own budgets); the agent setup
    # phase runs on BenchFlow's own install budget, so it must come out at 1.
    top = raw.get("timeout_multiplier", 1.0)
    for key in ("timeout_multiplier", "agent_timeout_multiplier", *_PHASE_MULTIPLIERS):
        value = raw.get(key)
        if value is not None and (
            not _number(value) or value <= 0 or not math.isfinite(value)
        ):
            problems.add((key,), f"{value!r}: must be a finite number above 0")
    setup = raw.get("agent_setup_timeout_multiplier")
    if setup is not None:
        if _number(setup) and setup != 1:
            problems.add(
                ("agent_setup_timeout_multiplier",),
                f"{setup!r}: BenchFlow installs agents with its own time budget, "
                "so it cannot scale the agent setup "
                "timeout; drop it or set it to 1",
            )
    elif _number(top) and top > 0 and top != 1:
        problems.add(
            ("timeout_multiplier",),
            f"{top!r}: BenchFlow scales the agent, verifier and environment build "
            "timeouts with it, but not the agent setup "
            "timeout (BenchFlow installs agents with its own time budget); set "
            "agent_setup_timeout_multiplier: 1 to run with the rest scaled",
        )

    metrics: Any = raw.get("metrics", [])
    if not isinstance(metrics, list):
        problems.add(("metrics",), "must be a list")
    else:
        for index, metric in enumerate(metrics):
            keys = ("metrics", str(index))
            if not isinstance(metric, dict):
                problems.mapping(keys, metric)
                continue
            spec = cast("dict[str, Any]", metric)
            kind = spec.get("type", "mean")
            if kind != "mean" or spec.get("kwargs"):
                problems.add(
                    keys,
                    f"{metric!r}: BenchFlow scores a job by its mean reward "
                    "(and pass@k over n_attempts) only; drop the entry, or "
                    "compute it from the trials' result.json files",
                )
            for extra in sorted(set(metric) - {"type", "kwargs"}):
                problems.unknown((*keys, str(extra)), frozenset({"type", "kwargs"}))

    paths = raw.get("extra_instruction_paths", [])
    if not isinstance(paths, list):
        problems.add(("extra_instruction_paths",), "must be a list of files")
    else:
        for index, entry in enumerate(paths):
            if not Path(str(entry)).expanduser().is_file():
                problems.add(
                    ("extra_instruction_paths", str(index)),
                    f"{entry!r}: file not found (read from {Path.cwd()})",
                )

    orch = raw.get("orchestrator")
    if orch is not None and problems.mapping(("orchestrator",), orch):
        for key in orch:
            if str(key) not in _ORCHESTRATOR_KEYS:
                problems.unknown(("orchestrator", str(key)), _ORCHESTRATOR_KEYS)
        if orch.get("type", "local") != "local":
            problems.add(
                ("orchestrator", "type"),
                f"{orch['type']!r}: BenchFlow runs trials on this machine; drop it",
            )
        if "retry" in orch and differs_from_default(orch["retry"], _RETRY_DEFAULT):
            problems.add(
                ("orchestrator", "retry"),
                f"{orch['retry']!r}: BenchFlow cannot honour this job setting; "
                + _JOB_ADVICE["retry"],
            )
        if orch.get("kwargs"):
            problems.add(("orchestrator", "kwargs"), "BenchFlow takes none; drop it")
        nested = orch.get("n_concurrent_trials")
        if nested is not None and not _count(nested):
            problems.add(
                ("orchestrator", "n_concurrent_trials"),
                f"{nested!r}: must be a whole number >= 1",
            )

    env = raw.get("environment")
    if env is not None and problems.mapping(("environment",), env):
        _check_defaults(
            problems, ("environment",), env, _ENVIRONMENT_DEFAULTS,
            _ENVIRONMENT_ADVICE, frozenset({"type"}),
        )  # fmt: skip

    groups: dict[str, Any] = {}
    agents = raw.get("agents")
    if agents is not None and not isinstance(agents, list):
        problems.add(("agents",), "must be a list")
    for index, agent in enumerate(agents if isinstance(agents, list) else []):
        keys = ("agents", str(index))
        if not problems.mapping(keys, agent):
            continue
        _check_defaults(
            problems, keys, agent, _AGENT_DEFAULTS, _AGENT_ADVICE, _AGENT_HONOURED
        )
        if "env" in agent and not isinstance(agent["env"], dict):
            problems.add((*keys, "env"), "must be a mapping of names to values")
        # Rules for the per-agent cap (job config
        # validate_agent_concurrency_limits). BenchFlow runs the cells one
        # after another, so a cap (or a group's shared cap) is the cell's.
        cap = agent.get("n_concurrent")
        top = raw.get("n_concurrent_trials")
        if top is None:
            top = (
                (orch or {}).get("n_concurrent_trials", 4)
                if isinstance(orch, dict)
                else 4
            )
        if cap is not None and not _count(cap):
            problems.add(
                (*keys, "n_concurrent"), f"{cap!r}: must be a whole number >= 1"
            )
        elif cap is not None and _count(top) and cap > top:
            problems.add(
                (*keys, "n_concurrent"),
                f"{cap}: cannot exceed n_concurrent_trials ({top}); lower it or "
                "raise n_concurrent_trials",
            )
        group = agent.get("concurrency_group")
        if group is not None:
            if cap is None:
                problems.add(
                    (*keys, "concurrency_group"),
                    f"{group!r}: an agent in a concurrency group must set n_concurrent",
                )
            elif groups.setdefault(str(group), cap) != cap:
                problems.add(
                    (*keys, "n_concurrent"),
                    f"{cap}: agents in concurrency_group {group!r} must use the "
                    f"same n_concurrent ({groups[str(group)]})",
                )

    # Implicit concurrency pools: agents with no
    # group and the same identity, n_concurrent and the group aside, share
    # one pool and must use the same cap (review B2-L1).
    pools: dict[str, tuple[int, Any]] = {}
    for index, agent in enumerate(agents if isinstance(agents, list) else []):
        if not isinstance(agent, dict) or agent.get("concurrency_group") is not None:
            continue
        identity = {
            **_AGENT_DEFAULTS,
            "env": {},
            "reasoning_effort": raw.get("reasoning_effort"),
            "harness": raw.get("harness", "acp"),
            "skip_install": False,
            **agent,
        }
        identity["skip_install"] = bool(identity["skip_install"])
        identity["name"] = identity.get("name") or (
            None if identity.get("import_path") else "oracle"
        )
        for key in ("n_concurrent", "concurrency_group"):
            identity.pop(key, None)
        key = json.dumps(
            {k: v for k, v in identity.items() if v is not None},
            sort_keys=True,
            default=str,
        )
        cap = agent.get("n_concurrent")
        if key not in pools:
            pools[key] = (index, cap)
        elif pools[key][1] != cap:
            first, other = pools[key]
            mine = "unset" if cap is None else cap
            theirs = "unset" if other is None else other
            problems.add(
                ("agents", str(index), "n_concurrent"),
                f"{mine}: agents[{first}] is the same agent with n_concurrent "
                f"{theirs}, and "
                "The same agent shares one pool with one cap; use the same "
                "value, or a concurrency_group each",
            )

    datasets = raw.get("datasets")
    if datasets is not None and not isinstance(datasets, list):
        problems.add(("datasets",), "must be a list")
    for index, dataset in enumerate(datasets if isinstance(datasets, list) else []):
        keys = ("datasets", str(index))
        if not problems.mapping(keys, dataset):
            continue
        # Remote sources are refused by the loader, with the way to run them.
        _check_defaults(
            problems, keys, {k: v for k, v in dataset.items() if k not in DATASET_REMOTE_KEYS},
            _DATASET_DEFAULTS, {}, _DATASET_HONOURED,
        )  # fmt: skip
        for key in ("task_names", "exclude_task_names"):
            names = dataset.get(key)
            if names is not None and not (
                isinstance(names, list) and all(isinstance(n, str) for n in names)
            ):
                problems.add((*keys, key), "must be a list of task names or globs")
        n_tasks = dataset.get("n_tasks")
        if n_tasks is not None and not _count(n_tasks):
            problems.add(
                (*keys, "n_tasks"), f"{n_tasks!r}: must be a whole number >= 1"
            )

    tasks = raw.get("tasks")
    for index, task in enumerate(tasks if isinstance(tasks, list) else []):
        keys = ("tasks", str(index))
        if not isinstance(task, dict):
            continue  # the loader names it
        for key, value in task.items():
            key = str(key)
            if key not in _TASK_KEYS:
                problems.unknown((*keys, key), _TASK_KEYS)
            elif key in _TASK_DEFAULTS and differs_from_default(
                value, _TASK_DEFAULTS[key]
            ):
                problems.add(
                    (*keys, key),
                    f"{value!r}: BenchFlow cannot honour this job setting; drop it",
                )
    return problems.items


def agent_timeout_multiplier(raw: Mapping[str, Any]) -> float | None:
    """Resolve the agent timeout multiplier:
    ``agent_timeout_multiplier``, else ``timeout_multiplier``."""
    own = raw.get("agent_timeout_multiplier")
    return own if own is not None else raw.get("timeout_multiplier")


def phase_timeout_multipliers(
    raw: Mapping[str, Any], *, explicit_only: bool = False
) -> dict[str, float]:
    """Resolve the verifier and build multipliers (the
    phase's own, else ``timeout_multiplier``, else 1), for the run's config
    overlay ``timeout_multipliers`` section. The build one is always there:
    it also bounds the whole environment start at any
    multiplier (review B2-M1); a verifier at 1 is left out. ``explicit_only``
    identifies phases supplied by the file, including explicit defaults,
    without counting the implicit build-start bound as a file override."""
    top = raw.get("timeout_multiplier")
    out: dict[str, float] = {}
    for key, phase in _OVERLAY_PHASES.items():
        own = raw.get(key)
        value = own if own is not None else top
        if explicit_only and value is None:
            continue
        value = 1.0 if value is None else float(value)
        if explicit_only or phase == "build" or value != 1:
            out[phase] = value
    return out


def extra_instruction(raw: Mapping[str, Any]) -> str | None:
    """Additional text appended to every task instruction: each
    ``extra_instruction_paths`` file, then each ``extra_instructions`` entry,
    joined by blank lines (BenchFlow adds the blank line before it)."""
    extra = raw.get("extra_instructions") or []
    if isinstance(extra, str):
        extra = [extra]
    parts = [
        Path(str(p)).expanduser().read_text()
        for p in raw.get("extra_instruction_paths") or []
    ]
    parts += [str(e) for e in extra]
    return "\n\n".join(parts) or None


def agent_env(agent: Mapping[str, Any], where: str, index: int) -> dict[str, str]:
    """An agent entry's ``env:``, with ``${VAR}`` and
    ``${VAR:-default}`` filled from this shell (an unset ``${VAR}`` with no
    default is refused)."""
    out: dict[str, str] = {}
    for key, value in (agent.get("env") or {}).items():
        value = str(value)
        match = _TEMPLATE.fullmatch(value)
        if match:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                value = os.environ[name]
            elif default is not None:
                value = default
            else:
                raise ValueError(
                    f"{where}agents[{index}].env.{key}: ${{{name}}} is not set "
                    "in this shell; export it, or give a default as "
                    f"${{{name}:-value}}"
                )
        out[str(key)] = value
    return out


def _alias_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "x"


@dataclass(frozen=True)
class JobCell:
    """One agent entry with one dataset entry (or the ``tasks:`` list)."""

    index: int
    agent_index: int
    # None: the file's tasks: list (an ad hoc dataset), or no datasets.
    dataset_index: int | None
    # The cell's folder name: agent, model when it has one, dataset.
    alias: str
    agent: str
    model: str | None
    dataset: str


@dataclass(frozen=True)
class JobPlan:
    """What a YAML job config runs: its cells, each ``attempts`` times."""

    cells: tuple[JobCell, ...]
    attempts: int
    n_agents: int
    job_name: str | None
    jobs_dir: Path
    quiet: bool

    @property
    def is_matrix(self) -> bool:
        """More than one job: several cells, or several attempts."""
        return len(self.cells) > 1 or self.attempts > 1

    def describe(self) -> str:
        return (
            f"{self.n_agents} agent(s) x {len(self.cells) // max(self.n_agents, 1)} "
            f"dataset(s) x n_attempts {self.attempts}"
        )


def plan_from_raw(raw: Mapping[str, Any]) -> JobPlan:
    """The cells of an already checked YAML job config (:func:`job_config_problems`)."""
    from benchflow.evaluation import DEFAULT_AGENT

    agents = raw.get("agents") or [{"name": "oracle"}]
    datasets: list[int | None] = [*range(len(raw.get("datasets") or []))]
    if not datasets:
        datasets = [None]
    labels: dict[int | None, str] = {None: "adhoc"}
    for index, dataset in enumerate(raw.get("datasets") or []):
        where = Path(str(dataset.get("path") or "tasks")).expanduser()
        labels[index] = where.resolve().name or where.name or "tasks"
    pairs = [
        (agent_index, agent, dataset_index)
        for agent_index, agent in enumerate(agents)
        for dataset_index in datasets
    ]
    natural = []
    for _, agent, dataset_index in pairs:
        name = str(agent.get("name") or DEFAULT_AGENT)
        model = agent.get("model_name") or None
        parts = [name, *([str(model)] if model else []), labels[dataset_index]]
        natural.append("__".join(_alias_part(p) for p in parts))
    # Every cell its own folder (review B-H1): each natural name is reserved
    # first, so a repeat takes the next suffix no other cell is named, and the
    # same file always gives the same folders (a rerun resumes into them).
    used: set[str] = set(natural)
    taken: set[str] = set()
    aliases = []
    for alias in natural:
        if alias in taken:
            n = 2
            while f"{alias}-{n}" in used:
                n += 1
            alias = f"{alias}-{n}"
            used.add(alias)
        taken.add(alias)
        aliases.append(alias)
    cells: list[JobCell] = []
    for (agent_index, agent, dataset_index), alias in zip(pairs, aliases, strict=True):
        name = str(agent.get("name") or DEFAULT_AGENT)
        model = agent.get("model_name") or None
        cells.append(
            JobCell(
                index=len(cells),
                agent_index=agent_index,
                dataset_index=dataset_index,
                alias=alias,
                agent=name,
                model=str(model) if model else None,
                dataset=labels[dataset_index],
            )
        )
    orch = raw.get("orchestrator") or {}
    return JobPlan(
        cells=tuple(cells),
        attempts=int(raw.get("n_attempts", 1)),
        n_agents=len(agents),
        job_name=str(raw["job_name"]) if raw.get("job_name") else None,
        jobs_dir=Path(raw.get("jobs_dir", "jobs")),
        quiet=bool(raw.get("quiet", orch.get("quiet", False))),
    )


def cell_raw(raw: Mapping[str, Any], cell: JobCell) -> dict[str, Any]:
    """The config of one cell: its agent and its dataset, run once."""
    out = dict(raw)
    if raw.get("agents"):
        out["agents"] = [raw["agents"][cell.agent_index]]
    if cell.dataset_index is not None:
        out["datasets"] = [raw["datasets"][cell.dataset_index]]
    return out


def job_plan(
    path: Path | str, *, agent: str | None = None, model: str | None = None
) -> JobPlan | None:
    """The plan of the YAML job config at ``path``, checked; None when the
    file is a BenchFlow run config. Raises the loader's errors.

    ``agent`` and ``model`` are the command line's ``--agent``/``--model``:
    with one agent in the file (or none), the cells are named for what runs
    (review B-M4). A file of several agents takes neither (the caller refuses).
    """
    from benchflow._utils.config_keys import parse_run_config
    from benchflow.evaluation import check_run_config_keys

    path = Path(path)
    text = path.read_text()
    raw = parse_run_config(text, path)
    if not isinstance(raw, dict) or not is_job_config(raw):
        return None
    raw = check_run_config_keys(raw, path=path, text=text, legacy=True)
    agents = raw.get("agents") or [{"name": "oracle"}]
    if (agent is not None or model is not None) and len(agents) == 1:
        entry = dict(agents[0])
        if agent is not None:
            entry["name"] = agent
            entry.pop("model_name", None)
        if model is not None:
            entry["model_name"] = model
        raw = {**raw, "agents": [entry]}
    return plan_from_raw(raw)
