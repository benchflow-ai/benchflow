"""What BenchFlow does with each part of a task.md draft 2 package.

:func:`plan_package` reads a package that the reference parser accepted and
decides, field by field, how BenchFlow runs it:

- **honored**: mapped onto a native BenchFlow setting, or implemented by the
  ``taskmd`` verifier strategy (rubric scoring, judge-prompt@1 judges);
- **refused**: a run-affecting field BenchFlow cannot honor. Loading the task
  fails and the message names the field. Nothing is ever skipped silently;
- **refused for agents**: honored when a scripted seat runs (the oracle, a
  control, or the do-nothing ``nop`` run), refused when an agent would, such
  as an agent's tool-call budget, which a script never approaches;
- **recorded**: metadata and scoring rules. Metadata changes nothing a
  runtime does; scoring rules are checked only for a result that claims to be
  scored, which no BenchFlow run claims yet, so they are recorded as not
  checked.

Every config key the reference parser accepts reaches a decision here: a key
the walk below does not name is refused by default (``_Walker.unknown``), and
``tests/test_taskmd_support.py`` checks that every key the reference tools
document (``KEY_DOCS``) has a row in :data:`SUPPORT`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.taskmd._vendor import judgeprompt as jp
from benchflow.taskmd._vendor import taskmd as ref

DEFAULT_VERIFIER_MOUNT = "/verifier"
DEFAULT_ORACLE_MOUNT = "/oracle"
# Where verifier/ and oracle/ can appear. BenchFlow uploads a native package's
# verifier/ to /verifier and a legacy tests/ to /tests, oracle/ to /oracle and
# a legacy solution/ to /solution, and locks those paths from the agent.
VERIFIER_DIRS = {"/verifier": "verifier", "/tests": "tests"}
ORACLE_DIRS = {"/oracle": "oracle", "/solution": "solution"}
# The one Compose file location honored: BenchFlow's compose backends read
# environment/docker-compose.yaml, and sandbox/ becomes environment/.
COMPOSE_PATH = "sandbox/docker-compose.yaml"
SCRIPTED_AGENTS = ("oracle", "nop")
HOST_VAR = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}")
SIZE = re.compile(r"^(?P<n>\d+(?:\.\d+)?) ?(?P<u>MB|GB|TB)$")
SIZE_MB = {"MB": 1, "GB": 1024, "TB": 1024 * 1024}  # 1 GB is 1024 MB, as in Docker
SERVICE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")  # Compose's grammar
MODEL_ROLES = ("llm", "vlm", "agent", "panel")
HONORED_EVIDENCE = ("trajectory", "trajectory:reasoning", "tests")


@dataclass(frozen=True)
class Finding:
    """One field and what BenchFlow does with it, or why it cannot."""

    field: str
    detail: str

    def __str__(self) -> str:
        return f"{self.field}: {self.detail}"


@dataclass
class JudgingPlan:
    """The rubric grading the ``taskmd`` verifier strategy performs."""

    rubric: dict[str, Any] | None
    # Model roles the merged rubric uses, with their resolved settings.
    roles: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Shared rubrics that ``extends`` reaches: URL -> local file.
    shared: dict[str, Path] = field(default_factory=dict)
    # Files outside verifier/ that judge prompts read (reference values).
    extra_files: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """How BenchFlow runs one draft 2 package."""

    task_dir: Path
    document: Any
    refused: list[Finding] = field(default_factory=list)
    agent_refused: list[Finding] = field(default_factory=list)
    honored: list[Finding] = field(default_factory=list)
    recorded: list[Finding] = field(default_factory=list)
    frontmatter: dict[str, Any] = field(default_factory=dict)
    verifier_dirname: str = "verifier"
    oracle_dirname: str = "oracle"
    image: str | None = None
    compose: dict[str, Any] | None = None
    compose_file: str | None = None
    judging: JudgingPlan | None = None
    # Prompts delivered after the instruction, in order (stages, M3).
    turns: list[dict[str, str]] = field(default_factory=list)
    # The verifier's effective network ("none", "open", or a list of hosts).
    verifier_network: Any = "open"
    family: dict[str, Any] | None = None
    # [agent] user: the oracle runs as it, and an agent must run as it.
    agent_user: Any = None

    @property
    def ok(self) -> bool:
        return not self.refused

    def refuse(self, where: str, why: str) -> None:
        self.refused.append(Finding(where, why))

    def refuse_agent(self, where: str, why: str) -> None:
        self.agent_refused.append(Finding(where, why))

    def honor(self, where: str, how: str) -> None:
        self.honored.append(Finding(where, how))

    def record(self, where: str, how: str) -> None:
        self.recorded.append(Finding(where, how))


# The support table ---------------------------------------------------------------------------------
# One row per documented key pattern (the reference tools' KEY_DOCS, as docs/keys.md shows them):
# (pattern, status, detail). status is "honored", "refused", "agent-refused", "partial", or
# "recorded". This table is documentation and a coverage check; the decisions are made by the
# walk below, which refuses anything it does not name.

HONORED, REFUSED, PARTIAL, AGENT_REFUSED, RECORDED = (
    "honored",
    "refused",
    "partial",
    "agent-refused",
    "recorded",
)

SUPPORT: list[tuple[str, str, str]] = [
    ("name, version, description, authors, keywords", RECORDED, "native task.name, version, description, authors, keywords"),
    ("title", RECORDED, "metadata.title"),
    ("[about] <key>", RECORDED, "native metadata"),
    ("[agent] timeout", HONORED, "agent.timeout_sec"),
    ("[agent] on_timeout", PARTIAL, '"grade" (BenchFlow grades what the agent left and records timed_out); "grade-flagged" and "fail" are refused'),
    ("[agent] budget, [agent.budget] tool_calls, tokens", AGENT_REFUSED, "BenchFlow enforces no tool-call or token budget on an agent; a scripted seat never approaches one"),
    ("[agent] user", AGENT_REFUSED, "the oracle runs as it; an agent runs as the run's --sandbox-user, so a different user is refused"),
    ("[agent] network", PARTIAL, "equal to [sandbox] network, or a host list over an open sandbox (an agent allowlist); anything else is refused"),
    ("[agent] network_reason", HONORED, "reviewer documentation: nothing to do at run time"),
    ("[agent] system_prompt_append", AGENT_REFUSED, "BenchFlow's harnesses take no system prompt addition"),
    ("[agent] timeout_basis", PARTIAL, '"wall" (BenchFlow counts wall time); "environment" is refused'),
    ("[sandbox] image", HONORED, "sandbox.docker_image, or FROM in environment/Dockerfile"),
    ("[sandbox] os", PARTIAL, '"linux"; "windows" is refused'),
    ("[sandbox] cpus, memory, disk", HONORED, "sandbox.cpus, memory_mb, storage_mb (a whole number of CPUs)"),
    ("[sandbox] build_timeout", HONORED, "sandbox.build_timeout_sec"),
    ("[sandbox] gpus", HONORED, "sandbox.gpus"),
    ("[sandbox] gpu_types, tpu", PARTIAL, "gpu_types to sandbox.gpu_types; tpu is refused"),
    ("[sandbox] network", PARTIAL, '"none" (no-network), "open" (public), a host list (allowlist); { block = [...] } is refused'),
    ("[sandbox] workdir", HONORED, "sandbox.workdir"),
    ("[sandbox] env", PARTIAL, "literal values to sandbox.env; ${VAR} values are refused"),
    ("[sandbox] skills", REFUSED, "BenchFlow installs task skills only in its with-skill mode"),
    ("[sandbox] mcp", REFUSED, "not mapped yet"),
    ("[sandbox] mounts", REFUSED, "task files are not mounted at start yet"),
    ("[sandbox] services, [[sandbox.services]] name, image, build, command, env, ready", PARTIAL, "Compose services beside main; refused with network = \"none\", or a build folder outside sandbox/"),
    ("[sandbox] compose", PARTIAL, "only sandbox/docker-compose.yaml"),
    ("[sandbox] ready, [sandbox.ready] run, interval, timeout, start_period, start_interval, retries", HONORED, "sandbox.healthcheck"),
    ("[sandbox] outputs, [[sandbox.outputs]] path, save_as, exclude", HONORED, "native artifacts; the judges read the saved outputs"),
    ("[[sandbox.outputs]] max_bytes, service", REFUSED, "per-output caps and service outputs are not implemented"),
    ("[sandbox] boundary", PARTIAL, '"container"; gvisor, microvm, and vm are refused'),
    ("[sandbox] clock, [sandbox.clock] start, advance, enforce", REFUSED, "BenchFlow sets no clock"),
    ("[sandbox] timezone", REFUSED, "BenchFlow cannot check that the image has the zone's data"),
    ("[verifier] timeout", HONORED, "verifier.timeout_sec"),
    ("[verifier] user", HONORED, "verifier.user"),
    ("[verifier] env", PARTIAL, "literal values to verifier.env; ${VAR} values are refused"),
    ("[verifier] network", PARTIAL, "equal to [sandbox] network; a shared verifier is taken offline for \"none\""),
    ("[verifier] isolation", HONORED, '"shared", or "separate" (verifier.sandbox_mode: separate)'),
    ("[verifier] sandbox, [verifier.sandbox] <key>", PARTIAL, "image, cpus, memory, disk, workdir, env, build_timeout; the rest is refused"),
    ("[verifier] snapshot, [[verifier.snapshot]] run, reads, service, timeout, user", REFUSED, "snapshot commands are not run"),
    ("[verifier] combine_stages, unreached_stages", HONORED, "no stage is graded on its own, so the task's verifier alone decides the reward"),
    ("[verifier] mount", PARTIAL, '"/verifier" or "/tests"'),
    ("[verifier] feedback", HONORED, "BenchFlow shows an agent none of its review"),
    ("[verifier] models", PARTIAL, "false; true (a script that calls a model) is refused"),
    ("[verifier] judges and its role tables", PARTIAL, "llm and agent roles run judge-loop@1 (docs/task-authoring-taskmd-v2.md); vlm, panel, effort, a hosted harness, resources, services are refused"),
    ("[verifier] human, [verifier.human] <key>", REFUSED, "human assessment is not supported"),
    ("[oracle] env", PARTIAL, "literal values to oracle.env"),
    ("[oracle] mount", PARTIAL, '"/oracle" or "/solution"'),
    ("[world], [tiers], [conventions] (every key)", REFUSED, "worlds, tiers, and conventions are not provided"),
    ("[stages.<name>] unlock", PARTIAL, "a chain from the instruction (at_start, on_submit, after:<previous>) becomes turns of one session; on_request and at_turn are refused"),
    ("[stages.<name>] submit, mounts, agent, verifier, gate, ready, outputs", REFUSED, "stages graded or set up on their own are not implemented"),
    ("[roles.<name>] <key>, [interaction] <key>", REFUSED, "multi-agent roles are not mapped (the spec leaves each role's prompt and order open)"),
    ("[user] <key>", REFUSED, "simulated users are not mapped"),
    ("[variants.<name>] <key>", RECORDED, "BenchFlow runs the base task"),
    ("[matrix] <key>", REFUSED, "condition grids are not run"),
    ("[family] generator, seed_param, params, splits, database", HONORED, "family@1 per seed (--seeds): the generator runs in a fresh container of the task's image"),
    ("[training] reward, learnable_band, difficulty, admit", RECORDED, "metadata"),
    ("[training] fork", HONORED, "a bench eval run sets every episode up from scratch"),
    ("[training] max_concurrent", REFUSED, "not enforced across a run's trials yet"),
    ("[trajectory] require", PARTIAL, '"tool-calls"; the rest is refused'),
    ("[preference] <key>", REFUSED, "the human-preference protocol is not run"),
    ("[integrity] profile", PARTIAL, '"shared-hardened", or "separated-verifier" with isolation = "separate"; "runner-separated" is refused'),
    ("[integrity] intended_use", HONORED, "a BenchFlow eval run is not training"),
    ("[integrity] canary, threat_model, residual_risks", RECORDED, "metadata"),
    ("[integrity] forbidden_sources", REFUSED, "not blocked yet"),
    ("[integrity] resources, [integrity.resources] <key>", REFUSED, "per-path resource classes are not enforced"),
    ("[integrity] answers", HONORED, "judges are never served files the verifier writes"),
    ("[integrity] controls and its keys", RECORDED, "scoring rule, not checked; controls run as control variants (Python API) and --agent nop"),
    ("[runs] <key>", RECORDED, "scoring rule, not checked"),
    ("[[credits]], [provenance], [import.<format>]", RECORDED, "metadata"),
    ("x- keys and [x-<name>]", PARTIAL, "metadata; [x-benchflow] is refused"),
]


# Values ---------------------------------------------------------------------------------------------


def seconds(value: Any) -> float | None:
    """A task.md duration in seconds, or None."""
    return ref.duration_s(value)


def megabytes(value: Any) -> int | None:
    """A task.md size in whole megabytes (1 GB = 1024 MB), rounded up, or None."""
    match = SIZE.match(value) if isinstance(value, str) else None
    if not match:
        return None
    return math.ceil(float(match["n"]) * SIZE_MB[match["u"]])


def _table(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _literal_env(plan: Plan, where: str, env: Any) -> dict[str, str] | None:
    """An env table as strings, refusing host templates BenchFlow would resolve unchecked."""
    if not isinstance(env, dict):
        plan.refuse(where, "must be a table of strings")
        return None
    out: dict[str, str] = {}
    for key, value in env.items():
        text = value if isinstance(value, str) else None
        if text is None:
            plan.refuse(f"{where}.{key}", "must be a string")
            continue
        if HOST_VAR.search(text):
            plan.refuse(
                f"{where}.{key}",
                "${VAR} resolves from the host only for variables the operator allows, "
                "and never for a provider key; BenchFlow resolves any host variable, so "
                "templated values are refused",
            )
            continue
        out[str(key)] = text
    return out


def _author(value: Any) -> dict[str, str] | None:
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        out = {"name": value["name"]}
        if isinstance(value.get("email"), str):
            out["email"] = value["email"]
        return out
    if isinstance(value, str):
        match = re.match(r"^(?P<name>[^<>]*?)(?: <(?P<email>[^<>@\s]+@[^<>\s]+)>)?$", value)
        if match and match["email"]:
            return {"name": match["name"], "email": match["email"]}
        return {"name": value}
    return None


def network_setting(value: Any) -> tuple[str, list[str] | None] | None:
    """A task.md network as (native network_mode, allowed_hosts), or None when it has no native form."""
    if value == "none":
        return "no-network", None
    if value == "open":
        return "public", None
    if isinstance(value, list) and value and all(isinstance(h, str) for h in value):
        return "allowlist", list(value)
    return None


# The walk -------------------------------------------------------------------------------------------


class _Walker:
    def __init__(self, plan: Plan, config: dict[str, Any]) -> None:
        self.plan = plan
        self.config = config

    def unknown(self, where: str) -> None:
        self.plan.refuse(where, "BenchFlow does not implement this setting yet")

    def x_keys(self, table: dict[str, Any], where: str) -> list[str]:
        """Record a table's x- keys (metadata to the spec) and return the rest."""
        rest = []
        for key in table:
            if isinstance(key, str) and key.startswith("x-"):
                self.plan.record(f"{where} {key}", "an extension, metadata to the spec")
            else:
                rest.append(key)
        return rest


def plan_package(document: Any, task_dir: Path) -> Plan:
    """Decide how BenchFlow runs a parsed draft 2 package.

    ``document`` is the reference parser's ``TaskDocument``; it must have no
    error diagnostics (the caller refuses those first).
    """
    config = document.config if isinstance(document.config, dict) else {}
    plan = Plan(task_dir=task_dir, document=document)
    walker = _Walker(plan, config)
    fm: dict[str, Any] = {"schema_version": "1.3"}
    plan.frontmatter = fm
    metadata: dict[str, Any] = {}

    _identity(plan, config, fm, metadata)
    sandbox_network = _sandbox(walker, _table(config.get("sandbox")), fm)
    _agent(walker, _table(config.get("agent")), fm, sandbox_network)
    _verifier(walker, _table(config.get("verifier")), fm, sandbox_network)
    _oracle(walker, _table(config.get("oracle")), fm)
    for key, value in config.items():
        if key in ref.IDENTITY or key in ("sandbox", "agent", "verifier", "oracle"):
            continue
        handler = _TOP_LEVEL.get(key)
        if handler is not None:
            handler(walker, value, metadata)
        elif isinstance(key, str) and key.startswith("x-"):
            if key == "x-benchflow" and value:
                plan.refuse(
                    "[x-benchflow]",
                    "BenchFlow owns this extension and would have to honor it, and it "
                    "does not read v0.6 settings from a draft 2 package",
                )
            else:
                plan.record(f"[{key}]", "an extension, metadata to the spec")
        else:
            walker.unknown(f"[{key}]")
    _blocks(plan, document)
    _integrity_profile(plan, config)
    _package_files(plan, task_dir, config)
    plan.judging = _judging(plan, document, task_dir, config)
    if metadata:
        fm["metadata"] = metadata
    return plan


def _identity(plan: Plan, config: dict[str, Any], fm: dict[str, Any], metadata: dict[str, Any]) -> None:
    task: dict[str, Any] = {}
    name = config.get("name")
    if isinstance(name, str) and re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", name) and ".." not in name and not name.startswith("."):
        task["name"] = name
    elif name is not None:
        plan.record("name", f"{name!r} is not a native org/name, so it stays in metadata.taskmd only")
    for key in ("version", "description"):
        if isinstance(config.get(key), str):
            task[key] = config[key]
    authors = [a for a in (_author(x) for x in config.get("authors") or []) if a]
    if authors:
        task["authors"] = authors
    if isinstance(config.get("keywords"), list):
        task["keywords"] = [str(k) for k in config["keywords"]]
    if task and "name" in task:
        fm["task"] = task
    elif task:
        plan.record("version, description, authors, keywords", "native keeps these beside a name, so they stay in metadata.taskmd")
    if isinstance(config.get("title"), str):
        metadata["title"] = config["title"]
    about = _table(config.get("about"))
    for key, value in about.items():
        if key in ("taskmd", "embodied", "embodiment", "title"):
            metadata.setdefault("taskmd_about", {})[key] = value
        else:
            metadata[key] = value
    for key in ref.IDENTITY:
        if key in config:
            plan.record(key, "metadata")
    if about:
        plan.record("[about]", "metadata, in native metadata")


def _sandbox(walker: _Walker, sandbox: dict[str, Any], fm: dict[str, Any]) -> Any:
    """[sandbox]; returns the sandbox's effective network."""
    plan = walker.plan
    out: dict[str, Any] = {}
    network = sandbox.get("network", "open")
    for key in walker.x_keys(sandbox, "[sandbox]"):
        value = sandbox[key]
        where = f"[sandbox] {key}"
        if key == "image":
            plan.image = str(value)
            plan.honor(where, "the agent's image")
        elif key == "os":
            if value == "linux":
                plan.honor(where, "linux")
            else:
                plan.refuse(where, f"{value!r} sandboxes are not provided; BenchFlow runs linux containers")
        elif key == "cpus":
            if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
                out["cpus"] = value
                plan.honor(where, "sandbox.cpus")
            else:
                plan.refuse(where, "BenchFlow allocates a whole number of CPUs, at least 1")
        elif key in ("memory", "disk"):
            mb = megabytes(value)
            if mb is None:
                plan.refuse(where, "not a size")
            else:
                out["memory_mb" if key == "memory" else "storage_mb"] = mb
                plan.honor(where, f"sandbox.{'memory_mb' if key == 'memory' else 'storage_mb'} = {mb}")
        elif key == "build_timeout":
            secs = seconds(value)
            if secs is None:
                plan.refuse(where, "not a duration")
            else:
                out["build_timeout_sec"] = secs
                plan.honor(where, "sandbox.build_timeout_sec")
        elif key == "gpus":
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                out["gpus"] = value
                plan.honor(where, "sandbox.gpus")
            else:
                plan.refuse(where, "a whole number of GPUs")
        elif key == "gpu_types":
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                out["gpu_types"] = list(value)
                plan.honor(where, "sandbox.gpu_types")
            else:
                plan.refuse(where, "a list of GPU type names")
        elif key == "network":
            mapped = network_setting(value)
            if mapped is None:
                plan.refuse(
                    where,
                    "{ block = [...] } (open except the listed hosts) has no BenchFlow form: "
                    "BenchFlow's denylist also blocks every subdomain of a listed host",
                )
            else:
                out["network_mode"], hosts = mapped
                if hosts:
                    out["allowed_hosts"] = hosts
                plan.honor(where, f"sandbox.network_mode = {mapped[0]}")
        elif key == "workdir":
            out["workdir"] = str(value)
            plan.honor(where, "sandbox.workdir")
        elif key == "env":
            env = _literal_env(plan, where, value)
            if env:
                out["env"] = env
                plan.honor(where, "sandbox.env")
        elif key == "ready":
            check = _healthcheck(plan, where, value)
            if check is not None:
                out["healthcheck"] = check
                plan.honor(where, "sandbox.healthcheck, run before the agent starts")
        elif key == "outputs":
            artifacts = _outputs(plan, value, where)
            if artifacts:
                fm["artifacts"] = artifacts
        elif key == "boundary":
            if value == "container":
                plan.honor(where, "every BenchFlow sandbox isolates at least a container")
            else:
                plan.refuse(where, f"{value!r} isolation is not guaranteed by BenchFlow's sandboxes")
        elif key == "services":
            plan.compose = _services(plan, value, network)
        elif key == "compose":
            if value == COMPOSE_PATH:
                plan.compose_file = COMPOSE_PATH
                plan.honor(where, "environment/docker-compose.yaml; its service main is the agent's container")
            else:
                plan.refuse(where, f"only {COMPOSE_PATH} is supported: BenchFlow reads the Compose file beside the agent's Dockerfile")
        elif key in ("skills", "mcp", "mounts", "tpu", "clock", "timezone"):
            plan.refuse(where, _REFUSED_SANDBOX[key])
        else:
            walker.unknown(where)
    if plan.compose is not None and plan.compose_file is not None:
        plan.refuse("[sandbox] services", "[[sandbox.services]] and [sandbox] compose both declare services; declare them in one place")
    if out:
        fm["sandbox"] = out
    return network


_REFUSED_SANDBOX = {
    "skills": "BenchFlow installs a task's skills only in its with-skill mode, and a task.md run always installs them",
    "mcp": "MCP servers are not mapped yet",
    "mounts": "task files are not mounted at start yet",
    "tpu": "TPUs are not mapped yet",
    "clock": "BenchFlow sets no clock: the agent would see the real time",
    "timezone": "BenchFlow cannot check that the image holds the zone's data, and without it the zone falls back to UTC without an error",
}


def _healthcheck(plan: Plan, where: str, value: Any) -> dict[str, Any] | None:
    ready = _table(value)
    out: dict[str, Any] = {"command": ready.get("run")}
    for key, native in (("interval", "interval_sec"), ("timeout", "timeout_sec"), ("start_period", "start_period_sec"), ("start_interval", "start_interval_sec")):
        if key in ready:
            secs = seconds(ready[key])
            if secs is None:
                plan.refuse(f"{where}.{key}", "not a duration")
                return None
            out[native] = secs
    if "retries" in ready:
        out["retries"] = ready["retries"]
    for key in ready:
        if key not in ref.READY_KEYS and not str(key).startswith("x-"):
            plan.refuse(f"{where}.{key}", "not a readiness key")
            return None
    return out


def _outputs(plan: Plan, outputs: Any, where: str) -> list[Any]:
    artifacts: list[Any] = []
    if not isinstance(outputs, list):
        plan.refuse(where, "a list of paths or tables")
        return artifacts
    for n, item in enumerate(outputs):
        at = f"{where}[{n}]"
        if isinstance(item, str):
            artifacts.append(item)
            continue
        entry = _table(item)
        native: dict[str, Any] = {"source": entry.get("path")}
        for key, value in entry.items():
            if key == "path" or str(key).startswith("x-"):
                continue
            if key == "save_as":
                native["destination"] = value
            elif key == "exclude":
                native["exclude"] = list(value) if isinstance(value, list) else [value]
            elif key in ("max_bytes", "service"):
                plan.refuse(f"{at} {key}", "per-output caps and service outputs are not implemented")
            else:
                plan.refuse(f"{at} {key}", "not an output key")
        artifacts.append(native)
    if artifacts:
        plan.honor(where, "native artifacts, saved after the agent's run")
    return artifacts


def _services(plan: Plan, services: Any, network: Any) -> dict[str, Any] | None:
    """[[sandbox.services]] as the Compose services BenchFlow runs beside main."""
    if network == "none":
        plan.refuse(
            "[sandbox] services",
            'with network = "none" BenchFlow cuts all of the agent container\'s networking, '
            "so the agent could not reach its services",
        )
        return None
    out: dict[str, Any] = {}
    waits: dict[str, Any] = {}
    for service in services if isinstance(services, list) else []:
        name = str(service.get("name"))
        where = f"[sandbox.services.{name}]"
        if not SERVICE_NAME.match(name):
            plan.refuse(where, "not a Compose service name")
            continue
        entry: dict[str, Any] = {}
        for key, value in service.items():
            if key == "name" or str(key).startswith("x-"):
                continue
            if key == "image":
                entry["image"] = str(value)
            elif key == "build":
                path = PurePosixPath(str(value))
                if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("sandbox",):
                    plan.refuse(f"{where} build", "must be a folder in sandbox/, which becomes the Compose build context")
                    continue
                entry["build"] = PurePosixPath(*path.parts[1:]).as_posix() or "."
            elif key == "command":
                entry["command"] = value
            elif key == "env":
                env = _literal_env(plan, f"{where} env", value)
                if env is not None:
                    entry["environment"] = env
            elif key == "ready":
                check = _healthcheck(plan, f"{where} ready", value)
                if check is not None:
                    entry["healthcheck"] = {
                        "test": ["CMD-SHELL", check["command"]],
                        "interval": f"{round(check.get('interval_sec', 5.0) * 1000)}ms",
                        "timeout": f"{round(check.get('timeout_sec', 30.0) * 1000)}ms",
                        "retries": check.get("retries", 3),
                    }
                    if check.get("start_period_sec"):
                        entry["healthcheck"]["start_period"] = f"{round(check['start_period_sec'] * 1000)}ms"
                    if "start_interval_sec" in check:
                        entry["healthcheck"]["start_interval"] = f"{round(check['start_interval_sec'] * 1000)}ms"
            else:
                plan.refuse(f"{where} {key}", "not a service key")
        out[name] = entry
        waits[name] = {"condition": "service_healthy" if "ready" in service else "service_started"}
    if not out:
        return None
    plan.honor("[sandbox] services", "Compose services beside main (environment/docker-compose.yaml)")
    return {"services": {"main": {"depends_on": waits}, **out}}


def _agent(walker: _Walker, agent: dict[str, Any], fm: dict[str, Any], sandbox_network: Any) -> None:
    plan = walker.plan
    out: dict[str, Any] = {}
    for key in walker.x_keys(agent, "[agent]"):
        value = agent[key]
        where = f"[agent] {key}"
        if key == "timeout":
            secs = seconds(value)
            if secs is None:
                plan.refuse(where, "not a duration")
            else:
                out["timeout_sec"] = secs
                plan.honor(where, "agent.timeout_sec")
        elif key == "on_timeout":
            if value == "grade":
                plan.honor(where, "BenchFlow grades what the agent left at the time limit and records timed_out")
            else:
                plan.refuse(where, f"{value!r} is not implemented: BenchFlow grades what the agent left at the time limit")
        elif key == "budget":
            plan.refuse_agent(where, "BenchFlow enforces no tool-call or token budget on an agent")
        elif key == "user":
            plan.agent_user = value
            out["user"] = value
            plan.honor(where, "the oracle and controls run as it; an agent must run with --sandbox-user set to it")
        elif key == "network":
            if value == sandbox_network:
                plan.honor(where, "the sandbox's network")
            elif sandbox_network == "open" and isinstance(value, list) and value:
                out["network_mode"] = "allowlist"
                out["allowed_hosts"] = list(value)
                plan.honor(where, "an agent allowlist over an open sandbox (agent.network_mode = allowlist)")
            else:
                plan.refuse(
                    where,
                    f"{value!r} differs from [sandbox] network ({sandbox_network!r}); BenchFlow enforces "
                    "the sandbox's network, and per phase only an agent allowlist over an open sandbox",
                )
        elif key == "network_reason":
            plan.honor(where, "reviewer documentation; nothing to do at run time")
        elif key == "system_prompt_append":
            plan.refuse_agent(where, "BenchFlow's harnesses take no system prompt addition")
        elif key == "timeout_basis":
            if value == "wall":
                plan.honor(where, "BenchFlow counts wall time")
            else:
                plan.refuse(where, "BenchFlow counts only wall time, and cannot freeze the sandbox while a policy generates")
        else:
            walker.unknown(where)
    if out:
        fm["agent"] = out


def _verifier(walker: _Walker, verifier: dict[str, Any], fm: dict[str, Any], sandbox_network: Any) -> None:
    plan = walker.plan
    out: dict[str, Any] = {}
    isolation = verifier.get("isolation", "shared")
    network = verifier.get("network", sandbox_network)
    plan.verifier_network = network
    for key in walker.x_keys(verifier, "[verifier]"):
        value = verifier[key]
        where = f"[verifier] {key}"
        if key == "timeout":
            secs = seconds(value)
            if secs is None:
                plan.refuse(where, "not a duration")
            else:
                out["timeout_sec"] = secs
                plan.honor(where, "verifier.timeout_sec; it bounds the scripts, and each judge session has its own")
        elif key == "user":
            out["user"] = value
            plan.honor(where, "verifier.user")
        elif key == "env":
            env = _literal_env(plan, where, value)
            if env:
                out["env"] = env
                plan.honor(where, "verifier.env")
        elif key == "network":
            if value == sandbox_network:
                plan.honor(where, "the sandbox's network")
            else:
                plan.refuse(where, f"{value!r} differs from [sandbox] network ({sandbox_network!r}); BenchFlow gives a verifier the sandbox's network")
        elif key == "isolation":
            if value == "separate":
                out["sandbox_mode"] = "separate"
                plan.honor(where, "verifier.sandbox_mode = separate: a fresh container with only the saved outputs")
            else:
                plan.honor(where, "the agent's container, after BenchFlow's hardening")
        elif key == "sandbox":
            vs = _verifier_sandbox(plan, _table(value), sandbox_network)
            if vs:
                out["sandbox"] = vs
        elif key == "snapshot":
            plan.refuse(where, "snapshot commands are not run before grading")
        elif key in ("combine_stages", "unreached_stages"):
            plan.honor(where, "no stage is graded on its own, so the task's verifier alone decides the reward")
        elif key == "mount":
            dirname = VERIFIER_DIRS.get(str(value))
            if dirname is None:
                plan.refuse(where, f"{value!r} is not supported: BenchFlow places the verifier at /verifier or /tests, the paths it locks from the agent")
            else:
                plan.verifier_dirname = dirname
                plan.honor(where, f"the package's {dirname}/ folder")
        elif key == "feedback":
            plan.honor(where, "BenchFlow shows an agent none of its review")
        elif key == "models":
            if value is True:
                plan.refuse(where, "a verifier script that calls a model needs the model proxy's seat token, which BenchFlow does not provide yet")
            else:
                plan.honor(where, "the verifier's scripts call no model")
        elif key == "judges":
            pass  # decided with the rubric (_judging)
        elif key == "human":
            plan.refuse(where, "human assessment is not supported")
        else:
            walker.unknown(where)
    if isolation == "separate" and isinstance(verifier.get("sandbox"), dict) and "image" not in verifier["sandbox"]:
        plan.refuse("[verifier.sandbox]", "without an image, BenchFlow would build the verifier's sandbox from verifier/Dockerfile only; give it an image")
    if out:
        fm["verifier"] = out


def _verifier_sandbox(plan: Plan, table: dict[str, Any], sandbox_network: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in table.items():
        where = f"[verifier.sandbox] {key}"
        if str(key).startswith("x-"):
            plan.record(where, "an extension, metadata to the spec")
        elif key == "image":
            out["docker_image"] = str(value)
            plan.honor(where, "verifier.sandbox.docker_image")
        elif key == "cpus" and isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            out["cpus"] = value
            plan.honor(where, "verifier.sandbox.cpus")
        elif key in ("memory", "disk") and megabytes(value) is not None:
            out["memory_mb" if key == "memory" else "storage_mb"] = megabytes(value)
            plan.honor(where, f"verifier.sandbox.{key}")
        elif key == "workdir":
            out["workdir"] = str(value)
            plan.honor(where, "verifier.sandbox.workdir")
        elif key == "env":
            env = _literal_env(plan, where, value)
            if env:
                out["env"] = env
                plan.honor(where, "verifier.sandbox.env")
        elif key == "build_timeout" and seconds(value) is not None:
            out["build_timeout_sec"] = seconds(value)
            plan.honor(where, "verifier.sandbox.build_timeout_sec")
        elif key == "network" and value == sandbox_network:
            plan.honor(where, "the sandbox's network")
        else:
            plan.refuse(where, "not mapped for a verifier's own sandbox yet")
    mapped = network_setting(sandbox_network)
    if out and mapped is not None:
        out.setdefault("network_mode", mapped[0])
        if mapped[1]:
            out["allowed_hosts"] = mapped[1]
    return out


def _oracle(walker: _Walker, oracle: dict[str, Any], fm: dict[str, Any]) -> None:
    plan = walker.plan
    out: dict[str, Any] = {}
    for key in walker.x_keys(oracle, "[oracle]"):
        value = oracle[key]
        where = f"[oracle] {key}"
        if key == "env":
            env = _literal_env(plan, where, value)
            if env:
                out["env"] = env
                plan.honor(where, "oracle.env")
        elif key == "mount":
            dirname = ORACLE_DIRS.get(str(value))
            if dirname is None:
                plan.refuse(where, f"{value!r} is not supported: BenchFlow places the oracle at /oracle or /solution")
            else:
                plan.oracle_dirname = dirname
                plan.honor(where, f"the package's {dirname}/ folder")
        else:
            walker.unknown(where)
    if out:
        fm["oracle"] = out


# Top-level tables other than [sandbox], [agent], [verifier], [oracle] ---------------------------------


def _about(walker: _Walker, value: Any, metadata: dict[str, Any]) -> None:
    del walker, value, metadata  # handled by _identity


def _refuse_table(table: str, reason: str) -> Any:
    def handler(walker: _Walker, value: Any, metadata: dict[str, Any]) -> None:
        del value, metadata
        walker.plan.refuse(f"[{table}]", reason)

    return handler


def _record_table(table: str, detail: str) -> Any:
    def handler(walker: _Walker, value: Any, metadata: dict[str, Any]) -> None:
        del value, metadata
        walker.plan.record(f"[{table}]", detail)

    return handler


def _stages(walker: _Walker, stages: Any, metadata: dict[str, Any]) -> None:
    del metadata  # turns are written by _blocks, which sees the stage prompts
    plan = walker.plan
    for name, stage in _table(stages).items():
        for key in _table(stage):
            where = f"[stages.{name}] {key}"
            if key == "unlock" or str(key).startswith("x-"):
                continue
            plan.refuse(where, "stages graded, set up, or unlocked by a file on their own are not implemented; a stage maps only as a prompt revealed after the previous one")


def _variants(walker: _Walker, variants: Any, metadata: dict[str, Any]) -> None:
    del metadata
    for name in _table(variants):
        walker.plan.record(f"[variants.{name}]", "BenchFlow runs the base task; selecting a variant is not implemented")


def _family(walker: _Walker, family: Any, metadata: dict[str, Any]) -> None:
    del metadata
    plan = walker.plan
    table = _table(family)
    plan.family = table
    for key in walker.x_keys(table, "[family]"):
        where = f"[family] {key}"
        if key in ("generator", "seed_param", "params", "splits"):
            plan.honor(where, "family@1, one package per seed (bench eval run --seeds)")
        elif key == "database":
            plan.honor(where, "not read under family@1, which writes an instance's files into agent/ and verifier/")
        else:
            walker.unknown(where)


def _training(walker: _Walker, training: Any, metadata: dict[str, Any]) -> None:
    del metadata
    plan = walker.plan
    for key in walker.x_keys(_table(training), "[training]"):
        where = f"[training] {key}"
        if key in ("reward", "learnable_band", "difficulty", "admit"):
            plan.record(where, "metadata")
        elif key == "fork":
            plan.honor(where, "bench eval run sets every episode up from scratch (bench eval branch does not check it yet)")
        elif key == "max_concurrent":
            plan.refuse(where, "BenchFlow does not limit how many trials of one task run at once yet")
        else:
            walker.unknown(where)


def _trajectory(walker: _Walker, table: Any, metadata: dict[str, Any]) -> None:
    del metadata
    plan = walker.plan
    for key in walker.x_keys(_table(table), "[trajectory]"):
        value = _table(table)[key]
        where = f"[trajectory] {key}"
        if key != "require":
            walker.unknown(where)
            continue
        for item in value if isinstance(value, list) else [value]:
            if item == "tool-calls":
                plan.honor(f"{where} tool-calls", "BenchFlow records every tool call")
            else:
                plan.refuse(f"{where} {item}", "BenchFlow cannot guarantee a trajectory holds this")


def _integrity(walker: _Walker, table: Any, metadata: dict[str, Any]) -> None:
    del metadata
    plan = walker.plan
    integrity = _table(table)
    for key in walker.x_keys(integrity, "[integrity]"):
        where = f"[integrity] {key}"
        if key == "profile":
            pass  # _integrity_profile, which needs the verifier's isolation
        elif key == "intended_use":
            plan.honor(where, "a BenchFlow eval run is not training")
        elif key in ("canary", "threat_model", "residual_risks"):
            plan.record(where, "metadata")
        elif key == "forbidden_sources":
            plan.refuse(where, "BenchFlow does not block these sources yet")
        elif key == "resources":
            plan.refuse(where, "per-path resource classes are not enforced")
        elif key == "answers":
            plan.honor(where, "judges are never served files the verifier writes, so no answer reaches them that way")
        elif key == "controls":
            plan.record(where, "a scoring rule, not checked by a development run; control scripts run as control variants (benchflow.taskmd), and --agent nop is the do-nothing run")
        else:
            walker.unknown(where)


def _runs(walker: _Walker, table: Any, metadata: dict[str, Any]) -> None:
    del metadata
    for key in _table(table):
        walker.plan.record(f"[runs] {key}", "a scoring rule, recorded as not checked: no BenchFlow run claims to be scored")


ROLES_REFUSED = (
    "multi-agent roles are not mapped: task.md leaves open what each role is told "
    "and in which order roles act"
)
USER_REFUSED = (
    "simulated users are not mapped: BenchFlow's user loop starts a fresh agent "
    "session each round and runs the verifier between rounds"
)

_TOP_LEVEL: dict[str, Any] = {
    "about": _about,
    "world": _refuse_table("world", "worlds (desktop, browser, simulator, robot, lab) are not provided"),
    "tiers": _refuse_table("tiers", "sim-to-real tiers are not provided"),
    "conventions": _refuse_table("conventions", "conventions are read by a world's profile, and BenchFlow provides no world"),
    "stages": _stages,
    "roles": _refuse_table("roles", ROLES_REFUSED),
    "user": _refuse_table("user", USER_REFUSED),
    "interaction": _refuse_table("interaction", "multi-agent interaction is not mapped"),
    "variants": _variants,
    "matrix": _refuse_table("matrix", "condition matrices are not run: BenchFlow would run only the base task"),
    "family": _family,
    "training": _training,
    "trajectory": _trajectory,
    "preference": _refuse_table("preference", "the human-preference protocol is not run"),
    "integrity": _integrity,
    "runs": _runs,
    "credits": _record_table("credits", "metadata"),
    "provenance": _record_table("provenance", "metadata"),
    "import": _record_table("import", "metadata: settings of an imported task that task.md does not read"),
}


def _integrity_profile(plan: Plan, config: dict[str, Any]) -> None:
    profile = _table(config.get("integrity")).get("profile")
    if profile is None:
        return
    isolation = _table(config.get("verifier")).get("isolation", "shared")
    where = "[integrity] profile"
    if profile == "shared-hardened":
        plan.honor(where, "BenchFlow hardens a shared verifier's container (kills the agent's processes, cleans test hooks, wipes reward files)")
    elif profile == "separated-verifier" and isolation == "separate":
        plan.honor(where, "a separate verifier sandbox")
    elif profile == "separated-verifier":
        plan.refuse(where, 'needs [verifier] isolation = "separate"')
    else:
        plan.refuse(where, f"{profile!r} is not provided")


def _blocks(plan: Plan, document: Any) -> None:
    """Stage, role, user, and notes blocks, and the canary comment."""
    config = document.config if isinstance(document.config, dict) else {}
    stages = _table(config.get("stages"))
    prompts = {b.arg: ref.prompt(b) for b in document.blocks if b.kind == "stage"}
    for block in document.blocks:
        if block.kind == "notes":
            plan.record("```notes", "author and reviewer notes, never shown to an agent")
        elif block.kind == "role":
            plan.refuse(f"```role {block.arg}", ROLES_REFUSED)
        elif block.kind == "user":
            plan.refuse("```user", USER_REFUSED)
    if document.canary:
        plan.record("canary comment", "stripped from the agent's view")
    if not stages:
        return
    order = list(stages)
    previous: str | None = None
    for index, name in enumerate(order):
        unlock = _table(stages[name]).get("unlock")
        if unlock == "at_start":
            if index != 0:
                plan.refuse(f"[stages.{name}] unlock", "at_start must be the first stage")
            previous = name
            plan.honor(f"[stages.{name}] unlock", "at_start: its prompt is the instruction")
            continue
        chained = unlock == "on_submit" or (isinstance(unlock, str) and unlock == f"after:{previous}")
        if previous is None and isinstance(unlock, str) and unlock.startswith("after:"):
            chained = False
        if not chained:
            plan.refuse(
                f"[stages.{name}] unlock",
                f"{unlock!r} is not mapped: BenchFlow reveals a stage only after the agent ends its previous turn, "
                "so stages must unlock in declaration order (on_submit, or after:<the stage before>)",
            )
            continue
        plan.turns.append({"stage": name, "prompt": prompts.get(name, "")})
        plan.honor(f"[stages.{name}] unlock", "a new turn of the same agent session, after the agent ends its previous turn")
        previous = name


def _package_files(plan: Plan, task_dir: Path, config: dict[str, Any]) -> None:
    if (task_dir / "stages").is_dir():
        plan.refuse("stages/", "stages graded on their own (stages/<name>/verifier/, oracle/) are not implemented")
    if (task_dir / "world").is_dir() and "world" not in config:
        plan.record("world/", "unused: the task declares no [world]")
    has_sandbox = (task_dir / "sandbox" / "Dockerfile").is_file()
    if plan.image is None and not has_sandbox:
        plan.refuse("sandbox/Dockerfile", "the task names no [sandbox] image and ships no sandbox/Dockerfile, so there is nothing to run the agent in")
    compose = task_dir / COMPOSE_PATH
    if compose.is_file() and plan.compose_file is None:
        plan.refuse(COMPOSE_PATH, "BenchFlow would start the services in this file beside the agent's container, but [sandbox] compose does not declare it")
    elif plan.compose_file is not None and not compose.is_file():
        plan.refuse("[sandbox] compose", f"names {COMPOSE_PATH}, which the package does not have")
    if plan.compose is not None and (task_dir / "sandbox" / "docker-compose.yaml").is_file():
        plan.refuse("[sandbox] services", "sandbox/docker-compose.yaml exists too; BenchFlow runs one Compose file")
    for service in _table(config.get("sandbox")).get("services") or []:
        build = service.get("build") if isinstance(service, dict) else None
        if isinstance(build, str) and not (task_dir / build / "Dockerfile").is_file():
            plan.refuse(f"[sandbox.services.{service.get('name')}] build", f"{build} has no Dockerfile")
    verifier = task_dir / "verifier"
    if (verifier / "verifier.md").is_file():
        plan.refuse("verifier/verifier.md", "a BenchFlow verifier.md would replace task.md's verifier contract")
    if (verifier / "docker-compose.yaml").is_file() and _table(config.get("verifier")).get("isolation") == "separate":
        plan.refuse("verifier/docker-compose.yaml", "a verifier image built from Compose is not supported")
    oracle = task_dir / "oracle"
    if oracle.is_dir() and not (oracle / "solve.sh").is_file():
        plan.record("oracle/", "no solve.sh, so --agent oracle cannot run")


# Judging (M2) ---------------------------------------------------------------------------------------


def _judging(plan: Plan, document: Any, task_dir: Path, config: dict[str, Any]) -> JudgingPlan:
    rubric = document.rubric if isinstance(document.rubric, dict) else None
    judging = JudgingPlan(rubric=rubric)
    behaviors = document.behaviors if isinstance(document.behaviors, dict) else None
    if behaviors is not None and (behaviors.get("watch") or behaviors.get("paired")):
        plan.refuse("verifier/behaviors.json", "watched and paired behaviors are not detected, and their consequences and tags are not applied")
    judges = _table(_table(config.get("verifier")).get("judges"))
    has_test = (task_dir / "verifier" / "test.sh").is_file()
    if rubric is None:
        if judges:
            plan.refuse("[verifier] judges", "the task has no task.md rubric for a judge to grade")
        if not has_test:
            plan.refuse("verifier/test.sh", "the task has neither a task.md rubric nor a test.sh, so nothing would grade it")
        else:
            plan.honor("verifier/test.sh", "the verifier's script writes the reward itself (reward.txt or reward.json), as in Harbor")
        return judging
    unresolved = []
    for url in rubric.get("extends") or []:
        name = str(url).rstrip("/").rsplit("/", 1)[-1] + ".json"
        try:
            path = jp._shared_path(str(url), None)
        except jp.JudgePromptError:
            path = None
        if path is None or not path.is_file():
            unresolved.append(str(url))
            plan.refuse(
                "verifier/rubric.json extends",
                f"{url} is not at hand: BenchFlow does not fetch shared rubrics; set "
                f"TASKMD_SHARED_RUBRICS to a folder holding {name}",
            )
        else:
            judging.shared[str(url)] = path
    if unresolved:
        return judging
    try:
        criteria = jp.merged_criteria(rubric, {u: str(p) for u, p in judging.shared.items()})
    except jp.JudgePromptError as exc:
        plan.refuse("verifier/rubric.json", str(exc))
        return judging
    streams = _table(_table(config.get("world")).get("record")).get("streams")
    streams = streams if isinstance(streams, list) else []
    tests = [c for c in criteria if c.get("judge") == "test"]
    if tests and not has_test:
        plan.refuse("verifier/test.sh", f"{len(tests)} criteria are decided by tests, and the package has no verifier/test.sh to report them")
    if tests:
        plan.honor("test-judged criteria", "decided from the verifier's CTRF report, verifier/ctrf.json")
    by_role: dict[str, list[dict[str, Any]]] = {}
    for criterion in criteria:
        judge = criterion.get("judge")
        where = f"verifier/rubric.json {criterion.get('id')}"
        if judge == "test":
            continue
        if judge in ("rule", "world", "replay", "human"):
            plan.refuse(where, f"criteria judged by {judge} are not implemented")
            continue
        if judge in ("vlm", "panel"):
            plan.refuse(where, f"the {judge} role is not implemented" + (" (task.md leaves the panel role undefined)" if judge == "panel" else ""))
            continue
        by_role.setdefault(str(judge), []).append(criterion)
        for item in criterion.get("evidence") or []:
            kind = str(item).split("#", 1)[0]
            if kind in HONORED_EVIDENCE:
                continue
            if kind in jp.RUNTIME_KINDS or kind in streams or kind.startswith("diff:"):
                plan.refuse(
                    f"{where} evidence {item}",
                    "not provided yet: BenchFlow serves files, trajectory, trajectory:reasoning, and tests",
                )
    if not by_role:
        if judges:
            plan.record("[verifier] judges", "no criterion goes to a model, so no judge runs")
        return judging
    verifier = _table(config.get("verifier"))
    isolation = verifier.get("isolation", "shared")
    for role, assigned in by_role.items():
        settings, _ = ref.resolve_judge(config, role, None, task_dir)
        judging.roles[role] = settings
        where = f"[verifier.judges.{role}]"
        for key, value in settings.items():
            if key == "effort":
                plan.refuse(f"{where} effort", "reasoning effort is not passed to the judge model yet")
            elif key == "harness" and value != "judge-loop@1":
                plan.refuse(f"{where} harness", f"{value!r}: BenchFlow runs judge-loop@1 only; hosted harnesses are not admitted")
            elif key == "resources" and value:
                plan.refuse(f"{where} resources", "a judge's runner uses the verifier sandbox's resources")
            elif key == "services" and value:
                plan.refuse(f"{where} services", "a judge's runner cannot reach the task's services")
            elif key == "views":
                for view in value or []:
                    if not (isinstance(view, dict) and view.get("format") == "trajectory-1"):
                        plan.refuse(f"{where} views", "only trajectory-1 views are defined")
        if role == "agent":
            if isolation != "separate":
                plan.refuse(where, 'an agent judge\'s commands run in a fresh copy of the submission\'s environment; BenchFlow provides that only with [verifier] isolation = "separate"')
            if plan.verifier_network != "none":
                plan.refuse(where, f'an agent judge\'s runners have no network; the verifier sandbox BenchFlow runs them in has network {plan.verifier_network!r}. Set [sandbox] network = "none"')
            if isinstance(verifier.get("sandbox"), dict) and verifier["sandbox"].get("image"):
                plan.refuse(where, "an agent judge's runner starts from the solver's image, and this verifier runs another image")
            if (task_dir / "verifier" / "Dockerfile").is_file():
                plan.refuse(where, "an agent judge's runner starts from the solver's image, and this verifier builds its own from verifier/Dockerfile")
        plan.honor(where, f"judge-loop@1 over the Anthropic Messages API, {len(assigned)} criteria")
    for criterion in criteria:
        reference = criterion.get("reference")
        if isinstance(reference, str):
            rel = reference.split("#", 1)[0]
            if not rel.startswith("verifier/"):
                judging.extra_files.append(rel)
    return judging
