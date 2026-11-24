"""The operator's credentials a trial puts in its sandbox, and the run flags
that let the riskier ones in.

:func:`in_sandbox` is the one place that decides what counts. Each entry
names a credential by its kind, its route, its variable names (never a
value) and where it is staged. A process in the sandbox can read what is
staged there: task code the agent runs, and, under ``public``, the task's own
processes that run on into the agent's turn as root. Only the proxy on the
host (Docker, ``public``) keeps an API key out of the sandbox: there the agent
holds the proxy's own key and nothing else.

A cloud credential file (the gcloud application default credentials, a user
refresh token or a service-account key) is a cloud account, not an inference
login, so a run that would stage one is refused unless the operator passes
``--allow-cloud-credential-file`` (:func:`refuse_cloud_credential_file`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from benchflow.errors import UserError

API_KEY = "api-key"
SUBSCRIPTION_LOGIN = "subscription-login"
CLOUD_CREDENTIAL_FILE = "cloud-credential-file"

#: The one way to keep an operator's API key out of the sandbox.
HOST_PROXY = "the proxy on the host (Docker, `public`)"

CLOUD_FILE_FLAG = "--allow-cloud-credential-file"
CLOUD_FILE_SETTING = "allow_cloud_credential_file"
TASK_PROCESSES_FLAG = "--allow-task-processes-with-credentials"
TASK_PROCESSES_SETTING = "allow_task_processes_with_credentials"
SCOPED_FLAG = "--scoped-credential"
SCOPED_SETTING = "scoped_credentials"

#: Whether an API key the proxy in the sandbox holds counts: it is in that
#: proxy's process environment, as root.
IN_SANDBOX_PROXY_COUNTS = True

#: The docs' line, said with every warning.
ADVICE = (
    f"With tasks you don't trust, use API keys through {HOST_PROXY}, not a "
    "subscription login; rotate a login that ran such tasks."
)
CLOUD_FILE_LIFTS = (
    "a cloud credential file (the gcloud application default credentials) "
    "is staged in the sandbox, where root can read it"
)

TASK_PROCESSES_LIFTS = (
    "a task process may run in main beside the operator's credentials, "
    "where it can read them"
)


def quiet_main_refusal(entries: Sequence[dict[str, Any]] = ()) -> str:
    """What the quiet-main refusal adds when an operator credential in the
    sandbox is its reason, naming the unscoped *entries*' variables."""
    names = sorted({n for e in entries for n in e["names"]})
    which = f" ({', '.join(names)})" if names else ""
    return (
        f"An operator credential{which} goes into the sandbox, where a task "
        f"process can read it: run the model through {HOST_PROXY}, pass "
        f"{TASK_PROCESSES_FLAG} (run config {TASK_PROCESSES_SETTING}: true) to "
        f"accept it, or, for a credential that is the run's own (a per-run "
        f"token), name it with {SCOPED_FLAG} NAME."
    )


#: The refusal's sentence for no named variable.
QUIET_MAIN_REFUSAL = quiet_main_refusal()

SANDBOX_PROXY_PROCESS = "the sandbox proxy's process (root)"
SANDBOX_PROXY_FOLDER = "the sandbox proxy's runtime folder (root)"
AGENT_ENVIRONMENT = "the agent's environment"
NATIVE_ADC_PATH = "{home}/.config/gcloud/application_default_credentials.json"
#: The files an agent's launcher also writes an API key to, with its value.
_LAUNCHER_KEY_FILES = {
    "OPENAI_API_KEY": ("codex-acp", "{home}/.codex/auth.json"),
}
#: The login files an agent's own login is also written to.
_LOGIN_FILES = {
    "CODEX_AUTH_JSON": ("codex-acp", "{home}/.codex/auth.json"),
    "MUSE_AUTH_JSON": (None, "{home}/.config/muse/auth.json"),
}

# The proxy's own run key, which a role's env can carry from an earlier
# connection.
_MASTER_KEY_ENV = "BENCHFLOW_LITELLM_MASTER_KEY"
# The key a continuation's agent holds for BenchFlow's replay proxy: a
# constant of BenchFlow's, never the operator's.
REPLAY_API_KEY = "sk-runtime-replay"
REPLAY_KEY_ENV = "LLM_API_KEY"

# A login of a subscription, not a pay-as-you-go key.
_SUBSCRIPTION_NAMES = frozenset(
    {
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_OAUTH_TOKEN",
        "CODEX_ACCESS_TOKEN",
        "CODEX_AUTH_JSON",
        "MUSE_AUTH_JSON",
    }
)


class CloudCredentialFileRefused(ValueError, UserError):
    """A run that would stage a cloud credential file without the operator's
    flag; raised before any sandbox exists."""


class ScopedCredentialRefused(ValueError, UserError):
    """A ``--scoped-credential`` name the run can't scope; raised before any
    sandbox exists."""


def literal_grant(value: Any, *, where: str) -> bool:
    """A run's grant as a config file or record gives it: only a boolean true
    grants, and false or no value does not. Anything else (a quoted
    ``"false"``, ``1``) is refused, naming *where*: ``bool("false")`` is
    true."""
    if value is None or isinstance(value, bool):
        return bool(value)
    raise ValueError(f"{where} must be true or false, not {value!r}")


def scoped_names(value: Any, *, where: str = SCOPED_SETTING) -> tuple[str, ...]:
    """The run's ``--scoped-credential`` names as a config file, record or
    object gives them: a list of variable names (none for no value).
    Anything else is refused, naming *where*."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{where} must be a list of variable names")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{where} must be a list of variable names")
        if item not in names:
            names.append(item)
    return tuple(names)


def refuse_unscopable(
    scoped: tuple[str, ...] | list[str],
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
    *,
    task_name: str,
) -> None:
    """Before any sandbox: each ``--scoped-credential`` name must be one the
    run's resolved agent env sets, a credential variable (not a setting such
    as a base URL), and one the agent gets in its environment, not a file (a
    login file, the cloud credential file). Names only, never a value."""
    from benchflow.agents.env import VERTEX_ADC_ENV
    from benchflow.agents.launch_env import (
        credential_policy,
        host_credential_env_names,
    )
    from benchflow.agents.registry import is_scripted_agent

    if not scoped or is_scripted_agent(agent) or agent == "task-runtime":
        return
    policy = credential_policy(agent, model, agent_env)
    files = {*policy.files, *policy.file_only, *_LOGIN_FILES}
    credentials = {*host_credential_env_names(), *policy.launch_names(agent_env)}
    for name in scoped:
        if name in (VERTEX_ADC_ENV, "GOOGLE_APPLICATION_CREDENTIALS"):
            raise ScopedCredentialRefused(
                f"{task_name}: {SCOPED_FLAG} {name}: the cloud credential file "
                f"can't be scoped; it goes into the sandbox only with {CLOUD_FILE_FLAG}."
            )
        if not agent_env.get(name):
            raise ScopedCredentialRefused(
                f"{task_name}: {SCOPED_FLAG} {name}: the run's agent env does not "
                f"set {name}; name a variable the run gives the agent."
            )
        if name in files:
            raise ScopedCredentialRefused(
                f"{task_name}: {SCOPED_FLAG} {name}: {name} is staged as a file "
                f"(a login file), not only in the agent's environment; "
                f"{SCOPED_FLAG} names environment credentials only."
            )
        if name not in credentials:
            raise ScopedCredentialRefused(
                f"{task_name}: {SCOPED_FLAG} {name}: {name} is not a credential "
                f"variable (an API key or token the agent holds); {SCOPED_FLAG} "
                "names credentials only."
            )


def _mark_scoped(
    entries: list[dict[str, Any]], scoped: tuple[str, ...]
) -> list[dict[str, Any]]:
    """*entries* with each declared environment name split into an entry of
    its own, marked ``scoped``: kind and route unchanged. A file never is."""
    if not scoped:
        return entries
    out: list[dict[str, Any]] = []
    for entry in entries:
        mine = [
            n
            for n in entry["names"]
            if n in scoped
            and entry["kind"] != CLOUD_CREDENTIAL_FILE
            and n not in _LOGIN_FILES
        ]
        rest = [n for n in entry["names"] if n not in mine]
        if rest or not entry["names"]:
            out.append({**entry, "names": rest})
        if mine:
            out.append({**entry, "names": mine, "scoped": True})
    return out


def unscoped(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The entries the operator did not scope: the ones quiet main is for."""
    return [e for e in entries if not e.get("scoped")]


def agent_home(sandbox_user: str | None) -> str:
    """The home a connection stages its logins in (Rollout's credential
    home)."""
    return f"/home/{sandbox_user}" if sandbox_user else "/root"


def _entry(kind: str, route: str, names, staged: str) -> dict[str, Any]:
    return {"kind": kind, "route": route, "names": sorted(names), "staged": staged}


def _cloud_file_present(model: str | None, agent_env: dict[str, str]) -> bool:
    """Whether the run has a gcloud credential file to give: passed in, or a
    Vertex model's (resolution reads the host's ADC for it, and refuses the
    run without one)."""
    from benchflow.agents.env import VERTEX_ADC_ENV
    from benchflow.agents.registry import is_vertex_model

    return bool(agent_env.get(VERTEX_ADC_ENV)) or bool(model and is_vertex_model(model))


def in_sandbox(
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
    *,
    proxy_in_sandbox: bool,
    home: str = "{home}",
    scoped: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """The operator credentials this agent's connection stages in its sandbox,
    from the run's credential table (``launch_env.credential_policy``) and
    where the proxy runs (*proxy_in_sandbox*, from
    ``litellm_runtime.proxy_runs_on_host``). Empty for the proxy on the host
    and for a scripted agent. Names only, never a value; a file is named in
    *home*, the connection's (:func:`agent_home`). A name in *scoped* (the
    run's ``--scoped-credential``) is an entry of its own marked
    ``scoped``."""
    return _mark_scoped(
        _in_sandbox(
            agent, model, agent_env, proxy_in_sandbox=proxy_in_sandbox, home=home
        ),
        scoped,
    )


def _in_sandbox(
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
    *,
    proxy_in_sandbox: bool,
    home: str,
) -> list[dict[str, Any]]:
    from benchflow.agents.env import VERTEX_ADC_ENV, uses_native_vertex_login
    from benchflow.agents.launch_env import (
        NONE,
        PROXY,
        credential_policy,
        host_credential_env_names,
    )
    from benchflow.agents.registry import AGENTS

    # A continuation's key for BenchFlow's replay proxy is not the operator's,
    # whatever the table (a resolved env's recorded one included) names.
    replay = (
        {REPLAY_KEY_ENV} if agent_env.get(REPLAY_KEY_ENV) == REPLAY_API_KEY else set()
    )
    policy = credential_policy(agent, model, agent_env)
    if policy.route == NONE:
        return []
    found: list[dict[str, Any]] = []
    if policy.route == PROXY:
        if not proxy_in_sandbox:
            return []
        if IN_SANDBOX_PROXY_COUNTS:
            # The proxy's process gets the agent's environment as resolved,
            # before its own key goes in (litellm_runtime._proxy_process_env);
            # that key is BenchFlow's, not the operator's.
            keys = [
                n
                for n in host_credential_env_names()
                if n not in (VERTEX_ADC_ENV, _MASTER_KEY_ENV) and agent_env.get(n)
            ]
            if keys:
                found.append(_entry(API_KEY, PROXY, keys, SANDBOX_PROXY_PROCESS))
        if _cloud_file_present(model, agent_env):
            from benchflow.agents.registry import is_vertex_model

            staged = (
                SANDBOX_PROXY_FOLDER
                if model and is_vertex_model(model)
                else SANDBOX_PROXY_PROCESS
            )
            found.append(_entry(CLOUD_CREDENTIAL_FILE, PROXY, [VERTEX_ADC_ENV], staged))
        return found
    route = policy.route
    held = sorted(policy.launch_names(agent_env) - replay)
    logins = [n for n in held if n in _SUBSCRIPTION_NAMES]
    keys = [n for n in held if n not in _SUBSCRIPTION_NAMES and n != VERTEX_ADC_ENV]
    for name in logins:
        staged = AGENT_ENVIRONMENT
        writer, path = _LOGIN_FILES.get(name, ("", ""))
        if path and writer in (None, agent.removeprefix("acpx:")):
            staged = f"{AGENT_ENVIRONMENT} and {path.format(home=home)}"
        found.append(_entry(SUBSCRIPTION_LOGIN, route, [name], staged))
    # A key the agent's launcher also writes to a file names that file.
    launched = [
        n
        for n in keys
        if _LAUNCHER_KEY_FILES.get(n, ("",))[0] == agent.removeprefix("acpx:")
    ]
    for name in launched:
        path = _LAUNCHER_KEY_FILES[name][1].format(home=home)
        found.append(_entry(API_KEY, route, [name], f"{AGENT_ENVIRONMENT} and {path}"))
    keys = [n for n in keys if n not in launched]
    if keys:
        found.append(_entry(API_KEY, route, keys, AGENT_ENVIRONMENT))
    if uses_native_vertex_login(agent, model) or VERTEX_ADC_ENV in {
        *policy.files,
        *policy.file_only,
    }:
        found.append(
            _entry(
                CLOUD_CREDENTIAL_FILE,
                route,
                [VERTEX_ADC_ENV],
                NATIVE_ADC_PATH.format(home=home),
            )
        )
    elif VERTEX_ADC_ENV in held:
        # A raw command's own slots: the file's contents in its environment.
        found.append(
            _entry(CLOUD_CREDENTIAL_FILE, route, [VERTEX_ADC_ENV], AGENT_ENVIRONMENT)
        )
    if policy.host_login:
        cfg = AGENTS.get(agent)
        login = getattr(cfg, "subscription_auth", None)
        for f in getattr(login, "files", None) or ():
            found.append(
                _entry(
                    SUBSCRIPTION_LOGIN, route, [], f.container_path.format(home=home)
                )
            )
    return found


def cloud_credential_file(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The cloud credential file among *entries*, or None."""
    return next((e for e in entries if e["kind"] == CLOUD_CREDENTIAL_FILE), None)


def refuse_cloud_credential_file(
    entries: list[dict[str, Any]], *, task_name: str, allowed: bool
) -> None:
    """Refuse a cloud credential file the run did not allow, naming the flag
    and the proxy on the host (Docker, ``public``)."""
    found = cloud_credential_file(entries)
    if found is None or allowed:
        return
    raise CloudCredentialFileRefused(
        f"{task_name}: a cloud credential file (the gcloud application default "
        f"credentials, {', '.join(found['names'])}) would be staged in the "
        f"sandbox ({found['staged']}). A cloud credential file needs an "
        f"operator flag: pass {CLOUD_FILE_FLAG} (run config "
        f"{CLOUD_FILE_SETTING}: true), or run the model through {HOST_PROXY}, "
        "which keeps the file out of the sandbox."
    )


def proxy_in_sandbox(environment: str, *, force_sandbox_local: bool) -> bool:
    """Whether the run's proxy runs in the sandbox (the one placement rule,
    ``litellm_runtime.proxy_runs_on_host``)."""
    from benchflow.providers.litellm_runtime import proxy_runs_on_host

    return not proxy_runs_on_host(environment, force_sandbox_local=force_sandbox_local)


def granted(
    entries: list[dict[str, Any]],
    *,
    allow_cloud_credential_file: bool,
    allow_task_processes_with_credentials: bool = False,
) -> list[dict[str, Any]]:
    """What the operator's flags let in, in the shape of the isolation
    records (``setting``, ``run``, ``lifts``): the trial's, the run card's
    and the provenance's ``granted``."""
    found = []
    if allow_cloud_credential_file and cloud_credential_file(entries):
        found.append(
            {
                "setting": CLOUD_FILE_SETTING,
                "run": CLOUD_FILE_FLAG,
                "lifts": CLOUD_FILE_LIFTS,
            }
        )
    if allow_task_processes_with_credentials and unscoped(entries):
        found.append(
            {
                "setting": TASK_PROCESSES_SETTING,
                "run": TASK_PROCESSES_FLAG,
                "lifts": TASK_PROCESSES_LIFTS,
            }
        )
    return found


def _said(entries: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{e['kind']} ({', '.join(e['names'])}{', scoped' if e.get('scoped') else ''}) in {e['staged']}"
        if e["names"]
        else f"{e['kind']} in {e['staged']}"
        for e in entries
    )


def _scoped_line(entries: list[dict[str, Any]]) -> str:
    names = sorted({n for e in entries if e.get("scoped") for n in e["names"]})
    if not names:
        return ""
    return (
        f" The operator scoped {', '.join(names)} to this run ({SCOPED_FLAG}): "
        "a task process in main is not refused for it."
    )


def warning(
    task_name: str,
    entries: list[dict[str, Any]],
    allowed: list[dict[str, Any]],
    *,
    quiet_main: str | None = None,
) -> str | None:
    """The launch's warning for *entries*, or None when there are none."""
    if not entries:
        return None
    line = (
        f"{task_name}: operator credentials go into the sandbox, where a "
        f"process there can read them: {_said(entries)}."
    )
    for record in allowed:
        line += (
            f" {record['lifts'].capitalize()}: the run allows it with {record['run']}."
        )
    if quiet_main:
        line += f" Main's processes are {quiet_main}."
    return f"{line}{_scoped_line(entries)} {ADVICE}"


def record(
    earlier: dict[str, Any] | None,
    entries: list[dict[str, Any]],
    allowed: list[dict[str, Any]],
    *,
    refused: bool = False,
    quiet_main: str | None = None,
) -> dict[str, Any] | None:
    """result.json's ``operator_credentials``: every connection's entries and
    grants, each once; None while there are none. A connection *refused*
    before it staged anything lists its entries under ``would_stage``;
    *quiet_main* says why main's processes were not checked."""
    found = dict(earlier or {"in_sandbox": [], "granted": []})
    where = "would_stage" if refused else "in_sandbox"
    for key, new in ((where, entries), ("granted", allowed)):
        found[key] = [
            *found.get(key, []),
            *(e for e in new if e not in found.get(key, [])),
        ]
    if quiet_main and entries:
        found["quiet_main"] = quiet_main
    names = sorted({n for e in entries if e.get("scoped") for n in e["names"]})
    if names:
        found["scoped"] = sorted({*found.get("scoped", []), *names})
    return (
        found
        if found["in_sandbox"] or found["granted"] or found.get("would_stage")
        else None
    )


def task_entries(
    task_config: Any,
    *,
    agent: str,
    model: str | None,
    agent_env: dict[str, str],
    environment: str,
    sandbox_user: str | None,
    scoped: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """:func:`in_sandbox` for one task of a run, before any trial: its proxy
    is in the sandbox on a sandbox-proxy backend, or under its own restricted
    network (no internet, an egress policy)."""
    from types import SimpleNamespace
    from typing import cast

    from benchflow.agents.registry import is_scripted_agent
    from benchflow.task.plan_validation import agent_egress_policy

    if is_scripted_agent(agent) or agent == "task-runtime":
        return []
    no_web = getattr(task_config.sandbox, "allow_internet", True) is False
    policy = agent_egress_policy(
        cast(Any, SimpleNamespace(config=task_config)),
        agent=agent,
        sandbox_user=sandbox_user,
        disallow_web_tools=no_web,
    )
    return in_sandbox(
        agent,
        model,
        agent_env,
        proxy_in_sandbox=proxy_in_sandbox(
            environment, force_sandbox_local=no_web or policy is not None
        ),
        home=agent_home(sandbox_user),
        scoped=scoped,
    )


def run_card_block(
    task_dirs: list[Any],
    *,
    agent: str | None,
    model: str | None,
    agent_env: dict[str, str],
    environment: str,
    sandbox_user: str | None,
    allow_cloud_credential_file: bool,
    allow_task_processes_with_credentials: bool = False,
    scoped_credentials: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """The run card's ``operator_credentials``: per task, what its trials
    stage in the sandbox, what the run's flags let in and the names the
    operator scoped; None when no task stages anything."""
    from benchflow.task.runtime_view import TaskRuntimeView

    if not agent:
        return None
    tasks: dict[str, list[dict[str, Any]]] = {}
    allowed: list[dict[str, Any]] = []
    for task_dir in task_dirs:
        try:
            config = TaskRuntimeView.from_task_dir(task_dir).config
            entries = task_entries(
                config,
                agent=agent,
                model=model,
                agent_env=agent_env,
                environment=environment,
                sandbox_user=sandbox_user,
                scoped=scoped_credentials,
            )
        except Exception:  # the trial's own gate reports a task that won't load
            continue
        if entries:
            tasks[task_dir.name] = entries
            for g in granted(
                entries,
                allow_cloud_credential_file=allow_cloud_credential_file,
                allow_task_processes_with_credentials=allow_task_processes_with_credentials,
            ):
                if g not in allowed:
                    allowed.append(g)
    if not tasks:
        return None
    block: dict[str, Any] = {"tasks": tasks, "granted": allowed}
    names = sorted(
        {n for es in tasks.values() for e in es if e.get("scoped") for n in e["names"]}
    )
    if names:
        block["scoped"] = names
    return block


def run_start_lines(block: dict[str, Any] | None) -> list[str]:
    """The warnings a run gives at start for its card's block."""
    if not isinstance(block, dict) or not block.get("tasks"):
        return []
    tasks = block["tasks"]
    shown = "; ".join(
        f"{name}: {_said(entries)}" for name, entries in list(tasks.items())[:8]
    )
    if len(tasks) > 8:
        shown += f"; and {len(tasks) - 8} more"
    line = (
        f"Operator credentials go into the sandbox of {len(tasks)} "
        f"task{'s' if len(tasks) > 1 else ''}, where a process there can read "
        f"them: {shown}."
    )
    for g in block.get("granted") or ():
        line += f" {g['lifts'].capitalize()}: the run allows it with {g['run']}."
    if block.get("scoped"):
        line += (
            f" The operator scoped {', '.join(block['scoped'])} to this run "
            f"({SCOPED_FLAG}): a task process in main is not refused for it."
        )
    return [f"{line} {ADVICE}"]
