"""The cloud metadata endpoints under an open network: the list, the
decision and the record.

Every cloud serves instance credentials on a link-local address that any
process in a VM can reach, and Docker's bridge NATs a container's traffic
out through the host, so an agent in a sandbox on a cloud VM reaches the VM's
metadata server unless something refuses it. Under ``denylist``,
``allowlist`` and a kept-open no-network the uid firewall already confines
the agent to loopback; under the default open network nothing did. Root now
refuses TCP from every non-root user to each endpoint on every port but 53
(GCE's 169.254.169.254 and Azure's 168.63.129.16 also answer DNS): the same
list and rule as the Rust ``bench`` 0.8.2. The rules themselves
are in :mod:`benchflow.sandbox.lockdown` (``block_cloud_metadata``); the
rollout reads the decision here and passes it to the sandbox factory, which
gives ``main`` NET_ADMIN only while the block is armed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow.errors import UserError

#: The link-local address most clouds serve instance credentials on (AWS, GCE,
#: Azure, Oracle, DigitalOcean, OpenStack), AWS's container credentials (ECS)
#: and EKS Pod Identity, Alibaba Cloud's metadata server, Azure's WireServer,
#: and Tencent Cloud's metadata server.
CLOUD_METADATA_V4 = (
    "169.254.169.254",
    "169.254.170.2",
    "169.254.170.23",
    "100.100.100.200",
    "168.63.129.16",
    "169.254.0.23",
)
#: AWS's IPv6 IMDS and EKS Pod Identity, GCE's, and Oracle Cloud's.
CLOUD_METADATA_V6 = (
    "fd00:ec2::254",
    "fd00:ec2::23",
    "fd20:ce::254",
    "fd00:c1::a9fe:a9fe",
)
#: The trial folder's record of the block (state ``on``, ``off``,
#: ``not_reachable``, ``provider-managed``, ``not_installed``, ``unproven``
#: or ``refused``, and why).
CLOUD_METADATA_RECORD = "cloud-metadata.json"
#: The record's state when the block's own audit refused sandbox startup
#: (:class:`CloudMetadataPowerError`): the agent never starts.
REFUSED = "refused"


class CloudMetadataPowerError(RuntimeError):
    """The agent could bypass the cloud metadata firewall or launch barrier.

    The message starts with "sandbox startup refused", so the trial is a
    sandbox-setup failure, never run with an unverified privilege boundary.
    This policy error is shared by admission and the concrete firewall audit.

    ``record``: the trial's ``cloud-metadata.json`` for the refusal (state
    ``refused``, why, and what the audit found), when the block got far
    enough to make one. The rollout writes it before the error goes on, so
    the trial folder and result.json name what refused the trial.
    """

    def __init__(self, *args: object, record: dict[str, Any] | None = None) -> None:
        super().__init__(*args)
        self.record = record


class CloudMetadataUnprotectedError(CloudMetadataPowerError):
    """The block is not proven on, and the run did not opt out: the agent
    does not start. The message says why and how to opt out."""


#: How the record names the run's own opt-out (``bench eval run``,
#: ``EvaluationConfig.allow_cloud_metadata``).
CLOUD_METADATA_RUN_OPT_OUT = "--allow-cloud-metadata"
#: Where the refusal points a run that means to let its agent reach them.
CLOUD_METADATA_OPT_OUT_HINT = (
    "run with --allow-cloud-metadata to let the agent reach them (a task's "
    "allow_cloud_metadata = true takes effect only with that flag)"
)
#: What a task's own opt-out lifts once the run grants it.
TASK_OPT_OUT_LIFTS = (
    "the cloud metadata block for every process of the trial: the agent's "
    "rule, main's rule for every user, the verifier's and the task's other "
    "compose services'"
)


class CloudMetadataOptInRefused(ValueError, UserError):
    """A task asks to lift the cloud metadata block (its own
    ``allow_cloud_metadata = true``, or ``metadata.x-allow-cloud-metadata =
    true``) and the run did not grant it with ``--allow-cloud-metadata``. A
    task can't weaken its own isolation: the trial, or the whole job, is
    refused before anything starts. The message names the tasks, their key
    and the flag."""

    fault = "task"


def task_opt_in(task_config: Any) -> str:
    """How the task asks to lift the block, as its record names it ("" when
    it does not ask): the declaration alone, whatever the run grants."""
    opt_out = getattr(task_config, "cloud_metadata_opt_out", "")
    return opt_out if isinstance(opt_out, str) else ""


def task_opt_ins(task_config: Any) -> list[tuple[str, str]]:
    """Every way the task asks to lift the block: (setting, how). Today its
    own opt-out only (an allowlist entry that names a metadata address is no
    way around it: main's rule for every user refuses it)."""
    how = task_opt_in(task_config)
    return [("allow_cloud_metadata", how)] if how else []


def opt_in_refusal(
    asked: list[tuple[str, str]], *, command: str | None = None
) -> CloudMetadataOptInRefused:
    """The refusal for ``asked``, (task name, how it asks) pairs. *command*
    (``bench eval continue``, ``bench eval branch``): a command that takes no
    grant, so the refusal says it can't lift the block and names the one
    that can."""
    shown = "; ".join(f"{name} ({how})" for name, how in asked[:8])
    if len(asked) > 8:
        shown += f"; and {len(asked) - 8} more"
    many = len(asked) > 1
    them = "them" if many else "it"
    head = (
        f"{len(asked)} task{'s' if many else ''} ask{'' if many else 's'} for "
        "the cloud metadata endpoints (a cloud VM's instance credentials), which "
        f"a task can't open by itself: {shown}. A task's allow_cloud_metadata "
        f"takes effect only when the run also passes {CLOUD_METADATA_RUN_OPT_OUT}"
    )
    if command:
        error = CloudMetadataOptInRefused(
            f"{head}, and {command} takes no such flag, so it can't lift the "
            f"block: run the task with bench eval run {CLOUD_METADATA_RUN_OPT_OUT} "
            "instead. Nothing was started."
        )
        error.hint = (
            f"{command} can't lift the cloud metadata block; run the task with "
            f"bench eval run {CLOUD_METADATA_RUN_OPT_OUT} (or bench run "
            f"{CLOUD_METADATA_RUN_OPT_OUT}) to let {them} reach the endpoints, or "
            f"leave {'these tasks' if many else 'the task'} out"
        )
        return error
    error = CloudMetadataOptInRefused(f"{head}. Nothing was started.")
    error.hint = (
        f"run with {CLOUD_METADATA_RUN_OPT_OUT} (the run config's "
        "allow_cloud_metadata: true; allow_cloud_metadata=True in "
        "EvaluationConfig, RolloutConfig or RuntimeConfig) to let "
        f"{them} reach the endpoints, or leave "
        f"{'these tasks' if many else 'the task'} out; `bench tasks check` at a "
        "level that starts a sandbox (--level acceptance-live, for one) takes no "
        "such flag, so it can't run such a task"
    )
    return error


def refuse_opt_in_for_command(
    task_dirs: list[Path],
    *,
    command: str,
    config_override: dict[str, Any] | None = None,
) -> None:
    """Refuse, before anything starts, the tasks in *task_dirs* that ask to
    lift the block when *command* runs them: a command with no grant to give
    (``bench eval continue``, ``bench eval branch``). A task that does not
    load is left to the run (:func:`task_dirs_opt_ins`)."""
    asked = task_dirs_opt_ins(task_dirs, config_override)
    if asked:
        raise opt_in_refusal(asked, command=command)


def refuse_ungranted_opt_in(
    task_config: Any, *, task_name: str, allow_cloud_metadata: bool
) -> None:
    """Refuse, before anything starts, a task that asks to lift the block
    on a run that does not grant it (:class:`CloudMetadataOptInRefused`)."""
    if strict_grant(allow_cloud_metadata):
        return
    asked = task_opt_ins(task_config)
    if asked:
        raise opt_in_refusal([(task_name, "; ".join(how for _, how in asked))])


def opt_in_records(
    task_config: Any, *, allow_cloud_metadata: bool
) -> list[dict[str, Any]]:
    """The trial's record of each way the task asked for the endpoints and
    the run granted (result.json's ``isolation_opt_ins``, the provenance's
    ``isolation_lifted``): the setting, how the task asked, the flag and what
    it lifts. Empty when the task does not ask, or the run did not grant it
    (that trial is refused)."""
    if not strict_grant(allow_cloud_metadata):
        return []
    return [
        {
            "setting": setting,
            "task": how,
            "run": CLOUD_METADATA_RUN_OPT_OUT,
            "lifts": TASK_OPT_OUT_LIFTS,
        }
        for setting, how in task_opt_ins(task_config)
    ]


#: The record's state for a root agent on a sandbox provider (Daytona, Modal,
#: AgentCore, Apple): the host and its metadata service are the provider's,
#: whose own metadata policy applies.
PROVIDER_MANAGED = "provider-managed"
#: The record's state on a sandbox whose provider gives it no route to any
#: cloud metadata endpoint (``cloud_metadata_unreachable``, Modal): nothing is
#: installed, and a connect to every endpoint as the agent's user proved it
#: in this trial. The agent starts, as on ``on``.
NOT_REACHABLE = "not_reachable"


def cloud_metadata_refusal(state: str, why: str) -> CloudMetadataUnprotectedError:
    """The error that stops a trial whose block is not ``on``."""
    return CloudMetadataUnprotectedError(
        "sandbox startup refused: the agent would start with the cloud "
        f"metadata endpoints open to it (the block is {state}: {why}). "
        f"The trial stops before the agent starts; {CLOUD_METADATA_OPT_OUT_HINT}."
    )


@dataclass(frozen=True)
class CloudMetadataPlan:
    """What a run does about the cloud metadata endpoints.

    ``armed``: root installs the block before the agent starts (and the
    compose backends give ``main`` NET_ADMIN for it, unless BenchFlow's
    helper adds the rules from outside ``main``). Otherwise ``why`` says why
    not, for the record, whose state is ``state``: ``off`` when the run opts
    out (``opt_out`` names how), ``not_installed`` when the block cannot go
    in safely (``main`` shares the host's network), which stops the trial.
    ``root_agent``: the agent runs as root, so the rules refuse root too and
    ``main`` must hold neither NET_ADMIN nor NET_RAW.
    ``uncovered``: the task's other compose services, which get no rule.
    """

    armed: bool
    why: str = ""
    state: str = "off"
    opt_out: str = ""
    # The task's require_cloud_metadata_block = true: a root agent on a
    # provider without BenchFlow's helper is refused, not provider-managed.
    require_block: bool = False
    root_agent: bool = False
    uncovered: tuple[str, ...] = ()
    # Who lifted the block: ``{"run": "--allow-cloud-metadata"}``, with
    # ``"task"`` (how it asked) when the task's own opt-out was granted.
    lifted_by: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ComposeTopology:
    """What the task's own compose file says about the block's reach.

    ``main_network_mode``: ``main``'s ``network_mode`` (``None`` without
    one). ``other_services``: every service but ``main`` that has a network,
    sorted; one with ``network_mode: none`` (and no ``networks`` or
    ``extends`` that could give it one) can't reach the endpoints.
    ``unresolved``: why ``main``'s network cannot be read from the file
    (it does not parse, it uses ``include``, ``main`` uses ``extends``, or it
    joins a network named ``host`` or by a variable); empty when
    it can.
    """

    main_network_mode: str | None = None
    other_services: tuple[str, ...] = ()
    unresolved: str = ""
    #: For ``network_mode: service:<x>`` (followed through further
    #: ``service:`` links): the service whose own namespace main joins.
    main_netns_owner: str = ""
    #: Where that chain ends: "" (a namespace of the project's own), "host"
    #: (the host's network), or "outside" (another container's, a variable,
    #: an unknown service, a loop).
    main_chain_end: str = ""

    @property
    def main_shares_outside_network(self) -> bool:
        """Whether ``main`` joins a network namespace the sandbox does not
        own: the host's (``host``), another container's (``container:``),
        another service's (``service:``, which may be the host's in turn),
        or one the file only names through a variable. Rules root adds there
        land in that namespace (the host's own firewall for ``host``) and
        outlast the trial. A file whose ``main`` network cannot be read
        counts as outside: the check fails closed."""
        if self.unresolved:
            return True
        mode = self.main_network_mode
        if mode is None:
            return False
        if mode.startswith("service:") and "$" not in mode:
            # Another service of the project: its namespace is the trial's own
            # unless the chain ends in the host's or leaves the project.
            return self.main_chain_end != ""
        return (
            mode == "host" or mode.startswith(("container:", "service:")) or "$" in mode
        )

    def outside_why(self) -> str:
        """The record's reason when ``main`` shares an outside network."""
        if self.unresolved:
            return MAIN_NETWORK_UNRESOLVED.format(reason=self.unresolved)
        if self.main_chain_end == "host":
            return MAIN_SHARES_HOST_NETWORK.format(
                mode=f"{self.main_network_mode}, whose network is the host's"
            )
        return MAIN_SHARES_HOST_NETWORK.format(mode=self.main_network_mode)


def _service_chain(
    services: dict[str, Any], mode: str, networks: Any
) -> tuple[str, str]:
    """Follow ``service:`` links from *mode*: (the service whose own namespace
    it ends in, where the chain ends: "" the project's own, "host", or
    "outside")."""
    owner, seen = "", set()
    while isinstance(mode, str) and mode.strip().startswith("service:"):
        name = mode.strip().split(":", 1)[1].strip()
        service = services.get(name)
        if name in seen or "$" in name or not isinstance(service, dict):
            return name, "outside"
        if "extends" in service:
            return name, "outside"
        seen.add(name)
        owner, mode = name, service.get("network_mode")
        if mode is None:
            return owner, "outside" if _main_host_network(service, networks) else ""
    mode = str(mode).strip()
    if mode == "host":
        return owner, "host"
    if mode.startswith("container:") or "$" in mode:
        return owner, "outside"
    return owner, ""


def _compose_external(value: Any) -> bool | None:
    """Normalize Compose's external flag; None means it is unresolved.

    Compose casts strings before resolving names (case-insensitively, with
    YAML's legacy yes/no/on/off/y/n forms). Every legacy mapping, even {},
    means external: true. Never use Python truthiness for either form.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return True
    if isinstance(value, str):
        if value.lower() in {"true", "y", "yes", "on"}:
            return True
        if value.lower() in {"false", "n", "no", "off"}:
            return False
    return None


def _main_host_network(main: dict[str, Any], networks: Any) -> str:
    """Why a network main joins is named ``host`` or still holds a variable.

    Compose attaches main to ``default`` when its networks are absent or
    empty, unless it has a network_mode instead. An explicit name wins
    over legacy external.name. Only external networks use their bare key
    as a name; Compose prefixes private keys with the project name.
    """
    joined = main.get("networks")
    if not joined and main.get("network_mode") is None:
        joined = ["default"]
    if not isinstance(joined, (dict, list)):
        return ""
    names = [str(n) for n in joined]
    declared = networks if isinstance(networks, dict) else {}
    for key in names:
        spec = declared.get(key)
        spec = spec if isinstance(spec, dict) else {}
        raw_external = spec.get("external")
        external = _compose_external(raw_external)
        if external is None:
            return f"main joins network {key!r} with an unresolved external flag"
        legacy_name = (
            raw_external.get("name") if isinstance(raw_external, dict) else None
        )
        name = str(spec.get("name") or legacy_name or (key if external else "")).strip()
        if name == "host" or "$" in name:
            kind = "external network" if external else "network"
            return f"main joins the {kind} {key!r} (name: {name})"
    return ""


def _compose_loader() -> Any:
    """A YAML safe loader that also reads compose's own tags: ``!override``
    (the value replaces the earlier file's, so it is the value) and
    ``!reset`` (the attribute goes back to its default, so it is absent)."""
    import yaml

    class ComposeLoader(yaml.SafeLoader):
        pass

    # Compose's untyped YAML uses only true/false as booleans. Legacy words
    # such as off/no are names; coerce them only in _compose_external.
    # Copy both levels so other SafeLoader consumers keep their resolver.
    ComposeLoader.yaml_implicit_resolvers = {
        key: [
            (tag, pattern) for tag, pattern in rules if tag != "tag:yaml.org,2002:bool"
        ]
        for key, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    ComposeLoader.add_implicit_resolver(
        "tag:yaml.org,2002:bool",
        re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
        list("tTfF"),
    )

    def override(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        assert isinstance(node, yaml.ScalarNode)
        return loader.construct_scalar(node)

    ComposeLoader.add_constructor("!override", override)
    ComposeLoader.add_constructor("!reset", lambda loader, node: None)
    return ComposeLoader


def _offline(service: Any) -> bool:
    """A compose service with no network: ``network_mode: none``, with no
    ``networks`` and no ``extends`` that could give it one."""
    return (
        isinstance(service, dict)
        and str(service.get("network_mode", "")).strip() == "none"
        and "extends" not in service
        and not service.get("networks")
    )


def resolved_topology(project: Any) -> ComposeTopology:
    """:class:`ComposeTopology` of the project as Compose resolved it for the
    run (``docker compose ... config --format json``): extends, include,
    interpolation and profiles are what Compose made of them, so nothing is
    left unresolved."""
    services = project.get("services") if isinstance(project, dict) else None
    services = services if isinstance(services, dict) else {}
    networks = project.get("networks") if isinstance(project, dict) else None
    others = tuple(
        sorted(
            str(n) for n, svc in services.items() if n != "main" and not _offline(svc)
        )
    )
    main = services.get("main")
    if not isinstance(main, dict):
        return ComposeTopology(other_services=others)
    mode = main.get("network_mode")
    owner, end = "", ""
    if isinstance(mode, str) and mode.strip().startswith("service:"):
        owner, end = _service_chain(services, mode, networks)
    return ComposeTopology(
        main_network_mode=None if mode is None else str(mode).strip(),
        other_services=others,
        unresolved="" if mode is not None else _main_host_network(main, networks),
        main_netns_owner=owner,
        main_chain_end=end,
    )


def task_compose_topology(environment_dir: Path | None) -> ComposeTopology:
    """Read the task's ``docker-compose.yaml`` (the file the Docker and
    Daytona compose backends stack on BenchFlow's own) in *environment_dir*.

    Compose's ``!reset`` and ``!override`` tags are read as compose reads
    them. An empty topology when there is no file. A file that cannot be
    read or does not parse (any other tag), a file with an ``include``
    list, a ``main`` that ``extends`` another service (whose
    ``network_mode`` it may take), and a ``main`` on a network named
    ``host`` or with an unresolved variable give an ``unresolved`` topology,
    which counts as sharing an outside network.
    """
    if environment_dir is None:
        return ComposeTopology()
    path = environment_dir / "docker-compose.yaml"
    if not path.is_file():
        return ComposeTopology()
    import yaml

    try:
        # A SafeLoader with two more tags: it builds no Python objects.
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_compose_loader())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        first = (str(e).strip().splitlines() or [type(e).__name__])[0][:200]
        return ComposeTopology(
            unresolved=f"the task's docker-compose.yaml does not parse ({first})"
        )
    if isinstance(loaded, dict) and "include" in loaded:
        return ComposeTopology(
            unresolved="the task's docker-compose.yaml includes other compose "
            "files (include), which may define main"
        )
    if not isinstance(loaded, dict):
        return ComposeTopology()
    services = loaded.get("services")
    if not isinstance(services, dict):
        services = {}
    others = tuple(
        sorted(
            str(n) for n, svc in services.items() if n != "main" and not _offline(svc)
        )
    )
    main = services.get("main")
    if main is None:
        # The backend's base file supplies main even when the task override
        # only declares networks or sidecars.
        main = {}
    if not isinstance(main, dict):
        return ComposeTopology(other_services=others)
    mode = main.get("network_mode")
    if "extends" in main:
        unresolved = (
            "main uses extends, so its network_mode may come from another service"
        )
    else:
        unresolved = _main_host_network(main, loaded.get("networks"))
    owner, end = "", ""
    if isinstance(mode, str) and mode.strip().startswith("service:"):
        owner, end = _service_chain(services, mode, loaded.get("networks"))
    return ComposeTopology(
        main_network_mode=None if mode is None else str(mode).strip(),
        other_services=others,
        unresolved=unresolved,
        main_netns_owner=owner,
        main_chain_end=end,
    )


#: Why the block is not installed when ``main`` shares an outside network.
MAIN_SHARES_HOST_NETWORK = (
    "main shares the host's network or another container's (network_mode: "
    "{mode} in the task's docker-compose.yaml), where a rule would change "
    "that firewall (the host's own, for host) and outlast the trial, so none "
    "is installed; block the endpoints at the host"
)
#: Why the block is not installed when ``main``'s network cannot be read.
MAIN_NETWORK_UNRESOLVED = (
    "main may share the host's network ({reason}), where a rule would change "
    "the host's own firewall and outlast the trial, so none is installed; "
    "block the endpoints at the host"
)


def network_is_open(sandbox_config: Any) -> bool:
    """An open network: ``public`` with the internet allowed."""
    from benchflow.task.config import NetworkMode

    mode = getattr(sandbox_config, "network_mode", NetworkMode.PUBLIC)
    return mode == NetworkMode.PUBLIC and (
        getattr(sandbox_config, "allow_internet", True) is not False
    )


def cloud_metadata_plan(
    task_config: Any,
    *,
    sandbox_user: str | None,
    launches_agent: bool,
    environment_dir: Path | None = None,
    allow_cloud_metadata: bool = False,
) -> CloudMetadataPlan | None:
    """Whether the run refuses the cloud metadata endpoints to its agent.

    ``None`` when the question does not arise: no agent runs (oracle, nop),
    or the agent's network is not open (``no-network``, ``denylist`` and
    ``allowlist`` already confine the agent user to loopback and the lists'
    own addresses). Under an open network the block is armed, for an agent
    that runs as root too (``root_agent``), unless the run opts out with
    ``allow_cloud_metadata`` (``--allow-cloud-metadata``), with or without
    the task's own ``allow_cloud_metadata = true`` (or its older
    ``metadata.x-allow-cloud-metadata = true``). The task's key alone never
    disarms it: such a trial is refused before it starts
    (:func:`refuse_ungranted_opt_in`), and the plan keeps the block if one
    gets here. ``lifted_by`` names the flag, and the task's key when it was
    granted. It is not installed when the
    task's compose file (in *environment_dir*) puts ``main`` on the host's
    network or another container's, and the trial then stops.
    The task's other compose services are named in ``uncovered``: the rules
    go in ``main`` only.
    """
    from benchflow.sandbox.egress_denylist import agent_network_sandbox_config

    if not launches_agent or task_config is None:
        return None
    if not network_is_open(agent_network_sandbox_config(task_config)):
        return None
    if strict_grant(allow_cloud_metadata):
        how = task_opt_in(task_config)
        lifted_by = {
            "run": CLOUD_METADATA_RUN_OPT_OUT,
            **({"task": how} if how else {}),
        }
        opt_out = (
            f"{how}, granted by {CLOUD_METADATA_RUN_OPT_OUT}"
            if how
            else CLOUD_METADATA_RUN_OPT_OUT
        )
        return CloudMetadataPlan(
            armed=False,
            why=f"the run opts out ({opt_out})",
            opt_out=opt_out,
            lifted_by=lifted_by,
        )
    root_agent = not sandbox_user or sandbox_user in {"root", "0"}
    topology = task_compose_topology(environment_dir)
    if topology.main_shares_outside_network:
        return CloudMetadataPlan(
            armed=False,
            state="not_installed",
            why=topology.outside_why(),
            root_agent=root_agent,
        )
    return CloudMetadataPlan(
        armed=True,
        uncovered=topology.other_services,
        root_agent=root_agent,
        require_block=bool(
            getattr(
                getattr(task_config, "sandbox", None),
                "require_cloud_metadata_block",
                False,
            )
        ),
    )


def uncovered_services_note(services: tuple[str, ...]) -> str:
    """The record's note when the task has compose services besides main."""
    names = ", ".join(services)
    return (
        f"; the rules cover main only: the task's other services ({names}) "
        "reach the endpoints, so an agent that runs code in one of them or "
        "steers it can too (block the endpoints at the host)"
    )


def cloud_metadata_list() -> str:
    """The ten endpoints, comma-separated, for messages."""
    return ", ".join(CLOUD_METADATA_V4 + CLOUD_METADATA_V6)


def cloud_metadata_record(state: str, why: str, **extra: Any) -> dict[str, Any]:
    """The trial's ``cloud-metadata.json``: the defense, its state, why, and
    the endpoints it covers."""
    return {
        "defense": "cloud_metadata",
        "state": state,
        "why": why,
        "addresses": list(CLOUD_METADATA_V4 + CLOUD_METADATA_V6),
        **extra,
    }


def task_dirs_opt_ins(
    task_dirs: list[Path], config_override: dict[str, Any] | None = None
) -> list[tuple[str, str]]:
    """(task name, how it asks) for each task in *task_dirs* that asks to
    lift the block, with the run's config overlay applied as each trial
    applies it. A task that does not load is left to the run, which reports
    it."""
    from benchflow.task.runtime_view import TaskRuntimeView

    asked = []
    for task_dir in task_dirs:
        try:
            config = TaskRuntimeView.from_task_dir(task_dir).config
            if config_override:
                from benchflow._utils.config_override import apply_config_override

                config = apply_config_override(config, config_override)
        except Exception:  # the run names a task that does not load
            continue
        hows = [how for _, how in task_opt_ins(config)]
        if hows:
            asked.append((Path(task_dir).name, "; ".join(hows)))
    return asked


def run_card_block(
    task_dirs: list[Path],
    *,
    allow_cloud_metadata: bool,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The run card's ``cloud_metadata``: the run's flag (None without it)
    and, under it, each task whose own opt-out it grants, with its key."""
    allow_cloud_metadata = strict_grant(allow_cloud_metadata)
    asked = (
        task_dirs_opt_ins(task_dirs, config_override) if allow_cloud_metadata else []
    )
    return {
        "run_flag": CLOUD_METADATA_RUN_OPT_OUT if allow_cloud_metadata else None,
        "lifted_for": [{"task": name, "key": how} for name, how in asked],
    }


def run_start_lines(block: dict[str, Any] | None) -> list[str]:
    """The warnings a run gives at start for its card's ``cloud_metadata``."""
    if not isinstance(block, dict) or not block.get("run_flag"):
        return []
    lifted = [row for row in block.get("lifted_for") or () if isinstance(row, dict)]
    if lifted:
        named = "; ".join(f"{row['task']} ({row['key']})" for row in lifted[:8])
        if len(lifted) > 8:
            named += f"; and {len(lifted) - 8} more"
        return [
            f"The cloud metadata block is lifted for {len(lifted)} "
            f"task{'s' if len(lifted) > 1 else ''}: {named}. Each asks for it "
            f"and this run grants it with {CLOUD_METADATA_RUN_OPT_OUT}, so its "
            "agent, its test.sh and its other services can reach the cloud "
            "metadata endpoints (a cloud VM's instance credentials); each "
            "trial's result.json names it (isolation_opt_ins)."
        ]
    return [
        f"{CLOUD_METADATA_RUN_OPT_OUT} is set: under an open network the "
        "agent's own cloud metadata rule and the task's other compose "
        "services' are not installed (on Docker main's rule for every user "
        "still holds); no task asks to lift the rest."
    ]


def strict_grant(value: Any, *, where: str = "allow_cloud_metadata") -> bool:
    """The run's grant read from a config file or record: true only for a
    boolean true or the words ``true``/``yes``/``on``/``1``, so a quoted
    ``"false"`` never becomes the permission (``bool("false")`` is true).
    Anything else is refused, naming *where*."""
    if value is None or isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "on", "1"}:
        return True
    if text in {"false", "no", "off", "0", ""}:
        return False
    raise ValueError(f"{where} must be true or false, not {value!r}")
