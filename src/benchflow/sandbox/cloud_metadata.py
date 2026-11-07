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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    "set allow_cloud_metadata = true in the task's [environment] table, or run "
    "with --allow-cloud-metadata, to let the agent reach them"
)


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
        return (
            mode == "host" or mode.startswith(("container:", "service:")) or "$" in mode
        )

    def outside_why(self) -> str:
        """The record's reason when ``main`` shares an outside network."""
        if self.unresolved:
            return MAIN_NETWORK_UNRESOLVED.format(reason=self.unresolved)
        return MAIN_SHARES_HOST_NETWORK.format(mode=self.main_network_mode)


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
    return ComposeTopology(
        main_network_mode=None if mode is None else str(mode).strip(),
        other_services=others,
        unresolved=unresolved,
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
    that runs as root too (``root_agent``), unless the run opts out: the
    task's ``allow_cloud_metadata = true`` (or its older
    ``metadata.x-allow-cloud-metadata = true``), or ``allow_cloud_metadata``
    from the run (``--allow-cloud-metadata``). It is not installed when the
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
    opt_out = (
        CLOUD_METADATA_RUN_OPT_OUT
        if allow_cloud_metadata
        else getattr(task_config, "cloud_metadata_opt_out", "")
        or (
            "the task's opt-out"
            if getattr(task_config, "allows_cloud_metadata", False)
            else ""
        )
    )
    if opt_out:
        return CloudMetadataPlan(
            armed=False, why=f"the run opts out ({opt_out})", opt_out=opt_out
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
