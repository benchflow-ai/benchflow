"""Authority domains: which resource classes an agent may observe or change.

Ported from BenchGuard (arXiv 2609.11028;
``src/benchflow/benchguard/authority.py``). The paper classifies resources by
authority domain rather than by path; I1 (observation) and I2 (authority) are
the forbidden crossings of an agent into a domain it does not own:

- ``AgentOwned``: state the agent may see and change (its workspace).
- ``DeclaredHandoff``: objects declared to cross into outcome computation.
- ``OutcomeOwned``: the outcome procedure's state (hidden tests, labels,
  scorer code, reset state).
- ``RewardOutput``: the reward channel.
- ``ReleasedEvidence``: logs and feedback released by policy.
- ``TrustedInput``: read-only inputs the agent may observe but not change.
"""

from __future__ import annotations

from typing import Literal

from benchflow.integrity.constants import ActorClass, ResourceClass

AuthorityDomain = Literal[
    "AgentOwned",
    "DeclaredHandoff",
    "OutcomeOwned",
    "RewardOutput",
    "ReleasedEvidence",
    "TrustedInput",
]

RESOURCE_CLASS_TO_AUTHORITY: dict[ResourceClass, AuthorityDomain] = {
    "Public": "TrustedInput",
    "AgentWritable": "AgentOwned",
    "DeclaredArtifact": "DeclaredHandoff",
    "UndeclaredArtifact": "AgentOwned",
    "Trusted": "TrustedInput",
    "VerifierOnly": "OutcomeOwned",
    "Hidden": "OutcomeOwned",
    "ResetState": "OutcomeOwned",
    "RewardOutput": "RewardOutput",
}

# The agent may read its own state, its declared handoffs, released evidence
# and trusted inputs; never outcome-owned state or the reward channel (I1).
AGENT_OBSERVABLE_DOMAINS: frozenset[AuthorityDomain] = frozenset(
    {"AgentOwned", "DeclaredHandoff", "ReleasedEvidence", "TrustedInput"}
)
# The agent may write only its own state and its declared handoffs (I2).
AGENT_WRITABLE_DOMAINS: frozenset[AuthorityDomain] = frozenset(
    {"AgentOwned", "DeclaredHandoff"}
)


def authority_of(resource_class: ResourceClass | None) -> AuthorityDomain | None:
    if resource_class is None:
        return None
    return RESOURCE_CLASS_TO_AUTHORITY.get(resource_class)


def agent_may_observe(resource_class: ResourceClass | None) -> bool:
    """True iff an agent may observe a resource of this class (I1)."""
    domain = authority_of(resource_class)
    if domain is None:
        # An unknown reward-relevant object fails closed (I5).
        return False
    return domain in AGENT_OBSERVABLE_DOMAINS


def agent_may_write(resource_class: ResourceClass | None) -> bool:
    """True iff an agent may write a resource of this class (I2)."""
    domain = authority_of(resource_class)
    if domain is None:
        return False
    return domain in AGENT_WRITABLE_DOMAINS


def crossing_is_forbidden(
    *,
    actor_class: ActorClass,
    resource_class: ResourceClass | None,
    write: bool,
) -> bool:
    """Whether an actor's action on a resource crosses into a forbidden domain.

    Only agent actors are restricted here; the trusted host and the verifier
    are covered by the reward-provenance and fail-closed checks.
    """

    if actor_class != "Agent":
        return False
    if write:
        return not agent_may_write(resource_class)
    return not agent_may_observe(resource_class)


__all__ = [
    "AGENT_OBSERVABLE_DOMAINS",
    "AGENT_WRITABLE_DOMAINS",
    "RESOURCE_CLASS_TO_AUTHORITY",
    "AuthorityDomain",
    "agent_may_observe",
    "agent_may_write",
    "authority_of",
    "crossing_is_forbidden",
]
