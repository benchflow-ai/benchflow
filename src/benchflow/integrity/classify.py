"""Map a sandbox path onto a resource class, contract first.

Ported from BenchGuard's action recorder (arXiv 2609.11028;
``src/benchflow/benchguard/action_recorder.py``: ``_classify_resource_class``
and ``_normalize_manifest_resources``). The built-in table is the set of roots
BenchFlow itself locks away from the agent (``sandbox/lockdown.py``) plus the
reward and trajectory folders under ``/logs``.
"""

from __future__ import annotations

from collections.abc import Iterable

from benchflow.integrity.constants import RESOURCE_CLASSES, ResourceClass

# Longest root first, so /logs/verifier wins over /logs.
BUILTIN_PROTECTED_ROOTS: tuple[tuple[str, ResourceClass], ...] = (
    ("/testbed_verify", "VerifierOnly"),
    ("/logs/verifier", "RewardOutput"),
    # BenchFlow publishes the trajectory here for the verifier: evidence the
    # agent may read but not forge.
    ("/logs/agent", "Trusted"),
    ("/verifier", "VerifierOnly"),
    ("/solution", "Hidden"),
    ("/oracle", "Hidden"),
    ("/tests", "VerifierOnly"),
)
BUILTIN_AGENT_WRITABLE_ROOTS: tuple[str, ...] = ("/app", "/workspace")
REWARD_OUTPUT_NAMES = frozenset({"reward.txt", "reward.json", "judge_result.json"})

type ResourceTable = tuple[tuple[str, ResourceClass], ...]


def normalize_manifest_resources(
    manifest_resources: Iterable[tuple[str, str]] | None,
) -> ResourceTable:
    """``(path, class)`` pairs as lower-case absolute roots, longest first."""

    normalized: list[tuple[str, ResourceClass]] = []
    for path, resource_class in manifest_resources or ():
        if resource_class not in RESOURCE_CLASSES or not path:
            continue
        root = str(path).replace("\\", "/").lower().rstrip("/")
        if not root.startswith("/"):
            root = f"/{root}"
        normalized.append((root, resource_class))  # type: ignore[arg-type]
    normalized.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(normalized)


def is_under(normalized: str, root: str) -> bool:
    """True when ``normalized`` is ``root`` itself or sits beneath it."""

    return normalized == root or normalized.startswith(f"{root}/")


def classify_resource_class(
    path: str,
    manifest_resources: ResourceTable = (),
) -> ResourceClass | None:
    """The class of ``path``: the contract's roots, then the built-in table.

    A relative path is taken relative to the container root, which is where
    every protected root lives. Callers that know the agent's working
    directory resolve observed paths against it first.
    """

    normalized = path.replace("\\", "/").lower()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    # Anchored at a path root, never a substring: /app/tests/x is the agent's
    # own folder, not the verifier's /tests.
    for root, resource_class in manifest_resources:
        if is_under(normalized, root):
            return resource_class
    for root, resource_class in BUILTIN_PROTECTED_ROOTS:
        if is_under(normalized, root):
            return resource_class
    if normalized.rsplit("/", 1)[-1] in REWARD_OUTPUT_NAMES:
        return "RewardOutput"
    for root in BUILTIN_AGENT_WRITABLE_ROOTS:
        if is_under(normalized, root):
            return "AgentWritable"
    if normalized.rsplit("/", 1)[-1] == "instruction.md":
        return "Public"
    return None


__all__ = [
    "BUILTIN_AGENT_WRITABLE_ROOTS",
    "BUILTIN_PROTECTED_ROOTS",
    "REWARD_OUTPUT_NAMES",
    "ResourceTable",
    "classify_resource_class",
    "is_under",
    "normalize_manifest_resources",
]
