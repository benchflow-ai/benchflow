"""Per-child branch deltas — the recorded change a branch child runs under.

A branch child's delta is a tuple over the three run-level variation axes plus
an injected prompt (rollout-branching RFC §3.3). Every member reuses an
existing, content-addressed mechanism: ``environment_ref`` is an S-axis
registry ref, ``config_override`` a C-axis allowlisted patch hashed exactly
like the run-level overlay (#790), ``skill_mode`` the install-time skills
toggle, and ``injected_prompt`` an explicit, recorded first message — never a
silent injection (#908).

:class:`BranchDelta` is the *schema*: all four fields exist now so the
artifact format is stable. Which fields the branch engine executes is the
engine's contract (:mod:`benchflow.rollout_branch`); provenance hashes raw
content (the prompt text never appears in artifacts, only its digest).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchflow._utils.config_override import overlay_hash
from benchflow._utils.content_address import sha256_prefixed
from benchflow.skill_policy import SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL

# The skill modes a branch child may flip to. ``self-gen`` is a run-level mode,
# not a per-child ablation axis, so it is not branchable.
_BRANCH_SKILL_MODES = frozenset({SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL})


@dataclass(frozen=True)
class BranchDelta:
    """The exactly-one-controlled-change a branch child runs under (RFC §3.3).

    All fields default to ``None`` (= inherit the parent's value); an
    all-``None`` delta is the zero-delta child, byte-for-byte today's branch
    behavior. ``skill_mode`` is validated against the branchable modes
    (``no-skill`` / ``with-skill``) at construction — a bad mode fails closed
    here, not at child run time.
    """

    environment_ref: str | None = None
    config_override: dict[str, Any] | None = None
    skill_mode: str | None = None
    injected_prompt: str | None = None

    def __post_init__(self) -> None:
        if self.skill_mode is not None and self.skill_mode not in _BRANCH_SKILL_MODES:
            raise ValueError(
                f"skill_mode must be one of {sorted(_BRANCH_SKILL_MODES)} when "
                f"set, got {self.skill_mode!r}"
            )

    @property
    def is_empty(self) -> bool:
        """True iff every field is unset — the zero-delta child."""
        return (
            self.environment_ref is None
            and self.config_override is None
            and self.skill_mode is None
            and self.injected_prompt is None
        )

    def provenance_dict(self) -> dict[str, Any]:
        """The delta as recorded in lineage artifacts (RFC §3.4).

        Small literal fields (``environment_ref``, ``skill_mode``) are recorded
        verbatim; content-bearing fields are recorded as sha256 content
        addresses only — ``config_override`` hashed over its canonical JSON
        (``sort_keys=True``, the run-level overlay's exact hash) and
        ``injected_prompt`` over its UTF-8 text. No raw prompt text ever
        appears in provenance. Unset fields serialize as ``null``.
        """
        return {
            "environment_ref": self.environment_ref,
            "config_override_sha256": (
                overlay_hash(self.config_override)
                if self.config_override is not None
                else None
            ),
            "skill_mode": self.skill_mode,
            "injected_prompt_sha256": (
                sha256_prefixed(self.injected_prompt.encode())
                if self.injected_prompt is not None
                else None
            ),
        }
