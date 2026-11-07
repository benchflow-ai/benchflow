"""Reward integrity for BenchFlow runs (BenchShield, arXiv 2609.11028).

``integrity: audit`` (every backend) records host-side evidence of what the
agent did and checks it against the task's contract; ``integrity: strict``
also runs the verifier in BenchFlow's separate verifier sandbox, where a
clean run can be certified. Each audited trial gets ``integrity/`` with the
contract, action records, a hash-chained event stream, the trace check and
``claim_verdict.json``. Verdicts are ``Checked``, ``VectorExposed``,
``AgentViolation`` and ``Rejected``; a verdict never changes a reward.

The lifecycle model, event schema, trace checker and claim rules are ported
from BenchGuard (arXiv 2609.11028).

Read verdicts with ``bf.load_trial(path).integrity`` or
``bf.load_job(path).integrity()``; re-verdict a stored trial with
:func:`audit_trial`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchflow.integrity.constants import (
    INTEGRITY_MODES,
    IntegrityMode,
    normalize_integrity_mode,
)

# The pipeline's names resolve on first use (PEP 562): every run imports
# ``benchflow.integrity.constants`` (this file first) to read its mode, and
# the contract, event and claim modules take about 30 ms that a run with
# integrity off never needs.
if TYPE_CHECKING:
    from benchflow.integrity.trial import (
        audit_trial,
        resolve_verifier_isolation,
        separate_launch_issues,
        strict_launch_issues,
    )
    from benchflow.integrity.verdict import (
        IntegrityReport,
        IntegrityVerdict,
        read_verdict,
    )

_LAZY = {
    "resolve_verifier_isolation": "benchflow.integrity.trial",
    "separate_launch_issues": "benchflow.integrity.trial",
    "audit_trial": "benchflow.integrity.trial",
    "strict_launch_issues": "benchflow.integrity.trial",
    "IntegrityReport": "benchflow.integrity.verdict",
    "IntegrityVerdict": "benchflow.integrity.verdict",
    "read_verdict": "benchflow.integrity.verdict",
}

__all__ = [
    "INTEGRITY_MODES",
    "IntegrityMode",
    "IntegrityReport",
    "IntegrityVerdict",
    "audit_trial",
    "normalize_integrity_mode",
    "read_verdict",
    "resolve_verifier_isolation",
    "separate_launch_issues",
    "strict_launch_issues",
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
