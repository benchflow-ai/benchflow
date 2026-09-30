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

from benchflow.integrity.constants import INTEGRITY_MODES, IntegrityMode
from benchflow.integrity.trial import (
    audit_trial,
    normalize_integrity_mode,
    strict_launch_issues,
)
from benchflow.integrity.verdict import (
    IntegrityReport,
    IntegrityVerdict,
    read_verdict,
)

__all__ = [
    "INTEGRITY_MODES",
    "IntegrityMode",
    "IntegrityReport",
    "IntegrityVerdict",
    "audit_trial",
    "normalize_integrity_mode",
    "read_verdict",
    "strict_launch_issues",
]
