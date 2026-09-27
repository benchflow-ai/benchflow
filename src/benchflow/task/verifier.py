"""Verifier ($V$) — maps agent completion to a reward signal.

Internalized from Harbor's Verifier class. Supports two verification methods,
selected by ``[verifier].type`` in ``task.toml``:

- ``"test-script"`` (default): run ``tests/test.sh`` inside the sandbox and
  parse ``reward.txt`` / ``reward.json``.
- ``"llm-judge"``: download the agent's deliverables and grade them against a
  human-authored rubric using an LLM judge (see #270).

This module is a thin façade. The implementation lives in sibling
``verifier_*`` modules:

- ``verifier_core`` — the ``Verifier`` class (kept whole).
- ``verifier_errors`` — ``VerifierResult`` and the exception hierarchy.
- ``verifier_scan`` — dep-install failure scanning.
- ``verifier_script_strategy`` — script-strategy command building.
- ``verifier_reward_kit`` — Reward Kit resolution and manifest building.
- ``verifier_ors_episode`` — ORS-episode reward evidence parsing.
- ``verifier_judge_inputs`` — agent-judge input reading and scoring.

It re-exports ``Verifier``, its result and error types, and the dep-install
scan helpers that callers import from ``benchflow.task.verifier``.
"""

from __future__ import annotations

from benchflow.task.verifier_core import Verifier as Verifier
from benchflow.task.verifier_errors import AddTestsDirError as AddTestsDirError
from benchflow.task.verifier_errors import AgentJudgeInputError as AgentJudgeInputError
from benchflow.task.verifier_errors import (
    DownloadVerifierDirError as DownloadVerifierDirError,
)
from benchflow.task.verifier_errors import ORSEpisodeInputError as ORSEpisodeInputError
from benchflow.task.verifier_errors import RewardFileEmptyError as RewardFileEmptyError
from benchflow.task.verifier_errors import (
    RewardFileNotFoundError as RewardFileNotFoundError,
)
from benchflow.task.verifier_errors import RubricNotFoundError as RubricNotFoundError
from benchflow.task.verifier_errors import (
    UnsupportedVerifierStrategyError as UnsupportedVerifierStrategyError,
)
from benchflow.task.verifier_errors import (
    VerifierOutputParseError as VerifierOutputParseError,
)
from benchflow.task.verifier_errors import VerifierResult as VerifierResult
from benchflow.task.verifier_scan import (
    _DEP_INSTALL_DIAGNOSTIC as _DEP_INSTALL_DIAGNOSTIC,
)
from benchflow.task.verifier_scan import _SCAN_CHUNK_BYTES as _SCAN_CHUNK_BYTES
from benchflow.task.verifier_scan import (
    _has_dep_install_failure as _has_dep_install_failure,
)
