"""Verifier ($V$) — maps agent completion to a reward signal.

Internalized from Harbor's Verifier class. Supports two verification methods,
selected by ``[verifier].type`` in ``task.toml``:

- ``"test-script"`` (default): run ``tests/test.sh`` inside the sandbox and
  parse ``reward.txt`` / ``reward.json``.
- ``"llm-judge"``: download the agent's deliverables and grade them against a
  human-authored rubric using an LLM judge (see #270).

This module is a thin façade. The implementation now lives in sibling
``verifier_*`` modules:

- ``verifier_core`` — the ``Verifier`` class (kept whole).
- ``verifier_errors`` — ``VerifierResult`` and the exception hierarchy.
- ``verifier_scan`` — dep-install failure scanning.
- ``verifier_script_strategy`` — script-strategy command building.
- ``verifier_reward_kit`` — Reward Kit resolution and manifest building.
- ``verifier_ors_episode`` — ORS-episode reward evidence parsing.
- ``verifier_judge_inputs`` — agent-judge input reading and scoring.

Every public and underscore symbol previously importable from
``benchflow.task.verifier`` (including patch targets like
``benchflow.task.verifier.Verifier``) is re-exported here so the import and
monkeypatch surface is byte-identical.
"""

from __future__ import annotations

import json as json
import logging as logging
import math as math
import shlex as shlex
import shutil as shutil
from dataclasses import asdict as asdict
from pathlib import Path as Path
from pathlib import PurePosixPath as PurePosixPath
from typing import Any as Any
from typing import cast as cast

from pydantic import BaseModel as BaseModel

from benchflow._utils.scoring import (
    VERIFIER_DEP_INSTALL_MARKERS as VERIFIER_DEP_INSTALL_MARKERS,
)
from benchflow._utils.scoring import (
    contains_verifier_dep_install_marker as contains_verifier_dep_install_marker,
)
from benchflow.rewards.events import Granularity as Granularity
from benchflow.rewards.events import RewardEvent as RewardEvent
from benchflow.rewards.events import Space as Space
from benchflow.rewards.protocol import VerifyResult as VerifyResult
from benchflow.rewards.rubric_config import (
    criteria_aggregate_policy_from_rubric as criteria_aggregate_policy_from_rubric,
)
from benchflow.rewards.validation import (
    apply_aggregate_policy as apply_aggregate_policy,
)
from benchflow.rewards.validation import (
    is_valid_reward_number as is_valid_reward_number,
)
from benchflow.rewards.validation import validate_reward_map as validate_reward_map
from benchflow.sandbox.lockdown import _exec_return_code as _exec_return_code
from benchflow.sandbox.lockdown import (
    clear_verifier_output_dir as clear_verifier_output_dir,
)
from benchflow.task.env import resolve_env_vars as resolve_env_vars
from benchflow.task.paths import RolloutPaths as RolloutPaths
from benchflow.task.paths import SandboxPaths as SandboxPaths
from benchflow.task.verifier_core import Verifier as Verifier
from benchflow.task.verifier_core import logger as logger
from benchflow.task.verifier_document import VerifierDocument as VerifierDocument
from benchflow.task.verifier_document import VerifierStrategy as VerifierStrategy
from benchflow.task.verifier_document import (
    load_verifier_document as load_verifier_document,
)
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
from benchflow.task.verifier_judge_inputs import (
    _AGENT_JUDGE_INPUT_CHAR_LIMIT as _AGENT_JUDGE_INPUT_CHAR_LIMIT,
)
from benchflow.task.verifier_judge_inputs import (
    _agent_judge_score as _agent_judge_score,
)
from benchflow.task.verifier_judge_inputs import (
    _local_rollout_input_path as _local_rollout_input_path,
)
from benchflow.task.verifier_judge_inputs import (
    _read_agent_judge_input as _read_agent_judge_input,
)
from benchflow.task.verifier_judge_inputs import (
    _safe_input_filename as _safe_input_filename,
)
from benchflow.task.verifier_ors_episode import (
    _bounded_ors_reward as _bounded_ors_reward,
)
from benchflow.task.verifier_ors_episode import (
    _count_ors_episode_records as _count_ors_episode_records,
)
from benchflow.task.verifier_ors_episode import _is_ors_response as _is_ors_response
from benchflow.task.verifier_ors_episode import (
    _load_ors_episode_records as _load_ors_episode_records,
)
from benchflow.task.verifier_ors_episode import _ors_event as _ors_event
from benchflow.task.verifier_ors_episode import _ors_events as _ors_events
from benchflow.task.verifier_ors_episode import _ors_granularity as _ors_granularity
from benchflow.task.verifier_ors_episode import _ors_items as _ors_items
from benchflow.task.verifier_ors_episode import (
    _ors_records_to_verify_result as _ors_records_to_verify_result,
)
from benchflow.task.verifier_ors_episode import _ors_space as _ors_space
from benchflow.task.verifier_reward_kit import _jsonable as _jsonable
from benchflow.task.verifier_reward_kit import (
    _llm_judge_input_dir as _llm_judge_input_dir,
)
from benchflow.task.verifier_reward_kit import (
    _reward_kit_criteria as _reward_kit_criteria,
)
from benchflow.task.verifier_reward_kit import (
    _reward_kit_criteria_policy as _reward_kit_criteria_policy,
)
from benchflow.task.verifier_reward_kit import (
    _reward_kit_manifest_json as _reward_kit_manifest_json,
)
from benchflow.task.verifier_reward_kit import _reward_kit_root as _reward_kit_root
from benchflow.task.verifier_reward_kit import _reward_kit_runner as _reward_kit_runner
from benchflow.task.verifier_reward_kit import (
    _safe_strategy_relative_path as _safe_strategy_relative_path,
)
from benchflow.task.verifier_scan import (
    _DEP_INSTALL_DIAGNOSTIC as _DEP_INSTALL_DIAGNOSTIC,
)
from benchflow.task.verifier_scan import _SCAN_CHUNK_BYTES as _SCAN_CHUNK_BYTES
from benchflow.task.verifier_scan import (
    _has_dep_install_failure as _has_dep_install_failure,
)
from benchflow.task.verifier_script_strategy import (
    _has_aggregate_declaration as _has_aggregate_declaration,
)
from benchflow.task.verifier_script_strategy import (
    _relative_posix_path as _relative_posix_path,
)
from benchflow.task.verifier_script_strategy import (
    _script_strategy_chmod_command as _script_strategy_chmod_command,
)
from benchflow.task.verifier_script_strategy import (
    _script_strategy_command as _script_strategy_command,
)
from benchflow.task.verifier_script_strategy import (
    _script_strategy_first_token as _script_strategy_first_token,
)
