"""Backward-compat shim — re-exports everything from benchflow.rollout.

All public names (Rollout, RolloutConfig, Trial, TrialConfig, etc.) are
available here so that ``from benchflow.trial import Trial`` keeps working.

Tests may also ``patch("benchflow.trial.<name>", ...)`` so we explicitly
pull in every name that rollout.py uses at module scope.
"""

# Re-export all public AND private names from rollout
# Re-export names that rollout.py imports at module scope so that
# ``patch("benchflow.trial.<name>", ...)`` in tests keeps working.
from benchflow._acp_run import connect_acp, execute_prompts  # noqa: F401
from benchflow._agent_env import resolve_agent_env  # noqa: F401
from benchflow._agent_setup import apply_web_tool_policy, deploy_skills  # noqa: F401
from benchflow._credentials import (  # noqa: F401
    upload_subscription_auth,
    write_credential_files,
)
from benchflow._env_setup import _create_environment  # noqa: F401
from benchflow.rollout import *  # noqa: F403

# Explicit re-exports of private helpers and constants used by tests and self_gen
