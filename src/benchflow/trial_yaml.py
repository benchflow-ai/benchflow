"""Compatibility wrappers for rollout YAML loading."""

from benchflow.rollouts.yaml import (
    job_config_from_yaml,
    rollout_config_from_dict,
    rollout_config_from_yaml,
)
from benchflow.rollouts.yaml import (
    load_rollout_yaml as load_trial_yaml,
)

trial_config_from_yaml = rollout_config_from_yaml
trial_config_from_dict = rollout_config_from_dict

__all__ = [
    "job_config_from_yaml",
    "load_trial_yaml",
    "rollout_config_from_dict",
    "rollout_config_from_yaml",
    "trial_config_from_dict",
    "trial_config_from_yaml",
]
