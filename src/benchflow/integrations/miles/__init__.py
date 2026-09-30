"""Train on BenchFlow tasks with Miles (radixark/miles).

BenchFlow runs each Miles rollout as one episode of the RL cookbooks'
bash/submit harness in a BenchFlow sandbox, with the model calls going to the
rollout's token-in/token-out session, and answers with an attribution-aware
reward (see :mod:`benchflow.integrations.miles.episode`). Miles reaches it
through a small environment server (``python -m benchflow.integrations.miles
serve``) that its agent function calls once per rollout.
"""

from benchflow.integrations.miles.data import dataset_rows, write_dataset
from benchflow.integrations.miles.episode import (
    ABORTED_EXIT_STATUS,
    BASH_TIMEOUT_SEC,
    HARNESS_MESSAGE,
    MAX_OUTPUT_CHARS,
    MAX_TURNS,
    SUBMIT_PATH,
    EpisodeOutcome,
    EpisodeRequest,
    EpisodeRequestError,
    EpisodeSettings,
    exit_status_for,
    run_episode,
)

__all__ = [
    "ABORTED_EXIT_STATUS",
    "BASH_TIMEOUT_SEC",
    "HARNESS_MESSAGE",
    "MAX_OUTPUT_CHARS",
    "MAX_TURNS",
    "SUBMIT_PATH",
    "EpisodeOutcome",
    "EpisodeRequest",
    "EpisodeRequestError",
    "EpisodeSettings",
    "dataset_rows",
    "exit_status_for",
    "run_episode",
    "write_dataset",
]
