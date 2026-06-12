"""Resume a previous, unfinished (timed-out) agent run to completion.

``benchflow continue <orig-run-output-folder>`` is a *standalone* tool that does
**not** touch benchflow's normal run path. It reconstructs a timed-out run's
exact workspace and agent memory from the recorded trajectory (record-replay),
then lets the agent continue as if the timeout had simply been larger, and
writes a new HF-compatible result folder linked to the parent.

The mechanism (agreed design):

1. Load the original run folder (``config.json`` + ``trajectory/llm_trajectory.jsonl``).
2. Boot a fresh *pristine* sandbox from the same base image.
3. Stand up a :class:`~benchflow.continue_run.replay_proxy.ReplayProxy` that
   OpenHands talks to via ``LLM_BASE_URL``. It serves the recorded LLM
   responses **in order**, so the agent re-executes its own past decisions for
   real — rebuilding the byte-exact workspace and its exact internal state.
4. When the recorded responses run out (the timeout cut-point), the proxy flips
   to the **live** model and the agent continues — no injected prompt.
5. Re-verify and write a new folder with ``continued_from`` provenance.

Only the ``openhands`` agent is supported for now (the LLM-proxy seam this relies
on is wired for openhands via ``LLM_BASE_URL``).
"""

from benchflow.continue_run.run_folder import (
    RunFolder,
    RunFolderError,
    load_run_folder,
)

__all__ = [
    "RunFolder",
    "RunFolderError",
    "load_run_folder",
]
