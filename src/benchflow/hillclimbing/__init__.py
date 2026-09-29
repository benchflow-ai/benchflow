"""``bench hillclimb`` / ``bf.hillclimb``: automated eval hill-climbing.

An optimizer agent edits a *surface* the agent under test receives (a skills
folder, or a prompt prefix) one targeted patch at a time. Each patch is kept
only if it improves both the train split, by at least ``min_gain``, and the
held-out test split; the optimizer runs in a sandbox that never holds the
test split. See docs/hillclimb.md.

Modules: :mod:`.split` (train/test), :mod:`.stats` (bootstrap intervals and
the noise gate), :mod:`.surface` (versions, diffs, history),
:mod:`.evaluate` (runs as BenchFlow jobs), :mod:`.proposer` (the sandboxed
optimizer), :mod:`.engine` (the loop), :mod:`.record` (``hillclimb.json``)
and :mod:`.report` (the HTML page).
"""

from benchflow.hillclimbing.engine import (
    Decision,
    HillclimbConfig,
    HillclimbError,
    HillclimbResult,
    ahillclimb,
    decide,
    hillclimb,
    load,
    summarize,
)
from benchflow.hillclimbing.proposer import ProposerSettings
from benchflow.hillclimbing.record import HillclimbDoc, json_schema, load_record
from benchflow.hillclimbing.split import Split, SplitError, load_split_file, make_split

__all__ = [
    "Decision",
    "HillclimbConfig",
    "HillclimbDoc",
    "HillclimbError",
    "HillclimbResult",
    "ProposerSettings",
    "Split",
    "SplitError",
    "ahillclimb",
    "decide",
    "hillclimb",
    "json_schema",
    "load",
    "load_record",
    "load_split_file",
    "make_split",
    "summarize",
]
