"""Human preferences and rubric scores on finished trials.

``issue`` makes blind pairs of trial outputs, ``import_votes`` takes the
votes a rater's tool collected, and ``rank`` turns them into ratings. See
docs/tasks/human-preferences.md.
"""

from benchflow.preferences.store import (
    Conflict,
    PreferenceError,
    import_votes,
    issue,
    rank,
)

__all__ = ["Conflict", "PreferenceError", "import_votes", "issue", "rank"]
