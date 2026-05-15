"""Rollout lifecycle entry point.

This module is the v0.4 home for the execution lifecycle. During the migration
it delegates to the existing decomposed lifecycle implementation; follow-up
commits will remove the legacy module name once all callsites move here.
"""

from __future__ import annotations

from benchflow.trial import Trial


class Rollout(Trial):
    """One attempt on one task in one sandbox."""
