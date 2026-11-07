"""Guards FF-sw-daytona against owner collisions at 363d815b."""

import pytest

from benchflow.sandbox import daytona_reaper as reaper
from benchflow.sandbox.daytona_content_snapshots import owner_prefix


@pytest.mark.parametrize(
    "owner", ["team/a", "team:a", "a" * 49, " a", "a ", "...", "équipe", "a\nb"]
)
@pytest.mark.parametrize(
    "operation",
    [reaper._benchflow_owned_labels, reaper.owner_snapshot_prefix, owner_prefix],
)
def test_ambiguous_owner_is_rejected(monkeypatch, owner, operation):
    """Guard 363d815b: no lossy normalization may authorize deletion."""
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", owner)
    with pytest.raises(ValueError, match="BENCHFLOW_DAYTONA_OWNER"):
        operation()


@pytest.mark.parametrize("owner", ["team-a", "team_A", "a.b", "a" * 48])
def test_canonical_owner_is_preserved(monkeypatch, owner):
    """Guard 363d815b: existing canonical ownership remains usable."""
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", owner)
    assert reaper._benchflow_owner_scope() == owner
