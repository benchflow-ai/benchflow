"""``--integrity audit`` with a relative ``--jobs-dir`` (DF-p4-upgrader-2).

Every 0.7 example uses a relative ``--jobs-dir``; the audit stamped every such
trial Rejected (``integrity_error:ValueError ... is not in the subpath of``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

# --- DF-2: --integrity audit with a relative --jobs-dir ------------------------


def test_the_audit_reads_a_relative_trial_folder(tmp_path, monkeypatch) -> None:
    from benchflow.integrity.emit import emit_integrity
    from tests.test_integrity import _evidence

    absolute = _evidence(tmp_path / "abs", [])
    relative = _evidence(tmp_path / "jobs" / "job", [])
    monkeypatch.chdir(tmp_path)
    relative = dataclasses.replace(relative, trial_dir=Path("jobs/job/trial"))

    emit_integrity(absolute)
    emit_integrity(relative)

    def read(trial: Path, name: str) -> dict:
        return json.loads((trial / "integrity" / name).read_text())

    rel = read(tmp_path / "jobs/job/trial", "claim_verdict.json")
    assert "integrity_error" not in json.dumps(rel)
    # The same verdict as with an absolute --jobs-dir.
    assert (
        rel["certification"]
        == read(absolute.trial_dir, "claim_verdict.json")["certification"]
    )
    provenance = read(tmp_path / "jobs/job/trial", "reward_provenance.json")
    assert provenance["reward_file"] == "verifier/reward.txt"


def test_a_reachable_path_is_relative_to_a_relative_folder(
    tmp_path, monkeypatch
) -> None:
    from benchflow._utils.reachable_write import ReachablePath

    (tmp_path / "jobs" / "t" / "verifier").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    reward = ReachablePath.rollout(Path("jobs/t"), "verifier") / "reward.txt"
    assert reward.relative_to(Path("jobs/t")) == Path("verifier/reward.txt")
    assert reward.is_relative_to(Path("jobs"))
    assert not reward.is_relative_to(Path("other"))
