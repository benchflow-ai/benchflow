"""What ``bench hub push`` uploads lets the Hub count by BenchFlow's scoring rule.

A peer framework's new hub averaged only the trials whose reward was valid,
while its local mean counted every trial with an error as 0: one job, two
scores. BenchFlow Hub is a separate program; the push now carries what it
needs to give BenchFlow's headline: the rule's id, each trial's
classification (outcome, who ended it, error class, setting row) and each
setting's figures, in ``scoring.json`` beside the job's trials.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from typer.testing import CliRunner

from benchflow._utils.scoring import score_summary_fields
from benchflow.cli.main import app
from benchflow.scoring import RULE_ID, score, trial_record
from benchflow.setting_groups import summary_settings
from tests import _run_records as rr
from tests.test_hub_client import hub  # noqa: F401  (the fake Hub fixture)


def _push(job: Path, hub_url: str) -> dict:
    rows = [json.loads(p.read_text()) for p in sorted(job.glob("*/result.json"))]
    summary = {"job_name": job.name, **score_summary_fields(rows)}
    summary["scoring"]["settings"] = summary_settings(job, [summary])
    (job / "summary.json").write_text(json.dumps(summary))
    out = CliRunner().invoke(app, ["hub", "push", str(job), "--hub", hub_url])
    assert out.exit_code == 0, out.output
    return summary


def _uploaded(hub_server) -> dict:
    data = hub_server.uploads[-1]["files"][0][2]
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        (member,) = [m for m in tar.getmembers() if m.name.endswith("/scoring.json")]
        handle = tar.extractfile(member)
        assert handle is not None
        return json.loads(handle.read())


def test_the_push_carries_the_rule_and_each_trial_s_class(tmp_path, hub) -> None:  # noqa: F811
    job = rr.evaluation_job(tmp_path / "jobs" / "run", ["a", "b", "c", "d"])
    calls = [
        rr.anthropic_call(
            model="anthropic/claude-opus-5",
            served="claude-opus-5-20260801",
            route="anthropic/claude-opus-5",
            effort="high",
        )
    ]
    for i, (task, reward, error, category) in enumerate(
        [
            ("a", 1.0, None, None),
            ("b", 0.0, None, None),
            (
                "c",
                None,
                "provider api error [api_error/transient]: HTTP 503",
                "api_error",
            ),
            ("d", None, "Agent idle for 600s", "idle_timeout"),
        ]
    ):
        rr.trial(job, task, i, reward=reward, model="anthropic/claude-opus-5",
                 effort="high", calls=calls, error=error, error_category=category)  # fmt: skip
    summary = _push(job, hub.url)
    document = _uploaded(hub)
    assert document["schema"] == "benchflow.scoring.v1"
    assert document["rule"]["id"] == RULE_ID == summary["scoring_rule"]["id"]
    # Each trial's classification is the rule's, read from its result.json.
    by_task = {t["task"]: t for t in document["trials"]}
    for path in sorted(job.glob("*/result.json")):
        result = json.loads(path.read_text())
        mine = by_task[result["task_name"]]
        expected = trial_record(result)
        assert {k: mine[k] for k in expected} == expected
        assert mine["setting"] == 0 and mine["control"] is None
    assert [by_task[t]["ending"] for t in "abcd"] == [None, None, "provider", "harness"]
    # The Hub's headline, applied to the trials, is BenchFlow's.
    again = score(document["trials"]).to_dict()
    assert again == document["headline"]
    for key in ("n", "passed", "score", "ci95", "outcomes", "endings"):
        assert document["headline"][key] == summary["scoring"][key]
    assert document["settings"]["mixed"] is False


def test_a_mixed_job_s_push_has_no_headline_and_one_row_per_setting(
    tmp_path,
    hub,  # noqa: F811
) -> None:
    from tests.test_setting_rows import BEDROCK, DIRECT, _resumed_job

    job = _resumed_job(tmp_path)
    _push(job, hub.url)
    document = _uploaded(hub)
    assert document["headline"] is None
    settings = document["settings"]
    assert settings["split"] is True and settings["warning"].startswith(
        "mixed settings: these trials ran 2 settings"
    )
    assert [row["route"] for row in settings["rows"]] == [DIRECT, BEDROCK]
    # Each row's figures are the rule over that row's trials.
    for index, row in enumerate(settings["rows"]):
        trials = [t for t in document["trials"] if t["setting"] == index]
        assert score(trials).to_dict() == row["scoring"]
