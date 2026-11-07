"""The run card (benchflow.run_card): what a run is set to measure, before it runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, _task_digest_for
from benchflow.run_card import (
    RUN_CARD,
    RunCardError,
    build_run_card,
    card_lines,
    check_card,
    find_reference,
    load_reference,
    reference_findings,
)


def _task(
    root: Path, name: str, *, agent_sec: int | None = 600, network: str = "public"
) -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    agent = f"timeout_sec = {agent_sec}\n" if agent_sec is not None else ""
    (task / "task.toml").write_text(
        'version = "1.0"\n'
        f"[agent]\n{agent}"
        "[verifier]\ntimeout_sec = 120\n"
        f'[environment]\nnetwork_mode = "{network}"\ncpus = 2\nmemory_mb = 4096\n'
    )
    (task / "instruction.md").write_text("Write 42 to /app/answer.txt.\n")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    return task


def _tasks(tmp_path: Path, n: int = 3) -> Path:
    root = tmp_path / "tasks"
    for i in range(n):
        _task(root, f"task-{i}", agent_sec=600 if i else 1200)
    return root


def _evaluation(tmp_path: Path, tasks_dir: Path, **config) -> Evaluation:
    base = {
        "agent": "claude-agent-acp",
        "model": "claude-haiku-4-5-20251001",
        "concurrency": 2,
    }
    base.update(config)
    return Evaluation(
        tasks_dir=str(tasks_dir),
        jobs_dir=str(tmp_path / "jobs" / "claude"),
        config=EvaluationConfig(**base),
        preflight=False,
    )


DRUG_TARGET = """\
[reference]
source = "The benchmark paper, Table 2"
agent = "mini-swe-agent"
trials = 1
agent_timeout_sec = 600
temperature = 0.6
thinking = true
"""


def test_the_card_counts_every_trial_and_bounds_the_time(tmp_path):
    job = _evaluation(tmp_path, _tasks(tmp_path), planned_trials=8, trial_number=1)
    card = build_run_card(job, job._get_task_dirs())
    assert card["tasks"]["count"] == 3
    assert card["trials"] == {
        "per_task": 8,
        "this_trial": 1,
        "total": 24,
        "retries": 2,
        "seeds": None,
    }
    assert card["limits"]["agent_sec"] == {"min": 600.0, "max": 1200.0}
    assert card["limits"]["verifier_sec"] == {"min": 120.0, "max": 120.0}
    assert card["network"] == {"public": 3}
    assert card["sandbox"]["cpus"] == {"min": 2, "max": 2}
    # 8 trial jobs one after another, each 3 tasks 2 at a time (2 rounds): 16
    # rounds at the longest agent and verifier limits (the cold-reader test
    # found the bound treating the jobs as concurrent).
    assert card["estimate"]["rounds"] == 16
    assert card["estimate"]["max_wall_sec"] == 16 * (1200 + 120)
    lines = card_lines(card, tmp_path / "j")
    assert any("8 repeats one after another" in line for line in lines)
    assert any("each repeat is a job of its own)" in line for line in lines)
    # "Trial" is one task attempt; a pass over the task set is a repeat.
    assert (
        "  Trials:    24 in all, 8 per task (--trials 8 runs the task set 8 times; "
        "each repeat is a job of its own, and this is repeat 1 of 8, folder "
        "trial-01); 2 at a time on docker"
    ) in lines
    assert card["model"]["id"] == "claude-haiku-4-5-20251001"
    assert "claude-agent-acp@" in (card["agent"]["installs"] or "")
    assert card["reference"] is None


def test_a_task_with_no_agent_limit_leaves_no_upper_bound(tmp_path):
    root = tmp_path / "tasks"
    _task(root, "timed")
    _task(root, "untimed", agent_sec=None)
    job = _evaluation(tmp_path, root)
    card = build_run_card(job, job._get_task_dirs())
    assert card["limits"]["untimed_tasks"] == 1
    assert card["estimate"]["max_wall_sec"] is None
    lines = card_lines(card, tmp_path / "job")
    assert any("no estimate: a task has no agent limit" in line for line in lines)


def test_the_reference_names_what_differs_and_what_to_check(tmp_path):
    tasks = _tasks(tmp_path)
    (tasks / "run-reference.toml").write_text(DRUG_TARGET)
    job = _evaluation(tmp_path, tasks, planned_trials=8)
    card = build_run_card(job, job._get_task_dirs())
    deviations = {d["key"]: d for d in card["reference"]["deviations"]}
    assert deviations["agent"]["kind"] == "differs"
    assert deviations["trials"] == {
        "key": "trials",
        "reference": 1,
        "run": 8,
        "kind": "differs",
    }
    # One task has 1200 s: the run's limits are not the reference's 600 s.
    assert deviations["agent_timeout_sec"]["run"] == "600, 1200"
    assert deviations["temperature"]["kind"] == "check"
    assert deviations["thinking"]["kind"] == "check"
    findings = reference_findings(card)
    assert "differs: trials 8 (reference 1)" in findings
    assert any(f.startswith("check: temperature 0.6") for f in findings)
    assert any(f.startswith("check: thinking true in the reference") for f in findings)
    # Not strict: listed, never refused.
    check_card(card)


def test_a_strict_reference_refuses_a_run_that_differs_until_it_says_why(tmp_path):
    tasks = _tasks(tmp_path)
    (tasks / "run-reference.toml").write_text(DRUG_TARGET + "strict = true\n")
    job = _evaluation(tmp_path, tasks, planned_trials=8)
    card = build_run_card(job, job._get_task_dirs())
    with pytest.raises(RunCardError) as refused:
        check_card(card)
    assert "trials 8 (reference 1)" in str(refused.value)
    assert "agent claude-agent-acp (reference mini-swe-agent)" in str(refused.value)
    assert "--accept-deviations" in (refused.value.hint or "")

    accepted = _evaluation(
        tmp_path, tasks, planned_trials=8, accept_deviations="pass@8 study"
    )
    card = build_run_card(accepted, accepted._get_task_dirs())
    check_card(card)
    assert card["reference"]["accepted"] == "pass@8 study"
    assert "accepted: pass@8 study" in reference_findings(card)


def test_a_strict_refusal_comes_before_the_job_folder_exists(tmp_path):
    tasks = _tasks(tmp_path)
    (tasks / "run-reference.toml").write_text(DRUG_TARGET + "strict = true\n")
    job = _evaluation(tmp_path, tasks)
    with pytest.raises(RunCardError):
        job._prepare_run_card()
    assert not (tmp_path / "jobs").exists()


def test_a_run_that_matches_a_strict_reference_runs(tmp_path):
    root = tmp_path / "tasks"
    for i in range(2):
        _task(root, f"task-{i}")
    (root / "run-reference.toml").write_text(
        '[reference]\nagent = "claude-agent-acp"\ntrials = 1\n'
        "agent_timeout_sec = 600\nstrict = true\n"
    )
    job = _evaluation(tmp_path, root)
    job._prepare_run_card()
    differs = [
        d for d in job._run_card["reference"]["deviations"] if d["kind"] == "differs"
    ]
    assert differs == []


def test_the_reference_beside_a_snapshot_or_a_set_is_found(tmp_path):
    snapshot = tmp_path / "snap"
    task = _task(snapshot / "tasks", "one")
    (snapshot / "run-reference.toml").write_text("[reference]\ntrials = 1\n")
    assert find_reference(snapshot / "tasks") == snapshot / "run-reference.toml"
    assert find_reference(snapshot / "tasks" / "one") is None
    (task.parent / "run-reference.toml").write_text("[reference]\ntrials = 2\n")
    # One task of a set: the set's reference.
    assert find_reference(task) == task.parent / "run-reference.toml"


@pytest.mark.parametrize(
    "text,message",
    [
        ("trials = 1\n", "one [reference] table"),
        ("[reference]\ntries = 1\n", "unknown key tries"),
        ("[reference]\ntrials = true\n", "trials must be a whole number"),
        ("[reference]\ntemperature = 'low'\n", "temperature must be a number"),
        ("[reference]\nstrict = 1\n", "strict must be true or false"),
    ],
)
def test_a_bad_reference_is_refused_with_what_to_fix(tmp_path, text, message):
    path = tmp_path / "run-reference.toml"
    path.write_text(text)
    with pytest.raises(RunCardError) as refused:
        load_reference(path)
    assert message in str(refused.value)
    assert "before-you-run.md#the-run-card-010" in (refused.value.hint or "")


def _control_job(
    jobs_dir: Path, agent: str, rewards: dict[str, float], digests: dict[str, str]
) -> Path:
    job = jobs_dir / "2026-11-04__08-00-00"
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "config": {"agent": agent}})
    )
    for name, reward in rewards.items():
        trial = job / f"{name}__0000aaaa"
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": name,
                    "rollout_name": trial.name,
                    "agent": agent,
                    "rewards": {"reward": reward},
                    "error": None,
                    "verifier_error": None,
                }
            )
        )
        (trial / "config.json").write_text(
            json.dumps({"agent": agent, "task_digest": digests[name]})
        )
    (job / "summary.json").write_text(json.dumps({"total": len(rewards)}))
    return job


def test_controls_count_only_trials_of_the_same_task_content(tmp_path):
    tasks = _tasks(tmp_path)
    job = _evaluation(tmp_path, tasks)
    task_dirs = job._get_task_dirs()
    digests = {d.name: _task_digest_for(d, job._config) for d in task_dirs}
    _control_job(
        tmp_path / "jobs" / "oracle",
        "oracle",
        {"task-0": 1.0, "task-1": 1.0, "task-2": 0.0},
        digests,
    )
    stale = dict(digests, **{"task-0": "sha256:older"})
    _control_job(
        tmp_path / "jobs" / "nop", "nop", {"task-0": 0.0, "task-1": 0.0}, stale
    )
    card = build_run_card(job, task_dirs)
    oracle, nop = card["controls"]["oracle"], card["controls"]["nop"]
    assert (oracle["tasks"], oracle["of"], oracle["as_expected"]) == (3, 3, 2)
    # nop ran task-0 on other content: only task-1 counts.
    assert (nop["tasks"], nop["as_expected"]) == (1, 1)
    line = next(line for line in card_lines(card, tmp_path / "j") if "Controls" in line)
    assert (
        "oracle on 3 of 3 tasks, 2 scored 1 (oracle/2026-11-04__08-00-00, same task digests)"
        in line
    )
    # A shortfall is said on the line: those tasks measure nothing as they are.
    assert (
        "(oracle/2026-11-04__08-00-00, same task digests); 1 did not: those tasks measure nothing"
        in line
    )
    assert "nop on 1 of 3 tasks, 1 scored 0" in line


def test_no_controls_names_the_commands_to_run_them(tmp_path):
    tasks = _tasks(tmp_path)
    job = _evaluation(tmp_path, tasks)
    card = build_run_card(job, job._get_task_dirs())
    line = next(line for line in card_lines(card, tmp_path / "j") if "Controls" in line)
    assert line.startswith("  Controls:  local jobs: oracle none found")
    assert f"oracle none found: bench run {tasks} --agent oracle" in line
    assert f"nop none found: bench run {tasks} --agent nop" in line


def test_a_model_from_another_provider_differs(tmp_path):
    """Round 5 of the release review: only the part after the last slash was
    compared, so two providers' deployments of one model matched."""
    from benchflow.run_card import _same_model

    assert not _same_model("openrouter/meta/llama-x", "together/meta/llama-x")
    assert not _same_model("xiaomi/mimo-v2.6-flash", "openai/mimo-v2.6-flash")
    assert _same_model("anthropic/claude-x", "claude-x")
    assert _same_model("Claude-X", "claude-x")
    assert not _same_model("claude-x", None)


def test_the_newest_control_wins_a_tie(tmp_path):
    """Round 5 of the release review: with two oracle jobs on the same tasks, the
    first one found was kept, so an older pass could hide a newer failure."""
    import os

    tasks = _tasks(tmp_path)
    job = _evaluation(tmp_path, tasks)
    task_dirs = job._get_task_dirs()
    digests = {d.name: _task_digest_for(d, job._config) for d in task_dirs}
    older = _control_job(
        tmp_path / "jobs" / "oracle", "oracle", dict.fromkeys(digests, 1.0), digests
    )
    newer = older.parent / "2026-11-04__09-00-00"
    newer.mkdir()
    for child in older.iterdir():
        target = newer / child.name
        if child.is_dir():
            target.mkdir()
            for f in child.iterdir():
                (target / f.name).write_text(f.read_text())
        else:
            target.write_text(child.read_text())
    result = newer / "task-0__0000aaaa" / "result.json"
    data = json.loads(result.read_text())
    data["rewards"] = {"reward": 0.0}
    result.write_text(json.dumps(data))
    os.utime(older / "evaluation.json", (1e9, 1e9))
    oracle = build_run_card(job, task_dirs)["controls"]["oracle"]
    assert oracle["job"] == str(newer)
    assert oracle["as_expected"] == 2


def test_a_control_from_another_release_is_said_and_this_ones_preferred(tmp_path):
    """Round 5 of the release review: controls matched by task content only, so
    a 0.9.1 nop control (0) stood for a task 0.10's nop passes."""
    from benchflow.evaluation import _benchflow_version

    tasks = _tasks(tmp_path)
    job = _evaluation(tmp_path, tasks)
    task_dirs = job._get_task_dirs()
    digests = {d.name: _task_digest_for(d, job._config) for d in task_dirs}
    old = _control_job(
        tmp_path / "jobs" / "oracle", "oracle", dict.fromkeys(digests, 1.0), digests
    )
    record = json.loads((old / "evaluation.json").read_text())
    (old / "evaluation.json").write_text(
        json.dumps({**record, "benchflow_version": "0.9.1"})
    )
    card = build_run_card(job, task_dirs)
    assert card["controls"]["oracle"]["benchflow_version"] == "0.9.1"
    line = next(line for line in card_lines(card, tmp_path / "j") if "Controls" in line)
    assert f"run with BenchFlow 0.9.1: rerun it on {_benchflow_version()}" in line
    # A control from this release wins over the older one, newer or not.
    mine = tmp_path / "jobs" / "oracle2" / "2026-11-01__08-00-00"
    _control_job(mine.parent, "oracle", dict.fromkeys(digests, 1.0), digests)
    (mine.parent / "2026-11-04__08-00-00").rename(mine)
    record = json.loads((mine / "evaluation.json").read_text())
    (mine / "evaluation.json").write_text(
        json.dumps({**record, "benchflow_version": _benchflow_version()})
    )
    import os

    os.utime(mine / "evaluation.json", (1e9, 1e9))
    assert build_run_card(job, task_dirs)["controls"]["oracle"]["job"] == str(mine)


def test_a_task_subset_is_shown_and_checked_against_the_reference(tmp_path):
    """Round 5 of the release review: `-i keep-fails` ran 1 of a folder's
    tasks, and the card said only "1 from" the folder."""
    tasks = _tasks(tmp_path)
    (tasks / "run-reference.toml").write_text(DRUG_TARGET)
    job = _evaluation(tmp_path, tasks, include_tasks={"task-1"})
    card = build_run_card(job, job._get_task_dirs())
    assert card["tasks"]["selection"]["include"] == ["task-1"]
    line = card_lines(card, tmp_path / "j")[1]
    assert line.endswith(", selected with -i task-1")
    assert (
        "check: this run selects tasks (-i task-1); the reference's numbers may be "
        "over the whole set"
    ) in reference_findings(card)


def test_install_pins_from_a_requirements_string_are_split(monkeypatch):
    """Round 5 of the release review: openhands' card line showed stray quotes
    and a literal \\n ('openhands-sdk==1.28.1\\nopenhands-tools==1.28.1)."""
    from types import SimpleNamespace

    from benchflow.agents import registry
    from benchflow.run_card import _installs

    command = (
        "printf 'openhands-sdk==1.28.1\\nopenhands-tools==1.28.1' > /tmp/r.txt && "
        "uv pip install -r /tmp/r.txt 'git+https://github.com/x/y@2df8a28'"
    )
    monkeypatch.setitem(
        registry.AGENTS, "fake-agent", SimpleNamespace(install_cmd=command)
    )
    assert _installs("fake-agent") == (
        "openhands-sdk==1.28.1 + openhands-tools==1.28.1 + git+https://github.com/x/y@2df8a28"
    )


def test_the_control_hint_runs_where_the_next_card_looks(tmp_path, monkeypatch):
    """Round 4 of the release review: the hint left out --jobs-dir, so a
    control run from it landed in ./jobs and the next card still found none."""
    monkeypatch.chdir(tmp_path)
    tasks = _tasks(tmp_path)
    job = _evaluation(tmp_path, tasks)  # jobs/claude
    task_dirs = job._get_task_dirs()
    card = build_run_card(job, task_dirs)
    lines = card_lines(card, job._jobs_dir / "2026-11-04__09-00-00")
    line = next(line for line in lines if "Controls" in line)
    assert f"bench run {tasks} --agent oracle --jobs-dir jobs/oracle" in line
    assert f"bench run {tasks} --agent nop --jobs-dir jobs/nop" in line
    digests = {d.name: _task_digest_for(d, job._config) for d in task_dirs}
    _control_job(
        tmp_path / "jobs" / "oracle", "oracle", dict.fromkeys(digests, 1.0), digests
    )
    assert build_run_card(job, task_dirs)["controls"]["oracle"]["tasks"] == 3
    # A trial job of --trials sits in <jobs dir>/<alias>/trial-NN: the hint
    # still names the folder beside the jobs dir the user gave.
    trial_job = tmp_path / "rc" / "claude" / "claude" / "trial-01" / "2026"
    line = next(line for line in card_lines(card, trial_job) if "Controls" in line)
    assert "--agent oracle --jobs-dir rc/oracle" in line
    # The default jobs folder needs no flag: a control starts its own job there.
    default = tmp_path / "jobs" / "2026-11-04__09-00-00"
    line = next(line for line in card_lines(card, default) if "Controls" in line)
    assert line.endswith(f"bench run {tasks} --agent nop")


def test_a_control_run_says_so(tmp_path):
    job = _evaluation(tmp_path, _tasks(tmp_path), agent="oracle", model=None)
    card = build_run_card(job, job._get_task_dirs())
    assert card["controls"] == {"this_run": "oracle"}
    lines = card_lines(card, tmp_path / "j")
    assert "  Model:     none (a control run)" in lines


def test_the_card_is_written_and_logged_with_the_reference_findings(tmp_path, caplog):
    tasks = _tasks(tmp_path)
    (tasks / "run-reference.toml").write_text(DRUG_TARGET)
    job = _evaluation(tmp_path, tasks, planned_trials=8)
    job._prepare_run_card()
    job_dir = job._jobs_dir / job._job_name
    job_dir.mkdir(parents=True)
    with caplog.at_level(logging.INFO, logger="benchflow.evaluation"):
        job._log_run_card(done=0)
    card = json.loads((job_dir / RUN_CARD).read_text())
    assert card["run_card"] == 1
    assert card["trials"]["total"] == 24
    assert card["trials"]["done_before_this_run"] == 0
    text = caplog.text
    assert "Run card (" in text
    assert "Trials:    24 in all, 8 per task" in text
    assert "The run differs from its reference setup" in text
    assert "differs: trials 8 (reference 1)" in text


def test_a_card_that_cannot_be_built_never_stops_the_run(tmp_path, monkeypatch, caplog):
    job = _evaluation(tmp_path, _tasks(tmp_path))

    def broken(*_args, **_kwargs):
        raise KeyError("unexpected")

    monkeypatch.setattr("benchflow.run_card.build_run_card", broken)
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        job._prepare_run_card()
    assert job._run_card is None
    assert "No run card for this job" in caplog.text


def test_a_trial_job_of_trials_finds_the_controls_beside_its_agent(tmp_path):
    tasks = _tasks(tmp_path)
    job = Evaluation(
        tasks_dir=str(tasks),
        jobs_dir=str(tmp_path / "jobs" / "pi-acp" / "trial-02"),
        config=EvaluationConfig(
            agent="pi-acp",
            model="deepseek/deepseek-flash",
            planned_trials=2,
            trial_number=2,
        ),
        preflight=False,
    )
    task_dirs = job._get_task_dirs()
    digests = {d.name: _task_digest_for(d, job._config) for d in task_dirs}
    _control_job(
        tmp_path / "jobs" / "oracle",
        "oracle",
        {"task-0": 1.0, "task-1": 1.0, "task-2": 1.0},
        digests,
    )
    card = build_run_card(job, task_dirs)
    assert card["controls"]["oracle"]["as_expected"] == 3
    assert card["trials"]["total"] == 6


def test_a_control_run_is_never_held_to_the_reference(tmp_path):
    tasks = _tasks(tmp_path)
    (tasks / "run-reference.toml").write_text(DRUG_TARGET + "strict = true\n")
    job = _evaluation(tmp_path, tasks, agent="nop", model=None)
    card = build_run_card(job, job._get_task_dirs())
    assert card["reference"]["compared"] is False
    assert card["reference"]["deviations"] == []
    check_card(card)
    assert "a control run is not compared" in card_lines(card, tmp_path / "j")[-1]


def test_a_trial_job_finds_controls_beside_the_runs_jobs_folder(tmp_path):
    """`bench run ... --jobs-dir jobs/pi --trials 2` puts trial jobs in
    jobs/pi/<alias>/trial-NN; the controls sit in jobs/oracle and jobs/nop."""
    tasks = _tasks(tmp_path)
    job = Evaluation(
        tasks_dir=str(tasks),
        jobs_dir=str(tmp_path / "jobs" / "pi" / "pi-acp" / "trial-01"),
        config=EvaluationConfig(
            agent="pi-acp",
            model="deepseek/deepseek-flash",
            planned_trials=2,
            trial_number=1,
        ),
        preflight=False,
    )
    task_dirs = job._get_task_dirs()
    digests = {d.name: _task_digest_for(d, job._config) for d in task_dirs}
    _control_job(
        tmp_path / "jobs" / "oracle", "oracle", dict.fromkeys(digests, 1.0), digests
    )
    _control_job(tmp_path / "jobs" / "nop", "nop", dict.fromkeys(digests, 0.0), digests)
    controls = build_run_card(job, task_dirs)["controls"]
    assert controls["oracle"]["as_expected"] == 3
    assert controls["nop"]["as_expected"] == 3


def test_a_hub_dataset_shows_its_digest_not_a_cut_commit(tmp_path):
    """A Hub dataset's provenance carries its dataset digest where a git commit would be; the card
    printed it cut to 8 characters ("sha256:3")."""
    job = _evaluation(tmp_path, _tasks(tmp_path), agent="oracle", model=None)
    card = build_run_card(job, job._get_task_dirs())
    card["tasks"].update(
        repo="http://127.0.0.1:18871",
        path="",
        dataset="robouse/sample@1",
        commit="sha256:" + "ab" * 32,
    )
    lines = card_lines(card, tmp_path / "j")
    line = next(x for x in lines if x.startswith("  Tasks:"))
    assert line.endswith(
        "from robouse/sample@1 on the Hub at http://127.0.0.1:18871, digest sha256:abababababab"
    )
    # the command to run a missing control names the dataset on that Hub, not the Hub's address alone
    card["controls"] = {}
    controls = next(x for x in card_lines(card, tmp_path / "j") if "Controls" in x)
    assert (
        "nop none found: bench run -d robouse/sample@1 --hub http://127.0.0.1:18871 --agent nop"
        in controls
    )
    # read from local job folders, which the Hub's own controls (printed before the run) may not match
    assert controls.startswith("  Controls:  local jobs, not the Hub: ")


def test_the_card_names_the_output_cap_benchflow_sets(tmp_path):
    """BenchFlow sets CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000 for Claude Code unless the
    operator does (agents/env.py); the card said it set no output cap."""
    job = _evaluation(tmp_path, _tasks(tmp_path))
    card = build_run_card(job, job._get_task_dirs())
    assert card["agent"]["output_cap"] == "128000"
    line = next(
        x for x in card_lines(card, tmp_path / "j") if x.startswith("  Sampling:")
    )
    assert (
        "caps Claude Code's output at 128000 tokens (CLAUDE_CODE_MAX_OUTPUT_TOKENS)"
        in line
    )
    assert "output cap for agent trials" not in line
    mine = _evaluation(
        tmp_path,
        _tasks(tmp_path / "b"),
        agent_env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000"},
    )
    card = build_run_card(mine, mine._get_task_dirs())
    assert card["agent"]["output_cap"] == "32000"


def test_an_operator_script_card_claims_no_harness_model_or_cap(tmp_path):
    """A script: agent's card said "harness acp" and named the ACP output cap, which
    its runner never sets."""
    script = tmp_path / "route.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    job = _evaluation(tmp_path, _tasks(tmp_path), agent=f"script:{script}", model=None)
    card = build_run_card(job, job._get_task_dirs())
    assert card["agent"]["output_cap"] is None
    lines = card_lines(card, tmp_path / "j")
    agent = next(x for x in lines if x.startswith("  Agent:"))
    assert "an operator script, run as the sandbox user with no harness" in agent
    assert "harness acp" not in agent
    assert "  Model:     none (an operator script runs no model)" in lines
    assert not any(x.startswith("  Sampling:") for x in lines)


def test_a_dry_run_with_trials_plans_every_trial(tmp_path):
    """The cold-reader test (an agent with only the docs and the CLI) found the
    dry run planning 1 trial per task for `--trials 8`."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    tasks = _tasks(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(tasks),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--trials",
            "8",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(
        result.output[result.output.index("{") : result.output.rindex("}") + 1]
    )
    assert plan["rollouts"] == 24
    assert plan["run_card"]["trials"]["per_task"] == 8
    assert plan["run_card"]["trials"]["total"] == 24


def test_a_dry_run_prints_the_card_a_person_reads(tmp_path):
    """A maintainer's 2026-11-04 case: 60 tasks, 8 tries where the paper used
    one, 8 sessions, Qwen3.8-27B; the dry run is where to see it, before the
    run spends anything."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    tasks = _tasks(tmp_path, n=6)
    (tasks / "run-reference.toml").write_text(DRUG_TARGET)
    argv = ["run", str(tasks), "--agent", "claude-agent-acp"]
    argv += ["--model", "hosted_vllm/Qwen/Qwen3.8-27B", "--sandbox", "docker"]
    argv += ["--trials", "8", "--concurrency", "8", "--dry-run"]
    argv += ["--jobs-dir", str(tmp_path / "jobs")]
    result = CliRunner().invoke(app, argv, env={"COLUMNS": "400"})
    assert result.exit_code == 0, result.output
    card = result.stderr
    assert "Run card (a dry run: nothing ran, and no job folder was made):" in card
    assert "  Trials:    48 in all, 8 per task" in card
    assert "  Vendor:    Qwen recommends for Qwen3.8-27B: thinking on" in card
    assert "  differs: trials 8 (reference 1)" in card
    assert "  check: thinking true in the reference" in card
    # The plan on stdout stays JSON.
    assert json.loads(result.stdout)["run_card"]["trials"]["total"] == 48


def test_a_dry_run_the_reference_would_refuse_exits_1(tmp_path):
    """Round 4 of the release review: the dry run printed the refusal and
    exited 0, so `bench run --dry-run && bench run` went on to the refusal."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    tasks = _tasks(tmp_path)
    (tasks / "run-reference.toml").write_text(DRUG_TARGET + "strict = true\n")
    argv = ["run", str(tasks), "--agent", "claude-agent-acp", "--sandbox", "docker"]
    argv += ["--jobs-dir", str(tmp_path / "jobs"), "--dry-run"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1, result.output
    plan = json.loads(
        result.output[result.output.index("{") : result.output.rindex("}") + 1]
    )
    assert "differs from the setup" in plan["run_card_refusal"]
    assert "--accept-deviations" in result.output
    accepted = CliRunner().invoke(app, [*argv, "--accept-deviations", "a test"])
    assert accepted.exit_code == 0, accepted.output


@pytest.mark.asyncio
async def test_a_sharded_run_is_refused_once_before_its_workers(tmp_path, monkeypatch):
    """A strict reference refuses a sharded run in the parent, before any
    worker starts; with a reason it gets to the workers."""
    from benchflow import eval_sharding

    tasks = _tasks(tmp_path)
    (tasks / "run-reference.toml").write_text(DRUG_TARGET + "strict = true\n")

    async def no_workers(*_args, **_kwargs):
        raise AssertionError("a worker started")

    monkeypatch.setattr(eval_sharding, "_run_worker_once", no_workers)
    config = EvaluationConfig(
        agent="claude-agent-acp", model="claude-haiku-4-5-20251001"
    )
    with pytest.raises(RunCardError, match="differs from the setup"):
        await eval_sharding.run_sharded_evaluation(
            tasks_dir=tasks,
            jobs_dir=tmp_path / "jobs",
            config=config,
            worker_concurrency=2,
            worker_retries=0,
            worker_start_stagger_sec=0,
        )


def test_a_multi_step_task_counts_each_step_s_limit_and_check(tmp_path):
    """A step without its own limit has the task's (task.steps.step_task); the
    first launch run printed 'no upper bound' for such a task."""
    root = tmp_path / "tasks"
    task = root / "ms"
    for name in ("write", "append"):
        (task / "steps" / name / "tests").mkdir(parents=True)
        (task / "steps" / name / "instruction.md").write_text(f"Do {name}.\n")
        (task / "steps" / name / "tests" / "test.sh").write_text("#!/bin/sh\n")
    (task / "task.toml").write_text(
        'version = "1.0"\nmulti_step_reward_strategy = "final"\n'
        "[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 60\n"
        '[environment]\ndocker_image = "python:3.12-slim"\n'
        '[[steps]]\nname = "write"\n[[steps]]\nname = "append"\n'
        "[steps.agent]\ntimeout_sec = 120\n"
    )
    job = _evaluation(tmp_path, root)
    card = build_run_card(job, job._get_task_dirs())
    assert card["tasks"]["multi_step"] == 1
    # 300 s for the step that sets none, 120 s for the one that does.
    assert card["limits"]["agent_sec"] == {"min": 420.0, "max": 420.0}
    # Agent time plus a soft check per step and the final check.
    assert card["estimate"]["longest_trial_sec"] == 420.0 + 3 * 60.0
    assert card["estimate"]["max_wall_sec"] == 600.0
    assert card["tasks"]["multi_step_mean"] == 0
    # Under mean the earlier steps score from soft checks: the card says so.
    toml = task / "task.toml"
    toml.write_text(toml.read_text().replace('"final"', '"mean"'))
    card = build_run_card(job, job._get_task_dirs())
    assert card["tasks"]["multi_step_mean"] == 1
    limits = next(
        line for line in card_lines(card, tmp_path / "j") if "Limits:" in line
    )
    assert "per task (a multi-step task's is the sum of its steps')" in limits
    line = card_lines(card, tmp_path / "j")[1]
    assert "each step's min_reward gate is a soft check" in line
    assert "1 scored by mean, from those soft checks" in line
    # The final strategy is exposed too on a /root workspace.
    assert "a workspace other than /root" in line


def test_a_run_config_override_reaches_the_card(tmp_path):
    """The card reads each task as the trial does, after the run config's
    config_override (a timeout or a network policy it changes)."""
    job = _evaluation(
        tmp_path,
        _tasks(tmp_path),
        config_override={
            "agent": {"timeout_sec": 900},
            "sandbox": {"network_mode": "no-network"},
            "timeout_multipliers": {"verifier": 2.0},
        },
    )
    card = build_run_card(job, job._get_task_dirs())
    assert card["limits"]["agent_sec"] == {"min": 900.0, "max": 900.0}
    assert card["limits"]["verifier_sec"] == {"min": 240.0, "max": 240.0}
    assert card["network"] == {"no-network": 3}


def test_controls_matched_by_name_say_their_content_was_not_checked(
    tmp_path, monkeypatch
):
    import benchflow.run_card as run_card

    monkeypatch.setattr(run_card, "_MAX_DIGEST_TASKS", 1)
    tasks = _tasks(tmp_path)
    job = _evaluation(tmp_path, tasks)
    task_dirs = job._get_task_dirs()
    _control_job(
        tmp_path / "jobs" / "oracle",
        "oracle",
        dict.fromkeys((d.name for d in task_dirs), 1.0),
        dict.fromkeys((d.name for d in task_dirs), "sha256:any"),
    )
    card = build_run_card(job, task_dirs)
    assert card["controls"]["oracle"]["matched_by"] == "name"
    line = next(line for line in card_lines(card, tmp_path / "j") if "Controls" in line)
    assert "matched by name: content not checked above 1 tasks" in line


def test_the_vendor_line_says_when_the_reference_asks_for_other_sampling(tmp_path):
    tasks = _tasks(tmp_path)
    (tasks / "run-reference.toml").write_text(DRUG_TARGET)
    job = _evaluation(tmp_path, tasks, model="hosted_vllm/Qwen/Qwen3.8-27B")
    card = build_run_card(job, job._get_task_dirs())
    vendor = next(
        line
        for line in card_lines(card, tmp_path / "j")
        if line.startswith("  Vendor:")
    )
    assert vendor.endswith(
        "(the reference asks for temperature 0.6: follow it to reproduce the "
        "reference's numbers, the vendor's to measure the model as its maker "
        "recommends)"
    )


def test_an_accepted_note_is_kept_with_no_reference(tmp_path):
    job = _evaluation(tmp_path, _tasks(tmp_path), accept_deviations="a smoke run")
    card = build_run_card(job, job._get_task_dirs())
    assert card["reference"] is None and card["accepted_deviations"] == "a smoke run"
    assert "  Note:      a smoke run" in card_lines(card, tmp_path / "j")


def test_a_dry_run_of_a_repository_says_its_card_comes_later(tmp_path):
    """A repository or dataset target's dry run printed
    "run_card": null and exited 0, while --help promised a refusal's exit 1."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    argv = ["run", "benchflow-ai/example-tasks", "--agent", "oracle", "--sandbox"]
    argv += ["docker", "--jobs-dir", str(tmp_path / "jobs"), "--dry-run"]
    result = CliRunner().invoke(app, argv, env={"COLUMNS": "300"})
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["run_card"] is None
    assert result.stderr.startswith("Run card: none yet. A repository or dataset")


def test_the_card_names_the_retry_policy_and_a_self_routed_model(tmp_path, monkeypatch):
    """0.10 can run an attempt again after the agent acted, and no card said how many
    times; an agent that calls its own model (model_route = "agent") showed "none (the agent's default)"."""
    import benchflow.agents.registry as registry
    from benchflow.evaluation import RetryConfig

    job = _evaluation(tmp_path, _tasks(tmp_path))
    card = build_run_card(job, job._get_task_dirs())
    assert card["trials"]["retries"] == job._config.retry.max_retries
    line = next(
        x for x in card_lines(card, tmp_path / "j") if x.startswith("  Retries:")
    )
    assert f"up to {job._config.retry.max_retries} per trial" in line
    assert "--retry-attempts 0 for none" in line
    none = _evaluation(
        tmp_path, _tasks(tmp_path / "b"), retry=RetryConfig(max_retries=0)
    )
    lines = card_lines(build_run_card(none, none._get_task_dirs()), tmp_path / "j")
    assert "  Retries:   none (--retry-attempts 0)" in lines

    monkeypatch.setattr(registry, "agent_routes_own_model", lambda agent: True)
    own = _evaluation(tmp_path, _tasks(tmp_path / "c"), model=None)
    lines = card_lines(build_run_card(own, own._get_task_dirs()), tmp_path / "j")
    model = next(x for x in lines if x.startswith("  Model:"))
    assert model.startswith(
        "  Model:     none set (the agent picks its own, if it calls one)"
    )
