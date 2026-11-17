"""bench run, a review resume and bench review take a tasks-root entry by one rule.

FP patch 32: ``bench run`` ran a task linked into its tasks root,
while a review resume of the same run refused it as outside --tasks-root,
and ``bench review --tasks-root`` skipped it with its own guard (#942). All
three now call ``tasks_root_entry_refusal``, so the tasks one runs are the
tasks the others review.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from benchflow._utils.task_authoring import task_digest

_TASK_MD = """---
schema_version: "1.3"
agent:
  timeout_sec: 60
---

## prompt

Do nothing.
"""


def _task(folder: Path) -> Path:
    (folder / "environment").mkdir(parents=True)
    (folder / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (folder / "verifier").mkdir()
    (folder / "verifier" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (folder / "oracle").mkdir()
    (folder / "oracle" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    (folder / "task.md").write_text(_TASK_MD)
    return folder


def _tasks_root(tmp_path: Path) -> Path:
    """One root holding every kind of entry, each a runnable task once reached."""
    store = tmp_path / "store"
    root = tmp_path / "tasks"
    _task(root / "plain")
    (root / "linked").symlink_to(_task(store / "linked"), target_is_directory=True)
    (root / "renamed").symlink_to(
        _task(store / "another-name"), target_is_directory=True
    )
    (tmp_path / "middle").mkdir()
    (tmp_path / "middle" / "chained").symlink_to(
        _task(store / "chained"), target_is_directory=True
    )
    (root / "chained").symlink_to(
        tmp_path / "middle" / "chained", target_is_directory=True
    )
    (store / "a-file").write_text("")
    (root / "a-file").symlink_to(store / "a-file")
    for folder in (root / "uploads-a-link", store / "linked-uploads-a-link"):
        shutil.rmtree(_task(folder) / "environment")
        (folder / "environment").symlink_to(
            root / "plain" / "environment", target_is_directory=True
        )
    (root / "linked-uploads-a-link").symlink_to(
        store / "linked-uploads-a-link", target_is_directory=True
    )
    return root


# What bench run skips of _tasks_root, with the reason it records.
SKIPPED = {
    "chained": "link to a link",
    "linked-uploads-a-link": "link inside the task",
    "renamed": "link to a different name",
    "uploads-a-link": "link inside the task",
}


def test_bench_run_and_a_review_resume_take_the_same_entries(tmp_path):
    """And ``bench review --tasks-root`` (``_source_task_dir``) takes the
    same set: #942's guard took a link to any folder inside the root and no
    link out of it, so it took uploads-a-link and refused linked."""
    import benchflow as bf
    from benchflow.review.resume import ReviewResumeError, _trusted_task
    from benchflow.review.runner import _source_task_dir

    root = _tasks_root(tmp_path)
    evaluation = bf.Evaluation(
        tasks_dir=root,
        jobs_dir=tmp_path / "jobs",
        config=bf.EvaluationConfig(agent="oracle", environment="daytona"),
        preflight=False,
    )
    run = {path.name for path in evaluation._get_task_dirs()}
    reviewed, bench_reviewed = set(), set()
    for entry in sorted(root.iterdir()):
        real = entry.resolve()
        digest = task_digest(real) if real.is_dir() else "sha256:0"
        try:
            _trusted_task(
                tmp_path / "rollout",
                {"task_name": entry.name, "task_digest": digest},
                root,
            )
        except ReviewResumeError:
            pass
        else:
            reviewed.add(entry.name)
        rollout = tmp_path / "rollouts" / entry.name
        rollout.mkdir(parents=True)
        (rollout / "config.json").write_text(
            json.dumps({"task_path": f"/elsewhere/{entry.name}", "task_digest": digest})
        )
        if _source_task_dir(rollout, tasks_root=root)[0] is not None:
            bench_reviewed.add(entry.name)

    assert run == reviewed == bench_reviewed == {"plain", "linked"}
    skips = evaluation._tasks_root_skips["skipped"]
    assert {s["name"]: s["reason"] for s in skips} == SKIPPED


async def _run(
    tmp_path: Path, root: Path, caplog, **config
) -> tuple[dict, dict, list[str]]:
    """Run *root*, every trial passing: summary.json, run-card.json and the
    log lines in order, a line ``trial <name>`` as each trial starts."""
    import benchflow as bf
    from benchflow.models import RolloutResult

    evaluation = bf.Evaluation(
        tasks_dir=root,
        jobs_dir=tmp_path / "jobs",
        job_name="job",
        config=bf.EvaluationConfig(agent="oracle", **config),
        preflight=False,
    )

    async def run_single(task_dir, cfg):
        logging.getLogger("benchflow.evaluation").info("trial %s", task_dir.name)
        return RolloutResult(task_name=task_dir.name, rewards={"reward": 1.0})

    async def no_sweep():
        return None

    evaluation._run_single_task = run_single
    evaluation._sweep_docker = no_sweep
    with caplog.at_level(logging.INFO):
        await evaluation.run()
    job = tmp_path / "jobs" / "job"
    return (
        json.loads((job / "summary.json").read_text()),
        json.loads((job / "run-card.json").read_text()),
        [record.getMessage() for record in caplog.records],
    )


COUNT = "2 of 6 listed; 4 skipped by the tasks-root rule"


@pytest.mark.asyncio
async def test_summary_json_lists_each_entry_the_rule_skipped(tmp_path, caplog):
    """The one rule skips two links bench run used
    to run (renamed, chained), so a skip must never shrink a run silently.
    summary.json and the run card list each skipped entry with its reason,
    and one warning comes before anything runs."""
    summary, _, _ = await _run(tmp_path, _tasks_root(tmp_path), caplog)
    assert summary["planned"] == 2
    assert summary["tasks_root_skips"]["count"] == COUNT
    skips = summary["tasks_root_skips"]["skipped"]
    assert {s["name"]: s["reason"] for s in skips} == SKIPPED


@pytest.mark.asyncio
async def test_the_run_card_lists_them_and_its_tasks_line_counts_them(tmp_path, caplog):
    _, card, lines = await _run(tmp_path, _tasks_root(tmp_path), caplog)
    assert card["tasks"]["count"] == 2
    assert card["tasks"]["tasks_root_skips"]["count"] == COUNT
    skips = card["tasks"]["tasks_root_skips"]["skipped"]
    assert {s["name"]: s["reason"] for s in skips} == SKIPPED
    assert any(
        line.startswith(
            f"  Tasks:     {COUNT} (chained: link to a link, linked-uploads-a-link: "
            "link inside the task, renamed: link to a different name, "
            "uploads-a-link: link inside the task) from "
        )
        for line in lines
    )


@pytest.mark.asyncio
async def test_a_run_warns_once_before_anything_runs(tmp_path, caplog):
    _, _, lines = await _run(tmp_path, _tasks_root(tmp_path), caplog)
    # The listing is read more than once before the trials start.
    detail = "renamed is a link to the folder another-name"
    warned = [i for i, line in enumerate(lines) if detail in line]
    assert len(warned) == 1, [lines[i] for i in warned]
    assert lines[warned[0]].startswith(
        "The tasks-root rule skips 4 of the 6 tasks listed in "
    )
    assert f"\n  Skipping task 'renamed': {detail}" in lines[warned[0]]
    tasks_line = next(i for i, line in enumerate(lines) if line.startswith("  Tasks:"))
    first_trial = next(i for i, line in enumerate(lines) if line.startswith("trial "))
    assert warned[0] < tasks_line < first_trial


@pytest.mark.asyncio
async def test_a_run_the_rule_skips_nothing_of_shows_nothing_new(tmp_path, caplog):
    root = tmp_path / "tasks"
    _task(root / "plain")
    (root / "linked").symlink_to(
        _task(tmp_path / "store" / "linked"), target_is_directory=True
    )
    summary, card, lines = await _run(tmp_path, root, caplog)
    assert summary["planned"] == 2
    assert "tasks_root_skips" not in summary
    assert "tasks_root_skips" not in card["tasks"]
    assert f"  Tasks:     2 from {card['tasks']['root']}" in lines
    assert not any("tasks-root rule" in line for line in lines)


def _reasons(record: dict) -> dict[str, str]:
    return {s["name"]: s["reason"] for s in record["skipped"]}


async def _run_sharded(tmp_path: Path, root: Path, monkeypatch, caplog) -> Path:
    """Run *root* as ``bench run --worker-concurrency 1 -j 2`` would, each
    worker in this process (its own Evaluation over its shard's names, as
    ``benchflow.eval_worker`` runs it) and every trial passing; the jobs
    folder."""
    import benchflow as bf
    from benchflow import eval_sharding, eval_worker
    from benchflow.evaluation import Evaluation
    from benchflow.models import RolloutResult

    async def run_single(self, task_dir, cfg):
        return RolloutResult(task_name=task_dir.name, rewards={"reward": 1.0})

    async def in_process(payload_path, log_path, *, max_retries):
        return await eval_worker.run_worker(payload_path)

    monkeypatch.setenv("BENCHFLOW_DAYTONA_AUTO_REAP", "off")
    monkeypatch.setattr(Evaluation, "_run_single_task", run_single)
    monkeypatch.setattr(eval_sharding, "_run_worker_with_retries", in_process)
    jobs = tmp_path / "jobs"
    config = bf.EvaluationConfig(agent="oracle", environment="daytona", concurrency=2)
    with caplog.at_level(logging.INFO):
        await eval_sharding.run_sharded_evaluation(
            tasks_dir=root,
            jobs_dir=jobs,
            config=config,
            worker_concurrency=1,
            worker_retries=0,
            worker_start_stagger_sec=0,
        )
    return jobs


@pytest.mark.asyncio
async def test_a_sharded_run_records_what_the_rule_skipped(
    tmp_path, monkeypatch, caplog
):
    """REV6B: a sharded run (``--worker-concurrency``) gives each
    worker its shard's names, so only the parent's listing saw what the rule
    skipped. Its summary.json and the shards' cards and summaries recorded
    nothing, and the job went from 6 tasks to 2 without a word in its folder.
    The aggregate summary.json, and each shard's run card and summary.json,
    now carry the job's record, as an unsharded run's do."""
    jobs = await _run_sharded(tmp_path, _tasks_root(tmp_path), monkeypatch, caplog)
    for path in (jobs / "summary.json", jobs / "worker-sharded" / "summary.json"):
        summary = json.loads(path.read_text())
        assert summary["total"] == 2
        assert "tasks_root_skips" in summary, (path, sorted(summary))
        assert summary["tasks_root_skips"]["count"] == COUNT
        assert _reasons(summary["tasks_root_skips"]) == SKIPPED
    cards = sorted((jobs / "worker-shards").glob("shard-*/jobs/*/run-card.json"))
    assert len(cards) == 2
    for path in cards:
        card = json.loads(path.read_text())
        summary = json.loads((path.parent / "summary.json").read_text())
        assert card["tasks"]["count"] == summary["planned"] == 1
        for record in (card["tasks"]["tasks_root_skips"], summary["tasks_root_skips"]):
            assert record["count"] == COUNT
            assert _reasons(record) == SKIPPED
    lines = [record.getMessage() for record in caplog.records]
    # Each shard's Tasks line gives its own count, then the job's.
    shard_lines = [
        line
        for line in lines
        if line.startswith(f"  Tasks:     1 from {COUNT} (chained: link to a link, ")
    ]
    assert len(shard_lines) == 2, [line for line in lines if "Tasks:" in line]
    # One warning, from the parent's listing; none from a worker's.
    assert sum(line.startswith("The tasks-root rule skips") for line in lines) == 1


@pytest.mark.asyncio
async def test_a_sharded_run_the_rule_skips_nothing_of_shows_nothing_new(
    tmp_path, monkeypatch, caplog
):
    root = tmp_path / "tasks"
    _task(root / "plain")
    (root / "linked").symlink_to(
        _task(tmp_path / "store" / "linked"), target_is_directory=True
    )
    jobs = await _run_sharded(tmp_path, root, monkeypatch, caplog)
    for path in (
        jobs / "summary.json",
        jobs / "worker-sharded" / "summary.json",
        *(jobs / "worker-shards").glob("shard-*/jobs/*/summary.json"),
        *(jobs / "worker-shards").glob("shard-*/worker_payload.json"),
    ):
        assert "tasks_root_skips" not in json.loads(path.read_text()), path
    for path in (jobs / "worker-shards").glob("shard-*/jobs/*/run-card.json"):
        assert "tasks_root_skips" not in json.loads(path.read_text())["tasks"]
    assert not any("tasks-root rule" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("sample_seed", [None, 7])
def test_n_tasks_samples_among_the_entries_the_rule_takes(tmp_path, sample_seed):
    """REV6B: ``--n-tasks`` sampled the root's entries before the
    rule, so a link the rule skips took a task's place: ``--n-tasks 2`` on a
    root with two runnable tasks ran one. The rule comes first now, and the
    sample is drawn from the entries it takes; the record still lists every
    entry it skipped."""
    import benchflow as bf

    root = _tasks_root(tmp_path)

    def listing(n_tasks):
        evaluation = bf.Evaluation(
            tasks_dir=root,
            jobs_dir=tmp_path / "jobs",
            config=bf.EvaluationConfig(
                agent="oracle",
                environment="daytona",
                n_tasks=n_tasks,
                sample_seed=sample_seed,
            ),
            preflight=False,
        )
        return [d.name for d in evaluation._get_task_dirs()], evaluation

    names, evaluation = listing(2)
    assert names == ["linked", "plain"]
    assert evaluation._tasks_root_skips["count"] == COUNT
    assert _reasons(evaluation._tasks_root_skips) == SKIPPED
    names, evaluation = listing(1)
    assert len(names) == 1 and names[0] in ("linked", "plain")
    assert _reasons(evaluation._tasks_root_skips) == SKIPPED
    # REV6C nit: the entry the sample left out is listed too, so the count
    # is of the root's task entries ("1 of 5 listed" left it out).
    assert evaluation._tasks_root_skips["count"] == (
        "1 of 6 listed; 4 skipped by the tasks-root rule"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sharded", [False, True], ids=["plain", "sharded"])
async def test_a_root_the_rule_skips_entirely_is_refused_naming_the_rule(
    tmp_path, monkeypatch, caplog, sharded
):
    """REV6B: a root whose every entry the rule skips is refused as
    an empty selection, but the error said its entries "have task files but
    are not runnable tasks" and sent the user to ``bench check``, which
    passes each of them, and the warning before it promised a run card and a
    summary.json the refused run never writes. The error now names the rule,
    its hint the fix, and the warning promises no record."""
    import benchflow as bf
    from benchflow import eval_sharding
    from benchflow.evaluation import EmptyTaskSelectionError

    monkeypatch.setenv("BENCHFLOW_DAYTONA_AUTO_REAP", "off")
    root = tmp_path / "tasks"
    root.mkdir()
    for name in ("x", "y"):
        (root / name).symlink_to(
            _task(tmp_path / "store" / f"other-{name}"), target_is_directory=True
        )
    jobs = tmp_path / "jobs"
    config = bf.EvaluationConfig(agent="oracle", environment="daytona")
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(EmptyTaskSelectionError) as refused,
    ):
        if sharded:
            await eval_sharding.run_sharded_evaluation(
                tasks_dir=root,
                jobs_dir=jobs,
                config=config,
                worker_concurrency=1,
                worker_retries=0,
                worker_start_stagger_sec=0,
            )
        else:
            await bf.Evaluation(
                tasks_dir=root, jobs_dir=jobs, config=config, preflight=False
            ).run()
    assert str(refused.value) == (
        f"No tasks selected in {root}: the tasks-root rule skips all 2 entries "
        "listed (first: x is a link to the folder other-x; a linked task keeps "
        "its folder's name). Refusing to publish an empty 0/0 summary."
    )
    assert refused.value.hint == (
        "link a task under its folder's own name (tasks/bar -> ../store/bar) or "
        "put the folder itself in the tasks root; the warning above gives each "
        "entry's reason"
    )
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("The tasks-root rule skips")
    ]
    assert len(warnings) == 1
    assert warnings[0].startswith(
        f"The tasks-root rule skips 2 of the 2 tasks listed in {root}, so no task "
        "is left to run:\n  Skipping task 'x': "
    )
    assert "run-card.json" not in warnings[0] and "summary.json" not in warnings[0]
    assert not [*jobs.rglob("summary.json"), *jobs.rglob("run-card.json")]


def test_the_hint_for_a_link_inside_the_task_is_to_copy_its_files(tmp_path):
    import benchflow as bf

    root = tmp_path / "tasks"
    _task(root / "plain")
    shutil.rmtree(_task(root / "uploads-a-link") / "environment")
    (root / "uploads-a-link" / "environment").symlink_to(
        root / "plain" / "environment", target_is_directory=True
    )
    evaluation = bf.Evaluation(
        tasks_dir=root,
        jobs_dir=tmp_path / "jobs",
        config=bf.EvaluationConfig(
            agent="oracle", environment="daytona", include_tasks={"uploads-a-link"}
        ),
        preflight=False,
    )
    assert evaluation._get_task_dirs() == []
    assert evaluation._empty_selection_message().startswith(
        f"No tasks selected in {root}: the tasks-root rule skips the 1 entry listed "
        "(first: "
    )
    assert evaluation._empty_selection_hint() == (
        "copy a task's linked files into it; the warning above gives each "
        "entry's reason"
    )


class _SeededFormat:
    """A task format that writes one package per seed (benchflow.task.formats)."""

    name = "tasks-root-seeded"

    def detect(self, task_dir: Path) -> bool:
        return (task_dir / "seeded.fmt").is_file()

    def _package(self, out: Path) -> Path:
        return out if (out / "task.md").is_file() else _task(out)

    def materialize(self, task_dir: Path, out_root: Path) -> Path:
        return self._package(out_root / task_dir.name)

    def materialize_variant(self, task_dir: Path, out_root: Path, *, seed: int) -> Path:
        return self._package(out_root / f"{task_dir.name}--seed-{seed}")


@pytest.mark.asyncio
async def test_with_seeds_the_record_counts_entries_not_seeded_tasks(
    tmp_path, monkeypatch, caplog
):
    """REV6B nit: with ``--seeds`` each entry gives one task per
    seed, and the record set the seeded tasks it ran against the entries it
    skipped: entries a and b at three seeds and one skipped link read "6 of 7
    listed". It counts entries now, "2 of 3 listed", and the card's Tasks line
    gives the 6 tasks it runs beside them."""
    from benchflow.task import formats

    monkeypatch.setattr(formats, "_registered", [])
    monkeypatch.setattr(formats, "_entry_point_formats", [])
    monkeypatch.setenv(formats.CACHE_ENV, str(tmp_path / "cache"))
    formats.register_task_format(_SeededFormat())
    root = tmp_path / "tasks"
    for folder in (root / "a", root / "b", tmp_path / "store" / "zz"):
        folder.mkdir(parents=True)
        (folder / "seeded.fmt").write_text("")
    (root / "c").symlink_to(tmp_path / "store" / "zz", target_is_directory=True)

    summary, card, lines = await _run(tmp_path, root, caplog, seeds=[1, 2, 3])
    count = "2 of 3 listed; 1 skipped by the tasks-root rule"
    assert summary["planned"] == card["tasks"]["count"] == 6
    for record in (summary["tasks_root_skips"], card["tasks"]["tasks_root_skips"]):
        assert record["count"] == count
        assert _reasons(record) == {"c": "link to a different name"}
    assert f"  Tasks:     6 from {count} (c: link to a different name) from {root}" in (
        lines
    )
    assert any(
        line.startswith(f"The tasks-root rule skips 1 of the 3 tasks listed in {root};")
        for line in lines
    )


def test_a_shard_left_with_no_task_does_not_blame_the_rule(tmp_path):
    """A shard's record is its job's (which took tasks): when its own listing
    finds none of its tasks, its empty selection is the filters', not the
    rule's."""
    import benchflow as bf

    root = _tasks_root(tmp_path)
    job = bf.Evaluation(
        tasks_dir=root,
        jobs_dir=tmp_path / "jobs",
        config=bf.EvaluationConfig(agent="oracle", environment="daytona"),
        preflight=False,
    )
    job._get_task_dirs()
    shard = bf.Evaluation(
        tasks_dir=root,
        jobs_dir=tmp_path / "shard",
        config=bf.EvaluationConfig(
            agent="oracle", environment="daytona", include_tasks={"gone"}
        ),
        preflight=False,
    )
    shard._job_tasks_root_skips = job._tasks_root_skips
    assert shard._get_task_dirs() == []
    assert shard._tasks_root_skips["count"] == COUNT
    assert "tasks-root rule" not in shard._empty_selection_message()
    assert shard._empty_selection_hint() is None


# --- follow-up: bench review --tasks-root ------------------------------


def _rubric(task: Path) -> Path:
    from benchflow.review.config import DEFAULT_RUBRIC_PATH

    (task / "verifier" / "rubric.json").write_bytes(DEFAULT_RUBRIC_PATH.read_bytes())
    return task


def _rollout_of(jobs: Path, task: Path) -> Path:
    """A finished rollout of *task*, its config.json naming the task by a
    path elsewhere and both records carrying the task's digest."""
    rollout = jobs / "job" / f"{task.name}__0001"
    (rollout / "trajectory").mkdir(parents=True)
    digest = task_digest(task.resolve())
    (rollout / "config.json").write_text(
        json.dumps({"task_path": f"/elsewhere/{task.name}", "task_digest": digest})
    )
    (rollout / "result.json").write_text(
        json.dumps(
            {"task_name": task.name, "rewards": {"reward": 1.0}, "task_digest": digest}
        )
    )
    (rollout / "trajectory" / "trajectory.json").write_text("[]")
    return rollout


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["physics", "plain"])
async def test_bench_review_uses_a_linked_tasks_rubric_and_files(
    tmp_path, monkeypatch, name
):
    """a review (ruling decision 2): on the reported layout,
    ``tasks38/physics -> batch/mimo/tasks/physics``, ``bench review
    --tasks-root tasks38`` skipped the linked task by #942's guard ("does
    not resolve to a directory inside --tasks-root"), so it was reviewed with
    the built-in rubric and without the task's files, while ``bench run``
    ran it and a review resume took it. It now uses the task's own rubric and
    files, as for the real folder beside it; the digest still pins them."""
    import benchflow
    from benchflow.review.runner import run_reviews
    from benchflow.rollout import RolloutConfig

    store = tmp_path / "batch" / "mimo" / "tasks"
    root = tmp_path / "tasks38"
    (root / "physics").parent.mkdir(parents=True)
    (root / "physics").symlink_to(
        _rubric(_task(store / "physics")), target_is_directory=True
    )
    _rubric(_task(root / "plain"))
    rollout = _rollout_of(tmp_path / "jobs", root / name)
    uploads: list[dict] = []

    async def fake_run(config: RolloutConfig):
        uploads.append(dict(config.uploads or {}))
        leaf = Path(config.jobs_dir) / "job" / "w__0000"
        (leaf / "verifier").mkdir(parents=True)
        (leaf / "config.json").write_text("{}")
        (leaf / "result.json").write_text(json.dumps({"rewards": {"reward": 0.0}}))

        class _R:
            error = None

        return _R()

    monkeypatch.setattr(benchflow, "run", fake_run)
    report, _ = await run_reviews(
        rollout, agent="gemini", out_dir=tmp_path / "out", tasks_root=root
    )
    trial = report.trials[0]
    assert not any("task evidence skipped" in note for note in trial.notes)
    assert trial.rubric_path == str(
        (root / name).resolve() / "verifier" / "rubric.json"
    )
    assert any("/evidence/task" in u.values() for u in uploads), uploads


@pytest.mark.parametrize(
    "shape", ["renamed", "renamed-inside", "chained", "outside", "nested"]
)
def test_bench_review_refuses_an_entry_bench_run_skips_with_the_rules_text(
    tmp_path, shape
):
    """``bench review --tasks-root`` refuses the entries the tasks-root rule
    refuses, and its note gives the rule's reason: a link to a folder of
    another name, in the root (#942 took it) or out of it; a link to a link;
    a link out of the root to another name, as a link to ``~/.ssh`` would
    be; and a link one level down, not directly in the root."""
    from benchflow._utils.task_authoring import tasks_root_entry_refusal
    from benchflow.review.runner import _source_task_dir

    root = tmp_path / "tasks"
    _task(root / "plain")
    name = shape
    if shape == "renamed":
        (root / name).symlink_to(
            _task(tmp_path / "store" / "another-name"), target_is_directory=True
        )
    elif shape == "renamed-inside":
        (root / name).symlink_to(root / "plain", target_is_directory=True)
    elif shape == "chained":
        (tmp_path / "middle").mkdir()
        (tmp_path / "middle" / name).symlink_to(
            _task(tmp_path / "store" / name), target_is_directory=True
        )
        (root / name).symlink_to(tmp_path / "middle" / name, target_is_directory=True)
    elif shape == "outside":
        secret = tmp_path / "home" / ".ssh"
        secret.mkdir(parents=True)
        (secret / "id_ed25519").write_text("key")
        (root / name).symlink_to(secret, target_is_directory=True)
    else:
        (root / "group").mkdir()
        (root / "group" / "physics").symlink_to(
            _task(tmp_path / "store" / "physics"), target_is_directory=True
        )
        name = "physics"
    target = (root / ("group" if shape == "nested" else "") / name).resolve()
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    (rollout / "config.json").write_text(
        json.dumps(
            {"task_path": f"/elsewhere/{name}", "task_digest": task_digest(target)}
        )
    )
    folder, skip = _source_task_dir(rollout, tasks_root=root)
    refusal = tasks_root_entry_refusal(root.resolve(), name)
    assert folder is None and refusal is not None
    assert skip == f"task evidence skipped: {refusal}"


def test_bench_review_takes_the_root_when_it_is_the_task(tmp_path):
    """A single-task run's ``--tasks-dir`` is the task folder, and a review
    resume takes ``--tasks-root`` itself when it has the task's name. #942's
    guard refused it (the root is not inside itself); ``bench review`` now
    takes it as the resume does, and the digest still pins its files."""
    from benchflow.review.runner import _source_task_dir

    task = _task(tmp_path / "physics")
    rollout = _rollout_of(tmp_path / "jobs", task)
    folder, skip = _source_task_dir(rollout, tasks_root=task)
    assert (folder, skip) == (task.resolve(), None)


# --- follow-up: what the tasks-root record counts ----------------------


def _with_shared_files(tmp_path: Path, root: Path) -> Path:
    """REV6C L1's entries, none of them a task: ``docs/``, a folder;
    ``shared -> ../common``, a folder of shared files; and ``common ->
    ../mid/common``, a link to a link to that folder."""
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "README.md").write_text("notes\n")
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "helpers.py").write_text("")
    (root / "shared").symlink_to("../common", target_is_directory=True)
    (tmp_path / "mid").mkdir()
    (tmp_path / "mid" / "common").symlink_to("../common", target_is_directory=True)
    (root / "common").symlink_to("../mid/common", target_is_directory=True)
    return root


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["default", "n_tasks", "sharded"])
async def test_a_link_to_a_folder_that_holds_no_task_is_not_a_skipped_task(
    tmp_path, monkeypatch, caplog, mode
):
    """REV6C L1: the rule ran on every folder of the root before
    anything checked it held a task, so ``shared -> ../common`` and ``common
    -> ../mid/common`` were counted and named as skipped tasks ("1 of 3
    listed; 2 skipped") on a root with one task. A link to a folder without a
    task file is left out unrecorded, as before the rule, on every path."""
    root = tmp_path / "tasks"
    _task(root / "plain")
    _with_shared_files(tmp_path, root)
    if mode == "sharded":
        jobs = await _run_sharded(tmp_path, root, monkeypatch, caplog)
        files = [
            jobs / "summary.json",
            jobs / "worker-sharded" / "summary.json",
            *(jobs / "worker-shards").glob("shard-*/jobs/*/summary.json"),
        ]
        assert json.loads(files[0].read_text())["total"] == 1
        for path in files:
            assert "tasks_root_skips" not in json.loads(path.read_text()), path
    else:
        config = {"n_tasks": 5} if mode == "n_tasks" else {}
        summary, card, _ = await _run(tmp_path, root, caplog, **config)
        assert summary["planned"] == 1
        assert "tasks_root_skips" not in summary
        assert "tasks_root_skips" not in card["tasks"]
    assert not any("tasks-root rule" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_with_n_tasks_the_count_and_the_warning_list_the_whole_root(
    tmp_path, caplog
):
    """REV6C nit: ``-n 1`` on a root of 4 entries (plain, linked, renamed,
    chained) read "1 of 3 listed" and "skips 2 of the 3 tasks listed", which
    left out the entry the sample dropped. Both count it now, so they match
    the root; the card's selection gives the sample."""
    root = tmp_path / "tasks"
    _task(root / "plain")
    store = tmp_path / "store"
    (root / "linked").symlink_to(_task(store / "linked"), target_is_directory=True)
    (root / "renamed").symlink_to(
        _task(store / "another-name"), target_is_directory=True
    )
    (tmp_path / "middle").mkdir()
    (tmp_path / "middle" / "chained").symlink_to(
        _task(store / "chained"), target_is_directory=True
    )
    (root / "chained").symlink_to(
        tmp_path / "middle" / "chained", target_is_directory=True
    )
    summary, card, lines = await _run(tmp_path, root, caplog, n_tasks=1)
    count = "1 of 4 listed; 2 skipped by the tasks-root rule"
    assert summary["planned"] == 1
    assert summary["tasks_root_skips"]["count"] == count
    assert card["tasks"]["tasks_root_skips"]["count"] == count
    assert summary["tasks_root_skips"]["listed"] == 4
    warned = [line for line in lines if line.startswith("The tasks-root rule skips")]
    assert len(warned) == 1
    assert warned[0].startswith(
        f"The tasks-root rule skips 2 of the 4 tasks listed in {root};"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sharded", [False, True], ids=["plain", "sharded"])
async def test_a_refused_root_names_a_task_folder_beside_the_skips_that_cannot_run(
    tmp_path, monkeypatch, caplog, sharded
):
    """REV6C nit: when the rule skips every link and a task folder beside
    them cannot run (task.md, no environment/), the error named only the
    rule. It names that folder too, with its first issue."""
    import benchflow as bf
    from benchflow import eval_sharding
    from benchflow._utils.task_authoring import runnable_task_issues
    from benchflow.evaluation import EmptyTaskSelectionError

    monkeypatch.setenv("BENCHFLOW_DAYTONA_AUTO_REAP", "off")
    root = tmp_path / "tasks"
    root.mkdir()
    for name in ("x", "y"):
        (root / name).symlink_to(
            _task(tmp_path / "store" / f"other-{name}"), target_is_directory=True
        )
    shutil.rmtree(_task(root / "incomplete") / "environment")
    _with_shared_files(tmp_path, root)
    issue = runnable_task_issues(root / "incomplete")[0]
    jobs = tmp_path / "jobs"
    config = bf.EvaluationConfig(agent="oracle", environment="daytona")
    with pytest.raises(EmptyTaskSelectionError) as refused:
        if sharded:
            await eval_sharding.run_sharded_evaluation(
                tasks_dir=root,
                jobs_dir=jobs,
                config=config,
                worker_concurrency=1,
                worker_retries=0,
                worker_start_stagger_sec=0,
            )
        else:
            await bf.Evaluation(
                tasks_dir=root, jobs_dir=jobs, config=config, preflight=False
            ).run()
    assert str(refused.value) == (
        f"No tasks selected in {root}: the tasks-root rule skips all 2 entries "
        "listed (first: x is a link to the folder other-x; a linked task keeps "
        "its folder's name); 1 other task folder is not runnable (first: "
        f"incomplete: {issue}). Refusing to publish an empty 0/0 summary."
    )


# --- follow-up: the warning prints once per process --------------------


@pytest.mark.parametrize("matrix", ["two-agents", "n-attempts-2"])
def test_a_config_matrix_warns_once_and_every_trial_job_keeps_the_record(
    tmp_path, monkeypatch, caplog, matrix
):
    """REV6C L2: "once" was per Evaluation, and a ``--config``
    matrix lists every cell up front, then builds a new Evaluation for each
    trial job, so two agents printed the warning 4 times and ``n_attempts:
    2`` 3 times, the later ones after trials had run. It prints once per
    process for a root and its skips; every trial job's summary.json and run
    card still carry the record, which lists only task entries."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.evaluation import Evaluation
    from benchflow.models import RolloutResult

    async def run_single(self, task_dir, cfg):
        return RolloutResult(task_name=task_dir.name, rewards={"reward": 1.0})

    async def no_sweep(self):
        return None

    monkeypatch.setattr(Evaluation, "_run_single_task", run_single)
    monkeypatch.setattr(Evaluation, "_sweep_docker", no_sweep)
    monkeypatch.chdir(tmp_path)
    _with_shared_files(tmp_path, _tasks_root(tmp_path))
    agents = [{"name": "oracle"}]
    job: dict = {"datasets": [{"path": "tasks"}]}
    if matrix == "two-agents":
        agents.append({"name": "nop"})
    else:
        job["n_attempts"] = 2
    job["agents"] = agents
    (tmp_path / "config.json").write_text(json.dumps(job))
    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(
            app, ["run", "--config", "config.json"], prog_name="bench"
        )
    assert result.exit_code == 0, result.output
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("The tasks-root rule skips")
    ]
    assert len(warned) == 1, warned
    # Each trial job's summary.json and run card, and its jobs folder's
    # summary.json.
    cards = sorted((tmp_path / "jobs").rglob("run-card.json"))
    summaries = [
        *(card.parent / "summary.json" for card in cards),
        *(card.parent.parent / "summary.json" for card in cards),
    ]
    assert len(cards) == 2, cards
    records = [
        json.loads(card.read_text())["tasks"]["tasks_root_skips"] for card in cards
    ]
    records += [json.loads(path.read_text())["tasks_root_skips"] for path in summaries]
    for record in records:
        assert record["count"] == COUNT
        assert _reasons(record) == SKIPPED
