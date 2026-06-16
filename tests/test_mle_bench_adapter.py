from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from benchflow.task.config import TaskConfig


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_converter():
    return _load_module(
        Path("benchmarks/mle-bench/benchflow.py").resolve(),
        "mle_bench_converter_under_test",
    )


def _load_parity():
    return _load_module(
        Path("benchmarks/mle-bench/parity_test.py").resolve(),
        "mle_bench_parity_under_test",
    )


def _write_fake_source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "mle-bench"
    data = tmp_path / "mle-data"
    comp_id = "Fake-Kaggle"
    comp_dir = source / "mlebench" / "competitions" / comp_id
    comp_dir.mkdir(parents=True)
    (source / "mlebench" / "__init__.py").write_text("")
    (source / "mlebench" / "utils.py").write_text(
        """
import importlib
import logging
import csv
from pathlib import Path

import yaml


def get_logger(name):
    return logging.getLogger(name)


def get_module_dir():
    return Path(__file__).parent.resolve()


def get_repo_dir():
    return get_module_dir().parent


def import_fn(import_string):
    module_name, fn_name = import_string.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def read_csv(path, **kwargs):
    del kwargs
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_answers(path):
    return read_csv(path)
""".lstrip()
    )
    (source / "mlebench" / "grade_helpers.py").write_text(
        """
from dataclasses import dataclass
from datetime import datetime

from mlebench.utils import import_fn


class InvalidSubmissionError(Exception):
    pass


class Grader:
    def __init__(self, name, grade_fn):
        self.name = name
        self.grade_fn = import_fn(grade_fn)

    @staticmethod
    def from_dict(data):
        return Grader(**data)

    def __call__(self, submission, answers):
        return round(float(self.grade_fn(submission, answers)), 5)

    def rank_score(self, score, leaderboard):
        return {
            "gold_threshold": 0.5,
            "silver_threshold": 0.5,
            "bronze_threshold": 0.5,
            "median_threshold": 0.5,
            "gold_medal": bool(score is not None and score >= 0.5),
            "silver_medal": False,
            "bronze_medal": False,
            "above_median": bool(score is not None and score >= 0.5),
        }

    def is_lower_better(self, leaderboard):
        return False


@dataclass(frozen=True)
class CompetitionReport:
    competition_id: str
    score: float | None
    gold_threshold: float
    silver_threshold: float
    bronze_threshold: float
    median_threshold: float
    any_medal: bool
    gold_medal: bool
    silver_medal: bool
    bronze_medal: bool
    above_median: bool
    submission_exists: bool
    valid_submission: bool
    is_lower_better: bool
    created_at: datetime
    submission_path: str

    def to_dict(self):
        return {
            "competition_id": self.competition_id,
            "score": self.score,
            "gold_threshold": self.gold_threshold,
            "silver_threshold": self.silver_threshold,
            "bronze_threshold": self.bronze_threshold,
            "median_threshold": self.median_threshold,
            "any_medal": self.any_medal,
            "gold_medal": self.gold_medal,
            "silver_medal": self.silver_medal,
            "bronze_medal": self.bronze_medal,
            "above_median": self.above_median,
            "submission_exists": self.submission_exists,
            "valid_submission": self.valid_submission,
            "is_lower_better": self.is_lower_better,
            "created_at": self.created_at.isoformat(),
            "submission_path": self.submission_path,
        }
""".lstrip()
    )
    (source / "mlebench" / "registry.py").write_text(
        """
from dataclasses import dataclass
from pathlib import Path

from mlebench.grade_helpers import Grader
from mlebench.utils import get_module_dir, get_repo_dir, load_yaml


@dataclass(frozen=True)
class Competition:
    id: str
    name: str
    description: str
    grader: Grader
    answers: Path
    sample_submission: Path
    leaderboard: Path


class Registry:
    def __init__(self, data_dir=Path(".")):
        self._data_dir = Path(data_dir)

    def set_data_dir(self, new_data_dir):
        return Registry(Path(new_data_dir))

    def get_competitions_dir(self):
        return get_module_dir() / "competitions"

    def get_competition(self, competition_id):
        config_path = self.get_competitions_dir() / competition_id / "config.yaml"
        config = load_yaml(config_path)
        description = (get_repo_dir() / config["description"]).read_text()
        dataset = config["dataset"]
        return Competition(
            id=config["id"],
            name=config["name"],
            description=description,
            grader=Grader.from_dict(config["grader"]),
            answers=self._data_dir / dataset["answers"],
            sample_submission=self._data_dir / dataset["sample_submission"],
            leaderboard=self.get_competitions_dir() / competition_id / "leaderboard.csv",
        )


registry = Registry()
""".lstrip()
    )
    (source / "mlebench" / "grade.py").write_text(
        """
from datetime import datetime
from pathlib import Path

from mlebench.grade_helpers import CompetitionReport
from mlebench.utils import load_answers, read_csv


def grade_csv(path_to_submission, competition):
    submission_path = Path(path_to_submission)
    submission_exists = submission_path.is_file() and submission_path.suffix == ".csv"
    score = None
    if submission_exists:
        score = competition.grader(read_csv(submission_path), load_answers(competition.answers))
    rank_info = competition.grader.rank_score(score, read_csv(competition.leaderboard))
    return CompetitionReport(
        competition_id=competition.id,
        score=score,
        gold_threshold=rank_info["gold_threshold"],
        silver_threshold=rank_info["silver_threshold"],
        bronze_threshold=rank_info["bronze_threshold"],
        median_threshold=rank_info["median_threshold"],
        any_medal=rank_info["gold_medal"] or rank_info["silver_medal"] or rank_info["bronze_medal"],
        gold_medal=rank_info["gold_medal"],
        silver_medal=rank_info["silver_medal"],
        bronze_medal=rank_info["bronze_medal"],
        above_median=rank_info["above_median"],
        submission_exists=submission_exists,
        valid_submission=score is not None,
        is_lower_better=False,
        created_at=datetime.now(),
        submission_path=str(submission_path),
    )
""".lstrip()
    )
    for filename in ("data.py", "metrics.py"):
        (source / "mlebench" / filename).write_text("# fake mlebench core\n")
    (source / "mlebench" / "competitions" / "__init__.py").write_text("")
    (source / "mlebench" / "competitions" / "utils.py").write_text("")
    (comp_dir / "description.md").write_text("Predict the fake label.\n")
    (comp_dir / "grade.py").write_text(
        "def grade(submission, answers):\n    return 1.0\n"
    )
    (comp_dir / "leaderboard.csv").write_text("team,score\nwinner,1.0\n")
    (comp_dir / "config.yaml").write_text(
        """
id: Fake-Kaggle
name: Fake Kaggle
competition_type: simple
description: mlebench/competitions/Fake-Kaggle/description.md
dataset:
  answers: Fake-Kaggle/prepared/private/test.csv
  sample_submission: Fake-Kaggle/prepared/public/sampleSubmission.csv
grader:
  name: accuracy
  grade_fn: mlebench.competitions.Fake-Kaggle.grade:grade
preparer: mlebench.competitions.Fake-Kaggle.prepare:prepare
""".lstrip()
    )

    splits = source / "experiments" / "splits"
    splits.mkdir(parents=True)
    (splits / "split75.txt").write_text(f"{comp_id}\n")
    (splits / "low.txt").write_text(f"{comp_id}\n")
    (source / "experiments" / "competition_categories.csv").write_text(
        "competition_id,category,dataset_size_GB,EnabledDate,DeadlineDate,Complexity\n"
        "Fake-Kaggle,Tabular,0.001,1/1/2024,1/2/2024,Low\n"
    )

    public = data / comp_id / "prepared" / "public"
    private = data / comp_id / "prepared" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    (public / "train.csv").write_text("id,label\n1,0\n")
    (public / "sampleSubmission.csv").write_text("id,label\n2,0\n")
    (private / "test.csv").write_text("id,label\n2,1\n")
    return source, data


def test_mle_bench_converter_keeps_private_data_out_of_environment(tmp_path: Path):
    converter = _load_converter()
    source, data = _write_fake_source(tmp_path)

    generated = converter.convert_all(
        source,
        tmp_path / "out",
        data_dir=data,
        split="split75",
        overwrite=True,
    )

    assert len(generated) == 1
    task_dir = generated[0]
    assert task_dir.name == "fake-kaggle"
    assert (task_dir / "environment" / "data" / "train.csv").is_file()
    assert (task_dir / "environment" / "data" / "sample_submission.csv").is_file()
    assert (task_dir / "environment" / "data" / "sampleSubmission.csv").is_file()
    assert not (task_dir / "environment" / "private-data").exists()
    assert (
        task_dir
        / "tests"
        / "private-data"
        / "Fake-Kaggle"
        / "prepared"
        / "private"
        / "test.csv"
    ).is_file()

    metadata = json.loads((task_dir / "tests" / "mlebench_task.json").read_text())
    assert metadata["competition_id"] == "Fake-Kaggle"
    assert metadata["slug"] == "fake-kaggle"
    assert metadata["reward"] == "1.0 for any medal, otherwise 0.0"

    config = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text())
    assert config.task is not None
    assert config.task.name == "mle-bench/fake-kaggle"
    assert config.environment.allow_internet is False

    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
    assert 'CMD ["sleep", "infinity"]' in dockerfile

    test_sh = (task_dir / "tests" / "test.sh").read_text()
    assert test_sh.startswith("#!/bin/bash\n")
    assert (
        subprocess.run(
            ["bash", "-n", str(task_dir / "tests" / "test.sh")],
            check=False,
        ).returncode
        == 0
    )


def test_mle_bench_converter_fails_when_core_package_file_is_missing(tmp_path: Path):
    converter = _load_converter()
    source, data = _write_fake_source(tmp_path)
    (source / "mlebench" / "grade.py").unlink()

    try:
        converter.convert_all(
            source,
            tmp_path / "out",
            data_dir=data,
            split="split75",
            overwrite=True,
        )
    except FileNotFoundError as exc:
        assert "Required MLE-bench core file missing" in str(exc)
    else:
        raise AssertionError("missing MLE-bench core file should fail conversion")


def test_mle_bench_verifier_fails_loudly_when_grader_package_is_broken(tmp_path: Path):
    converter = _load_converter()
    source, data = _write_fake_source(tmp_path)
    task_dir = converter.convert_all(
        source,
        tmp_path / "out",
        data_dir=data,
        split="split75",
        overwrite=True,
    )[0]
    (task_dir / "tests" / "mlebench" / "grade.py").unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(task_dir / "tests" / "verify.py"),
            "--competition-id",
            "Fake-Kaggle",
            "--submission",
            str(task_dir / "environment" / "data" / "sample_submission.csv"),
            "--data-dir",
            str(task_dir / "tests" / "private-data"),
            "--reward-file",
            str(tmp_path / "reward.txt"),
            "--reward-json",
            str(tmp_path / "reward.json"),
            "--report-file",
            str(tmp_path / "grading_report.json"),
        ],
        cwd=task_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert not (tmp_path / "reward.txt").exists()
    assert "mlebench.grade" in result.stderr


def test_mle_bench_converter_accepts_sanitized_task_ids(tmp_path: Path):
    converter = _load_converter()
    source, data = _write_fake_source(tmp_path)

    generated = converter.convert_all(
        source,
        tmp_path / "out",
        data_dir=data,
        task_ids=["fake-kaggle"],
        overwrite=True,
    )

    assert [path.name for path in generated] == ["fake-kaggle"]


def test_mle_bench_parity_checks_generated_subset(tmp_path: Path):
    converter = _load_converter()
    parity = _load_parity()
    source, data = _write_fake_source(tmp_path)
    out = tmp_path / "out"
    converter.convert_all(source, out, data_dir=data, split="low", overwrite=True)

    result = parity.full_parity(out)

    assert result["structural_parity"]["tasks_tested"] == 1
    assert result["structural_parity"]["passed"] == 1
    assert result["eval_parity"]["tasks_tested"] == 1
    assert result["eval_parity"]["passed"] == 1
    assert result["side_by_side_status"] == "recorded"
    assert len(result["conversion_parity"]["tasks"]) == 1
    assert len(result["reward_distribution_parity"]["samples"]) == 1
    assert result["reward_distribution_parity"]["samples"][0]["reward_delta"] == 0.0
