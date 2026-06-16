"""Convert OpenAI MLE-bench competitions into BenchFlow task dirs.

MLE-bench is a collection of Kaggle competitions with its own preparation and
grading code. This adapter keeps one competition as one BenchFlow task, exposes
only the prepared public data to the agent, and runs the upstream grader from
``tests/`` against the agent's ``/home/submission/submission.csv``.

The converter expects a local checkout of ``openai/mle-bench`` and a prepared
data directory produced by ``mlebench prepare``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_SPLIT = "split75"
_CORE_MLEBENCH_FILES = (
    "__init__.py",
    "data.py",
    "grade.py",
    "grade_helpers.py",
    "metrics.py",
    "registry.py",
    "utils.py",
)


@dataclass(frozen=True)
class MLEBenchCompetition:
    competition_id: str
    slug: str
    name: str
    description: str
    competition_type: str
    grader_name: str
    grade_fn: str
    dataset: dict[str, str]
    category: str
    complexity: str
    dataset_size_gb: float | None
    splits: list[str]


def _sanitize_id(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot derive task slug from {raw!r}")
    return slug


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_list(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(v) for v in values) + "]"


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src, dst, symlinks=False, ignore=shutil.ignore_patterns(".git", "__pycache__")
    )


def _resolve_source_dir(source_dir: Path | None) -> Path:
    candidates = []
    if source_dir is not None:
        candidates.append(source_dir)
    else:
        candidates.extend(
            [
                _SCRIPT_DIR / "source",
                Path.cwd() / "mle-bench",
                Path.cwd() / "mle-bench-upstream",
                Path.home() / "mle-bench",
                Path("/tmp/mle-bench-upstream"),
            ]
        )

    for candidate in candidates:
        root = candidate.resolve()
        if (root / "mlebench" / "competitions").is_dir():
            return root
        if root.name == "mlebench" and (root / "competitions").is_dir():
            return root.parent

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Cannot find an MLE-bench checkout. Pass --source-dir /path/to/mle-bench. "
        f"Searched: {searched}"
    )


def _resolve_data_dir(data_dir: Path | None, source_dir: Path) -> Path:
    if data_dir is not None:
        return data_dir.resolve()
    for env_name in ("MLE_BENCH_DATA_DIR", "MLEBENCH_DATA_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            return Path(env_value).expanduser().resolve()
    if (source_dir / "data").is_dir():
        return (source_dir / "data").resolve()
    return (Path.home() / ".cache" / "mle-bench" / "data").resolve()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _load_splits(source_dir: Path) -> dict[str, set[str]]:
    splits_dir = source_dir / "experiments" / "splits"
    if not splits_dir.is_dir():
        return {}

    splits: dict[str, set[str]] = {}
    for path in sorted(splits_dir.glob("*.txt")):
        ids = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        splits[path.stem] = ids
    return splits


def _load_category_metadata(source_dir: Path) -> dict[str, dict[str, str]]:
    path = source_dir / "experiments" / "competition_categories.csv"
    if not path.is_file():
        return {}

    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            competition_id = (row.get("competition_id") or "").strip()
            if not competition_id:
                continue
            out[competition_id] = {
                "category": (row.get("category") or "").strip(),
                "complexity": (row.get("Complexity") or "").strip().lower(),
                "dataset_size_GB": (row.get("dataset_size_GB") or "").strip(),
            }
    return out


def _dataset_size(meta: dict[str, str]) -> float | None:
    raw = meta.get("dataset_size_GB", "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def load_competitions(source_dir: Path | None = None) -> list[MLEBenchCompetition]:
    """Load competition metadata from a local MLE-bench checkout."""
    root = _resolve_source_dir(source_dir)
    split_map = _load_splits(root)
    category_meta = _load_category_metadata(root)
    competitions_dir = root / "mlebench" / "competitions"

    competitions: list[MLEBenchCompetition] = []
    for config_path in sorted(competitions_dir.glob("*/config.yaml")):
        config = _load_yaml(config_path)
        competition_id = str(config.get("id") or config_path.parent.name)
        slug = _sanitize_id(competition_id)
        description_path = root / str(config["description"])
        description = description_path.read_text(encoding="utf-8")
        dataset = {str(k): str(v) for k, v in dict(config.get("dataset", {})).items()}
        grader = dict(config.get("grader", {}))
        meta = category_meta.get(competition_id, {})
        splits = sorted(
            name for name, ids in split_map.items() if competition_id in ids
        )
        complexity = meta.get("complexity") or _infer_complexity_from_splits(splits)

        competitions.append(
            MLEBenchCompetition(
                competition_id=competition_id,
                slug=slug,
                name=str(config.get("name") or competition_id),
                description=description,
                competition_type=str(config.get("competition_type") or "unknown"),
                grader_name=str(grader.get("name") or "unknown"),
                grade_fn=str(grader.get("grade_fn") or ""),
                dataset=dataset,
                category=meta.get("category") or "machine-learning",
                complexity=complexity,
                dataset_size_gb=_dataset_size(meta),
                splits=splits,
            )
        )
    return competitions


def _infer_complexity_from_splits(splits: list[str]) -> str:
    for name in ("low", "medium", "high"):
        if name in splits:
            return name
    return "medium"


def _select_competitions(
    competitions: list[MLEBenchCompetition],
    *,
    source_dir: Path,
    split: str | None,
    task_ids: list[str] | None,
    limit: int | None,
) -> list[MLEBenchCompetition]:
    selected = competitions
    if task_ids:
        wanted = {item.strip() for item in task_ids if item.strip()}
        wanted |= {_sanitize_id(item) for item in wanted}
        selected = [
            task
            for task in selected
            if task.competition_id in wanted or task.slug in wanted
        ]
    elif split and split != "all":
        split_map = _load_splits(source_dir)
        if split not in split_map:
            valid = ", ".join(["all", *sorted(split_map)])
            raise ValueError(
                f"Unknown MLE-bench split {split!r}; expected one of: {valid}"
            )
        selected = [
            task for task in selected if task.competition_id in split_map[split]
        ]

    if limit is not None:
        selected = selected[:limit]
    return selected


def _public_dir(data_dir: Path, competition: MLEBenchCompetition) -> Path:
    return data_dir / competition.competition_id / "prepared" / "public"


def _private_dir(data_dir: Path, competition: MLEBenchCompetition) -> Path:
    return data_dir / competition.competition_id / "prepared" / "private"


def _answers_path(data_dir: Path, competition: MLEBenchCompetition) -> Path:
    return data_dir / competition.dataset["answers"]


def _sample_submission_path(
    data_dir: Path, competition: MLEBenchCompetition
) -> Path | None:
    sample = competition.dataset.get("sample_submission")
    if not sample:
        return None
    return data_dir / sample


def _copy_prepared_data(
    competition: MLEBenchCompetition,
    *,
    data_dir: Path,
    task_dir: Path,
    include_data: bool,
) -> None:
    public_dst = task_dir / "environment" / "data"
    private_root = task_dir / "tests" / "private-data"

    if not include_data:
        public_dst.mkdir(parents=True, exist_ok=True)
        (public_dst / "description.md").write_text(
            competition.description, encoding="utf-8"
        )
        return

    public_src = _public_dir(data_dir, competition)
    private_src = _private_dir(data_dir, competition)
    answers_src = _answers_path(data_dir, competition)

    if not public_src.is_dir():
        raise FileNotFoundError(
            f"Prepared public data missing for {competition.competition_id}: {public_src}. "
            f"Run `mlebench prepare -c {competition.competition_id} --data-dir {data_dir}` first."
        )
    if not private_src.is_dir():
        raise FileNotFoundError(
            f"Prepared private data missing for {competition.competition_id}: {private_src}. "
            f"Run `mlebench prepare -c {competition.competition_id} --data-dir {data_dir}` first."
        )
    if not answers_src.is_file():
        raise FileNotFoundError(
            f"Prepared answer file missing for {competition.competition_id}: {answers_src}"
        )

    _copytree(public_src, public_dst)
    if not (public_dst / "description.md").is_file():
        (public_dst / "description.md").write_text(
            competition.description, encoding="utf-8"
        )
    sample_src = _sample_submission_path(data_dir, competition)
    if sample_src is not None and sample_src.is_file():
        canonical_sample = public_dst / "sample_submission.csv"
        if sample_src.name != canonical_sample.name:
            shutil.copy2(sample_src, canonical_sample)

    private_dst = private_root / competition.competition_id / "prepared" / "private"
    _copytree(private_src, private_dst)

    answers_dst = private_root / competition.dataset["answers"]
    answers_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(answers_src, answers_dst)


def _copy_mlebench_package(
    source_dir: Path,
    task_dir: Path,
    competition: MLEBenchCompetition,
) -> None:
    src_pkg = source_dir / "mlebench"
    dst_pkg = task_dir / "tests" / "mlebench"
    dst_pkg.mkdir(parents=True, exist_ok=True)

    for filename in _CORE_MLEBENCH_FILES:
        src = src_pkg / filename
        if not src.is_file():
            raise FileNotFoundError(f"Required MLE-bench core file missing: {src}")
        shutil.copy2(src, dst_pkg / filename)

    competitions_dst = dst_pkg / "competitions"
    competitions_dst.mkdir(exist_ok=True)
    init = src_pkg / "competitions" / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(
            f"Required MLE-bench competitions package file missing: {init}"
        )
    shutil.copy2(init, competitions_dst / "__init__.py")
    utils = src_pkg / "competitions" / "utils.py"
    if not utils.is_file():
        raise FileNotFoundError(
            f"Required MLE-bench competitions utility file missing: {utils}"
        )
    shutil.copy2(utils, competitions_dst / "utils.py")

    comp_src = src_pkg / "competitions" / competition.competition_id
    if not comp_src.is_dir():
        raise FileNotFoundError(f"MLE-bench competition package missing: {comp_src}")
    _copytree(comp_src, competitions_dst / competition.competition_id)


def _render_task_toml(competition: MLEBenchCompetition) -> str:
    tags = [
        "mle-bench",
        _sanitize_id(competition.category),
        competition.complexity,
        competition.grader_name,
    ]
    dataset_size = (
        str(competition.dataset_size_gb)
        if competition.dataset_size_gb is not None
        else "0.0"
    )
    split_list = _toml_list(competition.splits)
    return textwrap.dedent(
        f"""\
        version = "1.0"

        [task]
        name = {_toml_string(f"mle-bench/{competition.slug}")}

        [metadata]
        author_name = "OpenAI"
        difficulty = {_toml_string(competition.complexity)}
        category = {_toml_string(competition.category)}
        tags = {_toml_list(tags)}

        [metadata.mle_bench]
        competition_id = {_toml_string(competition.competition_id)}
        competition_type = {_toml_string(competition.competition_type)}
        grader = {_toml_string(competition.grader_name)}
        grade_fn = {_toml_string(competition.grade_fn)}
        dataset_size_gb = {dataset_size}
        splits = {split_list}
        reward = "any_medal"

        [agent]
        timeout_sec = 86400

        [verifier]
        timeout_sec = 3600
        user = "root"

        [environment]
        build_timeout_sec = 3600
        cpus = 4
        memory_mb = 16384
        storage_mb = 10240
        allow_internet = false
        workdir = "/home"
        """
    )


def _render_instruction(competition: MLEBenchCompetition) -> str:
    size_note = (
        f"{competition.dataset_size_gb:g} GB"
        if competition.dataset_size_gb is not None
        else "unknown size"
    )
    splits = ", ".join(competition.splits) if competition.splits else "unlisted"
    return textwrap.dedent(
        f"""\
        # MLE-bench: {competition.name}

        You are solving an offline MLE-bench competition adapted to BenchFlow.
        Train or otherwise construct a model using the prepared public data and
        produce a Kaggle-style CSV submission.

        ## Required Output

        Write your final submission to:

        `/home/submission/submission.csv`

        The verifier only grades that file. If it is missing, malformed, or not
        a CSV, the reward is `0`.

        ## Data

        - Public competition data is in `/home/data/`.
        - The competition description is also available at `/home/data/description.md`.
        - The sample submission file, when provided upstream, is in `/home/data/sample_submission.csv`.
        - A lightweight CSV-shape check is available as `/home/validate_submission.sh`.

        ## Scoring

        BenchFlow runs the upstream MLE-bench grader for `{competition.competition_id}`.
        The scalar reward is `1.0` when the submission earns any Kaggle medal
        under MLE-bench thresholds, otherwise `0.0`. The full grading report is
        saved to `/logs/verifier/grading_report.json`.

        ## Metadata

        - Competition ID: `{competition.competition_id}`
        - Category: {competition.category}
        - Complexity: {competition.complexity}
        - Split membership: {splits}
        - Dataset size: {size_note}
        - Grader: `{competition.grader_name}`

        ## Competition Description

        {competition.description.strip()}
        """
    )


def _render_dockerfile() -> str:
    return textwrap.dedent(
        """\
        FROM python:3.11-slim

        ENV DEBIAN_FRONTEND=noninteractive

        RUN apt-get update -qq && apt-get install -y -qq \\
            build-essential \\
            curl \\
            ffmpeg \\
            git \\
            libgl1 \\
            libglib2.0-0 \\
            p7zip-full \\
            unzip \\
            zip \\
            && rm -rf /var/lib/apt/lists/*

        RUN pip install --no-cache-dir \\
            appdirs \\
            diskcache \\
            levenshtein \\
            numpy \\
            pandas \\
            pillow \\
            py7zr \\
            pycocotools \\
            pyquaternion \\
            pyyaml \\
            scikit-learn \\
            scipy \\
            shapely \\
            tenacity \\
            tqdm

        WORKDIR /home

        COPY data/ /home/data/
        COPY instructions.txt /home/instructions.txt
        COPY validate_submission.sh /home/validate_submission.sh

        RUN mkdir -p /home/submission /workspace /logs/verifier /logs/agent /logs/artifacts \\
            && chmod +x /home/validate_submission.sh \\
            && chmod -R a+rwX /home/submission /workspace /logs

        CMD ["sleep", "infinity"]
        """
    )


def _render_environment_instructions() -> str:
    return textwrap.dedent(
        """\
        You are participating in MLE-bench, an offline adaptation of Kaggle
        competitions for evaluating machine learning engineering agents.

        Use the data in /home/data/ and write your final submission CSV to
        /home/submission/submission.csv. The BenchFlow verifier grades only that
        file using the upstream MLE-bench grader.
        """
    )


def _render_validate_submission_sh() -> str:
    return textwrap.dedent(
        """\
        #!/bin/bash
        set -euo pipefail

        SUBMISSION="${1:-/home/submission/submission.csv}"
        SAMPLE="/home/data/sample_submission.csv"

        if [ ! -f "$SUBMISSION" ]; then
            echo "Submission invalid: $SUBMISSION does not exist."
            exit 1
        fi

        python3 - "$SUBMISSION" "$SAMPLE" <<'PY'
        import sys
        from pathlib import Path

        import pandas as pd

        submission = Path(sys.argv[1])
        sample = Path(sys.argv[2])

        try:
            submission_df = pd.read_csv(submission)
        except Exception as exc:
            print(f"Submission invalid: could not read CSV: {exc}")
            raise SystemExit(1)

        if sample.is_file():
            sample_df = pd.read_csv(sample, nrows=0)
            if list(submission_df.columns) != list(sample_df.columns):
                print(
                    "Submission invalid: columns differ from sample_submission.csv. "
                    f"Expected {list(sample_df.columns)}, got {list(submission_df.columns)}."
                )
                raise SystemExit(1)

        print("Submission shape looks valid. Final grading runs after the task.")
        PY
        """
    )


def _render_test_sh(competition: MLEBenchCompetition) -> str:
    return Template(
        textwrap.dedent(
            """\
        #!/bin/bash
        set -euo pipefail

        mkdir -p /logs/verifier /logs/artifacts

        python3 /tests/verify.py \\
          --competition-id "$competition_id" \\
          --submission /home/submission/submission.csv \\
          --data-dir /tests/private-data \\
          --reward-file /logs/verifier/reward.txt \\
          --reward-json /logs/verifier/reward.json \\
          --report-file /logs/verifier/grading_report.json
        """
        )
    ).substitute(competition_id=competition.competition_id)


VERIFY_PY = '''\
"""BenchFlow verifier for one converted MLE-bench competition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _write_outputs(
    *,
    reward: float,
    report: dict,
    reward_file: Path,
    reward_json: Path,
    report_file: Path,
) -> None:
    reward_file.parent.mkdir(parents=True, exist_ok=True)
    reward_json.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    reward_file.write_text(f"{reward:.6f}\\n")
    report_file.write_text(json.dumps(report, indent=2, sort_keys=True))
    reward_json.write_text(
        json.dumps(
            {
                "reward": reward,
                "any_medal": 1.0 if report.get("any_medal") else 0.0,
                "gold_medal": 1.0 if report.get("gold_medal") else 0.0,
                "silver_medal": 1.0 if report.get("silver_medal") else 0.0,
                "bronze_medal": 1.0 if report.get("bronze_medal") else 0.0,
                "above_median": 1.0 if report.get("above_median") else 0.0,
                "valid_submission": 1.0 if report.get("valid_submission") else 0.0,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-id", required=True)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--reward-file", required=True, type=Path)
    parser.add_argument("--reward-json", required=True, type=Path)
    parser.add_argument("--report-file", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from mlebench.grade import grade_csv
    from mlebench.registry import registry

    competition = registry.set_data_dir(args.data_dir).get_competition(
        args.competition_id
    )
    report = grade_csv(args.submission, competition).to_dict()
    reward = 1.0 if report.get("any_medal") else 0.0

    _write_outputs(
        reward=reward,
        report=report,
        reward_file=args.reward_file,
        reward_json=args.reward_json,
        report_file=args.report_file,
    )


if __name__ == "__main__":
    main()
'''


def _write_metadata(task_dir: Path, competition: MLEBenchCompetition) -> None:
    metadata = {
        "benchmark": "mle-bench",
        "upstream_repo": "https://github.com/openai/mle-bench",
        "competition_id": competition.competition_id,
        "slug": competition.slug,
        "name": competition.name,
        "competition_type": competition.competition_type,
        "category": competition.category,
        "complexity": competition.complexity,
        "dataset_size_gb": competition.dataset_size_gb,
        "splits": competition.splits,
        "grader_name": competition.grader_name,
        "grade_fn": competition.grade_fn,
        "dataset": competition.dataset,
        "reward": "1.0 for any medal, otherwise 0.0",
    }
    (task_dir / "tests" / "mlebench_task.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def convert(
    source_instance: MLEBenchCompetition,
    output_dir: Path,
    *,
    overwrite: bool = False,
    source_dir: Path | None = None,
    data_dir: Path | None = None,
    include_data: bool = True,
) -> Path:
    """Convert one MLE-bench competition into one BenchFlow task directory."""
    if not isinstance(source_instance, MLEBenchCompetition):
        raise TypeError("source_instance must be an MLEBenchCompetition")

    root = _resolve_source_dir(source_dir)
    data_root = _resolve_data_dir(data_dir, root)
    competition = source_instance
    task_dir = output_dir / competition.slug

    if task_dir.exists():
        if not overwrite:
            logger.debug("Skipping existing task %s", task_dir)
            return task_dir
        shutil.rmtree(task_dir)

    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()

    _copy_prepared_data(
        competition,
        data_dir=data_root,
        task_dir=task_dir,
        include_data=include_data,
    )
    _copy_mlebench_package(root, task_dir, competition)

    (task_dir / "task.toml").write_text(
        _render_task_toml(competition), encoding="utf-8"
    )
    (task_dir / "instruction.md").write_text(
        _render_instruction(competition), encoding="utf-8"
    )
    (task_dir / "environment" / "Dockerfile").write_text(
        _render_dockerfile(), encoding="utf-8"
    )
    (task_dir / "environment" / "instructions.txt").write_text(
        _render_environment_instructions(),
        encoding="utf-8",
    )
    validate_sh = task_dir / "environment" / "validate_submission.sh"
    validate_sh.write_text(_render_validate_submission_sh(), encoding="utf-8")
    validate_sh.chmod(0o755)

    test_sh = task_dir / "tests" / "test.sh"
    test_sh.write_text(_render_test_sh(competition), encoding="utf-8")
    test_sh.chmod(0o755)
    (task_dir / "tests" / "verify.py").write_text(VERIFY_PY, encoding="utf-8")
    _write_metadata(task_dir, competition)

    return task_dir


def convert_all(
    source_dir: Path | None,
    output_dir: Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    task_ids: list[str] | None = None,
    data_dir: Path | None = None,
    split: str | None = _DEFAULT_SPLIT,
    include_data: bool = True,
) -> list[Path]:
    """Convert selected MLE-bench competitions into BenchFlow task dirs."""
    root = _resolve_source_dir(source_dir)
    competitions = load_competitions(root)
    selected = _select_competitions(
        competitions,
        source_dir=root,
        split=split,
        task_ids=task_ids,
        limit=limit,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for competition in selected:
        generated.append(
            convert(
                competition,
                output_dir,
                overwrite=overwrite,
                source_dir=root,
                data_dir=data_dir,
                include_data=include_data,
            )
        )
        logger.info("Generated %s", competition.competition_id)

    logger.info("Generated %d MLE-bench tasks in %s", len(generated), output_dir)
    return generated


def _parse_task_ids(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenAI MLE-bench to BenchFlow."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Prepared MLE-bench data dir; defaults to MLE_BENCH_DATA_DIR or ~/.cache/mle-bench/data.",
    )
    parser.add_argument(
        "--split",
        default=_DEFAULT_SPLIT,
        help="MLE-bench split to convert (default: split75). Use 'all' for every config.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--task-ids", default=None, help="Comma-separated competition IDs or slugs"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Do not copy prepared data; generated tasks document structure but are not runnable.",
    )
    args = parser.parse_args()

    generated = convert_all(
        args.source_dir,
        args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        task_ids=_parse_task_ids(args.task_ids),
        data_dir=args.data_dir,
        split=args.split,
        include_data=not args.metadata_only,
    )
    logger.info("Generated %d mle-bench tasks in %s", len(generated), args.output_dir)


if __name__ == "__main__":
    main()
