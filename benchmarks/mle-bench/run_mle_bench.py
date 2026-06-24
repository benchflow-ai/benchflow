"""Convert MLE-bench tasks if needed, then run them through BenchFlow."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent


def _load_converter():
    spec = importlib.util.spec_from_file_location(
        "benchflow_mle_bench_converter", _HERE / "benchflow.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_task_ids(raw: list[str] | None) -> list[str] | None:
    """Accept both space-separated and comma-separated ``--task-ids`` values.

    The converter CLI and the README document the comma-separated form
    (``--task-ids a,b``); splitting here keeps that idiom working when the
    runner collects values via ``nargs="*"`` instead of silently treating
    ``"a,b"`` as one unmatched id.
    """
    if not raw:
        return None
    ids = [tid.strip() for token in raw for tid in token.split(",") if tid.strip()]
    return ids or None


def ensure_converted_tasks(args: argparse.Namespace) -> Path:
    """Convert the source benchmark into BenchFlow tasks under ``tasks/``."""
    converter = _load_converter()
    output_dir = args.output_dir or (_HERE / "tasks")
    converter.convert_all(
        args.source_dir,
        output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        task_ids=_normalize_task_ids(args.task_ids),
        data_dir=args.data_dir,
        split=args.split,
        include_data=not args.metadata_only,
    )
    return output_dir


async def run(args: argparse.Namespace) -> None:
    from benchflow.evaluation import Evaluation

    tasks_dir = ensure_converted_tasks(args)
    config = yaml.safe_load((_HERE / "mle-bench.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("mle-bench.yaml must contain a YAML mapping")
    config["tasks_dir"] = str(tasks_dir)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        handle.flush()
        result = await Evaluation.from_yaml(handle.name).run()
    print(f"Score: {result.passed}/{result.total} ({result.score:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLE-bench via BenchFlow.")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="split75")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
