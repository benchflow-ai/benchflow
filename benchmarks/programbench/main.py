"""CLI entry point for ProgramBench → BenchFlow task generation.

Thin delegator to :func:`benchmarks.programbench.benchflow.main` so the
documented ``python -m benchmarks.programbench.main`` invocation keeps
working while ``benchmarks/programbench/benchflow.py`` is the single
source of truth for the converter CLI.

Usage::

    python -m benchmarks.programbench.main --output-dir benchmarks/programbench/tasks
    python -m benchmarks.programbench.main --output-dir out --limit 5
    python -m benchmarks.programbench.main --output-dir out --task-ids jqlang__jq.b33a763
"""

from __future__ import annotations

from benchmarks.programbench.benchflow import main

if __name__ == "__main__":
    main()
