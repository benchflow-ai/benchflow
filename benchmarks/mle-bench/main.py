"""Run the Mle Bench converter CLI.

Thin delegator so ``python benchmarks/mle-bench/main.py --output-dir ...`` keeps
working while ``benchflow.py`` stays the single source of truth.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("benchflow.py")), run_name="__main__")
