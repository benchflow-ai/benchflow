#!/usr/bin/env python3
"""Re-run deterministic SkillsBench E2E audits on an existing output directory.

Usage:
    python benchmarks/scripts/skillsbench_e2e_audit.py jobs/skillsbench-e2e/<run-id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def main() -> int:
    from benchflow.integration.artifact_audit import write_artifact_audit
    from benchflow.integration.audit_agent import write_audit_outputs
    from benchflow.integration.parity import write_parity_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument(
        "--audit-prompt",
        type=Path,
        default=REPO_ROOT
        / "tasks"
        / "skillsbench-e2e"
        / "audit"
        / "trajectory-result-auditor.md",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        parser.error(f"run_dir does not exist: {run_dir}")

    artifact = write_artifact_audit(run_dir)
    parity = write_parity_report(run_dir, args.baseline_dir)
    findings = write_audit_outputs(run_dir, args.audit_prompt)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "artifact_ok": artifact["ok"],
                "artifact_errors": artifact["error_count"],
                "artifact_warnings": artifact["warning_count"],
                "current_results": parity["current_count"],
                "baseline_results": parity["baseline_count"],
                "failed_entries": findings["failed_entries"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
