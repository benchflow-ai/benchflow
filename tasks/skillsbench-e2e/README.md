# SkillsBench E2E matrix

This directory contains the file-driven ENG-6 E2E configuration.

Canonical dry run:

```bash
uv run bench eval create -f tasks/skillsbench-e2e/e2e.yaml --dry-run
```

Canonical live run:

```bash
export BENCHFLOW_RUN_SKILLSBENCH_E2E=1
export DAYTONA_API_KEY=...
export GEMINI_API_KEY=...
uv run bench eval create -f tasks/skillsbench-e2e/e2e.yaml
```

The live run executes the 9 selected SkillsBench tasks across every registered
BenchFlow agent using `gemini-3.1-flash-lite-preview` on Daytona with global
concurrency 30. It is intentionally gated and should not run on every commit.

Outputs are written under `jobs/skillsbench-e2e/<run-id>/`:

- `matrix_config.json`
- `matrix_summary.json`
- `artifact_audit.json`
- `parity_report.json`
- `audit_findings.json`
- `findings.md`
