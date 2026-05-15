"""Build deterministic findings and audit-agent prompt bundles for E2E runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_findings(run_dir: str | Path) -> dict[str, Any]:
    """Create machine-readable findings from deterministic audit reports."""
    run_dir = Path(run_dir)
    matrix = _load_json(run_dir / "matrix_summary.json")
    artifact = _load_json(run_dir / "artifact_audit.json")
    parity = _load_json(run_dir / "parity_report.json")

    entries = matrix.get("entries", []) if isinstance(matrix.get("entries"), list) else []
    failed_entries = [
        e
        for e in entries
        if e.get("status") not in {"completed", "passed", "failed", "planned"}
        or e.get("error")
    ]
    gemini_compat = [
        e
        for e in failed_entries
        if "gemini" in str(e.get("model", "")).lower()
        and any(
            token in str(e.get("error", "")).lower()
            for token in ("model", "provider", "api key", "unsupported", "auth")
        )
    ]

    missing_baselines = [
        t.get("task_name")
        for t in parity.get("tasks", [])
        if isinstance(t, dict) and t.get("missing_baseline")
    ]

    return {
        "run_dir": str(run_dir),
        "total_entries": len(entries),
        "failed_entries": len(failed_entries),
        "gemini_compatibility_failures": gemini_compat,
        "artifact_errors": artifact.get("error_count", 0),
        "artifact_warnings": artifact.get("warning_count", 0),
        "missing_baseline_tasks": sorted(t for t in missing_baselines if t),
        "token_fields_present": parity.get("token_fields_present"),
        "baseline_token_fields_present": parity.get("baseline_token_fields_present"),
    }


def render_findings_markdown(findings: dict[str, Any]) -> str:
    """Render deterministic findings as Markdown."""
    lines = [
        "# SkillsBench E2E findings",
        "",
        f"- Run directory: `{findings.get('run_dir')}`",
        f"- Matrix entries: {findings.get('total_entries', 0)}",
        f"- Failed/problem entries: {findings.get('failed_entries', 0)}",
        f"- Artifact errors: {findings.get('artifact_errors', 0)}",
        f"- Artifact warnings: {findings.get('artifact_warnings', 0)}",
        f"- Current token fields present: {findings.get('token_fields_present')}",
        f"- Baseline token fields present: {findings.get('baseline_token_fields_present')}",
        "",
        "## Gemini compatibility failures",
        "",
    ]
    compat = findings.get("gemini_compatibility_failures") or []
    if compat:
        lines.extend(
            f"- `{e.get('agent')}` / `{e.get('task_name')}`: {e.get('error')}"
            for e in compat
        )
    else:
        lines.append("- None detected by deterministic rules.")

    missing = findings.get("missing_baseline_tasks") or []
    lines.extend(["", "## Missing baseline tasks", ""])
    if missing:
        lines.extend(f"- `{task}`" for task in missing)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def render_audit_prompt(run_dir: str | Path, prompt_path: str | Path | None) -> str:
    """Render an audit-agent prompt bundle for a completed E2E output directory."""
    run_dir = Path(run_dir)
    rubric = Path(prompt_path).read_text() if prompt_path else ""
    bundle = {
        "matrix_summary": _load_json(run_dir / "matrix_summary.json"),
        "artifact_audit": _load_json(run_dir / "artifact_audit.json"),
        "parity_report": _load_json(run_dir / "parity_report.json"),
        "deterministic_findings": _load_json(run_dir / "audit_findings.json"),
    }
    return (
        f"{rubric.strip()}\n\n"
        "## Output bundle JSON\n\n"
        "```json\n"
        f"{json.dumps(bundle, indent=2)[:120000]}\n"
        "```\n"
    )


def write_audit_outputs(
    run_dir: str | Path,
    prompt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write deterministic findings, Markdown summary, and audit-agent prompt."""
    run_dir = Path(run_dir)
    findings = build_findings(run_dir)
    (run_dir / "audit_findings.json").write_text(json.dumps(findings, indent=2))
    (run_dir / "findings.md").write_text(render_findings_markdown(findings))
    if prompt_path:
        (run_dir / "audit_agent_prompt.md").write_text(
            render_audit_prompt(run_dir, prompt_path)
        )
    return findings
