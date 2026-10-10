"""Produce golden outputs for benchflow-task's conversion tests with task.md's tools/convert.py.

usage: python make_goldens.py <fixtures/convert dir> <task-md tools dir>

For every input under harbor/ and v06/ it runs convert.py's import (import_harbor / import_v06) into
expected/<kind>/<name>/, then exports that package back with export_harbor into
expected/roundtrip/<kind>/<name>/. For every task.md package under export/ it runs export_harbor into
expected/export/<name>/. Errors and export reports (minus paths and the tool name) go to expected/results.json.
"""

import json
import shutil
import sys
from pathlib import Path

fixtures, tools = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(tools))
import convert  # noqa: E402

expected = fixtures / "expected"
if expected.exists():
    shutil.rmtree(expected)
results: dict[str, dict] = {}


def export(src: Path, dst: Path, key: str) -> None:
    report = dst.parent / (dst.name + ".report.json")
    try:
        losses = convert.export_harbor(src, dst, allow_degraded=False, report=report)
        rep = json.loads(report.read_text())
    except convert.ExportRefused as exc:
        rep = exc.report
        losses = None
    if report.exists():
        report.unlink()
    results[key] = {
        "status": rep["status"],
        "losses": rep["losses"],
        "restored_originals": rep["restored_originals"],
        "rewritten": rep["rewritten"],
        "restored_extension_paths": rep["restored_extension_paths"],
    }
    if losses is None and dst.exists():
        shutil.rmtree(dst)


for kind in ("harbor", "v06"):
    for src in sorted((fixtures / kind).iterdir()):
        if not src.is_dir():
            continue
        dst = expected / kind / src.name
        key = f"{kind}/{src.name}"
        try:
            notes = convert.import_harbor(src, dst) if kind == "harbor" else convert.import_v06(src, dst)
            results[key] = {"notes": notes}
        except (ValueError, FileExistsError) as exc:
            results[key] = {"error": str(exc)}
            if dst.exists():
                shutil.rmtree(dst)
            continue
        export(dst, expected / "roundtrip" / kind / src.name, f"roundtrip/{kind}/{src.name}")

export_dir = fixtures / "export"
if export_dir.is_dir():
    for src in sorted(export_dir.iterdir()):
        if src.is_dir():
            export(src, expected / "export" / src.name, f"export/{src.name}")

(expected / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {len(results)} results")
