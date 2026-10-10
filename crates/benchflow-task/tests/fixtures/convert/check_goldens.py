"""Record what task.md's reference checker says about each imported package in expected/.

usage: python check_goldens.py <fixtures/convert dir> <task-md tools dir>

Runs tools/taskmd.py's check (parse, then the family instance check, as `taskmd.py check` does) on every package
under expected/harbor/ and expected/v06/, and writes the error messages, sorted, to expected/checked.json. Run it
after make_goldens.py. The packages there are convert.py's imports, which benchflow-task's imports match byte for
byte, so these are the errors the Rust checker must report on the Rust imports.
"""

import json
import sys
from pathlib import Path

fixtures, tools = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(tools))
import taskmd  # noqa: E402

expected = fixtures / "expected"
checked: dict[str, list[str]] = {}
for kind in ("harbor", "v06"):
    for pkg in sorted((expected / kind).iterdir()):
        if not (pkg / "task.md").is_file():
            continue
        doc = taskmd.parse(pkg)
        diags = doc.diagnostics + taskmd.family_instance_check(doc, pkg)
        checked[f"{kind}/{pkg.name}"] = sorted(d.message for d in diags if d.level == "error")

(expected / "checked.json").write_text(json.dumps(checked, indent=2, ensure_ascii=False) + "\n")
print(f"checked {len(checked)} packages")
