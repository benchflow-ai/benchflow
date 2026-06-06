#!/usr/bin/env python3
"""Complete the #914 skill-catalog fix: trim bloated SKILL.md frontmatter to the
canonical name+description so the OpenHands SDK stops silently dropping the skill.
The SDK drops a SKILL.md whose frontmatter has type-mismatched/extra typed fields
(compatibility, metadata, dependencies, examples, stats, ...). The skill BODY +
scripts are untouched, so behavior is identical — only the catalog metadata shrinks.
Surgical: only the tasks whose with-skill cells fail skill_posture."""
import glob, os, re
import yaml

SB = os.path.expanduser("~/skillsbench/tasks")
TASKS = ["dialogue-parser", "flink-query", "mario-coin-counting"]
n = 0
report = []
for t in TASKS:
    for sk in sorted(glob.glob(f"{SB}/{t}/environment/skills/*/SKILL.md")):
        raw = open(sk, encoding="utf-8").read()
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, re.S)
        if not m:
            report.append(f"  no-frontmatter: {sk}"); continue
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except Exception as e:
            report.append(f"  yaml-error {sk}: {e}"); continue
        body = m.group(2)
        name, desc = fm.get("name"), fm.get("description")
        if not name or not desc:
            report.append(f"  missing name/desc: {sk}"); continue
        before = len(fm)
        if before <= 2:
            report.append(f"  already-clean: {t}/{os.path.basename(os.path.dirname(sk))} ({before} fields)")
            continue
        new_fm = yaml.safe_dump({"name": name, "description": desc},
                                default_flow_style=False, sort_keys=False, allow_unicode=True)
        open(sk, "w", encoding="utf-8").write("---\n" + new_fm + "---\n" + body)
        n += 1
        report.append(f"  TRIMMED: {t}/{os.path.basename(os.path.dirname(sk))} ({before} -> 2 fields)")
print("\n".join(report))
print(f"\ntrimmed {n} SKILL.md to name+description")
