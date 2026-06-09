#!/usr/bin/env python3
"""Clean SkillsBench -> task.md adapter (review draft).

Wraps benchflow's lossless `migrate_task_to_task_md`, then post-processes the
generated package so the task.md is minimal and review-friendly:

  1. Drop empty-default scaffolding the canonical serializer emits but the task
     never uses: `artifacts: []`, `reward: {}`, every empty `env: {}`
     (verifier/oracle/environment/agent), `mcp_servers: []`, empty `oracle`.
  2. Drop dormant llm-judge-only blocks when `verifier.type == test-script`:
     `verifier.judge` (+ its dangling `rubric_path`) and `verifier.memory`.
  3. Drop the deprecated, redundant `environment.allow_internet`
     (`network_mode` is authoritative; the schema reconciles them anyway).
  4. Fix the real runtime bug: `verifier/test.sh` references `/tests/...` but
     the native verifier dir mounts at `/verifier` -> rewrite `/tests/`->`/verifier/`.
  5. Reorder frontmatter to: schema_version, metadata, environment, agent, verifier.
  6. (optional, --offline-no-network) set `network_mode: no-network` when the
     source declares no network -- least privilege for offline tasks. OFF by
     default to stay byte-faithful to the source.

Safety: the cleanup is PROVABLY lossless -- the script asserts the pruned
task.md reloads to the *identical* TaskConfig as benchflow's faithful migration
(every dropped key reloads to the same default). If that assertion ever fails,
the script errors instead of writing a divergent file.

    python adapt_clean.py --src ../../../skillsbench/tasks/ada-bathroom-plan-repair --out ./adapted-clean/ada-bathroom-plan-repair
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from benchflow._utils.task_authoring import check_task, migrate_task_to_task_md
from benchflow.task import TaskDocument

# root keys dropped when empty; ordering of what we KEEP
_FRONTMATTER_ORDER = ["schema_version", "metadata", "environment", "agent", "verifier"]
_DROP_IF_EMPTY_ROOT = ["artifacts", "reward", "oracle", "solution", "steps",
                       "source", "task", "multi_step_reward_strategy", "sandbox"]
_VERIFIER_DORMANT = ["judge", "memory"]          # only used when type == llm-judge
_ENV_DEPRECATED = ["allow_internet"]              # redundant with network_mode


def _is_empty(v) -> bool:
    return v is None or v == {} or v == [] or v == ""


def _drop_empty(obj):
    """Recursively drop keys whose value is an empty container / None / ''."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            v = _drop_empty(v)
            if not _is_empty(v):
                out[k] = v
        return out
    if isinstance(obj, list):
        return [_drop_empty(x) for x in obj]
    return obj


def _split_frontmatter(text: str):
    assert text.startswith("---"), "task.md must start with YAML frontmatter"
    # Line-based split (mirror benchflow parser) so a --- inside a YAML scalar
    # cannot truncate the frontmatter.
    lines = text.splitlines(keepends=True)
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    fm = "".join(lines[1:end])
    body = "".join(lines[end + 1:]).lstrip("\n")
    return yaml.safe_load(fm), body


def _clean_frontmatter(cfg: dict, *, offline_no_network: bool) -> dict:
    cfg = _drop_empty(cfg)

    ver = cfg.get("verifier") or {}
    if ver.get("type", "test-script") == "test-script":
        for k in _VERIFIER_DORMANT:
            ver.pop(k, None)
        if ver.get("service") == "main":      # serializer default
            ver.pop("service", None)
    cfg["verifier"] = ver

    env = cfg.get("environment") or {}
    for k in _ENV_DEPRECATED:
        env.pop(k, None)
    if offline_no_network and "network_mode" in env:
        # only harden if it's the permissive default; preserves explicit choices
        if env["network_mode"] == "public":
            env["network_mode"] = "no-network"
    cfg["environment"] = env

    # drop any now-empty root keys created by the above
    cfg = {k: v for k, v in cfg.items() if not _is_empty(v)}

    # stable, readable order: known keys first, then any leftovers
    ordered = {k: cfg[k] for k in _FRONTMATTER_ORDER if k in cfg}
    for k, v in cfg.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _fix_verifier_paths(out: Path) -> bool:
    ts = out / "verifier" / "test.sh"
    if not ts.exists():
        return False
    txt = ts.read_text()
    fixed = txt.replace("/tests/", "/verifier/")
    if fixed != txt:
        ts.write_text(fixed)
        return True
    return False


def adapt_one(src: Path, out: Path, *, offline_no_network: bool) -> list[str]:
    notes: list[str] = []
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)

    # 1. faithful, lossless migration (legacy -> task.md layout, in place)
    migrate_task_to_task_md(out, overwrite=True, remove_legacy=True)
    faithful_cfg = TaskDocument.from_path(out / "task.md").config.model_dump()

    # 2. fix the /tests -> /verifier runtime bug
    if _fix_verifier_paths(out):
        notes.append("fixed verifier/test.sh: /tests/ -> /verifier/")

    # 3. clean the frontmatter
    text = (out / "task.md").read_text()
    cfg, body = _split_frontmatter(text)
    cleaned = _clean_frontmatter(cfg, offline_no_network=offline_no_network)
    new_text = "---\n" + yaml.safe_dump(cleaned, sort_keys=False, width=100, allow_unicode=True) + "---\n\n" + body
    (out / "task.md").write_text(new_text)

    # 4. PROVE lossless: cleaned file must reload to the identical TaskConfig
    reloaded = TaskDocument.from_path(out / "task.md").config.model_dump()
    if offline_no_network:
        # network hardening is an intentional semantic change; ignore those fields
        for d in (faithful_cfg, reloaded):
            d.get("sandbox", {}).pop("network_mode", None)
            d.get("sandbox", {}).pop("allow_internet", None)
    if reloaded != faithful_cfg:
        diff = {k: (faithful_cfg.get(k), reloaded.get(k))
                for k in set(faithful_cfg) | set(reloaded) if faithful_cfg.get(k) != reloaded.get(k)}
        raise SystemExit(f"NON-LOSSLESS clean for {src.name}; differing fields: {diff}")
    notes.append("cleaned frontmatter is lossless (reloads to identical TaskConfig)")

    # 5. structural check
    issues = check_task(out)
    notes.append(f"check_task: {'OK' if not issues else issues}")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="skillsbench/tasks/<name>")
    ap.add_argument("--out", required=True, type=Path, help="output dir for the cleaned package")
    ap.add_argument("--offline-no-network", action="store_true",
                    help="set network_mode: no-network when source has the permissive default")
    a = ap.parse_args()
    notes = adapt_one(a.src, a.out, offline_no_network=a.offline_no_network)
    print(f"OK {a.src.name} -> {a.out}")
    for n in notes:
        print("  -", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
