#!/usr/bin/env python3
"""Publish reviewed-healthy SkillsBench max-effort cells to HF refs/pr/5 (v1.1 layout).

For each cell with review verdict 'pass' (review/<cell>.json) that has a local rollout
dir (runs/<cell_id>/<task>__<tid>/), upload the 5 canonical files into:
  submissions/skillsbench/v1.1/openhands-{with-skills|no-skills}__<slug>/<TS>-maxeffort/<task>__<tid>/
- normalizes config.json.include_task_skills to match the mode (with->true)
- generates timing.json from result.json.timing when the file is missing
- (re)writes the group metadata.yaml
- dedups vs PR5 by trial id; records published/<cell>.json

Usage: publish.py [--dry-run] [--ts 2026-06-04__hhmm] [--src-commit <sha>]
Read keys + paths default to this toolkit dir. Adapted from workspace/scripts/generic_push.py.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = "benchflow/skillsbench-leaderboard"
REPO_TYPE = "dataset"
PR_REF = "refs/pr/5"
V11 = "submissions/skillsbench/v1.1"
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.expanduser("~/Downloads/GitHub/bingran-you/.env")
CANON = ["config.json", "result.json", "timing.json",
         "trajectory/acp_trajectory.jsonl", "trajectory/llm_trajectory.jsonl"]

MODELS = {
    "opus-4.8": {"slug": "aws-bedrock-us.anthropic.claude-opus-4-8", "model_name": "aws-bedrock/us.anthropic.claude-opus-4-8",
                 "provider": "aws-bedrock", "display": "Claude Opus 4.8 (max thinking)", "org": "Anthropic"},
    "gemini-3.5-flash": {"slug": "gemini-3.5-flash", "model_name": "gemini-3.5-flash",
                 "provider": "google", "display": "Gemini 3.5 Flash (high)", "org": "Google"},
    "minimax-m3": {"slug": "minimax-MiniMax-M3", "model_name": "minimax/MiniMax-M3",
                 "provider": "minimax", "display": "MiniMax M3 (thinking/max)", "org": "MiniMax"},
}
HFMODE = {"with": "with-skills", "without": "no-skills"}

# --- secret hygiene: nothing with a credential ever leaves for the public PR ---
# benchflow already scrubs config/result/trajectories, but the publisher guarantees
# it too: drop secret-named keys, redact secret-shaped values, and ABORT a cell if
# any secret pattern survives in any file about to be uploaded.
SECRET_NAME_RE = re.compile(r"(_API_KEY$|^API_KEY$|SECRET|TOKEN|BEARER|PASSWORD|CREDENTIAL|ACCESS_KEY)", re.I)
SECRET_VAL_RE = re.compile(
    r"(AQ\.[A-Za-z0-9._-]{8,}|AIza[A-Za-z0-9._-]{10,}|sk-api-[A-Za-z0-9_-]{8,}|sk-[A-Za-z0-9-]{16,}|"
    r"ABSK[A-Za-z0-9+/=]{8,}|Bearer\s+[A-Za-z0-9._-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})"
)


def _scrub(o):
    if isinstance(o, dict):
        return {k: ("[REDACTED]" if SECRET_NAME_RE.search(k) else _scrub(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [_scrub(x) for x in o]
    if isinstance(o, str):
        return SECRET_VAL_RE.sub("[REDACTED]", o)
    return o


def safe_bytes(path, is_config=False, mode="without"):
    """Return scrubbed bytes for upload; raise if any secret survives."""
    raw = open(path, encoding="utf-8", errors="replace").read()
    if str(path).endswith(".jsonl"):
        text = SECRET_VAL_RE.sub("[REDACTED]", raw)
    else:
        obj = json.loads(raw)
        if is_config:
            obj["include_task_skills"] = (mode == "with")
        text = json.dumps(_scrub(obj), indent=2)
    if SECRET_VAL_RE.search(text):
        raise ValueError(f"secret pattern survived scrub in {path}")
    return text.encode()


def hf_token() -> str:
    if os.environ.get("HUGGING_FACE_TOKEN"):
        return os.environ["HUGGING_FACE_TOKEN"]
    for p in (ENV_PATH, os.path.expanduser("~/keys.env"), os.path.expanduser("~/.env")):
        try:
            for line in open(p):
                m = re.match(r'^\s*(?:export\s+)?HUGGING_FACE_TOKEN\s*=\s*["\']?([^"\'\s]+)', line)
                if m:
                    return m.group(1)
        except FileNotFoundError:
            continue
    raise SystemExit("no HUGGING_FACE_TOKEN (env, .env, or ~/keys.env)")


def metadata_yaml(model_key: str, mode: str, src_commit: str) -> str:
    m = MODELS[model_key]
    skills = "true" if mode == "with" else "false"
    origin = '"benchflow-ai/skillsbench task environment/skills"' if mode == "with" else "null"
    return f"""agent_url: https://github.com/All-Hands-AI/OpenHands
agent_display_name: "OpenHands ({HFMODE[mode]})"
agent_org_display_name: "All Hands AI"
agent_version: "openhands-sdk"

skills_used: {skills}
skills_dir_origin: {origin}

models:
  - model_name: {m["model_name"]}
    model_provider: {m["provider"]}
    model_display_name: "{m["display"]}"
    model_org_display_name: "{m["org"]}"

dataset:
  name: skillsbench
  version: v1.1
  source_repo: benchflow-ai/skillsbench
  source_commit: {src_commit}

contact: bingran@benchflow.ai
notes: |
  OpenHands ({HFMODE[mode]}) + {m["display"]} at maximum reasoning effort, generated
  2026-06 to fill the 91-task x 3-trial SkillsBench grid for this config. Every trial
  passed benchflow-experiment-review (healthy ACP+LLM trajectory, parseable
  config/result/timing, positive timing.total, no error, non-partial; with-skills
  trials show task_skills_loading=1, no-skills show 0).
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ts", default=datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S") + "-maxeffort")
    ap.add_argument("--src-commit", default=os.environ.get("SB_SRC_COMMIT", "unknown"))
    ap.add_argument("--runs-root", default=str(ROOT / "runs"),
                    help="dir holding <cell>/<task>__<tid>/ rollouts (use 'jobs' when running on the VM)")
    a = ap.parse_args()
    runs_root = Path(a.runs_root)

    os.environ.setdefault("HF_TOKEN", hf_token())
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=os.environ["HF_TOKEN"])

    # existing cells per group: {trial_id: cell_dir_path} for dedup AND path recovery, cached.
    _exist_cache: dict = {}

    def existing(group: str) -> dict:
        if group not in _exist_cache:
            ids: dict = {}
            try:
                for it in api.list_repo_tree(REPO, path_in_repo=f"{V11}/{group}", repo_type=REPO_TYPE, revision=PR_REF, recursive=True):
                    if it.path.endswith("/result.json"):
                        cell_dir = it.path.rsplit("/", 1)[0]
                        leaf = cell_dir.split("/")[-1]
                        if "__" in leaf:
                            ids[leaf.rsplit("__", 1)[1]] = cell_dir
            except Exception:
                pass
            _exist_cache[group] = ids
        return _exist_cache[group]

    review_files = glob.glob(str(ROOT / "review" / "*.json"))
    ops = []
    published = []
    groups_seen = {}
    skipped = 0
    for rf in review_files:
        rv = json.load(open(rf))
        if rv.get("verdict") != "pass":
            continue
        cell = rv["cell_id"]
        m = re.match(r"(?P<model>.+?)__(?P<mode>with|without)__(?P<task>.+)__t(?P<slot>\d+)$", cell)
        if not m:
            continue
        model_key, mode, task = m["model"], m["mode"], m["task"]
        # A cell can have several rollout dirs: failed creation attempts (result.json but
        # no trajectory) plus the one successful run. Pick a COMPLETE rollout (result.json
        # + llm_trajectory), not by sort position — else we grab an empty attempt and skip.
        roll = [d for d in glob.glob(str(runs_root / cell / "**" / f"{task}__*"), recursive=True)
                if Path(d).is_dir() and (Path(d) / "result.json").exists()
                and (Path(d) / "trajectory" / "llm_trajectory.jsonl").exists()]
        if not roll:
            print(f"  [skip] no complete rollout (result.json+llm_trajectory) for {cell}")
            continue
        inner = Path(sorted(roll)[-1])
        tid = inner.name.rsplit("__", 1)[1]
        group = f"openhands-{HFMODE[mode]}__{MODELS[model_key]['slug']}"
        exist = existing(group)
        if tid in exist:
            # already in PR5 (e.g. pushed earlier from another machine) — record its path
            # so the dashboard shows the HF link; don't re-upload.
            published.append({"cell_id": cell, "hf_path": exist[tid], "tid": tid, "already": True,
                              "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            skipped += 1
            continue
        dest = f"{V11}/{group}/{a.ts}/{task}__{tid}"
        rd = json.load(open(inner / "result.json"))
        cell_ops = []
        try:
            for f in CANON:
                src = inner / f
                if src.exists():
                    cell_ops.append(CommitOperationAdd(
                        path_in_repo=f"{dest}/{f}",
                        path_or_fileobj=io.BytesIO(safe_bytes(src, is_config=(f == "config.json"), mode=mode))))
                elif f == "timing.json" and isinstance(rd.get("timing"), dict) and rd["timing"]:
                    cell_ops.append(CommitOperationAdd(
                        path_in_repo=f"{dest}/{f}",
                        path_or_fileobj=io.BytesIO(json.dumps(_scrub(rd["timing"]), indent=2).encode())))
                else:
                    raise ValueError(f"missing canonical file {f}")
        except ValueError as e:
            print(f"  [skip] {cell}: {e}")
            continue
        ops.extend(cell_ops)
        groups_seen.setdefault((model_key, mode), group)
        published.append({"cell_id": cell, "hf_path": dest, "tid": tid,
                          "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    # group metadata.yaml (once per group)
    for (model_key, mode), group in groups_seen.items():
        ops.append(CommitOperationAdd(path_in_repo=f"{V11}/{group}/metadata.yaml",
                                      path_or_fileobj=io.BytesIO(metadata_yaml(model_key, mode, a.src_commit).encode())))

    new = [p for p in published if not p.get("already")]
    print(f"to publish: {len(new)} new + {len(published) - len(new)} already-in-PR5 recorded ({len(ops)} files); dedup {skipped}; groups={list(groups_seen.values())}")
    if a.dry_run:
        print("DRY RUN — nothing pushed")
        return 0
    # Always (re)write publish records so the dashboard links every cell that IS in PR5.
    (ROOT / "published").mkdir(exist_ok=True)
    for p in published:
        json.dump(p, open(ROOT / "published" / f"{p['cell_id']}.json", "w"), indent=2)
    if ops:
        CH = 400
        for i in range(0, len(ops), CH):
            api.create_commit(repo_id=REPO, repo_type=REPO_TYPE, revision=PR_REF, operations=ops[i:i + CH],
                              commit_message=f"max-effort fill: {len(new)} cells (batch {i // CH + 1})")
            print(f"  committed batch {i // CH + 1}: {len(ops[i:i+CH])} files")
    print(f"[done] {len(new)} uploaded, {len(published)} recorded -> {PR_REF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
