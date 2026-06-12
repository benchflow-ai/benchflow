#!/usr/bin/env python3
"""Publish reviewed-healthy SkillsBench max-effort cells to HF refs/pr/5 (v1.1 layout).

For each cell with review verdict 'pass' (review/<cell>.json) that has a local rollout
dir (runs/<cell_id>/<task>__<tid>/), upload the 5 canonical files into:
  submissions/skillsbench/v1.1/openhands-{with-skills|no-skills}__<slug>/<TS>-maxeffort/<task>__<tid>/
- normalizes config.json.include_task_skills to match the mode (with->true)
- generates timing.json from result.json.timing when the file is missing
- (re)writes the group metadata.yaml
- dedups vs PR5 by trial id; records published/<cell>.json
- optionally repairs existing PR5 cells from local rollouts with --repair-existing

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
from datetime import UTC, datetime
from pathlib import Path

REPO = "benchflow/skillsbench-leaderboard"
REPO_TYPE = "dataset"
PR_REF = "refs/pr/5"
V11 = "submissions/skillsbench/v1.1"
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
ENV_PATHS = (
    os.environ.get("BENCHFLOW_KEYS_ENV"),
    os.path.expanduser("~/Downloads/bingran-you/.env"),
    os.path.expanduser("~/Downloads/GitHub/bingran-you/.env"),
    os.path.expanduser("~/keys.env"),
    os.path.expanduser("~/.env"),
)
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
SECRET_NAME_RE = re.compile(
    r"(^API_KEY$|_API_KEY$|SECRET|BEARER|PASSWORD|CREDENTIAL|ACCESS_KEY|"
    r"(^|_)(AUTH_)?TOKEN$|^TOKEN$)",
    re.I,
)
SECRET_VAL_RE = re.compile(
    r"(AQ\.[A-Za-z0-9._-]{8,}|AIza[A-Za-z0-9._-]{10,}|sk-api-[A-Za-z0-9_-]{8,}|sk-[A-Za-z0-9-]{16,}|"
    r"ABSK[A-Za-z0-9+/=]{8,}|Bearer\s+[A-Za-z0-9._-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})"
)
TOKEN_COUNTER_KEYS = {
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cached_read_tokens",
    "cached_tokens",
    "cached_write_tokens",
    "completion_tokens",
    "input_tokens",
    "max_tokens",
    "n_cache_creation_tokens",
    "n_cache_read_tokens",
    "n_input_tokens",
    "n_output_tokens",
    "output_tokens",
    "prompt_tokens",
    "provider_total_tokens",
    "thought_tokens",
    "total_cached_tokens",
    "total_completion_tokens",
    "total_cost_usd",
    "total_input_tokens",
    "total_output_tokens",
    "total_prompt_tokens",
    "total_tokens",
}


def _is_token_counter(key: str, value) -> bool:
    """Keep numeric usage telemetry while still redacting credential tokens."""
    normalized = key.lower()
    if normalized not in TOKEN_COUNTER_KEYS and not normalized.endswith("_tokens"):
        return False
    return value is None or (
        isinstance(value, int | float) and not isinstance(value, bool)
    )


def _scrub(o):
    if isinstance(o, dict):
        return {
            k: (
                _scrub(v)
                if _is_token_counter(k, v) or not SECRET_NAME_RE.search(k)
                else "[REDACTED]"
            )
            for k, v in o.items()
        }
    if isinstance(o, list):
        return [_scrub(x) for x in o]
    if isinstance(o, str):
        return SECRET_VAL_RE.sub("[REDACTED]", o)
    return o


def safe_bytes(path, is_config=False, mode="without"):
    """Return scrubbed bytes for upload; raise if any secret survives."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
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
    checked = []
    for env_path in ENV_PATHS:
        if not env_path:
            continue
        p = os.path.expanduser(env_path)
        checked.append(p)
        try:
            with open(p) as fh:
                for line in fh:
                    m = re.match(r'^\s*(?:export\s+)?HUGGING_FACE_TOKEN\s*=\s*["\']?([^"\'\s]+)', line)
                    if m:
                        return m.group(1)
        except FileNotFoundError:
            continue
    raise SystemExit("no HUGGING_FACE_TOKEN in env or: " + ", ".join(checked))


def _load_only_cells(value: str) -> set[str]:
    if not value:
        return set()
    p = Path(value)
    if p.exists():
        cells: set[str] = set()
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                cells.add(line.split()[0])
            else:
                cells.add(str(rec.get("cell_id") or rec.get("cell") or "").strip())
        return {c for c in cells if c}
    return {c.strip() for c in value.split(",") if c.strip()}


def _load_repair_queue(value: str) -> dict[str, str]:
    if not value:
        return {}
    dests: dict[str, str] = {}
    for line in Path(value).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rec = json.loads(line)
        cell = str(rec.get("cell_id") or rec.get("cell") or "").strip()
        hf_path = str(rec.get("hf_path") or "").strip().rstrip("/")
        if cell and hf_path:
            dests[cell] = hf_path
    return dests


def configure_hf_token_env() -> str:
    token = os.environ.get("HF_TOKEN") or hf_token()
    os.environ["HF_TOKEN"] = token
    return token


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
  2026-06 to fill the 88-task x 3-trial SkillsBench grid for this config. Every trial
  passed benchflow-experiment-review (healthy ACP+LLM trajectory, parseable
  config/result/timing, positive timing.total, and either completed normally or
  reached a normal_timeout with complete verifier/reward/token evidence;
  with-skills trials show task_skills_loading=1, no-skills show 0).
"""


def _canonical_ready(rollout: Path) -> bool:
    return (
        rollout.is_dir()
        and (rollout / "result.json").exists()
        and (rollout / "config.json").exists()
        and (rollout / "trajectory" / "llm_trajectory.jsonl").exists()
        and (rollout / "trajectory" / "acp_trajectory.jsonl").exists()
    )


def _reviewed_rollout(rv: dict, runs_root: Path, task: str) -> Path | None:
    """Return the exact rollout reviewed by review_cell.py.

    Legacy review files did not carry ``rollout_dir``; for those, fall back to
    the old complete-rollout search so older already-reviewed cells remain
    publishable. New reviews must include ``rollout_dir`` so publish cannot
    accidentally upload a sibling attempt that was never reviewed.
    """
    reviewed = rv.get("rollout_dir")
    if reviewed:
        path = Path(reviewed)
        return path if _canonical_ready(path) else None
    return None


def _result_publishable(result: dict, review: dict) -> tuple[bool, str]:
    partial = bool(result.get("partial_trajectory"))
    summary = result.get("trajectory_summary") or {}
    summary_partial = bool(summary.get("partial_trajectory"))
    accepted_timeout = bool(review.get("accepted_normal_timeout")) and bool(
        review.get("timeout_complete_artifacts")
    )
    if (partial or summary_partial) and not accepted_timeout:
        return False, "partial trajectory lacks accepted_normal_timeout overlay"
    err = result.get("error")
    err_text = str(err).lower()
    if err and not (
        accepted_timeout and ("timeout" in err_text or "timed out" in err_text)
    ):
        return False, f"unaccepted run error: {err}"
    rew = (result.get("rewards") or {}).get("reward")
    try:
        reward_ok = rew is not None and 0.0 <= float(rew) <= 1.0
    except (TypeError, ValueError):
        reward_ok = False
    if not reward_ok:
        return False, "missing/invalid reward"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ts", default=datetime.now(UTC).strftime("%Y-%m-%d__%H-%M-%S") + "-maxeffort")
    ap.add_argument("--src-commit", default=os.environ.get("SB_SRC_COMMIT", "unknown"))
    ap.add_argument("--runs-root", default=str(ROOT / "runs"),
                    help="dir holding <cell>/<task>__<tid>/ rollouts (use 'jobs' when running on the VM)")
    ap.add_argument("--repair-existing", action="store_true",
                    help="overwrite canonical files for already-present trial ids from local reviewed rollouts")
    ap.add_argument("--repair-queue", default="",
                    help="JSONL with cell_id + hf_path; overwrite that HF path from a healthy local rerun")
    ap.add_argument("--only-cells", default="",
                    help="comma-separated cell ids or JSONL/text file with cell_id; limits publish/repair scope")
    a = ap.parse_args()
    runs_root = Path(a.runs_root)
    only_cells = _load_only_cells(a.only_cells)
    repair_dests = _load_repair_queue(a.repair_queue)

    token = configure_hf_token_env()
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token)

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
        with open(rf) as fh:
            rv = json.load(fh)
        if rv.get("verdict") != "pass":
            continue
        cell = rv["cell_id"]
        if only_cells and cell not in only_cells:
            continue
        m = re.match(r"(?P<model>.+?)__(?P<mode>with|without)__(?P<task>.+)__t(?P<slot>\d+)$", cell)
        if not m:
            continue
        model_key, mode, task = m["model"], m["mode"], m["task"]
        inner = _reviewed_rollout(rv, runs_root, task)
        if inner is None:
            print(f"  [skip] no reviewed complete rollout for {cell}")
            continue
        tid = inner.name.rsplit("__", 1)[1]
        if rv.get("trial_id") and str(rv["trial_id"]) != tid:
            print(f"  [skip] {cell}: reviewed trial_id={rv['trial_id']} != rollout trial_id={tid}")
            continue
        with open(inner / "result.json") as fh:
            rd = json.load(fh)
        ok, reason = _result_publishable(rd, rv)
        if not ok:
            print(f"  [skip] {cell}: {reason}")
            continue
        group = f"openhands-{HFMODE[mode]}__{MODELS[model_key]['slug']}"
        exist = existing(group)
        repair_dest = repair_dests.get(cell)
        if tid in exist and not a.repair_existing and not repair_dest:
            # already in PR5 (e.g. pushed earlier from another machine) — record its path
            # so the dashboard shows the HF link; don't re-upload.
            published.append({"cell_id": cell, "hf_path": exist[tid], "tid": tid, "already": True,
                              "updated_at": datetime.now(UTC).isoformat(timespec="seconds")})
            skipped += 1
            continue
        repairing = bool(repair_dest) or (tid in exist and a.repair_existing)
        dest = repair_dest or (exist[tid] if repairing else f"{V11}/{group}/{a.ts}/{task}__{tid}")
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
        published.append({"cell_id": cell, "hf_path": dest, "tid": tid, "repair": repairing,
                          "source_tid": tid, "accepted_normal_timeout": bool(rv.get("accepted_normal_timeout")),
                          "updated_at": datetime.now(UTC).isoformat(timespec="seconds")})

    # group metadata.yaml (once per group)
    for (model_key, mode), group in groups_seen.items():
        ops.append(CommitOperationAdd(path_in_repo=f"{V11}/{group}/metadata.yaml",
                                      path_or_fileobj=io.BytesIO(metadata_yaml(model_key, mode, a.src_commit).encode())))

    new = [p for p in published if not p.get("already") and not p.get("repair")]
    repaired = [p for p in published if p.get("repair")]
    print(f"to publish: {len(new)} new + {len(repaired)} repairs + {len(published) - len(new) - len(repaired)} already-in-PR5 recorded ({len(ops)} files); dedup {skipped}; groups={list(groups_seen.values())}")
    if a.dry_run:
        print("DRY RUN — nothing pushed")
        return 0
    if ops:
        CH = 400
        for i in range(0, len(ops), CH):
            api.create_commit(repo_id=REPO, repo_type=REPO_TYPE, revision=PR_REF, operations=ops[i:i + CH],
                              commit_message=f"max-effort fill: {len(new)} cells, repair {len(repaired)} (batch {i // CH + 1})")
            print(f"  committed batch {i // CH + 1}: {len(ops[i:i+CH])} files")
    # Always (re)write publish records so the dashboard links every cell that IS in PR5.
    # This must happen after create_commit, otherwise a failed HF upload creates
    # local dashboard credit for a cell that is not actually present in PR5.
    (ROOT / "published").mkdir(exist_ok=True)
    for p in published:
        with open(ROOT / "published" / f"{p['cell_id']}.json", "w") as fh:
            json.dump(p, fh, indent=2)
    print(f"[done] {len(new)} uploaded, {len(repaired)} repaired, {len(published)} recorded -> {PR_REF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
