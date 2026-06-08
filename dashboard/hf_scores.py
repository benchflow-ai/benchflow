"""HuggingFace PR score aggregation for the live Experiments dashboard.

The experiment fill ledger answers "what is still running?". The public
leaderboard-style score answers "among scored HF submissions, how often did
each harness/model pass?". Keep those as separate surfaces so operational fill
state cannot accidentally become final benchmark score.
"""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

REPO = "benchflow/skillsbench-leaderboard"
REPO_TYPE = "dataset"
PRS = (2, 3, 4, 5)
ANALYSIS_SUMMARY = "analysis/canonical_harness_model_skill_mode_status_cn6only_capped5.csv"
_DASH = Path(__file__).resolve().parent
CACHE_PATH = _DASH / "hf_scoreboard_cache.json"
DEFAULT_TTL_SECONDS = 10 * 60

_MODE_ALIASES = {
    "with": "with-skills",
    "with-skill": "with-skills",
    "with-skills": "with-skills",
    "skills": "with-skills",
    "skill": "with-skills",
    "no": "without-skills",
    "none": "without-skills",
    "no-skill": "without-skills",
    "no-skills": "without-skills",
    "noskills": "without-skills",
    "without": "without-skills",
    "without-skill": "without-skills",
    "without-skills": "without-skills",
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _to_int(value: Any) -> int:
    try:
        text = str(value or "").strip()
        if not text:
            return 0
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _norm_mode(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    return _MODE_ALIASES.get(text, text)


def _norm_harness(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if not text:
        return "unknown"
    for suffix in ("-without-skills", "-with-skills", "-no-skills", "-no-skill"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _harness_label(value: str) -> str:
    labels = {
        "claude-code": "Claude Code",
        "gemini-cli": "Gemini CLI",
        "openhands": "OpenHands",
        "codex": "Codex",
    }
    return labels.get(value, value.replace("-", " ").title())


def _model_slug(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    text = text.split("/")[-1]
    text = text.removeprefix("us.")
    text = text.removeprefix("anthropic.")
    text = text.replace("_", "-")
    return text


def _model_label(value: str) -> str:
    slug = _model_slug(value)
    lower = slug.lower()
    replacements = [
        (r"^claude-opus-4-8.*", "Opus 4.8"),
        (r"^claude-opus-4-7.*", "Opus 4.7"),
        (r"^gemini-3\.5-flash.*", "Gemini 3.5 Flash"),
        (r"^gemini-3\.1-pro.*", "Gemini 3.1 Pro"),
        (r"^gpt-5\.5.*", "GPT-5.5"),
        (r"^minimax-m3.*", "MiniMax M3"),
    ]
    for pattern, label in replacements:
        if re.match(pattern, lower):
            return label
    parts = re.split(r"[-.]+", slug)
    return " ".join(p.upper() if p.lower() in {"gpt", "cli"} else p.capitalize() for p in parts if p)


def _direct_path_info(path: str) -> tuple[str, str, str, str] | None:
    parts = path.split("/")
    try:
        sub_idx = parts.index("submissions")
    except ValueError:
        return None
    if len(parts) < sub_idx + 6 or parts[sub_idx + 1] != "skillsbench":
        return None
    family = parts[sub_idx + 3]
    if "__" not in family:
        return None
    harness_part, model_part = family.split("__", 1)
    harness = _norm_harness(harness_part)
    mode = "with-skills"
    if re.search(r"(^|-)(no|without)-skills?$", harness_part):
        mode = "without-skills"
    elif re.search(r"(^|-)with-skills?$", harness_part):
        mode = "with-skills"
    task_dir = parts[-2] if len(parts) >= 2 else ""
    task = task_dir.split("__", 1)[0]
    return harness, _model_slug(model_part), mode, task


def _result_reward(payload: dict[str, Any]) -> float | None:
    for key in ("reward", "score"):
        reward = _to_float(payload.get(key))
        if reward is not None:
            return reward
    rewards = payload.get("rewards")
    if isinstance(rewards, dict):
        for key in ("reward", "score", "passed"):
            reward = _to_float(rewards.get(key))
            if reward is not None:
                return reward
    passed = payload.get("passed")
    if isinstance(passed, bool):
        return 1.0 if passed else 0.0
    return _to_float(passed)


def _add_bucket(
    buckets: dict[tuple[str, str, str], dict[str, Any]],
    *,
    pr: int,
    harness: str,
    model: str,
    mode: str,
    passed: int,
    failed: int,
    errored: int = 0,
    tasks_seen: int = 0,
) -> None:
    mode = _norm_mode(mode)
    if mode not in {"with-skills", "without-skills"}:
        return
    total = passed + failed + errored
    if total <= 0:
        return
    key = (_norm_harness(harness), _model_slug(model), mode)
    bucket = buckets[key]
    bucket["harness"] = key[0]
    bucket["model"] = key[1]
    bucket["mode"] = key[2]
    bucket["passed"] += passed
    bucket["failed"] += failed
    bucket["errored"] += errored
    bucket["total"] += total
    bucket["tasks_seen"] = max(bucket["tasks_seen"], tasks_seen)
    bucket["prs"].add(pr)


def _read_analysis_prs(buckets: dict[tuple[str, str, str], dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"huggingface_hub unavailable: {exc}"]

    for pr in (2, 3):
        try:
            local = hf_hub_download(
                REPO,
                ANALYSIS_SUMMARY,
                repo_type=REPO_TYPE,
                revision=f"refs/pr/{pr}",
            )
        except Exception as exc:
            warnings.append(f"PR{pr} analysis unavailable: {exc}")
            continue
        with open(local, newline="") as fh:
            for row in csv.DictReader(fh):
                _add_bucket(
                    buckets,
                    pr=pr,
                    harness=row.get("harness") or "",
                    model=row.get("model") or "",
                    mode=row.get("skill_mode") or "",
                    passed=_to_int(row.get("pass")),
                    failed=_to_int(row.get("fail")),
                    errored=_to_int(row.get("usable_error")),
                    tasks_seen=_to_int(row.get("tasks_seen")),
                )
    return warnings


def _hf_raw_url(revision: str, path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{REPO}/resolve/"
        f"{quote(revision, safe='')}/{quote(path, safe='/._-')}"
    )


def _fetch_result_json(revision: str, path: str) -> tuple[str, dict[str, Any] | None, str | None]:
    headers = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    timeout = _to_float(os.environ.get("EXPERIMENTS_HF_SCORE_HTTP_TIMEOUT")) or 8.0
    try:
        resp = requests.get(_hf_raw_url(revision, path), headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return path, None, f"{path}: HTTP {resp.status_code}"
        payload = resp.json()
    except Exception as exc:
        return path, None, f"{path}: {exc}"
    if not isinstance(payload, dict):
        return path, None, f"{path}: result JSON is not an object"
    return path, payload, None


def _read_direct_prs(buckets: dict[tuple[str, str, str], dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"huggingface_hub unavailable: {exc}"]

    api = HfApi()
    workers = _to_int(os.environ.get("EXPERIMENTS_HF_SCORE_WORKERS")) or 20
    for pr in (4, 5):
        revision = f"refs/pr/{pr}"
        try:
            files = api.list_repo_files(REPO, repo_type=REPO_TYPE, revision=revision)
        except Exception as exc:
            warnings.append(f"PR{pr} file list unavailable: {exc}")
            continue
        result_paths = [
            path
            for path in files
            if path.endswith("/result.json")
            and path.startswith("submissions/skillsbench/")
        ]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(_fetch_result_json, revision, path): path
                for path in result_paths
            }
            for future in as_completed(futures):
                path, payload, warning = future.result()
                info = _direct_path_info(path)
                if info is None:
                    continue
                if warning:
                    warnings.append(f"PR{pr} result unavailable: {warning}")
                    continue
                if payload is None:
                    continue
                harness, path_model, path_mode, _task = info
                reward = _result_reward(payload)
                if reward is None:
                    continue
                model = _model_slug(
                    payload.get("model") or payload.get("model_slug") or path_model
                )
                mode = _norm_mode(
                    payload.get("skill_mode") or payload.get("mode") or path_mode
                )
                _add_bucket(
                    buckets,
                    pr=pr,
                    harness=payload.get("harness") or payload.get("agent") or harness,
                    model=model,
                    mode=mode,
                    passed=1 if reward >= 1.0 else 0,
                    failed=0 if reward >= 1.0 else 1,
                    tasks_seen=1,
                )
    return warnings


def _rows_for_mode(buckets: dict[tuple[str, str, str], dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (harness, model, bucket_mode), bucket in buckets.items():
        if bucket_mode != mode:
            continue
        total = int(bucket["total"])
        rate = (float(bucket["passed"]) / total) if total else 0.0
        rows.append(
            {
                "harness": harness,
                "harness_label": _harness_label(harness),
                "model": model,
                "model_label": _model_label(model),
                "label": f"{_harness_label(harness)} {_model_label(model)}",
                "mode": bucket_mode,
                "passed": int(bucket["passed"]),
                "failed": int(bucket["failed"]),
                "errored": int(bucket["errored"]),
                "total": total,
                "tasks_seen": int(bucket["tasks_seen"]),
                "pass_rate": rate,
                "pass_rate_pct": round(rate * 100.0, 1),
                "prs": sorted(bucket["prs"]),
            }
        )
    rows.sort(key=lambda row: (-row["pass_rate"], -row["total"], row["label"]))
    return rows


def _normalized_gain(rows_by_mode: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for mode, rows in rows_by_mode.items():
        for row in rows:
            by_key[(row["harness"], row["model"])][mode] = row
    gains: list[dict[str, Any]] = []
    for (_harness, _model), modes in by_key.items():
        with_row = modes.get("with-skills")
        without_row = modes.get("without-skills")
        if not with_row or not without_row:
            continue
        baseline = float(without_row["pass_rate"])
        with_rate = float(with_row["pass_rate"])
        gain = 0.0 if baseline >= 1.0 else (with_rate - baseline) / (1.0 - baseline)
        gains.append(
            {
                **with_row,
                "mode": "normalized-gain",
                "with_pass_rate": with_rate,
                "without_pass_rate": baseline,
                "gain": gain,
                "gain_pct": round(gain * 100.0, 1),
                "with_total": with_row["total"],
                "without_total": without_row["total"],
                "prs": sorted(set(with_row["prs"]) | set(without_row["prs"])),
            }
        )
    gains.sort(key=lambda row: (-row["gain"], -row["with_total"], row["label"]))
    return gains


def build_scoreboard() -> dict[str, Any]:
    """Build a leaderboard-style score summary from HF PR2/3/4/5 data."""
    buckets: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "passed": 0,
            "failed": 0,
            "errored": 0,
            "total": 0,
            "tasks_seen": 0,
            "prs": set(),
        }
    )
    warnings = []
    warnings.extend(_read_analysis_prs(buckets))
    if os.environ.get("EXPERIMENTS_HF_SCORE_SKIP_DIRECT") == "1":
        warnings.append("PR4/PR5 direct result scan skipped by EXPERIMENTS_HF_SCORE_SKIP_DIRECT")
    else:
        warnings.extend(_read_direct_prs(buckets))
    by_mode = {
        "with-skills": _rows_for_mode(buckets, "with-skills"),
        "without-skills": _rows_for_mode(buckets, "without-skills"),
    }
    by_mode["normalized-gain"] = _normalized_gain(by_mode)
    scored_trials = sum(bucket["total"] for bucket in buckets.values())
    return {
        "as_of": _now_iso(),
        "source": "HuggingFace PR2/PR3 analysis + PR4/PR5 result.json",
        "repo": REPO,
        "refs": [f"refs/pr/{pr}" for pr in PRS],
        "scored_trials": int(scored_trials),
        "groups": len(buckets),
        "by_mode": by_mode,
        "warnings": warnings[:25],
        "warning_count": len(warnings),
    }


def _load_cache(path: Path = CACHE_PATH) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_cache(data: dict[str, Any], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as fh:
        fh.write(payload)
        tmp = Path(fh.name)
    tmp.replace(path)


def snapshot(cache_path: str | Path | None = None) -> dict[str, Any]:
    """Return cached HF score summary.

    HF PR4/5 contain thousands of small JSON files, so live HTTP requests should
    not refresh them inline. Run this module as a background cache warmer; the
    dashboard reads the cache and keeps serving even when HF is slow.
    """
    path = Path(cache_path) if cache_path is not None else Path(
        os.environ.get("EXPERIMENTS_HF_SCORE_CACHE") or CACHE_PATH
    )
    ttl = _to_int(os.environ.get("EXPERIMENTS_HF_SCORE_TTL_SECONDS"))
    if ttl <= 0:
        ttl = DEFAULT_TTL_SECONDS
    cache = _load_cache(path)
    if cache is not None:
        age = time.time() - path.stat().st_mtime
        if age < ttl:
            return {**cache, "cached": True, "cache_age_s": round(age, 1)}
        if os.environ.get("EXPERIMENTS_HF_SCORE_REFRESH_INLINE") != "1":
            return {
                **cache,
                "cached": True,
                "cache_age_s": round(age, 1),
                "stale": True,
                "error": "HF score cache is stale; background refresh is pending",
            }
    elif os.environ.get("EXPERIMENTS_HF_SCORE_REFRESH_INLINE") != "1":
        return {
            "as_of": _now_iso(),
            "source": "HuggingFace PR2/PR3/PR4/PR5",
            "repo": REPO,
            "refs": [f"refs/pr/{pr}" for pr in PRS],
            "scored_trials": 0,
            "groups": 0,
            "by_mode": {"with-skills": [], "without-skills": [], "normalized-gain": []},
            "warnings": [],
            "warning_count": 0,
            "cached": False,
            "cache_age_s": None,
            "error": "HF score cache is not warm yet; background refresh is pending",
        }
    try:
        fresh = build_scoreboard()
        _write_cache(fresh, path)
        return {**fresh, "cached": False, "cache_age_s": 0}
    except Exception as exc:
        if cache is not None:
            age = time.time() - path.stat().st_mtime
            return {
                **cache,
                "cached": True,
                "cache_age_s": round(age, 1),
                "stale": True,
                "error": f"HF score refresh failed: {exc}",
            }
        return {
            "as_of": _now_iso(),
            "source": "HuggingFace PR2/PR3/PR4/PR5",
            "repo": REPO,
            "refs": [f"refs/pr/{pr}" for pr in PRS],
            "scored_trials": 0,
            "groups": 0,
            "by_mode": {"with-skills": [], "without-skills": [], "normalized-gain": []},
            "warnings": [],
            "warning_count": 0,
            "error": f"HF score refresh failed: {exc}",
        }


def main() -> None:
    fresh = build_scoreboard()
    _write_cache(fresh)
    print(
        json.dumps(
            {
                "as_of": fresh.get("as_of"),
                "scored_trials": fresh.get("scored_trials"),
                "groups": fresh.get("groups"),
                "warning_count": fresh.get("warning_count"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
