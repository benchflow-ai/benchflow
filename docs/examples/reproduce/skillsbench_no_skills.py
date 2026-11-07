"""Recompute SkillsBench's leaderboard entry for OpenHands on DeepSeek V4 Flash without skills (27.5 +- 7.4).

Downloads the published trials' small files from Hugging Face at a fixed revision, picks trials by the rule that
reproduces all 87 published per-task scores, and prints the score with the leaderboard's interval, then the same
with duplicate copies removed.

The rule: every openhands-no-skills__*v4-flash* folder except ...-ds-v4-flash-src-411dc68; per task, the 3 latest
trials (by started_at) that have a reward and no error; empty slots count 0; the score is the mean over tasks of
each task's mean reward, x100; the interval is 1.96 x the sample stdev of the per-task means / sqrt(tasks).

usage: python3 skillsbench_no_skills.py skillsbench-v1.1-tasks.txt [download dir]   (needs huggingface_hub)
"""
import glob
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict

from huggingface_hub import snapshot_download

REPO, REVISION = "benchflow/skillsbench-leaderboard", "f104580363a9642563593c475620196ecd36687d"
BASE = "submissions/skillsbench/v1.1/openhands-no-skills__*v4-flash*"
LEFT_OUT = "openhands-no-skills__deepseek-ds-v4-flash-src-411dc68"
roster = sorted(open(sys.argv[1]).read().split())
out = sys.argv[2] if len(sys.argv) > 2 else "hf"
snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=out,
                  allow_patterns=[f"{BASE}/*/*/result.json"])

trials = []
for path in sorted(glob.glob(os.path.join(out, BASE, "*", "*", "result.json"))):
    folder = path.split(os.sep)[-4]
    r = json.load(open(path))
    trial = os.path.basename(os.path.dirname(path))
    task = r.get("task_name") or trial.split("__")[0]
    reward = (r.get("rewards") or {}).get("reward")
    ok = reward is not None and not r.get("error") and not r.get("partial_trajectory")
    # The same trial copied into two folders has the same task, start, end, reward and tool calls.
    key = json.dumps([task, r.get("started_at"), r.get("finished_at"), r.get("rewards"), r.get("n_tool_calls"),
                      r.get("error")], default=str)
    trials.append({"folder": folder, "task": task, "reward": reward, "ok": ok, "started_at": r.get("started_at") or "",
                   "copy": hashlib.sha256(key.encode()).hexdigest()})
print(f"{len(trials)} trials in {len({t['folder'] for t in trials})} folders at {REVISION[:8]}")


def score(rows):
    pools = defaultdict(list)
    for t in rows:
        if t["folder"] != LEFT_OUT and t["task"] in roster and t["ok"]:
            pools[t["task"]].append(t)
    means = []
    for task in roster:
        latest = sorted(pools[task], key=lambda t: t["started_at"], reverse=True)[:3]
        rewards = [t["reward"] for t in latest] + [0.0] * (3 - len(latest))
        means.append(sum(rewards) / 3)
    mean = 100 * sum(means) / len(means)
    half = 100 * 1.96 * statistics.stdev(means) / len(means) ** 0.5
    return mean, half


kept = [t for t in trials if t["folder"] != LEFT_OUT]
unique = list({t["copy"]: t for t in reversed(kept)}.values())  # the first copy of each trial, in path order
for label, rows in (("published rule, copies counted", trials), ("same rule, copies removed", unique)):
    mean, half = score(rows)
    print(f"{label}: {mean:.3f} +- {half:.3f} over {len(roster)} tasks")
