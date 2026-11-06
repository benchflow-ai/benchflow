"""Compare two runs of the same tasks, task by task, with exact tests that respect the pairing.

usage: python3 paired.py <folder of arm A> <folder of arm B> [--success 1.0]

Each folder holds one or more BenchFlow jobs (a --trials run's agent folder works). For each task in both, the
script takes each arm's trials and their success (reward >= --success). It prints:
- per task: trials and passes in each arm, and the difference in pass rate;
- the mean of those differences over the tasks: the estimand, for these tasks, and its 95% t interval over tasks;
- an exact permutation test that shuffles trials between the arms within each task (are these tasks different?);
- an exact sign-flip test over tasks (would the difference hold for other tasks like these?), and the smallest p it
  can give with this many tasks.
"""
import glob
import itertools
import json
import os
import sys
from collections import defaultdict

args = [a for a in sys.argv[1:] if not a.startswith("--")]
success = float(sys.argv[sys.argv.index("--success") + 1]) if "--success" in sys.argv else 1.0
arm_a, arm_b = args[0], args[1]


def trials(folder):
    out = defaultdict(list)
    for path in glob.glob(os.path.join(folder, "**", "result.json"), recursive=True):
        trial_dir = os.path.dirname(path)
        if not os.path.exists(os.path.join(trial_dir, "config.json")):
            continue  # a job-level result.json, not a trial
        r = json.load(open(path))
        reward = (r.get("rewards") or {}).get("reward")
        if reward is None:
            continue  # unscored: left out here; count it yourself if it matters
        out[r.get("task_name") or os.path.basename(trial_dir).split("__")[0]].append(1 if reward >= success else 0)
    return out


A, B = trials(arm_a), trials(arm_b)
tasks = sorted(set(A) & set(B))
diffs = []
print(f"{'task':45} {'A':>7} {'B':>7}  B-A")
for t in tasks:
    a, b = A[t], B[t]
    d = sum(b) / len(b) - sum(a) / len(a)
    diffs.append(d)
    print(f"{t:45} {sum(a)}/{len(a):<5} {sum(b)}/{len(b):<5} {d:+.3f}")
obs = sum(diffs) / len(diffs)
print(f"\nmean difference in pass rate over {len(tasks)} tasks (B - A): {obs:+.3f}")
# 95% t interval over tasks: what the difference could be for other tasks like these.
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
if len(diffs) > 1:
    m = obs
    sd = (sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
    t = T975[len(diffs) - 2] if len(diffs) - 1 <= len(T975) else 1.96
    half = t * sd / len(diffs) ** 0.5
    print(f"95% t interval over tasks: {m - half:+.3f} to {m + half:+.3f}")

# Exact permutation within tasks: every way to split each task's pooled trials into the two arms' sizes.
per_task = []
for t in tasks:
    pool, na = A[t] + B[t], len(A[t])
    splits = []
    for idx in itertools.combinations(range(len(pool)), na):
        a = [pool[i] for i in idx]
        b = [pool[i] for i in range(len(pool)) if i not in idx]
        splits.append(sum(b) / len(b) - sum(a) / len(a))
    per_task.append(splits)
count = total = 0
for combo in itertools.product(*per_task):
    total += 1
    count += abs(sum(combo) / len(combo)) >= abs(obs) - 1e-12
print(f"within-task permutation test, two-sided: p = {count / total:.3f} ({total} splits)")

# Exact sign flip over tasks.
flips = [sum(s * d for s, d in zip(signs, diffs)) / len(diffs) for signs in itertools.product((1, -1), repeat=len(diffs))]
p = sum(abs(f) >= abs(obs) - 1e-12 for f in flips) / len(flips)
print(f"sign-flip test over tasks, two-sided: p = {p:.3f} (the smallest possible with {len(tasks)} tasks: {2 / len(flips):.3f})")
