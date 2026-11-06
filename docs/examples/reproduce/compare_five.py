"""Compare the published SkillsBench trials with a rerun, task by task.

The five tasks and their rewards are the ones on Reproduce a published score:
for each task, the 3 trials the published score used and the 3 trials of the
rerun with BenchFlow 0.9.1. Prints each task's means, then three ways to ask
whether the rerun differs: a t interval over the tasks' differences, an exact
sign-flip test across tasks, and an exact permutation test within tasks.

    python3 compare_five.py               # this page's five tasks
    python3 compare_five.py rewards.json  # your own: {"task": [[published...], [rerun...]], ...}
"""

import itertools
import json
import statistics
import sys

# task: (published rewards, rerun rewards)
TRIALS = {
    "3d-scan-calc": ([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    "citation-check": ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
    "crystallographic-wyckoff-position-analysis": ([0.64, 0.55, 0.55], [0.55, 0.55, 0.55]),
    "econ-detrending-correlation": ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
    "offer-letter-generator": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
}
# The t distribution's 97.5th percentile by degrees of freedom (4 is this page's case: five tasks).
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.7764451051977987, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def t975(df):
    """The table's value for df, or for the nearest smaller df in it (wider, so conservative)."""
    return T_975[max(d for d in T_975 if d <= df)]

if len(sys.argv) > 1:
    with open(sys.argv[1]) as f:
        TRIALS = {task: (pair[0], pair[1]) for task, pair in json.load(f).items()}


def mean100(rewards):
    return 100 * sum(rewards) / len(rewards)


diffs = []
for task, (published, rerun) in TRIALS.items():
    d = mean100(rerun) - mean100(published)
    diffs.append(d)
    print(f"{task:44} published {mean100(published):5.1f}  rerun {mean100(rerun):5.1f}  difference {d:+6.1f}")

n = len(diffs)
mean = statistics.mean(diffs)
half = t975(n - 1) * statistics.stdev(diffs) / n**0.5
print(f"\nmean difference {mean:+.1f} points; 95% t interval {mean - half:+.1f} to {mean + half:+.1f}")

# Sign flips: under no difference, each task's difference is as likely negative.
flips = [sum(s * d for s, d in zip(signs, diffs)) for signs in itertools.product((1, -1), repeat=n)]
p_flip = sum(abs(f) >= abs(sum(diffs)) - 1e-9 for f in flips) / len(flips)
print(f"sign-flip p = {p_flip:.3f} (the smallest possible with {n} tasks: {2 / 2**n:.4f})")

# Within each task, any of its rewards could have been the rerun's (as many as the rerun had).
splits = []
for published, rerun in TRIALS.values():
    pool = published + rerun
    splits.append(
        [
            mean100([pool[i] for i in pick]) - mean100([pool[i] for i in range(len(pool)) if i not in pick])
            for pick in itertools.combinations(range(len(pool)), len(rerun))
        ]
    )
observed = sum(diffs)
size = 1
for options in splits:
    size *= len(options)
if size <= 5_000_000:
    total = extreme = 0
    for combo in itertools.product(*splits):
        total += 1
        extreme += abs(sum(combo)) >= abs(observed) - 1e-9
    print(f"within-task permutation p = {extreme / total:.3f} ({total:,} splits)")
else:
    import random

    rng = random.Random(0)
    total = 200_000
    extreme = sum(abs(sum(rng.choice(o) for o in splits)) >= abs(observed) - 1e-9 for _ in range(total))
    print(f"within-task permutation p = {extreme / total:.3f} ({total:,} sampled splits of {size:,})")
