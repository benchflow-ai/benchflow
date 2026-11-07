"""Compare the published SkillsBench trials with a rerun, task by task.

The five tasks and their rewards are the ones on Reproduce a published score:
for each task, the 3 trials the published score used and the 3 trials of the
rerun with BenchFlow 0.9.1. Prints each task's means, then three ways to ask
whether the rerun differs: a t interval over the tasks' differences, an exact
sign-flip test across tasks, and an exact permutation test within tasks.

    python3 compare_five.py
"""

import itertools
import statistics

# task: (published rewards, rerun rewards)
TRIALS = {
    "3d-scan-calc": ([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    "citation-check": ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
    "crystallographic-wyckoff-position-analysis": ([0.64, 0.55, 0.55], [0.55, 0.55, 0.55]),
    "econ-detrending-correlation": ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
    "offer-letter-generator": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
}
# The t distribution's 97.5th percentile with 4 degrees of freedom.
T_4 = 2.7764451051977987


def mean100(rewards):
    return 100 * sum(rewards) / len(rewards)


diffs = []
for task, (published, rerun) in TRIALS.items():
    d = mean100(rerun) - mean100(published)
    diffs.append(d)
    print(f"{task:44} published {mean100(published):5.1f}  rerun {mean100(rerun):5.1f}  difference {d:+6.1f}")

n = len(diffs)
mean = statistics.mean(diffs)
half = T_4 * statistics.stdev(diffs) / n**0.5
print(f"\nmean difference {mean:+.1f} points; 95% t interval {mean - half:+.1f} to {mean + half:+.1f}")

# Sign flips: under no difference, each task's difference is as likely negative.
flips = [sum(s * d for s, d in zip(signs, diffs)) for signs in itertools.product((1, -1), repeat=n)]
p_flip = sum(abs(f) >= abs(sum(diffs)) - 1e-9 for f in flips) / len(flips)
print(f"sign-flip p = {p_flip:.3f} (the smallest possible with {n} tasks: {2 / 2**n:.4f})")

# Within each task, any 3 of its 6 rewards could have been the rerun's.
splits = []
for published, rerun in TRIALS.values():
    pool = published + rerun
    splits.append(
        [
            mean100([pool[i] for i in pick]) - mean100([pool[i] for i in range(6) if i not in pick])
            for pick in itertools.combinations(range(6), 3)
        ]
    )
observed = sum(diffs)
total = extreme = 0
for combo in itertools.product(*splits):
    total += 1
    extreme += abs(sum(combo)) >= abs(observed) - 1e-9
print(f"within-task permutation p = {extreme / total:.3f} ({total:,} splits)")
