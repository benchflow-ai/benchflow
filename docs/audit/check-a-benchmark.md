# Check a benchmark

Before you trust a benchmark's scores, check its tasks: each one should be solvable, none should pass for an agent that does nothing, the verifiers shouldn't be fooled by a shortcut, and a task shouldn't pass one time and fail the next for no reason. These checks need no model, so they cost minutes, not money, with two exceptions: a task with a rubric is reviewed on every run, controls included, by a reviewer that uses a model ([Rubric review](../tasks/rubric-review.md)); and a task that names its own agents in roles and scenes runs those agents even under `-a oracle` ([Multi-turn tasks](../tasks/multi-turn.md#several-agents-in-one-sandbox)). Run them on any benchmark you didn't write, and on your own before you publish it.

The examples check five tasks from SkillsBench, at commit f860ecf0, copied into a folder of their own:

```bash
git clone https://github.com/benchflow-ai/skillsbench.git
git -C skillsbench checkout f860ecf02ef53412a030f8661c83d644b739ff01
mkdir skillsbench-tasks
for t in 3d-scan-calc citation-check crystallographic-wyckoff-position-analysis \
         econ-detrending-correlation offer-letter-generator; do
  cp -r skillsbench/tasks/$t skillsbench-tasks/
done
```

## Run the oracle and nop over every task

```bash
bench run skillsbench-tasks -a oracle -j 3 --jobs-dir jobs/oracle
bench run skillsbench-tasks -a nop -j 3 --jobs-dir jobs/nop
```

```text
✓ Score: 5/5 (100.0%), mean reward 1.00, errors=0
Outcomes: 5 scored (5 passed, 0 failed), 0 unscored, 0 errored
...
• Score: 0/5 (0.0%), mean reward 0.00, errors=0
nop control: 0/5, as expected: doing nothing passes none of the tasks
```

`bench results` on the nop job lists each task with its control label:

```text
jobs/nop/2026-11-04__18-24-23 (evaluation): 0/5 attempted passed (0%), 0/5 scored passed (0%), 0 verifier errors, 0 unscored, only control runs, so they are counted
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Task                                       ┃ Reward ┃ Exec      ┃ Assess ┃ Control ┃ Agent ┃ Model ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 3d-scan-calc                               │ 0      │ completed │ scored │ empty   │ nop   │       │
│ citation-check                             │ 0      │ completed │ scored │ empty   │ nop   │       │
│ crystallographic-wyckoff-position-analysis │ 0      │ completed │ scored │ empty   │ nop   │       │
│ econ-detrending-correlation                │ 0      │ completed │ scored │ empty   │ nop   │       │
│ offer-letter-generator                     │ 0      │ completed │ scored │ empty   │ nop   │       │
└────────────────────────────────────────────┴────────┴───────────┴────────┴─────────┴───────┴───────┘
```

All five pass both controls. That's necessary, not enough: one of them can still be fooled, as the probe below shows.

## Read the control results

| Oracle | nop | What it means |
|---|---|---|
| passes | 0 | The task passes both controls. That is as far as controls can check it. |
| fails | any | The reference solution fails: the task is broken, or the oracle doesn't fit this sandbox. An agent's 0 here means nothing. |
| any | above 0 | Doing nothing earns credit: the test checks something already true. Every agent gets that credit for free, a full pass when the reward is 1. |
| unscored | any | BenchFlow or the sandbox failed before grading; rerun before you judge the task. |

For a pass-or-fail test, the oracle passes at 1. For a partial-credit test, compare its reward with the score its test calls a full answer, often 1. For a rubric-reviewed task, the oracle passes when its test and every blocker pass, even if the rubric gives it less than 1 ([Rubric review](../tasks/rubric-review.md#write-the-rubric)).

Leave a task that fails its controls out of every comparison, and say so in the report.

The oracle runs as root, and agents run as a user of their own, so an oracle pass doesn't show an agent can do the task: a test that reads a file only root could write passes for the oracle and fails for the agent.

## Terminal-Bench 2.0 on 0.9.1

A whole public benchmark, checked the same way: both controls on all 89 Terminal-Bench 2.0 tasks, with 0.9.1 and Docker 29.1.3, on a machine with 16 CPUs, 62 GB of memory and no GPU, on 5 November 2026. Each task ran as its own run, several at a time:

```bash
bench run -d terminal-bench@2.0 -i build-pmars -a oracle --jobs-dir jobs/oracle/build-pmars
bench run -d terminal-bench@2.0 -i build-pmars -a nop --jobs-dir jobs/nop/build-pmars
```

| Outcome | Oracle | nop |
|---|---|---|
| Scored 1 | 78 | 0 |
| Scored 0 | 9 | 87 |
| Unscored | 0 | 1 |
| Errored | 2 | 1 |

nop scored 1 on no task, so doing nothing passes none of them. These tasks fail a control, each for a cause in the task, the services it downloads from, or BenchFlow:

| Task | Oracle | nop | Cause | What the trial shows |
|---|---|---|---|---|
| `qemu-startup` | 0 | 0 | The test installs curl from Debian 11's security archive, which has moved to archive.debian.org, so no test runs | `404  Not Found` for `libnghttp2-14_1.43.0-1+deb11u3`, then `uvx: command not found` |
| `qemu-alpine-ssh` | 0 | 0 | The same | `404  Not Found` for `curl_7.74.0-1.3+deb11u16`, then `uvx: command not found` |
| `sam-cell-seg` | 0 | 0 | BenchFlow refuses a pytest plugin (anyio) the test installs while grading, and scores 0 | `Verifier plugin trust rejected: anyio` |
| `make-mips-interpreter` | errored | errored | BenchFlow refuses the image: a folder the system loader searches isn't root's alone | `LoaderCustodyError: ... /lib/mips-linux-gnu (owner 2952:1009, mode drwxr-xr-x), which is not root's alone` |
| `build-pmars` | 0 | 0 | The solution pins a package version Ubuntu no longer serves | `E: Version '1.22.21' for 'dpkg-dev' was not found` |
| `rstan-to-pystan` | 0 | 0 | The same | `E: Version '8.5.0-2ubuntu10.6' for 'curl' was not found` |
| `make-doom-for-mips` | 0 | 0 | Packages the solution installs are gone from Debian's servers | `404  Not Found` from apt, then no `frame.bmp` |
| `mcmc-sampling-stan` | 0 | 0 | The solution doesn't pin RcppParallel, whose current release needs cmake, so RStan doesn't install | `RcppParallel requires cmake (>= 3.5); cmake was not found` |
| `caffe-cifar-10` | errored | 0 | The solution downloads 162 MB of CIFAR-10 from a server that sent about 50 KB/s, past the task's 20-minute limit | `solution/solve.sh timed out after 1200s` |
| `build-cython-ext` | 0 | 0 | One of the repository's 18 tests fails after the solution | `KeyError: 'pos'` |
| `protein-assembly` | 0 | 0 | The solution's optimiser finds no answer, so it writes no file | `NoSolutionError`, then `File /app/gblock.txt does not exist` |
| `configure-git-webserver` | 1 | unscored | BenchFlow grades only with a root-owned `python3` in the sandbox; the solution installs one, nop doesn't | `Verifier hardening failed: ... the image of this task had no root-owned python3` |

No agent can pass the first four on 0.9.1: their tests can't run, or BenchFlow refuses them. For the next seven the oracle can't show the task is solvable, though an agent that solves them another way might pass. On the last, an agent that doesn't install `python3` ends unscored too.

Score a full run over the tasks that pass both controls, and name the rest with their cause. `-x` leaves a task out; this takes the twelve from the run's results file:

```bash
curl -fsSLO https://raw.githubusercontent.com/benchflow-ai/benchflow/main/docs/examples/controls/terminal-bench-2.0-controls-0.9.1.tsv
x=($(awk -F'\t' 'NR > 1 && $4 != "" { printf " -x %s", $1 }' terminal-bench-2.0-controls-0.9.1.tsv))
bench run -d terminal-bench@2.0 -a claude -m sonnet "${x[@]}"
```

`--dry-run` on the same command lists the twelve under `"exclude"`. `x` is an array, so bash and zsh both pass each flag as its own argument; a plain variable reaches `bench` as one argument in zsh. With an agent that runs as its own user, also report apart the four tasks below whose tests read root's writes and that pass both controls: `build-pov-ray`, `mailman`, `nginx-request-logging` and `sqlite-with-gcov`.

The error messages name a fix for BenchFlow's three: make the folder root's in the task's Dockerfile, pin the plugin in the task's `[verifier] pytest_plugin_releases`, install `python3` in the image. `--timeout-multiplier 3` gives `caffe-cifar-10` an hour.

A test that downloads its tools while grading can also fail on the network: `install-windows-3.11`'s test once got `curl: (22) The requested URL returned error: 504` from GitHub for uv and scored 0, and its rerun scored 1.

Both controls passing doesn't clear a task for agents either. The tasks were written for an agent that runs as root, and five tasks' tests read files only root could write; the oracle passed four of them, and an agent running as its own user can fail them ([the pitfall](./pitfalls.md#terminal-bench-20-tasks-fail-that-pass-in-published-runs)).

A full run needs disk. Docker listed 87 of the run's 89 task images at 103 GB, and `bench cache clean --remove` keeps every image a run pulled ([Sandboxes](../scale/sandboxes.md#what-benchflow-keeps-between-runs)), so remove each task's image with `docker image rm` once its runs finish. When our disk filled, trials ended with `[Errno 28] No space left on device` or `Cannot pull image`, which read like task failures; we reran those tasks. Every task's result is in [`terminal-bench-2.0-controls-0.9.1.tsv`](../examples/controls/terminal-bench-2.0-controls-0.9.1.tsv).

## Check for duplicate tasks and seen answers

Two tasks that ask nearly the same thing count one skill twice. [`dupes.py`](../examples/controls/dupes.py) compares every pair of instructions in a folder of tasks by the share of five-word runs they have in common. BenchFlow keeps a registry dataset's tasks under `.cache/datasets-v2/<org>/<repository>/` in the Git repository you ran from, or the current folder outside one. On the 89 Terminal-Bench 2.0 tasks it downloaded for the controls above, no pair shares half; the closest are variants of one subject:

```bash
curl -fsSLO https://raw.githubusercontent.com/benchflow-ai/benchflow/main/docs/examples/controls/dupes.py
python3 dupes.py .cache/datasets-v2/laude-institute/terminal-bench-2
```

```text
89 instructions, 3916 pairs
0.26  dna-assembly  dna-insert
0.24  polyglot-c-py  polyglot-rust-c
0.13  feal-differential-cryptanalysis  feal-linear-cryptanalysis
...
pairs at 0.5 or more: 0
```

A model may also have seen a public benchmark's tasks or their solutions in training, and no control shows that. Compare its scores on tasks published before and after its training cutoff, or keep tasks that were never published, and say in the report which you did.

## Find flaky tasks

A task whose reference solution passes only some of the time makes every score on it noisy. Run the oracle a few times:

```bash
bench run skillsbench-tasks -a oracle -j 3 --trials 3 --jobs-dir jobs/flaky
bench eval metrics jobs/flaky/oracle --k 1 --k 3
```

```text
                      Results: jobs/flaky/oracle
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric         ┃ Value                                             ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Total          │ 15                                                │
│ Passed         │ 15                                                │
│ Failed         │ 0                                                 │
│ Errored        │ 0                                                 │
│ Score          │ 100.0% (15 of 15 trials)                          │
...
│ pass@1         │ 100.0% · pass^1 100.0% (5 tasks)                  │
│ pass@3         │ 100.0% · pass^3 100.0% (5 tasks)                  │
└────────────────┴───────────────────────────────────────────────────┘
```

pass^3 is the share of tasks that passed all three times. At 100%, no failure was seen in three tries of any of these five, which doesn't prove a task never fails: one that fails 10% of the time still passes three tries in a row 73% of the time. To bound a task's failure rate, run more tries: with no failures in n tries, the 95% upper bound on its failure rate is about 3/n, so 30 clean tries bound it near 10%. A task that fails here fails for reasons that have nothing to do with an agent: a network fetch, a race, a timeout too tight for the machine.

## Check that a verifier accepts other correct answers

A test can be too strict: it fails a right answer written another way. `bench check --level equivalence` runs the oracle, then grades variants of its output, some wrong and some equal in value, and reports which the test got wrong. On 0.9.1, with the half-life task from these docs' examples, whose test accepts a range on purpose (0.10.0rc1 prints the same, and so does a path such as `../tasks/half-life` given from another folder):

```bash
git clone --depth 1 https://github.com/benchflow-ai/benchflow.git
mkdir -p tasks && cp -r benchflow/docs/examples/research-task/half-life tasks/
# the check is about the test; a rubric review needs a reviewer model
rm tasks/half-life/verifier/rubric.json
bench check tasks/half-life --level equivalence --sandbox docker
```

```text
✗ half-life — 1 issue(s):
  → equivalence: false positive [wrong-number] answer.json: +1% on 2 field(s): half_life_min 37.12 -> 37.4912; 
uncertainty_min 1.135 -> 1.14635 scored 1.0, not below oracle 1.0
```

Read each flag against the task's own rules before you change anything. Here the check raised the oracle's answer by 1% and expected the test to reject it; the half-life task accepts anything within 2.5 minutes of the true value on purpose, and 37.49 is a correct answer. A flag like this on a task with an exact answer would be a real bug in the test.

## Probe the verifiers with a cheating agent

A script agent (`-a script:<file>`) runs a shell script as the agent, with no model, so you can try a specific cheat. SkillsBench's `3d-scan-calc` reads a 3D scan from the workspace and checks the reported mass against the scan; this script ([`3d-scan-calc-cheat.sh`](../examples/probes/3d-scan-calc-cheat.sh)) replaces the scan with one triangle and reports that triangle's mass:

```bash
#!/bin/bash
# Replace the scan the verifier reads with one triangle, and report that triangle's mass.
python3 - <<'PY'
import json, struct
tri = struct.pack("<3f3f3f3fH", 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1)
open("/root/scan_data.stl", "wb").write(b"\0" * 80 + struct.pack("<I", 1) + tri)
json.dump({"main_part_mass": 1 / 6 * 0.10, "material_id": 1}, open("/root/mass_report.json", "w"))
PY
```

Save it as `cheat.sh`, make it executable, and run it as the agent, with the verifier in the agent's sandbox and then in a sandbox of its own:

```bash
bench run skillsbench-tasks/3d-scan-calc -a script:./cheat.sh
bench run skillsbench-tasks/3d-scan-calc -a script:./cheat.sh --verifier-sandbox separate
```

```text
[PASS] 3d-scan-calc (reward=1.00, tools=0)
✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
...
[PASS] 3d-scan-calc (reward=1.00, tools=0)
✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
```

The script solves nothing and scores 1, with the verifier in the agent's sandbox and in a separate one. The test computes the expected mass from the scan in the workspace, a file the agent can replace, so no sandboxing helps; the test has to read its own copy of the input. This is SkillsBench issue #1064, "18 task verifiers grade against files the agent can overwrite", open as of November 4, 2026. A benchmark with a task like this can report a pass for an agent that found the shortcut.

[Stop reward hacking](../tasks/reward-hacking.md#probe-your-verifier) shows more probes and how a task author closes each one.

## Search trajectories for shortcuts

Every trial keeps the agent's tool calls in `trajectory/acp_trajectory.jsonl`, one event per line, with each command in its `title`. Search them for the shortcuts that matter for your benchmark: reading the tests, fetching a known solution, writing into the verifier's folders, trying to become root.

Find every trajectory under the jobs folder, whatever its layout, count them, so an empty result means something, then search them. The commands work in bash and zsh, and with spaces in paths:

```bash
find jobs -path '*/trajectory/acp_trajectory.jsonl' | wc -l
find jobs -path '*/trajectory/acp_trajectory.jsonl' -exec \
  grep -h -o -E '"title": "[^"]*(sudo -n|/logs/verifier|/testbed_verify|/\.bfx)[^"]{0,60}' {} +
```

Over our example runs it searched 17 trajectories and found two agents probing, both OpenCode on DeepSeek. One, on a SkillsBench task, looked for the grader and for root before it did the work:

```text
"title": "cat /instruction.md; echo '---'; ls -laR /testbed_verify | head -100; echo '---'; ls -la /app /logs /opt
"title": "whoami; id; sudo -n true 2>&1 && echo SUDO_OK; ls -ld /testbed_verify; stat /testbed_verify
```

Every attempt was refused (`sudo: command not found`, and `/testbed_verify` is readable by root only), and the agent then solved the task the honest way; its reward stands. The other, in a task whose workspace it couldn't write, spent its whole turn on BenchFlow's own sandbox helper, trying to run commands as other users and to read `/etc/shadow`:

```text
"title": "for a in /bin/true /bin/echo /usr/bin/env /.bfx/bfx; do echo ===$a; /.bfx/bfx exec --json \
"title": "getent shadow root 2>&1; getent shadow agent 2>&1; python3 - <<'PY'\nimport subprocess,json\nfor d in [\n {'root':'/','paths':['etc/sha
```

Each try was denied (`exec /usr/bin/id: EPERM`), and the trial scored 0. Neither run broke anything, but both belong in a report: an agent that probes like this will take a shortcut where a benchmark leaves one, as `3d-scan-calc` does.

## Benchmarks with no reference solution

Without an oracle you can still run `nop`, the probes and a trajectory search. You can't check solvability or run the equivalence check; use a strong agent's passing trials as a stand-in, read a sample of them, and say in the report that the task set has no reference solutions.
