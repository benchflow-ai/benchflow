# Stop reward hacking

An agent rewarded for a passing test will, sooner or later, try to pass it without doing the work: edit the test, plant a file the test imports, write the reward itself, or find the answer somewhere.

## What BenchFlow does between the agent and the verifier

The agent works as its own user, not root, and never sees the tests: `verifier/` is copied in only after the agent's turn ends. What happens next depends on `--integrity`.

### In `audit` (the default) and `off`

`test.sh` runs as written, in the agent's sandbox, and the processes the agent left running stay alive. BenchFlow keeps the reward the one `test.sh` wrote:

- `/logs/verifier/` is closed to the agent from sandbox setup on every backend: root's, mode 755 (on local Docker, where it is the trial folder's bind mount, the host user's, mode 755). A sandbox where setup can't show it closed is refused before the agent's turn (category `integrity_unavailable`), and so is one whose sandbox user is an image account with the uid BenchFlow runs as on that host; BenchFlow's own sandbox user never gets that uid there;
- before `test.sh`, it empties `/logs/verifier/` and gives it to root (mode 755), so a process the agent left can't write there. If a process keeps the folder from being emptied, the trial ends as a verifier failure: counted, never retried;
- when the agent's workspace is root's and mode 1777 (the grant makes it so when it holds a provider's folder, such as Daytona's `.daytona` in `/root`, or a folder BenchFlow keeps for itself, as `/opt` and `/usr/local` do), a root `test.sh` meets it without group or other write (1755), as a root oracle's `solve.sh` does in every mode: it can rewrite the agent's files in place, which the kernel's `fs.protected_regular` refuses in a sticky folder others may write. A service the agent left running keeps reading the workspace and writing its own files there, but can no longer create or rename names directly in it while `test.sh` runs. The workspace gets the grant's 1777 back before another agent turn (the next step of a multi-step task). A step that doesn't take ends the trial as a verifier error, never a quiet 0. A `test.sh` that runs as a user of its own leaves the workspace as it is;
- it removes every link the agent left, on any mount, that leads into `/logs/verifier/`, and checks that each one is gone. It finds them with BenchFlow's own `bfx` binary. One step still uses the image's programs: root runs the image's `sh` and `readlink` (or `realpath`) on each link the agent made, and their answer can only add a link to those removed, never keep one. If they fail, the check is incomplete;
- right before `test.sh` starts, it asks the kernel whether any process of the agent's user is still alive (zombies count until reaped). The answer is in the record as `alive_at_test`;
- from the moment `test.sh` exits until its outputs are copied, the sandbox's other processes are held still. The reward is read from that copy;
- once nothing in the sandbox can write any more, the copy goes back over the trial's `verifier/`. Only the names `test.sh` left there and BenchFlow's own records stay: any other name (a file, a folder or a link that a process left running added after `test.sh` exited) is removed, and named in `integrity/verifier-inputs.json` as `planted`, with what it was (a file's size and SHA-256, a link's target);
- each trial's `integrity/verifier-inputs.json` records the checks under `provenance`: what ran, what it found, and a verdict, `verified` or `unverifiable`, with the reasons. `verified` means every check ran and passed, and, for an agent that runs as its own user, that none of its processes was alive when `test.sh` started. That check sees only the processes in the agent's sandbox: it can't see `test.sh` itself running code the agent wrote, or another container of the task's compose project acting during grading.

When a check is defeated, an agent that ran as its own user scores 0 for the trial, counted, with the reason (category `provenance_incomplete`). An agent that runs as root keeps the reward `test.sh` left, marked `unverifiable`. Where `bfx` can't run in the sandbox, a model agent's trial is refused before its turn, unscored (category `integrity_unavailable`); the oracle still runs.

**A process the agent leaves running costs the `verified` label, not the score.** In `audit` and `off`, a server, a watcher or any other process of the agent's that is still alive when `test.sh` starts could change what `test.sh` writes while it runs, where no check can see it. So that trial keeps the reward `test.sh` wrote, marked `unverifiable`, with the reason "the agent's processes were still running when test.sh ran". The same holds when the sandbox can't answer the question: BenchFlow asks through its own copy of the image's Python, made before the agent's turn (never the image's live `python3`, whose library the agent may have changed), or util-linux `ionice`, so an image with neither reads `unverifiable`; so does a PID 1 that runs as the agent's user, or a kernel that doesn't implement the call. On Modal, whose kernel (gVisor) answers the same for every user, BenchFlow sends signal 0 as the agent's account instead; an image to which Modal adds Python (plain `ubuntu:24.04`) gives it no Python to ask with. `strict` ends the agent's processes before grading, so there they cost nothing. In `audit` and `off` they end after grading, and `kill-after-grading.json` in the trial names each one (pid, account, command, start time) with the signal that ended it, inferred from its SIGTERM disposition, not measured (`signal_source`); an empty list means none was left. An agent that runs as root isn't stopped after grading: its processes end with the sandbox. If your verifier needs a service the agent started, expect `unverifiable` in `audit`; to grade it `verified`, have `test.sh` start the service itself. That includes a task with `x-keep-services`: the services it keeps run as the agent's account, so in `audit` and `off` its trials read `unverifiable`, reward kept. The oracle is checked the same way: a process its `solve.sh` leaves running (as root, or as the task's `[agent].user`) reads `unverifiable`, its reward kept. `strict` refuses such a task before the agent's turn, because it grades in a separate sandbox, which has none of the services.

One gap remains: a sandbox filled with millions of files can make grading slow or fail; such trials score 0 with a reason.

### In `strict`

Before `test.sh` runs, BenchFlow:

- stops every process the agent left running (`kill-before-grading.json` in the trial records it);
- removes start-up files the agent could plant to run code inside the test, such as `conftest.py`, `sitecustomize.py` and `.pth` files, and modules named after a test runner, such as `pytest.py`;
- deletes any reward file already in `/logs/verifier/`;
- gives the test a fresh, root-only home folder when the agent could write to root's;
- removes the git hooks, filters, aliases and other command settings the agent added to a repository in the workspace, and runs root's git with hooks, `fsmonitor`, external diff and signing off, from 0.9.2 on. A task whose answer is a hook or a filter sets `cleanup_git: false` under `verifier: hardening:`.

## What it can't stop

These are the task author's to close:

- **A test that trusts a file the agent can write.** If the test reads its expected value, or the data it checks against, from the workspace, the agent can change it. Keep expected values in `verifier/`.
- **A test that imports from the workspace.** Python run from the workspace imports the agent's modules first; run a script with `python3 -I`, and pytest with `$BENCHFLOW_PYTEST` ([Verifiers](./verifiers.md#run-pytest-in-a-test)).
- **Answers the agent can look up.** With open network access, an agent can fetch a published solution. [Network and secrets](./network-and-secrets.md) shows how to close it.
- **A grader that checks only the answer.** The half-life test can't tell a fitted 37.4 from a typed one; a [rubric review](./rubric-review.md) can.

## Probe your verifier

Write the cheats you're worried about as small scripts, and run them as the agent with `-a script:<file>`. A script agent runs the file in the sandbox as the agent's user, with no model. Three probes against the [half-life task](./write-a-task.md); the task and the probes come from the BenchFlow repository's examples:

```bash
git clone --depth 1 https://github.com/benchflow-ai/benchflow.git
mkdir -p tasks && cp -r benchflow/docs/examples/research-task/half-life tasks/
# these probes test the test; a rubric review needs a reviewer model
rm tasks/half-life/verifier/rubric.json
cp -r benchflow/docs/examples/research-task/probes .
bench run tasks/half-life -a script:probes/look.sh
bench run tasks/half-life -a script:probes/shadow-json.sh
bench run tasks/half-life -a script:probes/write-reward.sh
```

| Probe | What it tries | The test printed | Reward |
|---|---|---|---|
| `look.sh` | lists `/verifier`, `/tests`, `/oracle`, and searches for the answer | `FAIL: no usable /app/answer.json (...)` | 0 |
| `shadow-json.sh` | writes a wrong answer, plus a `json.py` that returns the right one to anything that imports it | `FAIL: 4.50 min from the true value, more than 2.5; ...` | 0 |
| `write-reward.sh` | writes a wrong answer, and `1` to `/logs/verifier/reward.txt` | `FAIL: 4.50 min from the true value, more than 2.5; ...` | 0 |

A script agent's output is in the trial's `agent/script.txt`. The `look.sh` probe shows what an agent can find while it works:

```text
/verifier: absent
/tests: absent
/oracle: absent
/solution: absent
/logs/verifier: 
test.sh, solve.sh or rubric.json anywhere: 
files in /app, /home and /tmp that contain 37.4:
```

No tests, no reference solution, no file holding the answer. The `json.py` probe fails because the test runs `python3 -I`; without `-I` it would pass. The reward written by the agent is deleted before the test runs. The probe scripts are in [`docs/examples/research-task/probes`](../examples/research-task/probes). A task holds when every probe scores 0 while the oracle scores 1.

## Audit how each reward was earned

`--integrity audit` records what the agent did to the files and network the reward depends on, checks it against the task's rules, and writes a verdict next to each reward. The verdict never changes the reward; only the rules above do. In `audit` and `off`, an agent that runs as its own user scores 0 when a check of where its reward came from doesn't complete (`provenance_incomplete`), and in `strict`, BenchFlow removes what the agent left before grading.

```bash
bench run tasks/half-life -a script:probes/write-reward.sh --integrity audit
```

The run prints the reward as usual (`[FAIL] half-life (reward=0.00, tools=0)`); the verdict is in the trial's `integrity/claim_verdict.json`. Its main fields:

```json
{
  "core_verdict": "VectorExposed",
  "exploited": false,
  "reason": "no violation observed; exposed by shared_verifier, access_unattested (pre-turn probe scripted)",
  "reward": 0.0,
  "reward_effect": "none",
  "reward_source": "TrustedVerifier"
}
```

`core_verdict` is one of four: `Checked` (nothing the agent did reached the reward, and every way it could have was closed), `VectorExposed` (nothing reached it, but a way stayed open), `AgentViolation` (something the agent did reached it) or `Rejected` (the evidence to judge is incomplete). Nothing the probe did reached the reward (`reward_effect: "none"`): its `reward.txt` was deleted before the test ran, and the reward came from BenchFlow's own run of the test. The verdict is `VectorExposed` because of how the trial ran, not what the agent did: the test shared the agent's sandbox (`shared_verifier`), and for a script agent no root check confirmed that the protected paths were closed to it (`access_unattested`). The audit judges what reached the reward, so the attempt itself isn't flagged; the trajectory shows it.

`--integrity strict` also runs the test in a separate sandbox that receives only the agent's outputs, on the sandboxes that can: Docker, remote Docker, Daytona and Modal ([Sandboxes](../scale/sandboxes.md#choose-a-sandbox)). The oracle's verdict on the half-life task is then:

```json
{"core_verdict": "Checked", "reason": "no violation observed; the separate verifier received only the transferred outputs", "reward": 1.0}
```

### Strict is experimental

`--integrity strict` is experimental: on a 170-task sample, 17-19% of honest reference runs score 0 in strict because of what strict transfers. Strict's plugin guard runs inside the test process, so strict raises the cost of changing the test machinery and refuses a pass that has no complete guard verdict; it does not claim to stop code running in the test process from forging that verdict.

The separate verifier sandbox receives only the transferred workspace and the task's declared artifacts. A correct answer kept anywhere else scores 0 there. In the sample (33 tasks on the 0.10 line, 29 on 0.9.3), every honest run strict scored 0 fell into one of these classes:

| Class | What happens | How to tell |
|---|---|---|
| The working directory is outside the transferred workspace | The image's `WORKDIR` is `/`, so the workspace is `/root`, and the solution's edits to `/testbed` are not transferred. | `verifier-sandbox/verifier-sandbox.json` lists the edits under `outside_workspace_writes`. |
| Answers written outside the workspace | The solution writes its answer to a fixed path the workspace does not contain. | The same `outside_workspace_writes` list, or test.sh reads a path outside the workspace. |
| Grading reads the agent's log | test.sh reads a file under `/logs/agent`, which the verifier sandbox doesn't receive. | test.sh names `/logs/agent`. |
| System-level installs | The solution installs a program into a system folder such as `/usr/local/bin`. | test.sh runs a program the workspace doesn't hold. |
| tox | test.sh runs the tests through tox, whose own environment pytest runs in. | test.sh runs `tox`. |
| A gold conftest | The reference solution edits a `conftest.py` the verifier keeps from the image. | The solution's diff changes a `conftest.py`. |
| A reward outside 0-1 | Strict scores a reward outside the task's range as 0; audit and off record it as test.sh wrote it, with a note. This class is not a transfer, so it is outside the 17-19%. | The trial's `result.json` has `reward_out_of_range_info` with `recorded_as_written: false`. |
| pytest named in a branch test.sh doesn't take | Strict decides from test.sh's text whether it starts pytest. A test.sh that names pytest in a branch it doesn't take, then scores another way, leaves no record of pytest, and strict scores its pass 0. This class is not a transfer either, and no task in the sample has it. | The trial's warning says "the guard left no record, though test.sh starts pytest". |

When strict scores a task 0 that the default mode (`audit`) passes, check these classes first: audit's score is the task's own verdict.

The sample's tasks in each class (those marked 0.10 or 0.9.3 scored 0 on that line only):

- **Working directory outside the workspace (0.10: 17, 0.9.3: 11).**
  - `featurebench`: `astropy__astropy.b0db0daa.test_compression_failures.48dc420b.lv1`, `mlflow__mlflow.93dab383.test_genai_eval_utils.2d5441f2.lv1` (0.10), `mlflow__mlflow.93dab383.test_log_image.439a816d.lv1` (0.10), `mlflow__mlflow.93dab383.test_validation.2e1c5076.lv1` (0.10), `mwaskom__seaborn.7001ebe7.test_algorithms.1f0181c2.lv1`, `pandas-dev__pandas.82fa2715.test_all_methods.c74b49a1.lv1`, `pandas-dev__pandas.82fa2715.test_info.d8a64ebf.lv1`, `pandas-dev__pandas.82fa2715.test_sample.5d2ba03e.lv1`, `pydantic__pydantic.e1dcaf9e.test_titles.e806bda8.lv1`.
  - `featurebench-lite`: `astropy__astropy.b0db0daa.test_table.48eef659.lv1`, `mwaskom__seaborn.7001ebe7.test_statistics.0f2ae277.lv1`.
  - `featurebench-modal`: `astropy__astropy.b0db0daa.test_tiled_compression.a4fe298a.lv1`, `mlflow__mlflow.93dab383.test_evaluation_dataset.4a6528c2.lv1` (0.10), `mlflow__mlflow.93dab383.test_git_versioning.b91d0418.lv1` (0.10), `mlflow__mlflow.93dab383.test_tracing.0dc1e3a3.lv1` (0.10), `pandas-dev__pandas.82fa2715.test_http_headers.aafb551e.lv1`, `sympy__sympy.c1097516.test_runtests_pytest.71d43c5f.lv1`.
- **Answers written outside the workspace (0.10: 6, 0.9.3: 7).** `lawbench`: `lawbench-2-2-18-one-shot`, `lawbench-3-5-24-one-shot`, `lawbench-3-6-3-one-shot` (`/app/answer.jsonl`). `reasoning-gym`: `reasoning-gym-algorithmic-letter-jumble-hard0292e21f-seed45-2`, `reasoning-gym-algorithmic-pool-matrix-easy00cea0a5-seed45-1`, `reasoning-gym-arithmetic-fraction-simplification-hard0292e21f-seed45-0` (`/workspace/answer.txt`). `medagentbench`: `task2_9` (0.9.3).
- **Grading reads the agent's log (3).** `mmmlu`: `mmmlu-ar-xy-00007`, `mmmlu-pt-br-00115` and the Bengali (Bangladesh) task numbered 00018.
- **System-level installs (0.10: 2, 0.9.3: 3).** `seta-env`: `651`, `1098` (0.9.3), `1274`.
- **tox (4).** `swtbench-verified`: `sphinx-doc__sphinx-9591`, `sphinx-doc__sphinx-9602`, `sphinx-doc__sphinx-9673`, `sphinx-doc__sphinx-10673`.
- **A gold conftest (1).** `swtbench-verified`: `pytest-dev__pytest-5787`.
- **A reward outside 0-1 (2).** `sldbench`: `easy_question_scaling_law` (-1.0). `algotune`: `algotune-robust-kalman-filter` (just above 1).

To run the probes over a whole benchmark you didn't write, see [Check a benchmark](../audit/check-a-benchmark.md#probe-the-verifiers-with-a-cheating-agent).
