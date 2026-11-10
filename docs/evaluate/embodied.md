# Embodied tasks (Robo Use)

An embodied task gives the agent a simulated robot. The simulator runs in a second container beside the agent's, with no network. The agent controls the robot only through the `robo` command (`robo observe`, `robo act`, `robo done`), and the verifier runs in the simulator's container and scores the robot's final physical state. A seed picks the starting state: in Meta-World's reach task it moves the goal, and the arm starts in the same place each time. The tasks and their simulators come from [Robo Use](https://github.com/benchflow-ai/robouse). Here the reference solution, the `oracle` agent, runs the reach task from three seeds on BenchFlow 0.9.1 with Docker; 0.9.2 runs it the same way.

## Set it up

BenchFlow reads Robo Use's task folders only when the `robouse` package is installed in the same Python environment as `bench`. Install BenchFlow with it, as on [Install](../start/install.md) plus `--with`:

```bash
uv tool install --python 3.12 'benchflow==0.9.2' --with 'robouse==0.6.1' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.9.2/constraints.txt
```

`--with` keeps Robo Use through later reinstalls of the tool. To add it to an install you already have, `uv pip install --python "$(uv tool dir)/benchflow/bin/python" "robouse==0.6.1"` adds only `robouse` and leaves the release's pinned packages as they were, but the next `uv tool install` of BenchFlow drops it. Without it, `bench run` stops at the first Robo Use setting in the task:

```text
.../robouse/tasks/metaworld/metaworld-reach/task.md: task.md parse error: robouse/tasks/metaworld/metaworld-reach/task.md:20: robouse: unknown key
  next: fix the task folder
```

The folder is fine; install the package. Then clone the tasks and check out a fixed commit, so a rerun gets the same tasks:

```bash
git clone https://github.com/benchflow-ai/robouse.git
git -C robouse checkout 5b2f77cddd7259a80287a216ee898e93021e8811
git -C robouse rev-parse HEAD
```

```text
5b2f77cddd7259a80287a216ee898e93021e8811
```

You also need Docker, with Compose 2.24.4 or newer as Robo Use asks. In 0.9.2 the simulator's image is built from a folder on the machine that runs `bench`, under `~/.cache/benchflow/task-formats/robouse/`; 0.10.0rc2 builds it from the task package itself, so embodied tasks also run on Daytona there.

## Run with seeds

```bash
export ROBOUSE_ORACLE_TOKEN=$(openssl rand -hex 16)
bench run robouse/tasks/metaworld/metaworld-reach -a oracle --seeds 0-2 \
  --allow-task-env ROBOUSE_ORACLE_TOKEN --sandbox docker -j 1 \
  --jobs-dir jobs --job seeded-reach
```

The token is for the reference solution; [The oracle token](#the-oracle-token) explains it. `--seeds 0-2` makes one trial per seed, named `metaworld-reach--seed-0` to `metaworld-reach--seed-2`; `--seeds` also takes a list, such as `0,3,7`. `-j 1` runs them one at a time. The first seed's lines and the summary:

```text
[metaworld-reach--seed-0] Building the task image for metaworld-reach--seed-0: the first run of a task builds it, later runs reuse it (this can take a few minutes)
[metaworld-reach--seed-0] Built the task image for metaworld-reach--seed-0 in 10 s
[metaworld-reach--seed-0] Setting up sandbox user: agent
[metaworld-reach--seed-0] Sandbox user agent ready (workspace=/app)
[metaworld-reach--seed-0] Oracle mode: running oracle solve.sh
[metaworld-reach--seed-0] test.sh ran without the image's PYTHONPATH: the root verifier clears these names, so a test that runs the image's own tools with them fails; set the value the test needs in test.sh or [verifier.env]
[metaworld-reach--seed-0] Running verifier...
[metaworld-reach--seed-0] Rewards: {'reward': 1.0, 'success_ever': 1.0, 'budget_used': 0.102}
[PASS] metaworld-reach--seed-0 (reward=1.00, tools=0)
...
Job complete: 3/3 (100.0%), mean_reward=1.00, errors=0, idle_timeouts=0, time=1.7min

✓ Score: 3/3 (100.0%), mean reward 1.00, errors=0
Outcomes: 3 scored (3 passed, 0 failed), 0 unscored, 0 errored
Cost:      none: no model tokens were used
Time:      1m 42s
Artifacts: jobs/seeded-reach
```

The image took 10 s because Docker had it from an earlier run; a machine's first build takes a few minutes. `budget_used` is the share of the task's budget of 500 steps that the episode used. Each trial's `artifacts/recording.mp4` is a video of its episode; the same file is at `verifier/episode/recording.mp4`, the path the export below names.

## Read the results by seed

```bash
bench embodied report jobs/seeded-reach
```

```text
task                                           n pass   mean    std    p@1    p@2    p@3
metaworld-reach                                3    3  1.000  0.000  1.000  1.000  1.000
all tasks                                      3       1.000         1.000  1.000  1.000
pass rate 1.000 (95% Wilson interval 0.439-1.000)
  sim: 3 trials, pass rate 1.000 [0.439, 1.000]
```

`n` counts trials and `pass` those that passed; `mean` and `std` are over their rewards, and `p@k` is pass@k ([Compare runs](../audit/compare-runs.md#choose-the-metric)). Across seeds, pass@k is the chance that at least one of k different starting states is solved. Three passes out of three still leave the pass rate anywhere from 0.44 to 1 at 95%, so compare agents on more seeds. `sim` says the robot was simulated. `--json` adds each seed's rewards, the mean return and the number of distinct starting states, 3 here.

Steps and return per seed are in the episode index that `bench embodied export` writes, next to `steps.jsonl`, which holds every step's observation (taken before the step's action), action and reward. So the last observation of a seed that passed can still show the arm short of the goal: the final action, which reached it, has no observation after it.

```bash
bench embodied export jobs/seeded-reach --out seeded-reach-data
jq -r '[.seed, .reward, .outcome, .steps_used, .return] | @tsv' seeded-reach-data/episodes.jsonl
```

```text
3 episode(s), 129 step(s) -> seeded-reach-data
0	1.0	success_reached	51	333.040193
1	1.0	success_reached	44	284.642213
2	1.0	success_reached	34	216.666856
```

`steps_used` counts simulator steps. `return` adds up Meta-World's own reward for each step; it doesn't decide the score. A seed gives the same starting state every time: a second run of the same command had the same `initial_state_sha256` for each seed, and the same steps and return.

## The oracle token

In a vision task the agent sees the robot's own state and camera images, and nothing else. Robo Use's reference solution needs the hidden state, which the simulator gives only to a caller holding a token you choose for the run. The task gives `$ROBOUSE_ORACLE_TOKEN` to the simulator and to the reference solution, not to other agents. BenchFlow passes your variables to a task only when you name them ([Environment variables](../reference/environment-variables.md#let-a-task-read-your-variables)), so for the oracle, export a fresh token and pass `--allow-task-env ROBOUSE_ORACLE_TOKEN`. Exported but not named, the token is replaced by the task's default, which is empty, and the run warns:

```text
warning: ROBOUSE_ORACLE_TOKEN is set in your environment or .env, but the task metaworld-reach-vision gets its default instead, since you did not name it. BenchFlow passes a variable from your environment into a task only when you name it: --allow-task-env ROBOUSE_ORACLE_TOKEN, or BENCHFLOW_ALLOW_TASK_ENV=ROBOUSE_ORACLE_TOKEN
```

On `robouse/tasks/vision/metaworld-reach-vision`, the vision version of reach, the reference solution then stopped with `KeyError: 'obs_vector'` before its first step, and the trial scored 0 with no error. With the token named, it passed in 51 steps. Plain reach shows the agent the goal, so its oracle passes either way. When the agent isn't the oracle, leave the token unset.

## Warnings you can ignore

`test.sh ran without the image's PYTHONPATH` appears once per trial, and in each trial's `result.json` under `warnings`. BenchFlow's verifier runs as root and clears the module and library paths the image sets, so nothing the agent could put on them runs as root. The task's `test.sh` sets the `PYTHONPATH` it needs itself. When the trial's `Rewards:` line follows, ignore the warning.

The token warning above is harmless when the agent isn't the oracle, and on tasks that aren't vision tasks (a vision task sets `obs_mode: vision` in its `task.md`).

## Not covered here

`bench --help` doesn't list `bench embodied` in 0.9.2, and [Commands](../reference/commands.md#commands-hidden-from-bench---help) lists it with the commands that may change, so check `bench embodied report --help` and `export --help` after an upgrade. `bench embodied check-spec` and `bench embodied robo-path` are there too, and each prints its own `--help`. Running a model as the agent, Robo Use's own `robouse` command and real robots are in [Robo Use's documentation](https://github.com/benchflow-ai/robouse/tree/5b2f77cddd7259a80287a216ee898e93021e8811/docs).
