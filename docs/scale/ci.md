# Run in CI

Run a benchmark in CI to catch a change that makes an agent, a task or a verifier worse. `--fail-under` turns the pass rate into an exit code, so the build fails when the rate drops below your line.

## Fail under a pass rate

`tasks/half-life` is the task [Write a task](../tasks/write-a-task.md) builds and shows how to copy.

```bash
bench run tasks/half-life -a nop --fail-under 0.5
echo "exit code $?"
```

```text
• Score: 0/1 (0.0%), mean reward 0.00, errors=0
nop control: 0/1, as expected: doing nothing passes none of the tasks
...
Gate failed: pass rate 0.00 < --fail-under 0.5
exit code 1
```

With the oracle, the same gate passes and the command exits 0. The pass rate is over every planned task, so a trial that errored or ended unscored counts against it.

## Exit codes

| Exit code | Means |
|---|---|
| 0 | the run finished, and every gate you set passed |
| 1 | a gate failed (`--fail-under`, `--fail-on`), or the run itself failed |
| 2 | the command was wrong: a bad flag or value, refused before anything ran |

`--fail-on timeout,error,verifier-error` fails the build when any trial ended one of those ways, even if the pass rate is fine. `--summary-out summary.json` writes the run's counts and the gate result to a file a later step can read.

## A GitHub Actions job

The job installs a pinned BenchFlow, runs the controls on your tasks, and fails under a pass rate. Docker is already on GitHub's Ubuntu runners.

```yaml
name: benchmark
on: [pull_request]
jobs:
  controls:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - run: echo "$HOME/.local/bin" >> "$GITHUB_PATH"
      - run: uv tool install --python 3.12 'benchflow==0.9.2' --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.9.2/constraints.txt
      - run: bench run tasks/ -a oracle --fail-under 1.0
      - run: bench run tasks/ -a nop --fresh --fail-on error,verifier-error --summary-out nop.json
      - run: jq -e '.passed == 0' nop.json
```

The oracle step fails the build when a task can no longer be solved, and the nop step when a task stops loading or grading. The last step catches a task that passes for free. A `nop` run that passes a task only warns, and exits 0; on a task whose test always writes 1:

```text
✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
nop control: 1/1 passed with nothing done: those tasks can be passed without solving them; check their verifiers
```

The summary file then holds `"passed": 1`, and `jq -e '.passed == 0' nop.json` prints `false` and exits 1, failing the build. A task with a rubric is reviewed on these control runs too, and 0.9.2's reviewer has no default model, so such a run stops before it starts: give both steps `--reviewer-agent` and `--reviewer-model` with the model's key as a secret, or keep rubric tasks out of them ([Rubric review](../tasks/rubric-review.md)). For an agent run in CI, pass the model key as a repository secret, and cap the spend with `--max-cost-usd` ([Concurrency and budgets](./cost.md#budgets)).
