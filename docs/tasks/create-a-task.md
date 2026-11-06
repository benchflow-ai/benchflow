# Create a task

This tutorial builds a task that asks for one file, checks it, and proves the check works. It takes about ten minutes and needs Docker.

## 1. Scaffold the task

```bash
bench new my-task
```

This writes `tasks/my-task/`:

```text
tasks/my-task/
├── task.md             # settings + instruction
├── environment/
│   └── Dockerfile      # the sandbox image
├── verifier/
│   ├── test.sh         # writes the reward
│   ├── test_outputs.py # optional: pytest checks test.sh can run
│   ├── verifier.md     # optional: a declared scoring strategy
│   └── rubrics/        # optional: rubric files verifier.md names
└── oracle/
    └── solve.sh        # the reference solution
```

Every file but the Dockerfile holds `[REPLACE: ...]` placeholders. Until you replace them the task fails: `test.sh` writes 0.0 and `solve.sh` exits 1. `bench check` lists what is left:

```bash
bench check tasks/my-task
```

## 2. Write the instruction

Replace the body of `task.md`, everything below the closing `---`, with:

```markdown
Create a file `/app/hello.txt` containing exactly `Hello, world!`.
```

Keep the frontmatter. Its first line is the task's canary: a unique ID that shows up if the task ever leaks into training data.

Use absolute paths. The agent starts in `/app`, the image's working directory, and the verifier and oracle must look in the same place.

## 3. Write the verifier

Replace `verifier/test.sh` with a check that writes the reward:

```bash
#!/bin/bash
REWARD=0
if [ "$(cat /app/hello.txt 2>/dev/null | tr -d '\n')" = "Hello, world!" ]; then
    REWARD=1
fi
echo "$REWARD" > /logs/verifier/reward.txt
```

## 4. Write the oracle

Replace `oracle/solve.sh` with a solution:

```bash
#!/bin/bash
echo 'Hello, world!' > /app/hello.txt
```

## 5. Remove the files you don't use

This `test.sh` does all the scoring, so delete the optional verifier files:

```bash
rm -r tasks/my-task/verifier/test_outputs.py tasks/my-task/verifier/verifier.md tasks/my-task/verifier/rubrics
bench check tasks/my-task
```

It prints `✓ my-task — valid (structural)`. At its default level `bench check` never starts a sandbox; [Check a task](./check.md) lists what each level adds.

Keep `verifier.md` and `rubrics/` when you want a declared scoring strategy; see [Verifier](./verifier.md#declare-a-strategy). Next time, `bench new my-task --no-pytest` leaves out `test_outputs.py`.

## 6. Prove the task both ways

Before any model sees the task, the oracle must score 1.0. `nop`, an agent that does nothing, must score 0.0. Neither needs a model or a key.

```bash
bench run tasks/my-task -a oracle --sandbox docker --jobs-dir jobs/my-task-oracle --fresh
bench run tasks/my-task -a nop --sandbox docker --jobs-dir jobs/my-task-nop --fresh
```

Each run prints the reward and ends with a `View:` line. Run that line to open the trial in your browser.

`--fresh` starts a new job every time. Without it, a rerun resumes the last job in that jobs folder; see [Continue a run](../jobs/resume.md).

If the oracle does not score 1.0, check:

- `solve.sh` and `test.sh` look at the same absolute paths;
- the Dockerfile installs everything both scripts need;
- `test.sh` writes `/logs/verifier/reward.txt` on every path, including failures.

If `nop` scores above 0, the verifier passes an untouched workspace. Fix the check before you run a model.

## 7. Run a model

```bash
bench run tasks/my-task -a claude -m claude-haiku-4-5-20251001 --sandbox docker
```

`-a claude` is short for `claude-agent-acp`. [Getting started](../getting-started/quick-start.md#run-a-task-with-the-oracle) covers logins and keys.

## Next steps

- [Configuration](./configuration.md): timeouts, resources and every other `task.md` setting.
- [Verifier](./verifier.md): reward files, `reward.json`, strategies and rubrics.
- [Environment](./environment.md): images, extra containers and the paths the sandbox sets up.
- [Network policies](./network-policies.md): what the agent may reach.
