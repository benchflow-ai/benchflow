# Authoring Tasks

BenchFlow authors tasks in the native `task.md` package format. A task is one Markdown document with YAML frontmatter plus sidecar directories for the sandbox, the verifier and the optional oracle (the reference solution):

```text
tasks/my-task/
├── task.md                  # settings (frontmatter) + the prompt the agent gets (body)
├── environment/
│   └── Dockerfile           # the sandbox image
├── verifier/
│   ├── test.sh              # writes a reward (0.0-1.0) to /logs/verifier/reward.txt
│   ├── test_outputs.py      # optional pytest checks that test.sh can run
│   ├── verifier.md          # optional: how the task is scored, for reviewers and tools
│   └── rubrics/             # optional: the rubric verifier.md refers to
└── oracle/
    └── solve.sh             # the reference solution, run by --agent oracle
```

## Your first task, step by step

Scaffold the package:

```bash
bench tasks init my-task
```

It writes `tasks/my-task/` with every file above. Each one holds `[REPLACE: ...]` placeholders, and the scaffold fails on purpose until you edit it (`test.sh` writes 0.0 and `solve.sh` exits 1), so an unedited task can never pass by accident. `bench tasks check` lists what is left:

```bash
bench tasks check tasks/my-task
```

A small complete task asks for a file and checks it. Replace the prompt below the frontmatter of `task.md` (keep the frontmatter; its first line is the canary that marks this file as benchmark data):

```markdown
Create a file `/app/hello.txt` containing exactly `Hello, world!`.
```

Use absolute paths in the prompt: the agent works in `/app`, the image's working directory, and the verifier and the oracle must look in the same place.

Replace `verifier/test.sh` with a check that writes the reward:

```bash
#!/bin/bash
REWARD=0
if [ "$(cat /app/hello.txt 2>/dev/null | tr -d '\n')" = "Hello, world!" ]; then
    REWARD=1
fi
echo "$REWARD" > /logs/verifier/reward.txt
```

Replace `oracle/solve.sh` with a solution:

```bash
#!/bin/bash
echo 'Hello, world!' > /app/hello.txt
```

This test.sh does all the scoring, so delete the optional files the scaffold wrote for richer verifiers: `verifier/test_outputs.py` (a pytest file that only runs if test.sh calls pytest), `verifier/verifier.md` and `verifier/rubrics/`. Keep them, with their placeholders replaced, when you want a declared scoring strategy or a rubric; see [Verifier package and strategy declaration](./task-authoring-task-md.md#verifier-package-and-strategy-declaration). `bench tasks init my-task --no-pytest` leaves out `test_outputs.py` from the start.

```bash
rm -r tasks/my-task/verifier/test_outputs.py tasks/my-task/verifier/verifier.md tasks/my-task/verifier/rubrics
bench tasks check tasks/my-task
```

The check prints `✓ my-task — valid (structural)`. Then prove the task both ways, before any model sees it: the oracle must score 1.0, and the empty `nop` agent, which does nothing, must score 0.0:

```bash
bench eval run --tasks-dir tasks/my-task --agent oracle --sandbox docker --jobs-dir jobs/my-task-oracle
bench eval run --tasks-dir tasks/my-task --agent nop --sandbox docker --jobs-dir jobs/my-task-nop
```

Neither needs a model or a key. Then run a model on it, for example `--agent claude --model claude-haiku-4-5-20251001` (see [Getting started](./getting-started.md#run-your-first-eval)).

The full authoring guide, including multi-container tasks, network policy and verifier strategies, lives in [Authoring native task.md tasks](./task-authoring-task-md.md), and the normative schema lives in [Task standard](./task-standard.md).

## Existing Split Packages

BenchFlow can still read and migrate older split packages so existing datasets have a direct upgrade path. Do not start new tasks in that layout.

```bash
bench tasks migrate tasks/old-task --remove-legacy
bench tasks check tasks/old-task
```

`--remove-legacy` promotes `tests/` to `verifier/`, promotes `solution/` to `oracle/`, and removes the old split entrypoint after the generated `task.md` round-trips successfully.

If you need to publish a compatibility artifact for another runner, export from the native package instead of hand-authoring the old layout:

```bash
bench tasks export tasks/my-task exported/my-task
```
