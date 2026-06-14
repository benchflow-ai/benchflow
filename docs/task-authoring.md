# Authoring Tasks

BenchFlow v0.6.2 authors tasks in the native `task.md` package format.
A task is one Markdown document with YAML frontmatter plus sidecar directories
for the sandbox, verifier, and optional oracle.

```text
tasks/my-task/
├── task.md
├── environment/
│   └── Dockerfile
├── verifier/
│   ├── test.sh
│   └── test_outputs.py
└── oracle/
    └── solve.sh
```

Start every new task with the native scaffold:

```bash
bench tasks init my-task
bench tasks check tasks/my-task
```

The full authoring guide lives in
[Authoring native task.md tasks](./task-authoring-task-md.md), and the
normative schema lives in [Task standard](./task-standard.md).

## Existing Split Packages

BenchFlow can still read and migrate older split packages so existing datasets
have a direct upgrade path. Do not start new tasks in that layout.

```bash
bench tasks migrate tasks/old-task --remove-legacy
bench tasks check tasks/old-task
```

`--remove-legacy` promotes `tests/` to `verifier/`, promotes `solution/` to
`oracle/`, and removes the old split entrypoint after the generated `task.md`
round-trips successfully.

If you need to publish a compatibility artifact for another runner, export from
the native package instead of hand-authoring the old layout:

```bash
bench tasks export tasks/my-task exported/my-task
```
