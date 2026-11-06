# Import and export

```bash
bench import tasks/old-task --remove-legacy   # task.toml → task.md
bench export tasks/my-task out/my-task-toml # task.md → task.toml
```

A task.toml task (`task.toml`, `instruction.md`, `solution/`, `tests/`) runs in BenchFlow as it is. Import it only when you want to keep working on it as a `task.md` task.

## Import a task

```bash
bench import tasks/my-task                  # write task.md, keep task.toml + instruction.md
bench import tasks/my-task --overwrite      # replace an existing task.md
bench import tasks/my-task --remove-legacy  # then delete them: tests/ → verifier/, solution/ → oracle/
```

| Option | Effect |
|---|---|
| (none) | Writes `task.md` beside the old files. `task.md` wins from then on. |
| `--overwrite` | Replaces an existing `task.md` |
| `--remove-legacy` | Deletes `task.toml` and `instruction.md`, and renames `tests/` to `verifier/` and `solution/` to `oracle/`. It does this only after the new `task.md` checks out. |

The import checks its own work. It parses the new `task.md` again, and fails unless the settings and the instruction match the originals. Nothing is lost silently.

`task.toml` keys BenchFlow does not model are kept, not dropped, under `benchflow.compat`:

```yaml
benchflow:
  compat:
    source: legacy
    extra_paths:
      - sandbox.modal.image
      - verifier.criteria_runner.metric
    extra:
      sandbox:
        modal:
          image: registry.example.com/task:latest
      verifier:
        criteria_runner:
          metric: exact_match
```

Then check and prove the result:

```bash
bench check tasks/my-task
bench run tasks/my-task -a oracle --sandbox docker --jobs-dir jobs/my-task-oracle
```

## Export a task

```bash
bench export tasks/my-task out/my-task-toml
bench export tasks/my-task --report-only        # print what would be lost, write nothing
```

| Option | Effect |
|---|---|
| `--overwrite` | Replace an existing export folder |
| `--report-only` | Print the loss report without writing files |
| `--target task-toml` | The only target, and the default |

The export writes `task.toml`, `instruction.md`, `solution/` and `tests/`. It puts the keys from `benchflow.compat.extra` back into `task.toml` without overwriting BenchFlow's own.

It also writes `compatibility/export-report.json`. The report lists:

- the files it chose and their hashes;
- name collisions;
- the keys it restored, under `restored_extension_paths`;
- anything the split layout cannot hold.

Keep the export in its own folder, not beside `task.md`. A folder with both fails `bench check --level publication-grade`.

## What a round trip keeps

A round trip keeps meaning, not bytes:

| Part | Compared as |
|---|---|
| Settings | Parsed and normalized |
| Instruction | Normalized text |
| `oracle/`, `verifier/` | SHA-256 of every regular file |

Comments, formatting and blank lines are not kept.

## Other formats

BenchFlow also runs [task.md draft 2](./taskmd-draft-2.md) folders, and other benchmark formats through adapters; see [Task formats](./task-formats.md).

## Next steps

- [Overview](./overview.md#task-folders): how BenchFlow's names map onto the task.toml format's.
- [Check a task](./check.md): validate the result.
- [CLI reference](../reference/cli.md): every flag of `bench import` and `bench export`.
