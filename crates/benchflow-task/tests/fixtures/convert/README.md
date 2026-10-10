# Conversion fixtures

`tests/convert_golden.rs` checks `bench import` and `bench export` against task.md's reference converter, `tools/convert.py` in the task.md repository.

| Folder | Holds |
|---|---|
| `harbor/` | Split-layout (Harbor) tasks: public BenchFlow tasks from this repository, and small tasks written to cover each import rule (comments, deprecated spellings, steps, services, cheat oracles, a CRLF instruction, values Harbor would refuse, an instruction with an open code fence) |
| `v06/` | BenchFlow 0.8 task.md packages: the examples in `docs/examples/task-md/`, demo and conformance tasks, and two written to cover roles, scenes, a simulated user, extension namespaces, `prompts/` files, `verifier/verifier.md`, a RewardKit judge file, bundled skills, and a stochastic oracle case |
| `export/` | task.md packages for export: one that exports losslessly, and ones that lose a clock, judges and a rubric, a role block, or only metadata |
| `expected/` | What `tools/convert.py` wrote: `harbor/` and `v06/` (imports), `roundtrip/` (each import exported back), `export/`, and `results.json` (errors, notes, and each export's status and losses); and `checked.json`, the errors task.md's reference checker (`tools/taskmd.py`) reports on each import |

Regenerate `expected/` after changing an input, with the task.md repository checked out beside this one and PyYAML installed:

```sh
python make_goldens.py . /path/to/task-md/tools
python check_goldens.py . /path/to/task-md/tools
```

The files are compared byte for byte, so `.gitattributes` keeps git from changing their line endings.
