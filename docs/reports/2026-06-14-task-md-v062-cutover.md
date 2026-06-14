# BenchFlow v0.6.2 task.md cutover

BenchFlow v0.6.2 makes native `task.md` packages the task authoring path:

- `bench tasks init <name>` scaffolds `task.md`, `environment/`, `verifier/`,
  and `oracle/`.
- The CLI no longer scaffolds new split-layout tasks.
- Existing split packages remain readable so they can be migrated in place with
  `bench tasks migrate <dir> --remove-legacy`.
- Compatibility exports are produced from native packages with
  `bench tasks export`.

This keeps the migration path available without leaving two first-class
authoring formats for contributors to choose between.
