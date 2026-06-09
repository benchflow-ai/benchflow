# Verifier Native Entrypoint Rubric

- `task_paths_entrypoint`: `TaskPaths` validates selected
  `verifier/verifier.md` strategies instead of hard-coding `test.sh`.
- `regression_tests`: tests cover valid Reward Kit packages without
  `test.sh` and missing-runner failure.
- `no_test_sh_dogfood`: this dogfood package passes structural and
  sandbox-aware task checks without a `verifier/test.sh` file.
