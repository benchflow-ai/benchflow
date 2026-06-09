The implementation should make native verifier package validity follow the
selected `verifier/verifier.md` strategy instead of requiring
`verifier/test.sh`.

Expected code shape:

- `TaskPaths.has_verifier_entrypoint()` returns true for selected executable
  verifier strategies.
- `TaskPaths.is_valid()` delegates verifier validity to that selected
  entrypoint check.
- A selected Reward Kit strategy without `verifier/test.sh` is valid only when
  its safe relative runner and criteria files exist.
- Malformed verifier documents or missing selected runners fail closed.
