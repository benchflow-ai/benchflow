# Prompt And User Semantics Rubric

- `prompt_composition`: append and explicit replace semantics are implemented
  with stable ordering and tests.
- `user_runtime`: document-declared `user`, `## user-persona`, and
  `benchflow.nudges` either compile into a concrete rollout user loop or fail
  sandbox-aware validation.
- `privacy`: private user facts are not materialized into solver prompts before
  scripted reveal conditions.
- `ergonomics`: task authors get clear validation errors and docs for supported
  vs metadata-only user semantics.

