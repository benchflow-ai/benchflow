# Prompt And User Semantics Rubric

- `prompt_composition`: append and explicit replace semantics compile into a
  package-level prompt plan with stable ordering and tests.
- `user_runtime`: document-declared `user`, `## user-persona`, and
  `benchflow.nudges` compile into a concrete linear rollout user loop across
  ordered document scenes. Human confirmation policies fail closed without an
  explicit handler, and branchable `ask_user` requests preserve option kinds
  while the first sequential shared-workspace team handoff subset executes
  explicit multi-role turns and unsupported full branch/team semantics are
  surfaced explicitly.
- `privacy`: private user facts are not materialized into solver prompts before
  scripted reveal conditions.
- `ergonomics`: task authors get clear validation errors and docs for supported
  vs metadata-only user semantics.
