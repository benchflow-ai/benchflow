Evaluate this coding-agent run.

You receive the task instruction, a bundle with the diff against the starting repository, any new files, and the agent's final message, and the agent's trajectory. The bundle is the deliverable. The trajectory is context about how the agent worked.

Read the task instruction at `/tests/judge/instruction.md` and the bundle at `/logs/verifier/judge-bundle.md` if they are not already included below. Do not modify any files.

Score each criterion from 0 to 100.

{criteria}

Prefer evidence-backed, maintainable changes that match the problem's scope.
Penalize superficial test edits, missing edge cases, and changes that only satisfy visible tests.
Penalize sophisticated-looking constructs that fail to deliver the invariant they imply.
Do not penalize adding tests, ADRs, migrations, or documentation when warranted by the task.
