# Clean body + sidecar prompts

The `task.md` body is the base prompt — free-form, no `## prompt` heading, exactly
like a `SKILL.md` body. The extra prompts live as their own files:

- `prompts/role.reviewer.md`
- `prompts/scene.investigate.md`
- `prompts/user-persona.md`

This keeps every prompt body free-form (no reserved-heading ceremony). Single-shot
tasks need none of the `prompts/` files at all.
