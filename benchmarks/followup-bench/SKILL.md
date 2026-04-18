---
name: followup-handling
description: Handle multi-turn follow-up conversations where the user corrects, clarifies, or changes requirements mid-task.
---

# Follow-up Handling Skill

Handle multi-turn conversations where users correct mistakes, clarify ambiguity, or change requirements after the initial request.

## Key behaviors

1. **Listen to corrections** — When a user says "actually X" or "no, I meant Y", discard the previous approach and follow the new instruction.
2. **Don't repeat work** — If the first turn produced partial correct output, preserve it and only change what the follow-up requires.
3. **Ask for clarification** — If a follow-up is ambiguous, ask before acting.
4. **Track state** — Remember what was done in previous turns and build on it, don't start from scratch.

## Anti-patterns to avoid

- Ignoring the follow-up and sticking with the original interpretation
- Starting over from scratch when only a small change is needed
- Applying the correction to the wrong part of the output
- Not acknowledging the correction before acting on it
