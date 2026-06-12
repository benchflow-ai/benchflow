---
name: skill-creator
description: Create a tiny task-specific skill pack for BenchFlow self-gen smoke tests.
---

# Skill Creator

When asked to create a skill pack for this smoke task, do it directly.

Create exactly one skill pack under the requested generated-skills directory:

```text
<generated-skills-root>/conformance-writer/SKILL.md
```

The generated `SKILL.md` should tell the solver:

- create `conformance.txt` in the current working directory
- write exactly `ok`
- list the current directory contents

Keep the generated skill short. Do not solve the task in the creator turn.
