Investigate why `report.py` produces negative totals and fix the root cause.
Leave the public API unchanged and add a regression test.

```role reviewer
You are a strict reviewer. Reject fixes that mask the symptom instead of the
cause, and require a regression test that fails before the change.
```

```stage investigate
First reproduce the negative-total case from the sample data and state the
root cause in one sentence before editing any code.
```

```user
A terse staff engineer who answers only when asked a direct, specific question.
```

```toml task
name = "benchflow/clean-body-roles-scenes"
description = "Authoring example — clean skill-style body, roles/scenes as files"

[about]
category = "authoring"
note = "Illustrates the native prompt surface; not a runnable package."

[import.v06]
schema_version = "1.3"
```
