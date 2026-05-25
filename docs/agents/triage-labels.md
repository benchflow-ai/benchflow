# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

**Note:** `wontfix` already exists in this repo. The other four labels are not yet created. Run these to create them when you start using the triage workflow:

```bash
gh label create needs-triage --description "Maintainer needs to evaluate" --color "fbca04"
gh label create needs-info --description "Waiting on reporter" --color "d876e3"
gh label create ready-for-agent --description "Fully specified, AFK-ready" --color "0e8a16"
gh label create ready-for-human --description "Needs human implementation" --color "5319e7"
```

Edit the right-hand column above if you later remap to existing or different label names.
