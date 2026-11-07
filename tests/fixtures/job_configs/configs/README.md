YAML job config inputs (stored as JSON) exercise explicit task selection and
agent defaults. These are compatible input files, not recorded run outputs.

- `git-task-config.json`: a task pinned by `git_url` and `git_commit_id`;
  the fabricated commit exercises refusal before any fetch.
- `glob-name-config.json`: two tasks, including a literal `hello-tw?` name.
- `datasets-no-agent-config.json`: a dataset with no agent; runs the oracle.
- `empty-agents-config.json`: an explicit `agents: []` beside a `tasks:` list.
