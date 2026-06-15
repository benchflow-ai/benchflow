Using the benchflow SDK, do the following:

1. Create a directory `/app/tasks/` with 3 symlinks pointing to tasks in `/app/sample-tasks/`:
   - task-a
   - task-b
   - task-c

2. Run a benchflow Job on `/app/tasks/` with:
   - agent: "claude-agent-acp"
   - model: "claude-haiku-4-5-20251001"
   - environment: "daytona"
   - concurrency: 3
   - jobs_dir: "/app/job-output"

3. After the job completes, use `collect_metrics` to analyze the results.

4. Write the metrics summary to `/app/metrics.json` as JSON.

The output JSON must contain at minimum: `total`, `passed`, `failed`, `score`.

The ANTHROPIC_API_KEY and DAYTONA_API_KEY are already set in the environment.
