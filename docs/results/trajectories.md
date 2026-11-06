# Trajectories

```python
import benchflow as bf

trial = bf.load_trial("jobs/my-job/hello-world-task__0000abcd")
for event in trial.trajectory:          # ACP events, oldest first
    print(event.get("type"))
```

A trajectory is what the agent did in a trial: its messages, thoughts and tool calls. BenchFlow saves it in the trial folder in several formats.

## Where the files are

| File | Format | Written |
|---|---|---|
| `trajectory/acp_trajectory.jsonl` | ACP events, one per line | every trial (older trials: `agent/acp_trajectory.jsonl`) |
| `trajectory/llm_trajectory.jsonl` | raw model requests and responses | when the model calls went through BenchFlow's proxy |
| `trainer/atif.json` | ATIF, the Agent Trajectory Interchange Format (`ATIF-v1.7`): one JSON document with `agent`, `steps` and `final_metrics` | every trial that writes `result.json`, unless its trajectory is empty |
| `trainer/adp.jsonl` | ADP, the Agent Data Protocol (schema 1.3.1) | every trial that writes `result.json` |
| `trainer/verifiers.jsonl` | a Verifiers record (prompt, completion, reward) | every trial that writes `result.json` |
| `agent/*.txt` | the agent's own logs | when the agent writes any |

Secrets are masked in `acp_trajectory.jsonl` and in the trainer files when they are written.

## Load them in Python

```python
import json
from pathlib import Path

import benchflow as bf

job = bf.load_job("jobs/my-job")
for trial in job.trials:
    print(trial.task_name, trial.reward, len(trial.trajectory))

# One trial as a document, trajectory included
doc = bf.load_trial("jobs/my-job/hello-world-task__0000abcd").to_json_dict()
print(len(doc["trajectory"]))

# The ATIF file
atif = json.loads(Path("jobs/my-job/hello-world-task__0000abcd/trainer/atif.json").read_text())
print(atif["schema_version"], len(atif["steps"]), atif["final_metrics"])
```

`trial.trajectory` reads `acp_trajectory.jsonl` and returns a list of event dicts. It is empty when nothing was captured.

`to_json_dict()` puts the same events in the `trajectory` field of a [`benchflow.trial` document](../reference/json-export.md). Job documents leave trajectories out unless you pass `include_trajectories=True`.

From the command line:

```bash
bench view jobs/my-job/hello-world-task__0000abcd            # read it in a browser
bench eval inspect jobs/my-job/hello-world-task__0000abcd --json --trajectories
```

## Next steps

- [Share a trajectory](./share-a-trajectory.md): upload a session to BenchFlow's dataset
- [Training signal](../reference/training-signal.md): turn trials into training data
- [Streaming rollouts to a trainer](../reference/rollout-stream.md)
- [JSON export](../reference/json-export.md)
