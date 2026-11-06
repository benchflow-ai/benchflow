# Scenes and simulated users

Run several agents in one task, or one agent with several prompts. All of these patterns are scenes: roles and turns in one sandbox.

```python
from pathlib import Path

import benchflow as bf
from benchflow import Role, RolloutConfig, Scene, Turn

config = RolloutConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(
            name="review-loop",
            roles=[
                Role("coder", "gemini", "gemini-3.1-flash-lite-preview"),
                Role("reviewer", "claude-agent-acp", "claude-sonnet-5"),
            ],
            turns=[
                Turn("coder"),
                Turn("reviewer", "Review /app. Write feedback to /app/review-feedback.md."),
                Turn("coder", "Read /app/review-feedback.md and revise your solution."),
            ],
        )
    ],
    environment="docker",
)
result = bf.run_sync(config)
print(result.rewards)
```

The verifier scores the sandbox once, after the last turn.

## How roles share work

The roles share the sandbox's files, not a conversation. When the turn passes to another role, BenchFlow stops the current agent and starts the next one in a new session. Back-to-back turns of the same role keep one session.

So pass work between roles through files, and name the file in the next prompt. BenchFlow doesn't copy messages between roles.

| | Carries over |
|---|---|
| Next turn of the same role | the session, with its memory, and the files |
| A turn of another role | the files only |
| The next scene | the files only |

## Paths in the sandbox

| Path | Holds |
|---|---|
| `/app` | the agent's workspace, which the verifier scores |
| `/instruction.md` | the task's instruction, for every task |
| `/oracle` | the reference solution of a `task.md` task (`/solution` in a split task), for `--agent oracle` runs only |

A turn with no prompt (`Turn("coder")`, or `- role: coder` in YAML) gets the task's instruction.

The oracle is uploaded only for `--agent oracle` runs. No other run has a copy in the sandbox, so no role can read it.

## Write scenes in YAML

Put one task's scenes in a YAML file:

```yaml
# review.yaml
task_path: tasks/my-task
environment: docker
scenes:
  - name: review-loop
    roles:
      - name: coder
        agent: gemini
        model: gemini-3.1-flash-lite-preview
      - name: reviewer
        agent: claude-agent-acp
        model: claude-sonnet-5
    turns:
      - role: coder
      - role: reviewer
        prompt: "Review /app. Write feedback to /app/review-feedback.md."
      - role: coder
        prompt: "Read /app/review-feedback.md and revise your solution."
```

Then load and run it:

```python
import benchflow as bf

result = bf.run_sync(bf.rollout_config_from_yaml("review.yaml"))
```

A scene with roles but no `turns` gives each role one turn, in order.

> **Note:** a run config for `bench run --config` has no `scenes` key; it refuses one as unknown. To give every task in a benchmark the same scenes, declare them in each task's `task.md` (see [task.md authoring](./overview.md)), or loop over the tasks in Python.

## Simulated user

A "user" role gives the assistant instructions a little at a time and reveals information gradually. A user role shares the agent's sandbox, so it never sees the solution. For a user that knows the solution, use a Python `User` with `oracle_access=True`, which reads `solve.sh` on the host ([scripted users](./scripted-users.md#oracle-access)).

```yaml
task_path: tasks/my-task
environment: daytona
scenes:
  - name: interactive-assist
    roles:
      - name: user
        agent: gemini
        model: gemini-3.1-flash-lite-preview
      - name: assistant
        agent: claude-agent-acp
        model: claude-sonnet-5
    turns:
      - role: user
        prompt: |
          You are a user who needs help with the task in /instruction.md.
          Write a high-level request to /app/user-guidance.md.
      - role: assistant
        prompt: "Read /app/user-guidance.md and do what it asks."
      - role: user
        prompt: |
          Compare the assistant's work in /app against the task in /instruction.md.
          If it is incomplete, add one specific missing requirement to /app/user-guidance.md.
      - role: assistant
        prompt: "/app/user-guidance.md has a new hint. Read it and finish the task."
```

Each assistant turn starts a new session. So every assistant prompt names the file that holds the guidance.

The user is a full agent with tools: it reads files and checks outputs. For a user written as Python rules instead of a second model, use `BaseUser`; see [Progressive disclosure](./scripted-users.md).

## Code review loop

A coder solves the task, a reviewer writes feedback, and the coder revises:

```yaml
task_path: tasks/my-task
environment: daytona
scenes:
  - name: review-loop
    roles:
      - name: coder
        agent: gemini
        model: gemini-3.1-flash-lite-preview
      - name: reviewer
        agent: gemini
        model: gemini-3.1-flash-lite-preview
    turns:
      - role: coder
      - role: reviewer
        prompt: |
          You are a code reviewer. Read /instruction.md and the coder's work in /app.
          Do NOT change any file in /app except /app/review-feedback.md.
          Write specific, actionable feedback to /app/review-feedback.md.
      - role: coder
        prompt: "Read /app/review-feedback.md and revise your solution."
```

The reviewer here can still write to `/app`: only its prompt tells it not to. To take away that access, give the review to a tool instead.

### Review as an MCP tool

The experimental MCP reviewer runs a reviewer model behind a tool. The coder calls `review_code` and gets feedback text back. The reviewer has no way to write files.

```python
from pathlib import Path

import benchflow as bf
from benchflow import Role, RolloutConfig, Scene, Turn
from benchflow.experimental.mcp.hooks import mcp_reviewer_hook

config = RolloutConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(
            name="solve-and-review",
            roles=[Role("coder", "gemini", "gemini-3.1-flash-lite-preview")],
            turns=[
                Turn("coder"),
                Turn("coder", "Call the review_code MCP tool for feedback, then fix the issues."),
            ],
        )
    ],
    environment="daytona",
    pre_agent_hooks=[mcp_reviewer_hook(port=8100, model="gemini-3.1-flash-lite")],
)
result = bf.run_sync(config)
```

The hook starts `benchflow.experimental.mcp.reviewer_server` as a background process in the same sandbox, before the agent. It serves `review_code` and `get_review_status` over streamable HTTP at `http://localhost:8100/mcp`. The task gives the agent that server in its sandbox settings:

```toml
[[sandbox.mcp_servers]]
name = "reviewer"
transport = "streamable-http"
url = "http://localhost:8100/mcp"
```

### Compare reviewers

Run the same tasks under each condition, with the same verifier:

| Condition | Turns |
|---|---|
| `baseline` | the coder alone, one turn |
| `reviewer` | coder, reviewer, coder |
| `reviewer+spec` | coder, a reviewer that re-reads the instruction, coder |

Whether a reviewer helps depends on the benchmark. Measure it; don't assume it.

## Cross-model review

Give the roles different models, for example a cheap coder and a strong reviewer. Change only the reviewer's `agent` and `model` in the review loop above:

```yaml
      - name: reviewer
        agent: claude-agent-acp
        model: claude-sonnet-5
```

Each variant is its own YAML file, with the same task and verifier:

| Variant | Coder | Reviewer | Question |
|---|---|---|---|
| Self-review | gemini-flash | gemini-flash | Does same-model review help? |
| Cross-model | gemini-flash | claude-sonnet | Does another model catch other bugs? |
| Strong reviewer | gemini-flash | claude-opus | Does a stronger reviewer help a weaker coder? |
| Weak reviewer | claude-opus | gemini-flash | Does a weaker reviewer hurt a stronger coder? |

## Follow-up prompts to one agent

One role gets several prompts in one session and keeps its memory between them:

```yaml
task_path: tasks/my-task
environment: daytona
scenes:
  - name: iterative-solve
    roles:
      - name: solver
        agent: gemini
        model: gemini-3.1-flash-lite-preview
    turns:
      - role: solver
      - role: solver
        prompt: "Review your solution. Run the tests if there are any. Fix what you find."
      - role: solver
        prompt: "Re-read the instruction and check your solution meets every requirement."
```

Use it for self-review, for build-test-fix cycles, or to split a task into phases.

## Generate a skill, then solve

Two scenes in a row: the first writes a skill file and is not scored, and the second solves the task with that file on disk.

```yaml
task_path: tasks/my-task
environment: daytona
scenes:
  - name: skill-gen
    roles:
      - name: gen
        agent: gemini
        model: gemini-3.1-flash-lite-preview
    turns:
      - role: gen
        prompt: |
          Read /instruction.md. Write a skill to /app/generated-skill.md that helps
          an agent solve this task: key steps, common pitfalls, useful commands
          and a solution outline.
  - name: solve
    roles:
      - name: solver
        agent: gemini
        model: gemini-3.1-flash-lite-preview
    turns:
      - role: solver
```

The solver starts a new session with no memory of the first scene. It gets the task's instruction and finds `/app/generated-skill.md` on disk. The verifier scores only the final `/app`.

The SkillsBench paper reports about zero lift from self-written skills with a generic prompt. A prompt specific to the task type ("write a skill for compiler tasks") is what helps.

## Tasks with mock services

A task can start mock Gmail, Calendar, Docs, Drive and Slack services in the sandbox. The agent calls real HTTP endpoints on `localhost`.

```python
from pathlib import Path

import benchflow as bf
from benchflow import SERVICES, RolloutConfig, Scene, build_service_hooks

services = [SERVICES["gmail"], SERVICES["gcal"], SERVICES["slack"]]

config = RolloutConfig(
    task_path=Path("tasks/schedule-meeting-from-email"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    environment="daytona",
    pre_agent_hooks=build_service_hooks(services),
)
result = bf.run_sync(config)
```

Start services with `pre_agent_hooks`. `RolloutConfig.services` is a record only: it starts nothing.

| Service | Command | Port | Database |
|---|---|---|---|
| `gmail` | `claw-gmail` | 9001 | `/data/gmail.db` |
| `slack` | `claw-slack` | 9002 | `/data/slack.db` |
| `gcal` | `claw-gcal` | 9003 | `/data/gcal.db` |
| `gdoc` | `claw-gdoc` | 9004 | `/data/gdoc.db` |
| `gdrive` | `claw-gdrive` | 9005 | `/data/gdrive.db` |

Each service runs as a background process in the sandbox, answers `/health` once it is up, and keeps its state in SQLite. To seed a service, put its database at that path in the task's image.

```text
tasks/schedule-meeting-from-email/
├── task.md                 # settings + instruction
├── environment/
│   ├── Dockerfile          # FROM benchflow/claws-base, which has every claw-* command; copies the seeds to /data
│   ├── gmail.db            # seed: an email from Alice asking for a meeting
│   └── gcal.db             # seed: the existing calendar
├── oracle/
│   └── solve.sh            # curl calls to the Gmail and Calendar APIs
└── verifier/
    └── test.sh             # checks /data/gcal.db has the new event
```

## Next steps

- What turns, rounds and scenes keep: [Core concepts](../core-concepts.md#turns-rounds-and-scenes).
- A simulated user in Python: [Progressive disclosure](./scripted-users.md).
- Fork a run at a checkpoint: [Branching](../jobs/branching.md).
