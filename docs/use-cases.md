# Multi-Agent and Multi-Scene Use Cases

BenchFlow's Scene-based lifecycle enables evaluation patterns that go far beyond single-turn "prompt and score." This document covers the key use cases for researchers migrating from Harbor who need multi-turn, multi-agent, or stateful environment capabilities.

**Context:** Harbor issue [#1316](https://github.com/harbor-ai/harbor/issues/1316) proposed a simulated-user API (a "user" agent that iteratively provides instructions), and PR [#1462](https://github.com/harbor-ai/harbor/pull/1462) added multi-turn support via Docker Compose sidecars. BenchFlow solves both with a simpler primitive: **Scenes with Roles and Turns**, all running in a single shared sandbox via ACP.

---

## 1. Interactive User Simulation (Harbor #1316 equivalent)

A "user" role provides instructions iteratively; the agent responds. The user has oracle access to the solution and reveals information gradually, simulating realistic human-agent interaction.

In Harbor, this required a FastMCP sidecar container running a simulated-user persona server, wired via Docker Compose networking. In BenchFlow, it is a two-role Scene where the "user" role is just another agent with a different prompt and (optionally) a different model.

### YAML

```yaml
task_dir: .ref/terminal-bench-2
environment: daytona
concurrency: 64

scenes:
  - name: interactive-assist
    roles:
      - name: user
        agent: gemini
        model: gemini-3.1-flash-lite-preview
      - name: assistant
        agent: claude-agent-acp
        model: claude-sonnet-4-6
    turns:
      - role: user
        prompt: |
          You are simulating a user who needs help with the task in /app/instruction.md.
          You have access to the solution in /solution/solve.sh.
          Give the assistant a high-level description of what you want. Do NOT reveal implementation details yet.
          Write your message to /app/.outbox/assistant.json.
      - role: assistant
      - role: user
        prompt: |
          Read the assistant's work in /app/. Compare against /solution/solve.sh.
          If incomplete, provide a targeted hint (one specific detail from the solution).
          Write to /app/.outbox/assistant.json.
      - role: assistant
        prompt: "The user provided additional guidance. Read it and continue working."
      - role: user
        prompt: |
          Final check. Read /app/ and compare to /solution/. If correct, write
          {"to": "assistant", "content": "LGTM"} to /app/.outbox/assistant.json.
          If not, give one final hint.
      - role: assistant
        prompt: "Address the user's latest feedback and finalize your solution."
```

### Python

```python
from benchflow.trial import TrialConfig, Scene, Role, Turn

config = TrialConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(name="interactive-assist",
              roles=[
                  Role("user", "gemini", "gemini-3.1-flash-lite-preview"),
                  Role("assistant", "claude-agent-acp", "claude-sonnet-4-6"),
              ],
              turns=[
                  Turn("user", "You are simulating a user. Read /app/instruction.md..."),
                  Turn("assistant"),  # None = use instruction.md
                  Turn("user", "Check the assistant's work against /solution/..."),
                  Turn("assistant", "The user provided additional guidance..."),
              ]),
    ],
    environment="daytona",
)
result = await bf.run(config)
```

### Why this is better than Harbor #1316

- No Docker Compose, no sidecar container, no FastMCP server to maintain.
- Both agents share the sandbox filesystem -- the "user" reads `/solution/` (which is locked from the assistant by `lockdown_paths`).
- The user agent is a real LLM with full tool access -- it can read files, check outputs, and give nuanced feedback, not just templated responses.
- Same task folder works for single-turn (baseline) and interactive (with user) via different YAML configs.

---

## 2. Code Review Loop (followup-bench)

A coder agent solves the task, then an independent reviewer agent critiques the solution. The coder revises based on the feedback. The reviewer never has write access to `/app/` -- it can only read and provide feedback.

### YAML

```yaml
task_dir: .ref/terminal-bench-2
environment: daytona
concurrency: 64

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
          You are an expert code reviewer. Read the task at /app/instruction.md
          and the coder's work in /app/. Write specific, actionable feedback.
          IMPORTANT: Do NOT modify any files in /app/ except /app/.outbox/coder.json.
          Write: {"to": "coder", "content": "Your specific feedback here."}
      - role: coder
        prompt: "Read the reviewer's feedback and revise your solution."
```

### Python (with MCP reviewer sidecar)

For stronger isolation, use the MCP reviewer server pattern. The reviewer runs as a sidecar service -- it has no filesystem write access at all. The coder calls the reviewer via a tool call:

```python
from benchflow.trial import TrialConfig, Scene, Role, Turn

config = TrialConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(name="solve-and-review",
              roles=[Role("coder", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[
                  Turn("coder"),
                  Turn("coder", "Call the review_code MCP tool to get feedback, then fix issues."),
              ]),
    ],
    services=["benchflow-reviewer:8100"],
    environment="daytona",
)
result = await bf.run(config)
```

The MCP reviewer server (`benchflow.mcp.reviewer_server`) runs as a background process in the sandbox. It exposes `review_code` and `get_review_status` tools via streamable-http. The reviewer LLM reads the code but has **no ability to write files** -- all it can do is return feedback text.

### Results

On Terminal-Bench 2, adding an independent reviewer approximately doubles the win rate on tasks where the baseline fails. Ablation experiments (`experiments/reviewer_ablation.py`) compare three conditions:

| Condition | Description |
|-----------|-------------|
| `baseline` | Single-agent, single-turn |
| `reviewer` | Coder + plain reviewer + coder revision |
| `reviewer+spec` | Coder + reviewer that re-reads instruction + coder revision |

The reviewer condition consistently outperforms baseline on complex tasks that require debugging or multi-file coordination.

### Why this beats Harbor

- Harbor PR #1462 required a separate container per agent and Docker Compose networking. BenchFlow runs both agents in the same sandbox -- cheaper, faster startup.
- The MCP pattern (`services: ["benchflow-reviewer:8100"]`) gives the reviewer tool-level isolation: it cannot write to the workspace, preventing reward hacking via reviewer collusion.
- Same task, same verifier -- just add the `scenes` key to your YAML.

---

## 3. Skill Generation (BYOS -- Bring Your Own Skill)

An agent generates a task-specific skill before solving. This is a two-scene trial: `prep` (unscored) and `solve` (scored). Both scenes share the sandbox, so the generated skill persists.

### YAML

```yaml
task_dir: .ref/skillsbench/tasks
environment: daytona
concurrency: 64

scenes:
  - name: skill-gen
    roles:
      - name: gen
        agent: gemini
        model: gemini-3.1-flash-lite-preview
    turns:
      - role: gen
        prompt: |
          Read /app/instruction.md. Analyze the task requirements.
          Write a skill document to /app/generated-skill.md that will help
          an agent solve this task. Include: key steps, common pitfalls,
          relevant commands or APIs, and a solution outline.
  - name: solve
    roles:
      - name: solver
        agent: gemini
        model: gemini-3.1-flash-lite-preview
    turns:
      - role: solver
```

### Python

```python
from benchflow.trial import TrialConfig, Scene, Role, Turn

config = TrialConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(name="skill-gen",
              roles=[Role("gen", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[Turn("gen", "Analyze the task and write a skill to /app/generated-skill.md")]),
        Scene(name="solve",
              roles=[Role("solver", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[Turn("solver")]),  # None prompt = use instruction.md
    ],
    environment="daytona",
)
result = await bf.run(config)
```

### How scenes work here

1. **Scene 1 (`skill-gen`)**: The `gen` agent reads the task instruction, analyzes it, and writes a skill file. This scene is unscored -- its output is an artifact that persists in the sandbox filesystem.
2. **Scene 2 (`solve`)**: A fresh agent session starts (no context from scene 1). The `solver` agent gets the standard `instruction.md` prompt and also sees `/app/generated-skill.md` on disk. The verifier scores only the final `/app/` state.

The key insight: `disconnect()` between scenes kills the agent process, so there is no context bleed. The only communication is through the shared filesystem.

### Research findings

From the SkillsBench paper: self-generated skills with generic prompts yield approximately 0 percentage points of lift over baseline. The BYOS pattern only helps when the skill-generation prompt is task-type-specific (e.g., "write a skill for compiler tasks" vs. "write a skill for this task"). This result informed the GEPA (Guided Evolution of Prompts and Agents) skill improvement pipeline.

---

## 4. Multi-turn Conversation

The same agent receives multiple prompts in sequence, maintaining full conversation context between turns. This is the simplest multi-turn pattern -- no role switching, just sequential prompts to a persistent ACP session.

### YAML

```yaml
task_dir: .ref/terminal-bench-2
environment: daytona
concurrency: 64

scenes:
  - name: iterative-solve
    roles:
      - name: solver
        agent: gemini
        model: gemini-3.1-flash-lite-preview
    turns:
      - role: solver
      - role: solver
        prompt: "Review your solution. Run the tests if available. Check for edge cases and fix any issues you find."
      - role: solver
        prompt: "Final check: re-read the original instruction and verify your solution addresses every requirement."
```

### Python

```python
from benchflow.trial import TrialConfig, Scene, Role, Turn

config = TrialConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(name="iterative-solve",
              roles=[Role("solver", "gemini", "gemini-3.1-flash-lite-preview")],
              turns=[
                  Turn("solver"),  # instruction.md
                  Turn("solver", "Review your solution. Run tests. Fix issues."),
                  Turn("solver", "Final check: verify every requirement is met."),
              ]),
    ],
    environment="daytona",
)
result = await bf.run(config)
```

### How it works

ACP sessions are persistent -- the agent process stays alive across all turns within a scene. The agent retains full conversation history (tool calls, outputs, reasoning) between prompts. Each `Turn` sends a new `prompt()` call on the existing session.

This is equivalent to what Harbor #1316 proposed as "multi-turn evaluation" -- but without needing a simulated user. The "user" in this case is the benchmark framework itself, issuing predetermined follow-up prompts.

### Why this is useful

- **Self-review**: The second prompt asks the agent to check its own work, catching obvious errors.
- **Iterative refinement**: Tasks that require build-test-fix cycles benefit from explicit prompts to test and iterate.
- **Decomposition**: Complex tasks can be broken into phases ("first set up the environment", "now implement the feature", "now write tests").

---

## 5. Cross-model Review

Different models fill different roles in the same scene. A cheap model codes, an expensive model reviews. Role-level model configuration makes this trivial.

### YAML

```yaml
task_dir: .ref/terminal-bench-2
environment: daytona
concurrency: 32

scenes:
  - name: cross-model-review
    roles:
      - name: coder
        agent: gemini
        model: gemini-3.1-flash-lite-preview
      - name: reviewer
        agent: claude-agent-acp
        model: claude-sonnet-4-6
    turns:
      - role: coder
      - role: reviewer
        prompt: |
          You are reviewing code written by a different agent.
          Read /app/instruction.md for the task requirements.
          Examine the coder's work in /app/. Write specific feedback
          to /app/.outbox/coder.json: {"to": "coder", "content": "..."}
      - role: coder
        prompt: "Read the reviewer's feedback and revise your solution."
```

### Python

```python
from benchflow.trial import TrialConfig, Scene, Role, Turn

config = TrialConfig(
    task_path=Path("tasks/my-task"),
    scenes=[
        Scene(name="cross-model-review",
              roles=[
                  Role("coder", "gemini", "gemini-3.1-flash-lite-preview"),
                  Role("reviewer", "claude-agent-acp", "claude-sonnet-4-6"),
              ],
              turns=[
                  Turn("coder"),
                  Turn("reviewer", "Review the coder's work..."),
                  Turn("coder", "Address the reviewer's feedback."),
              ]),
    ],
    environment="daytona",
)
result = await bf.run(config)
```

### Cost-performance tradeoff

The cross-model pattern lets you sweep the reviewer axis independently:

| Variant | Coder | Reviewer | Question |
|---------|-------|----------|----------|
| Self-review | gemini-flash | gemini-flash | Does same-model review help? |
| Cross-model | gemini-flash | claude-sonnet | Does a different model catch different bugs? |
| Strong reviewer | gemini-flash | claude-opus | Does a stronger reviewer help a weaker coder? |
| Weak reviewer | claude-opus | gemini-flash | Does a weaker reviewer hurt a stronger coder? |

Each variant is just a different YAML file -- same task folder, same verifier, different role configurations. This enables controlled experiments on the marginal value of reviewer quality.

---

## 6. Stateful Environment (ClawsBench)

Tasks that require agents to interact with live services -- Gmail, Calendar, Docs, Drive, Slack. Services run as sidecar processes in the sandbox, exposing REST APIs on localhost. The agent interacts with real HTTP endpoints, not mocked tool calls.

### YAML

```yaml
task_dir: .ref/clawsbench/tasks
environment: daytona
concurrency: 32

services:
  - gmail
  - gcal
  - slack
```

### Python

```python
from benchflow.trial import TrialConfig, Scene, Role, Turn
from benchflow import SERVICES, build_service_hooks

# Declare which services the task needs
services = [SERVICES["gmail"], SERVICES["gcal"], SERVICES["slack"]]

config = TrialConfig(
    task_path=Path("tasks/schedule-meeting-from-email"),
    scenes=[Scene.single(agent="gemini", model="gemini-3.1-flash-lite-preview")],
    environment="daytona",
    pre_agent_hooks=build_service_hooks(services),
)
result = await bf.run(config)
```

### Service registry

BenchFlow ships with 5 built-in services (from the SmolClaws project):

| Service | CLI binary | Port | Description |
|---------|-----------|------|-------------|
| `gmail` | `claw-gmail` | 9001 | Mock Gmail REST API (FastAPI + SQLite) |
| `slack` | `claw-slack` | 9002 | Mock Slack API |
| `gcal` | `claw-gcal` | 9003 | Mock Google Calendar API |
| `gdoc` | `claw-gdoc` | 9004 | Mock Google Docs API |
| `gdrive` | `claw-gdrive` | 9005 | Mock Google Drive API |

Each service:
- Runs as a background process in the same container.
- Exposes a health endpoint (`/health`) for startup detection.
- Uses SQLite for state -- pre-seeded from the task's `environment/` directory.
- Is indistinguishable from the real API from the agent's perspective.

### How it works vs Harbor

In Harbor, stateful services required Docker Compose with separate containers for each service. This meant:
- Separate Dockerfiles per service container.
- Docker Compose networking for inter-container communication.
- Complex task setup with volume mounts for shared databases.

In BenchFlow, services are lightweight processes in the same sandbox:
- One Dockerfile with the services pre-installed.
- `pre_agent_hooks` starts them before the agent connects.
- The agent hits `localhost:9001` for Gmail -- no network complexity.
- Auto-detection: if a task's Dockerfile references `claw-gmail`, the service is started automatically.

### Example task structure (ClawsBench)

```
tasks/schedule-meeting-from-email/
├── task.toml
├── instruction.md          # "Read the email from Alice, create a calendar event..."
├── environment/
│   ├── Dockerfile          # FROM benchflow/claws-base (has all claw-* binaries)
│   ├── gmail.db            # Pre-seeded: email from Alice with meeting request
│   └── gcal.db             # Pre-seeded: existing calendar entries
├── solution/
│   └── solve.sh            # Oracle: curl commands to Gmail + GCal APIs
└── tests/
    └── test.sh             # Verify: check gcal.db has the new event
```

---

## Migration from Harbor

| Harbor pattern | BenchFlow equivalent | Key difference |
|----------------|---------------------|----------------|
| Docker Compose + FastMCP sidecar (#1316) | Scene with user + agent roles | No Compose needed; agents share sandbox |
| Multi-container multi-agent (#1462) | Scene with N roles + turns | Single container, process-level isolation via ACP |
| `agent_timeout` + single prompt | Turn with `None` prompt | Same behavior, just wrapped in Scene |
| Docker Compose services | `pre_agent_hooks` + `SERVICES` registry | Lightweight same-container sidecars |
| Separate verifier container | Same -- BenchFlow uses Harbor's `Verifier` | No change needed for task authors |

### Porting a Harbor task

1. **Task files**: No changes needed. BenchFlow reads the same `task.toml`, `instruction.md`, `Dockerfile`, and `tests/` structure.
2. **Single-turn**: Works out of the box with `bench eval create -t your-task -a gemini`.
3. **Multi-turn**: Add a `scenes` key to your YAML config (or pass `TrialConfig` in Python).
4. **Multi-agent**: Define multiple roles in the scene. No Docker Compose required.
5. **Services**: Declare in TrialConfig or auto-detected from Dockerfile.
