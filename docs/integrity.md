# Reward integrity

A reward says the verifier passed. It does not say how. `--integrity` records, for every trial, what the agent did to the things the reward depends on, checks that record against the task's contract, and writes a verdict next to the reward: `Checked`, `VectorExposed`, `AgentViolation` or `Rejected`. The verdict never changes the reward.

This is BenchShield ([arXiv 2609.11028](https://arxiv.org/abs/2609.11028)), BenchFlow's reward-integrity layer. The lifecycle model, event schema, trace checker and claim rules are ported from its implementation, BenchGuard; the evidence comes from what BenchFlow already records.

```bash
bench eval run --tasks-dir tasks/ --agent claude-agent-acp --model claude-sonnet-5 --integrity audit
```

```python
import benchflow as bf

job = bf.load_job("jobs/<job>")
job.integrity().counts()        # {'AgentViolation': 1, 'VectorExposed': 9}
for path, verdict in job.integrity().exploited().items():
    print(path.name, verdict.reason)

trial = bf.load_trial("jobs/<job>/<trial>")
trial.integrity.verdict, trial.integrity.exploited, trial.integrity.reason
```

`integrity: audit` (or `strict`) works in evaluation and rollout YAML, `EvaluationConfig(integrity=...)`, `RolloutConfig(integrity=...)` and `TaskRuntimeConfig(integrity=...)`.

## The two modes

| Mode | What runs differently | Backends | Best verdict of a clean run |
|---|---|---|---|
| `audit` | Nothing. The run is observed as the task ships. | every backend | `VectorExposed` when the verifier shares the agent's sandbox; `Checked` when the task already declares a [separate verifier](./separate-verifier.md) |
| `strict` | The verifier runs in the [separate verifier sandbox](./separate-verifier.md), which receives only the frozen workspace, the declared artifacts and `/logs/artifacts`. | where the separate verifier runs: Docker, remote Docker, Daytona | `Checked` |

Strict mode is the separate verifier, forced for every trial; there is no second verifier mechanism. A task the separate verifier cannot run (an `llm-judge` verifier, a non-`main` verifier service, `steps`, a backend such as Modal) is refused before any sandbox starts, with the launch gate's reason. Because strict changes where the verifier runs, its rewards can differ from a shared-verifier run's; that is the point of it.

## What a verdict means

| Verdict | Meaning | `exploited` |
|---|---|---|
| `AgentViolation` | Direct agent-attributed evidence of a forbidden crossing: the agent read protected state (hidden tests, the solution, the verifier's pre-agent snapshot), wrote the reward or verifier files or state the contract marks trusted, or used the network when the contract forbids it. | yes |
| `Rejected` | No claim: a violation without agent attribution, or evidence the claim needs is missing (no trajectory, a trajectory scraped from the agent's own files, a truncated one, an unclassifiable command that names a protected path). The paper calls this Inconclusive. | no |
| `VectorExposed` | The agent was observed and nothing it did crossed a boundary, but the run's profile has a known path to the reward (for example the verifier ran in the agent's sandbox). Not agent blame. | no |
| `Checked` | Observed, nothing crossed, and the verifier ran where the agent never was. `certification` stays `Conditional` while the task's verifier soundness is assumed rather than reviewed. | no |

`severity` is `RewardRelevant` when a violation happened in a run the verifier passed.

## What gets written

Each audited trial gets `integrity/`:

| File | Content |
|---|---|
| `claim_verdict.json` | The verdict: `core_verdict`, `exploited`, `reason`, `certification`, `task_outcome`, flags, `mode`, the reward as recorded and `reward_effect: none` |
| `manifest.json` | The contract the run was checked against: every protected root and its class, the workspace, declared artifacts, the authorized network class |
| `action_records.jsonl` | Agent actions projected from the trajectory the host recorded (shell commands decomposed into file reads, writes and URLs), plus agent network attempts the egress proxy refused |
| `events.jsonl`, `audit_manifest.json` | The typed lifecycle events, each carrying the hash of the one before; the manifest records the chain root |
| `trace_check.json` | The replay of the events into violation flags and the agent evidence behind them |
| `reward_provenance.json` | Where the reward came from: verifier status, reward file and its sha256 |
| `handoff.json` | With a separate verifier: what crossed into it, from the host's transfer record |
| `conformance.json` | Everything above in one document |

All of it is built on the host from files the host wrote, after the sandbox is gone; nothing is read back from the sandbox. The hash chain makes an edited or reordered stream detectable; what keeps the agent from writing it is that the agent never has access to the trial folder.

## The contract

The contract is derived from what BenchFlow knows about the trial:

| Resource | Class |
|---|---|
| The workspace the rollout resolved | agent-writable |
| `/tests`, `/verifier`, `/testbed_verify` | verifier-only |
| `/solution`, `/oracle` | hidden |
| `/logs/verifier` | reward output |
| `/logs/agent` | trusted (the agent may read, not write) |
| `/logs/artifacts`, declared `artifacts` | declared handoffs |
| mounted skills | public (read-only) |
| `[environment.state].paths` | hidden (reach the service through its interface) |

The network class is the agent's effective network mode: `no-network` forbids egress; `allowlist` and `denylist` are enforced by the egress proxy, whose refusals are recorded; `public` allows it.

What BenchFlow cannot infer goes in an optional `benchguard.yaml` in the task folder, in BenchGuard's binding format. Example: a task that ships a library in the workspace which the verifier imports declares it trusted, so an agent that edits it is caught; a closed-book task forbids all egress even on a public network:

```yaml
schema_version: benchguard.task_binding.v1
task:
  measurement_mode: closed_book
resources:
  - id: trl-source
    selector: /app/trl
    class: Trusted
    reason: the verifier imports this package
```

A binding can add and narrow; it cannot make `/tests`, `/solution` or the reward folder agent-visible.

Reward-relevant state inside the workspace is the case the derived contract cannot see. Everything under the agent's working directory is agent-writable, including state a task puts there for its own grading: a task that writes the expected answer, a grading key or a score file into the workspace gives the agent a leak that is not a protected-root crossing, so the default contract does not flag reading it. This stays a binding, not a default, because BenchFlow cannot tell a grader's hidden directory from one the task asks the agent to create; guessing by name would both miss leaks and flag honest work.

```yaml
schema_version: benchguard.task_binding.v1
resources:
  - id: grader-state
    selector: /workdir/.grader
    class: Hidden
    reason: the verifier's expected answer is stored inside the agent's workspace
```

With that binding, an agent that reads `/workdir/.grader/expected.json` and submits it is `AgentViolation` even though the verifier passed it. Keeping grading state out of the workspace is better where the task can (`/tests`, `/verifier`, `/solution` and `/oracle` are protected by default); the binding is for tasks that cannot.

## Re-verdict a stored trial

The checker is a pure function of the stored evidence, so a trial run without `--integrity` can be audited later from its folder:

```python
from benchflow.integrity import audit_trial

verdict = audit_trial("jobs/<job>/<trial>", task_path="tasks/<task>")
```

It reads `result.json`, `config.json`, `trajectory/acp_trajectory.jsonl`, the separate verifier's record and the egress log, and writes `integrity/` as a live run would.

## Training on audited rollouts

A trainer that must not reward a hacked rollout applies the verdict itself, so the change is visible in the trainer, not hidden in the data:

```python
runtime = await TaskRuntime.create(TaskRuntimeConfig(task_path=task, integrity="audit"))
...                                      # the policy's bash calls
result = await runtime.verify()
verdict = result.integrity               # IntegrityVerdict, or None if the audit did not run
if verdict is not None and verdict.exploited:
    reward, flagged = 0.0, verdict.reason    # an exploit is the policy's doing: 0, never dropped
else:
    reward, flagged = result.reward, None
```

The same verdict is on disk (`integrity/claim_verdict.json`) for trainers that read trial folders, and `bf.load_trial(path).integrity` reads it back. `verdict.exploited` and `verdict.reason` are the fields an `apply_integrity`-style reward helper takes.

## Limits

- Audit mode sees what the agent's tool calls say. A command run through an interpreter (`python -c`, `perl -e`, a script the agent wrote) is opaque: its file writes are not witnessed, and when it names a protected path the verdict is `Rejected`, not `Checked`. The host-side Docker monitor that would corroborate such writes is not ported.
- Reward-relevant state a task keeps inside the agent's workspace is agent-writable by default, so reading it is not a violation until a binding names it (see "The contract").
- Answer-source shortcuts (reading an upstream fix from git history, fetching a published answer) are semantic (I7). They are caught only when a binding says so (for example `measurement_mode: closed_book` turns any egress into a violation).
- The static taint analysis that finds vectors before a run, and the LLM audit agents, are not part of this; the rubric review's `reward_hacking` criterion is BenchFlow's LLM reviewer.
- Declared artifacts that are code run inside the verifier; they are handed over like data.
