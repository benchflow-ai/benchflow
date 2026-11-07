# Why BenchFlow

A benchmark score is the last number out of a long process: a task, a sandbox, an agent and its model, a check of what the agent left, and a record of what it did. Han-chung Lee, a co-author of SkillsBench, argues that this process, the evaluation layer, [outlives any one model or training run](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#chat-eval-was-a-spreadsheet-agent-eval-is-a-system). For each part he says it needs, here is what BenchFlow does and where it stops.

## What an agent eval needs

His [post on evaluation infrastructure](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/) says of benchmark scores: "They are the artifacts left behind by a much larger system: task definitions, runtimes, snapshots, trace capture, state diffs, verifiers, judge calibration, replay, release gates, and production feedback."

| Need | What Han says | What BenchFlow does |
|---|---|---|
| Task definitions | Tasks, configuration and reports form a [control plane](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#rollouts), apart from the model, harness and runtime. | A task is a folder: an instruction, an environment, a verifier and usually a reference solution ([Write a task](../tasks/write-a-task.md)). |
| Runtimes | [Thousands of concurrent rollouts](https://leehanchung.github.io/blogs/2026/04/24/hidden-technical-debt-agent-runtime/#why-sandboxing-is-not-optional) each need their own filesystem, processes and network. | Every rollout gets a fresh sandbox built from the task's environment, on Docker, Daytona or Modal ([Sandboxes](../scale/sandboxes.md)). |
| Trace capture | A [structured record of each step](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#2-trace). | Each rollout keeps the agent's messages and tool calls and every model call through BenchFlow's proxy ([Read the results](../evaluate/read-results.md#what-the-agent-did)). BenchFlow neither sends nor receives OpenTelemetry, though an agent can send its own to a receiver you run ([Usage and telemetry](../scale/telemetry.md#opentelemetry)). |
| State diffs | Grading only the output [misses deleted tests or a wrong timezone](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#rollouts). | The verifier grades the sandbox after the agent stops. `--integrity audit` records what the agent did to the files and network the reward depends on ([Stop reward hacking](../tasks/reward-hacking.md#audit-how-each-reward-was-earned)). |
| Verifiers and release gates | "The verifier that gates a release can often be reused as, or translated into, the reward signal that shapes a policy." ([evaluation post](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#chat-eval-was-a-spreadsheet-agent-eval-is-a-system)) | `verifier/test.sh` writes a reward from 0 to 1 ([Verifiers](../tasks/verifiers.md)). `--fail-under` fails a CI build on it ([Run in CI](../scale/ci.md)), and training rows reuse it ([RL environment](../train/environment.md)). |
| Judge calibration | Use an LLM judge [only when no programmatic check will do](https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/#verifier). | A rubric review adds a reviewer agent to a test, and [its guide](../tasks/rubric-review.md#check-the-reviewer) checks the reviewer on three runs whose verdicts are known. Where no test fits, a model can grade alone ([Grade with a model](../tasks/rubric-review.md#grade-with-a-model-no-test-script)). |
| Agent or scorer at fault | Durable infrastructure [tells agent failures from scorer failures](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#evaluation-debt-and-durable-infrastructure). | The reference solution must score 1, and an agent that does nothing must score 0. A rollout whose verifier couldn't run is counted apart from the failures, as unscored ([Check a benchmark](../audit/check-a-benchmark.md)). |
| Snapshots and replay | [Replay from a snapshot](https://leehanchung.github.io/blogs/2026/04/24/hidden-technical-debt-agent-runtime/#why-sandboxing-is-not-optional) debugs a long run without running it again. | `bench regrade` reruns a fixed verifier on the workspaces a run saved ([Resume, retries and regrade](../scale/resume.md)). `bench branch`, which forks a rollout from a saved sandbox snapshot, is covered only by its [reference entry](../reference/commands.md#bench-branch). |

BenchFlow doesn't collect production feedback, the last item on his list: it runs agents on tasks in sandboxes.

## Harness, runtime and environment are separate

His [runtime post](https://leehanchung.github.io/blogs/2026/04/24/hidden-technical-debt-agent-runtime/) puts the layers in one sentence: "The agent is the harness plus the model, running inside the runtime." The [harness](https://leehanchung.github.io/blogs/2026/05/08/hidden-technical-debt-agent-harness/#agent-harness) is the orchestration around the model: prompts, tools, the rollout loop, memory, sub-agents and guardrails. The [runtime](https://leehanchung.github.io/blogs/2026/04/24/hidden-technical-debt-agent-runtime/#what-an-agent-runtime-actually-is) is the machine around both: compute, filesystem, network, state and lifecycle. Tasks come [bundled with their environment](https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/#canonical-loop), and on experiments he writes: "Decouple the task from the harness." ([evaluation post](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#experimentation-not-benchmarking))

His harness post charts one model [17.5 points apart in two harnesses](https://leehanchung.github.io/blogs/2026/05/08/hidden-technical-debt-agent-harness/#from-research-to-production). BenchFlow makes each layer a setting of its own, so a comparison can change one at a time:

- The harness is the agent (`-a`), and its model is set apart (`-m`). Most agents run over ACP, the Agent Client Protocol; Claude Code and Codex also run as their own command-line programs with `--harness native`. Each release pins the agents' versions ([Agents and models](../evaluate/agents.md)), and your own agent runs the same way ([Run your own agent](../evaluate/your-own-agent.md)).
- The runtime is the sandbox, chosen with `--sandbox`: Docker, Daytona, Modal and others ([Sandboxes](../scale/sandboxes.md)).
- The task brings its own environment and verifier, and the verifier reaches the sandbox only after the agent stops ([Task format](../tasks/task-format.md)).

`bench eval compare` checks that two runs differ only in the settings you name with `--vary` ([Compare runs](../audit/compare-runs.md#check-that-the-runs-differ-only-where-you-meant)).

## The rollout is the unit

His [taxonomy of RL environments](https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/#tasks) takes the word from reinforcement learning: "A single run from start to completion is an episode, and the process of executing a policy to generate a trajectory is called a rollout." He tells apart the trajectory a trainer reads (states, actions and rewards) from the trace an observability system reads (tool calls and observations). His [evaluation post](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#rollouts) makes the rollout, one bounded episode recorded as a trace, its core artifact.

In BenchFlow a rollout is one agent, driving one model, on one task in a fresh sandbox; the command line calls it a trial. It keeps both of Han's records: the agent's events and model calls under `trajectory/`, and rows with the reward under `trainer/` ([Trajectory and rollout formats](../train/formats.md)). [How BenchFlow works](./how-it-works.md) names the parts and follows one rollout through them.

## Report scores that hold up

pass@k is [the probability that at least one of k sampled tries passes](https://leehanchung.github.io/blogs/2025/09/08/pass-at-k/#conclusion), estimated from a finite number of tries. With n tries and c passes, the naive estimate 1 − (1 − c/n)^k assumes draws with replacement and [comes out low](https://leehanchung.github.io/blogs/2025/09/08/pass-at-k/#the-biased-estimator-problem). The Codex paper's estimator, 1 − C(n − c, k) / C(n, k), is [unbiased, with the lowest variance of the unbiased estimators](https://leehanchung.github.io/blogs/2025/09/08/pass-at-k/#u-statistics). For 10 tries with 3 passes, it puts pass@5 at [91.7%, against the naive estimate's 83.2%](https://leehanchung.github.io/blogs/2025/09/08/pass-at-k/#practical-example).

That post gives no intervals. His evaluation post asks for one with every number: "Pair the runs and report the uncertainty with a confidence interval and a minimum detectable effect alongside every number, so we can discount a “gain” that sits inside the noise band." ([evaluation post](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#experimentation-not-benchmarking))

`bench eval metrics` reports pass@k and pass^k with that estimator, and `bench results A B` pairs two runs task by task. Neither prints an interval in 0.9.1, so [Compare runs](../audit/compare-runs.md) computes intervals and paired tests from the trials. In its example, an agent passed 7 of 9 tries with skills and 6 of 9 without, and the difference's 95% interval runs from a 37-point loss to a 59-point gain.

## Further reading

- [A Taxonomy of RL Environments for LLM Agents](https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/) (21 March 2026)
- [Hidden Technical Debt of AI Systems: Agent Runtime](https://leehanchung.github.io/blogs/2026/04/24/hidden-technical-debt-agent-runtime/) (24 April 2026)
- [Hidden Technical Debt of AI Systems: Agent Harness](https://leehanchung.github.io/blogs/2026/05/08/hidden-technical-debt-agent-harness/) (8 May 2026)
- [Hidden Technical Debt of AI Systems: Agent Evaluation Infrastructure](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/) (13 June 2026)
- [Statistics for AI/ML, Part 4: pass@k and Unbiased Estimator](https://leehanchung.github.io/blogs/2025/09/08/pass-at-k/) (8 September 2025)
