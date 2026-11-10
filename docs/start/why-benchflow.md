# Why BenchFlow

Han-chung Lee, a co-author of SkillsBench, splits evaluation infrastructure into "a control plane that decides what to run and whether the result should change what we ship, and a data plane that runs the agent and records what actually happened" ([Agent Evaluation Infrastructure](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/)). He calls the data plane the hard part: "Scoring the final response is the easy part. The harder problem is building a data plane rich enough that the score is explainable, reproducible, and actionable." ([rollouts](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#rollouts))

Running an agent on a task in a sandbox, with a harness you can swap, is common ground: his [harness post](https://leehanchung.github.io/blogs/2026/05/08/hidden-technical-debt-agent-harness/#the-bitter-lesson-for-agent-harness) names BenchFlow among the evaluation frameworks that let you swap harnesses, and charts one model [17.5 points apart in two harnesses](https://leehanchung.github.io/blogs/2026/05/08/hidden-technical-debt-agent-harness/#from-research-to-production). In BenchFlow the agent (`-a`), its model (`-m`) and the sandbox (`--sandbox`) are separate settings, and each release pins the agents' versions ([Agents and models](../evaluate/agents.md)). Where BenchFlow goes further is the record each run leaves and the decisions it reads from that record. It is the evaluation infrastructure behind SkillsBench, Robo Use, a physics research benchmark and a design benchmark judged by people, and much of what follows came from their failures.

These pages cover 0.9.2. "(0.10)" marks what needs the 0.10.0rc3 pre-release ([Install](./install.md#install-the-010-prerelease)).

## What each rollout records

A rollout is one agent, driving one model, on one task in a fresh sandbox ([How BenchFlow works](./how-it-works.md)). BenchFlow keeps:

- **What the agent did:** its messages and tool calls, under the trial's `trajectory/` ([Read the results](../evaluate/read-results.md#what-the-agent-did)).
- **Each model call that passes through BenchFlow's model proxy,** with the model, effort, sampling and output cap it carried, its tokens, its HTTP status and its finish reason, in `trajectory/llm_trajectory.jsonl`. The record is what each request carried; it can't show whether the provider honoured it. A Claude or ChatGPT subscription login goes around the proxy, and so does an agent whose manifest sets `model_route = "agent"` (0.10); for those, BenchFlow has only what the agent itself reports, if anything ([Usage and telemetry](../scale/telemetry.md)).
- **The workspace the agent left,** graded after the agent stops. Before the test runs, BenchFlow stops the agent's processes and removes files the agent could plant to run code inside the test, and since 0.9.2 the git hooks it added ([Stop reward hacking](../tasks/reward-hacking.md)). The saved workspace lets `bench regrade` score the run again with a fixed verifier ([Resume, retries and regrade](../scale/resume.md)).
- **A sandbox to fork from.** `bench branch` runs a task up to a checkpoint you choose, snapshots the sandbox, and starts each labelled child from that snapshot, with a fresh agent session or the parent's conversation reloaded, to be scored by the task's verifier. The snapshot holds the sandbox's files and copies of its SQLite databases, and embodied tasks refuse it. Only its [reference entry](../reference/commands.md#bench-branch) covers it.

In 0.10, each trial also records how its model ended (completed, a refusal, a provider error), the effort that ran and the tools the agent was offered. The job checks the calls against the run as it goes: too many trials cut at the output cap, or calls that carried another model or effort than the run asked for, fail it ([The run checks](../evaluate/before-you-run.md#the-run-checks)). Each trial also writes its trace as OpenTelemetry spans and can send them to your collector ([Usage and telemetry](../scale/telemetry.md#opentelemetry)).

## What is decided from the record

His evaluation post asks infrastructure to "distinguish the agent failing the task from the scorer failing the agent" ([evaluation debt](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#evaluation-debt-and-durable-infrastructure)), and to "Pair the runs and report the uncertainty with a confidence interval and a minimum detectable effect alongside every number" ([experimentation](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#experimentation-not-benchmarking)).

- **Controls before the spend.** A task's reference solution should score 1, and an agent that does nothing should score 0. [Check a benchmark](../audit/check-a-benchmark.md) runs both; on 0.9.1, 12 of Terminal-Bench 2.0's 89 tasks failed one. In 0.10 the run card prints, before the first trial, the settings that change a score and the controls already run on the same task content, matched by its digest, with the commands for the missing ones ([The run card](../evaluate/before-you-run.md#the-run-card-010)).
- **Scores with intervals (0.10).** Every figure counts by one rule, `benchflow.score.v1`: every finished trial is in n, and only a scored pass counts 1. Each score comes with a 95% interval over tasks, and `bench results A B` gives the gap paired by task with its interval, and warns when the interval holds 0 or the gap is under 3 points. On 0.9.2, [Compare runs](../audit/compare-runs.md) computes the same from the trials. Neither release gives a minimum detectable effect.
- **Runs that differ in one setting.** `bench eval compare` checks that two runs differ only in the settings you name with `--vary`, and with `--on-mismatch raise` stops on any other difference; it doesn't compare the skills a run used ([Compare runs](../audit/compare-runs.md#check-that-the-runs-differ-only-where-you-meant)).
- **pass@k without the bias.** `bench eval metrics` reports pass@k and pass^k with the estimator from the Codex paper ([Chen et al., 2021](https://arxiv.org/abs/2107.03374)), 1 − C(n − c, k) / C(n, k). For 10 tries with 3 passes it puts pass@5 at [91.7%, where the naive estimate gives 83.2%](https://leehanchung.github.io/blogs/2025/09/08/pass-at-k/#practical-example).
- **A setup to match (0.10).** A task set's `run-reference.toml` declares the setup its numbers used, and a strict one refuses a run that differs unless you pass `--accept-deviations` ([Publish a benchmark](../tasks/publish.md#declare-the-setup-runs-should-match-010)).

## Built for research benchmarks

Each of these is in use on a benchmark we run:

| What | In use for | Release | Page |
|---|---|---|---|
| A reviewer agent reads the frozen workspace and the trajectory; a trial passes only if the tests pass and every blocker in the rubric holds | grading long research tasks in physics | 0.9.2 | [Rubric review](../tasks/rubric-review.md) |
| Blind pairs of outputs, votes from people or a model, and rankings with intervals and rater agreement | judging designs | 0.10 | [Human preferences](../tasks/human-preferences.md) |
| Static checks of skill folders, and the same agent run with and without a skill | SkillsBench | 0.9.2 | [Skills](../evaluate/skills.md) |
| Simulated robot tasks, a verifier that reads the robot's physical state, and a trace of each step | Robo Use | 0.9.2; on Daytona in 0.10 | [Embodied tasks](../evaluate/embodied.md) |
| An integrity audit of what the agent did to the files and network its reward depends on, as a verdict that never changes the reward | probing SkillsBench's verifiers | 0.9.2 | [Stop reward hacking](../tasks/reward-hacking.md#audit-how-each-reward-was-earned) |
| A published score recomputed from its trials, and a matched sample run again | a SkillsBench leaderboard entry | a guide | [Reproduce a published score](../audit/reproduce.md) |

## Where it stops

His post asks for more than BenchFlow does today:

- **The state change at each step.** "If your eval does not capture the state delta, it is not flexible enough to evaluate tasks that are stateful." ([environment](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/#4-environment)) BenchFlow grades the final state, and records state at each step only in embodied tasks.
- **Checkpoints with the whole context.** A branch saves the sandbox's files and databases. The agent's session, memory, clocks and network fixtures aren't saved, and replaying a known trajectory works only for OpenHands (`bench eval continue`).
- **Memory.** Skills can be switched on and off (`bench skills eval`); no record shows what an agent read from or wrote to its memory.
- **Faults on purpose.** No run injects tool failures, latency or a revoked permission.
- **A minimum detectable effect** beside each number: 0.10 gives intervals only.
- **Production.** Failures seen in production don't flow back into a task suite, and nothing monitors a deployed agent. BenchFlow runs agents on tasks in sandboxes.

## Further reading

- [A Taxonomy of RL Environments for LLM Agents](https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/) (21 March 2026), where BenchFlow took the word rollout ([How BenchFlow works](./how-it-works.md#why-the-rollout))
- [Hidden Technical Debt of AI Systems: Agent Runtime](https://leehanchung.github.io/blogs/2026/04/24/hidden-technical-debt-agent-runtime/) (24 April 2026)
- [Hidden Technical Debt of AI Systems: Agent Harness](https://leehanchung.github.io/blogs/2026/05/08/hidden-technical-debt-agent-harness/) (8 May 2026)
- [Hidden Technical Debt of AI Systems: Agent Evaluation Infrastructure](https://leehanchung.github.io/blogs/2026/06/13/hidden-technical-debt-agent-evaluation-infra/) (13 June 2026)
- [Statistics for AI/ML, Part 4: pass@k and Unbiased Estimator](https://leehanchung.github.io/blogs/2025/09/08/pass-at-k/) (8 September 2025)
