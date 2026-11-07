# Agent-phase network experiment

Status: an opt-in experiment. `bench run` and the task schema defaults are
unchanged; a later release may change the default once it is measured.

Run the experiment with:

```sh
python -m benchflow.experimental.agent_network --no-network-by-default -- ./task --agent claude-agent-acp --model PROVIDER/MODEL
```

The flag supplies the existing `bench run --config-override` mechanism with
`{"agent":{"network_mode":"no-network","allowed_hosts":[]}}`. It applies
to every task in that invocation, including tasks that request an agent
allowlist. With no flag, arguments pass through unchanged. The wrapper refuses
another `--config` or `--config-override` instead of silently replacing it.
For a custom overlay, invoke `bench run` directly. This is an experimental
launcher, not a new flag on the public `bench run` command.

The overlay is recorded through the existing task configuration and resume
checks. Setup, agent installation, data staging and the verifier retain their
existing policies. Immediately before the first agent turn, the current
no-network implementation restricts the agent UID and switches off hosted web
tools. A sandbox-local guarded model gateway remains reachable; supported
native subscription transports use their existing restricted inference route.
Root agents are refused because a UID firewall cannot confine root. Unsupported
backends or model routes must fail the existing capability gates.

The purpose is to measure how many tasks genuinely need agent-phase internet,
without allowing answer fetching through package repositories, readers, search,
or model-provider tools. The denylist fixes in the accompanying change reduce
known relay routes; arbitrary mirrors, binary encodings, encoded addresses and
services acting on partial uploads remain outside a denylist's guarantees.
An allowlist or no-network policy is the stronger choice.

Before any default change, it needs paired real runs with identical task digests, model, effort and budgets on Docker and Daytona:
ordinary offline coding; pip/git dependency acquisition; task-owned Compose
services; browser tasks; native subscription and gateway routes; multi-step
trials and resume. Measure setup versus agent time, refusals, reward changes
and infrastructure failures. Include a canary that attempts DNS, direct IP,
HTTP(S), UDP and provider-hosted fetches, plus an ordinary model-call control.
No model-driven paired campaign is claimed by this spike.

Unit tests cover opt-in argument construction, recorded configuration, setup
and verifier policy separation, unchanged public defaults and overlay conflicts.
The existing agent-network tests exercise offline connection, the local gateway
marker and refusal of a root agent. A default rollout should also expose explicit
task exceptions and the effective policy in the run card; those are release
integration decisions rather than hidden exceptions in this wrapper.
