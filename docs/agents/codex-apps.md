# Codex Apps

Scored runs turn off Codex Apps, the account-connected tools such as Drive or Gmail. So a Codex run is graded on the sandbox alone.

To let a task use Apps:

```sh
bench run ./tasks -a codex --codex-apps-policy inherit
```

`codex` is a short name for `codex-acp`.

## Set the policy

| Where | Setting |
|---|---|
| CLI | `--codex-apps-policy inherit` or `--codex-apps-policy disabled` |
| Python | `EvaluationConfig(codex_apps_policy="inherit")`, or `bf.run(RolloutConfig(task_path=..., agent="codex", codex_apps_policy="inherit"))` |
| Run config YAML | a top-level `codex_apps_policy: inherit` |

```yaml
agent: codex
codex_apps_policy: inherit
```

The CLI flag wins over YAML. Without the flag, the YAML value stays.

Hosted source environments (`--source-env`) refuse the option. BenchFlow does not control their Codex.

## Defaults

| Run | Default |
|---|---|
| a scored task run (`purpose="task"`, `skip_verify=False`) | `disabled` |
| a rubric reviewer, or a run with `skip_verify=True` | `inherit` |
| any run with `--codex-apps-policy` | what you pass |

The default reads the run's configuration. It does not know whether an interactive run will verify, so set the policy yourself when it matters.

## Requirements

The policy needs BenchFlow's managed Codex ACP launcher at the registry's exact pin (`_CODEX_ACP_PACKAGE` in `agents/registry.py`, currently 2.0.1). The Codex CLI it runs must satisfy the range that the adapter declares (`^0.159.1` for 2.0.1, that is 0.159.1 or a later 0.159.x).

These are refused before the agent starts:

- another adapter version, or a Codex CLI outside that range;
- a custom launcher or executable override;
- an agent running as root;
- a malformed `CODEX_CONFIG`.

Other harnesses and local task MCP servers are not affected.

## What the trial records

`codex_apps_policy.json` in the trial folder records the requested and applied policy, the adapter and Codex versions, and the result of the check that Apps are off. It holds no config, credentials or raw probe output.

This receipt shows the configuration was enforced. It does not prove that no hosted tool was reachable, and it says nothing about account or network isolation.

## Next steps

- [Codex Apps enforcement](../reference/codex-apps-enforcement.md): how the policy is installed and checked.
- [Native harness](./harness.md): run Codex through `codex exec`.
- [Auth](./auth.md): Codex logins and API keys.
