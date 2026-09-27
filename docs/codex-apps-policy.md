# Codex Apps policy

Scored task rollouts (`purpose="task"`, `skip_verify=False`) disable native
Codex Apps by default. Set the canonical `codex_apps_policy` option to
`inherit` for tasks that intentionally need account-connected Apps, or
`disabled` to force enforcement in other rollout modes.

```sh
bench eval run --tasks-dir ./tasks --agent codex --codex-apps-policy inherit
```

Programmatic equivalents are `EvaluationConfig(codex_apps_policy="inherit")`
and `bf.run(RolloutConfig(task_path=..., agent="codex", codex_apps_policy="inherit"))`. Evaluation YAML uses a top-level
option in both supported layouts:

```yaml
agent: codex
codex_apps_policy: inherit
```

An explicit CLI option overrides YAML; omitting it preserves YAML. The policy
travels with worker/shard configuration and is reapplied at the worker's actual
agent connection. Explicit policy requests for hosted source environments are
rejected because they do not use this local managed ACP control.

Automatic mode inherits for reviewer and skipped-verification runs: this is a
configuration classifier, not a guarantee that every interactive run uses
`skip_verify`. It defaults disabled only when `purpose="task"` and
`skip_verify=False`.

The policy requires the managed Codex ACP launcher at the registry's exact pin (`_CODEX_ACP_PACKAGE` in `agents/registry.py`, currently 1.6.0). The actual native executable version (`codex-acp cli -V`) must satisfy the `@openai/codex` range that the installed adapter declares in its `package.json` (`^0.148.0` for 1.6.0, that is 0.148.x). Only exact and caret ranges are evaluated; any other range form fails closed. A caret range keeps its first nonzero component fixed, as in npm. A pre-release native version, or an adapter `package.json` that declares no `@openai/codex` dependency, is refused as well. Bumping the registry pin retargets this check, so there is no second copy of the version to update; re-run the credential-free adapter fixture below when bumping, because the `features list` override probe is the behavioral check for each admitted build. Manifest/custom launch replacements are rejected when disabled. Other adapter versions, native versions outside the declared range, custom launchers/executable overrides, root agents, malformed `CODEX_CONFIG`, and failed policy setup abort before ACP initialization. Other harnesses and local task MCP servers are unchanged.

BenchFlow sets `features.apps=false` in the adapter's supported `CODEX_CONFIG`
and installs a root-owned `/etc/codex/requirements.toml` inside the sandbox.
Existing requirements are preserved byte-for-byte. A native `features list`
probe must report Apps false even with a CLI enable override. An existing
incompatible policy fails rather than being replaced or loosened. Inheritance
also never removes a previously installed stricter policy.

Enforcement runs after installation and before every initial/role connection,
including reconnects without credential changes. The original recovery image
is captured earlier, before uploads and installation; this policy does not
move that baseline. Verifier-only recovery does not start a Codex solver.

`codex_apps_policy.json` records requested/applied policy, adapter and native versions,
managed requirements digest, and the successful override probe. A failed
reattempt removes an older success receipt. The receipt contains no config,
credentials, or raw probe output. This is configuration-enforcement evidence;
authenticated hosted-tool absence still needs a separate controlled native
conformance test. It does not establish general account, network, or
cross-harness isolation.

Sources: [adapter 1.6.0 CLI and CODEX_PATH handling](https://github.com/agentclientprotocol/codex-acp/blob/50bd611451c02868cc2b50bd6a7fc61ae5ef9b41/src/index.ts),
[Codex managed requirements](https://github.com/openai/codex/tree/rust-v0.148.0/codex-rs/config).

The credential-free adapter fixture exercised the actual managed launcher,
non-root agent context, user/CLI enable overrides, repeated application,
conflicting existing requirements, and unsafe symlink/writable policy files.
It used a disconnected disposable container for all probes and made no model
or hosted-catalog requests. This is not the authenticated conformance test.
