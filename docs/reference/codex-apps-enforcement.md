# Codex Apps enforcement

How BenchFlow installs and checks the [Codex Apps](../agents/codex-apps.md) policy. This page is for maintainers and security reviewers.

## Version check

The policy requires the managed Codex ACP launcher at the registry's exact pin (`_CODEX_ACP_PACKAGE` in `agents/registry.py`).

- The native executable's version (`codex-acp cli -V`) must satisfy the `@openai/codex` range in the installed adapter's `package.json`.
- Only exact and caret ranges are evaluated. Any other range form fails closed.
- A caret range keeps its first nonzero component fixed, as in npm.
- A pre-release native version is refused.
- An adapter `package.json` with no `@openai/codex` dependency is refused.

Bumping the registry pin retargets this check, so there is no second copy of the version to update. Re-run the credential-free adapter fixture below when bumping. Its `features list` override probe is the behaviour check for each admitted build.

When the policy is `disabled`, manifest or custom launch replacements are rejected. Failed policy setup aborts before ACP initialization.

## What BenchFlow installs

- `features.apps=false` in the adapter's supported `CODEX_CONFIG`.
- A root-owned `/etc/codex/requirements.toml` inside the sandbox.

Existing requirements are kept byte for byte. A native `features list` probe must report Apps off, even with a CLI enable override. An existing incompatible policy fails the run rather than being replaced or loosened. `inherit` never removes a stricter policy installed earlier.

## When it runs

Enforcement runs after installation and before every initial or role connection, including reconnects without credential changes.

The recovery image is captured earlier, before uploads and installation. This policy does not move that baseline. Verifier-only recovery does not start a Codex solver.

## The receipt

`codex_apps_policy.json` records the requested and applied policy, the adapter and native versions, the managed requirements digest, and the successful override probe. A failed retry removes an older success receipt.

The receipt contains no config, credentials or raw probe output. It is evidence that the configuration was enforced. Proving that authenticated hosted tools are absent needs a separate, controlled native conformance test. The receipt does not establish account, network or cross-harness isolation.

## Evidence

The credential-free adapter fixture exercised:

- the actual managed launcher, in a non-root agent context;
- user and CLI enable overrides, and repeated application;
- conflicting existing requirements;
- unsafe symlinked or writable policy files.

It used a disconnected, disposable container for all probes and made no model or hosted-catalog requests. It is not the authenticated conformance test.

The fixture was last run against adapter 1.6.0. The pin is now 2.0.1, so re-run it at the next bump. The deterministic tier's receipts show the override probe holding Apps off on Codex 0.159.3.

Sources:

- [adapter 2.0.1 CLI and `CODEX_PATH` handling](https://github.com/agentclientprotocol/codex-acp/blob/7a8e00fe46b299264f5ebe9f250288636f1485cc/src/index.ts), unchanged since 1.13.1;
- [Codex managed requirements](https://github.com/openai/codex/tree/rust-v0.159.3/codex-rs/config), at the release 2.0.1 resolves to.

## Next steps

- [Codex Apps](../agents/codex-apps.md): the user guide.
- [Native harness internals](../contributing/native-harness-internals.md): the Codex pins.
