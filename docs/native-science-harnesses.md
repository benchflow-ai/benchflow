# OpenScience and DeepSeek Harness

BenchFlow includes two independent science-oriented native ACP harnesses. They
share DeepSeek-compatible routing in the integration matrix, but they are not
the same agent and do not share an upstream runtime.

## OpenScience

- Agent ID: `openscience`
- Upstream: `synthetic-sciences/OpenScience`
- Pin: `v2.0.127`
- Native server: `openscience acp`
- Skill root: `$HOME/.claude/skills`

BenchFlow installs the tagged release with upstream checksum verification and
launches it through a small configuration writer. State lives under
`$BENCHFLOW_AGENT_HOME/.openscience-benchflow`. The launcher disables bundled
skills, project configuration, update checks, environment bootstrap, LSP
downloads, title generation, nested sandboxing, and all tools except the local
file, shell, notebook, artifact, planning, and skill tools explicitly allowed
by the generated policy.

The launcher selects an OpenScience bundled provider adapter from
`BENCHFLOW_PROVIDER_PROTOCOL`:

- `openai-completions` uses `@ai-sdk/openai-compatible`;
- `anthropic-messages` uses `@ai-sdk/anthropic`;
- `openai-responses` is rejected explicitly because that route has not been
  given a native BenchFlow conformance contract.

For a LiteLLM run, the launcher registers the gateway alias under the custom
OpenScience provider `benchflow`. Direct runs retain the resolved BenchFlow
provider identity. Generated configuration contains only an environment
reference to the API key, never its literal value.

OpenScience creates an internal per-session scratch directory even when ACP is
started with the benchmark project as its `cwd`. BenchFlow therefore installs a
launcher-owned instruction that identifies `$BENCHFLOW_WORKSPACE` as the task
workspace and requires shell `workdir` and file paths to target that directory;
session scratch remains internal state. The native smoke runs as the sandbox
UID and verifies that a tool call writes a probe into the mounted task workspace.

## Official DeepSeek Harness

- Agent ID: `deepseek-harness`
- Alias: `dsh`
- Upstream: `deepseek-ai/deepseek-harness`
- Pin: `@deepseek-ai/dsh@0.1.6-alpha.2`
- Native server: `dsh --profile acp`
- Skill root: `$HOME/.agents/skills`

BenchFlow installs DSH and Node 22.20.0 under `/opt/benchflow`, not into the
task image's global Node prefix. The exact npm package integrity is checked
against the pinned SHA-512 value. The launcher writes a Cordis patch beneath
`$BENCHFLOW_AGENT_HOME/.dsh-benchflow`, selects the BenchFlow model (the
LiteLLM alias takes priority when present), selects DSH's native protocol from
`BENCHFLOW_PROVIDER_PROTOCOL`, and disables telemetry, HMR, user plugins, and
web tools. `openai-completions` maps to DSH `chat-completions`, while
`anthropic-messages` maps to DSH `messages`. OpenAI Responses is rejected
explicitly because DSH does not implement that wire format. DSH's inner permission
mode is `danger-full-access` because the outer BenchFlow container remains the
authoritative sandbox and approval boundary.

The registered `deepseek` provider defaults to OpenAI Chat Completions at
`https://api.deepseek.com/v1`. A run can select DeepSeek's official Anthropic
Messages surface with
`--agent-env BENCHFLOW_PROVIDER_PROTOCOL=anthropic-messages`; BenchFlow then
resolves `https://api.deepseek.com/anthropic`. For a custom compatible service,
also provide `BENCHFLOW_PROVIDER_BASE_URL` explicitly. DSH owns the final
`/chat/completions` or `/v1/messages` suffix.

For the first-party Anthropic API, use the explicit model prefix
`anthropic-direct/<model>`. It routes `ANTHROPIC_API_KEY` to
`https://api.anthropic.com` with the Anthropic Messages protocol. The legacy
`anthropic/<model>` prefix intentionally remains unregistered because existing
agent-native and LiteLLM paths depend on that behavior.

The DSH ACP model option uses opaque JSON route values, so BenchFlow does not
drive that option. The generated launch profile owns model selection instead;
the standard `reasoning_effort` option remains available through ACP.

## Extended ACP capabilities

Both harnesses accept task-declared MCP servers. OpenScience v2.0.127 advertises
ACP MCP, but its dynamically added remote connector validates against persisted
execution config and fails to connect, so BenchFlow writes an isolated native
`mcp` config before launch. DSH receives HTTP MCP through ACP `session/new`.
The client can send SDK-backed image content blocks. OpenScience advertises image
support unconditionally, while DSH reports it only when the selected
model/provider supports image input.

Session recovery follows the server's advertised method. BenchFlow prefers
`session/resume` when present and falls back to the older `session/load`
capability. OpenScience advertises both; DSH advertises resume.

OpenScience does not expose a separate reasoning-effort ACP configuration
option in v2.0.127. Its variants are represented in model choices, so the
registry deliberately leaves `acp_effort_config_id` empty. DSH exposes the
separate `reasoning_effort` option.

See `docs/agent-feature-grid.md` for the complete capability and validation
matrix, including availability-gated sandbox backends.

## Skill fidelity

Both integrations suppress upstream default/bundled skill roots. A no-skill
rollout therefore exposes an empty BenchFlow-owned root. A with-skill rollout
mounts only the selected skills into the registry path above. Neither launcher
scans host project configuration for additional skills.

## Upgrading OpenScience

1. Select an immutable upstream release and record its tag, commit, and license.
2. Update `_OPENSCIENCE_VERSION` and the immutable tagged installer URL together.
3. Review upstream config, environment flags, ACP model-option behavior, tool
   permissions, and skill discovery for drift.
4. Run `tests/test_openscience_agent.py`, registry invariants, and the native
   Docker smoke:

   ```bash
   conda run -n bf_openscience_harness_dev \
     python tests/integration/native_acp_harness_smoke.py openscience
   ```

5. Confirm checksum verification, exact version, ACP initialize/session/prompt,
   the full reasoning/tool/skill/model/error/timeout lifecycle over both
   `/v1/chat/completions` and `/v1/messages`, explicit OpenAI Responses
   rejection, and removal of temporary Docker resources.

## Upgrading DeepSeek Harness

1. Select an immutable `dsh-v*` tag and record its commit, package version,
   Node engine range, license, npm integrity, and shasum.
2. Update `_DEEPSEEK_HARNESS_VERSION` and
   `_DEEPSEEK_HARNESS_INTEGRITY` together.
3. Review the shipped `acp` bundle, Cordis row IDs, `model-control.ts` option
   IDs, DeepSeek adapter protocol/base-URL behavior, permissions, telemetry,
   plugin discovery, and skill-filesystem schema.
4. Run `tests/test_deepseek_harness_agent.py`, registry invariants, and:

   ```bash
   conda run -n bf_openscience_harness_dev \
     python tests/integration/native_acp_harness_smoke.py deepseek-harness
   ```

5. Confirm the exact npm integrity, protocol version 1, `model` and
   `reasoning_effort` options, mock prompt/cancel behavior, both the
   `/v1/chat/completions` and `/v1/messages` request paths, explicit rejection
   of OpenAI Responses, and cleanup. The Anthropic-format lane exercises the
   full reasoning, tool, skill, model, cancellation, timeout, provider-error,
   recovery, workspace, and ACP-trajectory lifecycle rather than only a basic
   text response.
