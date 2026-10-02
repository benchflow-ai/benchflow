# Agent feature grid

This is the canonical capability contract for BenchFlow's built-in agent
integrations. A supported capability is wired and covered by an executable
test; conditional and deferred cells state the remaining boundary.

| Feature | OpenScience | DeepSeek Harness |
|---|---|---|
| Agent ID | `openscience` | `deepseek-harness` (`dsh`) |
| Native protocol | ACP over stdio | ACP over stdio |
| Provider wire formats | OpenAI Chat Completions; Anthropic Messages | OpenAI Chat Completions; Anthropic Messages |
| OpenAI Responses | Unsupported; fails closed | Unsupported; fails closed |
| Model selection | ACP `model` option and compatibility `session/set_model` | Launch-owned model; ACP model option is opaque |
| Reasoning effort | No separate ACP option; upstream variants are encoded in model choices | ACP `reasoning_effort` option |
| Local tools | Yes | Yes |
| BenchFlow-selected skills | Yes; `$HOME/.claude/skills` | Yes; `$HOME/.agents/skills` |
| Task MCP | Native config HTTP, SSE, and stdio | ACP HTTP |
| Image prompt | Advertised and supported | Conditional on selected model/provider capability |
| Session recovery | `session/resume` preferred; `session/load` also advertised | `session/resume` |
| Authentication | Provider environment mapping; secrets omitted from generated config | Provider environment mapping; secrets omitted from generated patch |
| Canonical Anthropic route | `anthropic-direct/<model>` | `anthropic-direct/<model>` |
| Docker sandbox | Live native smoke | Live native smoke |
| Daytona, Modal, Apple Container, AgentCore | Static install/path contract only; live run availability-gated | Static install/path contract only; live run availability-gated |
| ACP trajectory | Message, reasoning, tool, user, and timeout events | Message, reasoning, tool, user, and timeout events |
| Credential-free Anthropic CI | Native `/v1/messages` lifecycle | Native `/v1/messages` lifecycle |
| Paid provider artifacts | Deferred until runtime credentials exist | Deferred until runtime credentials exist |

## Capability evidence

- Registry behavior is defined in `src/benchflow/agents/registry.py` and
  `src/benchflow/agents/providers.py`.
- Content-block and recovery behavior is defined in
  `src/benchflow/acp/client.py`.
- Credential-free native conformance is exercised by
  `tests/integration/native_acp_harness_smoke.py`.
- `tests/test_agent_feature_grid.py` prevents this document from drifting from
  the executable registry contract.

OpenScience v2.0.127 exposes only the `model` session configuration option.
BenchFlow therefore does not invent an `acp_effort_config_id`; model variants
remain the upstream mechanism. DeepSeek Harness v0.1.6-alpha.2 exposes the
separate `reasoning_effort` option.

OpenScience advertises ACP MCP, but v2.0.127's dynamically added remote
connector validates requests against persisted execution config and fails to
connect. BenchFlow therefore writes the equivalent isolated native `mcp`
configuration before launch. DSH's ACP HTTP path is used directly.

The explicit `anthropic-direct` provider exists so first-party Anthropic API
routing is available without changing the historical meaning of unregistered
`anthropic/...` model strings used by legacy agent/LiteLLM paths.
