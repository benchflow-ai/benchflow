# No-web subscription proxy

The checks behind [subscription logins on no-web tasks](../agents/no-web-subscription.md). This page is for maintainers and security reviewers.

## Admission

A direct `claude-agent-acp` subscription run can use the controller-owned TLS proxy for model requests. The sandbox user still cannot connect directly anywhere outside loopback. This includes automatic reviewers with `open_network=False`.

- The managed launcher `/opt/benchflow/bin/claude-agent-acp` hands the adapter the pinned CLI through `CLAUDE_CODE_EXECUTABLE`. The adapter prefers it to the CLI its SDK bundles.
- Setup reads the ACP, SDK and CLI versions from the installed packages and runs the CLI's `--version`.
- It compares the launcher with the one BenchFlow writes.
- It checks all of this together with the actual nonzero sandbox UID and any routing overrides.

Bumping either registry pin (`_CLAUDE_AGENT_ACP_PACKAGE`, `_CLAUDE_CODE_PACKAGE`) retargets the check. Re-run the streaming fixture below when bumping.

Unsupported native clients or conflicting custom routing fail before ACP starts. They never fall back to open networking or API-key authentication.

## Request rules

The proxy terminates TLS, so it reads each request body before any byte goes upstream.

| Rule | Detail |
|---|---|
| route | HTTPS `api.anthropic.com:443`, `POST`, path `/v1/messages` with no query or exactly `?beta=true` |
| tools | every `tools` entry must be a client-executed custom tool: no `type`, or `type: "custom"` |
| MCP | no `mcp_servers` |
| sources | no image or document `source` in `system` or `messages` other than `base64`, `text`, `content` or `file` |
| body | a UTF-8 JSON object with no repeated key, no `Content-Encoding`, at most 32 MiB |

So `web_search_*`, `web_fetch_*`, `code_execution_*`, `advisor_*`, `mcp_toolset`, Anthropic-schema client tools, MCP connector servers and URL sources are all refused.

Accepted bodies are forwarded unchanged with a `Content-Length`; chunked uploads are re-framed. Model responses still stream. Refusals are logged by rule name only, never by body content.

## Headers and routing

OAuth headers are the ones the native client sends. BenchFlow sets no ACP provider override and never converts a token into an API key. API-key and LiteLLM routes, and ordinary task denylists, are not affected by this proxy.

## The admission receipt

`native-oauth-network.json` records successful proxy setup, client versions and the sandbox UID. It is written after the firewall is installed and before ACP bootstrap.

- A new connection attempt clears the current receipt.
- Reconnects rebuild the transport.
- Role changes archive earlier proxy diagnostics under `network-transports/`.

The receipt proves transport setup. It does not prove a successful model call, or that every hosted capability is absent.

## Evidence

A fake-token, offline, native-client streaming fixture exercised this narrow set. It last ran against Claude Code 2.1.257 and has not been re-run on 2.1.280.

In that release Claude Code sends a server tool in two cases only: for its WebSearch tool, which no-web runs disable, and when an `advisorModel` setting is configured. The fixture does not establish compatibility with every future Claude feature or release.

## Next steps

- [Subscription logins on no-web tasks](../agents/no-web-subscription.md): the user guide.
- [Native harness internals](../contributing/native-harness-internals.md#model-routing-checks): how native runs are routed.
