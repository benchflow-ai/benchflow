# Native Claude in no-web tasks

Direct `claude-agent-acp` subscription runs can use the controller-owned TLS proxy for model requests while the sandbox user remains unable to connect directly outside loopback. This includes automatic reviewers with `open_network=False`.

The client must be the registry's `claude-agent-acp` pin (`_CLAUDE_AGENT_ACP_PACKAGE` in `agents/registry.py`, currently 0.73.0), the exact `@anthropic-ai/claude-agent-sdk` version that ACP release depends on (0.3.257), and the native Claude release that SDK declares as its `claudeCodeVersion` (2.1.257), on Linux with the managed executable layout. Setup reads these from the installed packages and checks them together with the actual nonzero sandbox UID, managed launcher and routing overrides. Bumping the registry pin retargets the check; re-run the streaming fixture below when bumping. Unsupported native clients or conflicting custom routing fail before ACP starts; they do not fall back to open networking or API-key authentication.

The proxy permits only HTTPS `api.anthropic.com:443`, method POST, path `/v1/messages` with no query or exactly `?beta=true`. All other origins, ports, methods and paths are refused. Optional client settings/hello requests are blocked.

The proxy terminates TLS, so it also reads each request body before any byte goes upstream. Anthropic runs server tools and fetches remote sources itself, which would give the sandbox web access (and an exfiltration path) through the model endpoint. A request is refused unless every `tools` entry is a client-executed custom tool (no `type`, or `type: "custom"`); it must also have no `mcp_servers`, and no image or document `source` anywhere in `system` or `messages` other than `base64`, `text`, `content` or `file`. So `web_search_*`, `web_fetch_*`, `code_execution_*`, `advisor_*`, `mcp_toolset`, Anthropic-schema client tools, MCP connector servers and URL sources are all refused. Bodies that are not UTF-8 JSON objects, repeat an object key, declare a `Content-Encoding`, or exceed 32 MiB are refused as well. Accepted bodies are forwarded unchanged with a `Content-Length` (chunked uploads are re-framed), and model responses still stream. Refusals are logged by rule name only, never by body content. Claude Code 2.1.257 sends a server tool only for its WebSearch tool, which no-web runs disable, or when an `advisorModel` setting is configured. An `advisorModel` setting therefore does not work on no-web runs: every request that carries its `advisor_*` server tool gets `403 Forbidden` (rule `native-model-body-server-tool` in `trajectory/egress_denylist.jsonl`), so leave the setting out of no-web runs. This narrow set was exercised by a fake-token offline native-client streaming fixture; it does not establish compatibility with every future Claude feature or native release.

OAuth headers remain those emitted by the native client. No ACP provider override or token-to-API-key conversion occurs. API-key/LiteLLM routes and ordinary task denylists retain their existing behavior.

`native-oauth-network.json` records successful proxy setup, client versions and sandbox UID after the firewall is installed and before ACP bootstrap. A new connection attempt clears this current admission receipt. Reconnects rebuild the transport; role changes archive prior proxy diagnostics under `network-transports/`. An admission receipt proves transport setup, not a successful model call or absence of every hosted capability.

## Task image and sandbox requirements

The proxy and firewall run inside the task's `main` container, so a no-web native OAuth run needs the following. None of them falls back to open networking or API-key authentication when missing.

- `python3` on `PATH`. The `claude-agent-acp` no-web setup writes the agent's `settings.json` with it (on every no-web Claude run, OAuth or API key), and the proxy is a Python script. Without it, setup stops before ACP starts with `... needs python3 in the task image`.
- `openssl` on `PATH`. The proxy signs its TLS certificates with it. Without it, proxy setup stops before ACP starts with `... needs openssl in the task image`.
- A system CA bundle, such as Debian/Ubuntu `ca-certificates`. The proxy verifies `api.anthropic.com` against the image's default trust store. This is not checked at setup: without a bundle, setup succeeds but every model request fails upstream certificate verification.
- A non-root `sandbox_user` with a nonzero UID.
- `iptables`, and `ip6tables` when the kernel exposes IPv6, for the sandbox-user firewall. When `iptables` is missing, setup installs the `iptables` package with apt, dnf or apk.
- The `NET_ADMIN` capability on `main`, which `iptables` needs. The Docker backend and the Daytona DinD strategy (tasks with a `docker-compose.yaml`) add the `docker-compose-net-admin.yaml` overlay automatically whenever they keep a no-network task's container open for an LLM agent, as they do for `network_mode: denylist`. Both backends decide this with the same rule (`compose_needs_net_admin` in `sandbox/_compose.py`), and automatic reviewers get the capability through it too. The task does not need its own `cap_add`. Daytona direct sandboxes run the firewall without it.

Stock `ubuntu:24.04` has none of the first three. A task image can add them with:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends python3 openssl ca-certificates && rm -rf /var/lib/apt/lists/*
```
