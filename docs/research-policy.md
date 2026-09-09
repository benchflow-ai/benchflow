# Filtered web research

Some research benchmarks need ordinary Internet access while keeping a small
set of papers or answer-bearing pages unavailable. Use a private, run-scoped
research policy:

```bash
bench eval run \
  --tasks-dir tasks/my-research-task \
  --agent codex-acp \
  --sandbox docker \
  --research-policy /secure/frontierphysics-policy.yaml
```

The policy is deliberately not part of the task package. Do not commit it next
to `task.md`, put its values in a prompt, or pass it through `--agent-env`.

## Policy format

```yaml
version: 1
tasks:
  my-research-task:
    blocked_urls:
      - https://example.org/papers/answer.html
    blocked_url_prefixes:
      - https://example.org/supplements/answer
    blocked_hosts:
      - private-corpus.example.org
    blocked_terms:
      - Exact Paper Title
      - 10.1234/example.doi
    blocked_content_sha256:
      - 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

Every selected task needs its own entry. A missing task or an entry that blocks
nothing stops the run. `blocked_urls` ignores query strings, fragments, and an
HTTP/HTTPS scheme change. `blocked_url_prefixes` also covers descendant paths.
`blocked_hosts` includes subdomains. Terms are case-insensitive and filter both
search-result titles and fetched text; content hashes cover exact binary or text
bodies.

The optional top-level `search_endpoint` selects an HTTP(S) search HTML
endpoint. It defaults to DuckDuckGo's Lite HTML endpoint.

## Enforcement model

On Docker, BenchFlow disables each harness's native web tools and provides the
same `benchflow-research` MCP server to ACP and native-MCP-config agents. It
offers `web_search`, `web_fetch`, and `web_download`. The server talks only to a
root-owned loopback gateway, which checks the destination, every redirect, and
the returned content.

The model provider uses a separate loopback proxy. Before the agent process is
launched, an owner-based IPv4/IPv6 firewall blocks every other connection from
the sandbox user. This prevents `curl`, sockets, or an unregistered harness tool
from bypassing the gateway. Private, loopback, link-local, and other non-global
fetch destinations are rejected to prevent SSRF.

Claude subscription authentication cannot be translated through LiteLLM because
there is no operator-owned API key. In that mode, the research gateway also
provides a fixed-destination loopback relay for the native Anthropic protocol.
The relay forwards only to `api.anthropic.com`; it is not a general HTTP proxy,
and rejects provider-side web-search/web-fetch tools plus remote MCP requests,
so the sandbox user remains unable to connect directly to research sites.

Policy-enabled runs currently require Docker, Python 3 in the task image, and a
non-root `sandbox_user`. Unsupported sandboxes and already-started external
sandboxes fail closed.

`config.json` records only the resolved policy SHA-256, rule counts, and whether
the gateway plus firewall became active. The private path and all rule values
are omitted from durable worker payloads and rollout artifacts.

This mechanism can guarantee that the sandbox cannot directly retrieve the
listed resources. It cannot guarantee that a model has never seen a paper in
pretraining or that an unlisted mirror/citation cannot reveal its existence.
Use terms, content hashes, and URLs for known mirrors when discovery leakage is
part of the benchmark threat model.
