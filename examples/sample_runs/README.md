# Sample runs — multi-agent hosted by BenchFlow (proxy-tracked + trajectories)

Every agent's raw LLM is routed through a loopback BenchFlow LiteLLM proxy
(`ensure_litellm_runtime`, `environment="local"`); the proxy aggregates per-call
usage/cost and persists raw exchanges to `<run>/trajectory/llm_trajectory.jsonl`
in BenchFlow's canonical format. The agent never sees the raw provider key
(proxy isolation invariant).

## Shared config
- model: `deepseek/deepseek-v4-pro` (provider-prefixed → deepseek upstream)
- proxy: `ensure_litellm_runtime(agent="deepagents", model=..., environment="local")`
- upstream key: `DEEPSEEK_API_KEY` (stripped from the agent env by the proxy)

## 1) Inter-agent arena — 3 concurrent seats  → `arena-floor-proxy/`
    set -a; . ./sb-run.env; set +a
    uv run python examples/arena/run_through_proxy.py 3
Each of 3 deepseek-v4 seats plays one shared high-card round concurrently via
`run_arena`. Outputs: `seat-*.trajectory.jsonl` (per-seat decisions) +
`trajectory/llm_trajectory.jsonl` (raw exchanges). Chips conserve at 3000.

## 2) Intra-agent LangGraph medical assistant  → `medical-assistant/`
    uv pip install langgraph langchain-openai
    set -a; . ./sb-run.env; set +a
    MEDICAL_CONFIDENCE_THRESHOLD=1.01 \
      uv run python examples/medical/run_through_proxy.py "side effects of metformin?"
Real `langgraph.StateGraph` (supervisor → KB/web specialists → confidence-gated
handoff → guardrail). Path: supervisor → retrieve_kb → answer → web_search →
answer → guardrail. Outputs: `trajectory/llm_trajectory.jsonl` (one exchange per
node LLM call).

## Note on capture fidelity
The per-seat / per-node *decision* trajectories (and each call's response usage)
are captured directly from each model response and are complete. The proxy's
canonical `trajectory/llm_trajectory.jsonl` is rebuilt from LiteLLM's async
callback log on stop and may capture N-1 of N calls when the final call's
callback hasn't flushed — a LiteLLM timing artifact, not a routing gap (every
call DID go through the proxy, as the aggregated usage/cost confirms).
