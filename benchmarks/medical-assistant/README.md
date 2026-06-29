# Multi-Agent Medical Assistant — BenchFlow Adapter

BenchFlow adapter that **hosts the [Multi-Agent-Medical-Assistant](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant)**
agent pattern as a first-class BenchFlow benchmark: it runs in a Docker sandbox
with its LLM routed through BenchFlow's LiteLLM proxy, so per-rollout usage/cost
and the raw-LLM trajectory are captured, and the agent never sees the raw
provider key.

## What is hosted

The upstream is a LangGraph **supervisor → specialists** agent. This adapter
reproduces its control flow as a registered BenchFlow agent (`medical-assistant`,
an ACP shim at `src/benchflow/agents/medical_acp_shim.py`):

```
supervisor (router)
   ├── retrieve_kb   (RAG-style knowledge-base specialist)
   └── web_search    (fallback specialist)
        → answer     (emits a CONFIDENCE; low confidence → confidence-gated
                       handoff back to web_search)
        → guardrail  (output safety check)
```

Each graph node is streamed back as its own ACP `tool_call` step, so the
multi-agent structure (which specialist ran, in what order) is visible in the
captured trajectory.

### Scope vs. the full upstream app

The upstream app's heavy stack — **Azure OpenAI embeddings, Qdrant RAG, torch CV
imaging weights, Tavily web search** — is **out of scope** on the deepseek-only
proxy path. The RAG corpus is replaced by a small in-process knowledge base so
the full *router → specialists → confidence handoff → guardrail* control flow runs
end-to-end. The CV/imaging specialists and live web search are not included. See
`benchmark.yaml` (`hosting.not_included`).

## Tasks & verification

Three clinical drug-safety questions (`tasks/`), each verified **deterministically**:
the agent writes its answer to `/app/answer.md`, and `tests/test.sh` scores it
against the task's `ground_truth.json` keyword groups (OR within a group, AND
across groups). Reward = matched groups / total groups → `/logs/verifier/reward.txt`.

| Task | Question |
|------|----------|
| `metformin-side-effects` | Main side effects of metformin |
| `aspirin-secondary-prevention` | When low-dose aspirin is used, and its main risk |
| `ibuprofen-renal-caution` | Cautions for ibuprofen use |

## Run

```bash
set -a; . ~/sb-run.env; set +a        # provides DEEPSEEK_API_KEY for the proxy
python benchmarks/medical-assistant/run_medical_assistant.py

# or via the CLI:
bench eval run --config benchmarks/medical-assistant/medical-assistant-deepseek.yaml
```

Every rollout runs in its own Docker sandbox (`environment: docker`). The agent is
installed into the sandbox (uv venv + `langgraph` + `langchain-openai`) and its
`deepseek-v4-pro` calls go through the LiteLLM proxy.
