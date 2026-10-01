"""Logprob check of the converted Tinker adapter on vLLM against Tinker's own sampler.

Prompts: mid-episode histories from the q38 base eval on the expert test split (the
first 3-5 turns of 4 episodes), so the adapters see their training distribution. For
each prompt, the next reply (temperature 0, 24 tokens, logprobs) from vLLM base (vb),
vLLM tinker (vt), Tinker base (tb), Tinker tinker (tt), vLLM prime (vp). Reports, per
prompt, whether the greedy replies match, and the mean absolute difference of the
chosen tokens' logprobs over the common prefix: vb-tb is the two stacks' noise floor;
vt-tt should be near it if the conversion is right; vt-vb and tt-tb show the adapter.
  lora_logprob_check.py <vllm-base-url> <out.json> <rollouts.jsonl> [tinker-checkpoint]
(TINKER_API_KEY in the environment; rollouts.jsonl from an evaluate.py run on the
training family, whose histories give the prompts.)
"""
import json, os, sys
from pathlib import Path
import httpx
from benchflow.integrations.trl import bash_tool_schemas

VLLM, OUT = sys.argv[1], Path(sys.argv[2])
TINKER = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
CKPT = sys.argv[4] if len(sys.argv) > 4 else "tinker://<run-id>:train:0/sampler_weights/final"
key = os.environ["TINKER_API_KEY"]
src = Path(sys.argv[3])
recs = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
prompts = []
for r in recs[::37][:4]:
    msgs = r["messages"]
    # cut after the 3rd tool result: user, (assistant, tool)*3
    cut, n = [], 0
    for m in msgs:
        cut.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id")})
        if m["role"] == "tool":
            n += 1
            if n == 3:
                break
    prompts.append(cut)

def ask(url, model, extra, headers, messages):
    body = {"model": model, "messages": messages, "tools": bash_tool_schemas(), "tool_choice": "auto",
            "max_tokens": 24, "temperature": 0.0, "logprobs": True, "top_logprobs": 1, **extra}
    r = httpx.post(url + "/chat/completions", json=body, headers=headers, timeout=300)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    lp = [(t.get("token"), t.get("logprob")) for t in ((ch.get("logprobs") or {}).get("content") or [])]
    m = ch["message"]
    text = (m.get("content") or "") + "|" + "|".join(c["function"]["name"] + ":" + c["function"]["arguments"] for c in (m.get("tool_calls") or []))
    return {"text": text, "lp": lp}

def diff(a, b):
    n, s = 0, 0.0
    for (ta, la), (tb, lb) in zip(a["lp"], b["lp"]):
        if ta != tb or la is None or lb is None:
            break
        n += 1; s += abs(la - lb)
    return {"common_tokens": n, "mean_abs_lp_diff": round(s / n, 4) if n else None, "same_text": a["text"] == b["text"]}

off_v = {"chat_template_kwargs": {"enable_thinking": False}}
off_t = {"reasoning_effort": False}
hv = {"Authorization": "Bearer EMPTY"}; ht = {"Authorization": f"Bearer {key}"}
res = []
for p in prompts:
    r = {"vb": ask(VLLM, "Qwen/Qwen3.8-27B", off_v, hv, p), "vt": ask(VLLM, "tinker", off_v, hv, p),
         "vp": ask(VLLM, "prime", off_v, hv, p)}
    try:
        r["tb"] = ask(TINKER, "Qwen/Qwen3.8-27B", off_t, ht, p); r["tt"] = ask(TINKER, CKPT, off_t, ht, p)
    except Exception as exc:
        r["tinker_error"] = f"{type(exc).__name__}: {exc}"[:300]
    cmp = {"vt-vb (adapter on vLLM)": diff(r["vt"], r["vb"]), "vp-vb (prime on vLLM)": diff(r["vp"], r["vb"])}
    if "tb" in r:
        cmp.update({"vb-tb (stack floor)": diff(r["vb"], r["tb"]), "vt-tt (conversion)": diff(r["vt"], r["tt"]),
                    "tt-tb (adapter on Tinker)": diff(r["tt"], r["tb"]), "vt-tb": diff(r["vt"], r["tb"]), "vb-tt": diff(r["vb"], r["tt"])})
    r["cmp"] = cmp
    res.append(r)
    print(json.dumps(cmp), flush=True)
    print("  texts:", {k: r[k]["text"][:90] for k in ("vb", "vt", "vp", "tb", "tt") if k in r}, flush=True)
OUT.write_text(json.dumps(res, indent=1))
