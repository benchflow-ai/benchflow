"""Convert the Tinker sampler checkpoint (PEFT LoRA r=32, Tinker module names) into a
PEFT adapter vLLM 0.30 loads for Qwen3.8-27B (Qwen3_5ForConditionalGeneration).

Exact, no approximation:
- Gated DeltaNet: Tinker trains separate LoRAs on in_proj_q, in_proj_k, in_proj_v. The
  checkpoint (and vLLM) has one fused in_proj_qkv (rows [q; k; v]); vLLM splits a packed
  in_proj_qkv LoRA's B by output slice and reuses its A for each slice. So
  A_qkv = [A_q; A_k; A_v] (96 x 5120) and B_qkv = blockdiag(B_q, B_k, B_v) (10240 x 96):
  slice q gets B_q A_q, k gets B_k A_k, v gets B_v A_v.
- Every other module is zero-padded from rank 32 to 96 (exact), so the adapter has one
  rank, r = lora_alpha = 96, and vLLM's scaling alpha/r stays 1.0 as in Tinker (32/32).
- unembed_tokens (Tinker's name for the output projection) becomes lm_head.
- Names follow the HF checkpoint (model.language_model.layers.N...), like prime-rl's adapter.
"""
import json, re, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
R = 96
w = load_file(str(src / "adapter_model.safetensors"))
cfg = json.loads((src / "adapter_config.json").read_text())
assert cfg["r"] == 32 and cfg["lora_alpha"] == 32 and not cfg.get("use_rslora") and not cfg.get("use_dora"), cfg
out = {}

def pad(a, b):
    r = a.shape[0]
    A = torch.zeros(R, a.shape[1], dtype=torch.float32); A[:r] = a
    B = torch.zeros(b.shape[0], R, dtype=torch.float32); B[:, :r] = b
    return A, B

def name(mod):
    return mod.replace("base_model.model.model.layers.", "model.language_model.layers.")

layers = sorted({int(m.group(1)) for k in w for m in [re.search(r"layers\.(\d+)\.linear_attn", k)] if m})
done = set()
for n in layers:
    p = f"base_model.model.model.layers.{n}.linear_attn."
    A = torch.cat([w[p + f"in_proj_{x}.lora_A.weight"] for x in "qkv"], 0)            # 96 x 5120
    Bs = [w[p + f"in_proj_{x}.lora_B.weight"] for x in "qkv"]
    B = torch.zeros(sum(b.shape[0] for b in Bs), R, dtype=torch.float32)
    row = 0
    for i, b in enumerate(Bs):
        B[row:row + b.shape[0], 32 * i:32 * (i + 1)] = b; row += b.shape[0]
    q = f"model.language_model.layers.{n}.linear_attn.in_proj_qkv"
    out[q + ".lora_A.weight"], out[q + ".lora_B.weight"] = A, B
    for x in "qkv":
        done |= {p + f"in_proj_{x}.lora_A.weight", p + f"in_proj_{x}.lora_B.weight"}
    # exactness check on this layer: per-slice products equal Tinker's
    off = 0
    for i, x in enumerate("qkv"):
        bq = Bs[i]; ref = bq @ w[p + f"in_proj_{x}.lora_A.weight"]
        got = B[off:off + bq.shape[0]] @ A; off += bq.shape[0]
        assert torch.allclose(ref, got, atol=0, rtol=0) or (ref - got).abs().max() < 1e-7, (n, x)
for k, v in w.items():
    if k in done or not k.endswith("lora_A.weight"):
        continue
    mod = k[: -len(".lora_A.weight")]
    A, B = pad(v, w[mod + ".lora_B.weight"])
    if mod == "base_model.model.model.unembed_tokens":
        nm = "lm_head"
    else:
        nm = name(mod)
        assert nm.startswith("model.language_model.layers."), mod
    out[nm + ".lora_A.weight"], out[nm + ".lora_B.weight"] = A, B
    done |= {k, mod + ".lora_B.weight"}
assert done == set(w), set(w) - done
out = {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}
dst.mkdir(parents=True, exist_ok=True)
save_file(out, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})
targets = sorted({k.split(".")[-3] for k in out})
new_cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": "Qwen/Qwen3.8-27B",
           "r": R, "lora_alpha": float(R), "lora_dropout": 0.0, "bias": "none", "use_rslora": False,
           "use_dora": False, "target_modules": targets, "modules_to_save": None,
           "source": "tinker://<run-id>:train:0/sampler_weights/final (r32, alpha32), converted by convert_tinker.py"}
(dst / "adapter_config.json").write_text(json.dumps(new_cfg, indent=2) + "\n")
print("tensors in", len(w), "out", len(out), "targets", targets)
print("lm_head delta norm vs nothing:", float((w["base_model.model.model.unembed_tokens.lora_B.weight"] @ w["base_model.model.model.unembed_tokens.lora_A.weight"]).norm()))
