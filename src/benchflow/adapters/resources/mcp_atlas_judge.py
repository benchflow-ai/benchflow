#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def collect_text(value):
    chunks = []
    if isinstance(value, str):
        chunks.append(value)
    elif isinstance(value, list):
        for item in value:
            chunks.extend(collect_text(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"text", "content", "message", "output"} or isinstance(item, (dict, list)):
                chunks.extend(collect_text(item))
    return chunks


def read_final_answer():
    path = Path("/logs/agent/acp_trajectory.jsonl")
    if not path.exists():
        return ""
    texts = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        texts.extend(collect_text(event))
    return "\n".join(texts[-20:])[-12000:]


def write_reward(reward, details):
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.txt").write_text(f"{reward}\n")
    (out / "reward.json").write_text(
        json.dumps({"reward": reward, "details": details}, indent=2)
    )


def judge(payload, answer):
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return 0.0, {"error": "OPENROUTER_API_KEY is not set"}
    model = os.environ.get("MCP_ATLAS_JUDGE_MODEL", "qwen/qwen-plus")
    claims = payload.get("claims") or []
    prompt = (
        "You are grading whether an agent answer satisfies factual claims for a tool-use task.\n"
        "Return only JSON with keys reward (0 or 1) and justification.\n\n"
        f"Task prompt:\n{payload.get('prompt', '')}\n\n"
        f"Required claims:\n{json.dumps(claims, indent=2)}\n\n"
        f"Agent answer/transcript excerpt:\n{answer}\n"
    )
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return 0.0, {
            "error": f"judge HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"
        }
    except Exception as e:
        return 0.0, {"error": f"judge failed: {e}"}
    content = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"reward": 0, "justification": content[:1000]}
    reward = 1.0 if float(parsed.get("reward", 0)) >= 0.5 else 0.0
    return reward, {"judge": parsed, "model": model}


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    answer = read_final_answer()
    if not answer.strip():
        write_reward(0.0, {"error": "empty agent answer/transcript"})
        return
    reward, details = judge(payload, answer)
    write_reward(reward, details)


if __name__ == "__main__":
    main()
