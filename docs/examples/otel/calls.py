"""Print each model call's tokens from Claude Code's OpenTelemetry and from BenchFlow's proxy record.

    python3 calls.py otlp.jsonl <trial folder>
"""
import json
import sys
from pathlib import Path


def value(v):
    return next(iter(v.values())) if v else None


otel = []
for line in open(sys.argv[1]):
    for logs in json.loads(line)["body"].get("resourceLogs", []):
        for scope in logs.get("scopeLogs", []):
            for record in scope.get("logRecords", []):
                a = {x["key"]: value(x["value"]) for x in record.get("attributes", [])}
                if a.get("event.name") == "api_request":
                    otel.append(a)

proxy = []
for line in (Path(sys.argv[2]) / "trajectory" / "llm_trajectory.jsonl").read_text().splitlines():
    if line.strip():
        usage = json.loads(line)["response"]["body"].get("usage") or {}
        proxy.append(usage)

print("call  otel: input  cache_read  output   proxy: prompt_tokens  completion_tokens")
for n in range(max(len(otel), len(proxy))):
    o = otel[n] if n < len(otel) else {}
    p = proxy[n] if n < len(proxy) else {}
    print(
        f"{n + 1:>4}  {o.get('input_tokens', '-'):>11}  {o.get('cache_read_tokens', '-'):>10}"
        f"  {o.get('output_tokens', '-'):>6}  {p.get('prompt_tokens', '-'):>21}"
        f"  {p.get('completion_tokens', '-'):>17}"
    )
