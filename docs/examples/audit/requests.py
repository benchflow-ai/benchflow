"""Print what each model request of a job carried, from every trial's trajectory/llm_trajectory.jsonl.

One line per request: the trial, the call's number and purpose, the model asked for and the model that answered,
the effort, temperature, top_p and output cap sent ("-" when the request didn't set one), the HTTP status and why the
reply ended. Only calls that went through BenchFlow's model proxy are recorded.

    python3 requests.py jobs/<job>
"""
import json
import sys
from pathlib import Path

COLUMNS = ("trial", "call", "purpose", "model", "answered", "effort", "temperature", "top_p", "max_tokens", "status", "finish")
print("\t".join(COLUMNS))
for path in sorted(Path(sys.argv[1]).rglob("trajectory/llm_trajectory.jsonl")):
    trial = path.parent.parent.name
    for call, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        request = record["request"]["body"]
        response = record["response"]
        body = response.get("body") or {}
        choice = (body.get("choices") or [{}])[0]
        row = (
            trial,
            call,
            record.get("metadata", {}).get("call_purpose"),
            request.get("model"),
            body.get("model"),
            request.get("reasoning_effort"),
            request.get("temperature"),
            request.get("top_p"),
            request.get("max_tokens") or request.get("max_completion_tokens"),
            response.get("status_code"),
            choice.get("finish_reason") or body.get("stop_reason"),
        )
        print("\t".join("-" if value is None else str(value) for value in row))
