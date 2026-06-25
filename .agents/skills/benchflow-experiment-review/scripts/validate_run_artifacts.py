#!/usr/bin/env python3
"""Fail-closed BenchFlow run-artifact health validator.

This script is intentionally self-contained so it can travel with the
benchflow-experiment-review skill. It checks completed rollout folders for the
minimum artifact contract required before a trial can be treated as a healthy
BenchFlow model result:

- result.json parses.
- trajectory/acp_trajectory.jsonl exists, is non-empty, parses as JSONL, and
  contains agent-side events.
- trajectory/llm_trajectory.jsonl exists, is non-empty, parses as JSONL, and
  contains real provider request and response records.
- token usage, timing, and tool usage metadata are present.

Any violation makes that rollout unhealthy. The script exits 1 if any checked
rollout is unhealthy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TOKEN_KEYS = {
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "promptTokenCount",
    "candidatesTokenCount",
    "totalTokenCount",
}

INFRA_ERROR_MARKERS = (
    "missing required",
    "api key",
    "credential",
    "provider",
    "sandbox",
    "docker",
    "daytona",
    "no space left",
    "transport",
    "pipe closed",
    "connection",
)


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(errors="replace"))
    except OSError as exc:
        return None, f"{path}: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"{path}: JSON parse error at line {exc.lineno}: {exc.msg}"
    if not isinstance(value, dict):
        return None, f"{path}: expected JSON object"
    return value, None


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    try:
        raw = path.read_text(errors="replace")
    except OSError as exc:
        return [], [f"{path}: {exc}"]
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"{path}:{lineno}: JSON parse error: {exc.msg}")
            continue
        if not isinstance(row, dict):
            issues.append(f"{path}:{lineno}: expected JSON object")
            continue
        rows.append(row)
    if not rows and not issues:
        issues.append(f"{path}: empty JSONL")
    return rows, issues


def iter_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        out = [value]
        for child in value.values():
            out.extend(iter_dicts(child))
        return out
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for child in value:
            out.extend(iter_dicts(child))
        return out
    return []


def has_token_usage(value: Any) -> bool:
    for obj in iter_dicts(value):
        for key in TOKEN_KEYS:
            token_value = obj.get(key)
            if (
                isinstance(token_value, (int, float))
                and not isinstance(token_value, bool)
                and token_value > 0
            ):
                return True
    return False


def numeric_token_total(result: dict[str, Any]) -> int | None:
    sources: list[dict[str, Any]] = []
    agent_result = result.get("agent_result")
    if isinstance(agent_result, dict):
        sources.append(agent_result)
    token_usage = result.get("token_usage")
    if isinstance(token_usage, dict):
        sources.append(token_usage)
    sources.append(result)

    for source in sources:
        total = source.get("total_tokens") or source.get("n_total_tokens")
        if isinstance(total, int) and not isinstance(total, bool):
            return total
        input_tokens = (
            source.get("n_input_tokens")
            if source.get("n_input_tokens") is not None
            else source.get("input_tokens")
        )
        output_tokens = (
            source.get("n_output_tokens")
            if source.get("n_output_tokens") is not None
            else source.get("output_tokens")
        )
        if (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
        ):
            return input_tokens + output_tokens
        prompt_tokens = source.get("prompt_tokens")
        completion_tokens = source.get("completion_tokens")
        if (
            isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)
            and isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool)
        ):
            return prompt_tokens + completion_tokens
    return None


def reward_present(result: dict[str, Any]) -> bool:
    rewards = result.get("rewards")
    if isinstance(rewards, dict):
        return rewards.get("reward") is not None
    return result.get("reward") is not None


def timing_present(result: dict[str, Any]) -> bool:
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    return bool(
        result.get("started_at")
        and (result.get("finished_at") or timing.get("total") is not None)
    ) or bool(
        timing.get("started_at")
        and (timing.get("ended_at") or timing.get("duration_seconds") is not None)
    )


def tool_usage_count(result: dict[str, Any]) -> int | None:
    n_tool_calls = result.get("n_tool_calls")
    if isinstance(n_tool_calls, int) and not isinstance(n_tool_calls, bool):
        return n_tool_calls
    tool_usage = result.get("tool_usage")
    if isinstance(tool_usage, dict):
        total = 0
        found = False
        for value in tool_usage.values():
            if isinstance(value, int) and not isinstance(value, bool):
                total += value
                found = True
        if found:
            return total
    return None


def is_oracle_result(result: dict[str, Any], run_config: dict[str, Any] | None) -> bool:
    values = [result.get("agent"), result.get("agent_name")]
    if run_config:
        values.extend([run_config.get("agent"), run_config.get("harness")])
    return any(str(value).strip().lower() == "oracle" for value in values if value)


def validate_acp(rows: list[dict[str, Any]], path: Path) -> list[str]:
    issues: list[str] = []
    agent_events = 0
    for index, row in enumerate(rows, start=1):
        event_type = row.get("type")
        if not isinstance(event_type, str) or not event_type:
            issues.append(f"{path}:{index}: missing string 'type'")
        phase = str(row.get("phase") or "").lower()
        if phase != "verifier":
            agent_events += 1
    if agent_events == 0:
        issues.append(f"{path}: no agent-side events")
    return issues


def validate_llm(rows: list[dict[str, Any]], path: Path) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    request_count = 0
    response_count = 0
    error_count = 0
    usage_count = 0
    for index, row in enumerate(rows, start=1):
        request = row.get("request")
        response = row.get("response")
        error = row.get("error")
        if isinstance(request, dict):
            body = request.get("body")
            if isinstance(body, dict):
                request_count += 1
            else:
                issues.append(f"{path}:{index}: request missing object body")
        if isinstance(response, dict):
            body = response.get("body")
            if isinstance(body, dict):
                response_count += 1
                if has_token_usage(body):
                    usage_count += 1
            else:
                issues.append(f"{path}:{index}: response missing object body")
        if isinstance(error, dict):
            error_count += 1
    if request_count == 0:
        issues.append(f"{path}: no provider request bodies")
    if response_count == 0:
        issues.append(f"{path}: no provider response bodies")
    if request_count and response_count and response_count < request_count:
        issues.append(
            f"{path}: provider response count {response_count} < request count {request_count}"
        )
    if usage_count == 0:
        issues.append(f"{path}: no provider token usage in response bodies")
    return issues, {
        "requests": request_count,
        "responses": response_count,
        "errors": error_count,
        "responses_with_usage": usage_count,
    }


def load_run_config(root: Path) -> dict[str, Any] | None:
    for name in ("run_config.json", "config.json", "metadata.json"):
        path = root / name
        if not path.is_file():
            continue
        data, _ = read_json(path)
        if data is not None:
            return data
    return None


def validate_rollout(root: Path, *, allow_oracle_without_llm: bool = False) -> dict[str, Any]:
    result_path = root / "result.json"
    result, result_error = read_json(result_path)
    issues: list[str] = []
    warnings: list[str] = []
    if result_error:
        issues.append(result_error)
        result = {}

    run_config = load_run_config(root)
    oracle = is_oracle_result(result, run_config)

    acp_path = root / "trajectory" / "acp_trajectory.jsonl"
    llm_path = root / "trajectory" / "llm_trajectory.jsonl"
    artifact_summary: dict[str, Any] = {}

    if not acp_path.is_file():
        issues.append(f"missing required artifact: {acp_path}")
    else:
        acp_rows, acp_issues = read_jsonl(acp_path)
        issues.extend(acp_issues)
        issues.extend(validate_acp(acp_rows, acp_path))
        artifact_summary["acp_events"] = len(acp_rows)

    if oracle and allow_oracle_without_llm:
        warnings.append("oracle rollout: llm_trajectory requirement bypassed by flag")
    elif not llm_path.is_file():
        issues.append(f"missing required artifact: {llm_path}")
    else:
        llm_rows, llm_issues = read_jsonl(llm_path)
        issues.extend(llm_issues)
        llm_health_issues, llm_summary = validate_llm(llm_rows, llm_path)
        issues.extend(llm_health_issues)
        artifact_summary["llm_exchanges"] = len(llm_rows)
        artifact_summary["llm"] = llm_summary

    tokens = numeric_token_total(result)
    if not tokens or tokens <= 0:
        issues.append("missing or zero token usage in result metadata")
    if not timing_present(result):
        issues.append("missing timing metadata")
    tools = tool_usage_count(result)
    if tools is None or tools <= 0:
        issues.append("missing or zero tool usage metadata")
    if not reward_present(result):
        issues.append("missing verifier reward/score")

    error_text = " ".join(
        str(result.get(key) or "")
        for key in ("error", "verifier_error", "error_category", "verifier_error_category")
    ).lower()
    if any(marker in error_text for marker in INFRA_ERROR_MARKERS):
        issues.append("result carries infra/provider error markers")

    return {
        "root": str(root),
        "status": "healthy" if not issues else "unhealthy",
        "healthy": not issues,
        "issues": issues,
        "warnings": warnings,
        "result": {
            "task_name": result.get("task_name"),
            "agent": result.get("agent"),
            "model": result.get("model"),
            "tokens": tokens,
            "tool_calls": tools,
            "reward_present": reward_present(result),
            "oracle": oracle,
        },
        "artifacts": artifact_summary,
    }


def discover_rollouts(path: Path) -> list[Path]:
    if path.is_file() and path.name == "result.json":
        return [path.parent]
    if (path / "result.json").is_file():
        return [path]
    return sorted({candidate.parent for candidate in path.rglob("result.json")})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Rollout dir, result.json, or jobs root to validate.",
    )
    parser.add_argument(
        "--allow-oracle-without-llm",
        action="store_true",
        help="Treat oracle reward-only runs as out of scope for LLM trajectory capture.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a concise text report.",
    )
    args = parser.parse_args(argv)

    rollouts: list[Path] = []
    for path in args.paths:
        rollouts.extend(discover_rollouts(path))
    deduped = sorted({path.resolve(): path for path in rollouts}.values())

    reports = [
        validate_rollout(
            rollout,
            allow_oracle_without_llm=args.allow_oracle_without_llm,
        )
        for rollout in deduped
    ]
    summary = {
        "checked": len(reports),
        "healthy": sum(1 for report in reports if report["healthy"]),
        "unhealthy": sum(1 for report in reports if not report["healthy"]),
    }
    payload = {
        "healthy": summary["unhealthy"] == 0 and summary["checked"] > 0,
        "summary": summary,
        "rollouts": reports,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            f"checked={summary['checked']} healthy={summary['healthy']} "
            f"unhealthy={summary['unhealthy']}"
        )
        for report in reports:
            marker = "OK" if report["healthy"] else "UNHEALTHY"
            print(f"{marker} {report['root']}")
            for issue in report["issues"]:
                print(f"  - {issue}")
            for warning in report["warnings"]:
                print(f"  warning: {warning}")

    return 0 if payload["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
