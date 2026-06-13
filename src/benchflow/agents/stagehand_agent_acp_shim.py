#!/usr/bin/env python3
"""ACP shim for the Stagehand Agent browser loop."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from benchflow.environment.browser_runtime import (
    browser_runtime_session,
    expected_from_prompt,
)

_DIAG_TRUNCATE = 2000
_TOOL_RESULT_TRUNCATE = 4000
_DEFAULT_MODEL = "google/gemini-2.5-flash"
_DEFAULT_NODE = "/opt/benchflow/node/bin/node"
_DEFAULT_NODE_PATH = "/opt/benchflow/stagehand-agent/node_modules"
_DEFAULT_BROWSERS_PATH = "/opt/benchflow/stagehand-ms-playwright"

_STAGEHAND_RUNNER_JS = r"""
import process from "node:process";
import { chromium } from "playwright";
import { Stagehand, AISdkClient } from "@browserbasehq/stagehand";
import { createGoogleGenerativeAI } from "@ai-sdk/google";

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

function normalizeModel(model) {
  if (!model) return "google/gemini-2.5-flash";
  if (model.startsWith("google/")) return model;
  if (model.startsWith("gemini/")) return `google/${model.slice("gemini/".length)}`;
  if (model.startsWith("openai/") || model.startsWith("anthropic/")) return model;
  return `google/${model}`;
}

function googleModelName(model) {
  return model.startsWith("google/") ? model.slice("google/".length) : model;
}

const input = JSON.parse(await readStdin());
const expected = input.expected || null;
const model = normalizeModel(input.model);
const apiKey =
  process.env.GEMINI_API_KEY ||
  process.env.GOOGLE_GENERATIVE_AI_API_KEY ||
  process.env.GOOGLE_API_KEY;
if (!apiKey && model.startsWith("google/")) {
  throw new Error("GEMINI_API_KEY, GOOGLE_GENERATIVE_AI_API_KEY, or GOOGLE_API_KEY is required");
}

process.env.PLAYWRIGHT_BROWSERS_PATH =
  process.env.PLAYWRIGHT_BROWSERS_PATH || "/opt/benchflow/stagehand-ms-playwright";
process.env.CHROME_PATH = process.env.CHROME_PATH || chromium.executablePath();

const google = createGoogleGenerativeAI({ apiKey });
const llmClient = new AISdkClient({ model: google(googleModelName(model)) });
const stagehand = new Stagehand({
  env: "LOCAL",
  llmClient,
  disablePino: true,
  verbose: 0,
  localBrowserLaunchOptions: {
    headless: true,
    chromiumSandbox: false,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  },
});

try {
  await stagehand.init();
  const page = stagehand.context.pages()[0];
if (input.url) {
  await page.goto(input.url);
}
let extracted = null;
if (input.url) {
  extracted = await stagehand.extract(
    "Extract the exact page status text from the main element."
  );
}
const agent = stagehand.agent({ mode: input.mode || "dom", model });
const result = await agent.execute({
  instruction:
    input.instruction ||
    (expected
      ? `Report the page status. Final answer must be exactly: ${expected}`
      : "Complete the browser task and return the final answer."),
    maxSteps: input.max_steps || 6,
  });
  let screenshot = "";
  try {
    screenshot = (await page.screenshot({ type: "png" })).toString("base64");
  } catch {
    screenshot = "";
  }
  const currentUrl = page.url();
  const actions = Array.isArray(result?.actions)
    ? result.actions.map((action) => ({
        type: action?.type || "unknown",
        taskCompleted:
          action?.taskCompleted ?? action?.taskComplete ?? action?.completed ?? null,
      }))
    : [];
  process.stdout.write(
    JSON.stringify({
      model,
      extracted,
      result: {
        success: result?.success ?? null,
        completed: result?.completed ?? null,
        message: result?.message || "",
        actions,
        usage: result?.usage || {},
        messages_count: Array.isArray(result?.messages) ? result.messages.length : 0,
      },
      current_url: currentUrl,
      screenshots_b64: screenshot ? [screenshot] : [],
    })
  );
} finally {
  await stagehand.close({ force: true }).catch(() => {});
}
"""


def send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def recv() -> dict[str, Any]:
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed")
        line = line.strip()
        if line:
            return json.loads(line)


def _prompt_text(params: dict[str, Any]) -> str:
    parts = params.get("prompt", [])
    if not isinstance(parts, list):
        return ""
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(str(part.get("text", "")))
    return "\n".join(text_parts)


def _emit_text(session_id: str, text: str) -> None:
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text[:_DIAG_TRUNCATE]},
                },
            },
        }
    )


def _emit_tool_call(
    session_id: str,
    tool_call_id: str,
    name: str,
    arguments: str,
) -> None:
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": tool_call_id,
                    "title": name,
                    "kind": "browser",
                    "status": "in_progress",
                    "input": arguments[:_DIAG_TRUNCATE],
                },
            },
        }
    )


def _emit_tool_result(
    session_id: str,
    tool_call_id: str,
    *,
    result: str,
    status: str = "completed",
) -> None:
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": tool_call_id,
                    "status": status,
                    "content": [
                        {
                            "type": "content",
                            "content": {
                                "type": "text",
                                "text": result[:_TOOL_RESULT_TRUNCATE],
                            },
                        }
                    ],
                },
            },
        }
    )


def _artifact_dir() -> Path:
    return Path(os.environ.get("BENCHFLOW_STAGEHAND_ARTIFACT_DIR", "/logs/artifacts"))


def _normalize_model(model: str) -> str:
    if not model:
        return _DEFAULT_MODEL
    if model.startswith("google/"):
        return model
    if model.startswith("gemini/"):
        return f"google/{model.split('/', 1)[1]}"
    if model.startswith(("openai/", "anthropic/")):
        return model
    return f"google/{model}"


def _max_steps_from_prompt(text: str) -> int | None:
    match = re.search(
        r"(?:stagehand\s+)?max(?:imum)?\s+browser\s+steps:\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _run_stagehand_node(payload: dict[str, Any]) -> dict[str, Any]:
    node = os.environ.get("STAGEHAND_AGENT_NODE", _DEFAULT_NODE)
    runner_cwd = Path(
        os.environ.get("STAGEHAND_AGENT_CWD", "/opt/benchflow/stagehand-agent")
    )
    env = {
        **os.environ,
        "NODE_PATH": os.environ.get("STAGEHAND_AGENT_NODE_PATH", _DEFAULT_NODE_PATH),
        "PLAYWRIGHT_BROWSERS_PATH": os.environ.get(
            "PLAYWRIGHT_BROWSERS_PATH", _DEFAULT_BROWSERS_PATH
        ),
    }
    kwargs: dict[str, Any] = {}
    if runner_cwd.is_dir():
        kwargs["cwd"] = str(runner_cwd)
    result = subprocess.run(
        [node, "--input-type=module", "-e", _STAGEHAND_RUNNER_JS],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        timeout=int(os.environ.get("STAGEHAND_AGENT_TIMEOUT", "180")),
        check=False,
        **kwargs,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Stagehand runner failed with rc {result.returncode}: {message}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Stagehand runner returned non-JSON output: {result.stdout!r}"
        ) from exc


def _run_stagehand_agent(
    *,
    cwd: Path,
    prompt: str,
    session_id: str,
    model: str,
) -> dict[str, Any]:
    expected = expected_from_prompt(prompt)
    started = time.perf_counter()
    artifact_dir = _artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with browser_runtime_session(cwd) as browser_session:
        browser_env = browser_session.handle
        if browser_env.url is not None and not browser_session.readiness.ok:
            raise RuntimeError(
                f"browser environment not ready: {browser_session.readiness.to_dict()}"
            )
        instruction = browser_session.task_instruction(
            prompt=prompt,
            expected=expected,
        )
        tool_call_id = f"stagehand-agent-{uuid.uuid4().hex[:8]}"
        normalized_model = _normalize_model(model)
        max_steps = _max_steps_from_prompt(instruction) or int(
            os.environ.get("STAGEHAND_AGENT_MAX_STEPS", "6")
        )
        _emit_tool_call(
            session_id,
            tool_call_id,
            "stagehand.agent.execute",
            json.dumps(
                {
                    "model": normalized_model,
                    "mode": "dom",
                    "environment": browser_session.environment,
                    "max_steps": max_steps,
                },
                sort_keys=True,
            ),
        )
        try:
            output = _run_stagehand_node(
                {
                    "url": browser_env.url,
                    "instruction": instruction,
                    "expected": expected,
                    "model": normalized_model,
                    "mode": "dom",
                    "max_steps": max_steps,
                }
            )
        except Exception as exc:
            _emit_tool_result(
                session_id,
                tool_call_id,
                result=f"{type(exc).__name__}: {exc}",
                status="failed",
            )
            raise

    result = output.get("result") if isinstance(output, dict) else {}
    if not isinstance(result, dict):
        result = {}
    extracted = output.get("extracted") if isinstance(output, dict) else {}
    screenshots = output.get("screenshots_b64") if isinstance(output, dict) else []
    if not isinstance(screenshots, list):
        screenshots = []
    actions = result.get("actions")
    if not isinstance(actions, list):
        actions = []
    serialized = json.dumps(output, sort_keys=True)
    if expected is not None and expected not in serialized:
        raise RuntimeError(
            "Stagehand Agent output did not contain expected value: "
            f"{result.get('message', '')!r}"
        )
    if result.get("success") is False:
        raise RuntimeError(
            f"Stagehand Agent reported failure: {result.get('message', '')!r}"
        )

    final_result = expected or str(
        result.get("message") or result.get("finalResult") or serialized
    )
    (cwd / "final_result.txt").write_text(final_result + "\n")
    artifact = browser_session.write_trace_artifact(
        artifact_dir / "browser-use-smoke-trace.json",
        framework="benchflow-stagehand-agent",
        steps=actions,
        screenshots_b64=[screenshot for screenshot in screenshots if screenshot],
        final_result=final_result,
        duration_sec=round(time.perf_counter() - started, 6),
        extra={
            "stagehand_message": str(result.get("message", "")),
            "stagehand_success": result.get("success"),
            "stagehand_completed": result.get("completed"),
            "stagehand_model": output.get("model"),
            "stagehand_current_url": output.get("current_url"),
            "stagehand_extracted": extracted,
            "stagehand_usage": result.get("usage") or {},
            "stagehand_messages_count": result.get("messages_count"),
        },
    )
    _emit_tool_result(
        session_id,
        tool_call_id,
        result=json.dumps(
            {
                "final_result": final_result,
                "actions": [
                    action.get("type") for action in actions if isinstance(action, dict)
                ],
                "screenshots": len(artifact["screenshots_b64"]),
                "success": result.get("success"),
            },
            sort_keys=True,
        ),
    )
    return artifact


def main() -> int:
    session_id = "stagehand-agent"
    cwd = Path("/app")
    model = _DEFAULT_MODEL

    while True:
        try:
            request = recv()
        except EOFError:
            return 0
        except Exception as exc:
            print(f"stagehand-agent ACP decode error: {exc}", file=sys.stderr)
            return 1

        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": 1,
                        "agentCapabilities": {
                            "loadSession": False,
                            "promptCapabilities": {"image": False, "audio": False},
                        },
                        "agentInfo": {"name": "stagehand-agent", "version": "0.1"},
                    },
                }
            )
        elif method == "session/new":
            session_id = f"stagehand-agent-{uuid.uuid4().hex[:8]}"
            cwd = Path(str(params.get("cwd", "/app")))
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"sessionId": session_id},
                }
            )
        elif method == "session/set_model":
            model = str(params.get("modelId", "")) or _DEFAULT_MODEL
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/prompt":
            prompt = _prompt_text(params)
            try:
                artifact = _run_stagehand_agent(
                    cwd=cwd,
                    prompt=prompt,
                    session_id=session_id,
                    model=model,
                )
            except Exception as exc:
                _emit_text(session_id, f"Stagehand Agent failed: {exc}")
            else:
                _emit_text(
                    session_id,
                    "Stagehand Agent final result: "
                    f"{artifact['final_result']} ({len(artifact['steps'])} actions)",
                )
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"stopReason": "end_turn"},
                }
            )
        elif method == "session/cancel":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
