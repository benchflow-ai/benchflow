"""Trajectory viewer — renders Claude Code stream-json as HTML.

Works directly with raw turn*.txt files. No ATIF conversion.
"""

import html
import json
import sys
from pathlib import Path


_THINKING_PREVIEW = 600  # max chars for thinking block preview
_ARGS_PREVIEW = 300  # max chars for tool args display
_CONTENT_PREVIEW = 200  # max chars for write/agent content preview
_RESULT_PREVIEW = 300  # max chars for result summary


def render_turn(events: list[dict], turn_number: int, prompt: str = "") -> str:
    """Render one turn's events as HTML blocks."""
    blocks = []

    # Prompt
    if prompt:
        blocks.append(
            f'<div class="step prompt">'
            f'<div class="step-header"><span class="label prompt">PROMPT (turn {turn_number})</span></div>'
            f'<div class="msg">{html.escape(prompt)}</div>'
            f"</div>"
        )

    # Group: thinking → text → tool_use → tool_result → thinking → ...
    pending_thinking = ""
    pending_text = ""

    for event in events:
        etype = event.get("type", "")

        if etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                btype = block.get("type", "")

                if btype == "thinking":
                    pending_thinking += block.get("thinking", "")

                elif btype == "text":
                    pending_text += block.get("text", "")

                elif btype == "tool_use":
                    # Emit accumulated thinking+text, then the tool call
                    parts = []
                    if pending_thinking:
                        parts.append(
                            f'<div class="thinking">{html.escape(pending_thinking[:_THINKING_PREVIEW])}'
                            f"{'...' if len(pending_thinking) > _THINKING_PREVIEW else ''}</div>"
                        )
                        pending_thinking = ""
                    if pending_text:
                        parts.append(
                            f'<div class="msg">{html.escape(pending_text)}</div>'
                        )
                        pending_text = ""

                    name = html.escape(block.get("name", ""))
                    args = block.get("input", {})
                    # Format args nicely
                    if name == "Bash":
                        arg_display = html.escape(args.get("command", ""))
                    elif name in ("Read", "Write", "Edit"):
                        arg_display = html.escape(
                            args.get("file_path", args.get("path", ""))
                        )
                        if name == "Write" and "content" in args:
                            content_preview = args["content"][:_CONTENT_PREVIEW]
                            arg_display += f"\n{html.escape(content_preview)}{'...' if len(args['content']) > _CONTENT_PREVIEW else ''}"
                    elif name == "Agent":
                        arg_display = html.escape(
                            str(args.get("prompt", ""))[:_CONTENT_PREVIEW]
                        )
                    else:
                        arg_display = html.escape(
                            json.dumps(args, indent=2)[:_ARGS_PREVIEW]
                        )

                    parts.append(
                        f'<div class="tool">'
                        f'<span class="tool-name">{name}</span>'
                        f'<pre class="tool-args">{arg_display}</pre>'
                        f"</div>"
                    )

                    blocks.append(f'<div class="step agent">{"".join(parts)}</div>')

        elif etype == "user":
            content = event.get("message", {}).get("content", "")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        raw = str(block.get("content", ""))[:500]
                        # Detect binary
                        printable = sum(
                            1 for c in raw if c.isprintable() or c in "\n\t"
                        )
                        if len(raw) > 20 and printable / len(raw) < 0.7:
                            display = "[binary content]"
                        else:
                            display = html.escape(raw[:400])
                        blocks.append(
                            f'<div class="step output"><pre>{display}</pre></div>'
                        )

        elif etype == "result":
            # Final summary
            cost = event.get("total_cost_usd", 0)
            turns = event.get("num_turns", "?")
            result_text = html.escape(event.get("result", "")[:_RESULT_PREVIEW])
            blocks.append(
                f'<div class="step result">'
                f'<div class="step-header"><span class="label result">RESULT</span>'
                f'<span class="meta-inline">turns={turns} cost=${cost:.4f}</span></div>'
                f'<div class="msg">{result_text}</div>'
                f"</div>"
            )

    # Flush remaining text
    if pending_thinking or pending_text:
        parts = []
        if pending_thinking:
            parts.append(
                f'<div class="thinking">{html.escape(pending_thinking[:_THINKING_PREVIEW])}</div>'
            )
        if pending_text:
            parts.append(f'<div class="msg">{html.escape(pending_text)}</div>')
        blocks.append(f'<div class="step agent">{"".join(parts)}</div>')

    return "\n".join(blocks)


def render_trial(trial_dir: Path, prompts: list[str] | None = None) -> str:
    """Render a full trial (multiple turns) as HTML.

    Auto-detects format:
    - turn*.txt → Claude Code stream-json
    - trajectory/acp_trajectory.jsonl → ACP session events
    - prompts.json → used for prompt labels if available
    """
    # Try loading prompts from prompts.json if not provided
    if prompts is None and (trial_dir / "prompts.json").exists():
        prompts = json.loads((trial_dir / "prompts.json").read_text())

    # Auto-detect format
    turn_files = sorted(trial_dir.glob("turn*.txt"))
    acp_traj = trial_dir / "trajectory" / "acp_trajectory.jsonl"

    if not turn_files and acp_traj.exists():
        return _render_acp_trajectory(trial_dir, acp_traj, prompts)

    if not turn_files:
        return "<p>No trajectory files found</p>"

    # Default prompts
    if prompts is None:
        prompts = [
            f"(turn {i + 1} prompt — not captured in stream)"
            for i in range(len(turn_files))
        ]

    # Pad prompts if fewer than turns
    while len(prompts) < len(turn_files):
        prompts.append("")

    # Extract session info from first turn
    def _parse_jsonl(text: str) -> list[dict]:
        events = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    first_events = _parse_jsonl(turn_files[0].read_text())
    sys_event = next((e for e in first_events if e.get("type") == "system"), {})
    total_cost = 0
    total_turns_count = 0

    all_blocks = []
    for i, tf in enumerate(turn_files):
        events = _parse_jsonl(tf.read_text())
        all_blocks.append(render_turn(events, i + 1, prompts[i]))
        for e in events:
            if e.get("type") == "result":
                total_cost += e.get("total_cost_usd", 0)
                total_turns_count += e.get("num_turns", 0)

    session_id = sys_event.get("session_id", "?")
    model = sys_event.get("model", "?")
    version = sys_event.get("claude_code_version", "?")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>benchflow — {trial_dir.name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; max-width: 960px; margin: 0 auto; }}
.header {{ border-bottom: 1px solid #30363d; padding-bottom: 16px; margin-bottom: 24px; }}
.header h1 {{ font-size: 20px; color: #f0f6fc; margin-bottom: 8px; }}
.meta {{ display: flex; gap: 12px; flex-wrap: wrap; font-size: 13px; color: #8b949e; }}
.meta span {{ background: #161b22; padding: 4px 10px; border-radius: 6px; border: 1px solid #30363d; }}
.step {{ margin-bottom: 4px; padding: 10px 14px; border-radius: 6px; }}
.step.prompt {{ background: #0d1f3c; border: 1px solid #1f3a5f; margin-bottom: 12px; }}
.step.agent {{ background: #161b22; border: 1px solid #30363d; }}
.step.output {{ background: #0d1117; border-left: 3px solid #238636; padding: 6px 14px; }}
.step.output pre {{ color: #7ee787; font-size: 12px; white-space: pre-wrap; word-break: break-word; }}
.step.result {{ background: #1a2f1a; border: 1px solid #238636; margin-top: 12px; }}
.step-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
.label {{ padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
.label.prompt {{ background: #1f3a5f; color: #58a6ff; }}
.label.result {{ background: #1a2f1a; color: #3fb950; }}
.meta-inline {{ font-size: 12px; color: #8b949e; }}
.msg {{ font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
.thinking {{ font-size: 13px; color: #8b949e; font-style: italic; margin-bottom: 6px; padding: 8px; background: #0d1117; border-radius: 4px; border-left: 3px solid #484f58; }}
.tool {{ margin-bottom: 4px; }}
.tool-name {{ background: #2d333b; color: #f0883e; padding: 2px 8px; border-radius: 4px; font-family: monospace; font-size: 13px; font-weight: 600; }}
.tool-args {{ margin-top: 4px; font-size: 12px; color: #c9d1d9; background: #0d1117; padding: 8px; border-radius: 4px; white-space: pre-wrap; word-break: break-word; }}
.turn-divider {{ border-top: 2px solid #30363d; margin: 20px 0; padding-top: 8px; }}
</style>
</head>
<body>
<div class="header">
<h1>{html.escape(trial_dir.name)}</h1>
<div class="meta">
<span>model: {html.escape(model)}</span>
<span>session: {html.escape(session_id[:16])}...</span>
<span>claude code: {html.escape(version)}</span>
<span>turns: {len(turn_files)}</span>
<span>total cost: ${total_cost:.4f}</span>
</div>
</div>
{_join_with_divider(all_blocks)}
</body>
</html>"""


def _render_acp_trajectory(
    trial_dir: Path, acp_path: Path, prompts: list[str] | None
) -> str:
    """Render an ACP trajectory JSONL file as HTML."""
    events = [
        json.loads(line) for line in acp_path.read_text().splitlines() if line.strip()
    ]

    # Load result.json for metadata
    result_data = {}
    result_path = trial_dir / "result.json"
    if result_path.exists():
        result_data = json.loads(result_path.read_text())

    blocks = []

    # Show prompts
    for i, prompt in enumerate(prompts or []):
        blocks.append(
            f'<div class="step prompt">'
            f'<div class="step-header"><span class="label prompt">PROMPT {i + 1}</span></div>'
            f'<div class="msg">{html.escape(prompt[:500])}</div>'
            f"</div>"
        )

    # Show events
    for event in events:
        etype = event.get("type", "")
        if etype == "tool_call":
            kind = html.escape(event.get("kind", ""))
            title = html.escape(event.get("title", ""))
            status = event.get("status", "")
            blocks.append(
                f'<div class="step agent">'
                f'<div class="tool"><span class="tool-name">{kind}</span> {title}</div>'
                f'<div class="metrics">{status}</div>'
                f"</div>"
            )
        elif etype == "agent_message":
            text = html.escape(event.get("text", ""))
            blocks.append(
                f'<div class="step agent"><div class="msg">{text[:500]}</div></div>'
            )
        elif etype == "agent_thought":
            text = html.escape(event.get("text", ""))
            blocks.append(
                f'<div class="step agent"><div class="thinking">{text[:500]}</div></div>'
            )

    # Result summary
    if result_data:
        agent = html.escape(result_data.get("agent_name", "?"))
        rewards = result_data.get("rewards", {})
        n_tools = result_data.get("n_tool_calls", 0)
        n_prompts = result_data.get("n_prompts", 0)
        blocks.append(
            f'<div class="step result">'
            f'<div class="step-header"><span class="label result">RESULT</span></div>'
            f'<div class="msg">Agent: {agent} | Rewards: {rewards} | '
            f"Tool calls: {n_tools} | Prompts: {n_prompts}</div>"
            f"</div>"
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>benchflow — {html.escape(trial_dir.name)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; max-width: 960px; margin: 0 auto; }}
.header {{ border-bottom: 1px solid #30363d; padding-bottom: 16px; margin-bottom: 24px; }}
.header h1 {{ font-size: 20px; color: #f0f6fc; }}
.step {{ margin-bottom: 4px; padding: 10px 14px; border-radius: 6px; }}
.step.prompt {{ background: #0d1f3c; border: 1px solid #1f3a5f; margin-bottom: 12px; }}
.step.agent {{ background: #161b22; border: 1px solid #30363d; }}
.step.result {{ background: #1a2f1a; border: 1px solid #238636; margin-top: 12px; }}
.step-header {{ margin-bottom: 6px; }}
.label {{ padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
.label.prompt {{ background: #1f3a5f; color: #58a6ff; }}
.label.result {{ background: #1a2f1a; color: #3fb950; }}
.msg {{ font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
.thinking {{ font-size: 13px; color: #8b949e; font-style: italic; padding: 8px; background: #0d1117; border-radius: 4px; border-left: 3px solid #484f58; }}
.tool {{ margin-bottom: 4px; }}
.tool-name {{ background: #2d333b; color: #f0883e; padding: 2px 8px; border-radius: 4px; font-family: monospace; font-size: 13px; font-weight: 600; }}
.metrics {{ font-size: 11px; color: #484f58; margin-top: 4px; }}
</style></head><body>
<div class="header"><h1>{html.escape(trial_dir.name)}</h1></div>
{"".join(blocks)}
</body></html>"""


def _join_with_divider(blocks: list[str]) -> str:
    return '<div class="turn-divider"></div>'.join(blocks)


def serve(trial_path: str, port: int = 8888, prompts: list[str] | None = None) -> None:
    """Serve a trial directory as a web page."""
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    path = Path(trial_path)
    if not path.is_dir():
        print(f"Not a directory: {path}")
        sys.exit(1)

    html_content = render_trial(path, prompts)
    (path / "trajectory.html").write_text(html_content)

    print(f"Trajectory viewer: http://localhost:{port}")
    print(f"Trial: {path}")
    print("Press Ctrl+C to stop\n")

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode())

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("localhost", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m benchflow.viewer <trial_dir> [port]")
        sys.exit(1)
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
    serve(sys.argv[1], port)
