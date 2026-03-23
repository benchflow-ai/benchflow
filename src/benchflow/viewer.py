"""Trajectory viewer — renders Claude Code stream-json as HTML.

Works directly with raw turn*.txt files. No ATIF conversion.
"""

import html
import json
import sys
from pathlib import Path


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
                            f'<div class="thinking">{html.escape(pending_thinking[:600])}'
                            f"{'...' if len(pending_thinking) > 600 else ''}</div>"
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
                            content_preview = args["content"][:200]
                            arg_display += f"\n{html.escape(content_preview)}{'...' if len(args['content']) > 200 else ''}"
                    elif name == "Agent":
                        arg_display = html.escape(str(args.get("prompt", ""))[:200])
                    else:
                        arg_display = html.escape(json.dumps(args, indent=2)[:300])

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
            result_text = html.escape(event.get("result", "")[:300])
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
                f'<div class="thinking">{html.escape(pending_thinking[:600])}</div>'
            )
        if pending_text:
            parts.append(f'<div class="msg">{html.escape(pending_text)}</div>')
        blocks.append(f'<div class="step agent">{"".join(parts)}</div>')

    return "\n".join(blocks)


def render_trial(trial_dir: Path, prompts: list[str] | None = None) -> str:
    """Render a full trial (multiple turns) as HTML."""
    turn_files = sorted(trial_dir.glob("turn*.txt"))
    if not turn_files:
        return "<p>No turn files found</p>"

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
    first_events = [
        json.loads(line)
        for line in turn_files[0].read_text().splitlines()
        if line.strip()
    ]
    sys_event = next((e for e in first_events if e.get("type") == "system"), {})
    total_cost = 0
    total_turns_count = 0

    all_blocks = []
    for i, tf in enumerate(turn_files):
        events = [
            json.loads(line) for line in tf.read_text().splitlines() if line.strip()
        ]
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
