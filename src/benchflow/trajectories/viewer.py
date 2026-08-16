"""Trajectory viewer — renders Claude Code stream-json, Codex sessions, and ACP JSONL as HTML.

Works with trial directories (`turn*.txt` or `trajectory/acp_trajectory.jsonl`)
and with a raw session JSONL file. No ATIF conversion.
"""

import html
import json
import sys
from pathlib import Path

_THINKING_PREVIEW = 600  # max chars for thinking block preview
_ARGS_PREVIEW = 300  # max chars for tool args display
_CONTENT_PREVIEW = 200  # max chars for write/agent content preview
_RESULT_PREVIEW = 300  # max chars for result summary

# Shared stylesheet for all viewer pages, matching the www.benchflow.ai design
# language (light monochrome: near-white page, white cards, near-black ink,
# Satoshi/Google Sans Code with system-safe fallbacks). Kept inline so pages
# work fully offline with no external requests; one constant so the three
# templates stop drifting apart.
#
# The site itself is deliberately achromatic, so tool-call accents below are
# muted GitHub-label-style pastels (pale chip background + darker same-hue
# text + a soft left border strip) that read as annotations rather than
# fighting the monochrome base. The dark #141414 code treatment is reserved
# for terminal output of shell commands; everything else stays light.
_VIEWER_CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --background: #fafafa; --card: #ffffff;
  --ink: #0a0a0a; --ink-secondary: #404040; --muted: #737373; --faint: #a1a1a1;
  --border: #e5e5e5; --rule-strong: #c7c7c7; --secondary: #f5f5f5;
  --code-bg: #141414; --code-ink: #ececec;
  --radius: 8px;
  --font-sans: "Satoshi", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "Google Sans Code", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
body { font-family: var(--font-sans); background: var(--background); color: var(--ink); padding: 28px 20px 48px; max-width: 960px; margin: 0 auto; line-height: 1.6; -webkit-font-smoothing: antialiased; }
::selection { background: var(--secondary); color: var(--ink); }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 9999px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
.wordmark { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
.wordmark svg { width: 18px; height: 18px; flex: none; color: var(--ink); }
.wordmark .brand { font-weight: 600; font-size: 15px; letter-spacing: -0.01em; color: var(--ink); }
.wordmark .app { font-family: var(--font-mono); font-size: 10.5px; font-weight: 500; color: var(--ink-secondary); background: var(--secondary); border: 1px solid var(--border); border-radius: 9999px; padding: 3px 10px; }
.header { border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 24px; }
.header h1 { font-size: 20px; font-weight: 600; letter-spacing: -0.02em; color: var(--ink); margin-bottom: 10px; overflow-wrap: anywhere; }
.meta { display: flex; gap: 8px; flex-wrap: wrap; font-size: 13px; color: var(--muted); }
.meta span { font-family: var(--font-mono); font-size: 11px; font-weight: 500; color: var(--ink-secondary); background: var(--secondary); padding: 3px 9px; border-radius: 4px; border: 1px solid var(--border); }
.step { margin-bottom: 8px; padding: 12px 16px; border-radius: var(--radius); background: var(--card); border: 1px solid var(--border); box-shadow: 0 1px 2px rgba(10, 10, 10, 0.04); }
.step.prompt { background: var(--secondary); border-color: var(--rule-strong); margin-bottom: 14px; }
.step.output { background: var(--card); border-color: var(--border); padding: 10px 16px; }
.step.output pre { color: var(--ink-secondary); font-family: var(--font-mono); font-size: 12px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; }
.step.output.term { background: var(--code-bg); border-color: var(--code-bg); }
.step.output.term pre { color: var(--code-ink); }
.step.result { background: var(--ink); border-color: var(--ink); margin-top: 14px; }
.step.result .msg { color: var(--background); }
.step-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.label { display: inline-flex; align-items: center; font-family: var(--font-mono); padding: 2px 10px; border-radius: 9999px; font-weight: 600; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.04em; }
.label.prompt { background: var(--ink); color: var(--background); }
.label.result { background: var(--background); color: var(--ink); }
.meta-inline { font-family: var(--font-mono); font-size: 11px; color: var(--muted); }
.step.result .meta-inline { color: var(--faint); }
.msg { font-size: 14px; line-height: 1.65; white-space: pre-wrap; word-break: break-word; }
.thinking { font-size: 13px; color: var(--muted); font-style: italic; margin-bottom: 8px; padding: 8px 12px; background: var(--secondary); border-radius: 4px; border-left: 2px solid var(--rule-strong); white-space: pre-wrap; word-break: break-word; }
.tool { margin-bottom: 6px; }
.tool-name { display: inline-flex; align-items: center; font-family: var(--font-mono); font-size: 11px; font-weight: 600; color: var(--acc-ink, var(--ink)); background: var(--acc-bg, var(--secondary)); border: 1px solid var(--acc-line, var(--border)); padding: 2px 9px; border-radius: 4px; }
.tool-args { margin-top: 6px; font-family: var(--font-mono); font-size: 12px; line-height: 1.7; color: var(--ink-secondary); background: var(--secondary); border: 1px solid var(--border); padding: 10px 12px; border-radius: 6px; white-space: pre-wrap; word-break: break-word; }
.step.tool-step { border-left: 3px solid var(--acc-strip, var(--rule-strong)); }
.acc-bash  { --acc-bg: #f7efda; --acc-line: #ecdcb2; --acc-ink: #8a5a12; --acc-strip: #dcb45e; }
.acc-edit  { --acc-bg: #e8f0fa; --acc-line: #d0dff1; --acc-ink: #1d4e89; --acc-strip: #7fa8d8; }
.acc-read  { --acc-bg: #e5f2ec; --acc-line: #cbe3d7; --acc-ink: #1a6b52; --acc-strip: #74bda0; }
.acc-agent { --acc-bg: #efeaf8; --acc-line: #ded3ef; --acc-ink: #5b3e96; --acc-strip: #a78fd6; }
.acc-web   { --acc-bg: #e3f1f6; --acc-line: #c8e2ea; --acc-ink: #176478; --acc-strip: #6fb6ca; }
.acc-other { --acc-bg: var(--secondary); --acc-line: var(--border); --acc-ink: var(--ink-secondary); --acc-strip: var(--rule-strong); }
.metrics { font-family: var(--font-mono); font-size: 11px; color: var(--faint); margin-top: 4px; }
.turn-divider { border-top: 1px solid var(--border); margin: 24px 0; }
"""

# Sticky bottom confirmation bar injected only in --confirm mode. Styling
# reuses the page's CSS variables (white surface, top border, ink text, pill
# buttons) so it reads as part of the same www.benchflow.ai design language.
# The <style> block rides along with the snippet so plain (non-confirm) pages
# carry zero confirm markup or CSS.
_CONFIRM_BAR_HTML = """\
<style>
body { padding-bottom: 104px; }
.confirm-bar { position: fixed; bottom: 0; left: 0; right: 0; background: var(--card); border-top: 1px solid var(--border); box-shadow: 0 -1px 3px rgba(10, 10, 10, 0.05); z-index: 10; }
.confirm-inner { max-width: 960px; margin: 0 auto; padding: 14px 20px; display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.confirm-question { font-size: 14px; font-weight: 500; color: var(--ink); }
.confirm-actions { display: flex; gap: 8px; flex: none; }
.confirm-btn { font-family: var(--font-sans); font-size: 13px; font-weight: 600; letter-spacing: -0.01em; padding: 8px 18px; border-radius: 9999px; cursor: pointer; transition: background 0.15s ease; }
.confirm-btn.approve { background: var(--ink); color: var(--background); border: 1px solid var(--ink); }
.confirm-btn.approve:hover { background: #262626; }
.confirm-btn.reject { background: var(--card); color: var(--ink); border: 1px solid var(--rule-strong); }
.confirm-btn.reject:hover { background: var(--secondary); }
</style>
<div class="confirm-bar">
<div class="confirm-inner" id="confirm-inner">
<span class="confirm-question">Submit this trajectory to the BenchFlow eval prize?</span>
<div class="confirm-actions">
<button class="confirm-btn approve" onclick="__benchDecide('approve')">Approve &amp; submit</button>
<button class="confirm-btn reject" onclick="__benchDecide('reject')">Not this one</button>
</div>
</div>
</div>
<script>
async function __benchDecide(choice) {
  try {
    await fetch("/decision", { method: "POST", body: choice });
  } catch (err) {
    // The server exits right after answering; a dropped socket is still a
    // delivered decision, so fall through to the confirmation note.
  }
  document.getElementById("confirm-inner").innerHTML =
    choice === "approve"
      ? '<span class="confirm-question">Approved — go back to your agent.</span>'
      : '<span class="confirm-question">Rejected — tell your agent which session instead.</span>';
}
</script>
"""


def _inject_confirm_bar(page: str) -> str:
    """Append the confirm bar just before </body> (or at the end as fallback)."""
    if "</body>" in page:
        return page.replace("</body>", f"{_CONFIRM_BAR_HTML}</body>", 1)
    return page + _CONFIRM_BAR_HTML


# Small BenchFlow wordmark header (inline SVG logo from the site's icon set —
# no external requests) shown above the page title on every viewer page.
_WORDMARK_HTML = (
    '<div class="wordmark">'
    '<svg viewBox="0 0 514 512" fill="currentColor" aria-hidden="true">'
    '<path fill-rule="evenodd" clip-rule="evenodd" d="M445.422 66.4597L511.882 0'
    "L389.022 293.965L295.129 387.859L0 511.882L69.3042 442.577L81.0101 454.283"
    "L89.0554 446.238L77.3493 434.532L130.65 381.232L162.469 413.051L170.514 40"
    "5.006L138.695 373.187L191.995 319.887L203.701 331.593L211.746 323.547L200."
    "04 311.841L253.34 258.541L285.16 290.36L293.205 282.315L261.386 250.496L31"
    "4.686 197.196L326.392 208.902L334.437 200.856L322.731 189.15L376.031 135.8"
    "5L407.851 167.669L415.896 159.624L384.077 127.805L437.377 74.5049L449.083 "
    "86.2108L457.128 78.1656L445.422 66.4597ZM399.127 389.865V299.369L513.197 2"
    "6.4333V503.935L399.127 389.865ZM391.061 397.931L505.132 512.001H29.1594L30"
    '0.605 397.931H391.061Z"/>'
    "</svg>"
    '<span class="brand">BenchFlow</span>'
    '<span class="app">trajectory viewer</span>'
    "</div>"
)


# Tool kind → accent class. Substring matching so it covers Claude Code tool
# names (Bash, Write, Edit, Read, Agent, WebSearch, ...), ACP kinds (execute,
# edit, read, search, fetch, ...), and Codex function names (shell, ...).
# Order matters: earlier entries win (e.g. "websearch" hits web before read).
_TOOL_ACCENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("acc-web", ("web", "search", "fetch", "grep", "glob", "browser")),
    ("acc-bash", ("bash", "shell", "exec", "terminal", "command")),
    ("acc-edit", ("write", "edit", "patch", "delete", "move", "notebook")),
    ("acc-read", ("read", "cat", "view", "ls", "list")),
    ("acc-agent", ("agent", "task", "skill", "oracle")),
)


def _tool_accent_class(name: str) -> str:
    """Map a tool name/kind to its accent CSS class (presentation only)."""
    lowered = name.lower()
    for css_class, needles in _TOOL_ACCENTS:
        if any(needle in lowered for needle in needles):
            return css_class
    return "acc-other"


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
    # tool_use id → accent class, so each tool_result can match its call's
    # accent and shell results alone keep the dark terminal treatment.
    accent_by_tool_id: dict[str, str] = {}

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

                    accent = _tool_accent_class(block.get("name", ""))
                    if block.get("id"):
                        accent_by_tool_id[str(block["id"])] = accent
                    blocks.append(
                        f'<div class="step agent tool-step {accent}">'
                        f"{''.join(parts)}</div>"
                    )

        elif etype == "user":
            content = event.get("message", {}).get("content", "")
            if isinstance(content, str) and content.strip():
                blocks.append(_user_prompt_html(content))
            elif isinstance(content, list):
                texts: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
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
                        accent = accent_by_tool_id.get(
                            str(block.get("tool_use_id", "")), "acc-other"
                        )
                        term = " term" if accent == "acc-bash" else ""
                        blocks.append(
                            f'<div class="step output tool-step {accent}{term}">'
                            f"<pre>{display}</pre></div>"
                        )
                    elif block.get("type") == "text" and block.get("text"):
                        texts.append(str(block["text"]))
                if texts:
                    blocks.append(_user_prompt_html("\n".join(texts)))

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


# Sentinel HTML returned by render_rollout when a directory holds no trajectory
# files. serve() keys off it to fail fast instead of writing/serving a blank page.
_NO_TRAJECTORIES_HTML = "<p>No trajectory files found</p>"


def _user_prompt_html(text: str) -> str:
    return (
        '<div class="step prompt">'
        '<div class="step-header"><span class="label prompt">USER</span></div>'
        f'<div class="msg">{html.escape(text[:2000])}</div>'
        "</div>"
    )


def _parse_jsonl(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def render_rollout(rollout_dir: Path, prompts: list[str] | None = None) -> str:
    """Render a full trial (multiple turns) as HTML.

    Auto-detects format:
    - turn*.txt → Claude Code stream-json
    - trajectory/acp_trajectory.jsonl → ACP session events
    - prompts.json → used for prompt labels if available
    """
    # Try loading prompts from prompts.json if not provided. A corrupt file is
    # auxiliary (it only supplies prompt labels) — degrade to default labels
    # rather than crashing the whole view with a raw JSONDecodeError traceback.
    if prompts is None and (rollout_dir / "prompts.json").exists():
        try:
            prompts = json.loads((rollout_dir / "prompts.json").read_text())
        except (json.JSONDecodeError, OSError):
            prompts = None

    # Auto-detect format
    turn_files = sorted(rollout_dir.glob("turn*.txt"))
    acp_traj = rollout_dir / "trajectory" / "acp_trajectory.jsonl"

    if not turn_files and acp_traj.exists():
        return _render_acp_trajectory(rollout_dir, acp_traj, prompts)

    if not turn_files:
        # The given dir has no trajectory of its own. If it's a job directory
        # (the natural value from `eval run`'s "Artifacts:" line), point at its
        # rollout subdirectories instead of showing a blank page.
        try:
            rollouts = sorted(
                d.name
                for d in rollout_dir.iterdir()
                if d.is_dir()
                and (
                    any(d.glob("turn*.txt"))
                    or (d / "trajectory" / "acp_trajectory.jsonl").exists()
                )
            )
        except OSError:
            rollouts = []
        if rollouts:
            items = "".join(f"<li><code>{html.escape(r)}</code></li>" for r in rollouts)
            return (
                f"<p>No trajectory here — <code>{html.escape(rollout_dir.name)}</code> "
                f"looks like a job directory with {len(rollouts)} rollout(s). "
                f"View one with <code>bench eval view {html.escape(rollout_dir.name)}/"
                f"&lt;rollout&gt;</code>:</p><ul>{items}</ul>"
            )
        return _NO_TRAJECTORIES_HTML

    # Default prompts
    if prompts is None:
        prompts = [
            f"(turn {i + 1} prompt — not captured in stream)"
            for i in range(len(turn_files))
        ]

    # Pad prompts if fewer than turns
    while len(prompts) < len(turn_files):
        prompts.append("")

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

    # `or "?"` (not just a .get default): a present-but-null value in the
    # stream-json (e.g. "session_id": null) bypasses the default and would crash
    # html.escape() / the [:16] slice below with a raw TypeError.
    session_id = str(sys_event.get("session_id") or "?")
    model = str(sys_event.get("model") or "?")
    version = str(sys_event.get("claude_code_version") or "?")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>benchflow — {rollout_dir.name}</title>
<style>
{_VIEWER_CSS}</style>
</head>
<body>
<div class="header">
{_WORDMARK_HTML}
<h1>{html.escape(rollout_dir.name)}</h1>
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
    rollout_dir: Path, acp_path: Path, prompts: list[str] | None
) -> str:
    """Render an ACP trajectory JSONL file as HTML."""
    events = _parse_jsonl(acp_path.read_text())
    result_data = _load_result_json(rollout_dir)
    return _render_acp_events(rollout_dir.name, events, result_data, prompts)


def _load_result_json(rollout_dir: Path) -> dict:
    result_path = rollout_dir / "result.json"
    if not result_path.exists():
        return {}
    try:
        parsed = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _render_acp_events(
    title: str,
    events: list[dict],
    result_data: dict | None = None,
    prompts: list[str] | None = None,
) -> str:
    result_data = result_data or {}
    blocks = []

    # Show prompts at top only if trajectory has no inline user_message events
    has_inline_prompts = any(e.get("type") == "user_message" for e in events)
    if not has_inline_prompts:
        for i, prompt in enumerate(prompts or []):
            blocks.append(
                f'<div class="step prompt">'
                f'<div class="step-header"><span class="label prompt">PROMPT {i + 1}</span></div>'
                f'<div class="msg">{html.escape(prompt[:500])}</div>'
                f"</div>"
            )

    # Show events
    prompt_counter = 0
    for event in events:
        etype = event.get("type", "")
        if etype == "user_message":
            prompt_counter += 1
            text = html.escape(event.get("text", ""))
            blocks.append(
                f'<div class="step prompt">'
                f'<div class="step-header"><span class="label prompt">PROMPT {prompt_counter}</span></div>'
                f'<div class="msg">{text[:500]}</div>'
                f"</div>"
            )
        elif etype == "tool_call":
            kind = html.escape(event.get("kind", ""))
            event_title = html.escape(event.get("title", ""))
            status = event.get("status", "")
            # ACP kinds are coarse ("other" covers Skill/Task/...); when the
            # kind doesn't map, fall back to the human title for the accent.
            accent = _tool_accent_class(event.get("kind", ""))
            if accent == "acc-other":
                accent = _tool_accent_class(event.get("title", ""))
            blocks.append(
                f'<div class="step agent tool-step {accent}">'
                f'<div class="tool"><span class="tool-name">{kind}</span> {event_title}</div>'
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
<html><head><meta charset="utf-8"><title>benchflow — {html.escape(title)}</title>
<style>
{_VIEWER_CSS}</style></head><body>
<div class="header">{_WORDMARK_HTML}<h1>{html.escape(title)}</h1></div>
{"".join(blocks)}
</body></html>"""


def _join_with_divider(blocks: list[str]) -> str:
    return '<div class="turn-divider"></div>'.join(blocks)


def _looks_like_codex(events: list[dict]) -> bool:
    return any(
        e.get("type") in {"session_meta", "response_item", "event_msg", "turn_context"}
        and isinstance(e.get("payload"), dict)
        for e in events[:30]
    )


def _looks_like_acp(events: list[dict]) -> bool:
    return any(
        e.get("type") in {"tool_call", "agent_thought", "user_message"}
        for e in events[:30]
    )


def _codex_message_text(payload: dict) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text") or block.get("input_text") or ""
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _codex_to_acp(events: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for event in events:
        raw_payload = event.get("payload")
        payload: dict = raw_payload if isinstance(raw_payload, dict) else {}
        top = event.get("type")
        if top == "event_msg":
            inner = payload.get("type")
            if inner == "user_message":
                converted.append(
                    {"type": "user_message", "text": str(payload.get("message") or "")}
                )
            elif inner == "agent_message":
                converted.append(
                    {"type": "agent_message", "text": str(payload.get("message") or "")}
                )
        elif top == "response_item":
            inner = payload.get("type")
            if inner == "function_call":
                args = payload.get("arguments") or ""
                if not isinstance(args, str):
                    args = json.dumps(args)
                converted.append(
                    {
                        "type": "tool_call",
                        "kind": str(payload.get("name") or "tool"),
                        "title": args[:300],
                        "status": str(payload.get("status") or ""),
                    }
                )
            elif inner == "message" and payload.get("role") in {"user", "assistant"}:
                text = _codex_message_text(payload)
                if not text:
                    continue
                kind = (
                    "user_message" if payload.get("role") == "user" else "agent_message"
                )
                converted.append({"type": kind, "text": text})
    return converted


def render_jsonl_file(path: Path) -> str:
    """Render a Claude Code, Codex, or ACP session JSONL file as HTML."""
    try:
        events = _parse_jsonl(path.read_text())
    except OSError:
        return _NO_TRAJECTORIES_HTML
    if not events:
        return _NO_TRAJECTORIES_HTML
    if _looks_like_codex(events):
        converted = _codex_to_acp(events)
        if not converted:
            return _NO_TRAJECTORIES_HTML
        return _render_acp_events(path.name, converted, {})
    if _looks_like_acp(events):
        return _render_acp_events(path.name, events, _load_result_json(path.parent))
    body = render_turn(events, 1, "")
    if not body.strip():
        return _NO_TRAJECTORIES_HTML
    return _stream_json_page(path.name, events, [body])


def _stream_json_page(title: str, events: list[dict], turn_blocks: list[str]) -> str:
    sys_event = next((e for e in events if e.get("type") == "system"), {})
    total_cost = 0.0
    for event in events:
        if event.get("type") == "result":
            total_cost += float(event.get("total_cost_usd") or 0)
    session_id = str(sys_event.get("session_id") or "?")
    model = str(sys_event.get("model") or "?")
    version = str(sys_event.get("claude_code_version") or "?")
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>benchflow — {html.escape(title)}</title>
<style>
{_VIEWER_CSS}</style>
</head>
<body>
<div class="header">
{_WORDMARK_HTML}
<h1>{html.escape(title)}</h1>
<div class="meta">
<span>model: {html.escape(model)}</span>
<span>session: {html.escape(session_id[:16])}...</span>
<span>claude code: {html.escape(version)}</span>
<span>total cost: ${total_cost:.4f}</span>
</div>
</div>
{_join_with_divider(turn_blocks)}
</body>
</html>"""


def serve(
    rollout_path: str,
    port: int = 8888,
    prompts: list[str] | None = None,
    confirm: bool = False,
) -> str | None:
    """Serve a trial directory or a session JSONL file as a web page.

    With ``confirm=True`` the page carries an Approve/Reject bar posting to
    ``/decision``; the first valid decision shuts the server down and is
    returned as ``"approved"`` or ``"rejected"`` after printing a
    machine-readable ``DECISION: <value>`` line to stdout. Without it the
    server runs until Ctrl+C and the return value is ``None`` — exactly the
    pre-confirm behavior (no bar, no POST endpoint).
    """
    import threading
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    path = Path(rollout_path)
    write_sidecar = False
    if path.is_file():
        html_content = render_jsonl_file(path)
    elif path.is_dir():
        html_content = render_rollout(path, prompts)
        write_sidecar = True
    else:
        print(f"Not a file or directory: {path}")
        sys.exit(1)

    if html_content == _NO_TRAJECTORIES_HTML:
        # Don't write a blank trajectory.html into an unrelated directory or
        # start a server for nothing — fail fast like the not-a-directory path.
        print(f"No trajectories found in {path}")
        sys.exit(1)
    if write_sidecar:
        # The sidecar stays the plain page: the confirm bar is a one-shot
        # interaction against this live server, not part of the artifact.
        (path / "trajectory.html").write_text(html_content)
    if confirm:
        html_content = _inject_confirm_bar(html_content)

    print(f"Trajectory viewer: http://localhost:{port}")
    print(f"Trial: {path}")
    if confirm:
        print("Waiting for Approve / Not this one in the browser (Ctrl+C to stop)\n")
    else:
        print("Press Ctrl+C to stop\n")

    decision: str | None = None

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode())

        def log_message(self, format, *args):
            pass

    handler_cls: type[SimpleHTTPRequestHandler] = Handler

    if confirm:

        class ConfirmHandler(Handler):
            def do_POST(self):
                nonlocal decision
                if self.path != "/decision":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode("utf-8", "replace").strip()
                if body not in ("approve", "reject"):
                    self.send_error(400, "Body must be 'approve' or 'reject'")
                    return
                decision = "approved" if body == "approve" else "rejected"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(decision.encode())
                # shutdown() blocks until serve_forever returns, and this
                # handler runs inside that loop — stop from a helper thread.
                threading.Thread(target=server.shutdown, daemon=True).start()

        handler_cls = ConfirmHandler

    server = HTTPServer(("localhost", port), handler_cls)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        return None
    finally:
        server.server_close()

    if decision is not None:
        print(f"DECISION: {decision}", flush=True)
    return decision


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m benchflow.viewer <rollout_dir_or_jsonl> [port]")
        sys.exit(1)
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
    serve(sys.argv[1], port)
