"""Trajectory viewer — renders Claude Code stream-json, Codex sessions, and ACP JSONL as HTML.

Works with trial directories (`turn*.txt` or `trajectory/acp_trajectory.jsonl`),
with a raw session JSONL file, and — for a directory *of* rollouts — as a
browsable multi-run server (sidebar + JSON API, see ``_serve_browse``). ACP
rollout directories render through the interactive ``viewer_template.html``
page; stream-json/Codex/raw-file paths keep the shared inline-CSS renderers.
No ATIF conversion.
"""

import html
import json
import os
import sys
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

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
.confirm-note { flex-basis: 100%; font-size: 12.5px; color: var(--muted); }
</style>
<div class="confirm-bar">
<div class="confirm-inner" id="confirm-inner">
<!--BENCHFLOW-REDACTION-NOTE-->
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


_REDACTION_NOTE_MARKER = "<!--BENCHFLOW-REDACTION-NOTE-->"


def _confirm_bar_html(redaction_summary: str | None) -> str:
    """Confirm-bar snippet, optionally carrying the redaction-summary note.

    The note is presentation-only text the caller composed (the upload skill
    lifts it from ``bench traj upload --dry-run``); the viewer itself never
    redacts, so without a summary the bar is byte-identical to the plain
    ``--confirm`` bar.
    """
    note = ""
    if redaction_summary:
        note = (
            '<div class="confirm-note">Before upload, BenchFlow masks: '
            f"{html.escape(redaction_summary)}. "
            "Originals never leave this machine.</div>"
        )
    return _CONFIRM_BAR_HTML.replace(_REDACTION_NOTE_MARKER, note, 1)


def _inject_confirm_bar(page: str, redaction_summary: str | None = None) -> str:
    """Append the confirm bar just before </body> (or at the end as fallback)."""
    bar = _confirm_bar_html(redaction_summary)
    if "</body>" in page:
        return page.replace("</body>", f"{bar}</body>", 1)
    return page + bar


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
            prompts = json.loads(
                (rollout_dir / "prompts.json").read_text(
                    encoding="utf-8", errors="replace"
                )
            )
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

    all_events: list[dict] = []
    all_blocks = []
    for i, tf in enumerate(turn_files):
        events = _parse_jsonl(tf.read_text(encoding="utf-8", errors="replace"))
        all_events.extend(events)
        all_blocks.append(render_turn(events, i + 1, prompts[i]))

    badges = _stream_header_badges(all_events)
    badges.append(("turns", str(len(turn_files))))
    total_cost = _result_cost(all_events)
    if total_cost is not None:
        badges.append(("total cost", f"${total_cost:.4f}"))

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
{_meta_badges_html(badges)}</div>
{_join_with_divider(all_blocks)}
</body>
</html>"""


# ── ACP (canonical) renderer: payload + interactive template ─────────────
#
# Rollout DIRECTORIES render through viewer_template.html: Python assembles
# one JSON payload (normalized steps + result/timing/verifier metadata) and
# the self-contained vanilla-JS page renders it client-side. Raw session
# FILES keep the inline `_render_acp_events` renderer below (its output is
# pinned server-side by the jsonl-session tests).

_TEMPLATE_NAME = "viewer_template.html"
_PAYLOAD_PLACEHOLDER = "__BENCHFLOW_PAYLOAD__"
_TITLE_PLACEHOLDER = "__BENCHFLOW_TITLE__"

# result.json diagnostic blocks surfaced in the header banner. The live list
# derives from DIAGNOSTIC_REGISTRY so new diagnostics (e.g. behavior flags)
# appear without this module drifting; the fallback pins the 0.7.4 set.
_DIAGNOSTIC_KEYS_FALLBACK = (
    "idle_timeout_info",
    "agent_timeout_info",
    "sandbox_startup_info",
    "transport_error_info",
    "verifier_timeout_info",
    "api_error_info",
    "suspected_api_error_info",
)


def _diagnostic_keys() -> tuple[str, ...]:
    """Diagnostic block keys, in registry display order."""
    try:
        from benchflow.diagnostics import DIAGNOSTIC_REGISTRY
    except Exception:  # pragma: no cover — registry is core; stay lenient
        return _DIAGNOSTIC_KEYS_FALLBACK
    return tuple(d.field for d in DIAGNOSTIC_REGISTRY)


def _load_json(path: Path) -> Any:
    # errors="replace": binary/mis-encoded sidecars degrade to a JSON parse
    # failure (→ None) instead of an unhandled UnicodeDecodeError.
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _tool_content_texts(content: Any) -> list[str]:
    """Flatten ACP tool_call content blocks to plain strings.

    Three shapes occur in captures: the canonical nested block
    ``{"type": "content", "content": {"type": "text", "text": ...}}``, a flat
    ``{"text": ...}``, and edit-tool diff blocks
    ``{"type": "diff", "path": ..., "oldText": ..., "newText": ...}``.
    """
    texts: list[str] = []
    if not isinstance(content, list):
        return texts
    for item in content:
        if not isinstance(item, dict):
            texts.append(str(item))
            continue
        inner = item.get("content")
        if isinstance(inner, dict) and "text" in inner:
            texts.append(str(inner.get("text", "")))
        elif "text" in item:
            texts.append(str(item.get("text", "")))
        elif item.get("type") == "diff":
            texts.append(
                f"diff {item.get('path', '')}\n"
                f"--- old\n{item.get('oldText') or ''}\n"
                f"+++ new\n{item.get('newText') or ''}"
            )
        else:
            texts.append(json.dumps(item, ensure_ascii=False))
    return [t for t in texts if t]


def _parse_ts(value: Any) -> float | None:
    """Lenient timestamp → epoch seconds; None for absent/unparseable.

    Accepts numeric epochs and anything ``datetime.fromisoformat`` reads
    (ISO-8601, and the space-separated ``str(datetime.now())`` form used by
    result.json's started_at).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def _normalize_steps(events: list[dict], prompts: list[str] | None) -> list[dict]:
    """Project raw ACP events onto renderer steps with server-side numbering.

    Prompt labels ("PROMPT 1") are computed here, not client-side, and the
    ``prompts`` list is embedded only when the trajectory has no inline
    user_message events — embedding both would duplicate prompt text in the
    emitted page (pinned by TestViewerCompatibility).

    Timestamps are forward-compatible passthrough: today's captures carry
    none (verified across the HF ground-truth uploads), but when the
    capture-side proposal lands (``ts`` on text events, ``started_at`` /
    ``finished_at`` on tool calls — benchflow#1033) steps gain ``t`` (epoch)
    and tools ``dur`` (seconds), and the renderer shows a timeline with no
    further changes.
    """
    steps: list[dict] = []
    prompt_counter = 0

    def add(kind: str, event: dict | None = None, **fields: Any) -> None:
        step = {"i": len(steps) + 1, "kind": kind, **fields}
        if event is not None:
            if kind == "tool":
                started = _parse_ts(event.get("started_at") or event.get("ts"))
                finished = _parse_ts(event.get("finished_at"))
                if started is not None:
                    step["t"] = started
                    if finished is not None and finished >= started:
                        step["dur"] = finished - started
            else:
                ts = _parse_ts(event.get("ts"))
                if ts is not None:
                    step["t"] = ts
        steps.append(step)

    has_inline_prompts = any(e.get("type") == "user_message" for e in events)
    if not has_inline_prompts:
        for text in prompts or []:
            prompt_counter += 1
            add("prompt", label=f"PROMPT {prompt_counter}", text=str(text))

    for event in events:
        etype = event.get("type", "")
        if etype == "user_message":
            prompt_counter += 1
            add(
                "prompt",
                event,
                label=f"PROMPT {prompt_counter}",
                text=str(event.get("text", "")),
            )
        elif etype == "agent_message":
            add("message", event, text=str(event.get("text", "")))
        elif etype == "agent_thought":
            add("thought", event, text=str(event.get("text", "")))
        elif etype == "tool_call":
            add(
                "tool",
                event,
                tool={
                    "id": event.get("tool_call_id", ""),
                    "kind": event.get("kind", "other"),
                    "title": event.get("title", ""),
                    "status": event.get("status", ""),
                    "content": _tool_content_texts(event.get("content")),
                },
            )
        elif etype == "agent_timeout":
            # Coerce to a list of strings — a crafted non-list value would
            # otherwise reach the client and break the renderer.
            pending_raw = event.get("pending_tool_call_ids")
            add(
                "timeout",
                event,
                timeout={
                    "reason": event.get("reason", ""),
                    "timeout_sec": event.get("timeout_sec"),
                    "pending": [str(x) for x in pending_raw]
                    if isinstance(pending_raw, list)
                    else [],
                    "complete": event.get("terminal_trajectory_complete"),
                },
            )
        else:
            # Unknown/future event types render as a generic card, never crash.
            add(
                "unknown",
                event,
                type=str(etype),
                text=json.dumps(event, ensure_ascii=False)[:2000],
            )
    return steps


def _build_meta(
    result_data: dict, timing: dict | None, steps: list[dict]
) -> dict[str, Any]:
    rewards = result_data.get("rewards")
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    agent_result = result_data.get("agent_result")
    usage = agent_result if isinstance(agent_result, dict) else {}

    errors: list[dict[str, str]] = []
    if result_data.get("error"):
        errors.append(
            {
                "label": str(result_data.get("error_category") or "error"),
                "text": str(result_data["error"]),
            }
        )
    if result_data.get("verifier_error"):
        errors.append(
            {
                "label": str(
                    result_data.get("verifier_error_category") or "verifier error"
                ),
                "text": str(result_data["verifier_error"]),
            }
        )
    if result_data.get("export_error"):
        errors.append(
            {"label": "export error", "text": str(result_data["export_error"])}
        )
    # Diagnostics that ride along an actual error render as error banners;
    # on an otherwise-clean rollout they are behavior flags (e.g. #1025's
    # chat-only completion) and render neutrally. All three error channels
    # count as "actual error".
    has_error = bool(
        result_data.get("error")
        or result_data.get("verifier_error")
        or result_data.get("export_error")
    )
    for key in _diagnostic_keys():
        value = result_data.get(key)
        if value:
            errors.append(
                {
                    "label": key.removesuffix("_info").replace("_", " "),
                    "text": json.dumps(value, ensure_ascii=False, default=str)[:2000],
                    "level": "error" if has_error else "info",
                }
            )

    return {
        "task_name": result_data.get("task_name"),
        "agent_name": result_data.get("agent_name") or result_data.get("agent"),
        "model": result_data.get("model"),
        "skill_mode": result_data.get("skill_mode"),
        "reward": reward,
        "usage": usage,
        "counts": {
            "prompts": sum(1 for s in steps if s["kind"] == "prompt"),
            "messages": sum(1 for s in steps if s["kind"] == "message"),
            "thoughts": sum(1 for s in steps if s["kind"] == "thought"),
            "tools": sum(1 for s in steps if s["kind"] == "tool"),
        },
        "timing": timing,
        "duration_sec": (timing or {}).get("total"),
        "errors": errors,
        "trajectory_source": result_data.get("trajectory_source"),
        "partial_trajectory": result_data.get("partial_trajectory"),
        "started_at": result_data.get("started_at"),
        "finished_at": result_data.get("finished_at"),
    }


def _load_verifier(rollout_dir: Path) -> dict[str, Any]:
    vdir = rollout_dir / "verifier"
    reward = _read_text(vdir / "reward.txt")
    ctrf_tests = None
    ctrf = _load_json(vdir / "ctrf.json")
    if isinstance(ctrf, dict):
        raw_tests = (ctrf.get("results") or {}).get("tests")
        if isinstance(raw_tests, list):
            ctrf_tests = [
                {
                    "name": str(t.get("name", "")),
                    "status": str(t.get("status", "")),
                    "duration": t.get("duration"),
                }
                for t in raw_tests
                if isinstance(t, dict)
            ]
    return {
        "reward": reward.strip() if reward else None,
        "stdout": _read_text(vdir / "test-stdout.txt"),
        "stderr": _read_text(vdir / "test-stderr.txt"),
        "ctrf": ctrf_tests,
    }


def _load_template() -> str:
    return (resources.files("benchflow.trajectories") / _TEMPLATE_NAME).read_text(
        encoding="utf-8"
    )


def _safe_json(obj: Any) -> str:
    """Serialize untrusted payload data to UTF-8-safe JSON text.

    Lone surrogates (possible in crafted trajectory text) survive
    ``json.dumps(ensure_ascii=False)`` but crash utf-8 encoding later in
    ``serve()``; replace them here.
    """
    data = json.dumps(obj, ensure_ascii=False, default=str)
    return data.encode("utf-8", errors="replace").decode("utf-8")


def _render_shell(title: str, boot: dict[str, Any]) -> str:
    """Inject a boot document into the template (script-breakout escaped)."""
    page = _load_template()
    page = page.replace(_TITLE_PLACEHOLDER, html.escape(title), 1)
    return page.replace(_PAYLOAD_PLACEHOLDER, _safe_json(boot).replace("</", "<\\/"), 1)


def _build_acp_payload(rollout_dir: Path, prompts: list[str] | None) -> dict[str, Any]:
    """Assemble the renderer payload for one canonical ACP rollout dir."""
    acp_path = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    try:
        text = acp_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    events = _parse_jsonl(text)

    if prompts is None:
        loaded = _load_json(rollout_dir / "prompts.json")
        prompts = loaded if isinstance(loaded, list) else None

    result_data = _load_result_json(rollout_dir)
    timing = _load_json(rollout_dir / "timing.json")
    if not isinstance(timing, dict):
        embedded = result_data.get("timing")
        timing = embedded if isinstance(embedded, dict) else None

    steps = _normalize_steps(events, prompts)
    return {
        "schema_version": 1,
        "rollout_name": rollout_dir.name,
        "meta": _build_meta(result_data, timing, steps),
        "steps": steps,
        "verifier": _load_verifier(rollout_dir),
    }


def _is_acp_rollout_dir(path: Path) -> bool:
    return (path / "trajectory" / "acp_trajectory.jsonl").exists()


def _runs_cap() -> int:
    """Browse-mode run cap (``BENCHFLOW_VIEWER_MAX_RUNS`` overrides, default 500)."""
    try:
        return max(1, int(os.environ.get("BENCHFLOW_VIEWER_MAX_RUNS", "500")))
    except ValueError:
        return 500


def _discover_rollouts(
    base: Path, max_depth: int = 4, cap: int | None = None
) -> list[str]:
    """Relative paths of ACP rollout dirs under ``base``, sorted, capped.

    The returned ids double as the ``/api/rollout?id=`` whitelist: an id is
    only ever resolved by exact membership here, so crafted ids (``../``
    traversal and the like) can never reach the filesystem. Directory
    symlinks under ``base`` are followed — serving a directory implies
    trusting what it links to.
    """
    if cap is None:
        cap = _runs_cap()
    found: list[str] = []

    def walk(d: Path, depth: int) -> None:
        if len(found) >= cap:
            return
        if _is_acp_rollout_dir(d):
            found.append(d.relative_to(base).as_posix())
            return  # rollout dirs don't nest
        if depth >= max_depth:
            return
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            return
        for child in children:
            if child.name.startswith("."):
                continue
            walk(child, depth + 1)

    try:
        top = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return found
    for child in top:
        if not child.name.startswith("."):
            walk(child, 1)
    return found


def _resolve_browse_rollout(base: Path, rid: str | None) -> Path | None:
    """Resolve an ``/api/rollout`` id strictly by whitelist membership.

    The id is never interpreted as a path unless a fresh scan discovered it,
    so crafted ids (``../`` traversal, absolute paths) return ``None`` even
    when the traversed-to path exists and is a real rollout.
    """
    if rid is None or rid not in set(_discover_rollouts(base)):
        return None
    return base / rid


def _rollout_summary(base: Path, rel_id: str) -> dict[str, Any]:
    """Sidebar row for one rollout: identity + score at a glance."""
    d = base / rel_id
    result_data = _load_result_json(d)
    rewards = result_data.get("rewards")
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    return {
        "id": rel_id,
        "name": d.name,
        "task_name": result_data.get("task_name") or d.name,
        "agent_name": result_data.get("agent_name") or result_data.get("agent"),
        "model": result_data.get("model"),
        "reward": reward,
        "has_error": bool(
            result_data.get("error") or result_data.get("verifier_error")
        ),
        "skill_mode": result_data.get("skill_mode"),
    }


def _render_acp_trajectory(
    rollout_dir: Path, acp_path: Path, prompts: list[str] | None
) -> str:
    """Render a canonical ACP rollout as a self-contained interactive page.

    Trajectory content is untrusted input: it travels as JSON data (``</``
    escaped so it cannot break out of the script tag) and the template
    renders it exclusively via ``textContent``.
    """
    payload = _build_acp_payload(rollout_dir, prompts)
    return _render_shell(rollout_dir.name, {"mode": "single", "payload": payload})


def _load_result_json(rollout_dir: Path) -> dict:
    parsed = _load_json(rollout_dir / "result.json")
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
        events = _parse_jsonl(path.read_text(encoding="utf-8", errors="replace"))
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
    return _stream_json_page(path.name, events, [body], session_fallback=path.stem)


def _stream_header_badges(
    events: list[dict], *, session_fallback: str | None = None
) -> list[tuple[str, str]]:
    """Header badges derivable from what the stream actually contains.

    ``claude -p`` stream-json carries a ``type: system`` init event
    (``session_id`` / ``model`` / ``claude_code_version``); real ``~/.claude``
    session files don't — their metadata lives per event (``sessionId``,
    ``version``) and on assistant events (``message.model``). Pull from
    whichever is present and omit anything unknown: a header of ``?`` badges
    at the approve moment reads as a broken viewer, not as missing data.
    Truthiness (not ``.get`` defaults) also swallows present-but-null values
    like ``"session_id": null``, which previously needed an explicit guard to
    avoid a TypeError.
    """
    sys_event = next((e for e in events if e.get("type") == "system"), {})

    def _first(values) -> object | None:
        return next((value for value in values if value), None)

    model = sys_event.get("model") or _first(
        event["message"].get("model")
        for event in events
        if event.get("type") == "assistant" and isinstance(event.get("message"), dict)
    )
    session_id = (
        sys_event.get("session_id")
        or _first(event.get("sessionId") for event in events)
        or session_fallback
    )
    version = sys_event.get("claude_code_version") or _first(
        event.get("version") for event in events
    )

    badges: list[tuple[str, str]] = []
    if model:
        badges.append(("model", str(model)))
    if session_id:
        sid = str(session_id)
        badges.append(("session", sid[:16] + ("..." if len(sid) > 16 else "")))
    if version:
        badges.append(("claude code", str(version)))
    return badges


def _result_cost(events: list[dict]) -> float | None:
    """Total cost summed from result events, or ``None`` when no event
    carries cost data — so the header can hide the badge instead of
    asserting a fictional ``$0.0000``."""
    total = 0.0
    seen = False
    for event in events:
        if event.get("type") == "result" and event.get("total_cost_usd") is not None:
            total += float(event.get("total_cost_usd") or 0)
            seen = True
    return total if seen else None


def _meta_badges_html(badges: list[tuple[str, str]]) -> str:
    """The header's ``.meta`` badge row; empty string when nothing is known."""
    if not badges:
        return ""
    spans = "\n".join(
        f"<span>{html.escape(label)}: {html.escape(value)}</span>"
        for label, value in badges
    )
    return f'<div class="meta">\n{spans}\n</div>\n'


def _stream_json_page(
    title: str,
    events: list[dict],
    turn_blocks: list[str],
    *,
    session_fallback: str | None = None,
) -> str:
    badges = _stream_header_badges(events, session_fallback=session_fallback)
    total_cost = _result_cost(events)
    if total_cost is not None:
        badges.append(("total cost", f"${total_cost:.4f}"))
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
{_meta_badges_html(badges)}</div>
{_join_with_divider(turn_blocks)}
</body>
</html>"""


# Sidecar files the viewer reads from a rollout dir; hf:// resolution fetches
# only these (large llm_trajectory.jsonl / trainer exports are skipped).
_HF_VIEWER_FILES = (
    "result.json",
    "timing.json",
    "prompts.json",
    "trajectory/acp_trajectory.jsonl",
    "verifier/*",
)


def _parse_hf_spec(spec: str) -> tuple[str, str | None, str]:
    """Split ``hf://org/name[@revision][/subpath]`` into its parts.

    Tolerates ``hf:/`` (a Path round-trip collapses the double slash) and
    returns ``(repo_id, revision, subpath)``.
    """
    raw = spec.split(":", 1)[1].lstrip("/")
    parts = [p for p in raw.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"invalid HF dataset spec {spec!r} — expected hf://<org>/<name>[/subpath]"
        )
    repo_id = "/".join(parts[:2])
    revision = None
    if "@" in parts[1]:
        name, _, revision = parts[1].partition("@")
        repo_id = f"{parts[0]}/{name}"
    return repo_id, revision, "/".join(parts[2:])


def _resolve_hf_source(spec: str) -> Path:
    """Materialize the viewer-relevant slice of an HF dataset locally.

    Downloads (into the shared huggingface_hub cache, so repeat views are
    incremental) only the files the viewer renders — trajectories plus
    result/timing/prompts/verifier sidecars — and returns the local
    directory to serve.
    """
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        print("huggingface_hub is required for hf:// sources")
        sys.exit(1)

    repo_id, revision, subpath = _parse_hf_spec(spec)
    scope = f"{subpath.rstrip('/')}/" if subpath else ""
    patterns = []
    for name in _HF_VIEWER_FILES:
        patterns.append(f"{scope}{name}")
        patterns.append(f"{scope}**/{name}")
    print(f"Fetching {repo_id}" + (f" @ {revision}" if revision else "") + " …")
    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
    )
    root = Path(local) / subpath if subpath else Path(local)
    if not root.is_dir():
        print(f"No such path in dataset {repo_id}: {subpath}")
        sys.exit(1)
    return root


def serve(
    rollout_path: str,
    port: int = 8888,
    prompts: list[str] | None = None,
    confirm: bool = False,
    redaction_summary: str | None = None,
) -> str | None:
    """Serve a trial directory, a session JSONL file, a directory of runs,
    or an ``hf://`` dataset source.

    ``hf://<org>/<name>[/subpath]`` fetches the viewer-relevant slice of a
    HuggingFace trajectory dataset (e.g. the community ground-truth uploads)
    into the local HF cache and serves it like any local directory.

    Single trajectories (a rollout directory, or a session JSONL file) keep
    the one-page behavior, including the ``confirm=True`` Approve/Reject
    contract documented on :func:`_serve_single`. A directory that is not
    itself a rollout but contains rollout directories is served in browse
    mode instead: a run sidebar plus ``/api/rollout?id=…`` endpoints the page
    uses to load traces dynamically (see :func:`_serve_browse`).

    ``confirm`` needs exactly one trajectory to approve, so combining it with
    a multi-run directory is an error.
    """
    if isinstance(rollout_path, str) and rollout_path.startswith("hf:"):
        rollout_path = str(_resolve_hf_source(rollout_path))
    path = Path(rollout_path)
    if (
        path.is_dir()
        and not _is_acp_rollout_dir(path)
        and not any(path.glob("turn*.txt"))
    ):
        cap = _runs_cap()
        rollouts = _discover_rollouts(path, cap=cap + 1)
        if rollouts:
            capped = len(rollouts) > cap
            n_runs = min(len(rollouts), cap)
            if confirm:
                print(
                    "--confirm needs a single rollout or session file, but "
                    f"{path} is a directory of {n_runs}{'+' if capped else ''} runs"
                )
                sys.exit(1)
            _serve_browse(path, port, n_runs=n_runs, capped=capped)
            return None
    return _serve_single(path, port, prompts, confirm, redaction_summary)


def _serve_single(
    path: Path,
    port: int,
    prompts: list[str] | None,
    confirm: bool,
    redaction_summary: str | None,
) -> str | None:
    """Serve one trial directory or session JSONL file as a web page.

    With ``confirm=True`` the page carries an Approve/Reject bar posting to
    ``/decision``; the first valid decision shuts the server down and is
    returned as ``"approved"`` or ``"rejected"`` after printing a
    machine-readable ``DECISION: <value>`` line to stdout. Without it the
    server runs until Ctrl+C and the return value is ``None`` — exactly the
    pre-confirm behavior (no bar, no POST endpoint).

    ``redaction_summary`` is an optional caller-composed line (e.g. ``"2 API
    keys, 1 bearer token"``) rendered inside the confirm bar so the reviewer
    sees what upload-time redaction would mask. Presentation-only; it has no
    effect without ``confirm=True``.
    """
    import threading
    from http.server import HTTPServer, SimpleHTTPRequestHandler

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
        html_content = _inject_confirm_bar(html_content, redaction_summary)

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


def _serve_browse(base: Path, port: int, n_runs: int, capped: bool = False) -> None:
    """Multi-rollout browser: sidebar shell + JSON API, rescanned per request."""
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from urllib.parse import parse_qs, urlsplit

    class Handler(SimpleHTTPRequestHandler):
        def _send(
            self,
            status: int,
            content_type: str,
            body: bytes,
            headers: tuple[tuple[str, str], ...] = (),
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _scan(self) -> tuple[list[str], bool]:
            """Fresh id list plus whether the cap truncated it."""
            cap = _runs_cap()
            ids = _discover_rollouts(base, cap=cap + 1)
            return ids[:cap], len(ids) > cap

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                ids, capped = self._scan()
                shell = _render_shell(
                    base.name,
                    {
                        "mode": "browse",
                        "capped": capped,
                        "rollouts": [_rollout_summary(base, r) for r in ids],
                    },
                )
                self._send(
                    200,
                    "text/html; charset=utf-8",
                    shell.encode("utf-8", errors="replace"),
                )
            elif parsed.path == "/api/rollouts":
                ids, truncated = self._scan()
                body = _safe_json([_rollout_summary(base, r) for r in ids])
                self._send(
                    200,
                    "application/json; charset=utf-8",
                    body.encode("utf-8", errors="replace"),
                    # API consumers must be able to detect truncation too; the
                    # body stays a bare list for backward compatibility.
                    headers=(("X-BenchFlow-Capped", "1"),) if truncated else (),
                )
            elif parsed.path == "/api/rollout":
                rid = (parse_qs(parsed.query).get("id") or [None])[0]
                rollout_dir = _resolve_browse_rollout(base, rid)
                if rollout_dir is None:
                    self._send(
                        404,
                        "application/json; charset=utf-8",
                        b'{"error": "unknown rollout id"}',
                    )
                    return
                body = _safe_json(_build_acp_payload(rollout_dir, None))
                self._send(
                    200,
                    "application/json; charset=utf-8",
                    body.encode("utf-8", errors="replace"),
                )
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")

        def log_message(self, format, *args):
            pass

    runs_desc = f"{n_runs} runs"
    if capped:
        runs_desc = f"first {n_runs} runs (capped — raise BENCHFLOW_VIEWER_MAX_RUNS)"
    print(f"Trajectory browser: http://localhost:{port}")
    print(f"Scanning: {base} ({runs_desc})")
    print("Press Ctrl+C to stop\n")

    server = HTTPServer(("localhost", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m benchflow.viewer <rollout_dir_or_jsonl> [port]")
        sys.exit(1)
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
    serve(sys.argv[1], port)
