/*
 * Pure payload -> HTML renderer for one BenchFlow run.
 *
 * Emits the DOM contract of the vendored PostTrainBench stylesheet
 * (.topbar / .layout / .rail / .summary-card / .event / .tool-call / …) so
 * BenchFlow traces read as the same product, then extends it where BenchFlow
 * carries data PostTrainBench has no concept of. Those additions are all
 * prefixed `bf-` and styled in benchflow.css, so a re-vendor of upstream's
 * stylesheet stays a straight overwrite.
 *
 * Why BenchFlow keeps its own renderer instead of vendoring upstream's
 * run.js: that file dispatches on Claude Code stream-json record types
 * (`assistant` / `user` / `system` / `result`) and hardcodes Claude Code tool
 * names (Bash, Edit, Read, TodoWrite), with a second branch bolted on for
 * Codex. BenchFlow normalizes 28 harnesses into ACP *once*, upstream of the
 * viewer — so rendering here is driven by ACP's kind/status, and reward,
 * verifier, skill mode, oracle and timeout events survive to the page.
 *
 * Every function is a string transform: no DOM, no globals, no fetch. boot.js
 * owns the DOM, and the published static site runs this same file against a
 * payload fetched from HuggingFace.
 *
 * Deliberately conservative JS (no optional chaining, no flat/flatMap) so the
 * node-based renderer test and old browsers agree with modern ones.
 */
var BFViewer = (function () {
  "use strict";

  var SCHEMA_VERSION = 1;
  var BINARY_PLACEHOLDER = "[binary output omitted]";

  // Upstream's inline icons, so BenchFlow's blocks carry the same marks.
  var ICON = {
    thought:
      '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" ' +
      'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
      'stroke-linejoin="round"><path d="M9 18h6"/><path d="M10 22h4"/>' +
      '<path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V18h6v-1.2c0-.8.4-1.6 ' +
      '1-2.1A7 7 0 0 0 12 2z"/></svg>',
    tool:
      '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" ' +
      'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
      'stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/>' +
      '<line x1="12" y1="19" x2="20" y2="19"/></svg>',
    output:
      '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" ' +
      'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
      'stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>'
  };

  // ── escaping ────────────────────────────────────────────────────────

  function esc(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // ── formatting ──────────────────────────────────────────────────────

  function isNumber(value) {
    return typeof value === "number" && isFinite(value);
  }

  function formatCount(value) {
    if (!isNumber(value)) return "—";
    return String(Math.round(value)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function formatDuration(value) {
    if (!isNumber(value)) return "—";
    if (value < 60) return value.toFixed(1) + "s";
    if (value < 3600) return (value / 60).toFixed(1) + "m";
    return (value / 3600).toFixed(2) + "h";
  }

  function formatCost(value) {
    if (!isNumber(value)) return "—";
    return "$" + value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  }

  function formatReward(value) {
    if (!isNumber(value)) return "—";
    return String(Math.round(value * 10000) / 10000);
  }

  function sourceLabel(source) {
    return source === "stream-json" ? "stream-json fallback" : "ACP";
  }

  // ── terminal output: ANSI SGR -> spans ──────────────────────────────

  var ANSI_COLORS = {
    30: "black", 31: "red", 32: "green", 33: "yellow",
    34: "blue", 35: "magenta", 36: "cyan", 37: "white",
    90: "bright-black", 91: "red", 92: "green", 93: "yellow",
    94: "blue", 95: "magenta", 96: "cyan", 97: "white"
  };

  /* Render terminal text: SGR colour/bold survive as spans, every other
   * control sequence is dropped rather than shown as mojibake. */
  function renderTerminal(text) {
    var out = "";
    var open = 0;
    var pattern = /\x1b\[([0-9;]*)m|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-9;?]*[A-Za-z]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g;
    var last = 0;
    var match;
    while ((match = pattern.exec(text)) !== null) {
      out += esc(text.slice(last, match.index));
      last = match.index + match[0].length;
      if (match[1] === undefined) continue; // non-SGR control: drop it
      var codes = match[1] === "" ? ["0"] : match[1].split(";");
      for (var i = 0; i < codes.length; i++) {
        var code = parseInt(codes[i], 10);
        if (!isFinite(code) || code === 0) {
          while (open > 0) { out += "</span>"; open--; }
        } else if (code === 1) {
          out += '<span class="ansi-bold">'; open++;
        } else if (code === 2) {
          out += '<span class="ansi-dim">'; open++;
        } else if (ANSI_COLORS[code]) {
          out += '<span class="ansi-' + ANSI_COLORS[code] + '">'; open++;
        }
      }
    }
    out += esc(text.slice(last));
    while (open > 0) { out += "</span>"; open--; }
    return out;
  }

  // ── markdown (small, safe subset) ───────────────────────────────────

  function inlineMarkdown(text) {
    var out = esc(text);
    out = out.replace(/`([^`]+)`/g, function (_m, code) {
      return "<code>" + code + "</code>";
    });
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/(^|[\s(])_([^_]+)_(?=$|[\s).,!?])/g, "$1<em>$2</em>");
    // Links are limited to http(s) so a trace can never inject javascript:.
    out = out.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" rel="noreferrer noopener" target="_blank">$1</a>'
    );
    return out;
  }

  /* Agent and user messages are markdown in practice; rendering them as
   * plain text is the single biggest quality gap against ptb. */
  function renderMarkdown(text) {
    var lines = String(text === null || text === undefined ? "" : text).split("\n");
    var out = [];
    var list = null;
    var paragraph = [];
    var fence = null;
    var code = [];

    function closeParagraph() {
      if (!paragraph.length) return;
      out.push("<p>" + inlineMarkdown(paragraph.join("\n")) + "</p>");
      paragraph = [];
    }
    function closeList() {
      if (!list) return;
      out.push("</" + list + ">");
      list = null;
    }
    function openList(tag) {
      if (list === tag) return;
      closeList();
      out.push("<" + tag + ">");
      list = tag;
    }

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      var fenceMatch = /^\s*```(\S*)\s*$/.exec(line);
      if (fence !== null) {
        if (fenceMatch) {
          out.push("<pre><code>" + esc(code.join("\n")) + "</code></pre>");
          fence = null;
          code = [];
        } else {
          code.push(line);
        }
        continue;
      }
      if (fenceMatch) {
        closeParagraph();
        closeList();
        fence = fenceMatch[1] || "";
        continue;
      }
      var heading = /^(#{1,6})\s+(.*)$/.exec(line);
      if (heading) {
        closeParagraph();
        closeList();
        var level = Math.min(6, heading[1].length + 2);
        out.push("<h" + level + ">" + inlineMarkdown(heading[2]) + "</h" + level + ">");
        continue;
      }
      var bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
      if (bullet) {
        closeParagraph();
        openList("ul");
        out.push("<li>" + inlineMarkdown(bullet[1]) + "</li>");
        continue;
      }
      var ordered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
      if (ordered) {
        closeParagraph();
        openList("ol");
        out.push("<li>" + inlineMarkdown(ordered[1]) + "</li>");
        continue;
      }
      var quote = /^\s*>\s?(.*)$/.exec(line);
      if (quote) {
        closeParagraph();
        closeList();
        out.push("<blockquote>" + inlineMarkdown(quote[1]) + "</blockquote>");
        continue;
      }
      if (!line.trim()) {
        closeParagraph();
        closeList();
        continue;
      }
      closeList();
      paragraph.push(line);
    }
    if (fence !== null) {
      out.push("<pre><code>" + esc(code.join("\n")) + "</code></pre>");
    }
    closeParagraph();
    closeList();
    return out.join("");
  }

  // ── tool observation blocks ─────────────────────────────────────────

  function renderClip(clip) {
    return (
      '<span class="bf-clip">… ' + formatCount(clip.dropped) +
      " characters truncated — full output in " + esc(clip.artifact) + "</span>"
    );
  }

  function renderTextBlock(block) {
    var text = block.text || "";
    if (!block.clip) return renderTerminal(text);
    var at = block.clip.at;
    return (
      renderTerminal(text.slice(0, at)) + "\n" + renderClip(block.clip) + "\n" +
      renderTerminal(text.slice(at))
    );
  }

  /* Upstream renders a diff only for Claude Code's Edit tool, off its
   * old_string/new_string input. ACP hands over a diff block directly, so
   * every harness's edits land in the same upstream .diff-remove/.diff-add
   * treatment. */
  function renderDiffBlock(block) {
    var out = block.path ? "<div>" + esc(block.path) + "</div>" : "";
    if (block.old) out += '<pre class="diff-remove">- ' + esc(block.old) + "</pre>";
    if (block.new) out += '<pre class="diff-add">+ ' + esc(block.new) + "</pre>";
    return out;
  }

  function renderBlockBody(block) {
    if (block.kind === "diff") return renderDiffBlock(block);
    if (block.kind === "binary") {
      return '<span class="bf-binary">' + BINARY_PLACEHOLDER + "</span>";
    }
    return renderTextBlock(block);
  }

  /* The observation goes in upstream's .tool-result / .tool-result-body
   * shell: that wrapper is what carries the 280px cap the "expand outputs"
   * control lifts, so an ACP observation clamps exactly like a Claude Code
   * tool result. */
  function renderObservation(blocks, status) {
    if (!blocks || !blocks.length) return "";
    var body = "";
    for (var i = 0; i < blocks.length; i++) {
      if (i > 0) body += "\n";
      body += renderBlockBody(blocks[i]);
    }
    var head = '<span class="tool-result-label">' + ICON.output + " Output</span>";
    var clipped = false;
    for (var j = 0; j < blocks.length; j++) if (blocks[j].clip) clipped = true;
    if (status === "failed") head += '<span class="chip bad">error</span>';
    if (clipped) head += '<span class="muted">truncated</span>';
    return (
      '<div class="tool-result"><div class="tool-result-head">' + head +
      '</div><div class="tool-result-body' +
      (status === "failed" ? " error" : "") + '">' + body + "</div></div>"
    );
  }

  // ── events ──────────────────────────────────────────────────────────

  function toolStatus(status) {
    if (!status) return "";
    return (
      '<span class="bf-tool-status bf-tool-status-' +
      esc(String(status).toLowerCase()) + '">' + esc(status) + "</span>"
    );
  }

  function renderToolCall(event) {
    var kind = event.kind || "tool";
    var command = event.title || kind;
    // The command line gets upstream's Bash treatment for every ACP tool:
    // `title` is already the most useful bounded rendering of the call.
    var args = '<div class="bash-cmd">' + esc(command) + "</div>";
    var id = event.tool_call_id
      ? '<span class="tool-id">' + esc(String(event.tool_call_id).slice(-8)) + "</span>"
      : "";
    return (
      '<details class="tool-call tool-bash" open><summary>' + ICON.tool +
      ' <span class="block-label-inline">Tool</span> <span class="tool-name">' +
      esc(kind) + "</span>" + id + toolStatus(event.status) + "</summary>" +
      '<div class="tool-args">' + args + "</div>" +
      renderObservation(event.blocks, event.status) + "</details>"
    );
  }

  function eventParts(event, turnNumber) {
    // Returns [markerHtml, bodyHtml, extraClass]. The turn number rides on
    // the user message that opens the turn, so #turn-N anchors land there.
    if (event.type === "user_message") {
      return [
        '<div class="turn-num">' + esc(turnNumber) + "</div>",
        '<div class="block-card agent-text">' + renderMarkdown(event.text) + "</div>",
        "role-user"
      ];
    }
    if (event.type === "agent_thought") {
      /* Some harnesses signal "the agent is thinking" but send no text —
       * @agentclientprotocol/claude-agent-acp emits agent_thought_chunk with
       * content.text === "". The event is still evidence that reasoning
       * happened here, so it is marked rather than dropped or rendered as an
       * empty expandable card. */
      if (!event.text) {
        return [
          '<div class="turn-role">think</div>',
          '<div class="bf-thought-empty">' + ICON.thought +
          " thought — the harness sent no text</div>",
          "bf-event-thought"
        ];
      }
      return [
        '<div class="turn-role">think</div>',
        '<details class="block-card agent-thinking" open><summary>' +
        ICON.thought + " <span>Thought</span></summary>" +
        '<div class="thinking-body">' + renderMarkdown(event.text) + "</div></details>",
        "bf-event-thought"
      ];
    }
    if (event.type === "agent_message") {
      return [
        '<div class="turn-role">agent</div>',
        '<div class="block-card agent-text">' + renderMarkdown(event.text) + "</div>",
        ""
      ];
    }
    if (event.type === "tool_call") {
      return ['<div class="turn-role">tool</div>', renderToolCall(event), ""];
    }
    if (event.type === "oracle") {
      return [
        '<div class="turn-role">oracle</div>',
        renderToolCall({
          kind: "oracle",
          title: event.title,
          status: event.status,
          blocks: event.blocks
        }),
        "bf-event-oracle role-system"
      ];
    }
    if (event.type === "agent_timeout") {
      var pending = event.pending_tool_call_ids;
      return [
        '<div class="turn-role bf-role-timeout">timeout</div>',
        '<div class="bf-timeout-card"><strong>Agent timed out</strong>' +
        "<div>" + esc(event.text) + "</div>" +
        '<dl class="bf-timeout-meta"><dt>Budget</dt><dd>' +
        esc(formatDuration(event.timeout_sec)) + "</dd><dt>Pending tools</dt><dd>" +
        esc(pending && pending.length ? pending.join(", ") : "none") +
        "</dd></dl></div>",
        "bf-event-timeout"
      ];
    }
    return ["", "", ""];
  }

  function renderEvent(event, turnNumber, anchor) {
    var parts = eventParts(event, turnNumber);
    if (!parts[1]) return "";
    // Node colour cycles per turn, repurposing upstream's session palette so
    // consecutive turns stay visually separable.
    var session = "session-" + (isNumber(turnNumber) ? turnNumber % 5 : 0);
    var cls = "event " + session + (parts[2] ? " " + parts[2] : "");
    var id = anchor ? ' id="' + anchor + '"' : "";
    var anchorAttr = anchor ? ' data-anchor="' + anchor + '"' : "";
    return (
      "<article" + id + ' class="' + cls + '">' +
      '<aside class="event-marker"' + anchorAttr + ">" + parts[0] + "</aside>" +
      '<div class="event-body">' + parts[1] + "</div></article>"
    );
  }

  function renderTurn(turn) {
    var isSetup = turn.number === null || turn.number === undefined;
    var out = isSetup ? '<div class="bf-setup-divider">Setup</div>' : "";
    for (var i = 0; i < turn.events.length; i++) {
      // Only the turn's first event carries the #turn-N anchor.
      var anchor = "";
      if (isSetup && i === 0) anchor = "setup";
      else if (!isSetup && i === 0) anchor = "turn-" + turn.number;
      out += renderEvent(turn.events[i], turn.number, anchor);
    }
    return out;
  }

  // ── summary rail ────────────────────────────────────────────────────

  function statRow(label, value) {
    if (value === null || value === undefined || value === "") return "";
    return "<dt>" + esc(label) + "</dt><dd>" + esc(value) + "</dd>";
  }

  function statusChip(status) {
    return (
      '<span class="bf-status bf-status-' + esc(status.slug) + '">' +
      esc(status.label) + "</span>"
    );
  }

  function renderTokens(usage) {
    var fields = [
      ["Input", usage.input], ["Output", usage.output],
      ["Cache write", usage.cache_creation], ["Cache read", usage.cache_read]
    ];
    var present = false;
    var rows = "";
    for (var i = 0; i < fields.length; i++) {
      if (isNumber(fields[i][1])) present = true;
      rows += statRow(fields[i][0], formatCount(fields[i][1]));
    }
    if (!present) return "";
    // The four rows above are billed separately but must reconcile to
    // total_tokens (this is why _usage_payload reads agent_result and not
    // final_metrics, which has no cache-creation field). Showing the sum makes
    // a capture bug visible instead of asking the reader to add four numbers.
    rows += statRow("Total", isNumber(usage.total) ? formatCount(usage.total) : null);
    // Naming the source is the difference between a measurement and a guess.
    var note = usage.source
      ? '<p class="bf-usage-note">usage: ' + esc(usage.source) +
        (usage.price_source ? " · price: " + esc(usage.price_source) : "") + "</p>"
      : "";
    return (
      '<div class="summary-tokens-block"><span class="summary-label">Token usage' +
      '</span><dl class="summary-tokens summary-stats">' + rows + "</dl>" + note +
      "</div>"
    );
  }

  function renderTiming(timing) {
    var phases = [
      ["Environment", "environment_setup"], ["Agent setup", "agent_setup"],
      ["Agent", "agent_execution"], ["Verifier", "verifier"]
    ];
    var present = [];
    for (var i = 0; i < phases.length; i++) {
      if (isNumber(timing[phases[i][1]])) {
        present.push([phases[i][0], timing[phases[i][1]]]);
      }
    }
    if (!present.length) return "";
    var denominator = isNumber(timing.total) && timing.total > 0 ? timing.total : 0;
    if (!denominator) {
      for (var j = 0; j < present.length; j++) denominator += present[j][1];
    }
    // The phases are instrumented one by one while `total` is the rollout's own
    // wall clock, so they do not tile it: agent install, task-file upload and
    // teardown all sit inside `total` and inside no phase. Naming the remainder
    // keeps the bars summing to the total instead of leaving a third of the
    // track blank and unexplained. Sub-0.1s remainders are rounding, not a
    // phase — dropping them avoids a permanent 1.5%-floor sliver.
    if (isNumber(timing.total)) {
      var covered = 0;
      for (var n = 0; n < present.length; n++) covered += present[n][1];
      var other = timing.total - covered;
      if (other > 0.05) present.push(["Other", other]);
    }
    var rows = "";
    for (var k = 0; k < present.length; k++) {
      var width = denominator > 0 ? (present[k][1] / denominator) * 100 : 0;
      rows +=
        '<div class="bf-timing-row"><div class="bf-timing-label"><span>' +
        esc(present[k][0]) + "</span><span>" + esc(formatDuration(present[k][1])) +
        '</span></div><div class="bf-timing-track"><span style="width:' +
        Math.max(1.5, width).toFixed(2) + '%"></span></div></div>';
    }
    var total = isNumber(timing.total)
      ? '<div class="bf-timing-total"><span>Total</span><span>' +
        esc(formatDuration(timing.total)) + "</span></div>"
      : "";
    return (
      '<section class="bf-timing"><span class="summary-label">Timing</span>' +
      rows + total + "</section>"
    );
  }

  function renderNotices(notices) {
    var out = "";
    for (var i = 0; i < notices.length; i++) {
      out +=
        '<div class="bf-notice bf-notice-' + esc(notices[i].level) + '"><strong>' +
        esc(notices[i].title) + "</strong><span>" + esc(notices[i].body) +
        "</span></div>";
    }
    return out;
  }

  function renderRail(payload) {
    var meta = payload.meta;
    var scored = isNumber(payload.reward);
    var percent = scored ? Math.max(0, Math.min(100, payload.reward * 100)) : 0;
    var score = scored
      ? '<div class="score-big">' + esc(formatReward(payload.reward)) +
        '<span class="bf-score-unit">reward</span></div>' +
        '<div class="score-bar"><div class="score-bar-fill" style="width:' +
        percent + '%"></div></div>'
      : '<div class="score-big bf-unscored">Not scored</div>';

    var stats =
      statRow("Agent", meta.agent_name || meta.agent) +
      statRow("Harness", meta.agent) +
      statRow("Model", meta.model) +
      statRow("Skill mode", meta.skill_mode) +
      statRow("Tool calls", formatCount(meta.n_tool_calls)) +
      (isNumber(meta.n_skill_invocations)
        ? statRow("Skill invocations", formatCount(meta.n_skill_invocations))
        : "") +
      statRow("Trace", sourceLabel(payload.source)) +
      statRow("Cost", formatCost(payload.usage.cost_usd));

    var download = payload.artifacts && payload.artifacts.trajectory
      ? '<a id="link-raw" href="#" class="btn btn-secondary btn-small">' +
        "Download trace</a>"
      : "";

    return (
      '<aside class="rail rail-left"><div class="card summary-card">' +
      '<div class="summary-meta"><h2 class="summary-title">' +
      esc(meta.task_name || payload.name) + "</h2>" +
      '<div class="summary-sub">' + esc(meta.model || meta.agent_name || "") +
      "</div></div>" +
      '<div class="summary-score">' + score +
      '<div class="score-sub bf-score-status">' + statusChip(payload.status) +
      "</div></div>" +
      '<div class="summary-details"><dl class="summary-stats">' + stats + "</dl>" +
      renderTokens(payload.usage) + renderTiming(payload.timing) +
      renderNotices(payload.notices) +
      '<div class="summary-footer">' + download +
      '<div class="run-id-block" id="run-id-box" title="Click to copy the run ID">' +
      '<code id="run-id-text">' + esc(payload.name) + "</code>" +
      '<button class="copy-btn" id="copy-id-btn" aria-label="Copy run ID">⧉</button>' +
      '<span id="copy-feedback" class="copy-feedback">copied</span>' +
      "</div></div></div></div></aside>"
    );
  }

  // ── page ────────────────────────────────────────────────────────────

  function renderRunHtml(payload) {
    if (!payload || typeof payload !== "object") {
      return '<div class="layout"><main class="content"><h1>No run data</h1></main></div>';
    }
    if (payload.schema_version !== SCHEMA_VERSION) {
      return (
        '<div class="layout"><main class="content"><h1>Unsupported trace</h1>' +
        "<p>This page renders payload schema " + SCHEMA_VERSION +
        ", but the data is schema " + esc(payload.schema_version) +
        ". Regenerate it with a matching <code>bench eval view</code>.</p>" +
        "</main></div>"
      );
    }

    var trace = "";
    var turns = 0;
    var events = 0;
    for (var i = 0; i < payload.turns.length; i++) {
      trace += renderTurn(payload.turns[i]);
      events += payload.turns[i].events.length;
      if (payload.turns[i].number !== null && payload.turns[i].number !== undefined) {
        turns++;
      }
    }
    if (!trace) {
      trace =
        '<div class="bf-empty-trace"><strong>No captured events</strong><br>' +
        "The rollout has result metadata but its trajectory is empty.</div>";
    }

    var toolbar =
      '<header class="trace-toolbar"><div class="trace-toolbar-summary">' +
      '<span id="event-count" class="trace-toolbar-count">' + turns +
      " turns · " + events + " events</span></div>" +
      '<div class="trace-toolbar-controls">' +
      '<label class="trace-jump" title="Jump to a turn number (1-based)">' +
      '<input type="number" id="jump-turn" min="1" inputmode="numeric" ' +
      'placeholder="turn #" aria-label="Jump to turn number" /></label>' +
      '<label class="toggle-label"><input type="checkbox" id="expand-outputs" />' +
      " expand outputs</label></div></header>";

    return (
      '<header class="topbar"><div class="topbar-inner">' +
      '<a href="#" class="logo">Bench<span class="logo-accent">Flow</span></a>' +
      '<span class="logo-sub">/ traces</span>' +
      '<div class="topbar-meta" id="topbar-meta">' + statusChip(payload.status) +
      "</div>" +
      '<button id="theme-toggle" class="theme-toggle topbar-action" ' +
      'aria-label="Toggle theme">◐</button></div></header>' +
      '<div class="layout bf-no-right-rail" id="run-layout" data-active-tab="trace">' +
      renderRail(payload) +
      '<main class="content"><nav class="tab-nav" id="tab-nav" role="tablist">' +
      '<button class="tab-btn active" data-tab="trace" role="tab">' +
      "<span>Run trace</span></button></nav>" +
      '<section id="section-trace" class="section active">' + toolbar +
      '<div id="trace">' + trace + "</div>" +
      "</section></main></div>"
    );
  }

  return {
    SCHEMA_VERSION: SCHEMA_VERSION,
    renderRunHtml: renderRunHtml,
    renderMarkdown: renderMarkdown,
    renderTerminal: renderTerminal,
    formatDuration: formatDuration,
    formatCount: formatCount
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = BFViewer;
