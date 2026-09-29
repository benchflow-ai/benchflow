BF.detail = (() => {
  const {
    EXECUTION_LABELS,
    el,
    fmtDuration,
    fmtTokens,
    plural,
    requirePayload,
    textWithRedaction,
  } = BF.core;

  const COLLAPSE_CHARS = 700;
  const COLLAPSE_LINES = 12;
  const TRACE_CHUNK = 300;
  const PANES = new Map([
    ["trace", "view-trace"],
    ["verifier", "view-verifier"],
    ["metrics", "view-metrics"],
    ["rubric", "view-rubric"],
    ["lineage", "view-lineage"],
  ]);
  const STEP_KINDS = new Set(["prompt", "message", "thought", "tool", "timeout", "unknown", "subagent"]);
  const GROUP_REASONS = new Map([
    ["parent_not_captured", "The tool call these events name as their parent is not in the capture."],
    ["too_deep", "These events are nested deeper than the subagent depth limit shared with the ATIF export."],
    ["cyclic", "The parent_tool_call_id attribution of these events forms a cycle."],
  ]);
  const TOOL_HUES = new Set(["read", "edit", "execute", "fetch", "search", "think", "skill", "other"]);
  const TOOL_STATUSES = new Set(["completed", "failed", "cancelled", "pending", "in_progress", "unknown"]);
  // What the Lineage tab's derived numbers mean (see rollout_branch.py).
  const FORK_VALUE_NOTE = "Value: mean reward of the fork's children, recorded only when every child was scored.";
  const REWARD_SOURCE_NOTES = new Map([
    ["runner_return", "Runner return: the reward the custom child runner returned, not read from the verifier by the branch engine (the runner may still have run the verifier itself)."],
  ]);

  let currentPayload = null;
  let currentOptions = {};
  let timelineBase = null;
  let traceGeneration = 0;
  let disclosureSequence = 0;
  const state = {
    focus: true,
    kinds: new Set(),
    failedOnly: false,
    query: "",
    matches: [],
    matchIndex: -1,
    activeMatch: null,
  };

  function cancel() {
    traceGeneration += 1;
  }

  function resetState() {
    state.focus = true;
    state.kinds.clear();
    state.failedOnly = false;
    state.query = "";
    state.matches = [];
    state.matchIndex = -1;
    state.activeMatch = null;
  }

  function fmtOffset(seconds) {
    const total = Math.max(0, Math.round(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const remainder = total % 60;
    const minuteText = hours ? String(minutes).padStart(2, "0") : String(minutes);
    return "+" + (hours ? hours + ":" + minuteText : minuteText) + ":" + String(remainder).padStart(2, "0");
  }

  function fmtShortDuration(seconds) {
    if (typeof seconds !== "number" || !Number.isFinite(seconds)) return null;
    if (seconds < 1) return "<1s";
    if (seconds < 60) return seconds.toFixed(1).replace(/\.0$/, "") + "s";
    const total = Math.round(seconds);
    return Math.floor(total / 60) + "m " + (total % 60) + "s";
  }

  function asRecord(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  // Visit every step, subagent steps included, depth-first in display order.
  function walkSteps(steps, visit) {
    (Array.isArray(steps) ? steps : []).forEach((step, index) => {
      visit(step, index);
      const trace = asRecord(asRecord(step).subagent);
      if (Array.isArray(trace.steps)) walkSteps(trace.steps, visit);
    });
  }

  function countEvents(steps) {
    let count = 0;
    walkSteps(steps, (step) => { if (normalizedKind(step) !== "subagent") count += 1; });
    return count;
  }

  function withDetail(label, detail) {
    return detail ? label + " (" + detail + ")" : label;
  }

  function statusColor(value) {
    if (/^(completed|complete|scored|restored|available|passed|yes)\b/.test(value)) return "var(--ok-ink)";
    if (/^(errored|timed out|failed|unscored|cancelled|partial|unavailable|no)\b/.test(value)) return "var(--bad-ink)";
    return "";
  }

  function renderHeader(payload) {
    const meta = payload.meta || {};
    const steps = payload.steps || [];
    const header = document.getElementById("hdr");
    header.textContent = "";

    const titleRow = el("div", "titlerow");
    const heading = el("h1", null, meta.task_name || payload.rollout_name || "trajectory");
    heading.tabIndex = -1;
    titleRow.appendChild(heading);
    const reward = meta.reward;
    if (reward === null || reward === undefined) {
      titleRow.appendChild(el("span", "badge unscored", "unscored"));
    } else if (reward >= 1) {
      titleRow.appendChild(el("span", "badge pass", "pass " + reward));
    } else {
      titleRow.appendChild(el("span", "badge fail", "fail " + reward));
    }
    if (meta.partial_trajectory) titleRow.appendChild(el("span", "badge warn", "partial trajectory"));
    header.appendChild(titleRow);

    const subtitle = [];
    if (payload.rollout_name) subtitle.push(payload.rollout_name);
    if (meta.trajectory_source) subtitle.push("source: " + meta.trajectory_source);
    const branch = asRecord(meta.branch);
    if (branch.node_id) {
      subtitle.push(
        "branch child " + branch.node_id + " of fork " + String(branch.fork_id || "").slice(0, 8)
          + (branch.parent_node ? " at node " + branch.parent_node : ""),
      );
    }
    header.appendChild(el("div", "sub", subtitle.join("  \u00b7  ")));

    const stats = el("div");
    stats.id = "stats";
    const identity = el("div", "statrow identity");
    const numbers = el("div", "statrow");
    function tile(row, label, value, mono, modifier) {
      if (value === null || value === undefined || value === "") return;
      const item = el("div", "stat");
      item.appendChild(el("span", "lbl", label));
      item.appendChild(el("span", "val" + (mono ? " mono" : "") + (modifier ? " " + modifier : ""), value));
      row.appendChild(item);
    }

    const usage = meta.usage || {};
    tile(identity, "harness", meta.agent_name);
    tile(identity, "model", meta.model, true);
    tile(
      identity,
      "skills",
      meta.skill_mode,
      false,
      meta.skill_mode ? (String(meta.skill_mode).startsWith("no") ? "skill-off" : "skill-on") : null,
    );
    const status = asRecord(meta.status);
    if (status.execution) {
      tile(identity, "execution", withDetail(EXECUTION_LABELS.get(status.execution) || status.execution, status.execution_detail));
      tile(identity, "assessment", withDetail(status.assessment, status.assessment_detail));
    }
    tile(numbers, "duration", fmtDuration(meta.duration_sec));
    tile(numbers, "tokens in", fmtTokens(usage.n_input_tokens));
    tile(numbers, "tokens out", fmtTokens(usage.n_output_tokens));
    tile(numbers, "cached", fmtTokens(usage.n_cache_read_tokens));
    tile(numbers, "total", fmtTokens(usage.total_tokens));
    if (typeof usage.cost_usd === "number" && Number.isFinite(usage.cost_usd)) {
      tile(numbers, "cost", "$" + usage.cost_usd.toFixed(4));
    }
    tile(numbers, "events", countEvents(steps) || null);
    tile(numbers, "tool calls", (meta.counts || {}).tools);
    tile(numbers, "prompts", (meta.counts || {}).prompts);
    tile(numbers, "usage", usage.usage_source);
    if (identity.children.length) stats.appendChild(identity);
    if (numbers.children.length) stats.appendChild(numbers);
    header.appendChild(stats);

    const errors = Array.isArray(meta.errors) ? meta.errors : [];
    if (errors.length) {
      const box = el("div");
      box.id = "errors";
      errors.forEach((error) => {
        const record = error && typeof error === "object" ? error : {};
        const item = el("div", "errbox" + (record.level === "info" ? " info" : ""));
        item.appendChild(el("span", "elabel", record.label || "error"));
        collapsibleText(item, record.text || "", "etext");
        box.appendChild(item);
      });
      header.appendChild(box);
    }
    return heading;
  }

  function selectPane(id, focusTab) {
    PANES.forEach((paneId, key) => {
      const pane = document.getElementById(paneId);
      const selected = key === id;
      pane.classList.toggle("hidden", !selected);
      pane.setAttribute("aria-hidden", selected ? "false" : "true");
    });
    document.getElementById("toolbar").classList.toggle("hidden", id !== "trace");
    document.querySelectorAll("#tabs [role='tab']").forEach((tab) => {
      const selected = tab.dataset.pane === id;
      tab.setAttribute("aria-selected", selected ? "true" : "false");
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focusTab) tab.focus();
    });
  }

  function renderTabs(payload) {
    const meta = payload.meta || {};
    const verifier = payload.verifier || {};
    const available = [["trace", "Trace"]];
    const status = asRecord(meta.status);
    if (
      verifier.reward !== null && verifier.reward !== undefined
      || verifier.stdout
      || verifier.stderr
      || (Array.isArray(verifier.ctrf) && verifier.ctrf.length)
      || verifier.recovery
      || status.scoring
      || (status.execution === "completed" && status.assessment === "unscored")
    ) {
      available.push(["verifier", "Verifier"]);
    }
    if (
      meta.timing && typeof meta.timing === "object" && Object.keys(meta.timing).length
      || meta.usage && typeof meta.usage === "object" && Object.keys(meta.usage).length
    ) {
      available.push(["metrics", "Metrics"]);
    }
    if (payload.rubric && typeof payload.rubric === "object") {
      available.push(["rubric", "Rubric"]);
    }
    if (payload.lineage && typeof payload.lineage === "object") {
      available.push(["lineage", "Lineage"]);
    }

    const tabs = document.getElementById("tabs");
    tabs.textContent = "";
    tabs.setAttribute("aria-label", "Trajectory sections");
    available.forEach(([id, label], index) => {
      const button = el("button", null, label);
      button.type = "button";
      button.id = "tab-" + id;
      button.dataset.pane = id;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", PANES.get(id));
      button.setAttribute("aria-selected", index === 0 ? "true" : "false");
      button.tabIndex = index === 0 ? 0 : -1;
      button.addEventListener("click", () => selectPane(id, false));
      button.addEventListener("keydown", (event) => {
        const buttons = [...tabs.querySelectorAll("[role='tab']")];
        const current = buttons.indexOf(button);
        let next = null;
        if (event.key === "ArrowRight") next = (current + 1) % buttons.length;
        if (event.key === "ArrowLeft") next = (current - 1 + buttons.length) % buttons.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = buttons.length - 1;
        if (next === null) return;
        event.preventDefault();
        selectPane(buttons[next].dataset.pane, true);
      });
      tabs.appendChild(button);
      const pane = document.getElementById(PANES.get(id));
      pane.setAttribute("aria-labelledby", button.id);
      pane.tabIndex = 0;
    });
    selectPane("trace", false);
  }

  function setFocus(enabled, focusButton, fullButton) {
    state.focus = enabled;
    focusButton.setAttribute("aria-pressed", enabled ? "true" : "false");
    fullButton.setAttribute("aria-pressed", enabled ? "false" : "true");
    if (enabled) {
      document.querySelectorAll("#view-trace .card.k-tool .expander[aria-expanded='true']")
        .forEach((button) => button.click());
    }
    applyCardVisibility();
  }

  function renderToolbar() {
    const toolbar = document.getElementById("toolbar");
    toolbar.textContent = "";

    const modeGroup = el("div", "group seg");
    modeGroup.setAttribute("role", "group");
    modeGroup.setAttribute("aria-label", "Trace detail level");
    const focusButton = el("button", null, "Focus");
    focusButton.type = "button";
    focusButton.id = "btn-focus";
    const fullButton = el("button", null, "Full");
    fullButton.type = "button";
    focusButton.setAttribute("aria-pressed", "true");
    fullButton.setAttribute("aria-pressed", "false");
    focusButton.addEventListener("click", () => setFocus(true, focusButton, fullButton));
    fullButton.addEventListener("click", () => setFocus(false, focusButton, fullButton));
    modeGroup.append(focusButton, fullButton);
    toolbar.appendChild(modeGroup);

    const disclosureGroup = el("div", "group");
    const expand = el("button", null, "Expand all");
    const collapse = el("button", null, "Collapse all");
    expand.type = collapse.type = "button";
    expand.addEventListener("click", () => {
      document.querySelectorAll("#view-trace .expander[aria-expanded='false']").forEach((button) => button.click());
    });
    collapse.addEventListener("click", () => {
      document.querySelectorAll("#view-trace .expander[aria-expanded='true']").forEach((button) => button.click());
    });
    disclosureGroup.append(expand, collapse);
    toolbar.appendChild(disclosureGroup);

    const filterGroup = el("div", "group");
    filterGroup.setAttribute("role", "group");
    filterGroup.setAttribute("aria-label", "Event filters");
    [["prompt", "Prompts"], ["message", "Messages"], ["thought", "Thoughts"], ["tool", "Tools"]]
      .forEach(([kind, label]) => {
        const button = el("button", "chip", label);
        button.type = "button";
        button.setAttribute("aria-pressed", "false");
        button.addEventListener("click", () => {
          if (state.kinds.has(kind)) state.kinds.delete(kind);
          else state.kinds.add(kind);
          button.setAttribute("aria-pressed", state.kinds.has(kind) ? "true" : "false");
          applyCardVisibility();
        });
        filterGroup.appendChild(button);
      });
    const failed = el("button", "chip", "Failed only");
    failed.type = "button";
    failed.setAttribute("aria-pressed", "false");
    failed.addEventListener("click", () => {
      state.failedOnly = !state.failedOnly;
      failed.setAttribute("aria-pressed", state.failedOnly ? "true" : "false");
      applyCardVisibility();
    });
    filterGroup.appendChild(failed);
    toolbar.appendChild(filterGroup);

    const searchGroup = el("div", "group");
    const label = el("label", "visually-hidden", "Search trajectory events");
    label.htmlFor = "search";
    const search = el("input");
    search.id = "search";
    search.type = "search";
    search.placeholder = "search...";
    search.addEventListener("input", () => runSearch(search.value));
    search.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        jumpMatch(event.shiftKey ? -1 : 1);
      }
    });
    const previous = el("button", null, "\u2191");
    const next = el("button", null, "\u2193");
    previous.type = next.type = "button";
    previous.setAttribute("aria-label", "Previous search match");
    next.setAttribute("aria-label", "Next search match");
    previous.addEventListener("click", () => jumpMatch(-1));
    next.addEventListener("click", () => jumpMatch(1));
    const info = el("span");
    info.id = "matchinfo";
    info.setAttribute("aria-live", "polite");
    searchGroup.append(label, search, previous, next, info);
    toolbar.appendChild(searchGroup);

    const progress = el("span");
    progress.id = "progress";
    progress.setAttribute("aria-live", "polite");
    toolbar.appendChild(progress);
  }

  function cardPassesFilters(card) {
    const kind = card.dataset.kind;
    // An unattributed-subagent group is a container: under a filter it shows
    // only through the events inside it.
    if (kind === "subagent") return !state.kinds.size && !state.failedOnly;
    const filterKind = kind === "timeout" || kind === "unknown" ? "tool" : kind;
    if (state.kinds.size && !state.kinds.has(filterKind)) return false;
    if (state.focus && kind === "thought") return false;
    if (state.failedOnly && kind !== "timeout" && !card.classList.contains("failed")) return false;
    return true;
  }

  function applyCardVisibility() {
    // Innermost cards first, so a card that holds a subagent trace stays
    // visible while any event inside it passes the filters.
    [...document.querySelectorAll("#view-trace .card")].reverse().forEach((card) => {
      const isActiveMatch = state.activeMatch !== null && card.id === state.activeMatch;
      let visible = isActiveMatch || cardPassesFilters(card);
      if (!visible && card.dataset.subagent) {
        visible = card.querySelector(".subtrace .card:not(.hidden)") !== null;
      }
      card.classList.toggle("hidden", !visible);
    });
  }

  // Open every collapsed subagent trace that contains the target card.
  function revealCard(target) {
    for (let node = target.parentElement; node && node.id !== "view-trace"; node = node.parentElement) {
      if (!node.classList.contains("subtrace") || !node.classList.contains("hidden")) continue;
      const toggle = node.previousElementSibling;
      if (toggle && toggle.classList.contains("subagent-toggle")) toggle.click();
    }
  }

  function collapsibleText(container, text, className, language = null) {
    const value = String(text);
    const lines = value.split("\n");
    const body = el("div", className);
    if (value.length <= COLLAPSE_CHARS && lines.length <= COLLAPSE_LINES) {
      textWithRedaction(body, value);
      BF.highlight.paint(body, value, language);
      container.appendChild(body);
      return;
    }

    const preview = lines.slice(0, 6).join("\n").slice(0, COLLAPSE_CHARS);
    const bodyId = "disclosure-body-" + (++disclosureSequence);
    body.id = bodyId;
    textWithRedaction(body, preview + " ...");
    container.appendChild(body);
    const button = el("button", "expander", "\u25b8 expand (" + value.length.toLocaleString() + " chars)");
    button.type = "button";
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-controls", bodyId);
    button.addEventListener("click", () => {
      const open = button.getAttribute("aria-expanded") === "true";
      body.textContent = "";
      textWithRedaction(body, open ? preview + " ..." : value);
      if (!open) BF.highlight.paint(body, value, language);
      button.setAttribute("aria-expanded", open ? "false" : "true");
      button.textContent = open
        ? "\u25b8 expand (" + value.length.toLocaleString() + " chars)"
        : "\u25be collapse";
    });
    container.appendChild(button);
  }

  function eventId(step, index) {
    if (step && typeof step === "object" && step.i !== undefined && step.i !== null) return String(step.i);
    return String(index + 1);
  }

  function cardDomId(step, index) {
    if (normalizedKind(step) === "subagent") return String(asRecord(step).gid || "g" + (index + 1));
    return "e" + eventId(step, index);
  }

  function subagentLabel(trace, open) {
    const parts = ["subagent"];
    if (trace.subagent_type) parts[0] += " " + trace.subagent_type;
    if (trace.description) parts.push(trace.description);
    const count = countEvents(trace.steps);
    return (open ? "\u25be " : "\u25b8 ") + parts.join(": ") + " (" + count + " event" + (count === 1 ? "" : "s") + ")";
  }

  // Nested, collapsed-by-default trace of the events a subagent emitted.
  function appendSubagent(card, trace, ownerId) {
    const body = el("div", "subtrace hidden");
    body.id = "subtrace-" + ownerId;
    (Array.isArray(trace.steps) ? trace.steps : []).forEach((step, index) => {
      body.appendChild(buildStepNode(step, index));
    });
    const button = el("button", "expander subagent-toggle", subagentLabel(trace, false));
    button.type = "button";
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-controls", body.id);
    button.addEventListener("click", () => {
      const open = button.getAttribute("aria-expanded") !== "true";
      body.classList.toggle("hidden", !open);
      button.setAttribute("aria-expanded", open ? "true" : "false");
      button.textContent = subagentLabel(trace, open);
    });
    card.dataset.subagent = "1";
    card.append(button, body);
  }

  function buildGroupCard(step, index) {
    const trace = asRecord(step.subagent);
    const card = el("article", "card k-unknown");
    card.id = cardDomId(step, index);
    card.dataset.kind = "subagent";
    const head = el("div", "chead");
    head.appendChild(el("span", "klabel", "unattributed subagent"));
    head.appendChild(el("span", "ttitle", trace.parent_tool_call_id || ""));
    card.appendChild(head);
    card.appendChild(el("div", "body", GROUP_REASONS.get(step.reason) || "These events have no spawning tool call in this trace."));
    appendSubagent(card, trace, card.id);
    return card;
  }

  function buildStepNode(step, index) {
    try {
      return normalizedKind(step) === "subagent" ? buildGroupCard(step, index) : buildCard(step, index);
    } catch (error) {
      const card = el("article", "card k-unknown");
      card.dataset.kind = "unknown";
      card.appendChild(el("span", "klabel", "render error"));
      card.appendChild(el("div", "body", "Could not render event #" + eventId(step, index) + ": " + error));
      return card;
    }
  }

  function normalizedKind(step) {
    const raw = step && typeof step === "object" ? step.kind : null;
    return STEP_KINDS.has(raw) ? raw : "unknown";
  }

  function buildCard(step, index) {
    const record = step && typeof step === "object" ? step : {};
    const kind = normalizedKind(record);
    const id = eventId(record, index);
    const card = el("article", "card k-" + kind);
    card.id = "e" + id;
    card.dataset.kind = kind;
    const head = el("div", "chead");
    const sequence = el("a", "seq", "#" + id);
    sequence.href = "#e" + encodeURIComponent(id);
    head.appendChild(sequence);

    if (typeof record.t === "number" && Number.isFinite(record.t) && timelineBase !== null) {
      head.appendChild(el("span", "tstamp", fmtOffset(record.t - timelineBase)));
    }
    if (kind === "prompt") head.appendChild(el("span", "klabel", record.label || "prompt"));
    else if (kind === "message") head.appendChild(el("span", "klabel", "agent"));
    else if (kind === "thought") head.appendChild(el("span", "klabel", "thought"));
    else if (kind === "timeout") head.appendChild(el("span", "klabel", "agent timeout"));
    else if (kind === "unknown") head.appendChild(el("span", "klabel", record.type || record.kind || "unknown"));

    let toolHue = null;
    if (kind === "tool") {
      const tool = record.tool && typeof record.tool === "object" ? record.tool : {};
      toolHue = TOOL_HUES.has(tool.hue) ? tool.hue : "other";
      const rawStatus = typeof tool.status === "string" ? tool.status : "unknown";
      const statusClass = TOOL_STATUSES.has(rawStatus) ? rawStatus : "unknown";
      card.classList.add("tb-" + toolHue);
      head.appendChild(el("span", "kindbadge", tool.kind || "tool"));
      head.appendChild(el("span", "ttitle", tool.title || ""));
      head.appendChild(el("span", "status " + statusClass, rawStatus));
      const duration = fmtShortDuration(record.dur);
      if (duration) head.appendChild(el("span", "tstamp", duration));
      if (statusClass === "failed" || statusClass === "cancelled") card.classList.add("failed");
    }
    card.appendChild(head);

    if (kind === "tool") {
      const tool = record.tool && typeof record.tool === "object" ? record.tool : {};
      const outputClass = "tout" + (toolHue === "execute" ? " term" : "");
      const content = Array.isArray(tool.content) ? tool.content : [];
      content.forEach((item, position) => {
        const wrapper = el("div");
        const language = BF.highlight.languageFor(String(item), toolHue, tool.title, position);
        collapsibleText(wrapper, item, outputClass, language);
        card.appendChild(wrapper);
      });
      if (record.subagent && typeof record.subagent === "object") appendSubagent(card, record.subagent, card.id);
    } else if (kind === "timeout") {
      const timeout = record.timeout && typeof record.timeout === "object" ? record.timeout : {};
      const pending = Array.isArray(timeout.pending) ? timeout.pending : [];
      card.appendChild(el(
        "div",
        "body",
        "Hit wall-clock timeout after " + (timeout.timeout_sec ?? "?") + "s (" + (timeout.reason || "timeout") + "). "
          + (pending.length ? "Interrupted tool calls: " + pending.join(", ") : "No tool calls were pending."),
      ));
    } else if (record.text) {
      collapsibleText(card, record.text, "body");
    }
    return card;
  }

  function renderTrace(payload) {
    const steps = payload.steps || [];
    timelineBase = null;
    for (const step of steps) {
      if (step && typeof step.t === "number" && Number.isFinite(step.t)) {
        timelineBase = step.t;
        break;
      }
    }

    const main = document.getElementById("view-trace");
    main.textContent = "";
    if (!steps.length) {
      const empty = el("div", null, "No events in this trajectory.");
      empty.id = "empty";
      main.appendChild(empty);
      return;
    }

    const generation = traceGeneration;
    const progress = document.getElementById("progress");
    let index = 0;
    function pump() {
      if (generation !== traceGeneration) return;
      const fragment = document.createDocumentFragment();
      for (let count = 0; count < TRACE_CHUNK && index < steps.length; count += 1, index += 1) {
        fragment.appendChild(buildStepNode(steps[index], index));
      }
      main.appendChild(fragment);
      applyCardVisibility();
      if (index < steps.length) {
        if (progress) progress.textContent = "rendering " + index + " / " + steps.length + " ...";
        requestAnimationFrame(pump);
        return;
      }
      if (progress) progress.textContent = "";
      let hashId = null;
      if (location.hash.startsWith("#e")) {
        try { hashId = decodeURIComponent(location.hash.slice(1)); }
        catch { hashId = location.hash.slice(1); }
      }
      const target = state.activeMatch ? document.getElementById(state.activeMatch) : (hashId ? document.getElementById(hashId) : null);
      if (target) {
        revealCard(target);
        target.scrollIntoView({ block: "center" });
        target.classList.add("flash");
      }
    }
    pump();
  }

  function stepSearchText(step) {
    if (!step || typeof step !== "object") return "";
    let value = (step.text || "") + " " + (step.label || "");
    if (step.tool && typeof step.tool === "object") {
      const content = Array.isArray(step.tool.content) ? step.tool.content : [];
      value += " " + (step.tool.title || "") + " " + (step.tool.kind || "") + " " + content.join(" ");
    }
    const trace = asRecord(step.subagent);
    if (trace.parent_tool_call_id) {
      value += " subagent " + (trace.subagent_type || "") + " " + (trace.description || "")
        + (normalizedKind(step) === "subagent" ? " unattributed " + trace.parent_tool_call_id : "");
    }
    return value.toLowerCase();
  }

  function runSearch(query) {
    state.query = query.trim().toLowerCase();
    state.matches = [];
    state.matchIndex = -1;
    state.activeMatch = null;
    const info = document.getElementById("matchinfo");
    if (!state.query) {
      if (info) info.textContent = "";
      applyCardVisibility();
      return;
    }
    walkSteps((currentPayload && currentPayload.steps) || [], (step, index) => {
      if (stepSearchText(step).includes(state.query)) state.matches.push(cardDomId(step, index));
    });
    if (info) info.textContent = state.matches.length + " match" + (state.matches.length === 1 ? "" : "es");
    if (state.matches.length) jumpMatch(1);
    else applyCardVisibility();
  }

  function jumpMatch(direction) {
    if (!state.matches.length) return;
    state.matchIndex = (state.matchIndex + direction + state.matches.length) % state.matches.length;
    state.activeMatch = state.matches[state.matchIndex];
    applyCardVisibility();
    const target = document.getElementById(state.activeMatch);
    if (target) {
      revealCard(target);
      target.scrollIntoView({ block: "center" });
      target.classList.remove("flash");
      void target.offsetWidth;
      target.classList.add("flash");
    }
    const info = document.getElementById("matchinfo");
    if (info) info.textContent = (state.matchIndex + 1) + " / " + state.matches.length;
  }

  function factsTable(rows) {
    const table = el("table", "plain");
    rows.forEach(([label, value, color]) => {
      const row = el("tr");
      row.appendChild(el("th", null, label));
      const cell = el("td", null, value);
      if (color) cell.style.color = color;
      row.appendChild(cell);
      table.appendChild(row);
    });
    return table;
  }

  // A heading is a label or [label, hover text]; a cell is a value, a Node,
  // or [value, color, hover text].
  function gridTable(headings, rows) {
    const table = el("table", "plain");
    const heading = el("tr");
    headings.forEach((entry) => {
      const [label, hint] = Array.isArray(entry) ? entry : [entry, ""];
      const th = el("th", null, label);
      if (hint) th.title = hint;
      heading.appendChild(th);
    });
    table.appendChild(heading);
    rows.forEach((cells) => {
      const row = el("tr");
      cells.forEach((cell) => {
        if (cell instanceof Node) {
          const td = el("td");
          td.appendChild(cell);
          row.appendChild(td);
          return;
        }
        const [value, color, hint] = Array.isArray(cell) ? cell : [cell, "", ""];
        const td = el("td", null, value);
        if (color) td.style.color = color;
        if (hint) td.title = hint;
        row.appendChild(td);
      });
      table.appendChild(row);
    });
    return table;
  }

  function scoringText(scoring) {
    if (scoring.status !== "complete") return "error: " + (scoring.error || "no explanation");
    const blockers = scoring.all_blockers_pass
      ? "all blockers pass"
      : "failed blockers: " + (Array.isArray(scoring.failed_blockers) ? scoring.failed_blockers.join(", ") : "?");
    return [
      scoring.passed ? "complete, passed" : "complete, not passed",
      scoring.tests_pass ? "tests pass" : "tests fail",
      blockers,
      "verifier reward " + scoring.verifier_reward,
      "rubric reward " + scoring.rubric_reward,
    ].join(" \u00b7 ");
  }

  // Execution vs assessment plus verifier-recovery evidence, one row each.
  function assessmentRows(payload) {
    const status = asRecord(asRecord(payload.meta).status);
    const recovery = asRecord(asRecord(payload.verifier).recovery);
    const rows = [];
    if (status.execution) {
      const execution = withDetail(EXECUTION_LABELS.get(status.execution) || status.execution, status.execution_detail);
      const assessment = withDetail(status.assessment, status.assessment_detail);
      rows.push(["Execution", execution, statusColor(execution)]);
      rows.push(["Assessment", assessment, statusColor(assessment)]);
    }
    const scoring = asRecord(status.scoring);
    if (scoring.status) rows.push(["Scoring (tests + rubric)", scoringText(scoring), statusColor(scoring.status)]);
    if (recovery.pointer) rows.push(["Verification pointer", "verification.json \u2192 " + recovery.pointer]);
    else if (recovery.pointer_error) rows.push(["Verification pointer", "verification.json " + recovery.pointer_error, "var(--bad-ink)"]);
    const attempts = Array.isArray(recovery.attempts) ? recovery.attempts.map(asRecord) : [];
    const admitted = attempts.find((attempt) => attempt.admitted);
    const shown = admitted || attempts[attempts.length - 1];
    if (shown) {
      const state = shown.status || "receipt not available";
      rows.push([admitted ? "Recovery attempt" : "Recovery attempt (not admitted)", state + " \u00b7 " + shown.attempt, statusColor(state)]);
      if (shown.original_error) rows.push(["Original verifier error", shown.original_error]);
      if (shown.error) rows.push(["Recovery error", shown.error, "var(--bad-ink)"]);
      if (shown.reward !== null && shown.reward !== undefined) rows.push(["Recovered reward", String(shown.reward)]);
      const verifierTime = fmtShortDuration(shown.verifier_sec);
      const evidence = [
        shown.evidence,
        shown.solver_replayed === false ? "solver not replayed" : (shown.solver_replayed ? "solver replayed" : null),
        verifierTime ? "verifier " + verifierTime : null,
      ].filter(Boolean);
      if (evidence.length) rows.push(["Recovery evidence", evidence.join(" \u00b7 ")]);
      if (shown.publication_error) {
        rows.push([
          "Publication error",
          shown.publication_error + " \u2014 verifier/ still holds the outputs from before recovery; the recovered outputs are in "
            + shown.attempt + "/verifier/.",
          "var(--bad-ink)",
        ]);
      }
      if (shown.cleanup_error) rows.push(["Recovery cleanup error", shown.cleanup_error]);
      if (shown.admission) rows.push(["Admission", shown.admission]);
    }
    const others = attempts.filter((attempt) => attempt !== shown);
    if (others.length) {
      rows.push(["Other recovery attempts", others.map((attempt) => (attempt.status || "no receipt") + " (" + attempt.attempt + ")").join(", ")]);
    }
    return rows;
  }

  function renderVerifier(payload) {
    const verifier = payload.verifier || {};
    const main = document.getElementById("view-verifier");
    main.textContent = "";
    if (verifier.reward !== null && verifier.reward !== undefined) {
      const block = el("div", "panelblock");
      block.appendChild(el("h2", null, "Reward"));
      const number = Number(verifier.reward);
      const reward = el("div", "bigreward", verifier.reward);
      reward.style.color = number >= 1 ? "var(--ok-ink)" : "var(--bad-ink)";
      block.appendChild(reward);
      main.appendChild(block);
    }
    const assessment = assessmentRows(payload);
    if (assessment.length) {
      const block = el("div", "panelblock");
      block.appendChild(el("h2", null, "Assessment"));
      block.appendChild(factsTable(assessment));
      main.appendChild(block);
    }
    if (Array.isArray(verifier.ctrf) && verifier.ctrf.length) {
      const block = el("div", "panelblock");
      block.appendChild(el("h2", null, "Tests (CTRF)"));
      const table = el("table", "plain");
      const heading = el("tr");
      ["Test", "Status", "Duration"].forEach((label) => heading.appendChild(el("th", null, label)));
      table.appendChild(heading);
      verifier.ctrf.forEach((test) => {
        const record = test && typeof test === "object" ? test : {};
        const row = el("tr");
        row.appendChild(el("td", null, record.name));
        const status = el("td", null, record.status);
        status.style.color = record.status === "passed"
          ? "var(--ok-ink)"
          : (record.status === "failed" ? "var(--bad-ink)" : "var(--muted)");
        row.appendChild(status);
        row.appendChild(el("td", null, record.duration !== undefined && record.duration !== null ? record.duration + " ms" : ""));
        table.appendChild(row);
      });
      block.appendChild(table);
      main.appendChild(block);
    }
    [["stdout", "Verifier stdout"], ["stderr", "Verifier stderr"]].forEach(([key, title]) => {
      if (!verifier[key]) return;
      const block = el("div", "panelblock");
      block.appendChild(el("h2", null, title));
      collapsibleText(block, verifier[key], "tout");
      main.appendChild(block);
    });
    if (!main.children.length) main.appendChild(el("div", "panelblock", "No verifier artifacts in this rollout."));
  }

  function renderRubric(payload) {
    const rubric = payload.rubric && typeof payload.rubric === "object" ? payload.rubric : null;
    const main = document.getElementById("view-rubric");
    main.textContent = "";
    if (!rubric) {
      main.appendChild(el("div", "panelblock", "No review report found for this rollout."));
      return;
    }
    const scoring = rubric.scoring && typeof rubric.scoring === "object" ? rubric.scoring : {};
    const head = el("div", "panelblock");
    head.appendChild(el("h2", null, "Review"));
    const facts = el("table", "plain");
    const rows = [
      ["Reviewer", rubric.reviewer_model || ""],
      ["Valid", rubric.review_valid ? "yes" : "no"],
      ["Verifier pass", scoring.deterministic_pass === undefined ? "" : (scoring.deterministic_pass ? "yes" : "no")],
      ["Blockers", scoring.all_blockers_pass === undefined ? ""
        : (scoring.all_blockers_pass ? "all pass" : "failed: " + (scoring.failed_blockers || []).join(", "))],
      ["Raw quality", scoring.raw_quality === undefined ? "" : String(scoring.raw_quality)],
      ["Gated quality", scoring.gated_quality === undefined ? "" : String(scoring.gated_quality)],
      ["Decision", scoring.decision || ""],
    ];
    rows.forEach(([label, value]) => {
      const row = el("tr");
      row.appendChild(el("th", null, label));
      const cell = el("td", null, value);
      if (label === "Blockers" || label === "Verifier pass") {
        cell.style.color = /fail|^no$/.test(value) ? "var(--bad-ink)" : (value ? "var(--ok-ink)" : "");
      }
      row.appendChild(cell);
      facts.appendChild(row);
    });
    head.appendChild(facts);
    if (rubric.summary) collapsibleText(head, rubric.summary, "body");
    main.appendChild(head);

    const criteria = Array.isArray(rubric.criteria) ? rubric.criteria : [];
    if (criteria.length) {
      const block = el("div", "panelblock");
      block.appendChild(el("h2", null, "Criteria"));
      const table = el("table", "plain");
      const heading = el("tr");
      ["Criterion", "Kind", "Verdict", "Reviewer note"].forEach((label) => heading.appendChild(el("th", null, label)));
      table.appendChild(heading);
      criteria.forEach((criterion) => {
        const record = criterion && typeof criterion === "object" ? criterion : {};
        const row = el("tr");
        row.appendChild(el("td", null, record.name));
        // blocker true: v0.2 blocker; false: v0.2 scored; null: legacy v0.1 (outcome only)
        const scored = record.blocker === false;
        const kind = record.blocker === true ? "blocker" : (scored ? "weight " + (record.weight ?? "?") : "criterion");
        row.appendChild(el("td", null, kind));
        const verdict = el("td", null, scored ? ((record.score ?? "?") + " / 2") : (record.outcome || ""));
        if (!scored && record.outcome) {
          verdict.style.color = record.outcome === "pass" ? "var(--ok-ink)"
            : (record.outcome === "fail" ? "var(--bad-ink)" : "var(--muted-foreground)");
        }
        row.appendChild(verdict);
        row.appendChild(el("td", null, record.explanation || ""));
        table.appendChild(row);
      });
      block.appendChild(table);
      main.appendChild(block);
    }
  }

  function branchLink(ref, label) {
    const link = el("a", null, label);
    link.href = currentOptions.branchHref ? currentOptions.branchHref(ref) : "?branch=" + encodeURIComponent(ref);
    if (currentOptions.openBranch) {
      link.addEventListener("click", (event) => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        currentOptions.openBranch(ref);
      });
    }
    return link;
  }

  function renderLineage(payload) {
    const main = document.getElementById("view-lineage");
    main.textContent = "";
    if (!payload.lineage || typeof payload.lineage !== "object") return;
    const lineage = payload.lineage;
    if (lineage.error) {
      main.appendChild(el("div", "panelblock", "tree.json could not be read: " + lineage.error));
      return;
    }
    const forks = Array.isArray(lineage.forks) ? lineage.forks.map(asRecord) : [];
    if (!forks.length) {
      main.appendChild(el("div", "panelblock", "tree.json records no forks."));
      return;
    }
    const overview = el("div", "panelblock");
    overview.appendChild(el("h2", null, "Forks (" + plural(lineage.nodes || 0, "tree node") + ")"));
    overview.appendChild(gridTable(
      ["Fork", "Parent \u2192 children", "Status", ["Value", FORK_VALUE_NOTE], "Snapshot", "Parent restore"],
      forks.map((fork) => {
        const children = Array.isArray(fork.children) ? fork.children : [];
        const captured = Array.isArray(fork.captured_layers) ? fork.captured_layers : [];
        const requested = Array.isArray(fork.requested_layers) ? fork.requested_layers : [];
        const status = [fork.status || "", fork.error, fork.artifact_error ? "artifact: " + fork.artifact_error : null]
          .filter(Boolean).join(" \u2014 ");
        const restore = [fork.parent_restore || "", fork.parent_restore_error].filter(Boolean).join(" \u2014 ");
        return [
          String(fork.id || "").slice(0, 8),
          (fork.parent_node || "?") + " \u2192 " + children.length + (fork.requested_children ? " of " + fork.requested_children : ""),
          [status, statusColor(status)],
          fork.value === null || fork.value === undefined ? "\u2014" : String(fork.value),
          (captured.length ? captured.join(", ") : "none captured")
            + (requested.join(",") !== captured.join(",") ? " (requested " + requested.join(", ") + ")" : ""),
          [restore, statusColor(restore)],
        ];
      }),
    ));
    overview.appendChild(el("p", "note", FORK_VALUE_NOTE));
    main.appendChild(overview);

    forks.forEach((fork) => {
      const children = Array.isArray(fork.children) ? fork.children.map(asRecord) : [];
      const block = el("div", "panelblock");
      block.appendChild(el("h2", null, "Fork " + String(fork.id || "").slice(0, 8) + ": children of " + (fork.parent_node || "?")));
      const sourceNotes = new Set();
      block.appendChild(gridTable(
        ["Child", "Status", "Reward", "Intervention", "Trajectory"],
        children.map((child) => {
          const node = child.node_id || "#" + child.index;
          const status = [child.status || "", child.error, child.cleanup_error ? "cleanup: " + child.cleanup_error : null]
            .filter(Boolean).join(" \u2014 ");
          const hasReward = typeof child.reward === "number";
          const source = child.reward_source && child.reward_source !== "verifier" ? " (" + child.reward_source.replace(/_/g, " ") + ")" : "";
          const sourceNote = hasReward ? REWARD_SOURCE_NOTES.get(child.reward_source) || "" : "";
          if (sourceNote) sourceNotes.add(sourceNote);
          const intervention = asRecord(child.intervention);
          const parts = [
            intervention.label,
            intervention.requested ? "requested: " + intervention.requested : null,
            intervention.execution && intervention.execution !== "unspecified" ? "execution: " + intervention.execution : null,
            intervention.evidence ? "evidence: " + intervention.evidence : null,
          ].filter(Boolean);
          const artifacts = [child.artifacts_status || "", child.artifacts_path].filter(Boolean).join(" ");
          return [
            node,
            [status, statusColor(status)],
            [
              hasReward ? child.reward + source : "\u2014",
              hasReward ? (child.reward >= 1 ? "var(--ok-ink)" : "var(--bad-ink)") : "",
              sourceNote,
            ],
            parts.length ? parts.join(" \u00b7 ") : "\u2014",
            child.ref ? branchLink(child.ref, "open " + node) : (artifacts || "\u2014"),
          ];
        }),
      ));
      sourceNotes.forEach((note) => block.appendChild(el("p", "note", note)));
      main.appendChild(block);
    });
  }

  function renderMetrics(payload) {
    const meta = payload.meta || {};
    const main = document.getElementById("view-metrics");
    main.textContent = "";
    const timing = meta.timing && typeof meta.timing === "object" ? meta.timing : {};
    const phases = ["environment_setup", "agent_setup", "agent_execution", "verifier"]
      .filter((key) => typeof timing[key] === "number" && Number.isFinite(timing[key]));
    if (phases.length) {
      const block = el("div", "panelblock");
      block.appendChild(el("h2", null, "Phase timing"));
      const maximum = Math.max(...phases.map((key) => timing[key]), 1);
      const colors = new Map([
        ["environment_setup", "var(--chart-2)"],
        ["agent_setup", "var(--chart-3)"],
        ["agent_execution", "var(--chart-1)"],
        ["verifier", "var(--chart-5)"],
      ]);
      phases.forEach((key) => {
        const row = el("div", "barrow");
        row.appendChild(el("span", null, key.replace(/_/g, " ")));
        const track = el("div");
        const bar = el("div", "bar");
        bar.style.width = Math.max(2, (timing[key] / maximum) * 100) + "%";
        bar.style.background = colors.get(key) || "var(--rule-strong)";
        track.appendChild(bar);
        row.appendChild(track);
        row.appendChild(el("span", "num", timing[key].toFixed(1) + " s"));
        block.appendChild(row);
      });
      if (typeof timing.total === "number" && Number.isFinite(timing.total)) {
        const row = el("div", "barrow");
        row.appendChild(el("span", null, "total"));
        row.appendChild(el("div"));
        row.appendChild(el("span", "num", timing.total.toFixed(1) + " s"));
        block.appendChild(row);
      }
      main.appendChild(block);
    }

    const usage = meta.usage && typeof meta.usage === "object" ? meta.usage : {};
    const usageRows = [
      ["input tokens", usage.n_input_tokens],
      ["output tokens", usage.n_output_tokens],
      ["cache read", usage.n_cache_read_tokens],
      ["cache write", usage.n_cache_creation_tokens],
      ["thought tokens", (usage.usage_details || {}).thought_tokens],
      ["total", usage.total_tokens],
    ].filter(([, value]) => value !== null && value !== undefined);
    if (usageRows.length) {
      const block = el("div", "panelblock");
      block.appendChild(el("h2", null, "Token usage" + (usage.usage_source ? " - " + usage.usage_source : "")));
      const table = el("table", "plain");
      usageRows.forEach(([label, value]) => {
        const row = el("tr");
        row.appendChild(el("td", null, label));
        row.appendChild(el("td", null, Number(value).toLocaleString()));
        table.appendChild(row);
      });
      block.appendChild(table);
      main.appendChild(block);
    }

    const counts = meta.counts && typeof meta.counts === "object" ? meta.counts : {};
    const countRows = Object.entries(counts).filter(([, value]) => value !== null && value !== undefined);
    if (countRows.length) {
      const block = el("div", "panelblock");
      block.appendChild(el("h2", null, "Event counts"));
      const table = el("table", "plain");
      countRows.forEach(([label, value]) => {
        const row = el("tr");
        row.appendChild(el("td", null, label.replace(/_/g, " ")));
        row.appendChild(el("td", null, value));
        table.appendChild(row);
      });
      block.appendChild(table);
      main.appendChild(block);
    }
    if (!main.children.length) main.appendChild(el("div", "panelblock", "No metrics in this rollout."));
  }

  function clearDetail() {
    ["hdr", "tabs", "view-trace", "view-verifier", "view-metrics", "view-rubric", "view-lineage"].forEach((id) => {
      document.getElementById(id).textContent = "";
    });
    document.getElementById("toolbar").textContent = "";
    document.getElementById("toolbar").classList.add("hidden");
    selectPane("trace", false);
  }

  function showMessage(label, message, isError) {
    cancel();
    currentPayload = null;
    currentOptions = {};
    clearDetail();
    const main = document.getElementById("view-trace");
    const box = el("div", isError ? "errbox" : "panelblock");
    if (isError) box.appendChild(el("span", "elabel", label));
    box.appendChild(el("span", isError ? "etext" : null, message));
    main.appendChild(box);
  }

  function showLoading(runId) {
    showMessage("", "Loading " + runId + "...", false);
    document.title = "loading - benchflow trajectory";
  }

  function showError(message, label = "load error") {
    showMessage(label, String(message), true);
    document.title = "load error - benchflow trajectory";
  }

  function loadPayload(value, options = {}) {
    const payload = requirePayload(value);
    cancel();
    resetState();
    currentPayload = payload;
    currentOptions = options;
    const heading = renderHeader(payload);
    renderTabs(payload);
    renderToolbar();
    renderTrace(payload);
    renderVerifier(payload);
    renderMetrics(payload);
    renderRubric(payload);
    renderLineage(payload);
    document.title = (payload.meta && payload.meta.task_name) || payload.rollout_name || "benchflow trajectory";
    if (options.focusHeading) requestAnimationFrame(() => heading.focus());
  }

  return { cancel, loadPayload, showError, showLoading };
})();
