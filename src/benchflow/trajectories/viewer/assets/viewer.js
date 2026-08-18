"use strict";
/* Boot data: {"mode":"single","payload":{...}} — embedded, offline-capable —
   or {"mode":"browse","rollouts":[...]} — payloads fetched per run from
   /api/rollout?id=… . All dynamic content flows through textContent. */
const BOOT = JSON.parse(document.getElementById("bf-payload").textContent);
let CUR = null;      /* current payload */
let T_BASE = null;   /* epoch of the first timestamped step (timeline origin) */
let renderGen = 0;   /* bumped per load; cancels stale chunked renders */
const state = { focus: false, kinds: new Set(), failedOnly: false, matches: [], matchIdx: -1 };
const COLLAPSE_CHARS = 700, COLLAPSE_LINES = 12;
const PANES = { trace: "view-trace", verifier: "view-verifier", metrics: "view-metrics" };

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}
/* Render text with ***REDACTED*** tokens highlighted, safely (DOM split, no HTML). */
function textWithRedaction(target, text) {
  const parts = String(text).split("***REDACTED***");
  parts.forEach((p, i) => {
    if (i > 0) target.appendChild(el("span", "redacted", "REDACTED"));
    if (p) target.appendChild(document.createTextNode(p));
  });
}
function fmtTokens(n) {
  if (n === null || n === undefined) return null;
  return n >= 10000 ? (n / 1000).toFixed(1) + "k" : String(n);
}
function fmtDuration(sec) {
  if (sec === null || sec === undefined || typeof sec !== "number") return null;
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return m ? m + "m " + s + "s" : s + "s";
}
/* timeline offset "+m:ss" (hours appear as needed) from the first
   timestamped step; renders only when the capture carries timestamps */
function fmtOffset(sec) {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return "+" + (h ? h + ":" + mm : mm) + ":" + String(s).padStart(2, "0");
}
function fmtDurShort(sec) {
  if (typeof sec !== "number") return null;
  if (sec < 1) return "<1s";
  if (sec < 60) return sec.toFixed(1).replace(/\.0$/, "") + "s";
  return Math.floor(sec / 60) + "m " + Math.round(sec % 60) + "s";
}

/* ── header ─────────────────────────────────────────────── */
function renderHeader(p) {
  const META = p.meta || {}, STEPS = p.steps || [];
  const h = document.getElementById("hdr");
  h.textContent = "";
  const tr = el("div", "titlerow");
  tr.appendChild(el("h1", null, META.task_name || p.rollout_name || "trajectory"));
  const r = META.reward;
  if (r === null || r === undefined) tr.appendChild(el("span", "badge unscored", "unscored"));
  else if (r >= 1) tr.appendChild(el("span", "badge pass", "pass " + r));
  else tr.appendChild(el("span", "badge fail", "fail " + r));
  if (META.partial_trajectory) tr.appendChild(el("span", "badge warn", "partial trajectory"));
  h.appendChild(tr);
  const subBits = [];
  if (p.rollout_name) subBits.push(p.rollout_name);
  if (META.trajectory_source) subBits.push("source: " + META.trajectory_source);
  h.appendChild(el("div", "sub", subBits.join("  ·  ")));

  /* Stat band, two rows: identity first (harness / model / skills — the
     primary review dimensions), numbers second. */
  const stats = el("div", null); stats.id = "stats";
  const idRow = el("div", "statrow identity");
  const numRow = el("div", "statrow");
  function tile(row, label, value, mono, extraCls) {
    if (value === null || value === undefined || value === "") return;
    const s = el("div", "stat");
    s.appendChild(el("span", "lbl", label));
    s.appendChild(el("span", "val" + (mono ? " mono" : "") + (extraCls ? " " + extraCls : ""), value));
    row.appendChild(s);
  }
  const u = META.usage || {};
  tile(idRow, "harness", META.agent_name);
  tile(idRow, "model", META.model, true);
  tile(idRow, "skills", META.skill_mode, false,
    META.skill_mode ? (String(META.skill_mode).startsWith("no") ? "skill-off" : "skill-on") : null);
  tile(numRow, "duration", fmtDuration(META.duration_sec));
  tile(numRow, "tokens in", fmtTokens(u.n_input_tokens));
  tile(numRow, "tokens out", fmtTokens(u.n_output_tokens));
  tile(numRow, "cached", fmtTokens(u.n_cache_read_tokens));
  tile(numRow, "total", fmtTokens(u.total_tokens));
  if (u.cost_usd !== null && u.cost_usd !== undefined) tile(numRow, "cost", "$" + Number(u.cost_usd).toFixed(4));
  tile(numRow, "events", STEPS.length || null);
  tile(numRow, "tool calls", (META.counts || {}).tools);
  tile(numRow, "prompts", (META.counts || {}).prompts);
  tile(numRow, "usage", u.usage_source);
  if (idRow.children.length) stats.appendChild(idRow);
  if (numRow.children.length) stats.appendChild(numRow);
  h.appendChild(stats);

  const errs = META.errors || [];
  if (errs.length) {
    const box = el("div", null); box.id = "errors";
    errs.forEach(e => {
      const b = el("div", "errbox" + (e.level === "info" ? " info" : ""));
      b.appendChild(el("span", "elabel", e.label));
      b.appendChild(el("span", "etext", e.text));
      box.appendChild(b);
    });
    h.appendChild(box);
  }
}

/* ── tabs ───────────────────────────────────────────────── */
function selectPane(id) {
  Object.entries(PANES).forEach(([k, pid]) => document.getElementById(pid).classList.toggle("hidden", k !== id));
  document.getElementById("toolbar").classList.toggle("hidden", id !== "trace");
}
function renderTabs(p) {
  const META = p.meta || {};
  const tabs = document.getElementById("tabs");
  tabs.textContent = "";
  const avail = [["trace", "Trace"]];
  const v = p.verifier || {};
  if (v.reward !== null && v.reward !== undefined || v.stdout || v.ctrf) avail.push(["verifier", "Verifier"]);
  if ((META.timing && Object.keys(META.timing).length) || Object.keys(META.usage || {}).length) avail.push(["metrics", "Metrics"]);
  avail.forEach(([id, label], i) => {
    const b = el("button", null, label);
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", i === 0 ? "true" : "false");
    b.addEventListener("click", () => {
      tabs.querySelectorAll("button").forEach(x => x.setAttribute("aria-selected", "false"));
      b.setAttribute("aria-selected", "true");
      selectPane(id);
    });
    tabs.appendChild(b);
  });
  selectPane("trace");
}

/* ── toolbar ────────────────────────────────────────────── */
function renderToolbar(p) {
  const t = document.getElementById("toolbar");
  t.textContent = "";
  const g1 = el("div", "group seg");
  const focusBtn = el("button", null, "Focus"); focusBtn.id = "btn-focus";
  const fullBtn = el("button", null, "Full");
  fullBtn.setAttribute("aria-pressed", state.focus ? "false" : "true");
  focusBtn.setAttribute("aria-pressed", state.focus ? "true" : "false");
  focusBtn.addEventListener("click", () => setFocus(true, focusBtn, fullBtn));
  fullBtn.addEventListener("click", () => setFocus(false, focusBtn, fullBtn));
  g1.appendChild(focusBtn); g1.appendChild(fullBtn);
  t.appendChild(g1);
  const g2 = el("div", "group");
  const ex = el("button", null, "Expand all");
  const co = el("button", null, "Collapse all");
  ex.addEventListener("click", () => document.querySelectorAll("#view-trace .expander[data-open='0']").forEach(b => b.click()));
  co.addEventListener("click", () => document.querySelectorAll("#view-trace .expander[data-open='1']").forEach(b => b.click()));
  g2.appendChild(ex); g2.appendChild(co);
  t.appendChild(g2);
  const g3 = el("div", "group");
  [["prompt", "Prompts"], ["message", "Messages"], ["thought", "Thoughts"], ["tool", "Tools"]].forEach(([k, label]) => {
    const c = el("button", "chip", label);
    c.setAttribute("aria-pressed", state.kinds.has(k) ? "true" : "false");
    c.addEventListener("click", () => {
      if (state.kinds.has(k)) state.kinds.delete(k); else state.kinds.add(k);
      c.setAttribute("aria-pressed", state.kinds.has(k) ? "true" : "false");
      applyFilters();
    });
    g3.appendChild(c);
  });
  const failed = el("button", "chip", "Failed only");
  failed.setAttribute("aria-pressed", state.failedOnly ? "true" : "false");
  failed.addEventListener("click", () => {
    state.failedOnly = !state.failedOnly;
    failed.setAttribute("aria-pressed", state.failedOnly ? "true" : "false");
    applyFilters();
  });
  g3.appendChild(failed);
  t.appendChild(g3);
  const g4 = el("div", "group");
  const s = el("input"); s.id = "search"; s.type = "search"; s.placeholder = "search…";
  s.addEventListener("input", () => runSearch(s.value));
  s.addEventListener("keydown", e => { if (e.key === "Enter") jumpMatch(e.shiftKey ? -1 : 1); });
  const info = el("span", null, ""); info.id = "matchinfo";
  const prev = el("button", null, "↑"), next = el("button", null, "↓");
  prev.title = "previous match"; next.title = "next match";
  prev.addEventListener("click", () => jumpMatch(-1));
  next.addEventListener("click", () => jumpMatch(1));
  g4.appendChild(s); g4.appendChild(prev); g4.appendChild(next); g4.appendChild(info);
  t.appendChild(g4);
  const prog = el("span", null, ""); prog.id = "progress";
  t.appendChild(prog);
}

function setFocus(on, focusBtn, fullBtn) {
  state.focus = on;
  focusBtn.setAttribute("aria-pressed", on ? "true" : "false");
  fullBtn.setAttribute("aria-pressed", on ? "false" : "true");
  if (on) {
    /* collapse any open tool outputs; thoughts are hidden by the filter below */
    document.querySelectorAll("#view-trace .card.k-tool .expander[data-open='1']").forEach(b => b.click());
  }
  applyFilters();
}
function applyFilters() {
  document.querySelectorAll("#view-trace .card").forEach(c => {
    const k = c.dataset.kind;
    /* timeout/unknown cards ride with the Tools chip */
    const effective = (k === "timeout" || k === "unknown") ? "tool" : k;
    let hide = state.kinds.size > 0 && !state.kinds.has(effective);
    if (state.focus && k === "thought") hide = true;
    if (state.failedOnly && !(k === "timeout" || c.classList.contains("failed"))) hide = true;
    c.classList.toggle("hidden", hide);
  });
}

/* ── collapsible body helper ────────────────────────────── */
function collapsibleText(container, text, cls) {
  const s = String(text);
  const lines = s.split("\n");
  const long = s.length > COLLAPSE_CHARS || lines.length > COLLAPSE_LINES;
  const body = el("div", cls);
  if (!long) { textWithRedaction(body, s); container.appendChild(body); return; }
  const preview = lines.slice(0, 6).join("\n").slice(0, COLLAPSE_CHARS);
  textWithRedaction(body, preview + " …");
  container.appendChild(body);
  const btn = el("button", "expander", "▸ expand (" + s.length.toLocaleString() + " chars)");
  btn.dataset.open = "0";
  btn.addEventListener("click", () => {
    const open = btn.dataset.open === "1";
    body.textContent = "";
    textWithRedaction(body, open ? preview + " …" : s);
    btn.dataset.open = open ? "0" : "1";
    btn.textContent = open ? "▸ expand (" + s.length.toLocaleString() + " chars)" : "▾ collapse";
  });
  container.appendChild(btn);
}

/* ── trace stream (chunked render) ──────────────────────── */
function buildCard(step) {
  const card = el("article", "card k-" + step.kind);
  card.id = "e" + step.i;
  card.dataset.kind = step.kind;
  const head = el("div", "chead");
  const seq = el("a", "seq", "#" + step.i);
  seq.href = "#e" + step.i;
  head.appendChild(seq);
  if (typeof step.t === "number" && T_BASE !== null) {
    head.appendChild(el("span", "tstamp", fmtOffset(step.t - T_BASE)));
  }
  if (step.kind === "prompt") head.appendChild(el("span", "klabel", step.label || "prompt"));
  else if (step.kind === "message") head.appendChild(el("span", "klabel", "agent"));
  else if (step.kind === "thought") head.appendChild(el("span", "klabel", "thought"));
  else if (step.kind === "timeout") head.appendChild(el("span", "klabel", "agent timeout"));
  else if (step.kind === "unknown") head.appendChild(el("span", "klabel", step.type || "unknown"));
  let toolKind = null;
  if (step.kind === "tool") {
    const t = step.tool || {};
    /* class names only from this fixed set — kind is untrusted input */
    const KINDS = ["read", "edit", "execute", "fetch", "search", "think", "skill"];
    let kk = KINDS.includes(t.kind) ? t.kind : "other";
    if (kk === "other") {
      /* ACP kinds are coarse ("other" covers Skill/Task/..., and kinds like
         "delete"/"move" fall outside KINDS) — infer the hue from kind+title,
         mirroring upstream's _tool_accent_class fallback chain. */
      const hay = (String(t.kind || "") + " " + String(t.title || "")).toLowerCase();
      const INFER = [
        ["search", ["web", "search", "fetch", "grep", "glob", "browser"]],
        ["execute", ["bash", "shell", "exec", "terminal", "command"]],
        ["edit", ["write", "edit", "patch", "delete", "move", "notebook"]],
        ["read", ["read", "cat", "view", "ls", "list"]],
        ["skill", ["agent", "task", "skill", "oracle"]],
      ];
      for (const [hue, needles] of INFER) {
        if (needles.some(n => hay.includes(n))) { kk = hue; break; }
      }
    }
    toolKind = kk;
    card.classList.add("tb-" + kk);
    head.appendChild(el("span", "kindbadge", t.kind || "tool"));
    head.appendChild(el("span", "ttitle", t.title || ""));
    head.appendChild(el("span", "status " + (t.status || ""), t.status || "?"));
    const durTxt = fmtDurShort(step.dur);
    if (durTxt) head.appendChild(el("span", "tstamp", durTxt));
    if (t.status === "failed" || t.status === "cancelled") card.classList.add("failed");
  }
  card.appendChild(head);
  if (step.kind === "tool") {
    /* dark terminal treatment for shell output only, like the site renderer */
    const outCls = "tout" + (toolKind === "execute" ? " term" : "");
    ((step.tool || {}).content || []).forEach(c => {
      const wrap = el("div", null);
      collapsibleText(wrap, c, outCls);
      card.appendChild(wrap);
    });
  } else if (step.kind === "timeout") {
    const d = step.timeout || {};
    const pend = Array.isArray(d.pending) ? d.pending : [];
    const b = el("div", "body",
      "Hit wall-clock timeout after " + (d.timeout_sec ?? "?") + "s (" + (d.reason || "timeout") + "). " +
      (pend.length ? "Interrupted tool calls: " + pend.join(", ") : "No tool calls were pending."));
    card.appendChild(b);
  } else if (step.text) {
    collapsibleText(card, step.text, "body");
  }
  return card;
}

function renderTrace(p) {
  const STEPS = p.steps || [];
  T_BASE = null;
  for (const s of STEPS) {
    if (typeof s.t === "number") { T_BASE = s.t; break; }
  }
  const main = document.getElementById("view-trace");
  main.textContent = "";
  if (!STEPS.length) {
    const e = el("div", null, "No events in this trajectory."); e.id = "empty";
    main.appendChild(e);
    return;
  }
  const CHUNK = 300;
  const gen = renderGen;
  let i = 0;
  const prog = document.getElementById("progress");
  function pump() {
    if (gen !== renderGen) return; /* a newer run was loaded; stop this render */
    const frag = document.createDocumentFragment();
    for (let n = 0; n < CHUNK && i < STEPS.length; n++, i++) {
      /* One malformed step degrades to a single error card — it must never
         halt the loop and blank the rest of the trace. */
      try {
        frag.appendChild(buildCard(STEPS[i]));
      } catch (err) {
        const c = el("article", "card k-unknown");
        c.dataset.kind = "unknown";
        c.appendChild(el("span", "klabel", "render error"));
        c.appendChild(el("div", "body", "Could not render event #" + (STEPS[i] && STEPS[i].i) + ": " + err));
        frag.appendChild(c);
      }
    }
    main.appendChild(frag);
    if (i < STEPS.length) {
      if (prog) prog.textContent = "rendering " + i + " / " + STEPS.length + " …";
      requestAnimationFrame(pump);
    } else {
      if (prog) prog.textContent = "";
      applyFilters();
      if (location.hash && location.hash.startsWith("#e")) {
        const target = document.getElementById(location.hash.slice(1));
        if (target) { target.scrollIntoView(); target.classList.add("flash"); }
      }
    }
  }
  pump();
  if (STEPS.length > 2000 && !state.focus) {
    const fb = document.getElementById("btn-focus");
    if (fb) fb.click(); /* auto-focus mode on big trajectories */
  }
}

/* ── search ─────────────────────────────────────────────── */
function stepSearchText(step) {
  let s = (step.text || "") + " " + (step.label || "");
  if (step.tool) s += " " + (step.tool.title || "") + " " + (step.tool.kind || "") + " " + (step.tool.content || []).join(" ");
  return s.toLowerCase();
}
function runSearch(q) {
  q = q.trim().toLowerCase();
  state.matches = []; state.matchIdx = -1;
  const info = document.getElementById("matchinfo");
  if (!q) { if (info) info.textContent = ""; return; }
  ((CUR && CUR.steps) || []).forEach(step => { if (stepSearchText(step).includes(q)) state.matches.push(step.i); });
  if (info) info.textContent = state.matches.length + " match" + (state.matches.length === 1 ? "" : "es");
  if (state.matches.length) jumpMatch(1);
}
function jumpMatch(dir) {
  if (!state.matches.length) return;
  state.matchIdx = (state.matchIdx + dir + state.matches.length) % state.matches.length;
  const id = "e" + state.matches[state.matchIdx];
  const c = document.getElementById(id);
  if (!c) return;
  c.classList.remove("hidden");
  c.scrollIntoView({ block: "center" });
  c.classList.remove("flash"); void c.offsetWidth; c.classList.add("flash");
  const info = document.getElementById("matchinfo");
  if (info) info.textContent = (state.matchIdx + 1) + " / " + state.matches.length;
}

/* ── verifier tab ───────────────────────────────────────── */
function renderVerifier(p) {
  const v = p.verifier || {};
  const main = document.getElementById("view-verifier");
  main.textContent = "";
  if (v.reward !== null && v.reward !== undefined) {
    const b = el("div", "panelblock");
    b.appendChild(el("h2", null, "Reward"));
    const n = Number(v.reward);
    const big = el("div", "bigreward", v.reward);
    big.style.color = n >= 1 ? "var(--ok-ink)" : "var(--bad-ink)";
    b.appendChild(big);
    main.appendChild(b);
  }
  if (v.ctrf && v.ctrf.length) {
    const b = el("div", "panelblock");
    b.appendChild(el("h2", null, "Tests (CTRF)"));
    const tbl = el("table", "plain");
    const hr = el("tr"); ["Test", "Status", "Duration"].forEach(x => hr.appendChild(el("th", null, x)));
    tbl.appendChild(hr);
    v.ctrf.forEach(t => {
      const r = el("tr");
      r.appendChild(el("td", null, t.name));
      const st = el("td", null, t.status);
      st.style.color = t.status === "passed" ? "var(--ok-ink)" : (t.status === "failed" ? "var(--bad-ink)" : "var(--muted)");
      r.appendChild(st);
      r.appendChild(el("td", null, t.duration !== undefined && t.duration !== null ? t.duration + " ms" : ""));
      tbl.appendChild(r);
    });
    b.appendChild(tbl);
    main.appendChild(b);
  }
  [["stdout", "Verifier stdout"], ["stderr", "Verifier stderr"]].forEach(([k, title]) => {
    if (!v[k]) return;
    const b = el("div", "panelblock");
    b.appendChild(el("h2", null, title));
    collapsibleText(b, v[k], "tout");
    main.appendChild(b);
  });
  if (!main.children.length) main.appendChild(el("div", "panelblock", "No verifier artifacts in this rollout."));
}

/* ── metrics tab ────────────────────────────────────────── */
function renderMetrics(p) {
  const META = p.meta || {};
  const main = document.getElementById("view-metrics");
  main.textContent = "";
  const timing = META.timing || {};
  const phases = ["environment_setup", "agent_setup", "agent_execution", "verifier"].filter(k => typeof timing[k] === "number");
  if (phases.length) {
    const b = el("div", "panelblock");
    b.appendChild(el("h2", null, "Phase timing"));
    const max = Math.max(...phases.map(k => timing[k]), 1);
    /* strip tones from the shared accent palette */
    const PHASE_COLOR = {
      environment_setup: "#74bda0", agent_setup: "#6fb6ca",
      agent_execution: "#7fa8d8", verifier: "#a78fd6",
    };
    phases.forEach(k => {
      const row = el("div", "barrow");
      row.appendChild(el("span", null, k.replace(/_/g, " ")));
      const track = el("div", null);
      const bar = el("div", "bar");
      bar.style.width = Math.max(2, (timing[k] / max) * 100) + "%";
      bar.style.background = PHASE_COLOR[k] || "var(--rule-strong)";
      track.appendChild(bar);
      row.appendChild(track);
      row.appendChild(el("span", "num", timing[k].toFixed(1) + " s"));
      b.appendChild(row);
    });
    if (typeof timing.total === "number") {
      const row = el("div", "barrow");
      row.appendChild(el("span", null, "total"));
      row.appendChild(el("div", null));
      row.appendChild(el("span", "num", timing.total.toFixed(1) + " s"));
      b.appendChild(row);
    }
    main.appendChild(b);
  }
  const u = META.usage || {};
  const rows = [
    ["input tokens", u.n_input_tokens], ["output tokens", u.n_output_tokens],
    ["cache read", u.n_cache_read_tokens], ["cache write", u.n_cache_creation_tokens],
    ["thought tokens", (u.usage_details || {}).thought_tokens], ["total", u.total_tokens],
  ].filter(([, v2]) => v2 !== null && v2 !== undefined);
  if (rows.length) {
    const b = el("div", "panelblock");
    b.appendChild(el("h2", null, "Token usage" + (u.usage_source ? " — " + u.usage_source : "")));
    const tbl = el("table", "plain");
    rows.forEach(([k, v2]) => {
      const r = el("tr");
      r.appendChild(el("td", null, k));
      r.appendChild(el("td", null, v2.toLocaleString()));
      tbl.appendChild(r);
    });
    b.appendChild(tbl);
    main.appendChild(b);
  }
  const counts = META.counts || {};
  const crows = Object.entries(counts).filter(([, v2]) => v2 !== null && v2 !== undefined);
  if (crows.length) {
    const b = el("div", "panelblock");
    b.appendChild(el("h2", null, "Event counts"));
    const tbl = el("table", "plain");
    crows.forEach(([k, v2]) => {
      const r = el("tr");
      r.appendChild(el("td", null, k));
      r.appendChild(el("td", null, v2));
      tbl.appendChild(r);
    });
    b.appendChild(tbl);
    main.appendChild(b);
  }
  if (!main.children.length) main.appendChild(el("div", "panelblock", "No metrics in this rollout."));
}

/* ── payload loading ────────────────────────────────────── */
function resetState() {
  state.focus = false; state.kinds.clear(); state.failedOnly = false;
  state.matches = []; state.matchIdx = -1;
}
function loadPayload(p) {
  CUR = p;
  renderGen++;
  resetState();
  renderHeader(p);
  renderTabs(p);
  renderToolbar(p);
  renderTrace(p);
  renderVerifier(p);
  renderMetrics(p);
  document.title = (p.meta && p.meta.task_name) || p.rollout_name || "benchflow trajectory";
}
function showLoadError(message) {
  const main = document.getElementById("view-trace");
  main.textContent = "";
  const b = el("div", "errbox");
  b.appendChild(el("span", "elabel", "load error"));
  b.appendChild(el("span", "etext", message));
  main.appendChild(b);
}

/* ── browse mode: run sidebar + dynamic fetch ───────────── */
function renderSidebar(rollouts, activeId) {
  const side = document.getElementById("side");
  side.textContent = "";
  side.classList.remove("hidden");
  const heading = BOOT.capped
    ? "first " + rollouts.length + " runs (capped)"
    : rollouts.length + " runs";
  const h2 = el("h2", null, heading);
  if (BOOT.capped) h2.title = "Raise BENCHFLOW_VIEWER_MAX_RUNS to discover more runs.";
  side.appendChild(h2);
  rollouts.forEach(r => {
    const st = (r.reward === null || r.reward === undefined) ? "unscored" : (r.reward >= 1 ? "pass" : "fail");
    const b = el("button", "run " + st);
    b.setAttribute("aria-current", r.id === activeId ? "true" : "false");
    b.dataset.id = r.id;
    const name = el("div", "rname", r.task_name || r.name || r.id);
    const sub = el("div", "rsub");
    const dot = el("span", "dot " + st);
    sub.appendChild(dot);
    sub.appendChild(el("span", null,
      (r.reward === null || r.reward === undefined ? (r.has_error ? "error" : "unscored") : (r.reward >= 1 ? "pass" : "fail " + r.reward))));
    if (r.agent_name) sub.appendChild(el("span", null, r.agent_name));
    if (r.model) sub.appendChild(el("span", "mono", r.model));
    if (r.skill_mode) sub.appendChild(el("span", null, r.skill_mode));
    b.appendChild(name); b.appendChild(sub);
    b.addEventListener("click", () => selectRun(r.id, true));
    side.appendChild(b);
  });
}
function markActive(id) {
  document.querySelectorAll("#side .run").forEach(b =>
    b.setAttribute("aria-current", b.dataset.id === id ? "true" : "false"));
}
async function selectRun(id, push) {
  markActive(id);
  try {
    const resp = await fetch("/api/rollout?id=" + encodeURIComponent(id));
    if (!resp.ok) { showLoadError("HTTP " + resp.status + " loading run: " + id); return; }
    const payload = await resp.json();
    loadPayload(payload);
    if (push) {
      const url = new URL(location.href);
      url.searchParams.set("run", id);
      url.hash = "";
      history.pushState({ run: id }, "", url);
    }
  } catch (err) {
    showLoadError("Failed to load run " + id + ": " + err);
  }
}

/* ── boot ───────────────────────────────────────────────── */
if (BOOT.mode === "browse") {
  const rollouts = BOOT.rollouts || [];
  const requested = new URLSearchParams(location.search).get("run");
  if (requested && !rollouts.some(r => r.id === requested)) {
    /* Never silently render a different run than the URL asked for. */
    renderSidebar(rollouts, null);
    showLoadError(
      'Run "' + requested + '" is not among the ' + rollouts.length +
      " discovered runs" +
      (BOOT.capped ? " (list is capped — raise BENCHFLOW_VIEWER_MAX_RUNS)." : "."));
  } else {
    const initial = requested || (rollouts[0] && rollouts[0].id);
    renderSidebar(rollouts, initial || null);
    if (initial) selectRun(initial, false);
    else showLoadError("No runs found under this directory.");
  }
  window.addEventListener("popstate", () => {
    const id = new URLSearchParams(location.search).get("run");
    if (id && rollouts.some(r => r.id === id)) selectRun(id, false);
  });
} else {
  loadPayload(BOOT.payload || {});
}
