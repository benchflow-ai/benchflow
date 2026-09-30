/* Job-level views: Outcomes grid, Pareto chart, Training view.

   The server precomputes everything (benchflow.outcomes/1, see outcomes.py):
   this module only regroups trial indices and draws. All text goes through
   textContent / createTextNode; data never becomes markup. */
BF.jobviews = (() => {
  const { el, plural } = BF.core;
  const SVG = "http://www.w3.org/2000/svg";
  const VIEWS = [["runs", "Runs"], ["outcomes", "Outcomes"], ["pareto", "Pareto"], ["training", "Training"]];
  const UNRECORDED = "(unrecorded)";

  let doc = null;
  let loadError = null;
  let loading = null;
  let mode = "browse";
  let openRun = null;
  let enabled = false;
  const state = {
    view: "runs",
    rows: "task",
    cols: "model",
    transpose: false,
    sort: "name",
    color: "reward",
    others: false,
    query: "",
    group: "model",
    part: "dataset",
    x: "tokens",
    y: "mean_reward",
    logx: true,
    table: false,
    metric: "mean_reward",
    before: null,
    after: null,
  };

  // ── formatting ─────────────────────────────────────────────────────────
  // Whole percents, one decimal near 0 and 1 (so 99.6% never reads 100%).
  function pct(v) {
    if (typeof v !== "number" || !Number.isFinite(v)) return "n/a";
    if (v === 0 || v === 1) return 100 * v + "%";
    return (100 * v).toFixed(v < 0.1 || v > 0.99 ? 1 : 0) + "%";
  }
  function num(v, digits = 2) {
    return typeof v === "number" ? v.toFixed(digits) : "n/a";
  }
  function fmtMetric(key, v) {
    if (typeof v !== "number" || !Number.isFinite(v)) return "n/a";
    if (key === "usd") return v >= 1 ? "$" + v.toFixed(2) : "$" + v.toFixed(v >= 0.01 ? 3 : 4);
    if (key === "tokens") return v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? (v / 1e3).toFixed(1) + "k" : String(Math.round(v));
    if (key === "sandbox_sec" || key === "wall_sec") return v >= 3600 ? (v / 3600).toFixed(1) + "h" : v >= 120 ? (v / 60).toFixed(1) + "m" : Math.round(v) + "s";
    if (key === "mean_reward" || key === "solve_rate") return pct(v);
    return num(v, 3);
  }
  function naturalSort(values, dim) {
    const order = dim === "step" && doc.training ? doc.training.steps : null;
    return values.slice().sort((a, b) => {
      const va = doc.dims[dim].values[a];
      const vb = doc.dims[dim].values[b];
      if ((va === UNRECORDED) !== (vb === UNRECORDED)) return va === UNRECORDED ? 1 : -1;
      if (order) {
        const ia = order.indexOf(va);
        const ib = order.indexOf(vb);
        if (ia !== ib) return (ia < 0 ? 1e9 : ia) - (ib < 0 ? 1e9 : ib);
      }
      return va.localeCompare(vb, undefined, { numeric: true });
    });
  }
  function hasSpread(dim) {
    return doc.dims[dim] && doc.dims[dim].values.length > 1;
  }
  // Whether a dimension takes more than one value among agent runs.
  function agentSpread(dim) {
    const d = doc.dims[dim];
    if (!d) return false;
    let first = -1;
    for (let i = 0; i < doc.n; i += 1) {
      if (doc.columns.role[i] !== 0) continue;
      if (first < 0) first = d.codes[i];
      else if (d.codes[i] !== first) return true;
    }
    return false;
  }
  const PREFERRED = ["model", "agent", "harness", "step", "split", "seed", "dataset", "job"];
  let defaultsApplied = false;
  // First time the data is shown: columns and Pareto points default to a
  // dimension that varies among agent runs (unless the URL chose one).
  function applyDefaults() {
    if (defaultsApplied) return;
    defaultsApplied = true;
    const params = new URLSearchParams(location.search);
    const varied = PREFERRED.find(agentSpread);
    if (!params.has("cols")) state.cols = varied || "outcome";
    if (!params.has("pg")) {
      const groups = Object.keys(doc.pareto);
      state.group = PREFERRED.find((d) => groups.includes(d) && agentSpread(d)) || groups[0] || "model";
    }
  }

  // ── URL state ──────────────────────────────────────────────────────────
  const URL_KEYS = { rows: "rows", cols: "cols", transpose: "tr", group: "pg", part: "pp", x: "px", y: "py" };
  function readURL(params) {
    const view = params.get("view");
    state.view = VIEWS.some(([key]) => key === view) ? view : (mode === "export" ? "outcomes" : "runs");
    Object.entries(URL_KEYS).forEach(([key, name]) => {
      const value = params.get(name);
      if (value === null) return;
      state[key] = key === "transpose" ? value === "1" : value;
    });
  }
  // A run URL keeps only the view, so going back returns to it.
  function addParams(params, hasRun) {
    if (!enabled || state.view === "runs") return;
    params.set("view", state.view);
    if (hasRun) return;
    Object.entries(URL_KEYS).forEach(([key, name]) => {
      const value = state[key];
      if (key === "transpose") { if (value) params.set(name, "1"); } else params.set(name, value);
    });
  }
  function writeURL() {
    if (mode === "export") {
      const params = new URLSearchParams();
      addParams(params, false);
      history.replaceState({}, "", location.pathname + (params.toString() ? "?" + params : ""));
    } else {
      BF.catalog.writeURL(null, false);
    }
  }

  // ── tabs and loading ───────────────────────────────────────────────────
  function renderTabs() {
    const nav = document.getElementById("jobtabs");
    nav.textContent = "";
    nav.classList.remove("hidden");
    VIEWS.forEach(([key, label]) => {
      if (key === "runs" && mode === "export") return;
      if (key === "training" && doc && !doc.training) return;
      const button = el("button", "jtab" + (state.view === key ? " on" : ""), label);
      button.type = "button";
      button.setAttribute("aria-pressed", state.view === key ? "true" : "false");
      button.addEventListener("click", () => {
        if (state.view === key) return;
        state.view = key;
        if (mode === "export") writeURL();
        else BF.catalog.writeURL(null, true);
        show();
      });
      nav.appendChild(button);
    });
  }

  function ensureData() {
    if (doc || loadError || loading) return loading;
    loading = fetch("/api/outcomes")
      .then(async (response) => {
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.error || "HTTP " + response.status);
        if (!BF.core.isRecord(body) || !BF.core.isRecord(body.columns)) throw new Error("malformed outcomes document");
        doc = body;
      })
      .catch((error) => { loadError = error.message; })
      .finally(() => { loading = null; show(); });
    return loading;
  }

  // Show the current view (called after the catalog is shown, and on tab clicks).
  function show() {
    if (!enabled) return;
    renderTabs();
    const index = document.getElementById("view-index");
    const main = document.getElementById("view-job");
    if (state.view === "runs") {
      main.classList.add("hidden");
      if (mode !== "export") index.classList.remove("hidden");
      return;
    }
    index.classList.add("hidden");
    main.classList.remove("hidden");
    main.textContent = "";
    if (!doc) {
      if (loadError) {
        main.appendChild(el("div", "jnote bad", "Could not build the job views: " + loadError));
      } else {
        main.appendChild(el("div", "jnote", "Reading the job's trials… (built once on the server)"));
        ensureData();
      }
      return;
    }
    if (state.view === "training" && !doc.training) state.view = "outcomes";
    applyDefaults();
    renderHeader(main);
    if (state.view === "outcomes") renderOutcomes(main);
    else if (state.view === "pareto") renderPareto(main);
    else renderTraining(main);
  }

  function renderHeader(main) {
    const head = el("div", "jhead");
    const agents = doc.columns.role.filter((r) => r === 0).length;
    head.appendChild(el("span", null, plural(doc.n, "trial") + (agents !== doc.n ? " (" + agents + " agent runs)" : "")));
    head.appendChild(el("span", null, plural(doc.dims.task.values.length, "task")));
    if (doc.roots.length > 1) head.appendChild(el("span", null, "jobs: " + doc.roots.join(", ")));
    if (doc.redaction) head.appendChild(el("span", null, "masked for sharing: " + doc.redaction));
    main.appendChild(head);
    (doc.notes || []).forEach((note) => main.appendChild(el("div", "jnote", note)));
  }

  function control(container, id, label, options, value, onChange) {
    const wrap = el("label", "jctl");
    wrap.appendChild(el("span", null, label));
    const select = el("select");
    select.id = id;
    options.forEach(([key, text]) => {
      const option = el("option", null, text);
      option.value = key;
      option.selected = key === value;
      select.appendChild(option);
    });
    select.addEventListener("change", () => { onChange(select.value); writeURL(); show(); });
    wrap.appendChild(select);
    container.appendChild(wrap);
  }
  function checkbox(container, id, label, value, onChange) {
    const wrap = el("label", "jctl jcheck");
    const box = el("input");
    box.type = "checkbox";
    box.id = id;
    box.checked = value;
    box.addEventListener("change", () => { onChange(box.checked); writeURL(); show(); });
    wrap.append(box, el("span", null, label));
    container.appendChild(wrap);
  }

  // ── tooltip ────────────────────────────────────────────────────────────
  let tip = null;
  function tooltip() {
    if (!tip) {
      tip = el("div", "jtip hidden");
      tip.setAttribute("role", "tooltip");
      document.body.appendChild(tip);
    }
    return tip;
  }
  function showTip(lines, event) {
    const box = tooltip();
    box.textContent = "";
    lines.filter(Boolean).forEach(([cls, text]) => box.appendChild(el("div", cls, text)));
    box.classList.remove("hidden");
    const pad = 14;
    const { innerWidth, innerHeight } = window;
    const rect = box.getBoundingClientRect();
    let left = event.clientX + pad;
    let top = event.clientY + pad;
    if (left + rect.width > innerWidth - 8) left = event.clientX - rect.width - pad;
    if (top + rect.height > innerHeight - 8) top = event.clientY - rect.height - pad;
    box.style.left = Math.max(4, left) + window.scrollX + "px";
    box.style.top = Math.max(4, top) + window.scrollY + "px";
  }
  function hideTip() {
    if (tip) tip.classList.add("hidden");
  }

  // ── Outcomes ───────────────────────────────────────────────────────────
  function statLine(stat) {
    if (!stat) return "no agent trials";
    if (stat.solve_rate === null) return plural(stat.trials, "trial") + ", none scored";
    let text = "pass@1 " + pct(stat.solve_rate);
    if (stat.interval) text += " (" + pct(stat.interval[0]) + "–" + pct(stat.interval[1]) + ")";
    const top = stat.at_k.length ? stat.at_k[stat.at_k.length - 1] : null;
    if (top && top[0] > 1 && top[1] !== null) text += " · pass@" + top[0] + " " + pct(top[1]);
    if (stat.mean_reward !== null && Math.abs(stat.mean_reward - stat.solve_rate) > 1e-9) text += " · mean " + num(stat.mean_reward);
    text += " · n=" + stat.scored + (stat.unscored ? " (+" + stat.unscored + " unscored)" : "");
    return text;
  }

  function trialLines(i, rowLabel, colLabel) {
    const c = doc.columns;
    const lines = [["tt", doc.dims.task.values[doc.dims.task.codes[i]]], ["tm", c.name[i]]];
    const context = [rowLabel, colLabel].filter((v) => v && v !== "all trials").join(" · ");
    if (context) lines.push(["tm", context]);
    const outcome = doc.vocab.outcome[c.outcome[i]];
    lines.push(["tv", c.reward[i] === null ? "unscored (no reward, not counted as 0)" : outcome + ", reward " + num(c.reward[i], 3)]);
    const execution = doc.vocab.execution[c.execution[i]];
    if (execution !== "completed") lines.push(["tm", "execution: " + execution.replace("_", " ")]);
    if (c.cause[i] >= 0) {
      const cause = doc.causes[c.cause[i]];
      lines.push(["tc", cause.label + " (" + cause.fault_words + ")"]);
      if (cause.next_step) lines.push(["tm", "next: " + cause.next_step]);
    }
    if (c.detail[i]) lines.push(["td", c.detail[i]]);
    const attempts = doc.attempt_outcomes[String(i)];
    if (attempts) {
      lines.push(["tm", plural(attempts.length, "attempt") + ", oldest first: " + attempts.map(([code, reward]) =>
        reward === null ? "unscored" : doc.vocab.outcome[code] + " " + num(reward, 2)).join(", ")]);
    }
    const verdict = doc.integrity_details[String(i)];
    if (verdict) {
      lines.push([verdict.exploited ? "tx" : "tm", "BenchShield: " + verdict.verdict + (verdict.exploited ? " (exploit, " + verdict.severity + ")" : "") + (verdict.reason ? ": " + verdict.reason : "")]);
    }
    const costs = [
      c.usd[i] !== null ? fmtMetric("usd", c.usd[i]) + (c.usd_estimated[i] ? " (estimated from the session log)" : "") : null,
      c.tokens[i] !== null ? fmtMetric("tokens", c.tokens[i]) + " tokens" : null,
      c.sandbox_sec[i] !== null ? fmtMetric("sandbox_sec", c.sandbox_sec[i]) + " sandbox" : null,
    ].filter(Boolean);
    if (costs.length) lines.push(["tm", costs.join(" · ") + (attempts ? " (all attempts)" : "")]);
    if (c.link[i] && mode !== "export") lines.push(["tl", "click to open the trial"]);
    return lines;
  }

  function renderOutcomes(main) {
    const c = doc.columns;
    const controls = el("div", "jcontrols");
    const rowDims = ["task", "dataset", "all"].filter((d) => d !== "dataset" || hasSpread(d));
    if (!rowDims.includes(state.rows)) state.rows = "task";
    const colDims = ["model", "agent", "harness", "seed", "step", "split", "dataset", "job", "outcome"].filter(hasSpread);
    if (state.cols !== "all" && !colDims.includes(state.cols)) state.cols = colDims[0] || "all";
    control(controls, "jrows", "rows", rowDims.map((d) => [d, doc.dims[d].label]), state.rows, (v) => { state.rows = v; });
    control(controls, "jcols", "columns", [["all", "none"], ...colDims.map((d) => [d, doc.dims[d].label])], state.cols, (v) => { state.cols = v; });
    checkbox(controls, "jtranspose", "transpose", state.transpose, (v) => { state.transpose = v; });
    control(controls, "jsort", "sort rows", [["name", "name"], ["solve", "pass@1"], ["mean", "mean reward"], ["trials", "trials"]], state.sort, (v) => { state.sort = v; });
    control(controls, "jcolor", "color", [["reward", "reward (continuous)"], ["pass", "passed or not"]], state.color, (v) => { state.color = v; });
    const hasOthers = c.role.some((r) => r !== 0);
    if (hasOthers) checkbox(controls, "jothers", "show control and optimizer runs", state.others, (v) => { state.others = v; });
    const search = el("input");
    search.type = "search";
    search.placeholder = "filter rows…";
    search.value = state.query;
    search.setAttribute("aria-label", "Filter rows");
    search.addEventListener("input", () => { state.query = search.value; renderGrid(); });
    controls.appendChild(search);
    main.appendChild(controls);
    main.appendChild(legend());
    const holder = el("div", "jgridwrap");
    main.appendChild(holder);

    function renderGrid() {
      holder.textContent = "";
      const R = state.transpose ? state.cols : state.rows;
      const C = state.transpose ? state.rows : state.cols;
      const rd = doc.dims[R];
      const cd = doc.dims[C];
      const query = state.query.trim().toLowerCase();
      const cells = new Map();
      const colSeen = new Set();
      for (let i = 0; i < doc.n; i += 1) {
        if (!state.others && c.role[i] !== 0) continue;
        const r = rd.codes[i];
        if (query && !rd.values[r].toLowerCase().includes(query)) continue;
        const k = cd.codes[i];
        colSeen.add(k);
        if (!cells.has(r)) cells.set(r, new Map());
        const row = cells.get(r);
        if (!row.has(k)) row.set(k, []);
        row.get(k).push(i);
      }
      const colCodes = naturalSort([...colSeen], C);
      const stats = doc.stats[R];
      const key = {
        name: null,
        solve: (s) => (s && s.solve_rate !== null ? s.solve_rate : -1),
        mean: (s) => (s && s.mean_reward !== null ? s.mean_reward : -1),
        trials: (s) => (s ? s.trials : -1),
      }[state.sort];
      let rowCodes = naturalSort([...cells.keys()], R);
      if (key) rowCodes = rowCodes.slice().sort((a, b) => key(stats[b]) - key(stats[a]));
      if (!rowCodes.length) {
        holder.appendChild(el("div", "jnote", doc.n ? "No rows match." : "No trials."));
        return;
      }
      const table = el("table", "jgrid" + (state.color === "pass" ? " discrete" : ""));
      const thead = el("thead");
      const hr = el("tr");
      hr.appendChild(el("th", "rh", rd.label + " ↓ / " + cd.label + " →"));
      colCodes.forEach((k) => {
        const th = el("th", "ch");
        th.appendChild(el("span", "chl", cd.values[k]));
        const s = doc.stats[C][k];
        if (s && C !== "all") th.appendChild(el("span", "chs", s.solve_rate === null ? "" : "pass@1 " + pct(s.solve_rate)));
        th.title = cd.values[k] + (s ? "\n" + statLine(s) : "");
        hr.appendChild(th);
      });
      thead.appendChild(hr);
      table.appendChild(thead);
      const tbody = el("tbody");
      const byOutcome = (a, b) => c.outcome[a] - c.outcome[b] || (c.reward[b] ?? -1) - (c.reward[a] ?? -1) || a - b;
      rowCodes.forEach((r) => {
        const tr = el("tr");
        const th = el("th", "rh");
        th.scope = "row";
        th.appendChild(el("span", "rl", rd.values[r]));
        th.appendChild(el("span", "rs", statLine(stats[r])));
        tr.appendChild(th);
        const row = cells.get(r);
        colCodes.forEach((k) => {
          const td = el("td");
          const members = row.get(k);
          if (members) {
            td.dataset.r = r;
            td.dataset.k = k;
            const frag = document.createDocumentFragment();
            members.sort(byOutcome).forEach((i) => {
              const sq = document.createElement("span");
              let cls = "sq";
              if (c.reward[i] === null) cls += " u";
              else {
                cls += " s";
                if (c.outcome[i] === 0) cls += " p";
                sq.style.setProperty("--r", String(Math.max(0, Math.min(1, c.reward[i]))));
                if (c.execution[i] !== 0) cls += " e";
              }
              if (c.attempts[i] > 1) cls += " t";
              if (c.integrity[i] === 2) cls += " x";
              else if (c.integrity[i] === 1) cls += " v";
              if (c.role[i] !== 0) cls += " o";
              if (c.link[i] && mode !== "export") cls += " l";
              sq.className = cls;
              sq.dataset.i = i;
              frag.appendChild(sq);
            });
            td.appendChild(frag);
          }
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      table.addEventListener("mouseover", (event) => {
        const sq = event.target.closest(".sq");
        if (!sq) { hideTip(); return; }
        const i = Number(sq.dataset.i);
        const td = sq.parentElement;
        showTip(trialLines(i, rd.values[Number(td.dataset.r)], cd.values[Number(td.dataset.k)]), event);
      });
      table.addEventListener("mouseleave", hideTip);
      table.addEventListener("click", (event) => {
        const sq = event.target.closest(".sq.l");
        if (!sq || !openRun) return;
        hideTip();
        openRun(c.link[Number(sq.dataset.i)]);
      });
      holder.appendChild(table);
    }
    renderGrid();
  }

  function legend() {
    const box = el("div", "jlegend");
    const ramp = el("span", "lg");
    ramp.appendChild(el("span", null, "reward 0"));
    ramp.appendChild(el("span", "ramp"));
    ramp.appendChild(el("span", null, "1"));
    box.appendChild(ramp);
    const item = (cls, text) => {
      const span = el("span", "lg");
      span.appendChild(el("span", "sq " + cls));
      span.appendChild(el("span", null, text));
      box.appendChild(span);
    };
    item("u", "unscored: infrastructure, setup or verifier failure (hover for the cause)");
    item("s e", "errored or timed out, then scored");
    item("s t", "retried (hover for every attempt)");
    if (doc.columns.integrity.some((v) => v >= 0)) {
      item("s x", "BenchShield exploit (AgentViolation)");
      item("s v", "vector exposed");
    } else {
      box.appendChild(el("span", "lg muted", "no BenchShield verdicts (run with --integrity audit)"));
    }
    return box;
  }

  // ── SVG helpers ────────────────────────────────────────────────────────
  function svg(tag, attrs, parent) {
    const node = document.createElementNS(SVG, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, String(v)));
    if (parent) parent.appendChild(node);
    return node;
  }
  function text(parent, x, y, value, attrs = {}) {
    const node = svg("text", { x, y, ...attrs }, parent);
    node.textContent = value;
    return node;
  }
  function niceTicks(lo, hi, count = 5) {
    if (!(hi > lo)) return [lo];
    const span = hi - lo;
    const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * step0).find((s) => span / s <= count) || step0 * 10;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
    return out;
  }
  function logTicks(lo, hi) {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e += 1) {
      [1, 2, 5].forEach((m) => {
        const v = m * Math.pow(10, e);
        if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
      });
    }
    return out.length >= 3 ? out : niceTicks(lo, hi, 4).filter((v) => v > 0);
  }
  function catVar(index) {
    return "var(--viz-cat-" + ((index % 8) + 1) + ")";
  }

  // ── Pareto ─────────────────────────────────────────────────────────────
  function renderPareto(main) {
    const groups = Object.keys(doc.pareto);
    if (!groups.length) { main.appendChild(el("div", "jnote", "No agent trials to plot.")); return; }
    if (!groups.includes(state.group)) state.group = groups[0];
    const parts = Object.keys(doc.pareto[state.group]);
    if (!parts.includes(state.part)) state.part = parts.includes("dataset") ? "dataset" : parts[0];
    if (!(state.x in doc.x_metrics)) state.x = "tokens";
    if (!(state.y in doc.y_metrics)) state.y = "mean_reward";
    const controls = el("div", "jcontrols");
    control(controls, "jpg", "points", groups.map((g) => [g, doc.dims[g].label]), state.group, (v) => { state.group = v; });
    control(controls, "jpp", "frontier per", parts.map((p) => [p, p === "none" ? "none (one frontier)" : doc.dims[p].label]), state.part, (v) => { state.part = v; });
    control(controls, "jpx", "x", Object.entries(doc.x_metrics), state.x, (v) => { state.x = v; });
    control(controls, "jpy", "y", Object.entries(doc.y_metrics), state.y, (v) => { state.y = v; });
    checkbox(controls, "jplog", "log x", state.logx, (v) => { state.logx = v; });
    checkbox(controls, "jptable", "table", state.table, (v) => { state.table = v; });
    main.appendChild(controls);

    const { points, frontiers } = doc.pareto[state.group][state.part];
    const X = state.x;
    const Y = state.y;
    const plotted = points.filter((p) => p.m[X][0] !== null && p.m[Y][0] !== null);
    const missing = points.filter((p) => p.m[X][0] === null || p.m[Y][0] === null);
    const partitions = [...new Set(points.map((p) => p.partition))].sort();
    const caption = el("div", "jcaption", "Each point is one " + doc.dims[state.group].label + " (per " + (state.part === "none" ? "job" : doc.dims[state.part].label)
      + "): mean " + doc.x_metrics[X] + " per trial (retries included) against " + doc.y_metrics[Y]
      + " over scored trials. Bars are 95% bootstrap intervals over tasks; lines are Pareto frontiers (cheaper and better).");
    main.appendChild(caption);
    if (!plotted.length) {
      main.appendChild(el("div", "jnote", "No point has both " + doc.x_metrics[X] + " and " + doc.y_metrics[Y] + " recorded."));
    } else {
      main.appendChild(paretoChart(plotted, points, frontiers[X + "|" + Y] || {}, partitions, X, Y));
    }
    if (missing.length) {
      main.appendChild(el("div", "jnote", "Not plotted (no " + doc.x_metrics[X] + " or " + doc.y_metrics[Y] + " recorded): "
        + missing.map((p) => p.group + (state.part === "none" ? "" : " / " + p.partition)).join(", ")));
    }
    if (state.table) main.appendChild(paretoTable(points, X, Y));
  }

  function paretoChart(plotted, all, frontier, partitions, X, Y) {
    const W = 820, H = 460, m = { l: 64, r: 150, t: 16, b: 48 };
    const xs = plotted.flatMap((p) => [p.m[X][0], p.m[X][1], p.m[X][2]]).filter((v) => typeof v === "number");
    const ys = plotted.flatMap((p) => [p.m[Y][0], p.m[Y][1], p.m[Y][2]]).filter((v) => typeof v === "number");
    const log = state.logx && Math.min(...xs) > 0;
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    if (log) { x0 /= 1.25; x1 *= 1.25; } else { const pad = (x1 - x0 || Math.abs(x1) || 1) * 0.08; x0 = Math.max(0, x0 - pad); x1 += pad; }
    const unit = Y === "mean_reward" || Y === "solve_rate";
    let y0 = unit ? 0 : Math.min(0, ...ys), y1 = unit ? 1 : Math.max(...ys) * 1.05 || 1;
    const sx = (v) => m.l + (log ? (Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) : (v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
    const sy = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "jchart", role: "img", "aria-label": "Pareto chart" });
    const grid = svg("g", { class: "grid" }, root);
    (unit ? [0, 0.25, 0.5, 0.75, 1] : niceTicks(y0, y1)).forEach((v) => {
      svg("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }, grid);
      text(grid, m.l - 8, sy(v) + 4, fmtMetric(Y, v), { "text-anchor": "end", class: "tick" });
    });
    (log ? logTicks(x0, x1) : niceTicks(x0, x1)).forEach((v) => {
      svg("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "v" }, grid);
      text(grid, sx(v), H - m.b + 16, fmtMetric(X, v), { "text-anchor": "middle", class: "tick" });
    });
    text(root, (m.l + W - m.r) / 2, H - 10, doc.x_metrics[X] + " per trial" + (log ? " (log)" : ""), { "text-anchor": "middle", class: "axis" });
    text(root, 14, (m.t + H - m.b) / 2, doc.y_metrics[Y], { "text-anchor": "middle", class: "axis", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` });
    const color = new Map(partitions.map((p, i) => [p, catVar(i)]));
    // Frontiers first, under the points.
    Object.entries(frontier).forEach(([part, idx]) => {
      const pts = idx.map((j) => all[j]).filter((p) => p.m[X][0] !== null);
      if (pts.length < 2) return;
      let d = "";
      pts.forEach((p, j) => {
        const px = sx(p.m[X][0]), py = sy(p.m[Y][0]);
        d += j === 0 ? `M${px},${py}` : `H${px}V${py}`;
      });
      svg("path", { d, class: "frontier", stroke: color.get(part) }, root);
    });
    const onFrontier = new Set(Object.values(frontier).flat());
    const marks = svg("g", {}, root);
    plotted.forEach((p) => {
      const j = all.indexOf(p);
      const g = svg("g", { class: "pt", tabindex: 0 }, marks);
      const px = sx(p.m[X][0]), py = sy(p.m[Y][0]);
      const stroke = color.get(p.partition);
      if (p.m[X][1] !== null) svg("line", { x1: sx(Math.max(p.m[X][1], log ? x0 : -Infinity)), x2: sx(p.m[X][2]), y1: py, y2: py, class: "err", stroke }, g);
      if (p.m[Y][1] !== null) svg("line", { x1: px, x2: px, y1: sy(p.m[Y][1]), y2: sy(p.m[Y][2]), class: "err", stroke }, g);
      svg("circle", { cx: px, cy: py, r: onFrontier.has(j) ? 5.5 : 4.5, fill: stroke, class: onFrontier.has(j) ? "on" : "off" }, g);
      svg("circle", { cx: px, cy: py, r: 12, class: "hit" }, g);
      if (onFrontier.has(j) || plotted.length <= 8) text(g, px + 8, py - 8, p.group, { class: "dl" });
      const lines = [
        ["tt", p.group + (state.part === "none" ? "" : " · " + p.partition)],
        ["tm", plural(p.trials, "trial") + ", " + plural(p.tasks, "task") + (onFrontier.has(j) ? " · on the frontier" : "")],
        ["tv", doc.y_metrics[Y] + " " + fmtMetric(Y, p.m[Y][0]) + (p.m[Y][1] !== null ? " (" + fmtMetric(Y, p.m[Y][1]) + "–" + fmtMetric(Y, p.m[Y][2]) + ")" : "") + ", n=" + p.m[Y][3]],
        ["tv", doc.x_metrics[X] + " " + fmtMetric(X, p.m[X][0]) + (p.m[X][1] !== null ? " (" + fmtMetric(X, p.m[X][1]) + "–" + fmtMetric(X, p.m[X][2]) + ")" : "") + ", n=" + p.m[X][3]],
      ];
      g.addEventListener("mousemove", (event) => showTip(lines, event));
      g.addEventListener("mouseleave", hideTip);
    });
    // Legend: one entry per partition.
    if (state.part !== "none") {
      const lg = svg("g", { class: "lgd" }, root);
      partitions.forEach((p, i) => {
        const y = m.t + 8 + i * 18;
        svg("circle", { cx: W - m.r + 18, cy: y, r: 5, fill: color.get(p) }, lg);
        text(lg, W - m.r + 28, y + 4, p.length > 18 ? p.slice(0, 17) + "…" : p, { class: "tick" });
      });
    }
    const wrap = el("div", "jchartwrap");
    wrap.appendChild(root);
    return wrap;
  }

  function paretoTable(points, X, Y) {
    const table = el("table", "jtable");
    const head = el("tr");
    [doc.dims[state.group].label, state.part === "none" ? null : doc.dims[state.part].label, "trials", "tasks", doc.x_metrics[X], "95% interval", doc.y_metrics[Y], "95% interval"]
      .filter((h) => h !== null).forEach((h) => head.appendChild(el("th", null, h)));
    table.appendChild(head);
    points.forEach((p) => {
      const tr = el("tr");
      const cells = [p.group, state.part === "none" ? null : p.partition, p.trials, p.tasks,
        fmtMetric(X, p.m[X][0]), p.m[X][1] === null ? "" : fmtMetric(X, p.m[X][1]) + "–" + fmtMetric(X, p.m[X][2]),
        fmtMetric(Y, p.m[Y][0]), p.m[Y][1] === null ? "" : fmtMetric(Y, p.m[Y][1]) + "–" + fmtMetric(Y, p.m[Y][2])];
      cells.filter((v) => v !== null).forEach((v) => tr.appendChild(el("td", null, v)));
      table.appendChild(tr);
    });
    return table;
  }

  // ── Training ───────────────────────────────────────────────────────────
  function renderTraining(main) {
    const t = doc.training;
    const controls = el("div", "jcontrols");
    control(controls, "jtm", "y", [["mean_reward", "mean reward"], ["solve_rate", "solve rate"]], state.metric, (v) => { state.metric = v; });
    const steps = t.steps;
    if (!steps.includes(state.before)) state.before = steps[0];
    if (!steps.includes(state.after)) state.after = steps[steps.length - 1];
    main.appendChild(controls);
    const source = (doc.sources.step || []).join("; ");
    main.appendChild(el("div", "jcaption", "Reward per step for each " + doc.dims[t.group_dim].label + " (steps from: " + (source || "recorded fields")
      + "). Bands are 95% intervals: bootstrap over tasks for mean reward, Wilson (clustered) for the solve rate."));
    main.appendChild(trainingChart(t));
    const held = el("div", "jsection");
    held.appendChild(el("h3", null, (t.heldout_split ? "Held-out tasks (" + t.heldout_split.join(", ") + ")" : "Every task") + ": mean reward before and after"));
    const pick = el("div", "jcontrols");
    control(pick, "jtb", "before", steps.map((s) => [s, s]), state.before, (v) => { state.before = v; });
    control(pick, "jta", "after", steps.map((s) => [s, s]), state.after, (v) => { state.after = v; });
    held.appendChild(pick);
    held.appendChild(dumbbell(t, steps.indexOf(state.before), steps.indexOf(state.after)));
    main.appendChild(held);
  }

  function trainingChart(t) {
    const W = 820, H = 340, m = { l: 56, r: 130, t: 16, b: 40 };
    const steps = t.steps;
    const band = (W - m.l - m.r) / Math.max(1, steps.length - 1);
    const sx = (k) => m.l + (steps.length === 1 ? (W - m.l - m.r) / 2 : k * band);
    const sy = (v) => H - m.b - v * (H - m.t - m.b);
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "jchart", role: "img", "aria-label": "reward over steps" });
    const grid = svg("g", { class: "grid" }, root);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      svg("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) }, grid);
      text(grid, m.l - 8, sy(v) + 4, pct(v), { "text-anchor": "end", class: "tick" });
    });
    steps.forEach((s, k) => text(grid, sx(k), H - m.b + 18, s, { "text-anchor": "middle", class: "tick" }));
    const key = state.metric;
    const ivKey = key === "mean_reward" ? "mean_interval" : "interval";
    t.series.forEach((series, n) => {
      const colorVar = catVar(n);
      const pts = series.points.map((p, k) => (p && p[key] !== null ? { k, p } : null)).filter(Boolean);
      const withIv = pts.filter(({ p }) => p[ivKey]);
      if (withIv.length > 1) {
        const top = withIv.map(({ k, p }) => `${sx(k)},${sy(p[ivKey][1])}`);
        const bottom = withIv.slice().reverse().map(({ k, p }) => `${sx(k)},${sy(p[ivKey][0])}`);
        svg("polygon", { points: [...top, ...bottom].join(" "), class: "band", fill: colorVar }, root);
      }
      if (pts.length > 1) svg("polyline", { points: pts.map(({ k, p }) => `${sx(k)},${sy(p[key])}`).join(" "), class: "line", stroke: colorVar }, root);
      pts.forEach(({ k, p }) => {
        const g = svg("g", { class: "pt" }, root);
        svg("circle", { cx: sx(k), cy: sy(p[key]), r: 4.5, fill: colorVar, class: "on" }, g);
        svg("circle", { cx: sx(k), cy: sy(p[key]), r: 12, class: "hit" }, g);
        const lines = [["tt", series.group + " · " + steps[k]],
          ["tv", (key === "mean_reward" ? "mean reward " : "solve rate ") + pct(p[key]) + (p[ivKey] ? " (" + pct(p[ivKey][0]) + "–" + pct(p[ivKey][1]) + ")" : "")],
          ["tm", p.scored + " scored of " + plural(p.trials, "trial")]];
        g.addEventListener("mousemove", (event) => showTip(lines, event));
        g.addEventListener("mouseleave", hideTip);
      });
      const last = pts[pts.length - 1];
      if (last) text(root, sx(last.k) + 10, sy(last.p[key]) + 4, series.group, { class: "dl" });
    });
    const lg = svg("g", { class: "lgd" }, root);
    t.series.forEach((series, n) => {
      const y = m.t + 8 + n * 18;
      svg("line", { x1: W - m.r + 40, x2: W - m.r + 56, y1: y, y2: y, stroke: catVar(n), class: "line" }, lg);
      text(lg, W - m.r + 62, y + 4, series.group, { class: "tick" });
    });
    const wrap = el("div", "jchartwrap");
    wrap.appendChild(root);
    return wrap;
  }

  function dumbbell(t, b, a) {
    const rows = t.heldout.filter((r) => r.mean[b] !== null || r.mean[a] !== null)
      .map((r) => ({ ...r, delta: (r.mean[a] ?? 0) - (r.mean[b] ?? 0) }))
      .sort((x, y) => y.delta - x.delta || x.task.localeCompare(y.task));
    if (!rows.length) return el("div", "jnote", "No scored trials at these steps.");
    const W = 820, row = 22, m = { l: 230, r: 90, t: 24, b: 28 };
    const H = m.t + m.b + rows.length * row;
    const sx = (v) => m.l + v * (W - m.l - m.r);
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "jchart", role: "img", "aria-label": "held-out tasks before and after" });
    const grid = svg("g", { class: "grid" }, root);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      svg("line", { x1: sx(v), x2: sx(v), y1: m.t - 6, y2: H - m.b, class: "v" }, grid);
      text(grid, sx(v), H - m.b + 16, pct(v), { "text-anchor": "middle", class: "tick" });
    });
    text(root, sx(0), 14, "○ " + t.steps[b] + "   ● " + t.steps[a], { class: "tick" });
    rows.forEach((r, n) => {
      const y = m.t + n * row + row / 2;
      text(root, m.l - 10, y + 4, r.task.length > 32 ? r.task.slice(0, 31) + "…" : r.task, { "text-anchor": "end", class: "tick" });
      const g = svg("g", { class: "pt" }, root);
      if (r.mean[b] !== null && r.mean[a] !== null) svg("line", { x1: sx(r.mean[b]), x2: sx(r.mean[a]), y1: y, y2: y, class: "conn" }, g);
      if (r.mean[b] !== null) svg("circle", { cx: sx(r.mean[b]), cy: y, r: 4.5, class: "before" }, g);
      if (r.mean[a] !== null) svg("circle", { cx: sx(r.mean[a]), cy: y, r: 4.5, class: "after" }, g);
      svg("rect", { x: m.l, y: y - row / 2, width: W - m.l - m.r, height: row, class: "hit" }, g);
      const d = r.mean[b] !== null && r.mean[a] !== null ? (r.delta >= 0 ? "+" : "") + (100 * r.delta).toFixed(0) + " pts" : "";
      text(root, W - m.r + 10, y + 4, d, { class: "tick" });
      const lines = [["tt", r.task], ["tv", t.steps[b] + ": " + (r.mean[b] === null ? "no scored trial" : pct(r.mean[b]) + " (n=" + r.n[b] + ")")],
        ["tv", t.steps[a] + ": " + (r.mean[a] === null ? "no scored trial" : pct(r.mean[a]) + " (n=" + r.n[a] + ")")]];
      g.addEventListener("mousemove", (event) => showTip(lines, event));
      g.addEventListener("mouseleave", hideTip);
    });
    const wrap = el("div", "jchartwrap");
    wrap.appendChild(root);
    return wrap;
  }

  // ── entry points ───────────────────────────────────────────────────────
  function init(options) {
    enabled = true;
    mode = "browse";
    openRun = options.openRun;
  }

  function startExport(outcomes) {
    enabled = true;
    mode = "export";
    doc = outcomes;
    document.getElementById("content").classList.add("hidden");
    document.getElementById("view-index").classList.add("hidden");
    const app = document.querySelector(".wordmark .app");
    if (app) app.textContent = "job outcomes";
    readURL(new URLSearchParams(location.search));
    if (state.view === "runs") state.view = "outcomes";
    show();
  }

  function hide() {
    ["jobtabs", "view-job"].forEach((id) => document.getElementById(id).classList.add("hidden"));
    hideTip();
  }

  return { addParams, hide, init, readURL, show, startExport, isEnabled: () => enabled };
})();
