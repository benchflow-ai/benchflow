// Compare, outcomes, Pareto, and branches views (new in the Rust viewer). The server computes
// every number; this script draws the tables and keeps the job selection
// and filters in the query string, so a link reproduces the view. A change
// fetches /api/<view> with the new query and redraws.
// Extends the BF namespace declared in viewer.js.

BF.views = (() => {
  const { el, fmtDuration, isRecord, plural } = BF.core;
  const OUTCOMES = ["passed", "failed", "unscored", "errored", "stopped"];
  const PHASES = [
    ["environment_setup", "environment setup"],
    ["agent_execution", "agent execution"],
    ["verifier", "verifier"],
  ];
  // The engine's owners, worded as `bench` words them.
  const OWNERS = new Map([
    ["task", "the task"],
    ["agent", "the agent"],
    ["model", "the model provider"],
    ["sandbox", "the sandbox"],
    ["you", "your setup"],
    ["benchflow", "BenchFlow (a bug)"],
  ]);
  const SOURCES = new Map([
    ["problem", "problem"],
    ["error", "error (0.8)"],
    ["verifier_error", "verifier_error (0.8)"],
    ["result.json", "result.json"],
  ]);

  let mode = "compare";
  let data = null;
  let request = null;
  let typing = null;
  // runs: the jobs named in the URL (none means every job).
  const state = { runs: [], status: new Set(), task: "", agent: "" };

  // ── formatting ───────────────────────────────────────────
  const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
  const pct = (value) => (isNumber(value) ? (100 * value).toFixed(1) + "%" : "—");
  const reward = (value) => (isNumber(value) ? String(Number(value.toFixed(3))) : "—");
  const signed = (value) => (value > 0 ? "+" : value < 0 ? "−" : "") + Math.abs(value).toFixed(3);
  const seconds = (value) => {
    if (!isNumber(value)) return "—";
    return value < 10 ? value.toFixed(1) + "s" : fmtDuration(value);
  };
  const letter = (index) => (index < 26 ? String.fromCharCode(65 + index) : "#" + (index + 1));
  const ownerText = (owner) => (owner ? OWNERS.get(owner) || owner : "not recorded (0.8)");

  // ── the URL ──────────────────────────────────────────────
  // Job ids are encoded one by one and joined with commas; "/" stays readable.
  const encodeId = (id) => encodeURIComponent(id).replace(/%2F/g, "/");
  const decode = (text) => {
    try { return decodeURIComponent(text.replace(/\+/g, " ")); } catch { return text; }
  };

  function readURL() {
    const raw = new Map();
    location.search.replace(/^\?/, "").split("&").filter(Boolean).forEach((pair) => {
      const at = pair.indexOf("=");
      const key = decode(at < 0 ? pair : pair.slice(0, at));
      const value = at < 0 ? "" : pair.slice(at + 1);
      if (value && !raw.has(key)) raw.set(key, value);
    });
    const list = (key) => (raw.get(key) || "").split(",").map(decode).filter(Boolean);
    state.runs = [...new Set([...list("runs"), ...list("run")])];
    state.status = new Set(list("status").filter((status) => OUTCOMES.includes(status)));
    state.task = decode(raw.get("task") || "");
    state.agent = decode(raw.get("agent") || "");
  }

  function queryFor(view) {
    const parts = [];
    if (state.runs.length) parts.push((view === "outcomes" ? "run" : "runs") + "=" + state.runs.map(encodeId).join(","));
    if (state.status.size && view !== "pareto" && view !== "branches") parts.push("status=" + OUTCOMES.filter((status) => state.status.has(status)).join(","));
    if (state.task) parts.push("task=" + encodeURIComponent(state.task));
    if (state.agent) parts.push("agent=" + encodeURIComponent(state.agent));
    return parts.length ? "?" + parts.join("&") : "";
  }

  // The other view's link carries the same jobs and filters.
  function updateNav() {
    document.querySelectorAll("#bf-views a").forEach((link) => {
      const view = link.getAttribute("href").split("?")[0];
      if (view === "/compare") link.setAttribute("href", "/compare" + queryFor("compare"));
      if (view === "/outcomes") link.setAttribute("href", "/outcomes" + queryFor("outcomes"));
      if (view === "/pareto") link.setAttribute("href", "/pareto" + queryFor("pareto"));
      if (view === "/branches") link.setAttribute("href", "/branches" + queryFor("branches"));
    });
  }

  async function reload() {
    history.replaceState({}, "", location.pathname + queryFor(mode));
    updateNav();
    if (request) request.abort();
    const controller = new AbortController();
    request = controller;
    setStatus("loading…");
    try {
      const response = await fetch("/api/" + mode + queryFor(mode), { signal: controller.signal });
      if (!response.ok) throw new Error("HTTP " + response.status);
      const next = await response.json();
      if (!isRecord(next)) throw new TypeError("the answer is not a JSON object");
      if (controller !== request) return;
      data = next;
      refreshControls();
      renderBody();
    } catch (error) {
      if (controller.signal.aborted) return;
      setStatus("could not load: " + error.message);
    } finally {
      if (controller === request) request = null;
    }
  }

  // ── controls ─────────────────────────────────────────────
  function setStatus(text) {
    const status = document.querySelector("#vcontrols .vstatus");
    if (status) status.textContent = text;
  }

  function selectedJobs() {
    return Array.isArray(data.query && data.query.runs) ? data.query.runs : [];
  }

  function renderControls() {
    const controls = document.getElementById("vcontrols");
    controls.textContent = "";

    const jobs = el("details", "vjobs");
    jobs.appendChild(el("summary"));
    const list = el("div", "vjoblist");
    jobs.appendChild(list);
    controls.appendChild(jobs);

    const statusField = el("div", "vfield");
    statusField.setAttribute("role", "group");
    statusField.setAttribute("aria-label", "status filter");
    statusField.appendChild(el("span", "vlabel", mode === "compare" ? "rows with" : "status"));
    OUTCOMES.forEach((status) => {
      const button = el("button", "vtoggle", status);
      button.type = "button";
      button.dataset.status = status;
      button.addEventListener("click", () => {
        if (state.status.has(status)) state.status.delete(status);
        else state.status.add(status);
        reload();
      });
      statusField.appendChild(button);
    });
    // A pass rate needs every outcome, so the Pareto view has no status
    // filter; the branches view chooses parents, whose status says nothing
    // of their children.
    if (mode !== "pareto" && mode !== "branches") controls.appendChild(statusField);

    const taskField = el("div", "vfield");
    const taskLabel = el("label", "vlabel", "task");
    taskLabel.htmlFor = "vtask";
    const task = el("input");
    task.id = "vtask";
    task.type = "search";
    task.placeholder = "name contains…";
    task.value = state.task;
    task.addEventListener("input", () => {
      state.task = task.value;
      clearTimeout(typing);
      typing = setTimeout(reload, 200);
    });
    taskField.append(taskLabel, task);
    controls.appendChild(taskField);

    const agentField = el("div", "vfield");
    const agentLabel = el("label", "vlabel", "agent");
    agentLabel.htmlFor = "vagent";
    const agent = el("select");
    agent.id = "vagent";
    agent.addEventListener("change", () => {
      state.agent = agent.value;
      reload();
    });
    agentField.append(agentLabel, agent);
    controls.appendChild(agentField);

    controls.appendChild(el("span", "vstatus"));
    refreshControls();
  }

  function refreshControls() {
    const jobs = Array.isArray(data.jobs) ? data.jobs : [];
    const selected = selectedJobs();
    document.querySelector("#vcontrols .vjobs summary").textContent =
      "jobs: " + (state.runs.length ? selected.length + " of " + jobs.length : "all " + jobs.length) + " ▾";
    const list = document.querySelector("#vcontrols .vjoblist");
    list.textContent = "";
    if (jobs.length > 1) {
      const all = el("button", "vtoggle", "all jobs");
      all.type = "button";
      all.setAttribute("aria-pressed", state.runs.length ? "false" : "true");
      all.addEventListener("click", () => {
        state.runs = [];
        reload();
      });
      list.appendChild(all);
    }
    jobs.forEach((job) => {
      const label = el("label");
      const box = el("input");
      box.type = "checkbox";
      box.checked = selected.includes(job.id);
      const order = selected.indexOf(job.id);
      // The last selected job cannot be unchecked: no selection means all.
      box.disabled = box.checked && selected.length === 1;
      box.addEventListener("change", () => {
        const base = state.runs.length ? state.runs.slice() : selected.slice();
        state.runs = box.checked ? [...base.filter((id) => id !== job.id), job.id] : base.filter((id) => id !== job.id);
        reload();
      });
      label.appendChild(box);
      if (mode !== "outcomes" && order >= 0) label.appendChild(el("b", null, String(order + 1)));
      label.appendChild(el("span", "vjobname", job.name));
      label.appendChild(el("span", "vcount", plural(job.trials, "trial")));
      list.appendChild(label);
    });

    document.querySelectorAll("#vcontrols button[data-status]").forEach((button) => {
      button.setAttribute("aria-pressed", state.status.has(button.dataset.status) ? "true" : "false");
    });

    const agent = document.getElementById("vagent");
    agent.textContent = "";
    const agents = Array.isArray(data.agents) ? data.agents.slice() : [];
    if (state.agent && !agents.includes(state.agent)) agents.push(state.agent);
    [["", "all agents"], ...agents.map((name) => [name, name])].forEach(([value, text]) => {
      const option = el("option", null, text);
      option.value = value;
      option.selected = value === state.agent;
      agent.appendChild(option);
    });
    setStatus("");
  }

  // ── shared pieces ────────────────────────────────────────
  function section(title, note) {
    const node = el("section", "vsection");
    node.appendChild(el("h2", null, title));
    if (note) node.appendChild(el("p", "vnote", note));
    return node;
  }

  function table(headings) {
    const wrap = el("div", "vscroll");
    const node = el("table", "vtable");
    const head = el("thead");
    const row = el("tr");
    headings.forEach((heading) => {
      const th = typeof heading === "string" ? el("th", null, heading) : heading;
      th.scope = "col";
      row.appendChild(th);
    });
    head.appendChild(row);
    const body = el("tbody");
    node.append(head, body);
    wrap.appendChild(node);
    return { wrap, body };
  }

  function cell(text, cls) {
    return el("td", cls, text);
  }

  function numTh(text) {
    return el("th", "num", text);
  }

  function chip(trial) {
    const link = el("a", "vchip " + trial.outcome);
    link.href = "/?run=" + encodeURIComponent(trial.id);
    const scored = trial.outcome === "passed" || trial.outcome === "failed";
    link.textContent = scored && isNumber(trial.reward) ? reward(trial.reward) : trial.outcome;
    link.title = trial.outcome + (isNumber(trial.reward) ? ", reward " + reward(trial.reward) : "") + " — " + trial.id;
    return link;
  }

  function warnings(body) {
    const unknown = Array.isArray(data.unknown) ? data.unknown : [];
    if (unknown.length) {
      body.appendChild(el("div", "vwarn", "Not found here: " + unknown.join(", ") + ". The jobs on offer are in the jobs menu."));
    }
    if (data.capped) {
      body.appendChild(el("div", "vwarn", "Only the first trials were read (raise BENCHFLOW_VIEWER_MAX_RUNS to read more)."));
    }
  }

  function countText(count, total) {
    return count + (total ? " (" + pct(count / total) + ")" : "");
  }

  // ── compare ──────────────────────────────────────────────
  function columnLabel(column, index) {
    return letter(index) + " · " + column.job_name + (column.arm ? " · " + column.arm : "");
  }

  function renderCompare(body) {
    const columns = Array.isArray(data.columns) ? data.columns : [];
    const rows = Array.isArray(data.rows) ? data.rows : [];
    const pairs = Array.isArray(data.pairs) ? data.pairs : [];
    if (!columns.length || !rows.length) {
      body.appendChild(el("div", "vempty", "No trials match."));
      return;
    }

    const jobs = section(
      "Pass rate by job",
      "Pass rate counts every trial in the rows shown (errored and unscored trials count as not passed; trials you stopped are left out), with its 95% Wilson interval. "
        + "A job that ran several agents or models gets one column for each.",
    );
    const summary = table(["", "job", "agent · model", numTh("tasks"), numTh("trials"), numTh("passed"), numTh("failed"),
      numTh("unscored"), numTh("errored"), numTh("stopped"), numTh("pass rate (95% CI)"), numTh("mean reward")]);
    columns.forEach((column, index) => {
      const row = el("tr");
      const counts = isRecord(column.counts) ? column.counts : {};
      const interval = Array.isArray(column.pass_interval) ? " (" + pct(column.pass_interval[0]) + "–" + pct(column.pass_interval[1]) + ")" : "";
      row.append(
        cell(letter(index), "vmono"),
        cell(column.job_name + (column.never_ran ? " (" + column.never_ran + " never ran)" : ""), "vtask"),
        cell(column.arm || "—"),
        cell(column.tasks, "num"),
        cell(column.trials, "num"),
        ...OUTCOMES.map((status) => cell(counts[status] ?? 0, "num")),
        cell(pct(column.pass_rate) + interval, "num"),
        cell(reward(column.mean_reward), "num"),
      );
      summary.body.appendChild(row);
    });
    jobs.appendChild(summary.wrap);
    body.appendChild(jobs);

    if (pairs.length) {
      const paired = section(
        "Paired difference",
        "Columns by their letter above. Over tasks with a scored trial on both sides: each task's mean reward on the left minus the right (a task's trials are averaged first), "
          + "with a 95% t-interval (none from fewer than 3 shared tasks or when every task differs by the same amount). Left out: tasks both sides ran that one side did not score. One side only: tasks the other side did not run.",
      );
      const pairTable = table(["left", "right", numTh("shared tasks"), numTh("mean difference (95% CI)"), numTh("left better"),
        numTh("right better"), numTh("tied"), numTh("left out"), numTh("one side only")]);
      pairs.forEach((pair) => {
        const row = el("tr");
        const left = columns[pair.a];
        const right = columns[pair.b];
        let difference = "—";
        let tone = null;
        if (isNumber(pair.mean_difference)) {
          difference = signed(pair.mean_difference);
          if (Array.isArray(pair.interval)) {
            difference += " (" + signed(pair.interval[0]) + " to " + signed(pair.interval[1]) + ")";
            if (pair.interval[0] > 0) tone = "num vpos";
            else if (pair.interval[1] < 0) tone = "num vneg";
          } else if (pair.no_interval) {
            difference += " (no interval: " + pair.no_interval + ")";
          }
        }
        const side = (column, index) => {
          const td = cell(letter(index), "vmono");
          if (column) td.title = columnLabel(column, index);
          return td;
        };
        row.append(
          side(left, pair.a),
          side(right, pair.b),
          cell(pair.shared, "num"),
          cell(difference, tone || "num"),
          cell(pair.better, "num"),
          cell(pair.worse, "num"),
          cell(pair.tied, "num"),
          cell(pair.excluded, "num"),
          cell(pair.one_side, "num"),
        );
        pairTable.body.appendChild(row);
      });
      paired.appendChild(pairTable.wrap);
      body.appendChild(paired);
    }

    const tasks = section(
      "Tasks",
      plural(rows.length, "task") + ". Rows where the columns disagree come first: the widest spread of mean reward, then mixed outcomes. "
        + "Each chip is a trial (its reward, or its status when unscored or errored); click one to open it.",
    );
    const headings = [el("th", null, "task")];
    columns.forEach((column, index) => {
      const th = el("th", "vcol", letter(index) + " · " + column.job_name);
      if (column.arm) th.appendChild(el("span", "varm", column.arm));
      headings.push(th);
    });
    const grid = table(headings);
    rows.forEach((taskRow) => {
      const row = el("tr", taskRow.spread > 0 || taskRow.mixed ? "vdisagree" : null);
      row.appendChild(cell(taskRow.task, "vtask"));
      (Array.isArray(taskRow.cells) ? taskRow.cells : []).forEach((trials) => {
        const td = el("td");
        if (Array.isArray(trials) && trials.length) {
          const box = el("div", "vcell");
          trials.forEach((trial) => box.appendChild(chip(trial)));
          td.appendChild(box);
        } else {
          td.appendChild(el("span", "vmissing", "—"));
        }
        row.appendChild(td);
      });
      grid.body.appendChild(row);
    });
    tasks.appendChild(grid.wrap);
    body.appendChild(tasks);
  }

  // ── outcomes ─────────────────────────────────────────────
  // The jobs with a nonzero count, one per line (only the count for one job).
  function byJobCell(summary, counts) {
    const td = el("td", "vmsg");
    if (summary.length < 2) {
      td.textContent = "\u2014";
      return td;
    }
    const lines = summary.filter((job) => isRecord(counts) && counts[job.job]);
    lines.forEach((job) => td.appendChild(el("span", "vsub", job.name + ": " + counts[job.job])));
    if (!lines.length) td.textContent = "\u2014";
    return td;
  }

  function renderOutcomes(body) {
    const summary = Array.isArray(data.summary) ? data.summary : [];
    const problems = Array.isArray(data.problems) ? data.problems : [];
    const owners = Array.isArray(data.owners) ? data.owners : [];
    const timing = Array.isArray(data.timing) ? data.timing : [];
    if (!summary.some((job) => job.trials > 0)) {
      body.appendChild(el("div", "vempty", "No trials match."));
      return;
    }

    const outcomes = section("Outcomes", "How each trial ended. Unscored: the agent ran but no verdict was recorded. Errored: the trial could not finish.");
    const legend = el("div", "vlegend");
    OUTCOMES.forEach((status) => legend.appendChild(el("span", status, status)));
    outcomes.appendChild(legend);
    const counts = table(["job", numTh("trials"), "", ...OUTCOMES.map(numTh)]);
    summary.forEach((job) => {
      const row = el("tr");
      const jobCounts = isRecord(job.counts) ? job.counts : {};
      const bar = el("div", "vbar");
      OUTCOMES.forEach((status) => {
        const count = jobCounts[status] || 0;
        if (!count || !job.trials) return;
        const part = el("span", status);
        part.style.width = (100 * count / job.trials) + "%";
        part.title = status + ": " + count;
        bar.appendChild(part);
      });
      const barCell = el("td");
      barCell.appendChild(bar);
      row.append(cell(job.name, "vtask"), cell(job.trials, "num"), barCell,
        ...OUTCOMES.map((status) => cell(countText(jobCounts[status] || 0, job.trials), "num")));
      counts.body.appendChild(row);
    });
    outcomes.appendChild(counts.wrap);
    body.appendChild(outcomes);

    const byOwner = section(
      "Problems by owner",
      problems.length
        ? "Who should act, from each trial's problem. 0.8 trials record an error and its category but no owner."
        : "No trial recorded a problem.",
    );
    if (owners.length) {
      const ownerTable = table(["owner", numTh("trials"), "by job"]);
      owners.forEach((owner) => {
        const row = el("tr");
        row.append(cell(ownerText(owner.owner), "vtask"), cell(owner.total, "num"), byJobCell(summary, owner.by_job));
        ownerTable.body.appendChild(row);
      });
      byOwner.appendChild(ownerTable.wrap);
    }
    body.appendChild(byOwner);

    if (problems.length) {
      const byCode = section(
        "Problems by code",
        "Grouped by code and owner. For 0.8 trials the code is the error's category (error_category or verifier_error_category, or the viewer's classification of the text). "
          + "The message and next step shown are the most common in the group.",
      );
      const codeTable = table(["code", "owner", "recorded in", "message", "next step", numTh("trials"), "by job", "trials"]);
      problems.forEach((problem) => {
        const row = el("tr");
        const trials = el("td");
        const list = Array.isArray(problem.trials) ? problem.trials : [];
        const details = el("details", "vtrials");
        details.appendChild(el("summary", null, plural(list.length, "trial")));
        const box = el("div", "vcell");
        list.forEach((trial) => {
          const link = chip(trial);
          link.textContent = trial.task;
          box.appendChild(link);
        });
        details.appendChild(box);
        trials.appendChild(details);
        row.append(
          cell(problem.code, "vmono"),
          cell(ownerText(problem.owner)),
          cell(SOURCES.get(problem.source) || problem.source),
          cell(problem.message || "—", "vmsg"),
          cell(problem.next || "—", "vmsg"),
          cell(problem.total, "num"),
          byJobCell(summary, problem.by_job),
          trials,
        );
        codeTable.body.appendChild(row);
      });
      byCode.appendChild(codeTable.wrap);
      body.appendChild(byCode);
    }

    const time = section(
      "Time per phase",
      "Median and 90th percentile in seconds, from timing.json (or the timing in result.json), over the trials that recorded the phase. "
        + "The 90th percentile interpolates between trials.",
    );
    const headings = ["job"];
    PHASES.forEach(([, name]) => headings.push(numTh(name + " median"), numTh("p90"), numTh("n")));
    const timeTable = table(headings);
    timing.forEach((job) => {
      const row = el("tr");
      row.appendChild(cell(job.name, "vtask"));
      PHASES.forEach(([key]) => {
        const phase = isRecord(job.phases) && isRecord(job.phases[key]) ? job.phases[key] : {};
        row.append(cell(seconds(phase.median), "num"), cell(seconds(phase.p90), "num"), cell(phase.n ?? 0, "num"));
      });
      timeTable.body.appendChild(row);
    });
    time.appendChild(timeTable.wrap);
    body.appendChild(time);
  }

  // ── pareto ───────────────────────────────────────────────
  const SVG = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs) => {
    const node = document.createElementNS(SVG, tag);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, String(value)));
    return node;
  };
  // $0.0347, $0.04, $0.10, $1.25
  const usd = (value) => (isNumber(value) ? "$" + (value < 0.1 ? value.toFixed(4).replace(/0{1,2}$/, "") : value.toFixed(2)) : "—");
  const tokens = (value) => {
    if (!isNumber(value)) return "—";
    if (value >= 1e6) return (value / 1e6).toFixed(1) + "M";
    if (value >= 1e3) return (value / 1e3).toFixed(1) + "k";
    return String(Math.round(value));
  };
  // Each axis: its key, the chart's title, the x label, a formatter, and
  // what the axis is when no job recorded it.
  const AXES = [
    ["cost_per_trial", "Pass rate against cost per trial", "mean cost of a trial's model calls (agent_result.cost_usd)", usd,
      "No trial here recorded a cost (agent_result.cost_usd), so there is no cost frontier."],
    ["median_seconds", "Pass rate against time", "median time of a trial, start to finish (timing.json's total)", seconds,
      "No trial here recorded its time (timing.json's total)."],
    ["tokens_per_trial", "Pass rate against tokens per trial", "mean input plus output tokens of a trial (agent_result.total_tokens)", tokens,
      "No trial here recorded token usage."],
  ];

  // Round numbers for an axis covering `low` to `high` (never below 0).
  function ticks(low, high) {
    if (!(high > low)) {
      const pad = Math.abs(high) * 0.1 || 1;
      low = Math.max(0, low - pad);
      high += pad;
    }
    const raw = (high - low) / 4;
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((m) => m >= raw);
    const out = [];
    for (let value = Math.floor(low / step) * step; value <= high + step * 1e-9; value += step) out.push(value);
    if (out[out.length - 1] < high) out.push(out[out.length - 1] + step);
    return out;
  }

  function paretoChart(points, key, xLabel, format) {
    const width = 720, height = 320;
    const pad = { left: 52, right: 24, top: 24, bottom: 46 };
    const plotted = points.map((point, index) => ({ point, index })).filter(({ point }) => isNumber(point[key]));
    const values = plotted.map(({ point }) => point[key]);
    const xTicks = ticks(Math.min(...values), Math.max(...values));
    const xMin = xTicks[0];
    const xMax = xTicks[xTicks.length - 1];
    const x = (value) => pad.left + ((value - xMin) / (xMax - xMin)) * (width - pad.left - pad.right);
    const y = (rate) => pad.top + (1 - rate) * (height - pad.top - pad.bottom);
    const root = svg("svg", { viewBox: "0 0 " + width + " " + height, class: "vchart", role: "img",
      "aria-label": "pass rate (vertical) against " + xLabel + " (horizontal), one point per job" });
    [0, 0.25, 0.5, 0.75, 1].forEach((rate) => {
      root.appendChild(svg("line", { x1: pad.left, x2: width - pad.right, y1: y(rate), y2: y(rate), class: "vgrid" }));
      const label = svg("text", { x: pad.left - 8, y: y(rate) + 4, class: "vtick", "text-anchor": "end" });
      label.textContent = Math.round(rate * 100) + "%";
      root.appendChild(label);
    });
    xTicks.forEach((value) => {
      root.appendChild(svg("line", { x1: x(value), x2: x(value), y1: height - pad.bottom, y2: height - pad.bottom + 4, class: "vaxis" }));
      const label = svg("text", { x: x(value), y: height - pad.bottom + 17, class: "vtick", "text-anchor": "middle" });
      label.textContent = format(value);
      root.appendChild(label);
    });
    root.appendChild(svg("line", { x1: pad.left, x2: width - pad.right, y1: height - pad.bottom, y2: height - pad.bottom, class: "vaxis" }));
    const xTitle = svg("text", { x: (pad.left + width - pad.right) / 2, y: height - 8, class: "vaxistitle", "text-anchor": "middle" });
    xTitle.textContent = xLabel + " →";
    root.appendChild(xTitle);
    const yTitle = svg("text", { x: 12, y: (pad.top + height - pad.bottom) / 2, class: "vaxistitle", "text-anchor": "middle",
      transform: "rotate(-90 12 " + (pad.top + height - pad.bottom) / 2 + ")" });
    yTitle.textContent = "pass rate ↑";
    root.appendChild(yTitle);
    // The frontier as a staircase: up and to the right, best first.
    const front = plotted.filter(({ point }) => point.frontier && point.frontier[key])
      .sort((a, b) => a.point[key] - b.point[key] || a.point.pass_rate - b.point.pass_rate);
    if (front.length > 1) {
      let path = "M" + x(front[0].point[key]) + "," + y(front[0].point.pass_rate);
      front.slice(1).forEach(({ point }) => {
        path += " H" + x(point[key]) + " V" + y(point.pass_rate);
      });
      root.appendChild(svg("path", { d: path, class: "vfront" }));
    }
    // Each letter goes to the first spot near its point that no other
    // letter or point takes.
    const taken = plotted.map(({ point }) => ({ x: x(point[key]) - 6, y: y(point.pass_rate) - 6, w: 12, h: 12 }));
    const overlaps = (box) => taken.some((other) => box.x < other.x + other.w && other.x < box.x + box.w
      && box.y < other.y + other.h && other.y < box.y + box.h);
    const place = (cx, cy, text) => {
      const w = 7.5 * text.length;
      const spots = [[8, -16], [8, 4], [-8 - w, -16], [-8 - w, 4], [8, -30], [8, 18], [-8 - w, -30], [-8 - w, 18]];
      const spot = spots.map(([dx, dy]) => ({ x: cx + dx, y: cy + dy, w, h: 12 })).find((box) => box.y >= 0 && box.x >= 0 && box.x + box.w <= width && !overlaps(box))
        || { x: cx + 8, y: cy - 16, w, h: 12 };
      taken.push(spot);
      return spot;
    };
    // The frontier is drawn last, so another job at the same spot never hides it.
    const onFront = ({ point }) => !!(point.frontier && point.frontier[key]);
    [...plotted.filter((p) => !onFront(p)), ...plotted.filter(onFront)].forEach(({ point, index }) => {
      const cx = x(point[key]);
      const cy = y(point.pass_rate);
      const on = !!(point.frontier && point.frontier[key]);
      const group = svg("g", { class: "vpoint" + (on ? " on" : "") });
      if (Array.isArray(point.pass_interval)) {
        group.appendChild(svg("line", { x1: cx, x2: cx, y1: y(point.pass_interval[1]), y2: y(point.pass_interval[0]), class: "vci" }));
      }
      group.appendChild(svg("circle", { cx, cy, r: 5, class: "vdot" }));
      const spot = place(cx, cy, letter(index));
      const label = svg("text", { x: spot.x, y: spot.y + 10, class: "vlabel" });
      label.textContent = letter(index);
      group.appendChild(label);
      const title = svg("title");
      title.textContent = letter(index) + " · " + point.job_name + (point.arm ? " · " + point.arm : "")
        + "\npass rate " + pct(point.pass_rate)
        + (Array.isArray(point.pass_interval) ? " (" + pct(point.pass_interval[0]) + "–" + pct(point.pass_interval[1]) + ")" : "")
        + ", " + point.passed + " of " + plural(point.trials, "trial")
        + "\n" + xLabel.split(" (")[0] + ": " + format(point[key]) + " over " + plural(point[key + "_n"], "trial")
        + (on ? "\non the frontier" : "");
      group.appendChild(title);
      group.appendChild(svg("circle", { cx, cy, r: 14, class: "vhit" }));
      root.appendChild(group);
    });
    return root;
  }

  function renderPareto(body) {
    const points = Array.isArray(data.points) ? data.points.filter((point) => point.trials > 0) : [];
    if (!points.length) {
      body.appendChild(el("div", "vempty", "No trials match."));
      return;
    }
    if (data.same_tasks === false) {
      body.appendChild(el("div", "vwarn", "These jobs did not all run the same tasks, so their pass rates do not compare directly. "
        + "Pick jobs that ran the same tasks in the jobs menu, or narrow them with the task filter."));
    }
    const legend = el("div", "vlegend vpareto-legend");
    legend.append(el("span", "on", "on the frontier: no other job passes as often for less"),
      el("span", "off", "not on it"), el("span", "ci", "95% Wilson interval of the pass rate"));
    AXES.forEach(([key, title, xLabel, format, none]) => {
      const node = section(title, null);
      if (!points.some((point) => isNumber(point[key]))) {
        node.appendChild(el("p", "vnote", none));
      } else {
        const missing = points.filter((point) => !isNumber(point[key])).map((point) => letter(points.indexOf(point)));
        node.appendChild(legend.cloneNode(true));
        const box = el("div", "vscroll vchartbox");
        box.appendChild(paretoChart(points, key, xLabel, format));
        node.appendChild(box);
        if (missing.length) node.appendChild(el("p", "vnote", "Not plotted (none of their trials recorded it): " + missing.join(", ") + "."));
      }
      body.appendChild(node);
    });
    const listing = section(
      "Jobs",
      "One point per job, or per agent and model when a job ran several. Pass rate counts every trial (errored and unscored trials count as not passed). "
        + "Cost and tokens are means, time is the median, each over the trials that recorded it (n). The horizontal axes start near the smallest value, not at zero. Frontier: the axes on which no other point passes at least as often for no more, with one of the two strictly better.",
    );
    const list = table(["", "job", "agent · model", numTh("trials"), numTh("pass rate (95% CI)"),
      numTh("cost / trial (n)"), numTh("median time (n)"), numTh("tokens / trial (n)"), "frontier"]);
    const withN = (text, n) => text + (n ? " (" + n + ")" : "");
    points.forEach((point, index) => {
      const row = el("tr");
      const ci = Array.isArray(point.pass_interval) ? " (" + pct(point.pass_interval[0]) + "–" + pct(point.pass_interval[1]) + ")" : "";
      const on = AXES.filter(([key]) => point.frontier && point.frontier[key]).map(([key]) => ({
        cost_per_trial: "cost", median_seconds: "time", tokens_per_trial: "tokens" })[key]);
      row.append(
        cell(letter(index), "vmono"),
        cell(point.job_name, "vtask"),
        cell(point.arm || "—"),
        cell(point.trials, "num"),
        cell(pct(point.pass_rate) + ci, "num"),
        cell(withN(usd(point.cost_per_trial), point.cost_per_trial_n), "num"),
        cell(withN(seconds(point.median_seconds), point.median_seconds_n), "num"),
        cell(withN(tokens(point.tokens_per_trial), point.tokens_per_trial_n), "num"),
        cell(on.length ? on.join(", ") : "—", on.length ? "vfrontier" : null),
      );
      list.body.appendChild(row);
    });
    listing.appendChild(list.wrap);
    body.appendChild(listing);
  }

  // ── branches ─────────────────────────────────────────────
  function renderBranches(body) {
    const forks = Array.isArray(data.forks) ? data.forks : [];
    const totals = isRecord(data.totals) ? data.totals : {};
    const listing = section(
      "Forks",
      plural(forks.length, "fork") + ", " + plural(totals.children || 0, "child", "children") + " (" + (totals.scored || 0) + " scored). "
        + "Each fork ran its parent to the fork point, stored a checkpoint, and graded each child from it. The value is the mean of the children's rewards, "
        + "only when every child was scored; a child's advantage is its reward minus that value. A continuing parent's reward is its own, never in the value. "
        + "A child's label opens its trajectory.",
    );
    if (!forks.length) {
      listing.appendChild(el("p", "vnote", "No trial in the selected jobs has a tree.json (run `bench branch` to make one)."));
      body.appendChild(listing);
      return;
    }
    const list = table(["trial · fork", "child", "status", numTh("reward"), numTh("advantage"), "note"]);
    forks.forEach((fork) => {
      const head = el("tr", "vgroup");
      const name = el("td");
      name.colSpan = 6;
      const link = el("a", "vtask", fork.task || fork.run);
      link.href = "/?run=" + encodeURIComponent(fork.run);
      name.appendChild(link);
      if (fork.error && !fork.fork_id) {
        name.appendChild(el("span", "vnote", "  tree.json unreadable: " + fork.error));
        head.appendChild(name);
        list.body.appendChild(head);
        return;
      }
      const parts = [
        fork.fork_point || "?",
        fork.status || "?",
        "value " + reward(fork.value) + (isNumber(fork.value_stderr) ? " ± " + reward(fork.value_stderr) : ""),
        "checkpoint " + (fork.provider || "none") + (isNumber(fork.checkpoint_seconds) ? " " + seconds(fork.checkpoint_seconds) : "") + (fork.retention ? ", " + fork.retention : ""),
        fork.agent_session ? "session " + fork.agent_session : null,
        (fork.parent_restore === "skipped" ? "parent discarded"
          : fork.parent_restore === "not_attempted" ? "parent never forked"
          : fork.parent_restore === "not_needed" || fork.parent_restore === "restored" ? "parent kept"
          : "parent " + (fork.parent_restore || "unknown"))
          + (isNumber(fork.parent_reward) ? ", reward " + reward(fork.parent_reward) : ""),
        fork.arm || null,
      ].filter(Boolean);
      name.appendChild(el("span", "vnote", "  " + parts.join(" · ")));
      if (fork.error) name.appendChild(el("div", "vwarn", fork.error));
      head.appendChild(name);
      list.body.appendChild(head);
      (Array.isArray(fork.children) ? fork.children : []).forEach((child) => {
        const row = el("tr");
        const label = (child.label || child.node_id || "?") + (child.own_prompt ? " (own prompt)" : "");
        let who = cell(label);
        if (child.ref) {
          who = el("td");
          const open = el("a", null, label);
          open.href = "/?run=" + encodeURIComponent(fork.run) + "&branch=" + encodeURIComponent(child.ref);
          who.appendChild(open);
        }
        const adv = isNumber(child.advantage) ? (child.advantage === 0 ? "0.000" : signed(child.advantage)) : "—";
        row.append(
          cell(child.node_id || "", "vmono"),
          who,
          cell(child.status || "—"),
          cell(reward(child.reward), "num"),
          cell(adv, "num"),
          cell(child.error || ""),
        );
        list.body.appendChild(row);
      });
    });
    listing.appendChild(list.wrap);
    body.appendChild(listing);
  }

  function renderBody() {
    const body = document.getElementById("vbody");
    body.textContent = "";
    warnings(body);
    if (mode === "compare") renderCompare(body);
    else if (mode === "pareto") renderPareto(body);
    else if (mode === "branches") renderBranches(body);
    else renderOutcomes(body);
  }

  function start(boot) {
    mode = ["outcomes", "pareto", "branches"].includes(boot.mode) ? boot.mode : "compare";
    data = boot;
    readURL();
    renderControls();
    updateNav();
    renderBody();
  }

  return { start };
})();

(() => {
  const node = document.getElementById("bf-payload");
  let boot = null;
  try {
    boot = JSON.parse(node ? node.textContent : "");
  } catch (error) {
    boot = null;
  }
  BF.theme.init();
  if (!BF.core.isRecord(boot)) {
    document.getElementById("vbody").textContent = "The embedded view data is not a JSON object.";
    return;
  }
  BF.views.start(boot);
})();
