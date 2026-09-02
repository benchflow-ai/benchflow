BF.catalog = (() => {
  const { el, fmtDuration, fmtTokens } = BF.core;
  const PAGE_SIZE = 100;
  const GROUPERS = new Map([
    ["task", (run) => run.task_name || "(unknown task)"],
    ["agent", (run) => [run.agent_name, run.model].filter(Boolean).join(" \u00b7 ") || "(unknown agent)"],
    ["none", () => ""],
  ]);
  const SORTS = new Map([
    ["name", (left, right) => String((left.task_name || "") + (left.name || "")).localeCompare(String((right.task_name || "") + (right.name || "")))],
    ["reward", (left, right) => (right.reward ?? -1) - (left.reward ?? -1)],
    ["duration", (left, right) => (right.duration_sec ?? -1) - (left.duration_sec ?? -1)],
    ["cost", (left, right) => (right.cost_usd ?? -1) - (left.cost_usd ?? -1)],
  ]);

  let rollouts = [];
  let capped = false;
  let onSelectRun = null;
  const state = {
    group: "task",
    sort: "name",
    query: "",
    toggled: new Set(),
    pages: new Map(),
    scrollY: 0,
  };

  function init(boot, selectRun) {
    rollouts = boot.rollouts.slice();
    capped = Boolean(boot.capped);
    onSelectRun = selectRun;
  }

  function runStatus(run) {
    if (run.reward === null || run.reward === undefined) return "unscored";
    return run.reward >= 1 ? "pass" : "fail";
  }

  function fmtCost(value) {
    return typeof value === "number" && Number.isFinite(value) ? "$" + value.toFixed(2) : null;
  }

  function readURL() {
    const params = new URLSearchParams(location.search);
    const group = params.get("group");
    const sort = params.get("sort");
    state.group = GROUPERS.has(group) ? group : "task";
    state.sort = SORTS.has(sort) ? sort : "name";
    state.query = params.get("q") || "";
    const decode = (value) => {
      try { return decodeURIComponent(value); } catch { return value; }
    };
    state.toggled = new Set((params.get("toggled") || "").split(",").filter(Boolean).map(decode));
    return params.get("run");
  }

  function writeURL(runId, push) {
    const params = new URLSearchParams();
    if (state.group !== "task") params.set("group", state.group);
    if (state.sort !== "name") params.set("sort", state.sort);
    if (state.query) params.set("q", state.query);
    if (state.toggled.size) {
      params.set("toggled", [...state.toggled].map(encodeURIComponent).join(","));
    }
    if (runId) params.set("run", runId);
    const url = location.pathname + (params.toString() ? "?" + params.toString() : "");
    history[push ? "pushState" : "replaceState"]({}, "", url);
  }

  function matchesQuery(run, query) {
    if (!query) return true;
    return [run.task_name, run.name, run.agent_name, run.model, run.skill_mode, runStatus(run)]
      .filter(Boolean)
      .join(" ")
      .toLowerCase()
      .includes(query);
  }

  function appendSelect(container, id, labelText, options, value, onChange) {
    const label = el("label", null, labelText);
    label.htmlFor = id;
    const select = el("select");
    select.id = id;
    options.forEach(([optionValue, text]) => {
      const option = el("option", null, text);
      option.value = optionValue;
      option.selected = optionValue === value;
      select.appendChild(option);
    });
    select.addEventListener("change", () => {
      onChange(select.value);
      writeURL(null, false);
      renderRuns();
    });
    container.append(label, select);
  }

  function renderIndex() {
    const main = document.getElementById("view-index");
    main.textContent = "";
    const stats = el("div");
    stats.id = "ixstats";
    main.appendChild(stats);

    const controls = el("div");
    controls.id = "ixcontrols";
    appendSelect(
      controls,
      "ixgroup",
      "group",
      [["task", "task"], ["agent", "model + harness"], ["none", "none"]],
      state.group,
      (value) => {
        state.group = value;
        state.pages.clear();
      },
    );
    appendSelect(
      controls,
      "ixsort",
      "sort",
      [["name", "name"], ["reward", "reward"], ["duration", "duration"], ["cost", "cost"]],
      state.sort,
      (value) => { state.sort = value; },
    );
    const searchLabel = el("label", "visually-hidden", "Filter runs");
    searchLabel.htmlFor = "ixsearch";
    const search = el("input");
    search.id = "ixsearch";
    search.type = "search";
    search.placeholder = "filter runs...";
    search.value = state.query;
    search.addEventListener("input", () => {
      state.query = search.value;
      state.pages.clear();
      writeURL(null, false);
      renderRuns();
    });
    controls.append(searchLabel, search);
    main.appendChild(controls);

    const list = el("div");
    list.id = "ixruns";
    main.appendChild(list);
    renderRuns();
  }

  function renderStats(rows) {
    const stats = document.getElementById("ixstats");
    stats.textContent = "";
    const counts = new Map([["pass", 0], ["fail", 0], ["unscored", 0]]);
    rows.forEach((run) => counts.set(runStatus(run), counts.get(runStatus(run)) + 1));
    const total = el("span");
    total.appendChild(el("b", null, rows.length));
    total.appendChild(document.createTextNode(
      (rows.length === rollouts.length ? "" : " / " + rollouts.length)
        + " runs"
        + (capped ? " (capped - raise BENCHFLOW_VIEWER_MAX_RUNS)" : ""),
    ));
    stats.appendChild(total);
    ["pass", "fail", "unscored"].forEach((status) => {
      const item = el("span", null, status + " ");
      item.appendChild(el("b", null, counts.get(status)));
      stats.appendChild(item);
    });
  }

  function focusGroup(key) {
    const button = [...document.querySelectorAll(".group-head")]
      .find((candidate) => candidate.dataset.groupKey === key);
    if (button) button.focus({ preventScroll: true });
  }

  function renderRuns(groupToFocus = null) {
    const list = document.getElementById("ixruns");
    if (!list) return;
    list.textContent = "";
    const query = state.query.trim().toLowerCase();
    const rows = rollouts.filter((run) => matchesQuery(run, query));
    renderStats(rows);
    if (!rows.length) {
      const empty = el("div", null, rollouts.length ? "No runs match the filter." : "No runs found.");
      empty.id = "ixempty";
      list.appendChild(empty);
      return;
    }

    const grouper = GROUPERS.get(state.group);
    const groups = new Map();
    rows.forEach((run) => {
      const key = grouper(run);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(run);
    });
    const keys = [...groups.keys()].sort((left, right) => left.localeCompare(right));
    const defaultOpen = new Set(rollouts.map(grouper)).size <= 2;
    const filtering = query !== "";

    keys.forEach((key, groupIndex) => {
      const members = groups.get(key).slice().sort(SORTS.get(state.sort));
      const section = el("section", "group");
      const body = el("div", "group-body");
      body.id = "group-body-" + groupIndex;
      const open = state.group === "none" || filtering || defaultOpen !== state.toggled.has(key);

      if (state.group !== "none") {
        const heading = el("button", "group-head");
        heading.type = "button";
        heading.dataset.groupKey = key;
        heading.setAttribute("aria-expanded", open ? "true" : "false");
        heading.setAttribute("aria-controls", body.id);
        heading.appendChild(el("span", "chev", open ? "\u25be" : "\u25b8"));
        heading.appendChild(el("span", "gname", key));
        const groupCounts = new Map([["pass", 0], ["fail", 0], ["unscored", 0]]);
        members.forEach((run) => groupCounts.set(runStatus(run), groupCounts.get(runStatus(run)) + 1));
        const summary = el("span", "gstats");
        summary.appendChild(el("span", null, members.length + " runs"));
        if (groupCounts.get("pass")) summary.appendChild(el("span", "gpass", groupCounts.get("pass") + " pass"));
        if (groupCounts.get("fail")) summary.appendChild(el("span", "gfail", groupCounts.get("fail") + " fail"));
        if (groupCounts.get("unscored")) summary.appendChild(el("span", null, groupCounts.get("unscored") + " unscored"));
        const scored = groupCounts.get("pass") + groupCounts.get("fail");
        if (scored) summary.appendChild(el("span", null, Math.round(100 * groupCounts.get("pass") / scored) + "%"));
        heading.appendChild(summary);
        heading.addEventListener("click", () => {
          if (state.toggled.has(key)) state.toggled.delete(key);
          else state.toggled.add(key);
          writeURL(null, false);
          renderRuns(key);
        });
        section.appendChild(heading);
      }

      if (open) {
        const extra = state.pages.get(key) || 0;
        const limit = PAGE_SIZE + extra;
        members.slice(0, limit).forEach((run) => body.appendChild(runRow(run)));
        if (members.length > limit) {
          const remaining = members.length - limit;
          const more = el(
            "button",
            "showmore",
            "Show " + Math.min(PAGE_SIZE * 2, remaining) + " more (" + remaining + " hidden)",
          );
          more.type = "button";
          more.addEventListener("click", () => {
            state.pages.set(key, extra + PAGE_SIZE * 2);
            renderRuns();
          });
          body.appendChild(more);
        }
      }
      section.appendChild(body);
      list.appendChild(section);
    });
    if (groupToFocus !== null) focusGroup(groupToFocus);
  }

  function runRow(run) {
    const status = runStatus(run);
    const button = el("button", "runrow");
    button.type = "button";
    button.dataset.runId = run.id;
    button.setAttribute("aria-label", "Open " + (run.task_name || run.name || run.id));
    button.appendChild(el("span", "dot " + status));
    const main = el("span", "rmain");
    main.appendChild(el("span", "rtitle", state.group === "task" ? run.name : run.task_name));
    const subtitle = el("span", "rsub");
    if (state.group !== "agent") {
      if (run.agent_name) subtitle.appendChild(el("span", null, run.agent_name));
      if (run.model) subtitle.appendChild(el("span", null, run.model));
    }
    if (run.skill_mode) subtitle.appendChild(el("span", null, run.skill_mode));
    if (run.has_error) subtitle.appendChild(el("span", null, "error"));
    main.appendChild(subtitle);
    button.appendChild(main);

    const stats = el("span", "rstats");
    stats.appendChild(el(
      "span",
      "rreward " + status,
      status === "unscored" ? "\u2014" : (status === "pass" ? "pass" : "fail " + run.reward),
    ));
    const duration = fmtDuration(run.duration_sec);
    if (duration) stats.appendChild(el("span", null, duration));
    const tokens = fmtTokens(run.total_tokens);
    if (tokens) stats.appendChild(el("span", null, tokens + " tok"));
    const cost = fmtCost(run.cost_usd);
    if (cost) stats.appendChild(el("span", null, cost));
    button.appendChild(stats);
    button.addEventListener("click", () => onSelectRun(run.id, button));
    return button;
  }

  function show(options = {}) {
    document.getElementById("content").classList.add("hidden");
    const main = document.getElementById("view-index");
    main.classList.remove("hidden");
    renderIndex();
    window.scrollTo(0, state.scrollY);
    if (options.focusRun) {
      requestAnimationFrame(() => {
        const button = [...document.querySelectorAll(".runrow")]
          .find((candidate) => candidate.dataset.runId === options.focusRun);
        if (button) button.focus({ preventScroll: true });
      });
    }
  }

  function rememberScroll() {
    state.scrollY = window.scrollY;
  }

  function hasRun(runId) {
    return rollouts.some((run) => run.id === runId);
  }

  function unknownRunMessage(runId) {
    return 'Run "' + runId + '" is not among the ' + rollouts.length + " discovered runs"
      + (capped ? " (list is capped - raise BENCHFLOW_VIEWER_MAX_RUNS)." : ".");
  }

  return {
    hasRun,
    init,
    readURL,
    rememberScroll,
    show,
    unknownRunMessage,
    writeURL,
  };
})();
