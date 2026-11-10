// One rollout as one object, in three panes: the rail lists the runs, the
// centre header carries the run's meta row and a timeline of its events,
// and the side panel holds the verdict: reward, the sentence that says why,
// the checks with their evidence, phase times and usage. A summary opens
// its proof (the failing assertion, the interrupted call, the verifier's
// output) on hover or keyboard focus; Escape closes it. Popovers and the
// timeline tooltip are built on demand, so a long run pays for neither.
// Extends the BF namespace declared in viewer.js.

BF.runcard = (() => {
  const { el, plural } = BF.core;
  const PHASES = [
    ["environment_setup", "Environment setup"],
    ["agent_setup", "Agent setup"],
    ["agent_execution", "Agent"],
    ["verifier", "Verifier"],
  ];
  const TRACE_LINES = 12;
  const LIST_LIMIT = 12;
  let popSequence = 0;
  let openTip = null;
  let lastClose = 0;

  function asRecord(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  // 3.5s, 27s, 14m 36s, 3h 3m: precise where it is short, coarse where long.
  function fmtSpan(seconds) {
    const value = finite(seconds);
    if (value === null) return null;
    if (value < 10) return (value < 0.1 ? "<0.1" : value.toFixed(1).replace(/\.0$/, "")) + "s";
    if (value < 60) return Math.round(value) + "s";
    const total = Math.round(value);
    if (total < 3600) return Math.floor(total / 60) + "m " + (total % 60) + "s";
    return Math.floor(total / 3600) + "h " + Math.floor((total % 3600) / 60) + "m";
  }

  // 950, 12.4k, 33.9M
  function fmtCount(value) {
    const number = finite(value);
    if (number === null) return null;
    if (number >= 1e6) return (number / 1e6).toFixed(1) + "M";
    if (number >= 1e4) return (number / 1e3).toFixed(1) + "k";
    return String(number);
  }

  function fmtReward(value) {
    const number = finite(value);
    if (number === null) return null;
    return number === Math.round(number * 10) / 10 ? number.toFixed(1) : String(number);
  }

  function walk(steps, visit) {
    (Array.isArray(steps) ? steps : []).forEach((step) => {
      visit(step);
      const trace = asRecord(asRecord(step).subagent);
      if (Array.isArray(trace.steps)) walk(trace.steps, visit);
    });
  }

  function eventSteps(steps) {
    const found = [];
    walk(steps, (step) => {
      if (asRecord(step).kind !== "subagent") found.push(asRecord(step));
    });
    return found;
  }

  function shortTest(name) {
    const text = String(name || "");
    return text.split("::").pop() || text;
  }

  function lab(text, cls) {
    return el("span", "lab" + (cls ? " " + cls : ""), text);
  }

  // ── popovers ────────────────────────────────────────────────────────────

  function closeTip(tip, suppress) {
    if (!tip) return;
    tip.pop.classList.remove("open");
    tip.trigger.removeAttribute("data-open");
    if (suppress) tip.trigger.dataset.suppressed = "1";
    if (openTip === tip) {
      openTip = null;
      lastClose = Date.now();
    }
  }

  // Popovers sit in the viewport, beside or under their trigger, so a narrow
  // side panel does not squeeze the evidence.
  function placeTip(tip) {
    const pop = tip.pop;
    const box = tip.trigger.getBoundingClientRect();
    const width = Math.min(520, window.innerWidth - 16);
    pop.style.width = width + "px";
    let left = Math.min(box.right, window.innerWidth - 8) - width;
    if (left < 8) left = Math.max(8, Math.min(box.left, window.innerWidth - width - 8));
    const roomBelow = window.innerHeight - box.bottom;
    const useAbove = roomBelow < pop.offsetHeight + 12 && box.top > pop.offsetHeight + 12;
    pop.style.left = left + "px";
    pop.style.top = (useAbove ? box.top - pop.offsetHeight - 6 : box.bottom + 6) + "px";
    pop.dataset.side = useAbove ? "top" : "bottom";
  }

  function showTip(tip) {
    if (tip.trigger.dataset.suppressed) return;
    if (openTip && openTip !== tip) closeTip(openTip, false);
    if (!tip.built) {
      tip.build(tip.pop);
      tip.built = true;
    }
    // A second popover opened right after another skips the fade.
    tip.pop.classList.toggle("instant", Date.now() - lastClose < 300 || openTip !== null);
    tip.pop.classList.add("open");
    tip.trigger.dataset.open = "1";
    openTip = tip;
    placeTip(tip);
  }

  function attachTip(trigger, build) {
    const pop = el("div", "pop");
    pop.id = "pop-" + (++popSequence);
    pop.setAttribute("role", "tooltip");
    trigger.classList.add("tip");
    trigger.tabIndex = 0;
    trigger.setAttribute("aria-describedby", pop.id);
    document.body.appendChild(pop);
    const tip = { trigger, pop, build, built: false, timer: 0 };
    const enter = () => {
      clearTimeout(tip.timer);
      showTip(tip);
    };
    const leave = () => {
      clearTimeout(tip.timer);
      tip.timer = setTimeout(() => {
        if (document.activeElement !== trigger) closeTip(tip, false);
      }, 120);
    };
    trigger.addEventListener("mouseenter", enter);
    trigger.addEventListener("mouseleave", () => {
      delete trigger.dataset.suppressed;
      leave();
    });
    trigger.addEventListener("focus", enter);
    trigger.addEventListener("blur", () => {
      delete trigger.dataset.suppressed;
      closeTip(tip, false);
    });
    pop.addEventListener("mouseenter", () => clearTimeout(tip.timer));
    pop.addEventListener("mouseleave", leave);
    return tip;
  }

  function clearTips() {
    if (openTip) closeTip(openTip, false);
    document.querySelectorAll("body > .pop, body > .tl-tip").forEach((node) => node.remove());
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && openTip) closeTip(openTip, true);
  });
  window.addEventListener("resize", () => {
    if (openTip) placeTip(openTip);
  });
  document.addEventListener("scroll", () => {
    if (openTip && document.activeElement !== openTip.trigger) closeTip(openTip, false);
  }, true);

  function popHead(pop, title, meta, mono) {
    const head = el("div", "pop-h");
    head.appendChild(el("span", mono ? "pop-title mono" : "pop-title", title));
    if (meta) head.appendChild(el("span", "pop-meta", meta));
    pop.appendChild(head);
  }

  function popRow(pop, label, value, tone) {
    if (value === null || value === undefined || value === "") return;
    const row = el("div", "pop-row");
    row.appendChild(el("span", "pop-label", label));
    row.appendChild(el("span", "pop-value" + (tone ? " " + tone : ""), value));
    pop.appendChild(row);
  }

  // Code lines on a rule: a failure's explanation lines in the fail hue.
  function popCode(pop, lines, failPattern) {
    const block = el("div", "pop-code");
    lines.forEach((line) => {
      const row = el("div", "pop-line" + (failPattern && failPattern.test(line) ? " bad" : ""));
      BF.core.textWithRedaction(row, line || " ");
      block.appendChild(row);
    });
    pop.appendChild(block);
  }

  function popNote(pop, text) {
    pop.appendChild(el("p", "pop-note", text));
  }

  function stepLink(step, onStep, label) {
    const link = el("a", "step-link", label || "#" + step.i);
    link.href = "#e" + encodeURIComponent(step.i);
    link.addEventListener("click", (event) => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      onStep(step.i);
    });
    return link;
  }

  // The failing frame and its explanation: from the ">" line on, else the tail.
  function traceLines(test) {
    const text = String(test.trace || test.message || "");
    const lines = text.replace(/\s+$/, "").split("\n");
    let start = lines.findIndex((line) => /^>\s/.test(line));
    if (start < 0) start = Math.max(0, lines.length - TRACE_LINES);
    return lines.slice(start, start + TRACE_LINES + 4);
  }

  function buildTestPop(test) {
    return (pop) => {
      popHead(pop, String(test.name || "test"), [test.location, fmtSpan(test.seconds)].filter(Boolean).join(" · "), true);
      const lines = traceLines(test);
      if (lines.some((line) => line.trim())) popCode(pop, lines, /^E\s/);
      else popNote(pop, "The test report carries no trace for this failure.");
    };
  }

  function buildToolPop(step, onStep) {
    return (pop) => {
      const tool = asRecord(step.tool);
      const meta = [tool.kind, tool.status, fmtSpan(step.dur)].filter(Boolean).join(" · ");
      popHead(pop, String(tool.title || tool.kind || "tool call"), meta, true);
      const content = Array.isArray(tool.content) ? tool.content : [];
      const first = content.find((item) => String(item).trim());
      if (first) {
        const lines = String(first).split("\n");
        popCode(pop, lines.slice(0, TRACE_LINES).concat(lines.length > TRACE_LINES ? ["…"] : []), null);
      } else {
        popNote(pop, "This call recorded no input or output.");
      }
      const foot = el("p", "pop-note");
      foot.append("Event ", stepLink(step, onStep), " in the trace");
      pop.appendChild(foot);
    };
  }

  // ── state ───────────────────────────────────────────────────────────────

  function stateOf(payload) {
    const meta = asRecord(payload.meta);
    const status = asRecord(meta.status);
    const tone = BF.core.scoreStatus(status.score_outcome, meta.reward);
    const interrupted = status.execution === "timed_out" || status.execution === "errored";
    let label = tone === "pass" ? "Passed" : tone === "fail" ? "Failed" : "Not scored";
    if (status.execution === "timed_out") label = tone === "unscored" ? "Timed out" : label + " · timed out";
    else if (status.execution === "errored") label = tone === "unscored" ? "Errored" : label + " · errored";
    const word = tone === "pass" ? "ok" : tone === "fail" ? "bad" : "warn";
    return { tone, word, interrupted, label, status };
  }

  // ── centre header: title, meta row, timeline ────────────────────────────

  function renderHeader(payload, header, state, onStep) {
    const meta = asRecord(payload.meta);
    const counts = asRecord(meta.counts);
    const timing = asRecord(meta.timing);
    const titleRow = el("div", "titlerow");
    const heading = el("h1", null, meta.task_name || payload.rollout_name || "trajectory");
    heading.tabIndex = -1;
    titleRow.appendChild(heading);
    header.appendChild(titleRow);

    const metaRow = el("div", "metarow");
    const badge = el("span", "badge st " + state.tone + " " + state.word, state.label);
    metaRow.appendChild(badge);
    [
      meta.agent_name,
      meta.model,
      meta.skill_mode,
      typeof counts.tools === "number" ? plural(counts.tools, "tool call") : null,
      fmtSpan(finite(timing.total) ?? meta.duration_sec),
    ].filter(Boolean).forEach((item) => metaRow.appendChild(el("span", "m", item)));
    header.appendChild(metaRow);

    const sub = el("div", "sub");
    const parts = [];
    if (payload.rollout_name) parts.push(payload.rollout_name);
    if (meta.trajectory_source) parts.push("source: " + meta.trajectory_source);
    const branch = asRecord(meta.branch);
    if (branch.node_id) {
      parts.push(
        "branch child " + branch.node_id + " of fork " + String(branch.fork_id || "").slice(0, 8)
          + (branch.parent_node ? " at node " + branch.parent_node : ""),
      );
    }
    if (meta.partial_trajectory) parts.push("partial trajectory");
    parts.forEach((part) => sub.appendChild(el("span", null, part)));
    header.appendChild(sub);

    const timeline = renderTimeline(payload, onStep);
    if (timeline) header.appendChild(timeline);
    return heading;
  }

  // Lanes of the run's events on one axis: wall-clock time when the capture
  // has timestamps, event order when it has none. Each mark is a tool call,
  // a model turn or a prompt; the tooltip names it and its duration.
  function renderTimeline(payload, onStep) {
    const events = eventSteps(payload.steps).filter((step) => typeof step.i === "number");
    if (!events.length) return null;
    const timed = events.filter((step) => finite(step.t) !== null);
    const useTime = timed.length >= Math.max(2, events.length * 0.5);
    let start = 0;
    let end = events.length;
    if (useTime) {
      start = Math.min(...timed.map((step) => step.t));
      end = Math.max(...timed.map((step) => step.t + (finite(step.dur) || 0)));
      if (end <= start) end = start + 1;
    }
    const position = (step, index) => (useTime
      ? ((finite(step.t) ?? start) - start) / (end - start)
      : index / Math.max(1, events.length));
    const width = (step) => (useTime && finite(step.dur) ? step.dur / (end - start) : 0);

    const verdictStep = asRecord(payload.verdict).step;
    const box = el("section", "timeline");
    const head = el("div", "tl-head");
    head.appendChild(lab("Timeline"));
    head.appendChild(el("span", "tl-span", useTime
      ? fmtSpan(end - start) + " · " + plural(events.length, "event")
      : plural(events.length, "event") + " · by order, no timestamps"));
    box.appendChild(head);

    const lanes = [
      ["Prompts", (step) => step.kind === "prompt"],
      ["Model", (step) => step.kind === "message" || step.kind === "thought"],
      ["Tools", (step) => step.kind === "tool" || step.kind === "timeout"],
    ];
    const grid = el("div", "tl-grid" + (events.length > 160 ? " dense" : ""));
    lanes.forEach(([label, keep]) => {
      grid.appendChild(el("span", "tl-label", label));
      const lane = el("div", "tl-lane");
      events.forEach((step, index) => {
        if (!keep(step)) return;
        const tool = asRecord(step.tool);
        const mark = el("span", "tl-mark k-" + step.kind);
        if (tool.status === "failed" || tool.status === "cancelled") mark.classList.add("bad");
        if (tool.status === "pending" || tool.status === "in_progress" || step.kind === "timeout") mark.classList.add("warn");
        if (step.i === verdictStep) mark.classList.add("named");
        mark.style.left = (position(step, index) * 100).toFixed(3) + "%";
        const span = width(step);
        if (span > 0.002) mark.style.width = (span * 100).toFixed(3) + "%";
        mark.dataset.i = step.i;
        lane.appendChild(mark);
      });
      grid.appendChild(lane);
    });
    grid.appendChild(el("span", "tl-label"));
    const axis = el("div", "tl-axis");
    for (let tick = 0; tick <= 4; tick += 1) {
      const label = useTime ? fmtSpan(((end - start) * tick) / 4) || "0s" : "#" + Math.max(1, Math.round((events.length * tick) / 4));
      const mark = el("span", null, tick === 0 && useTime ? "0s" : label);
      mark.style.left = tick * 25 + "%";
      axis.appendChild(mark);
    }
    grid.appendChild(axis);
    box.appendChild(grid);

    const byIndex = new Map(events.map((step) => [String(step.i), step]));
    const tooltip = el("div", "tl-tip");
    tooltip.setAttribute("role", "tooltip");
    document.body.appendChild(tooltip);
    grid.addEventListener("mouseover", (event) => {
      const mark = event.target.closest(".tl-mark");
      if (!mark) return;
      const step = byIndex.get(mark.dataset.i);
      if (!step) return;
      tooltip.textContent = "";
      const tool = asRecord(step.tool);
      const kind = step.kind === "tool" ? (tool.kind || "tool") : step.kind;
      tooltip.appendChild(el("div", "tl-tip-kind", kind + " · #" + step.i));
      const text = step.kind === "tool" ? tool.title : step.kind === "timeout" ? "agent timeout" : step.text || step.label;
      tooltip.appendChild(el("div", "tl-tip-title", String(text || "").split("\n")[0].slice(0, 140)));
      const facts = [tool.status, fmtSpan(step.dur)].filter(Boolean).join(" · ");
      if (facts) tooltip.appendChild(el("div", "tl-tip-meta", facts));
      const rect = mark.getBoundingClientRect();
      tooltip.classList.add("open");
      const left = Math.max(8, Math.min(rect.left, window.innerWidth - tooltip.offsetWidth - 8));
      tooltip.style.left = left + "px";
      tooltip.style.top = rect.bottom + 6 + "px";
    });
    grid.addEventListener("mouseleave", () => tooltip.classList.remove("open"));
    grid.addEventListener("click", (event) => {
      const mark = event.target.closest(".tl-mark");
      if (mark) onStep(Number(mark.dataset.i));
    });
    return box;
  }

  // ── side panel: the verdict and its evidence ────────────────────────────

  function section(side, title, extra) {
    const block = el("section", "side-block");
    const head = el("div", "side-h");
    head.appendChild(lab(title));
    if (extra) head.appendChild(el("span", "side-h-meta", extra));
    block.appendChild(head);
    side.appendChild(block);
    return block;
  }

  function kvRow(parent, label, value, options = {}) {
    if (value === null || value === undefined || value === "") return null;
    const row = el("div", "kv-row" + (options.cls ? " " + options.cls : ""));
    row.appendChild(el("span", "kv-k", label));
    const cell = el("span", "kv-v" + (options.mono ? " mono" : ""));
    if (value instanceof Node) cell.appendChild(value);
    else cell.textContent = value;
    row.appendChild(cell);
    parent.appendChild(row);
    return row;
  }

  function renderSide(payload, side, state, actions) {
    const meta = asRecord(payload.meta);
    const verifier = asRecord(payload.verifier);
    const verdict = asRecord(payload.verdict);
    const timing = asRecord(meta.timing);
    const usage = asRecord(meta.usage);
    const steps = eventSteps(payload.steps);
    const tools = steps.filter((step) => step.kind === "tool");
    const tests = Array.isArray(verifier.ctrf) ? verifier.ctrf.map(asRecord) : [];
    const failing = tests.filter((test) => test.status === "failed");
    const passed = tests.filter((test) => test.status === "passed");
    const onStep = actions.showStep;

    // Verdict: the reward, the state word and the sentence.
    const top = section(side, "Verdict");
    const score = el("div", "score");
    const reward = fmtReward(meta.reward);
    const value = el("span", "score-v " + state.word, reward === null ? "—" : reward);
    score.appendChild(value);
    const scoreMeta = el("span", "score-m");
    scoreMeta.appendChild(el("span", "lab plain", "Reward"));
    scoreMeta.appendChild(el("span", "st " + state.word, state.label));
    score.appendChild(scoreMeta);
    top.appendChild(score);
    attachTip(value, (pop) => {
      const status = state.status;
      popHead(pop, "Reward " + (reward === null ? "not recorded" : reward), status.summary || null, false);
      popRow(pop, "execution", [status.execution ? String(status.execution).replace("_", " ") : null, status.execution_detail].filter(Boolean).join(", "));
      popRow(pop, "assessment", [status.assessment, status.assessment_detail].filter(Boolean).join(", "));
      if (verifier.reward !== null && verifier.reward !== undefined) popRow(pop, "verifier/reward.txt", String(verifier.reward));
    });
    if (verdict.sentence) {
      const why = el("p", "why");
      why.id = "why";
      BF.core.textWithRedaction(why, verdict.sentence);
      top.appendChild(why);
    }
    const namedStep = typeof verdict.step === "number" ? tools.find((step) => step.i === verdict.step) : null;
    let action = null;
    if (failing.length) action = ["Show failing test", () => actions.showTest(failing[0].name)];
    else if (namedStep && state.interrupted) action = ["Show call #" + namedStep.i, () => onStep(namedStep.i)];
    if (action) {
      const button = el("button", "btn rc-action", action[0]);
      button.type = "button";
      button.addEventListener("click", action[1]);
      top.appendChild(button);
    }

    // The call a timeout or error cut off.
    if (namedStep && state.interrupted) {
      const tool = asRecord(namedStep.tool);
      const running = tool.status === "pending" || tool.status === "in_progress";
      const block = section(side, running ? "Interrupted call" : "Last call");
      const row = el("div", "call-row");
      row.appendChild(el("span", "call-title", tool.title || tool.kind || "tool call"));
      row.appendChild(el("span", "call-meta", "#" + namedStep.i + (tool.status ? " · " + tool.status : "")));
      block.appendChild(row);
      attachTip(row, buildToolPop(namedStep, onStep));
    }

    // Checks: one row per test, failures first, each failure one hover
    // from its assertion.
    if (tests.length || verifier.ctrf_tool || finite(timing.verifier) !== null) {
      const runner = [verifier.ctrf_tool, finite(verifier.ctrf_seconds) !== null ? fmtSpan(verifier.ctrf_seconds) : null]
        .filter(Boolean).join(" · ");
      const block = section(side, "Checks", tests.length ? passed.length + " of " + tests.length + " passed" : null);
      const table = el("div", "checks");
      failing.concat(tests.filter((test) => test.status !== "failed")).slice(0, LIST_LIMIT).forEach((test) => {
        const row = el("div", "check-row " + (test.status === "failed" ? "bad" : test.status === "passed" ? "ok" : ""));
        row.appendChild(el("span", "flag " + (test.status === "failed" ? "bad" : test.status === "passed" ? "ok" : ""), test.status || "unknown"));
        row.appendChild(el("span", "check-name", shortTest(test.name)));
        row.appendChild(el("span", "check-t", fmtSpan(test.seconds) || ""));
        table.appendChild(row);
        if (test.status === "failed") attachTip(row, buildTestPop(test));
      });
      if (tests.length > LIST_LIMIT) table.appendChild(el("div", "check-more", (tests.length - LIST_LIMIT) + " more in the Verifier tab"));
      if (tests.length) block.appendChild(table);
      const runnerRow = kvRow(block, "Runner", runner || "not recorded", { mono: true, cls: "runner" });
      if (runnerRow) {
        attachTip(runnerRow, (pop) => {
          popHead(pop, verifier.ctrf_tool || "verifier", fmtSpan(timing.verifier) ? "phase " + fmtSpan(timing.verifier) : null, true);
          if (tests.length) {
            popRow(pop, "tests", plural(tests.length, "test") + (runner ? " in " + fmtSpan(verifier.ctrf_seconds) : ""));
            popRow(pop, "passed", String(passed.length), passed.length ? "ok" : null);
            popRow(pop, "failed", String(failing.length), failing.length ? "bad" : null);
          }
          if (verifier.reward !== null && verifier.reward !== undefined) popRow(pop, "reward.txt", String(verifier.reward));
          const output = String(verifier.stdout || verifier.stderr || "");
          const tail = output.split("\n").map((line) => line.replace(/\s+$/, "")).filter(Boolean).slice(-6);
          if (tail.length) {
            popNote(pop, verifier.stdout ? "Last lines of test-stdout.txt" : "Last lines of test-stderr.txt");
            popCode(pop, tail, /\bfailed\b|Error/);
          } else {
            popNote(pop, "The verifier wrote no output file.");
          }
        });
      }
    }

    // Phases: where the time went.
    const phases = PHASES.filter(([key]) => finite(timing[key]) !== null);
    if (phases.length) {
      const block = section(side, "Phases", fmtSpan(timing.total));
      phases.forEach(([key, label]) => kvRow(block, label, fmtSpan(timing[key]), { cls: "num" }));
    }

    // Agent: the calls by kind, and the ones that failed.
    if (tools.length) {
      const kinds = new Map();
      tools.forEach((step) => {
        const kind = String(asRecord(step.tool).kind || "other");
        kinds.set(kind, (kinds.get(kind) || 0) + 1);
      });
      const block = section(side, "Tool calls", String(tools.length));
      const strip = el("div", "kinds");
      [...kinds.entries()].sort((a, b) => b[1] - a[1]).forEach(([kind, count]) => {
        const item = el("span", "kind");
        item.appendChild(el("span", "kind-k", kind));
        item.appendChild(el("span", "kind-n", String(count)));
        strip.appendChild(item);
      });
      block.appendChild(strip);
      const failedTools = tools.filter((step) => ["failed", "cancelled"].includes(asRecord(step.tool).status));
      failedTools.slice(0, 6).forEach((step) => {
        const row = el("div", "call-row bad");
        row.appendChild(el("span", "call-title", asRecord(step.tool).title || "tool call"));
        const link = el("span", "call-meta");
        link.appendChild(stepLink(step, onStep));
        row.appendChild(link);
        block.appendChild(row);
      });
      if (failedTools.length > 6) block.appendChild(el("div", "check-more", (failedTools.length - 6) + " more failed calls"));
    }

    // Usage.
    const tokenRows = [
      ["Input", usage.n_input_tokens],
      ["Output", usage.n_output_tokens],
      ["Cache read", usage.n_cache_read_tokens],
      ["Total", usage.total_tokens],
    ].filter(([, number]) => finite(number) !== null);
    if (tokenRows.length || finite(usage.cost_usd) !== null) {
      const block = section(side, "Usage", usage.usage_source || null);
      tokenRows.forEach(([label, number]) => kvRow(block, label, fmtCount(number), { cls: "num" }));
      if (finite(usage.cost_usd) !== null) kvRow(block, "Cost", "$" + usage.cost_usd.toFixed(4), { cls: "num" });
    }
  }

  // ── rail: the runs in this view ─────────────────────────────────────────

  function renderRail(items, currentId, onOpen, outline = null) {
    const rail = document.getElementById("rail-list");
    if (!rail) return;
    rail.textContent = "";
    if (!items.length) return;
    renderRuns(rail, items, currentId, onOpen);
    if (outline) renderOutline(rail, outline);
  }

  // The open run's landmarks: its prompts, and the calls that failed or were
  // cut off. Each opens its event in the transcript.
  function renderOutline(rail, outline) {
    const steps = eventSteps(outline.payload.steps);
    const verdictStep = asRecord(outline.payload.verdict).step;
    const marks = steps.filter((step) => {
      const status = asRecord(step.tool).status;
      return step.kind === "prompt" || step.kind === "timeout" || status === "failed" || status === "cancelled" || step.i === verdictStep;
    });
    if (!marks.length) return;
    const head = el("div", "rail-h");
    head.appendChild(lab("Landmarks"));
    rail.appendChild(head);
    marks.slice(0, 60).forEach((step) => {
      const tool = asRecord(step.tool);
      const row = el("button", "rail-row sub");
      row.type = "button";
      const tone = step.kind === "prompt" ? "" : step.i === verdictStep || step.kind === "timeout" ? "warn" : "bad";
      row.appendChild(el("span", "tick " + tone));
      const name = step.kind === "prompt"
        ? String(step.label || "Prompt").replace(/^PROMPT\b/, "Prompt")
        : step.kind === "timeout" ? "Agent timeout" : String(tool.title || tool.kind || "tool call");
      row.appendChild(el("span", "rail-name", name));
      row.appendChild(el("span", "rail-meta", "#" + step.i));
      row.addEventListener("click", () => outline.onStep(step.i));
      rail.appendChild(row);
    });
    if (marks.length > 60) rail.appendChild(el("div", "rail-more", (marks.length - 60) + " more in the transcript"));
  }

  function renderRuns(rail, items, currentId, onOpen) {
    const head = el("div", "rail-h");
    head.appendChild(lab(plural(items.length, "run")));
    rail.appendChild(head);
    items.forEach((item) => {
      const row = el(onOpen ? "button" : "div", "rail-row" + (item.id === currentId ? " on" : ""));
      if (onOpen) {
        row.type = "button";
        row.addEventListener("click", () => onOpen(item.id));
      }
      if (item.id === currentId) row.setAttribute("aria-current", "page");
      row.appendChild(el("span", "dot " + item.tone));
      row.appendChild(el("span", "rail-name", item.name));
      row.appendChild(el("span", "rail-meta", item.meta || ""));
      rail.appendChild(row);
    });
  }

  function render(payload, header, side, actions) {
    clearTips();
    const state = stateOf(payload);
    const heading = renderHeader(payload, header, state, actions.showStep);
    renderSide(payload, side, state, actions);
    return heading;
  }

  return { clearTips, fmtCount, fmtReward, fmtSpan, render, renderRail, stateOf };
})();
