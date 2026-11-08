// The run card: one rollout as one object. The phase rows (setup, agent,
// verifier) carry their durations; the checks under them carry the test
// counts, reward and total time. Every summary on the card opens its
// evidence (the failing assertion, the tool call, the verifier output) on
// hover or keyboard focus; Escape closes it. Popovers are built on first
// open, so a long run pays nothing for evidence nobody looks at.
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

  function toolSteps(steps) {
    const found = [];
    walk(steps, (step) => {
      if (asRecord(step).kind === "tool") found.push(step);
    });
    return found;
  }

  function shortTest(name) {
    const text = String(name || "");
    return text.split("::").pop() || text;
  }

  // Line icons (12px, currentColor) for the state marks.
  const ICONS = {
    ok: ["M2.5 6.5 5 9l4.5-5.5"],
    bad: ["M3 3l6 6", "M9 3 3 9"],
    warn: ["M6 1.8 11 10.2H1z", "M6 5v2.4", "M6 8.8v.1"],
  };

  function icon(kind) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 12 12");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("class", "rc-icon " + kind);
    (ICONS[kind] || []).forEach((d) => {
      const path = document.createElementNS(ns, "path");
      path.setAttribute("d", d);
      svg.appendChild(path);
    });
    return svg;
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

  function placeTip(tip) {
    const card = tip.card;
    const cardBox = card.getBoundingClientRect();
    const box = tip.trigger.getBoundingClientRect();
    const pop = tip.pop;
    pop.style.left = "0px";
    pop.style.top = "0px";
    const width = Math.min(pop.offsetWidth, cardBox.width - 16);
    let left = box.left - cardBox.left;
    left = Math.max(8, Math.min(left, cardBox.width - width - 8));
    const below = box.bottom - cardBox.top + 6;
    const roomBelow = window.innerHeight - box.bottom;
    const above = box.top - cardBox.top - pop.offsetHeight - 6;
    const useAbove = roomBelow < pop.offsetHeight + 12 && box.top > pop.offsetHeight + 12;
    pop.style.left = left + "px";
    pop.style.top = (useAbove ? above : below) + "px";
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

  function attachTip(trigger, card, build) {
    const pop = el("div", "pop");
    pop.id = "pop-" + (++popSequence);
    pop.setAttribute("role", "tooltip");
    trigger.classList.add("tip");
    trigger.tabIndex = 0;
    trigger.setAttribute("aria-describedby", pop.id);
    card.appendChild(pop);
    const tip = { trigger, pop, card, build, built: false, timer: 0 };
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

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && openTip) closeTip(openTip, true);
  });
  window.addEventListener("resize", () => {
    if (openTip) placeTip(openTip);
  });

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

  // Code lines with a colored rule: the explanation lines of a failure in
  // the fail hue, the rest gray.
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

  function stepLink(step, onStep) {
    const link = el("a", "pop-link", "#" + step.i);
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
    return lines.slice(start, start + TRACE_LINES + 4).filter((line, index, all) => line.trim() || index < all.length - 1);
  }

  function buildTestPop(test) {
    return (pop) => {
      popHead(pop, String(test.name || "test"), [test.location, fmtSpan(test.seconds)].filter(Boolean).join(" · "), true);
      const lines = traceLines(test);
      if (lines.length && lines.some((line) => line.trim())) popCode(pop, lines, /^E\s/);
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

  function buildTestListPop(tests, label) {
    return (pop) => {
      popHead(pop, plural(tests.length, label + " test", label + " tests"), null, false);
      tests.slice(0, LIST_LIMIT).forEach((test) => {
        popRow(pop, shortTest(test.name), [test.location, fmtSpan(test.seconds)].filter(Boolean).join(" · ") || "—");
      });
      if (tests.length > LIST_LIMIT) popNote(pop, (tests.length - LIST_LIMIT) + " more in the Verifier tab.");
    };
  }

  // ── the card ────────────────────────────────────────────────────────────

  function stateOf(payload) {
    const meta = asRecord(payload.meta);
    const status = asRecord(meta.status);
    const tone = BF.core.scoreStatus(status.score_outcome, meta.reward);
    const interrupted = status.execution === "timed_out" || status.execution === "errored";
    let label = tone === "pass" ? "Passed" : tone === "fail" ? "Failed" : "Not scored";
    if (status.execution === "timed_out") label = tone === "unscored" ? "Timed out" : label + " · timed out";
    else if (status.execution === "errored") label = tone === "unscored" ? "Errored" : label + " · errored";
    return { tone, interrupted, label, status };
  }

  function row(list, name, options = {}) {
    const item = el("li", "rc-row" + (options.cls ? " " + options.cls : ""));
    item.appendChild(el("span", "rc-node" + (options.node ? " " + options.node : "")));
    const main = el("span", "rc-main");
    main.appendChild(el("span", "rc-name", name));
    if (options.chip) main.appendChild(el("span", "rc-chip", options.chip));
    item.appendChild(main);
    item.appendChild(el("span", "rc-meta", options.meta || ""));
    list.appendChild(item);
    return item;
  }

  function render(payload, header, actions) {
    const meta = asRecord(payload.meta);
    const verifier = asRecord(payload.verifier);
    const verdict = asRecord(payload.verdict);
    const timing = asRecord(meta.timing);
    const usage = asRecord(meta.usage);
    const counts = asRecord(meta.counts);
    const steps = Array.isArray(payload.steps) ? payload.steps : [];
    const state = stateOf(payload);
    const onStep = actions.showStep;

    const card = el("section", "runcard tone-" + state.tone + (state.interrupted ? " interrupted" : ""));
    card.setAttribute("aria-label", "Run summary");

    // Title row: task, then the state on the right.
    const titleRow = el("div", "titlerow");
    const heading = el("h1", null, meta.task_name || payload.rollout_name || "trajectory");
    heading.tabIndex = -1;
    titleRow.appendChild(heading);
    const badge = el("span", "badge " + state.tone + (state.interrupted ? " interrupted" : ""));
    badge.appendChild(el("span", "dot"));
    badge.appendChild(document.createTextNode(state.label));
    titleRow.appendChild(badge);
    card.appendChild(titleRow);

    const subtitle = [];
    if (payload.rollout_name) subtitle.push(payload.rollout_name);
    if (meta.agent_name) subtitle.push(meta.agent_name);
    if (meta.skill_mode) subtitle.push(meta.skill_mode);
    if (meta.trajectory_source) subtitle.push("source: " + meta.trajectory_source);
    const branch = asRecord(meta.branch);
    if (branch.node_id) {
      subtitle.push(
        "branch child " + branch.node_id + " of fork " + String(branch.fork_id || "").slice(0, 8)
          + (branch.parent_node ? " at node " + branch.parent_node : ""),
      );
    }
    const sub = el("div", "sub");
    subtitle.forEach((part) => sub.appendChild(el("span", null, part)));
    card.appendChild(sub);
    if (meta.partial_trajectory) card.appendChild(el("div", "rc-partial", "This trajectory is partial: result.json marks the capture incomplete."));

    // Phase rows: the run as a small graph.
    const graph = el("ol", "rc-graph");
    const tools = toolSteps(steps);
    const failedTools = tools.filter((step) => ["failed", "cancelled"].includes(asRecord(step.tool).status));
    const tests = Array.isArray(verifier.ctrf) ? verifier.ctrf.map(asRecord) : [];
    const failing = tests.filter((test) => test.status === "failed");
    const passed = tests.filter((test) => test.status === "passed");
    PHASES.slice(0, 2).forEach(([key, label]) => {
      const seconds = finite(timing[key]);
      if (seconds !== null) row(graph, label, { meta: fmtSpan(seconds) });
    });

    const toolCount = typeof counts.tools === "number" ? counts.tools : tools.length;
    const agentMeta = [plural(toolCount, "tool call"), fmtSpan(timing.agent_execution)].filter(Boolean).join(" · ");
    const agentRow = row(graph, "Agent", {
      chip: meta.model || null,
      meta: agentMeta,
      node: state.interrupted ? "warn" : "solid",
    });
    attachTip(agentRow, card, (pop) => {
      const kinds = new Map();
      tools.forEach((step) => {
        const kind = String(asRecord(step.tool).kind || "other");
        kinds.set(kind, (kinds.get(kind) || 0) + 1);
      });
      popHead(pop, plural(toolCount, "tool call"), [
        typeof counts.prompts === "number" ? plural(counts.prompts, "prompt") : null,
        typeof counts.messages === "number" ? plural(counts.messages, "message") : null,
      ].filter(Boolean).join(" · "), false);
      [...kinds.entries()].sort((a, b) => b[1] - a[1]).forEach(([kind, count]) => popRow(pop, kind, String(count)));
      if (failedTools.length) {
        popHead(pop, plural(failedTools.length, "failed call"), null, false);
        failedTools.slice(0, LIST_LIMIT).forEach((step) => {
          const line = el("div", "pop-row");
          const label = el("span", "pop-label bad");
          label.textContent = String(asRecord(step.tool).title || "tool call");
          line.appendChild(label);
          const value = el("span", "pop-value");
          value.appendChild(stepLink(step, onStep));
          line.appendChild(value);
          pop.appendChild(line);
        });
      }
    });

    // The call a timeout or error cut off: one hover from its content.
    const namedStep = typeof verdict.step === "number" ? tools.find((step) => step.i === verdict.step) : null;
    if (namedStep && state.interrupted) {
      const tool = asRecord(namedStep.tool);
      const running = tool.status === "pending" || tool.status === "in_progress";
      const callRow = row(graph, (running ? "Interrupted: " : "Last call: ") + (tool.title || tool.kind || "tool call"), {
        cls: "rc-sub warn",
        meta: "#" + namedStep.i + (tool.status ? " · " + tool.status : ""),
      });
      attachTip(callRow, card, buildToolPop(namedStep, onStep));
    }

    const hasVerifier = finite(timing.verifier) !== null || tests.length || verifier.stdout || verifier.reward !== null && verifier.reward !== undefined;
    if (hasVerifier) {
      const verifierNode = state.tone === "pass" ? "ok" : state.tone === "fail" ? "bad" : "";
      const verifierRow = row(graph, "Verifier", {
        chip: verifier.ctrf_tool || null,
        meta: [tests.length ? passed.length + " of " + plural(tests.length, "test") + " passed" : null, fmtSpan(timing.verifier)]
          .filter(Boolean).join(" · "),
        node: verifierNode,
      });
      attachTip(verifierRow, card, (pop) => {
        popHead(pop, verifier.ctrf_tool || "verifier", fmtSpan(timing.verifier) ? "phase " + fmtSpan(timing.verifier) : null, Boolean(verifier.ctrf_tool));
        if (tests.length) {
          popRow(pop, "tests", plural(tests.length, "test") + (finite(verifier.ctrf_seconds) !== null ? " in " + fmtSpan(verifier.ctrf_seconds) : ""));
          popRow(pop, "passed", String(passed.length), passed.length ? "ok" : null);
          popRow(pop, "failed", String(failing.length), failing.length ? "bad" : null);
        }
        if (verifier.reward !== null && verifier.reward !== undefined) popRow(pop, "reward.txt", String(verifier.reward));
        const output = String(verifier.stdout || verifier.stderr || "");
        const tail = output.split("\n").map((line) => line.replace(/\s+$/, "")).filter(Boolean).slice(-5);
        if (tail.length) {
          popNote(pop, verifier.stdout ? "Last lines of test-stdout.txt" : "Last lines of test-stderr.txt");
          popCode(pop, tail, /\bfailed\b|Error/);
        }
      });
      failing.slice(0, 3).forEach((test) => {
        const testRow = row(graph, shortTest(test.name), {
          cls: "rc-sub bad",
          meta: [test.location, fmtSpan(test.seconds)].filter(Boolean).join(" · "),
        });
        attachTip(testRow, card, buildTestPop(test));
      });
      if (failing.length > 3) row(graph, (failing.length - 3) + " more failed", { cls: "rc-sub bad" });
    }
    card.appendChild(graph);

    // Checks: each summary opens its proof.
    const checks = el("div", "rc-checks");
    function check(mark, label, value, tone, build) {
      const item = el("span", "rc-check" + (tone ? " " + tone : ""));
      if (mark) item.appendChild(icon(mark));
      item.appendChild(el("span", "rc-check-label", label));
      if (value) item.appendChild(el("span", "rc-check-value", value));
      checks.appendChild(item);
      if (build) attachTip(item, card, build);
      return item;
    }
    if (passed.length) check("ok", "Tests", passed.length + " passed", "ok", buildTestListPop(passed, "passed"));
    if (failing.length) {
      check("bad", "Tests", failing.length + " failed", "bad", (pop) => {
        failing.slice(0, 4).forEach((test, index) => {
          if (index) pop.appendChild(el("hr", "pop-rule"));
          buildTestPop(test)(pop);
        });
        if (failing.length > 4) popNote(pop, (failing.length - 4) + " more in the Verifier tab.");
      });
    }
    const reward = fmtReward(meta.reward);
    check(null, "Reward", reward === null ? "none" : reward, null, (pop) => {
      const status = state.status;
      popHead(pop, "Reward " + (reward === null ? "not recorded" : reward), status.summary || null, false);
      popRow(pop, "execution", [status.execution ? String(status.execution).replace("_", " ") : null, status.execution_detail].filter(Boolean).join(", "));
      popRow(pop, "assessment", [status.assessment, status.assessment_detail].filter(Boolean).join(", "));
      if (verifier.reward !== null && verifier.reward !== undefined) popRow(pop, "verifier/reward.txt", String(verifier.reward));
    });
    const total = fmtSpan(finite(timing.total) ?? meta.duration_sec);
    if (total) {
      check(null, "Total", total, null, (pop) => {
        popHead(pop, "Time and usage", null, false);
        Object.entries(timing).forEach(([key, value]) => {
          if (key !== "total" && finite(value) !== null) popRow(pop, key.replace(/_/g, " "), fmtSpan(value));
        });
        popRow(pop, "total", total);
        if (finite(usage.cost_usd) === null) popNote(pop, "This run recorded no cost.");
      });
    }
    const tokens = fmtCount(usage.total_tokens);
    if (tokens) {
      check(null, "Tokens", tokens, null, (pop) => {
        popHead(pop, "Tokens", usage.usage_source || null, false);
        popRow(pop, "input", fmtCount(usage.n_input_tokens));
        popRow(pop, "output", fmtCount(usage.n_output_tokens));
        popRow(pop, "cache read", fmtCount(usage.n_cache_read_tokens));
        popRow(pop, "cache write", fmtCount(usage.n_cache_creation_tokens));
        popRow(pop, "total", tokens);
        if (finite(usage.cost_usd) !== null) popRow(pop, "cost", "$" + usage.cost_usd.toFixed(4));
      });
    }
    if (finite(usage.cost_usd) !== null) check(null, "Cost", "$" + usage.cost_usd.toFixed(2), null, null);
    card.appendChild(checks);

    // Footer: where the run ended, and the one next action, beside it.
    const foot = el("div", "rc-foot");
    const status = el("span", "rc-status");
    let action = null;
    if (failing.length) {
      status.appendChild(icon("bad"));
      status.append(plural(failing.length, "test") + " failed");
      action = ["Show failing test", () => actions.showTest(failing[0].name)];
    } else if (namedStep && state.interrupted) {
      status.appendChild(icon("warn"));
      status.append(state.status.execution === "timed_out" ? "Stopped by the time limit" : "Stopped by an error");
      action = ["Show call #" + namedStep.i, () => onStep(namedStep.i)];
    } else if (state.tone === "pass") {
      status.appendChild(icon("ok"));
      status.append(tests.length ? "All tests passed" : "Scored " + reward);
    } else if (state.tone === "fail") {
      status.appendChild(icon("bad"));
      status.append("Scored " + reward);
    } else {
      status.appendChild(icon("warn"));
      status.append("Not scored");
    }
    foot.appendChild(status);
    if (action) {
      const button = el("button", "rc-action", action[0]);
      button.type = "button";
      button.addEventListener("click", action[1]);
      foot.appendChild(button);
    }
    card.appendChild(foot);
    header.appendChild(card);

    if (verdict.sentence) {
      const why = el("p", "why");
      why.id = "why";
      BF.core.textWithRedaction(why, verdict.sentence);
      header.appendChild(why);
    }
    return heading;
  }

  return { render, fmtSpan, fmtReward };
})();
