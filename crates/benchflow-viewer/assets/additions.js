// What the Rust viewer adds to 0.8's run page, which is otherwise 0.8's
// unchanged: the trial's defense record in its header, and a views "runs"
// link that keeps the run list's filter. Loaded after detail.js and before
// boot.js, so the first trial a page opens already gets the record.
BF.additions = (() => {
  const { el } = BF.core;

  // States in the order the summary counts them; any other state follows.
  const STATES = ["enforced", "failed", "off", "not-supported", "not recorded"];
  const STATE_COLORS = new Map([
    ["enforced", "var(--ok-ink)"],
    ["failed", "var(--bad-ink)"],
    ["off", "var(--warn-ink)"],
    ["not-supported", "var(--warn-ink)"],
  ]);
  // A row with no state reads the same in the summary and in its cell.
  const NO_STATE = "no state";
  // The rows that answer "was the agent's network protected?". A record can
  // lack one or both: an engine before F1 recorded no cloud_metadata, and
  // network_policy only for a network other than open; a trial that stopped
  // before its agent's network was set up has neither; a trial whose agent
  // network step failed (R108, R109) records only the failed one; and a
  // regrade's record covers only the verifier. Without them such a record
  // would read as an all-clear, so a record that has rows but lacks one of
  // these shows it as "not recorded", with no note: the state says it all.
  // Only these two: the record stays untyped, and other rows show as written.
  const NETWORK_ROWS = ["network_policy", "cloud_metadata"];
  const NOT_RECORDED = "not recorded";

  // Inline styles cannot hold a media query: below 720 px each row puts name
  // and state on one line and the note under them, instead of squeezing the
  // note beside the fixed name and state columns (at 521 px it had 193 px).
  document.head.appendChild(el(
    "style",
    null,
    "#defenses { margin-top: 12px; font-size: 12px; }\n"
      + "#defenses summary { cursor: pointer; color: var(--ink-secondary); }\n"
      + "#defenses .lbl { font-family: var(--font-mono); font-size: 10px; font-weight: 600;"
      + " letter-spacing: 0.08em; text-transform: uppercase; color: var(--faint); }\n"
      + "#defenses table { margin-top: 6px; font-size: inherit; }\n"
      + "#defenses td { vertical-align: top; }\n"
      + "#defenses td:first-child { font-family: var(--font-mono); }\n"
      + "#defenses td:nth-child(2) { white-space: nowrap; }\n"
      + "#defenses col.name { width: 16em; }\n"
      + "#defenses col.state { width: 8em; }\n"
      + "@media (max-width: 720px) {\n"
      + "  #defenses table { table-layout: auto; }\n"
      + "  #defenses col.name, #defenses col.state { width: auto; }\n"
      + "  #defenses tr { display: grid; grid-template-columns: auto 1fr; column-gap: 10px;"
      + " border-bottom: 1px solid var(--border); }\n"
      + "  #defenses td { padding: 6px 0; border-bottom: none; }\n"
      + "  #defenses td:nth-child(3) { grid-column: 1 / -1; padding-top: 0; }\n"
      + "  #defenses tr:last-child { border-bottom: none; }\n"
      + "  #defenses td:nth-child(3):empty { display: none; }\n"
      + "}\n",
  ));

  // One line under the stats, "defenses 13 enforced · 2 off · …", that opens
  // to every row with its state and why. It starts open when a defense failed.
  function renderDefenses(payload) {
    const header = document.getElementById("hdr");
    const old = document.getElementById("defenses");
    if (old) old.remove();
    const meta = payload && payload.meta;
    const rows = (meta && Array.isArray(meta.defenses) ? meta.defenses : [])
      .filter((row) => row && typeof row === "object" && typeof row.name === "string")
      .map((row) => ({
        name: row.name,
        state: typeof row.state === "string" ? row.state : NO_STATE,
        note: typeof row.note === "string" ? row.note : "",
      }));
    if (!header || !rows.length) return;
    NETWORK_ROWS.filter((name) => !rows.some((row) => row.name === name)).forEach((name) => {
      rows.push({ name, state: NOT_RECORDED, note: "" });
    });

    const counts = new Map();
    rows.forEach((row) => counts.set(row.state, (counts.get(row.state) || 0) + 1));
    const order = [...STATES.filter((state) => counts.has(state)), ...[...counts.keys()].filter((state) => !STATES.includes(state))];

    const box = el("details");
    box.id = "defenses";
    box.open = counts.has("failed");
    const summary = el("summary");
    summary.appendChild(el("span", "lbl", "defenses"));
    order.forEach((state, index) => {
      summary.appendChild(document.createTextNode(index ? " · " : " "));
      const count = el("span", null, counts.get(state) + " " + state);
      if (STATE_COLORS.has(state)) count.style.color = STATE_COLORS.get(state);
      summary.appendChild(count);
    });
    box.appendChild(summary);

    const table = el("table", "plain");
    const columns = el("colgroup");
    ["name", "state", "note"].forEach((name) => columns.appendChild(el("col", name)));
    table.appendChild(columns);
    rows.forEach((row) => {
      const line = el("tr");
      line.appendChild(el("td", null, row.name));
      const state = el("td", null, row.state);
      if (STATE_COLORS.has(row.state)) state.style.color = STATE_COLORS.get(row.state);
      line.appendChild(state);
      line.appendChild(el("td", null, row.note));
      table.appendChild(line);
    });
    box.appendChild(table);

    const errors = document.getElementById("errors");
    header.insertBefore(box, errors && errors.parentNode === header ? errors : null);
  }

  const loadPayload = BF.detail.loadPayload;
  BF.detail.loadPayload = (value, options) => {
    loadPayload(value, options);
    renderDefenses(value);
  };

  // The views link back to the run list, from a trial: the run list's own
  // query (filter, grouping, sort, open groups) without the trial it has
  // open. Set when the link is used, since the run list rewrites the URL as
  // it is filtered. On the run list itself it stays the plain "/", the
  // one-click way to the unfiltered list.
  const runsLink = document.querySelector("#bf-views a[href='/']");
  if (runsLink) {
    const sync = () => {
      const params = new URLSearchParams(location.search);
      if (!params.has("run")) {
        runsLink.setAttribute("href", "/");
        return;
      }
      params.delete("run");
      params.delete("branch");
      const query = params.toString();
      runsLink.setAttribute("href", "/" + (query ? "?" + query : ""));
    };
    ["pointerenter", "focus", "click"].forEach((type) => runsLink.addEventListener(type, sync));
  }

  return { renderDefenses };
})();
