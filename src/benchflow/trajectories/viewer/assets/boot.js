BF.navigation = (() => {
  let generation = 0;
  let request = null;
  let loadedRun = null;
  let selectedRun = null;
  // The branch child on screen (null for a run or the catalog): the back
  // button then leads to its parent run instead of the run list.
  let shownBranch = null;

  function beginTransition() {
    generation += 1;
    if (request) request.abort();
    request = null;
    BF.detail.cancel();
    return generation;
  }

  // A branch child is addressed by its parent run plus a tree.json ref.
  function runKey(runId, branch) {
    return runId + (branch ? "\u0000" + branch : "");
  }

  function setBackTarget(branch) {
    shownBranch = branch || null;
    document.getElementById("backbtn").textContent = shownBranch ? "\u2190 parent run" : "\u2190 runs";
  }

  function showCatalog(push) {
    beginTransition();
    loadedRun = null;
    setBackTarget(null);
    if (push) BF.catalog.writeURL(null, true);
    BF.catalog.show({ focusRun: selectedRun });
    document.title = "runs - benchflow trajectory";
  }

  async function openRun(runId, push, sourceButton = null, branch = null) {
    const index = document.getElementById("view-index");
    if (!index.classList.contains("hidden")) BF.catalog.rememberScroll();
    selectedRun = runId;
    const transition = beginTransition();
    loadedRun = null;
    if (push) BF.catalog.writeURL(runId, true, branch);
    BF.core.showDetailShell(true);
    setBackTarget(branch);
    const label = branch ? runId + " (branch " + branch + ")" : runId;
    BF.detail.showLoading(label);

    const controller = new AbortController();
    request = controller;
    try {
      const url = "/api/rollout?id=" + encodeURIComponent(runId)
        + (branch ? "&branch=" + encodeURIComponent(branch) : "");
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) throw new Error("HTTP " + response.status + " loading run " + label);
      const body = await response.text();
      let payload;
      try {
        payload = JSON.parse(body);
      } catch (error) {
        throw new TypeError("malformed JSON from the rollout API: " + error.message);
      }
      BF.core.requirePayload(payload);
      if (transition !== generation) return;
      BF.detail.loadPayload(payload, {
        focusHeading: Boolean(sourceButton),
        branchHref: (ref) => BF.catalog.urlFor(runId, ref),
        openBranch: (ref) => openRun(runId, true, null, ref),
      });
      loadedRun = runKey(runId, branch);
      window.scrollTo(0, 0);
    } catch (error) {
      if (controller.signal.aborted || transition !== generation) return;
      BF.detail.showError("Failed to load run " + label + ": " + error.message);
    } finally {
      if (transition === generation) request = null;
    }
  }

  function showUnknownRun(runId) {
    selectedRun = runId;
    beginTransition();
    loadedRun = null;
    BF.core.showDetailShell(true);
    setBackTarget(null);
    BF.detail.showError(BF.catalog.unknownRunMessage(runId));
  }

  function applyLocation() {
    const runId = BF.catalog.readURL();
    const branch = BF.catalog.readBranch();
    if (runId && runKey(runId, branch) === loadedRun && !document.getElementById("content").classList.contains("hidden")) {
      return;
    }
    if (runId && BF.catalog.hasRun(runId)) openRun(runId, false, null, branch);
    else if (runId) showUnknownRun(runId);
    else showCatalog(false);
  }

  function startBrowse(boot) {
    BF.catalog.init(boot, (runId, sourceButton) => openRun(runId, true, sourceButton));
    if (boot.jobviews) {
      BF.jobviews.init({
        openRun: (runId) => {
          BF.catalog.allowRun(runId);
          openRun(runId, true);
        },
      });
    }
    const back = document.getElementById("backbtn");
    back.addEventListener("click", () => {
      if (shownBranch && selectedRun) openRun(selectedRun, true);
      else showCatalog(true);
    });
    window.addEventListener("popstate", applyLocation);
    applyLocation();
  }

  // A single-trajectory page embeds its branch children; ?branch=<ref>
  // selects one, so the page works without a server (trajectory.html).
  function startSingle(payload, branches) {
    beginTransition();
    BF.core.showDetailShell(false);
    const ref = new URLSearchParams(location.search).get("branch");
    if (ref) {
      // A branch child's way back is its parent: this page without ?branch=.
      setBackTarget(ref);
      document.getElementById("backbar").classList.remove("hidden");
      document.getElementById("backbtn").addEventListener("click", () => {
        const parent = new URL(location.href);
        parent.searchParams.delete("branch");
        parent.hash = "";
        location.assign(parent.href);
      });
    }
    const embedded = BF.core.isRecord(branches) ? branches : {};
    if (ref && !Object.prototype.hasOwnProperty.call(embedded, ref)) {
      BF.detail.showError('Branch child "' + ref + '" is not in this rollout\'s lineage.', "load error");
      return;
    }
    try {
      BF.detail.loadPayload(ref ? embedded[ref] : payload, {
        branchHref: (child) => "?branch=" + encodeURIComponent(child),
      });
    } catch (error) {
      BF.detail.showError("The embedded trajectory payload is malformed: " + error.message, "viewer data error");
    }
  }

  return { startBrowse, startSingle };
})();

(() => {
  function bootError(message) {
    BF.core.showDetailShell(false);
    BF.detail.showError(message, "viewer data error");
  }

  const node = document.getElementById("bf-payload");
  let boot;
  try {
    boot = JSON.parse(node ? node.textContent : "");
  } catch (error) {
    bootError("The embedded viewer data is not valid JSON: " + error.message);
    return;
  }
  if (!BF.core.isRecord(boot)) {
    bootError("The embedded viewer data must be a JSON object.");
    return;
  }
  BF.theme.init();
  if (boot.mode === "single") {
    BF.navigation.startSingle(boot.payload, boot.branches);
    return;
  }
  if (boot.mode === "browse") {
    if (!Array.isArray(boot.rollouts)) {
      bootError("Browse-mode viewer data must contain a rollouts array.");
      return;
    }
    const invalid = boot.rollouts.find((run) => !BF.core.isRecord(run) || typeof run.id !== "string" || !run.id);
    if (invalid) {
      bootError("A catalog entry is malformed: every rollout must be an object with a non-empty string id.");
      return;
    }
    BF.navigation.startBrowse(boot);
    return;
  }
  if (boot.mode === "export") {
    if (!BF.core.isRecord(boot.outcomes) || !BF.core.isRecord(boot.outcomes.columns)) {
      bootError("Export viewer data must contain an outcomes document.");
      return;
    }
    BF.jobviews.startExport(boot.outcomes);
    return;
  }
  bootError('Unknown viewer mode: expected "single", "browse" or "export".');
})();
