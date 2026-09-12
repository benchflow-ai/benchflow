BF.navigation = (() => {
  let generation = 0;
  let request = null;
  let loadedRun = null;
  let selectedRun = null;

  function beginTransition() {
    generation += 1;
    if (request) request.abort();
    request = null;
    BF.detail.cancel();
    return generation;
  }

  function showCatalog(push) {
    beginTransition();
    loadedRun = null;
    if (push) BF.catalog.writeURL(null, true);
    BF.catalog.show({ focusRun: selectedRun });
    document.title = "runs - benchflow trajectory";
  }

  async function openRun(runId, push, sourceButton = null) {
    const index = document.getElementById("view-index");
    if (!index.classList.contains("hidden")) BF.catalog.rememberScroll();
    selectedRun = runId;
    const transition = beginTransition();
    loadedRun = null;
    if (push) BF.catalog.writeURL(runId, true);
    BF.core.showDetailShell(true);
    BF.detail.showLoading(runId);

    const controller = new AbortController();
    request = controller;
    try {
      const response = await fetch("/api/rollout?id=" + encodeURIComponent(runId), {
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("HTTP " + response.status + " loading run " + runId);
      const body = await response.text();
      let payload;
      try {
        payload = JSON.parse(body);
      } catch (error) {
        throw new TypeError("malformed JSON from the rollout API: " + error.message);
      }
      BF.core.requirePayload(payload);
      if (transition !== generation) return;
      BF.detail.loadPayload(payload, { focusHeading: Boolean(sourceButton) });
      loadedRun = runId;
      window.scrollTo(0, 0);
    } catch (error) {
      if (controller.signal.aborted || transition !== generation) return;
      BF.detail.showError("Failed to load run " + runId + ": " + error.message);
    } finally {
      if (transition === generation) request = null;
    }
  }

  function showUnknownRun(runId) {
    selectedRun = runId;
    beginTransition();
    loadedRun = null;
    BF.core.showDetailShell(true);
    BF.detail.showError(BF.catalog.unknownRunMessage(runId));
  }

  function applyLocation() {
    const runId = BF.catalog.readURL();
    if (runId && runId === loadedRun && !document.getElementById("content").classList.contains("hidden")) {
      return;
    }
    if (runId && BF.catalog.hasRun(runId)) openRun(runId, false);
    else if (runId) showUnknownRun(runId);
    else showCatalog(false);
  }

  function startBrowse(boot) {
    BF.catalog.init(boot, (runId, sourceButton) => openRun(runId, true, sourceButton));
    const back = document.getElementById("backbtn");
    back.addEventListener("click", () => showCatalog(true));
    window.addEventListener("popstate", applyLocation);
    applyLocation();
  }

  function startSingle(payload) {
    beginTransition();
    BF.core.showDetailShell(false);
    try {
      BF.detail.loadPayload(payload);
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
    BF.navigation.startSingle(boot.payload);
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
  bootError('Unknown viewer mode: expected "single" or "browse".');
})();
