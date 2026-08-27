"use strict";

/* Shared, deliberately small primitives. The focused assets loaded after this
   file extend BF with detail, catalog, and navigation modules. */
const BF = {};

BF.core = (() => {
  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function isRecord(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function requirePayload(value) {
    if (!isRecord(value)) throw new TypeError("payload must be a JSON object");
    if (value.steps !== undefined && !Array.isArray(value.steps)) {
      throw new TypeError("payload.steps must be an array");
    }
    for (const key of ["meta", "verifier"]) {
      if (value[key] !== undefined && !isRecord(value[key])) {
        throw new TypeError("payload." + key + " must be an object");
      }
    }
    return value;
  }

  function textWithRedaction(target, text) {
    String(text).split("***REDACTED***").forEach((part, index) => {
      if (index > 0) target.appendChild(el("span", "redacted", "REDACTED"));
      if (part) target.appendChild(document.createTextNode(part));
    });
  }

  function fmtTokens(value) {
    if (value === null || value === undefined) return null;
    return value >= 10000 ? (value / 1000).toFixed(1) + "k" : String(value);
  }

  function fmtDuration(seconds) {
    if (typeof seconds !== "number" || !Number.isFinite(seconds)) return null;
    const total = Math.round(seconds);
    const minutes = Math.floor(total / 60);
    const remainder = total % 60;
    return minutes ? minutes + "m " + remainder + "s" : remainder + "s";
  }

  function showDetailShell(isBrowse) {
    document.getElementById("view-index").classList.add("hidden");
    document.getElementById("content").classList.remove("hidden");
    document.getElementById("backbar").classList.toggle("hidden", !isBrowse);
  }

  return {
    el,
    fmtDuration,
    fmtTokens,
    isRecord,
    requirePayload,
    showDetailShell,
    textWithRedaction,
  };
})();
