// Color theme (light/dark) and syntax highlighting of tool content.
//
// The theme is a data attribute on <html>; template.html applies the stored
// or system preference before first paint, this module owns the toggle.
// Highlighting is deliberately conservative: a block is colored only when its
// language is known from a markdown fence, a diff signature, or the tool kind,
// never from auto-detection, so plain logs stay plain.
window.BF = window.BF || {};

BF.theme = (() => {
  const KEY = "bf-theme";

  function current() {
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }

  function apply(theme) {
    if (theme === "dark") document.documentElement.dataset.theme = "dark";
    else delete document.documentElement.dataset.theme;
    const button = document.getElementById("theme-toggle");
    if (button) button.textContent = theme === "dark" ? "light" : "dark";
  }

  function toggle() {
    const next = current() === "dark" ? "light" : "dark";
    try { localStorage.setItem(KEY, next); } catch (e) { /* storage may be disabled */ }
    apply(next);
  }

  function init() {
    apply(current());
    const button = document.getElementById("theme-toggle");
    if (button) button.addEventListener("click", toggle);
  }

  return { init, toggle, current };
})();

BF.highlight = (() => {
  const FENCE = /^```([A-Za-z0-9_+-]*)\s*\n/;
  const HEREDOC = /^(.*?\b(?:python3?|python -)\b[^\n]*<<-?\s*['"]?(\w+)['"]?\s*\n)([\s\S]*?)(\n\2\b[\s\S]*)$/;
  const EXTENSIONS = {
    py: "python", sh: "bash", bash: "bash", json: "json", yaml: "yaml", yml: "yaml",
    toml: "ini", ini: "ini", md: "markdown", c: "c", h: "c", cpp: "cpp", hpp: "cpp",
    cc: "cpp", js: "javascript", mjs: "javascript", ts: "typescript", rs: "rust",
    diff: "diff", patch: "diff",
  };
  const FENCE_ALIASES = { console: "shell", sh: "bash", shell: "shell", text: null, "": null };

  function known(language) {
    return language && window.hljs && hljs.getLanguage(language) ? language : null;
  }

  function fromFence(text) {
    const match = text.match(FENCE);
    if (!match) return null;
    const name = match[1].toLowerCase();
    return known(name in FENCE_ALIASES ? FENCE_ALIASES[name] : name);
  }

  function isDiff(text) {
    const head = text.slice(0, 400);
    return /^\*\*\* Begin Patch|^diff |^--- [^\n]*\n\+\+\+ |^@@ /m.test(head);
  }

  // Language for one content block of a tool card, or null to leave it plain.
  function languageFor(text, hue, title, position) {
    if (!window.hljs) return null;
    const fenced = fromFence(text);
    if (fenced) return fenced;
    if (isDiff(text)) return "diff";
    if (text.startsWith("```")) return null;
    if (hue === "execute" && position === 0) return "bash";
    if ((hue === "read" || hue === "edit") && typeof title === "string") {
      const match = title.match(/\.([A-Za-z0-9]+)\s*$/);
      if (match) return known(EXTENSIONS[match[1].toLowerCase()]);
    }
    return null;
  }

  function render(text, language) {
    const options = { language, ignoreIllegals: true };
    const heredoc = language === "bash" ? text.match(HEREDOC) : null;
    if (heredoc && hljs.getLanguage("python")) {
      return hljs.highlight(heredoc[1], options).value
        + hljs.highlight(heredoc[3], { language: "python", ignoreIllegals: true }).value
        + hljs.highlight(heredoc[4], options).value;
    }
    return hljs.highlight(text, options).value;
  }

  // Replace the block's text nodes with highlighted markup. Blocks that carry
  // redaction markers keep their spans and are left uncolored.
  function paint(block, text, language) {
    if (!language || !window.hljs || block.querySelector(".redacted")) return;
    if (block.textContent !== text) return;
    block.innerHTML = render(text, language);
    block.classList.add("hljs");
    block.dataset.language = language;
  }

  return { languageFor, paint };
})();
