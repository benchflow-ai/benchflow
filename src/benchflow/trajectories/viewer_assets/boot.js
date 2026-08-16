/*
 * DOM bootstrap. Everything that is not a pure string transform lives here:
 * payload acquisition, mount, theme, and the interactive controls.
 *
 * Payload acquisition has two modes on purpose. `bench eval view` inlines the
 * data island, so the file works offline. The published site (slice 5) sets
 * window.BF_CONFIG = {dataUrl: "…/viewer_data/<run_id>.json"} and the same
 * render.js draws it — one renderer, two delivery paths.
 *
 * The controls mirror the vendored PostTrainBench viewer's: theme toggle,
 * jump-to-turn, expand outputs, copy run id, download. Upstream's Focus/Full
 * toggle is deliberately not carried over yet — see TODO.md. Every event the
 * normalizer emits is rendered, which is upstream's "Full".
 */
(function () {
  "use strict";

  var root = document.documentElement;
  var storageKey = "benchflow-viewer-theme";
  var mount = document.querySelector("#bf-app");

  function preferredTheme() {
    try {
      var saved = window.localStorage.getItem(storageKey);
      if (saved === "dark" || saved === "light") return saved;
    } catch (err) {
      // localStorage is unavailable for a directly opened file:// page.
    }
    return window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  }

  function setTheme(theme, persist) {
    root.setAttribute("data-theme", theme);
    var button = document.querySelector("#theme-toggle");
    if (button) {
      var next = theme === "dark" ? "light" : "dark";
      button.setAttribute("aria-label", "Switch to " + next + " theme");
      button.setAttribute("title", "Switch to " + next + " theme");
    }
    if (persist) {
      try {
        window.localStorage.setItem(storageKey, theme);
      } catch (err) {
        // The selected theme still applies for this page load.
      }
    }
  }

  function inlinePayload() {
    var island = document.querySelector("#bf-run-data");
    if (!island) return null;
    try {
      return JSON.parse(island.textContent);
    } catch (err) {
      return null;
    }
  }

  function download(payload) {
    var blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json"
    });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = payload.name + ".trace.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 0);
  }

  function copyText(text, feedback) {
    function flash() {
      if (!feedback) return;
      feedback.classList.add("visible");
      setTimeout(function () {
        feedback.classList.remove("visible");
      }, 1200);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(flash, function () {});
      return;
    }
    var field = document.createElement("textarea");
    field.value = text;
    document.body.appendChild(field);
    field.select();
    try {
      document.execCommand("copy");
      flash();
    } catch (err) {
      // Clipboard is unavailable; the id stays selectable by hand.
    }
    document.body.removeChild(field);
  }

  function wire(payload) {
    var themeButton = document.querySelector("#theme-toggle");
    if (themeButton) {
      themeButton.addEventListener("click", function () {
        setTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark", true);
      });
    }

    var expandSwitch = document.querySelector("#expand-outputs");
    if (expandSwitch) {
      expandSwitch.addEventListener("change", function () {
        var all = document.querySelectorAll("details.tool-call");
        for (var i = 0; i < all.length; i++) {
          all[i].classList.toggle("expanded", expandSwitch.checked);
          all[i].open = true;
        }
      });
    }

    var jump = document.querySelector("#jump-turn");
    if (jump) {
      jump.addEventListener("keydown", function (event) {
        if (event.key !== "Enter") return;
        var target = document.querySelector("#turn-" + String(jump.value).trim());
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }

    if (payload) {
      var feedback = document.querySelector("#copy-feedback");
      var copyButton = document.querySelector("#copy-id-btn");
      if (copyButton) {
        copyButton.addEventListener("click", function () {
          copyText(payload.name, feedback);
        });
      }
      var link = document.querySelector("#link-raw");
      if (link) {
        link.addEventListener("click", function (event) {
          event.preventDefault();
          download(payload);
        });
      }
    }

    // Clicking an event marker copies a permalink to that turn.
    var markers = document.querySelectorAll(".event-marker[data-anchor]");
    for (var j = 0; j < markers.length; j++) {
      (function (marker) {
        marker.addEventListener("click", function () {
          var anchor = marker.getAttribute("data-anchor");
          copyText(
            window.location.href.split("#")[0] + "#" + anchor,
            document.querySelector("#copy-feedback")
          );
          window.location.hash = anchor;
        });
      })(markers[j]);
    }
  }

  function draw(payload) {
    if (mount) mount.innerHTML = BFViewer.renderRunHtml(payload);
    setTheme(preferredTheme(), false);
    wire(payload);
    if (window.location.hash) {
      var target = document.querySelector(window.location.hash);
      if (target) target.scrollIntoView({ block: "start" });
    }
  }

  setTheme(preferredTheme(), false);

  var config = window.BF_CONFIG || {};
  if (!mount) {
    // A message page (job-directory hint) ships its own server-rendered body.
    wire(null);
  } else if (config.dataUrl) {
    fetch(config.dataUrl)
      .then(function (response) {
        return response.json();
      })
      .then(draw)
      .catch(function () {
        mount.innerHTML =
          '<div class="layout"><main class="content"><h1>Could not load trace</h1>' +
          "</main></div>";
      });
  } else {
    draw(inlinePayload());
  }
})();
