/* STS2026 docs: single-page navigation, theme, charts and comparison table. */
(function () {
  "use strict";

  /* Versions own their reports; pending work belongs to To-do.
     Prefix dates use d/m: version creation or task setup, not run completion. */
  var NAV = [
    { group: "Project", defaultOpen: true, items: [
      { href: "#overview", label: "Overview" }
    ]},
    { group: "To-do", defaultOpen: true, items: [
      { href: "#task-tts", label: "Continue TTS training", ver: "1/10" },
      { href: "#task-rvc", label: "Basic RVC implementation", ver: "1/10" },
      { href: "#turino-listening", label: "Listening improvements · Turino", ver: "1/10" }
    ]},
    { group: "Versions", defaultOpen: true, groups: [
      { group: "17/9 · Original", items: [
        { href: "#version-v0", label: "Model overview" }
      ]},
      { group: "20/9 · Original_eval", items: [
        { href: "#version-v1", label: "Model and baseline" },
        { href: "#jacques-enc-dec", label: "Original vs new encoder/decoder" }
      ]},
      { group: "24/9 · Original_eval_gan", items: [
        { href: "#version-original-eval-gan", label: "Model overview" }
      ]},
      { group: "30/9 · TTS_0.01", items: [
        { href: "#version-full-training", label: "Notebook overview" },
        { href: "#full-train-gan", label: "GAN vs no GAN" }
      ]},
      { group: "7/10 · TTS_0.02", items: [
        { href: "#version-speaker-conditioning", label: "Speaker-ID conditioning" }
      ]}
    ]}
  ];

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }

  /* English is the only maintained website language. */
  function t(s) { return s; }
  function tf(s, vars) { return s.replace(/\{(\w+)\}/g, function (_, k) { return vars[k]; }); }

  /* ---- theme ---- */
  var saved = store("sts-theme");
  if (saved === "dark" || saved === "light") document.documentElement.setAttribute("data-theme", saved);
  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function buildNav() {
    var host = document.querySelector("[data-sidenav]");
    if (!host) return;
    var here = location.hash || "#overview";
    var html = "<nav aria-label=\"" + esc(t("Documentation")) + "\">";
    function containsActive(g) {
      return (g.items || []).some(function (it) { return it.href === here; }) ||
        (g.groups || []).some(containsActive);
    }
    function renderGroup(g) {
      var key = g.group.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      var saved = store("sts-nav-group-" + key);
      var open = containsActive(g) || (saved === null ? g.defaultOpen === true : saved === "true");
      var content = "<details class=\"nav-group\" data-nav-group=\"" + esc(key) + "\"" + (open ? " open" : "") + "><summary>" +
        "<span>" + esc(t(g.group)) + "</span><span class=\"nav-chevron\" aria-hidden=\"true\"></span></summary><ul>";
      (g.items || []).forEach(function (it) {
        var cur = it.href === here ? " aria-current=\"location\"" : "";
        var ver = it.ver ? "<span class=\"ver " + (it.cls || "") + "\">" + esc(it.ver) + "</span>" : "";
        content += "<li><a href=\"" + esc(it.href) + "\"" + (it.sub ? " class=\"sub\"" : "") + cur + ">" + ver + "<span>" + esc(t(it.label)) + "</span></a></li>";
      });
      (g.groups || []).forEach(function (child) {
        content += "<li>" + renderGroup(child) + "</li>";
      });
      return content + "</ul></details>";
    }
    NAV.forEach(function (g) { html += renderGroup(g); });
    host.innerHTML = html + "</nav>";
    host.querySelectorAll(".nav-group").forEach(function (group) {
      group.addEventListener("toggle", function () {
        store("sts-nav-group-" + group.getAttribute("data-nav-group"), group.open ? "true" : "false");
      });
    });
  }

  function addHeadingAnchors() {
    var heads = Array.prototype.filter.call(document.querySelectorAll("main h2[id], main h3[id]"),
      function (h) { return !h.closest("[hidden]"); });
    heads.forEach(function (h) {
      if (h.querySelector(".anchor")) return;
      h.querySelectorAll(".anchor").forEach(function (old) { old.remove(); });
      var a = document.createElement("a");
      a.className = "anchor"; a.href = "#" + h.id; a.textContent = "#"; a.setAttribute("aria-label", t("Link to this section"));
      h.appendChild(a);
    });
  }

  /* Track the reading position without changing history or keyboard focus. */
  function wireScrollTracking() {
    var host = document.querySelector("[data-sidenav]");
    if (!host) return;
    var pending = false, active = null;
    function update() {
      pending = false;
      var links = Array.prototype.slice.call(host.querySelectorAll("a[href^='#']"));
      var selected = links[0], latestTop = -Infinity;
      var section = document.querySelector(".doc-section");
      var padding = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
      var margin = section ? parseFloat(getComputedStyle(section).scrollMarginTop) || 0 : 0;
      var readingLine = padding + margin + 8;
      links.forEach(function (link) {
        var target = document.getElementById(link.getAttribute("href").slice(1));
        if (!target) return; // Result headings appear after lazy loading.
        var rect = target.getBoundingClientRect();
        var end = target.tagName === "H3" ? target.parentElement.getBoundingClientRect().bottom : rect.bottom;
        if (rect.top <= readingLine && end > readingLine && rect.top >= latestTop) {
          selected = link; latestTop = rect.top;
        }
      });
      if (window.scrollY > 0 && window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
        // A short final section cannot reach the reading line at the end of the page.
        for (var i = links.length - 1; i >= 0; i--) {
          if (document.getElementById(links[i].getAttribute("href").slice(1))) { selected = links[i]; break; }
        }
      }
      if (!selected) return;
      links.forEach(function (link) {
        if (link === selected) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
      var href = selected.getAttribute("href");
      if (href === active) return;
      active = href;
      var parent = selected.closest("details");
      while (parent) { parent.open = true; parent = parent.parentElement.closest("details"); }
      // Scroll only the navigation rail; scrollIntoView would also move the article.
      var rail = host.getBoundingClientRect(), item = selected.getBoundingClientRect();
      if (item.top < rail.top) host.scrollTop -= rail.top - item.top + 8;
      else if (item.bottom > rail.bottom) host.scrollTop += item.bottom - rail.bottom + 8;
    }
    function schedule() {
      if (!pending) { pending = true; window.requestAnimationFrame(update); }
    }
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    window.addEventListener("hashchange", function () { active = null; schedule(); });
    // Lazy results and expandable curves can change section positions without scrolling.
    if (window.ResizeObserver) new ResizeObserver(schedule).observe(document.querySelector(".article"));
    schedule();
  }

  /* Slots: a slot whose .slot-body has any content switches to the "filled" style. */
  function markSlots() {
    document.querySelectorAll(".slot").forEach(function (s) {
      var body = s.querySelector(".slot-body");
      s.classList.toggle("filled", !!body && body.innerHTML.replace(/<!--[\s\S]*?-->/g, "").trim() !== "");
      var head = s.querySelector(".slot-id");
      if (head && s.dataset.slot) head.textContent = t("slot") + ": " + s.dataset.slot;
    });
    var open = document.querySelectorAll(".slot:not(.filled)").length;
    var badge = document.querySelector("[data-open-slots]");
    if (badge) badge.textContent = tf(open === 1 ? "{n} open team section" : "{n} open team sections", { n: open });
  }

  var themeLabel = function () {};
  function wireChrome() {
    var btn = document.querySelector("[data-theme-toggle]");
    if (btn) {
      themeLabel = function () { btn.textContent = isDark() ? t("Light") : t("Dark"); };
      themeLabel();
      btn.addEventListener("click", function () {
        var next = isDark() ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        store("sts-theme", next); themeLabel(); redrawAll();
      });

    }
    var m = document.querySelector("[data-menu]"), shell = document.querySelector(".shell");
    var mobile = window.matchMedia ? window.matchMedia("(max-width: 760px)") : { matches: false };
    function syncNavButton() {
      if (!m || !shell) return;
      if (mobile.matches) {
        var open = document.body.classList.contains("nav-open");
        m.textContent = open ? t("Close") : t("Menu");
        m.setAttribute("aria-expanded", open ? "true" : "false");
        m.setAttribute("aria-label", open ? t("Close navigation") : t("Open navigation"));
      } else {
        var collapsed = shell.classList.contains("nav-collapsed");
        m.textContent = collapsed ? t("Show nav") : t("Hide nav");
        m.setAttribute("aria-expanded", collapsed ? "false" : "true");
        m.setAttribute("aria-label", collapsed ? t("Show navigation") : t("Hide navigation"));
      }
    }
    function syncViewport() {
      if (mobile.matches) {
        shell.classList.remove("nav-collapsed");
        document.body.classList.remove("nav-open");
      } else {
        document.body.classList.remove("nav-open");
        shell.classList.toggle("nav-collapsed", store("sts-nav-collapsed") === "true");
      }
      syncNavButton();
    }
    syncViewport();
    if (m) m.addEventListener("click", function () {
      if (mobile.matches) document.body.classList.toggle("nav-open");
      else {
        shell.classList.toggle("nav-collapsed");
        store("sts-nav-collapsed", shell.classList.contains("nav-collapsed") ? "true" : "false");
      }
      syncNavButton();
    });
    var nav = document.querySelector("[data-sidenav]");
    if (nav) nav.addEventListener("click", function (ev) {
      if (mobile.matches && ev.target.closest("a")) {
        document.body.classList.remove("nav-open"); syncNavButton();
      }
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && mobile.matches && document.body.classList.contains("nav-open")) {
        document.body.classList.remove("nav-open"); syncNavButton();
      }
    });
    if (mobile.addEventListener) mobile.addEventListener("change", syncViewport);
    else if (mobile.addListener) mobile.addListener(syncViewport);
  }

  /* ---------------- charts ---------------- */
  var charts = [];
  function niceTicks(min, max, n) {
    var span = max - min || 1, step = Math.pow(10, Math.floor(Math.log10(span / n)));
    var err = span / n / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step, out = [];
    for (var v = lo; v <= hi + step / 2; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function fmt(v) { return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2); }

  /* opts: { x:[..], series:[{name, values, color:"var(--accent)", dash}], xLabel, height, markers:[x] } */
  function lineChart(el, opts) {
    charts.push(function () { drawLine(el, opts); });
    drawLine(el, opts);
  }
  function drawLine(el, o) {
    var W = o.width || 640, H = o.height || 240, L = 46, R = 14, T = 12, B = 34;
    var xs = o.x, all = [];
    o.series.forEach(function (s) { s.values.forEach(function (v) { if (v != null) all.push(v); }); });
    var yt = niceTicks(Math.min.apply(null, all), Math.max.apply(null, all), 4);
    var y0 = o.yMin != null ? o.yMin : yt[0], y1 = yt[yt.length - 1];
    var x0 = o.xMin != null ? o.xMin : xs[0], x1 = xs[xs.length - 1];
    function px(x) { return L + (x - x0) / (x1 - x0) * (W - L - R); }
    function py(y) { return T + (1 - (y - y0) / (y1 - y0)) * (H - T - B); }
    var svg = "<svg class=\"chart\" viewBox=\"0 0 " + W + " " + H + "\" role=\"img\" aria-label=\"" + esc(o.label || "chart") + "\">";
    yt.forEach(function (v) {
      if (v < y0) return;
      svg += "<line class=\"grid\" x1=\"" + L + "\" x2=\"" + (W - R) + "\" y1=\"" + py(v) + "\" y2=\"" + py(v) + "\"/>";
      svg += "<text x=\"" + (L - 6) + "\" y=\"" + (py(v) + 4) + "\" text-anchor=\"end\">" + fmt(v) + "</text>";
    });
    (o.xTicks || niceTicks(x0, x1, 5)).forEach(function (v) {
      if (v < x0 || v > x1) return;
      var anchor = v === x1 ? "end" : v === x0 ? "start" : "middle";
      svg += "<text x=\"" + px(v) + "\" y=\"" + (H - B + 16) + "\" text-anchor=\"" + anchor + "\">" + (v >= 10000 ? v / 1000 + "k" : v) + "</text>";
    });
    svg += "<line class=\"axis\" x1=\"" + L + "\" x2=\"" + (W - R) + "\" y1=\"" + (H - B) + "\" y2=\"" + (H - B) + "\"/>";
    if (o.xLabel) svg += "<text x=\"" + (W - R) + "\" y=\"" + (H - 4) + "\" text-anchor=\"end\">" + esc(t(o.xLabel)) + "</text>";
    (o.markers || []).forEach(function (m) {
      svg += "<line x1=\"" + px(m) + "\" x2=\"" + px(m) + "\" y1=\"" + T + "\" y2=\"" + (H - B) + "\" style=\"stroke:var(--rule);stroke-dasharray:3 3\"/>";
    });
    o.series.forEach(function (s) {
      var d = "", pts = "";
      s.values.forEach(function (v, i) {
        if (v == null) return;
        d += (d ? "L" : "M") + px(xs[i]).toFixed(1) + " " + py(v).toFixed(1);
        if (s.dots) pts += "<circle cx=\"" + px(xs[i]) + "\" cy=\"" + py(v) + "\" r=\"3.2\" style=\"fill:var(--surface);stroke:" + s.color + ";stroke-width:1.8\"/>";
      });
      svg += "<path d=\"" + d + "\" style=\"fill:none;stroke:" + s.color + ";stroke-width:" + (s.width || 1.8) + (s.dash ? ";stroke-dasharray:" + s.dash : "") + ";stroke-linejoin:round\"/>" + pts;
      var li = s.values.length - 1;
      if (!s.dots && s.values[li] != null) svg += "<circle cx=\"" + px(xs[li]) + "\" cy=\"" + py(s.values[li]) + "\" r=\"3.2\" style=\"fill:" + s.color + "\"/>";
    });
    svg += "<line class=\"hover-line\" x1=\"0\" x2=\"0\" y1=\"" + T + "\" y2=\"" + (H - B) + "\" style=\"stroke:var(--ink-3);opacity:0\"/>";
    svg += "<rect x=\"" + L + "\" y=\"" + T + "\" width=\"" + (W - L - R) + "\" height=\"" + (H - T - B) + "\" style=\"fill:transparent\" class=\"hit\"/></svg>";
    var legend = "<div class=\"legend\">" + o.series.map(function (s) {
      return "<span><i class=\"line\" style=\"border-color:" + s.color + (s.dash ? ";border-top-style:dashed" : "") + "\"></i>" + esc(s.name) + "</span>";
    }).join("") + "<span class=\"readout muted\" style=\"margin-left:auto\"></span></div>";
    el.innerHTML = svg + legend;
    var svgEl = el.querySelector("svg"), hit = el.querySelector(".hit"), line = el.querySelector(".hover-line"), out = el.querySelector(".readout");
    hit.addEventListener("mousemove", function (ev) {
      var r = svgEl.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W;
      var best = 0;
      xs.forEach(function (v, i) { if (Math.abs(px(v) - x) < Math.abs(px(xs[best]) - x)) best = i; });
      line.setAttribute("x1", px(xs[best])); line.setAttribute("x2", px(xs[best])); line.style.opacity = 0.6;
      out.textContent = t(o.xName || "iteration") + " " + xs[best] + " · " + o.series.map(function (s) {
        return s.name + " " + (s.values[best] == null ? t("n/a") : s.values[best].toFixed(4));
      }).join(" · ");
    });
    hit.addEventListener("mouseleave", function () { line.style.opacity = 0; out.textContent = ""; });
  }
  function redrawAll() { charts.forEach(function (f) { f(); }); drawSpectro(); }

  /* ---------------- comparison table ---------------- */
  var METRICS = [
    { key: "log_mel_l1", label: "Log-mel L1" },
    { key: "stft_sc", label: "STFT SC" },
    { key: "stft_logmag", label: "STFT log-mag" }
  ];
  var STATUS = {
    completed: ["ok", "completed"], running: ["acc", "running"], "in-progress": ["gan", "in progress"],
    planned: ["", "planned"], failed: ["warn", "failed"]
  };
  function num(v, d) { return v == null ? "<span class=\"delta-na\">—</span>" : v.toFixed(d == null ? 4 : d); }
  function delta(v, b) {
    if (v == null || b == null) return "<span class=\"delta-na\">" + esc(t("not measured")) + "</span>";
    var p = (v - b) / b * 100;
    if (Math.abs(p) < 0.005) return "<span class=\"delta-na\">±0.00%</span>";
    return "<span class=\"" + (p < 0 ? "delta-good" : "delta-bad") + "\">" + (p > 0 ? "+" : "") + p.toFixed(2) + "%</span>";
  }
  function renderComparison(el) {
    var S = window.STS; if (!S) return;
    var mode = el.getAttribute("data-mode") || "mean";
    var groups = S.GROUPS || [{ id: null, reference: S.BASELINE_ID }];
    var out = "";
    groups.forEach(function (g) {
      var rows = S.EXPERIMENTS.filter(function (e) { return g.id == null || e.group === g.id; });
      var base = S.EXPERIMENTS.filter(function (e) { return e.id === g.reference; })[0];
      if (!rows.length) return;
      if (g.id != null) {
        out += "<div class=\"board-group\"><h3>" + esc(g.title) + "</h3><p class=\"small muted\">" + esc(g.note || "") +
          (g.report ? " <a href=\"" + g.report + "\">" + esc(t("Open report")) + "</a>" : "") + "</p></div>";
      }
      var h = "<div class=\"table-wrap\"><table><thead><tr><th>" + esc(t("Variant")) + "</th><th>" + esc(t("Status")) + "</th>";
      METRICS.forEach(function (m) { h += "<th class=\"num\">" + esc(t(m.label)) + "</th><th class=\"num\">" + esc(t("Δ vs base")) + "</th>"; });
      h += "<th class=\"num\">" + esc(t("Iter / updates")) + "</th></tr></thead><tbody>";
      rows.forEach(function (e) {
        var st = STATUS[e.status] || ["", e.status];
        var isBase = base && e.id === base.id;
        h += "<tr class=\"" + (isBase ? "is-base" : "") + "\"><td><strong>" + esc(e.name) + "</strong><br><span class=\"small muted\">" + esc(e.change) + " · <code>" + esc(e.branch) + "</code></span>";
        if (e.trainingBy) h += "<br><span class=\"small muted\">" + esc(t("Training by")) + ": " + esc(e.trainingBy) + (e.completedDate ? " · " + esc(e.completedDate) : "") + "</span>";
        h += "</td>";
        h += "<td><span class=\"chip " + st[0] + "\"><span class=\"dot\"></span>" + esc(t(st[1])) + "</span></td>";
        METRICS.forEach(function (m) {
          var v = e.final[mode][m.key], b = base ? base.final[mode][m.key] : null;
          h += "<td class=\"num\">" + num(v) + "</td><td class=\"num\">" + (isBase ? "<span class=\"delta-na\">" + esc(t("base")) + "</span>" : delta(v, b)) + "</td>";
        });
        h += "<td class=\"num\">" + (e.iterations == null ? "<span class=\"delta-na\">—</span>" : e.iterations + " / " + e.updates) + "</td></tr>";
      });
      out += h + "</tbody></table></div>";
    });
    el.innerHTML = out;
  }

  /* ---------------- hero spectrogram (decorative, deterministic) ---------------- */
  function drawSpectro() {
    var c = document.querySelector("canvas.spectro"); if (!c) return;
    var dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    if (!w) return;
    c.width = w * dpr; c.height = h * dpr;
    var g = c.getContext("2d"); g.scale(dpr, dpr);
    var cs = getComputedStyle(document.documentElement);
    var bg = cs.getPropertyValue("--sunken").trim(), a = cs.getPropertyValue("--accent").trim(), b = cs.getPropertyValue("--baseline").trim();
    g.fillStyle = bg; g.fillRect(0, 0, w, h);
    var cols = Math.floor(w / 4), rows = 40, cw = w / cols, rh = h / rows, seed = 7;
    function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }
    // three "formant" tracks gliding like LJSpeech syllables; energy gaps mark word boundaries
    for (var x = 0; x < cols; x++) {
      var t = x / cols, env = Math.max(0, Math.sin(t * 38) * 0.5 + 0.55) * (Math.sin(t * 7.3) > -0.85 ? 1 : 0.08);
      var f = [6 + 3 * Math.sin(t * 11), 15 + 5 * Math.sin(t * 5 + 1), 26 + 4 * Math.sin(t * 8 + 2)];
      for (var y = 0; y < rows; y++) {
        var e = 0;
        f.forEach(function (fc, k) { e += Math.exp(-Math.pow(y - fc, 2) / (2 + k)) * (1 - k * 0.25); });
        e += (y % 3 === 0 && y < 22 ? 0.25 : 0) + rnd() * 0.12;
        e *= env * (1 - y / rows * 0.6);
        if (e < 0.08) continue;
        g.globalAlpha = Math.min(1, e);
        g.fillStyle = t > 0.62 && t < 0.62 + 32 / cols * 2.2 ? b : a;
        g.fillRect(x * cw, h - (y + 1) * rh, cw + 0.4, rh + 0.4);
      }
    }
    g.globalAlpha = 1;
  }

  window.STS = window.STS || {};
  window.STS.t = t;

  document.addEventListener("DOMContentLoaded", function () {
    document.documentElement.lang = "en";
    buildNav(); addHeadingAnchors(); markSlots(); wireChrome();
    document.querySelectorAll("[data-comparison]").forEach(renderComparison);
    document.querySelectorAll("[data-mode-switch]").forEach(function (sw) {
      sw.addEventListener("change", function (ev) {
        document.querySelectorAll("[data-comparison]").forEach(function (t) { t.setAttribute("data-mode", ev.target.value); renderComparison(t); });
      });
    });
    drawSpectro();
    var rt; window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(drawSpectro, 120); });
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
    var api = {
      lineChart: lineChart, esc: esc,
      resetCharts: function () { charts.length = 0; },
      addHeadingAnchors: addHeadingAnchors, buildNav: buildNav
    };
    [window.onDocsReady, window.onBaselineDocsReady, window.onReportDocsReady].forEach(function (init) {
      if (typeof init === "function") init(api);
    });
    window.addEventListener("hashchange", buildNav);
    wireScrollTracking();
  });
})();
