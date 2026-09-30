/* STS2026 docs: shared shell (nav, contents, theme), charts and comparison table. */
(function () {
  "use strict";

  /* ---- page registry: add a page here and it appears in every sidebar ---- */
  var NAV = [
    { group: "Project", items: [
      { href: "index.html", label: "Overview" }
    ]},
    { group: "Versions", items: [
      { href: "prototype.html", label: "Prototype", ver: "v0" },
      { href: "original-eval.html", label: "Original + eval", ver: "v1", cls: "base" }
    ]},
    { group: "Improvements", items: [
      { href: "improvements.html", label: "Comparison board", ver: "v2+", cls: "gan" },
      { href: "improvements.html#v2-gan", label: "Adversarial training (GAN)", ver: "v2", cls: "gan" },
      { href: "listening-turino-2026-10-01.html", label: "Listening improvements · Turino", ver: "10-01" }
    ]},
    { group: "Experiment reports", items: [
      { href: "report.html#jacques-enc-dec", label: "Jacques: original vs new enc/dec", ver: "09-20", cls: "base" },
      { href: "report.html#full-train-gan", label: "Full training: GAN vs no-GAN", ver: "09-29", cls: "gan" }
    ]},
    { group: "Repository", items: [
      { href: "../EVALUATION.md", label: "Evaluation protocol (md)" },
      { href: "../LOCAL_BASELINE_RESULT.md", label: "Baseline record (md)" },
      { href: "../LOCAL_EVALUATION.md", label: "Local runner (md)" },
      { href: "../../CONTRIBUTING.md", label: "Contributing (md)" }
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
    var here = (location.pathname.split("/").pop() || "index.html");
    var html = "<nav aria-label=\"" + esc(t("Documentation")) + "\">";
    NAV.forEach(function (g) {
      html += "<div><h4>" + esc(t(g.group)) + "</h4><ul>";
      g.items.forEach(function (it) {
        var parts = it.href.split("#"), cur = parts[0] === here && (!parts[1] || "#" + parts[1] === location.hash)
          ? " aria-current=\"page\"" : "";
        var ver = it.ver ? "<span class=\"ver " + (it.cls || "") + "\">" + esc(it.ver) + "</span>" : "";
        html += "<li><a href=\"" + it.href + "\"" + cur + ">" + ver + "<span>" + esc(t(it.label)) + "</span></a></li>";
      });
      html += "</ul></div>";
    });
    host.innerHTML = html + "</nav>";
  }

  var tocObserver = null;
  function buildToc() {
    var host = document.querySelector("[data-toc]");
    var heads = Array.prototype.filter.call(document.querySelectorAll("main h2[id], main h3[id]"),
      function (h) { return !h.closest("[hidden]"); });
    heads.forEach(function (h) {
      h.querySelectorAll(".anchor").forEach(function (old) { old.remove(); });
      var a = document.createElement("a");
      a.className = "anchor"; a.href = "#" + h.id; a.textContent = "#"; a.setAttribute("aria-label", t("Link to this section"));
      h.appendChild(a);
    });
    if (!host) return;
    if (!heads.length) { host.innerHTML = ""; return; }
    var html = "<h4>" + esc(t("On this page")) + "</h4><ul>";
    heads.forEach(function (h) {
      var text = Array.prototype.filter.call(h.childNodes, function (n) { return !(n.classList && n.classList.contains("anchor")); })
        .map(function (n) { return n.textContent; }).join("");
      html += "<li><a href=\"#" + h.id + "\"" + (h.tagName === "H3" ? " class=\"sub\"" : "") + ">" + esc(text.trim()) + "</a></li>";
    });
    host.innerHTML = html + "</ul>";
    var links = host.querySelectorAll("a");
    if (!("IntersectionObserver" in window)) return;
    if (tocObserver) tocObserver.disconnect();
    var io = tocObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        links.forEach(function (l) { l.classList.toggle("active", l.getAttribute("href") === "#" + e.target.id); });
      });
    }, { rootMargin: "-80px 0px -70% 0px" });
    heads.forEach(function (h) { io.observe(h); });
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
    var m = document.querySelector("[data-menu]");
    if (m) m.textContent = t("Menu");
    if (m) m.addEventListener("click", function () {
      var open = document.body.classList.toggle("nav-open");
      m.setAttribute("aria-expanded", open ? "true" : "false");
    });
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
    buildNav(); buildToc(); markSlots(); wireChrome();
    document.querySelectorAll("[data-comparison]").forEach(renderComparison);
    document.querySelectorAll("[data-mode-switch]").forEach(function (sw) {
      sw.addEventListener("change", function (ev) {
        document.querySelectorAll("[data-comparison]").forEach(function (t) { t.setAttribute("data-mode", ev.target.value); renderComparison(t); });
      });
    });
    drawSpectro();
    var rt; window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(drawSpectro, 120); });
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
    if (typeof window.onDocsReady === "function") window.onDocsReady({
      lineChart: lineChart, esc: esc, resetCharts: function () { charts.length = 0; }, buildToc: buildToc, buildNav: buildNav
    });
    window.addEventListener("hashchange", buildNav);
  });
})();
