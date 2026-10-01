/* Report page: loads reports/<id>.data.js and reports/<id>.notes.js for the id in the URL hash. */
  window.STS_REPORT_DEFAULTS = {
    metrics: document.getElementById("r-metrics-description").innerHTML,
    late: document.getElementById("r-late-title").textContent,
    training: document.getElementById("r-training-description").innerHTML
  };
  window.onReportDocsReady = function (api) {
    var COLORS = ["var(--accent)", "var(--gan)", "var(--baseline)", "var(--good)", "var(--bad)"];
    var ML = { log_mel_l1: "Log-mel L1", stft_spectral_convergence: "STFT SC", stft_log_magnitude: "STFT log-mag" };
    var $ = function (id) { return document.getElementById(id); };
    var esc = api.esc, T = function (s) { return window.STS.t(s); };
    var loaded = {};

    function loc(value) {
      if (value == null || typeof value === "string") return value || "";
      return value.en || "";
    }
    function load(src) {
      return new Promise(function (resolve, reject) {
        if (loaded[src]) return resolve();
        var s = document.createElement("script");
        s.src = src; s.onload = function () { loaded[src] = true; resolve(); }; s.onerror = reject;
        document.head.appendChild(s);
      });
    }
    function f4(v) { return v == null ? "—" : Number(v).toFixed(4); }
    function display(v) { return v == null ? "—" : String(v); }
    function numeric(v) { return typeof v === "number" && Number.isFinite(v); }
    function defaultText(key, fallback) {
      return fallback;
    }
    function pct(a, b) {
      var p = (b - a) / a * 100;
      return "<span class=\"" + (p < 0 ? "delta-good" : "delta-bad") + "\">" + (p > 0 ? "+" : "") + p.toFixed(1) + "%</span>";
    }
    function smooth(a, w) {
      return a.map(function (_, i) {
        var s = a.slice(Math.max(0, i - w + 1), i + 1).filter(function (v) { return v != null; });
        return s.length ? s.reduce(function (x, y) { return x + y; }, 0) / s.length : null;
      });
    }

    function renderIndex() {
      $("r-index").hidden = false; $("r-body").hidden = true; $("r-verdict").hidden = true;
      $("r-title").textContent = T("Experiment reports");
      $("r-lede").textContent = T("Each report compares finished runs under one protocol. Open one below.");
      $("r-chips").innerHTML = "";
      $("r-list").innerHTML = (window.STS_REPORT_INDEX || []).map(function (r) {
        return "<a class=\"card\" href=\"#" + r.id + "\"><h3>" + esc(loc(r.title)) + "</h3><p>" + esc(loc(r.summary)) +
          "</p><span class=\"card-foot\">" + esc(r.date) + " · " + esc(r.id) + "</span></a>";
      }).join("");
    }

    function renderReport(id) {
      var D = window.STS_REPORTS[id], N = (window.STS_REPORT_NOTES || {})[id] || {};
      var arms = Object.keys(D.arms), ref = D.reference, others = arms.filter(function (a) { return a !== ref; });
      $("r-metrics-description").innerHTML = loc(N.metricsDescription) || defaultText("rep.metrics.p", window.STS_REPORT_DEFAULTS.metrics);
      $("r-late-title").textContent = loc(N.lateTitle) || defaultText("rep.late", window.STS_REPORT_DEFAULTS.late);
      $("r-training-description").innerHTML = loc(N.trainingDescription) || defaultText("rep.training.p", window.STS_REPORT_DEFAULTS.training);
      var name = function (a) { return loc((N.arms || {})[a]) || a; };
      var color = function (a) { return COLORS[arms.indexOf(a) % COLORS.length]; };
      var sw = function (a) { return "<span class=\"swatch\" style=\"background:" + color(a) + "\"></span>"; };
      $("r-index").hidden = true; $("r-body").hidden = false;
      $("r-eyebrow").textContent = T("Experiment report") + " · " + (N.date || "");
      $("r-title").textContent = loc(N.title) || id;
      $("r-lede").innerHTML = loc(N.lede) + (loc(N.provenance) || "");
      $("r-chips").innerHTML = arms.map(function (a) {
        return "<span class=\"chip\">" + sw(a) + esc(name(a)) + (a === ref ? " · " + esc(T("reference")) : "") + "</span>";
      }).join("") + (N.chips || []).map(function (c) { return "<span class=\"chip\">" + esc(loc(c)) + "</span>"; }).join("");
      var verdict = loc(N.verdict);
      $("r-verdict").hidden = !verdict;
      $("r-verdict").innerHTML = verdict ? "<div class=\"callout-title\" style=\"color:var(--gan)\">" + esc(T("Conclusion")) + "</div>" + verdict : "";

      // runs
      var runRows = [
        ["Training by", function () { return esc(N.trainingBy || "—"); }],
        ["Status", function (a) { return D.arms[a].status.status; }],
        ["Iterations / updates", function (a) { var s = D.arms[a].status; return s.step.toLocaleString() + " / " + s.actual_optimizer_updates.toLocaleString(); }],
        ["Wall time", function (a) { var v = D.arms[a].status.session_wall_seconds; return numeric(v) ? (v / 3600).toFixed(2) + " h" : "—"; }],
        ["Seconds per iteration", function (a) { var v = (D.arms[a].train.sec_per_iter || []).filter(numeric); return v.length ? (v.reduce(function (x, y) { return x + y; }, 0) / v.length).toFixed(3) : "—"; }],
        ["Peak allocated memory", function (a) { var v = D.arms[a].status.peak_allocated_GiB; return numeric(v) ? v + " GiB" : "—"; }],
        ["Completed (UTC)", function (a) { return D.arms[a].status.completed_at_utc; }],
        ["Resumed", function (a) { var v = D.arms[a].status.resumed; return v == null ? "—" : v ? T("yes") : T("no"); }],
        ["Run folder", function (a) { return "<code>" + esc(D.arms[a].run) + "</code>"; }]
      ];
      if (N.completedDate) runRows.unshift(["Team completion date", function () { return esc(N.completedDate) + "<br><span class=\"small muted\">" + esc(loc(N.dateBasis)) + "</span>"; }]);
      if (N.componentBy) runRows.unshift(["Component development", function () { return esc(N.componentBy) + " · " + esc(N.componentDate || ""); }]);
      if (arms.some(function (a) { return D.arms[a].status.amp_skipped_updates != null; })) {
        runRows.splice(2, 0, ["AMP skips", function (a) { return display(D.arms[a].status.amp_skipped_updates); }]);
      }
      $("t-runs").innerHTML = "<thead><tr><th></th>" + arms.map(function (a) { return "<th>" + sw(a) + esc(name(a)) + "</th>"; }).join("") + "</tr></thead><tbody>" +
        runRows.map(function (r) { return "<tr><td>" + esc(T(r[0])) + "</td>" + arms.map(function (a) { return "<td>" + r[1](a) + "</td>"; }).join("") + "</tr>"; }).join("") + "</tbody>";

      // comparability
      var cfg = function (a) { return D.arms[a].config; };
      var short = function (h) { return h ? h.slice(0, 16) + "…" : "—"; };
      var weightsExcept = function (a) {
        var w = Object.assign({}, cfg(a).loss_weights || {}); (N.changed || []).forEach(function (k) { delete w[k]; });
        return Object.keys(w).map(function (k) { return k.replace("_weight", "") + " " + w[k]; }).join(", ");
      };
      var checks = [
        ["Dataset", function (a) { return cfg(a).data_dir; }], ["Training clips", function (a) { return display(cfg(a).training_clips); }],
        ["Metadata hash", function (a) { return short(cfg(a).metadata_sha256); }],
        ["Seed / batch size", function (a) { return display(cfg(a).train_seed) + " / " + display(cfg(a).batch_size); }],
        ["Iterations / evaluation interval", function (a) { return cfg(a).total_steps + " / " + cfg(a).eval_every; }],
        ["Notebook hash", function (a) { return short(cfg(a).notebook_sha256); }],
        ["Evaluator hash", function (a) { return short(cfg(a).eval_source_sha256); }],
        ["Evaluation batch hash", function (a) { return short(D.arms[a].batch_sha256); }],
        ["Other loss weights", weightsExcept],
        ["GPU / PyTorch", function (a) { return display(cfg(a).gpu) + " · " + display(cfg(a).torch); }]
      ];
      $("t-check").innerHTML = "<thead><tr><th>" + esc(T("Item")) + "</th>" + arms.map(function (a) { return "<th>" + esc(name(a)) + "</th>"; }).join("") + "<th>" + esc(T("Match")) + "</th></tr></thead><tbody>" +
        checks.map(function (c) {
          var vals = arms.map(c[1]), known = vals.every(function (v) { return v != null && v !== "—"; });
          var same = known && vals.every(function (v) { return String(v) === String(vals[0]); });
          return "<tr><td>" + esc(T(c[0])) + "</td>" + vals.map(function (v) { return "<td class=\"small\">" + esc(String(v)) + "</td>"; }).join("") +
            "<td class=\"" + (!known ? "muted" : same ? "ok" : "no") + "\">" + (!known ? "—" : same ? "✓" : "✗") + "</td></tr>";
        }).join("") +
        (N.changed || []).map(function (k) {
          return "<tr><td><code>" + esc(k) + "</code></td>" + arms.map(function (a) { return "<td>" + esc(String((cfg(a).loss_weights || {})[k])) + "</td>"; }).join("") +
            "<td class=\"muted\">" + esc(T("change under test")) + "</td></tr>";
        }).join("") + "</tbody>";

      // metrics
      var evalOf = function (a, mode) { return D.arms[a].eval.filter(function (r) { return r.mode === mode; }); };
      var stat = function (a, mode, k, which) {
        var rows = evalOf(a, mode);
        if (which === "final") return rows[rows.length - 1][k];
        var last = rows.filter(function (r) { return r.step > 0; }).slice(-5); return last.reduce(function (x, r) { return x + r[k]; }, 0) / last.length;
      };
      [["t-final", "final"], ["t-late", "late"]].forEach(function (spec) {
        var h = "<thead><tr><th>" + esc(T("Mode · metric")) + "</th>" + arms.map(function (a) { return "<th class=\"num\">" + sw(a) + esc(name(a)) + "</th>"; }).join("") +
          others.map(function (a) { return "<th class=\"num\">Δ " + esc(name(a)) + "</th>"; }).join("") + "</tr></thead><tbody>";
        ["mean", "sample"].forEach(function (m) {
          Object.keys(ML).forEach(function (k) {
            var r = stat(ref, m, k, spec[1]);
            h += "<tr><td>" + m + " · " + ML[k] + "</td>" + arms.map(function (a) { return "<td class=\"num\">" + f4(stat(a, m, k, spec[1])) + "</td>"; }).join("") +
              others.map(function (a) { return "<td class=\"num\">" + pct(r, stat(a, m, k, spec[1])) + "</td>"; }).join("") + "</tr>";
          });
        });
        $(spec[0]).innerHTML = h + "</tbody>";
      });
      var wh = "<thead><tr><th>" + esc(T("Mode · metric")) + "</th>";
      others.forEach(function (a) { wh += "<th class=\"num\">" + esc(name(a)) + ": " + esc(T("all points")) + "</th><th class=\"num\">" + esc(T("second half")) + "</th><th class=\"num\">" + esc(T("utterances, final")) + "</th>"; });
      wh += "</tr></thead><tbody>";
      ["mean", "sample"].forEach(function (m) {
        Object.keys(ML).forEach(function (k) {
          wh += "<tr><td>" + m + " · " + ML[k] + "</td>";
          others.forEach(function (a) {
            var base = {}; evalOf(ref, m).forEach(function (r) { base[r.step] = r[k]; });
            var rows = evalOf(a, m).filter(function (r) { return r.step > 0 && base[r.step] != null; });
            var half = rows.filter(function (r) { return r.step >= rows[rows.length - 1].step / 2; });
            var win = function (list) { return list.filter(function (r) { return r[k] < base[r.step]; }).length; };
            var ps = {}; D.arms[ref].per_sample_final.filter(function (r) { return r.mode === m; }).forEach(function (r) { ps[r.sample_id] = r[k]; });
            var cmp = D.arms[a].per_sample_final.filter(function (r) { return r.mode === m; });
            var cls = function (x, n) { return x / n >= 0.75 ? "delta-good" : x / n <= 0.25 ? "delta-bad" : ""; };
            var w1 = win(rows), w2 = win(half), w3 = cmp.filter(function (r) { return r[k] < ps[r.sample_id]; }).length;
            wh += "<td class=\"num " + cls(w1, rows.length) + "\">" + w1 + "/" + rows.length + "</td><td class=\"num " + cls(w2, half.length) + "\">" + w2 + "/" + half.length +
              "</td><td class=\"num " + cls(w3, cmp.length) + "\">" + w3 + "/" + cmp.length + "</td>";
          });
          wh += "</tr>";
        });
      });
      $("t-wins").innerHTML = wh + "</tbody>";

      // charts
      api.resetCharts();
      $("c-eval").innerHTML = ""; $("c-train").innerHTML = "";
      var chartBox = function (host, title) {
        var d = document.createElement("div"); d.innerHTML = "<div class=\"chart-title\">" + esc(title) + "</div><div></div>";
        host.appendChild(d); return d.lastChild;
      };
      ["mean", "sample"].forEach(function (m) {
        ["log_mel_l1", "stft_spectral_convergence"].forEach(function (k) {
          var steps = evalOf(ref, m).filter(function (r) { return r.step > 0; }).map(function (r) { return r.step; });
          api.lineChart(chartBox($("c-eval"), ML[k] + " · " + m), {
            label: ML[k], x: steps, width: 360, height: 230, xLabel: "iteration",
            xTicks: [steps[0], steps[Math.floor(steps.length / 2)], steps[steps.length - 1]],
            series: arms.map(function (a) {
              var map = {}; evalOf(a, m).forEach(function (r) { map[r.step] = r[k]; });
              return { name: name(a), color: color(a), dots: true, values: steps.map(function (s) { return map[s]; }) };
            })
          });
        });
      });
      var tr = D.arms[ref].train, xs = tr.step;
      var ticks = [0, xs[Math.floor(xs.length / 2)], xs[xs.length - 1]];
      [["mel", "Mel L1 (unweighted)"], ["stft", "Multi-resolution STFT (unweighted)"], ["duration", "Duration loss"], ["kl", "KL loss"]].forEach(function (c) {
        api.lineChart(chartBox($("c-train"), T(c[1])), {
          label: c[1], x: xs, xMin: 0, xTicks: ticks, width: 360, height: 230, xLabel: "iteration",
          series: arms.map(function (a) { return { name: name(a), color: color(a), values: smooth(D.arms[a].train[c[0]], 10) }; })
        });
      });
      arms.filter(function (a) { return (cfg(a).loss_weights || {}).gan_weight > 0; }).forEach(function (a) {
        var t2 = D.arms[a].train;
        api.lineChart(chartBox($("c-train"), T("Discriminator scores") + " · " + name(a)), {
          label: "discriminator", x: t2.step, xMin: 0, xTicks: ticks, width: 360, height: 230, xLabel: "iteration", yMin: 0,
          series: [{ name: T("real"), color: "var(--good)", values: smooth(t2.discriminator_real_score, 10) },
                   { name: T("generated"), color: "var(--bad)", values: smooth(t2.discriminator_fake_score, 10) }]
        });
        api.lineChart(chartBox($("c-train"), T("GAN losses") + " · " + name(a)), {
          label: "gan", x: t2.step, xMin: 0, xTicks: ticks, width: 360, height: 230, xLabel: "iteration", yMin: 0,
          series: [{ name: T("discriminator"), color: "var(--baseline)", values: smooth(t2.gan_discriminator, 10) },
                   { name: T("generator adversarial"), color: color(a), values: smooth(t2.gan, 10) }]
        });
      });

      $("r-listen").innerHTML = loc(N.listen);
      $("r-findings").innerHTML = loc(N.findings);
      $("r-storage").innerHTML = loc(N.storage);
      $("t-ckpt").innerHTML = "<thead><tr><th>" + esc(T("Checkpoint")) + "</th>" + arms.map(function (a) { return "<th>" + esc(name(a)) + "</th>"; }).join("") + "</tr></thead><tbody>" +
        D.arms[ref].checkpoints.map(function (c) {
          return "<tr><td><code>" + esc(c) + "</code></td>" + arms.map(function (a) { return "<td>" + (D.arms[a].checkpoints.indexOf(c) >= 0 ? "✓" : "—") + "</td>"; }).join("") + "</tr>";
        }).join("") + "</tbody>";
      $("r-team").innerHTML = loc(N.team);
      var slot = $("r-team").closest(".slot");
      slot.classList.toggle("filled", !!loc(N.team));
      api.addHeadingAnchors();
    }

    function scrollToReportTarget(section) {
      var reportSection = $("reports");
      var target = section ? $(section) : $("r-title");
      if (!reportSection || !target) return;
      reportSection.scrollIntoView({ behavior: "auto", block: "start" });
      window.requestAnimationFrame(function () {
        window.requestAnimationFrame(function () {
          target.scrollIntoView({ behavior: "auto", block: "start" });
        });
      });
    }

    var current = null, routeSequence = 0;
    function route() {
      var hash = decodeURIComponent(location.hash.replace(/^#/, ""));
      var reportId = null, section = null;
      if ((window.STS_REPORT_INDEX || []).some(function (r) { return r.id === hash; })) {
        reportId = hash;
      } else {
        var match = /^report-(jacques-enc-dec|full-train-gan)(?:-(runs|comparability|metrics|training|listening|findings|storage|team-notes))?$/.exec(hash);
        if (match) { reportId = match[1]; section = match[2] || null; }
      }
      if (!reportId) {
        if (!hash || hash === "reports") { current = null; renderIndex(); return; }
        if (document.getElementById(hash)) {
          if (current === null) renderIndex();
          return;
        }
        if (current === null) renderIndex();
        return;
      }
      var sequence = ++routeSequence;
      Promise.all([load("reports/" + reportId + ".data.js"), load("reports/" + reportId + ".notes.js")])
        .then(function () {
          if (sequence !== routeSequence) return;
          current = reportId; renderReport(reportId);
          scrollToReportTarget(section);
        })
        .catch(function () {
          if (sequence === routeSequence) $("r-title").textContent = T("Report files are missing") + ": " + reportId;
        });
    }
    window.addEventListener("hashchange", route);
    route();
  };
