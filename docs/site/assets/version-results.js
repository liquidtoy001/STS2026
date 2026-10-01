/* Compact results live once, under their notebook revision. Data remains canonical. */
window.onReportDocsReady = function (api) {
  "use strict";
  var specs = [
    { id: "jacques-enc-dec", host: "encoder-results", title: "Original vs new encoder/decoder",
      context: "Jacques · 2,000 iterations · eight training utterances · reconstruction-only objective. Component identity remains unverified: both runs record identical model/source metadata. Differences cannot yet be attributed to the revised architecture." },
    { id: "full-train-gan", host: "gan-results", title: "GAN vs no GAN",
      context: "29/9 runs · 50,000 iterations · full-corpus LJSpeech · complete objective · Kaggle T4. GAN weight 0 vs 1 is the change under test. These results are not comparable with the 2,000-iteration Original_eval baseline." }
  ];
  var metrics = { log_mel_l1: "Log-mel L1", stft_spectral_convergence: "STFT SC", stft_log_magnitude: "STFT log-magnitude" };
  var tasks = {};
  function load(src) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = src; script.onload = resolve; script.onerror = reject;
      document.head.appendChild(script);
    });
  }
  function render(spec) {
    var data = window.STS_REPORTS[spec.id], host = document.getElementById(spec.host);
    var arms = Object.keys(data.arms), names = spec.id === "full-train-gan" ? ["No GAN", "GAN"] : ["Original", "New enc/dec folder (identity pending)"];
    var final = function (arm, mode) { return data.arms[arm].eval.filter(function (row) { return row.mode === mode; }).slice(-1)[0]; };
    var html = "<h3 id=\"" + spec.id + "\">" + api.esc(spec.title) + "</h3><p>" + api.esc(spec.context) + "</p>";
    html += "<p class=\"small muted\">Completed (UTC): " + arms.map(function (arm, i) {
      return api.esc(names[i] + " " + data.arms[arm].status.completed_at_utc);
    }).join("; ") + ".</p><p class=\"small muted\">Final fixed-crop reconstruction errors; mean and sample modes are separate. Lower is better. Δ is relative to " + api.esc(names[0]) + ".</p>";
    html += "<div class=\"table-wrap\"><table><thead><tr><th>Mode · metric</th>" + names.map(function (name) { return "<th>" + api.esc(name) + "</th>"; }).join("") + "<th>Δ</th></tr></thead><tbody>";
    ["mean", "sample"].forEach(function (mode) {
      Object.keys(metrics).forEach(function (key) {
        var values = arms.map(function (arm) { return final(arm, mode)[key]; });
        var change = (values[1] - values[0]) / values[0] * 100;
        html += "<tr><td>" + mode + " · " + metrics[key] + "</td>" + values.map(function (value) { return "<td class=\"num\">" + value.toFixed(4) + "</td>"; }).join("") + "<td class=\"num\">" + (change > 0 ? "+" : "") + change.toFixed(2) + "%</td></tr>";
      });
    });
    html += "</tbody></table></div><details><summary>Evaluation and training curves</summary><div class=\"chart-grid\" data-result-charts></div></details>";
    html += "<p class=\"small muted\">Training-batch reconstruction does not establish held-out or text-to-speech quality. Spectral metrics alone cannot decide which audio sounds better.</p>";
    if (spec.id === "full-train-gan") html += "<p>No GAN has lower sample-mode log-mel and STFT log-magnitude errors at every recorded evaluation point. GAN improves final mean-mode STFT log-magnitude, while its other final errors are higher.</p>";
    host.innerHTML = html;
    var grid = host.querySelector("[data-result-charts]");
    ["mean", "sample"].forEach(function (mode) {
      var box = document.createElement("div"); box.className = "chart-box"; grid.appendChild(box);
      var rows = data.arms[arms[0]].eval.filter(function (row) { return row.mode === mode; });
      api.lineChart(box, { label: "Log-mel L1 · " + mode, x: rows.map(function (row) { return row.step; }), xLabel: "iteration", width: 360, height: 230,
        series: arms.map(function (arm, i) { return { name: names[i] + " · " + mode, color: i ? "var(--gan)" : "var(--accent)", values: data.arms[arm].eval.filter(function (row) { return row.mode === mode; }).map(function (row) { return row.log_mel_l1; }) }; }) });
    });
    ["total", "mel", "stft", "wave"].forEach(function (key) {
      var box = document.createElement("div"); box.className = "chart-box"; grid.appendChild(box);
      api.lineChart(box, { label: "Training loss · " + key, x: data.arms[arms[0]].train.step, xLabel: "iteration", width: 360, height: 230,
        series: arms.map(function (arm, i) { return { name: names[i] + " · " + key, color: i ? "var(--gan)" : "var(--accent)", values: data.arms[arm].train[key] }; }) });
    });
    api.addHeadingAnchors();
  }
  function ensure(spec) {
    if (!tasks[spec.id]) tasks[spec.id] = load("reports/" + spec.id + ".data.js").then(function () { render(spec); }).catch(function () {
      document.getElementById(spec.host).textContent = "Results could not be loaded. Reload this page to retry.";
    });
    return tasks[spec.id];
  }
  function route() {
    var hash = location.hash.slice(1);
    var spec = specs.filter(function (item) { return hash === item.id || hash.indexOf("report-" + item.id) === 0; })[0];
    if (spec) ensure(spec).then(function () { document.getElementById(spec.host).scrollIntoView({ block: "start" }); });
    else if (hash === "reports") document.getElementById("versions").scrollIntoView({ block: "start" });
  }
  if (window.IntersectionObserver) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) { if (entry.isIntersecting) {
        ensure(specs.filter(function (spec) { return spec.host === entry.target.id; })[0]); observer.unobserve(entry.target);
      }});
    }, { rootMargin: "500px" });
    specs.forEach(function (spec) { document.getElementById(spec.host).style.minHeight = "120px"; observer.observe(document.getElementById(spec.host)); });
  } else specs.forEach(ensure);
  window.addEventListener("hashchange", route);
  route();
};
