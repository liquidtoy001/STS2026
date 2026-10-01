window.onBaselineDocsReady = function (api) {
    var H = STS.BASELINE_EVAL_HISTORY, steps = [0, 500, 1000, 1500, 2000];
    function pick(mode, key) { return steps.map(function (s) { return H.filter(function (r) { return r.step === s && r.mode === mode; })[0][key]; }); }
    [["v1-ch-mel", "log_mel_l1"], ["v1-ch-sc", "stft_sc"]].forEach(function (c) {
      api.lineChart(document.getElementById(c[0]), {
        label: c[1] + " by evaluation step", x: steps, xTicks: [0, 1000, 2000], width: 340, height: 230, yMin: 0, xLabel: "iteration",
        series: [
          { name: "mean", values: pick("mean", c[1]), color: "var(--accent)", dots: true },
          { name: "sample", values: pick("sample", c[1]), color: "var(--baseline)", dash: "5 4", dots: true }
        ]
      });
    });
    var T = STS.BASELINE_TRAIN_HISTORY;
    api.lineChart(document.getElementById("v1-ch-train"), {
      label: "training losses", x: T.step, xMin: 0, xTicks: [0, 500, 1000, 1500, 2000], height: 230, yMin: 0, xLabel: "iteration",
      markers: [500, 1000, 1500],
      series: [
        { name: "MR-STFT", values: T.stft, color: "var(--baseline)" },
        { name: "mel", values: T.mel, color: "var(--accent)" },
        { name: "waveform L1", values: T.wave, color: "var(--ink-3)", dash: "3 3" }
      ]
    });
  };
