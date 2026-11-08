/* English-only website content. Preserve measured values and provenance. */
(window.STS_REPORT_NOTES = window.STS_REPORT_NOTES || {})["full-train-gan"] = {
  "date": "2026-09-29",
  "trainingBy": "tuirno",
  "componentBy": "Mou and Warneke",
  "componentDate": "2026-09-23",
  "provenance": "<p>GAN developed by Mou and Warneke: <a href=\"https://github.com/liquidtoy001/STS2026/tree/gan-group\">gan-group</a> and <a href=\"https://github.com/liquidtoy001/STS2026/tree/Mou\">Mou</a> branches. Completion commit <a href=\"https://github.com/liquidtoy001/STS2026/commit/07e663dba683af7df352bdaf6b27308596b3ff94\">07e663d</a>: 2026-09-23. The <a href=\"https://github.com/liquidtoy001/STS2026/tree/gan-loss-completed\">gan-loss-completed</a> tag currently points to the earlier 2026-09-22 commit 1f3fd0f, not that completion commit. Both training runs were executed by tuirno.</p>",
  "changed": [
    "gan_weight"
  ],
  "arms": {
    "gan0": "No GAN (gan0)",
    "gan1": "GAN (gan1)"
  },
  "title": "Full training: GAN vs no-GAN",
  "lede": "Both runs trained the complete model for 50,000 iterations on LJSpeech on a Kaggle Tesla T4, using <code>notebooks/vits-full-train.ipynb</code> (benlu's merged GAN trainer). The only difference is <code>gan_weight</code>: 0 or 1.",
  "chips": [
    "Training: tuirno · GAN components: Mou and Warneke",
    "LJSpeech 13,084 clips · batch 8 · seed 2026",
    "Both completed, not resumed"
  ],
  "verdict": "<ul><li><b>The comparison is valid.</b> Dataset, clip count, seed, notebook, evaluator and evaluation batch are identical; only <code>gan_weight</code> differs.</li><li><b>On the spectral probe, no-GAN is slightly better or equal.</b> In sample mode, log-mel L1 and log-magnitude are lower without GAN at all 20 evaluation points (final log-mel L1 0.942 vs 0.973).</li><li><b>GAN wins one metric consistently:</b> mean-mode STFT log-magnitude, lower at 10 of the last 11 evaluation points.</li><li><b>These metrics cannot decide which sounds better.</b> The GAN optimises perceived quality, not spectral error. Decide by listening.</li><li><b>The discriminator overpowers the generator late in training</b> (mean real score 0.98, generated 0.02 over the last 10,000 iterations).</li></ul>",
  "listen": "<p>Audio is not stored in Git. Each run folder in the team's Google Drive (<code>full-train-2026-09-29/runs/</code>, not part of this repository) contains:</p><ul class=\"list\"><li><code>samples/step_&lt;N&gt;_sample_{1,2}.wav</code>: text-only synthesis of two fixed sentences every 5,000 iterations.</li><li><code>evaluation/step_&lt;N&gt;/</code>: target, posterior-mean and posterior-sample reconstructions of the eight probe crops at every 10,000 iterations.</li></ul><p>The folder <code>full-train-2026-09-29/report/</code> has a local HTML page with players for both runs side by side; download it together with <code>runs/</code>.</p>",
  "findings": "<h3>Discriminator too strong</h3><p>The discriminator loss stays around 0.1–0.3 for the first 10,000 iterations, then falls to mostly below 0.05 (minimum 0.001) while the generator's adversarial loss settles near 1.0. Options: a lower discriminator learning rate, fewer discriminator updates, HiFi-GAN multi-period/multi-scale discriminators, or a feature-matching loss.</p><h3>16 clips missing from training</h3><p>Both runs read 13,084 of 13,100 LJSpeech clips. <code>pd.read_csv(..., sep=\"|\")</code> treats <code>\"</code> as a CSV quote, which leaves 16 normalized transcriptions empty; <code>dropna()</code> then removes them. <code>quoting=csv.QUOTE_NONE</code> keeps all 13,100. Both runs are affected equally, so the comparison stands; fix it before the next training round.</p><h3>Cost</h3><p>No-GAN 4.69 h (0.337 s/iteration), GAN 5.21 h (0.375 s/iteration): about 11% more time and 0.12 GiB more memory with the discriminator.</p><h3>Limits of the probe</h3><p>The eight probe utterances are in the training set and the probe measures reconstruction only. It says nothing about duration prediction or prior sampling. Comparing text-to-speech quality needs a held-out split, full-sentence synthesis and a listening test or ASR word error rate.</p>",
  "storage": "<div class=\"table-wrap\"><table><thead><tr><th>Where</th><th>What</th></tr></thead><tbody><tr><td>GitHub · <code>docs/records/full-train-gan/</code></td><td>run_config, status, evaluation history, evaluation manifest and Kaggle log for each run (about 60 KB each)</td></tr><tr><td>GitHub · <code>docs/site/reports/full-train-gan.data.js</code></td><td>trimmed metrics and training curves used by this page</td></tr><tr><td>Google Drive · <code>full-train-2026-09-29/checkpoints/</code></td><td>five checkpoints per run (10k–50k; 170–221 MB each)</td></tr><tr><td>Google Drive · <code>full-train-2026-09-29/runs/</code></td><td>complete run folders: audio, per-sample metrics, full training history, curves.png</td></tr><tr><td>Google Drive · <code>full-train-2026-09-29/report/</code> and <code>logs/</code></td><td>local listening report and Kaggle logs</td></tr><tr><td>Not kept</td><td>the Kaggle result zips (fully extracted), the Kaggle source snapshot (identical to the committed notebook, hash 294ebcc…), duplicate log downloads</td></tr></tbody></table></div>",
  "team": ""
};
