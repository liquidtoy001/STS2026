/* English-only website content. Preserve measured values and provenance. */
(window.STS_REPORT_NOTES = window.STS_REPORT_NOTES || {})["jacques-enc-dec"] = {
  "date": "2026-09-20",
  "trainingBy": "Jacques",
  "completedDate": "2026-09-20",
  "dateBasis": "Team-reported completion date; the new enc/dec source log says 2026-09-30. Raw timestamps are preserved for reconciliation.",
  "changed": [],
  "arms": {
    "original": "Jacques original run",
    "new-enc-dec": "New enc/dec folder (identity pending)"
  },
  "title": "Jacques: original versus new encoder/decoder",
  "lede": "Jacques was commissioned to compare original_eval and revised encoder/decoder components. Two supplied Kaggle runs completed 2,000 iterations on the same eight training utterances. Their measurements are available, but both carry identical source and model metadata; the architecture change remains unverified.",
  "chips": [
    "2,000 iterations · 1,990 updates · 10 AMP skips",
    "Reconstruction only · training batch",
    "Component identity pending verification"
  ],
  "metricsDescription": "Fixed crops from LJ001-0001…0008, 8,192 samples at 22,050 Hz (about 0.37 s). Mean and sample modes are separate; lower is better. The accompanying average covers four post-training evaluations (500, 1,000, 1,500 and 2,000), excluding initialization. Checkpoints are not independent repeats. Differences cannot yet be attributed to architecture.",
  "lateTitle": "Mean of four post-training evaluations",
  "trainingDescription": "Logged every 50 iterations; a 10-point moving average is displayed. These are random-crop training losses. Duration and KL are logged diagnostics with zero objective weights; GAN is disabled.",
  "verdict": "<ul><li><b>Completed measurements:</b> both runs finished with consistent per-sample and aggregate records.</li><li><b>Final checkpoint:</b> all six reconstruction errors are slightly higher in the new enc/dec folder; sample log-mel L1 is 0.5830 versus 0.5582.</li><li><b>Intermediate behavior:</b> the new enc/dec folder has lower errors across all six metrics at iteration 1,500. No consistent improvement or uncertainty estimate is established.</li><li><b>Provenance pending:</b> run_config files are byte-identical, including variant, model description and notebook hash. The folder name alone cannot verify new components.</li></ul>",
  "listen": "<p>Audio stays in the team's Google Drive (folder <code>Jacques Eval</code>, not part of this repository). Open each run’s <code>evaluation/Step_002000/</code> to compare matching <code>000…007_target.wav</code>, <code>_mean.wav</code> and <code>_sample.wav</code>. These short crops are reconstructions, not full-sentence synthesis. No listening verdict has been supplied.</p>",
  "findings": "<h3>Resolve executed component identity</h3><p>Request the executed notebooks/source snapshots, changed component definitions, and checkpoint/configuration. Both supplied variants are labelled <code>original</code> and carry notebook hash <code>ab8385ddd7711f0852b6a7870cd38f3d36841fafe2ccf099a871c16727778c29</code>. Stale provenance is possible; these records do not prove duplicate runs or missing edits.</p><h3>Scope of a paired component change</h3><p>If both encoder and decoder changed, only their joint effect is studied. Separate ablations and independent seeds are needed to identify individual effects and run variation.</p><p>Full analysis and the final metric table: <a href=\"../JACQUES_EVAL.md\">JACQUES_EVAL.md</a>.</p>",
  "storage": "<p>Both run folders are in the team's Google Drive, not part of this repository.</p><ul><li><code>records/</code>: configuration, status and training history.</li><li><code>evaluation/</code>: manifest, history and Step_* metrics; final-step audio stays with its metrics.</li><li><code>figures/</code>: curves; <code>screenshots/</code>: notebook captures.</li><li><code>_archive/</code>: three duplicate downloads per run, preserved.</li><li>Git records: <code>docs/records/jacques-enc-dec/</code>. Raw JSON bytes are unchanged; inventory and cloud IDs are recorded separately.</li><li>No source code or checkpoints are included in the supplied ZIP.</li></ul>",
  "team": ""
};
