/*
 * Hand-written narrative for report.html#full-train-gan. Numbers and charts come from
 * full-train-gan.data.js (generated). Each text field is { en, zh, de?, fr? }; English is the fallback.
 * Put the team's listening verdict and discussion in `team`.
 */
(window.STS_REPORT_NOTES = window.STS_REPORT_NOTES || {})["full-train-gan"] = {
  date: "2026-09-29",
  changed: ["gan_weight"],
  arms: {
    gan0: { en: "No GAN (gan0)", zh: "无 GAN（gan0）", de: "Ohne GAN (gan0)", fr: "Sans GAN (gan0)" },
    gan1: { en: "GAN (gan1)", zh: "有 GAN（gan1）", de: "Mit GAN (gan1)", fr: "Avec GAN (gan1)" }
  },
  title: {
    en: "Full training: GAN vs no-GAN",
    zh: "全量训练：有 GAN 与无 GAN",
    de: "Volltraining: mit und ohne GAN",
    fr: "Entraînement complet : avec et sans GAN"
  },
  lede: {
    en: "Both runs trained the complete model for 50,000 iterations on LJSpeech on a Kaggle Tesla T4, using <code>notebooks/vits-full-train.ipynb</code> (benlu's merged GAN trainer). The only difference is <code>gan_weight</code>: 0 or 1.",
    zh: "两次运行都在 Kaggle Tesla T4 上，用 <code>notebooks/vits-full-train.ipynb</code>（benlu 合并的 GAN 训练器）在 LJSpeech 上训练完整模型 50,000 次迭代。唯一的差别是 <code>gan_weight</code>：0 或 1。"
  },
  chips: [
    { en: "LJSpeech 13,084 clips · batch 8 · seed 2026", zh: "LJSpeech 13,084 条 · batch 8 · seed 2026" },
    { en: "Both completed, not resumed", zh: "两次均完成，未续跑" }
  ],
  verdict: {
    en: "<ul>" +
      "<li><b>The comparison is valid.</b> Dataset, clip count, seed, notebook, evaluator and evaluation batch are identical; only <code>gan_weight</code> differs.</li>" +
      "<li><b>On the spectral probe, no-GAN is slightly better or equal.</b> In sample mode, log-mel L1 and log-magnitude are lower without GAN at all 20 evaluation points (final log-mel L1 0.942 vs 0.973).</li>" +
      "<li><b>GAN wins one metric consistently:</b> mean-mode STFT log-magnitude, lower at 10 of the last 11 evaluation points.</li>" +
      "<li><b>These metrics cannot decide which sounds better.</b> The GAN optimises perceived quality, not spectral error. Decide by listening.</li>" +
      "<li><b>The discriminator overpowers the generator late in training</b> (mean real score 0.98, generated 0.02 over the last 10,000 iterations).</li>" +
      "</ul>",
    zh: "<ul>" +
      "<li><b>这组对比成立。</b>数据集、条数、随机种子、notebook、评估代码和评估批次完全一致，只有 <code>gan_weight</code> 不同。</li>" +
      "<li><b>按频谱类指标，无 GAN 略好或打平。</b>sample 模式下 log-mel L1 和 log-magnitude 在全部 20 个评估点上都是无 GAN 更低（最终 log-mel L1：0.942 对 0.973）。</li>" +
      "<li><b>有 GAN 只在一项上稳定占优：</b>mean 模式的 STFT log-magnitude，后 11 个评估点中赢 10 个。</li>" +
      "<li><b>这些指标判断不了谁更好听。</b>GAN 优化的是听感，不是频谱误差，要以试听为准。</li>" +
      "<li><b>训练后期判别器压制了生成器</b>（最后 1 万步真实音频平均得分 0.98，生成音频 0.02）。</li>" +
      "</ul>"
  },
  listen: {
    en: "<p>Audio is not stored in Git. Each run folder on the <a href=\"https://drive.google.com/drive/folders/1RNj-DalkXNUA9ZGDNreAvGdHG8QWwDfU\">team Google Drive</a> (<code>full-train-2026-09-29/runs/</code>) contains:</p>" +
      "<ul class=\"list\"><li><code>samples/step_&lt;N&gt;_sample_{1,2}.wav</code>: text-only synthesis of two fixed sentences every 5,000 iterations.</li>" +
      "<li><code>evaluation/step_&lt;N&gt;/</code>: target, posterior-mean and posterior-sample reconstructions of the eight probe crops at every 10,000 iterations.</li></ul>" +
      "<p>The folder <code>full-train-2026-09-29/report/</code> has a local HTML page with players for both runs side by side; download it together with <code>runs/</code>.</p>",
    zh: "<p>音频不进 Git。<a href=\"https://drive.google.com/drive/folders/1RNj-DalkXNUA9ZGDNreAvGdHG8QWwDfU\">团队谷歌云盘</a>里每个运行目录（<code>full-train-2026-09-29/runs/</code>）包含：</p>" +
      "<ul class=\"list\"><li><code>samples/step_&lt;N&gt;_sample_{1,2}.wav</code>：每 5,000 步用两个固定句子做的纯文本合成。</li>" +
      "<li><code>evaluation/step_&lt;N&gt;/</code>：每 10,000 步 8 个探针裁剪的目标、后验均值和后验采样重建。</li></ul>" +
      "<p><code>full-train-2026-09-29/report/</code> 里有一个本地 HTML 页面，可以并排试听两次运行；需要和 <code>runs/</code> 一起下载后打开。</p>"
  },
  findings: {
    en: "<h3>Discriminator too strong</h3><p>The discriminator loss stays around 0.1–0.3 for the first 10,000 iterations, then falls to mostly below 0.05 (minimum 0.001) while the generator's adversarial loss settles near 1.0. Options: a lower discriminator learning rate, fewer discriminator updates, HiFi-GAN multi-period/multi-scale discriminators, or a feature-matching loss.</p>" +
      "<h3>16 clips missing from training</h3><p>Both runs read 13,084 of 13,100 LJSpeech clips. <code>pd.read_csv(..., sep=\"|\")</code> treats <code>\"</code> as a CSV quote, which leaves 16 normalized transcriptions empty; <code>dropna()</code> then removes them. <code>quoting=csv.QUOTE_NONE</code> keeps all 13,100. Both runs are affected equally, so the comparison stands; fix it before the next training round.</p>" +
      "<h3>Cost</h3><p>No-GAN 4.69 h (0.337 s/iteration), GAN 5.21 h (0.375 s/iteration): about 11% more time and 0.12 GiB more memory with the discriminator.</p>" +
      "<h3>Limits of the probe</h3><p>The eight probe utterances are in the training set and the probe measures reconstruction only. It says nothing about duration prediction or prior sampling. Comparing text-to-speech quality needs a held-out split, full-sentence synthesis and a listening test or ASR word error rate.</p>",
    zh: "<h3>判别器过强</h3><p>前 10,000 步判别器损失在 0.1–0.3 左右，之后大多降到 0.05 以下（最低 0.001），而生成器的对抗损失停在 1.0 附近。可以考虑：降低判别器学习率、减少判别器更新次数、改用 HiFi-GAN 的多周期/多尺度判别器，或加入特征匹配损失。</p>" +
      "<h3>训练少了 16 条</h3><p>两次运行都只读到 LJSpeech 13,100 条中的 13,084 条。<code>pd.read_csv(..., sep=\"|\")</code> 默认把 <code>\"</code> 当作 CSV 引用符，导致 16 条 normalized 文本为空，随后被 <code>dropna()</code> 删掉。改用 <code>quoting=csv.QUOTE_NONE</code> 可以保留全部 13,100 条。两次运行受影响相同，对比仍然成立；下一轮训练前应修复。</p>" +
      "<h3>成本</h3><p>无 GAN 4.69 小时（0.337 秒/步），有 GAN 5.21 小时（0.375 秒/步）：判别器多花约 11% 时间和 0.12 GiB 显存。</p>" +
      "<h3>评估探针的局限</h3><p>这 8 条探针语音在训练集中，而且只衡量重建，不反映时长预测和先验采样。要比较 TTS 质量，需要留出集、整句合成，以及试听或 ASR 词错率。</p>"
  },
  storage: {
    en: "<div class=\"table-wrap\"><table><thead><tr><th>Where</th><th>What</th></tr></thead><tbody>" +
      "<tr><td>GitHub · <code>docs/records/full-train-gan/</code></td><td>run_config, status, evaluation history, evaluation manifest and Kaggle log for each run (about 60 KB each)</td></tr>" +
      "<tr><td>GitHub · <code>docs/site/reports/full-train-gan.data.js</code></td><td>trimmed metrics and training curves used by this page</td></tr>" +
      "<tr><td>Google Drive · <code>full-train-2026-09-29/checkpoints/</code></td><td>five checkpoints per run (10k–50k; 170–221 MB each)</td></tr>" +
      "<tr><td>Google Drive · <code>full-train-2026-09-29/runs/</code></td><td>complete run folders: audio, per-sample metrics, full training history, curves.png</td></tr>" +
      "<tr><td>Google Drive · <code>full-train-2026-09-29/report/</code> and <code>logs/</code></td><td>local listening report and Kaggle logs</td></tr>" +
      "<tr><td>Not kept</td><td>the Kaggle result zips (fully extracted), the Kaggle source snapshot (identical to the committed notebook, hash 294ebcc…), duplicate log downloads</td></tr>" +
      "</tbody></table></div>",
    zh: "<div class=\"table-wrap\"><table><thead><tr><th>位置</th><th>内容</th></tr></thead><tbody>" +
      "<tr><td>GitHub · <code>docs/records/full-train-gan/</code></td><td>每次运行的 run_config、status、评估历史、评估清单和 Kaggle 日志（各约 60 KB）</td></tr>" +
      "<tr><td>GitHub · <code>docs/site/reports/full-train-gan.data.js</code></td><td>本页使用的精简指标和训练曲线</td></tr>" +
      "<tr><td>谷歌云盘 · <code>full-train-2026-09-29/checkpoints/</code></td><td>每次运行 5 个检查点（10k–50k，每个 170–221 MB）</td></tr>" +
      "<tr><td>谷歌云盘 · <code>full-train-2026-09-29/runs/</code></td><td>完整运行目录：音频、逐条指标、完整训练历史、curves.png</td></tr>" +
      "<tr><td>谷歌云盘 · <code>full-train-2026-09-29/report/</code> 和 <code>logs/</code></td><td>本地试听报告和 Kaggle 日志</td></tr>" +
      "<tr><td>不保留</td><td>Kaggle 结果 zip（已完整解压）、Kaggle 源码快照（与已提交的 notebook 相同，哈希 294ebcc…）、重复下载的日志</td></tr>" +
      "</tbody></table></div>"
  },
  team: ""
};
