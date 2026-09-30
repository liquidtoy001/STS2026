/* List of experiment reports shown at report.html. Newest first. Titles/summaries: en required, other languages optional. */
window.STS_REPORT_INDEX = [
  {
    id: "full-train-gan",
    date: "2026-09-29",
    title: { en: "Full training: GAN vs no-GAN (50,000 iterations)", zh: "全量训练：有 GAN 与无 GAN（各 50,000 次迭代）" },
    summary: {
      en: "Two Kaggle T4 runs that differ only in gan_weight. Spectral probe metrics slightly favour no-GAN; the discriminator overpowers the generator late in training.",
      zh: "两次只有 gan_weight 不同的 Kaggle T4 运行。频谱类指标上无 GAN 略好；训练后期判别器压制了生成器。"
    }
  }
];
