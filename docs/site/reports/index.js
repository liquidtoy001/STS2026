/* English-only website content. Preserve measured values and provenance. */
window.STS_REPORT_INDEX = [
  {
    "id": "full-train-gan",
    "date": "2026-09-29",
    "title": "Full training: GAN vs no-GAN (50,000 iterations)",
    "summary": "Two Kaggle T4 runs that differ only in gan_weight. Spectral probe metrics slightly favour no-GAN; the discriminator overpowers the generator late in training."
  },
  {
    "id": "jacques-enc-dec",
    "date": "2026-09-20",
    "title": "Jacques: original vs new enc/dec (2,000 iterations)",
    "summary": "Two completed fixed-batch reconstruction runs. Final errors are slightly higher in the new enc/dec folder; identical source metadata leaves component identity pending verification."
  }
];
