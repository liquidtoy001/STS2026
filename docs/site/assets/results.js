/*
 * STS2026 measured results and experiment registry.
 *
 * This is the ONE file to edit when a new comparison run finishes.
 * The "Improvements" page builds one comparison table per entry in GROUPS, with deltas
 * relative to that group's reference run. Detailed experiment reports are generated with
 * tools/build_site_report.py and shown at report.html#<id>.
 *
 * Rules (from docs/EVALUATION.md):
 *  - Copy numbers from the run's evaluation/summary.json (step 2000). Never round by hand.
 *  - Leave a metric as null until it has been measured. null renders as "not measured".
 *  - Record iterations AND actual optimizer updates (AMP can skip updates).
 *  - Only runs with status "completed" count as comparisons.
 */
window.STS = window.STS || {};

STS.BASELINE_ID = "v1-original-eval";

/*
 * Comparison groups. Each group has its own protocol and reference run; deltas are
 * only computed inside a group. Runs from different groups are never compared.
 */
STS.GROUPS = [
  {
    id: "v1-overfit",
    title: "v1 protocol: 2,000-iteration overfit on 8 utterances",
    note: "Reconstruction-only objective (duration and KL weight 0), AMP, measured locally on 2026-09-19.",
    reference: "v1-original-eval",
    i18n: {
      zh: { title: "v1 协议：在 8 条语音上过拟合 2,000 次迭代", note: "只训练重建（duration 和 KL 权重为 0），开启 AMP，2026-09-19 本地实测。" },
      de: { title: "v1-Protokoll: 2.000 Iterationen Overfitting auf 8 Äußerungen", note: "Nur Rekonstruktion (duration und KL mit Gewicht 0), AMP, lokal gemessen am 2026-09-19." },
      fr: { title: "Protocole v1 : 2 000 itérations de surapprentissage sur 8 énoncés", note: "Reconstruction seule (duration et KL de poids 0), AMP, mesuré localement le 2026-09-19." }
    }
  },
  {
    id: "full-train",
    title: "Full training: 50,000 iterations on all of LJSpeech",
    note: "Complete objective (mel 1, STFT 0.5, wave 10, duration 1, KL 1), no AMP, Kaggle T4. Probe: same 8 fixed crops (in the training set).",
    reference: "full-gan0",
    report: "report.html#full-train-gan",
    i18n: {
      zh: { title: "全量训练：在全部 LJSpeech 上训练 50,000 次迭代", note: "完整训练目标（mel 1、STFT 0.5、wave 10、duration 1、KL 1），不开 AMP，Kaggle T4。评估探针：同样的 8 个固定裁剪（在训练集中）。" },
      de: { title: "Volltraining: 50.000 Iterationen auf ganz LJSpeech", note: "Vollständiges Ziel (mel 1, STFT 0,5, wave 10, duration 1, KL 1), ohne AMP, Kaggle T4. Probe: dieselben 8 festen Ausschnitte (im Trainingsset)." },
      fr: { title: "Entraînement complet : 50 000 itérations sur tout LJSpeech", note: "Objectif complet (mel 1, STFT 0,5, wave 10, duration 1, KL 1), sans AMP, Kaggle T4. Sonde : les 8 mêmes extraits fixes (dans l’ensemble d’entraînement)." }
    }
  }
];

/* Source: output/local-eval/original_20260919T151630615757Z/evaluation/evaluation_history.json */
STS.BASELINE_EVAL_HISTORY = [
  { step: 0,    mode: "mean",   log_mel_l1: 3.3404852151870728, stft_sc: 2.202135920524597,  stft_logmag: 3.5854020714759827 },
  { step: 0,    mode: "sample", log_mel_l1: 1.9617528468370438, stft_sc: 2.196268618106842,  stft_logmag: 2.0086960196495056 },
  { step: 500,  mode: "mean",   log_mel_l1: 0.8649029061198235, stft_sc: 0.9182565435767174, stft_logmag: 1.4590812772512436 },
  { step: 500,  mode: "sample", log_mel_l1: 0.8270140439271927, stft_sc: 0.9169581010937691, stft_logmag: 1.2963262051343918 },
  { step: 1000, mode: "mean",   log_mel_l1: 0.7155695483088493, stft_sc: 0.8933847695589066, stft_logmag: 1.2822148203849792 },
  { step: 1000, mode: "sample", log_mel_l1: 0.7048437222838402, stft_sc: 0.8926491439342499, stft_logmag: 1.1893363744020462 },
  { step: 1500, mode: "mean",   log_mel_l1: 0.636958584189415,  stft_sc: 0.8717164024710655, stft_logmag: 1.2314483374357224 },
  { step: 1500, mode: "sample", log_mel_l1: 0.6340814046561718, stft_sc: 0.8709701150655746, stft_logmag: 1.1652337312698364 },
  { step: 2000, mode: "mean",   log_mel_l1: 0.5600882470607758, stft_sc: 0.8627395927906036, stft_logmag: 1.1601064205169678 },
  { step: 2000, mode: "sample", log_mel_l1: 0.55783960968256,   stft_sc: 0.8618461862206459, stft_logmag: 1.1059232503175735 }
];

/* Source: same run, training_history.json. Training losses use random crops (not the fixed eval crops). */
STS.BASELINE_TRAIN_HISTORY = {
  step:  [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000],
  total: [17.2909,16.2573,16.263,14.8959,15.5912,14.357,14.7261,13.1725,12.8202,12.1574,11.7941,12.1617,11.0266,11.109,11.0002,10.9658,10.5812,9.9264,10.714,10.4851,10.4917,10.3389,10.1129,10.155,9.4306,8.9603,9.9872,9.2977,8.7485,9.0136,9.2818,8.9053,9.8464,9.3545,8.403,9.2022,8.8526,9.0105,8.9827,8.9998],
  mel:   [1.3921,1.3039,1.311,1.1787,1.2474,1.1245,1.1613,1.0061,0.9824,0.9203,0.8902,0.9181,0.8242,0.8388,0.8169,0.822,0.7815,0.7377,0.7913,0.7757,0.7855,0.7642,0.7394,0.7456,0.6824,0.6532,0.7383,0.6691,0.6428,0.6473,0.6668,0.6402,0.7004,0.6749,0.5927,0.654,0.6312,0.6418,0.6398,0.6272],
  stft:  [2.7668,2.6549,2.6991,2.588,2.6563,2.4775,2.5335,2.3857,2.3796,2.2711,2.2702,2.2883,2.1724,2.2246,2.1497,2.15,2.2074,2.0547,2.1366,2.1145,2.1205,2.0763,2.0667,2.0461,1.9986,2.0052,2.0646,2.0012,1.9358,1.9819,1.9907,2.0055,2.0563,1.9655,1.9286,2.0021,1.9633,1.951,1.937,1.9607],
  wave:  [0.0603,0.0563,0.0454,0.0521,0.0461,0.0634,0.058,0.0726,0.0617,0.0684,0.0622,0.0692,0.0612,0.0497,0.0682,0.0596,0.0558,0.0494,0.0665,0.0614,0.0516,0.0621,0.0652,0.0653,0.0608,0.0423,0.054,0.0605,0.0384,0.0559,0.0623,0.0498,0.0786,0.064,0.0547,0.066,0.0578,0.0641,0.0647,0.0767]
};

/*
 * Experiment registry. Add one object per completed (or planned) comparison.
 * final: step-2000 summary, per mode. Lower is better for every metric.
 * status: "completed" | "running" | "in-progress" | "planned" | "failed"
 * i18n: optional translated { name, change } per language code (zh, de, fr).
 */
STS.EXPERIMENTS = [
  {
    id: "v1-original-eval",
    group: "v1-overfit",
    i18n: {
      zh: { name: "原始模型（评估基准）", change: "无：冻结的参照" },
      de: { name: "Original (Evaluations-Baseline)", change: "Keine: eingefrorene Referenz" },
      fr: { name: "Original (référence d’évaluation)", change: "Aucun : référence figée" }
    },
    name: "Original (eval baseline)",
    change: "None: frozen reference",
    branch: "main",
    status: "completed",
    params: 12141186,
    iterations: 2000,
    updates: 1988,
    ampSkipped: 12,
    seconds: 640.92,
    peakAllocGiB: 0.789,
    final: {
      mean:   { log_mel_l1: 0.5600882470607758, stft_sc: 0.8627395927906036, stft_logmag: 1.1601064205169678 },
      sample: { log_mel_l1: 0.55783960968256,   stft_sc: 0.8618461862206459, stft_logmag: 1.1059232503175735 }
    },
    evidence: "output/local-eval/original_20260919T151630615757Z/"
  },
  {
    id: "full-gan0",
    group: "full-train",
    name: "Full training, no GAN (gan_weight = 0)",
    change: "Reference for this pair: full model, 50,000 iterations on all of LJSpeech",
    branch: "Turino",
    i18n: {
      zh: { name: "全量训练，无 GAN（gan_weight = 0）", change: "本组参照：完整模型，在全部 LJSpeech 上训练 50,000 次迭代" },
      de: { name: "Volltraining ohne GAN (gan_weight = 0)", change: "Referenz dieses Paars: vollständiges Modell, 50.000 Iterationen auf ganz LJSpeech" },
      fr: { name: "Entraînement complet sans GAN (gan_weight = 0)", change: "Référence de la paire : modèle complet, 50 000 itérations sur tout LJSpeech" }
    },
    status: "completed",
    params: 12141186, iterations: 50000, updates: 50000, ampSkipped: 0,
    seconds: 16874.4, peakAllocGiB: 1.134,
    final: {
      mean:   { log_mel_l1: 1.2183219343423843, stft_sc: 0.9410169497132301, stft_logmag: 1.7540749609470367 },
      sample: { log_mel_l1: 0.9420206770300865, stft_sc: 0.9165768325328827, stft_logmag: 1.1164949089288712 }
    },
    evidence: "Drive: full-train-2026-09-29/runs/full_gan0_20260929T175436031732Z; docs/records/full-train-gan/gan0"
  },
  {
    id: "full-gan1",
    group: "full-train",
    name: "Full training + GAN (LSGAN, gan_weight = 1)",
    change: "Adds the 1-D conv discriminator and adversarial loss; everything else identical",
    branch: "Turino",
    i18n: {
      zh: { name: "全量训练 + GAN（LSGAN，gan_weight = 1）", change: "加入一维卷积判别器和对抗损失，其余完全相同" },
      de: { name: "Volltraining + GAN (LSGAN, gan_weight = 1)", change: "Ergänzt 1-D-Conv-Diskriminator und adversariellen Loss; sonst identisch" },
      fr: { name: "Entraînement complet + GAN (LSGAN, gan_weight = 1)", change: "Ajoute le discriminateur conv 1-D et la perte adversariale ; le reste est identique" }
    },
    status: "completed",
    params: 12141186, iterations: 50000, updates: 50000, ampSkipped: 0,
    seconds: 18748.1, peakAllocGiB: 1.253,
    final: {
      mean:   { log_mel_l1: 1.3442358523607254, stft_sc: 0.9474663287401199, stft_logmag: 1.7193482369184494 },
      sample: { log_mel_l1: 0.973286584019661, stft_sc: 0.9181935712695122, stft_logmag: 1.1861073970794678 }
    },
    evidence: "Drive: full-train-2026-09-29/runs/full_gan1_20260929T170528763415Z; docs/records/full-train-gan/gan1"
  },
  {
    id: "v3-new-encoder",
    group: "full-train",
    i18n: {
      zh: { name: "新的后验编码器", change: "替换 PosteriorEncoder；解码器不变" },
      de: { name: "Neuer Posterior-Encoder", change: "Ersetzt PosteriorEncoder; Decoder unverändert" },
      fr: { name: "Nouvel encodeur a posteriori", change: "Remplace PosteriorEncoder ; décodeur inchangé" }
    },
    name: "New posterior encoder",
    change: "Replaces PosteriorEncoder; decoder unchanged",
    branch: "TBD",
    status: "planned",
    params: null, iterations: null, updates: null, ampSkipped: null, seconds: null, peakAllocGiB: null,
    final: { mean: { log_mel_l1: null, stft_sc: null, stft_logmag: null }, sample: { log_mel_l1: null, stft_sc: null, stft_logmag: null } },
    evidence: null
  },
  {
    id: "v4-new-decoder",
    group: "full-train",
    i18n: {
      zh: { name: "新的波形解码器", change: "替换 Decoder；编码器不变" },
      de: { name: "Neuer Waveform-Decoder", change: "Ersetzt Decoder; Encoder unverändert" },
      fr: { name: "Nouveau décodeur de forme d’onde", change: "Remplace Decoder ; encodeur inchangé" }
    },
    name: "New waveform decoder",
    change: "Replaces Decoder; encoder unchanged",
    branch: "TBD",
    status: "planned",
    params: null, iterations: null, updates: null, ampSkipped: null, seconds: null, peakAllocGiB: null,
    final: { mean: { log_mel_l1: null, stft_sc: null, stft_logmag: null }, sample: { log_mel_l1: null, stft_sc: null, stft_logmag: null } },
    evidence: null
  }
];
