# Full training: GAN vs no-GAN on Kaggle

`notebooks/vits-full-train.ipynb` trains the complete model on all of LJSpeech. It
combines the v1 model definitions with benlu's merged GAN + evaluation trainer
(2026-09-24): no DataParallel, no autocast, loss weights mel 1, STFT 0.5, wave 10,
duration 1, KL 1. The comparison is a pair of runs that differ only in `GAN_WEIGHT`
(0 and 1): 50,000 iterations each, same seed, data and settings.

These runs are not comparable with the v1 baseline numbers, which used a different
objective, loss weights and a fixed eight-utterance training batch.

## What the notebook adds

- Row-vectorised MAS. It returns the same alignment paths as the original loop (300
  random cases including ties) and ran 3–5× faster locally.
- `run_full_training`: training history every 100 iterations, fixed-crop evaluation
  every 2,500 iterations on LJ001-0001…0008, checkpoints every 5,000 (milestones every
  10,000 are kept), and text-to-speech samples for two fixed sentences at each save.
- A time budget of 11.3 h. Kaggle stops sessions at 12 h, so the notebook stops
  cleanly, saves a checkpoint and records `stopped_time_budget`.
- Resume from a stopped run. A resumed run keeps its history, but its data order
  differs from an uninterrupted run; `run_config.json` records the resume.

The evaluation utterances are part of the training data. The probe compares the two
runs; it is not a held-out or text-to-speech quality result.

## Run on Kaggle

1. Build the exports: `python tools/build_kaggle_full_train.py`. This writes
   `output/kaggle/vits-full-train_gan0.ipynb` and `..._gan1.ipynb`.
2. For each file: Kaggle → New Notebook → File → Import Notebook.
3. Add Input: an LJSpeech-1.1 dataset. The notebook finds `metadata.csv` and `wavs/`.
4. Settings → Accelerator: a GPU. Only one GPU is used.
5. Save Version → **Save & Run All (Commit)**. The run continues with the browser closed.
6. Check the log after about 15 minutes. Every 1,000 iterations it prints seconds per
   iteration and the estimated remaining time.
7. When the version finishes, download `sts2026-full-runs/` from the Output tab.

### If a run stops at the time budget

1. Open the stopped version's Output and create a dataset from it (or add that
   notebook output as Input to a new notebook).
2. Import the same export again, add both LJSpeech and the previous output as Input.
3. In the settings cell set `RESUME_FROM` to the previous run folder, for example
   `/kaggle/input/<previous-output>/sts2026-full-runs/full_gan1_<time>`.
4. Save & Run All again.

## Local smoke test

The export runs locally on the eight-clip subset from `tools/prepare_ljspeech_subset.py`:

```sh
STS_RUNTIME_ROOT=tmp/runtime STS_RUN_ROOT=tmp/runs STS_DATA_DIR=data/LJSpeech-1.1 \
STS_ALLOW_SUBSET=1 STS_NUM_WORKERS=0 STS_TOTAL_STEPS=30 STS_EVAL_EVERY=10 \
STS_SAVE_EVERY=10 STS_KEEP_EVERY=20 STS_LOG_EVERY=5 \
jupyter nbconvert --to notebook --execute output/kaggle/vits-full-train_gan1.ipynb --output tmp/smoke.ipynb
```

`STS_ALLOW_SUBSET=1` is for smoke tests only; without it the notebook refuses to train
when any LJSpeech clip is missing.

## After both runs

- Check each `status.json`: `completed`, step 50,000, and whether it was resumed.
- Upload both run folders to the [team Google Drive](https://drive.google.com/drive/folders/1RNj-DalkXNUA9ZGDNreAvGdHG8QWwDfU).
  Each keeps five 221 MB checkpoints (10k–50k); upload at least the final one.
- Add the final metrics to `docs/site/assets/results.js` with the no-GAN run as the
  reference for this pair, and listen to the samples before writing conclusions.
