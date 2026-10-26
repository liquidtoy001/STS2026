# Chunk-aware fine-tuning: run manual

For the teammate who runs the two fine-tuning runs. Hands-on time about 30 minutes; then 1–3 hours
unattended on an L40S or RTX PRO 6000, or about 5 hours on Kaggle T4 ×2. Why these runs exist:
[VOICE_AGENT_TTS.md](VOICE_AGENT_TTS.md); what the options do and the local 5,000-step check:
[CHUNK_FINETUNE.md](CHUNK_FINETUNE.md).

## What you will run

Two runs of arm R0, both starting from the finished R0 checkpoint at 40,000 steps and training
20,000 more steps. They differ only in chunk-aware training:

| Run tag | Chunk-aware training | Checkpoints kept |
|---|---|---|
| `ftchunk20k` | on: `--chunk-prob 0.5` (1–8 words, half of the crops are sentence prefixes) | 5k, 10k, 15k, 20k |
| `ftctrl20k` | off (the control: shows what 20,000 more steps do without chunks) | 5k, 10k, 15k, 20k |

The 5,000-step checkpoints repeat the local check on another machine; 10k–20k show whether
longer fine-tuning helps. Everything else (batch 16, learning rate 2e-4, fp16, seed 2026,
optimizer state carried over from R0) is fixed by arm R0 and the defaults. Do not change it.

Checklist: ☐ machine ready ☐ code ☐ data and R0 checkpoint verified ☐ durations file ☐ smoke
test ☐ both runs started ☐ both completed ☐ uploaded ☐ run cards posted.

Pick the section for your machine: **A** Linux GPU server (L40S, RTX PRO 6000, …), **B** Google
Colab, **C** Kaggle T4 ×2. The R0 and O1 runs were recorded with data in `/content/LJSpeech-1.1`
(a Google Colab path) on an RTX PRO 6000; if that is where you train, use B.

## A. Linux GPU server

### A1. Check the machine

```sh
nvidia-smi                 # a GPU with at least 8 GB free; note what else is running
df -h .                    # at least 10 GB free (two runs × four checkpoints of 815 MB, plus logs)
python3 --version          # 3.10 or newer
```

### A2. Python environment

If you trained R0 or O1 on this machine, activate that environment and go to A3. Otherwise:

```sh
python3 -m venv ~/sts-venv
source ~/sts-venv/bin/activate
python -m pip install --upgrade pip
# PyTorch + TorchAudio for this machine's CUDA driver: copy the command from
# https://pytorch.org/get-started/locally/ (Linux, Pip, CUDA)
python -c "import torch, torchaudio; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU name. The remaining packages are installed in A3.

### A3. Code

```sh
git clone https://github.com/liquidtoy001/STS2026.git    # skip if you already have the repository
cd STS2026
git fetch origin
git checkout --detach origin/Turino
git rev-parse --short HEAD                               # write this commit on the run card
python -m pip install -r requirements/notebook.txt      # only for a new environment (A2)
python tools/train_tts010.py --help | grep -e --init-from -e --chunk-prob
```

The last command must print both options; if it prints nothing, the checkout is too old.

### A4. Data and the R0 checkpoint

You need the complete LJSpeech-1.1 (the same folder as for R0) and the R0 checkpoint
`tts010_R0_20261014T070647303632Z/checkpoints/vits_step_40000.pt` (815 MB, team Google Drive).
Set two variables for the rest of this section and check both files:

```sh
DATA=/path/to/LJSpeech-1.1
R0=/path/to/vits_step_40000.pt
sha256sum $DATA/metadata.csv     # 852bce5c79e5184ba5bf2ea98aad4bcf49a5d787def5fd1f04fa2ab8aa6dc535
ls $DATA/wavs | wc -l            # 13100
sha256sum $R0                    # 975cff7157966f9379b9f7ed63964a9c78c71054c281c210748467f17b54f81e
```

If a hash differs, stop and ask Turino: a different checkpoint or metadata makes the runs
incomparable.

### A5. Alignment durations (about 2 minutes)

```sh
mkdir -p runs-ft
python tools/chunk_alignments.py --checkpoint $R0 --data-dir $DATA \
    --out runs-ft/chunk_durations_R0_40k.json 2>&1 | tee runs-ft/chunk_alignments.log
```

The last line reads `13000 clips in … s -> runs-ft/chunk_durations_R0_40k.json sha256 …`. The
local file had `cec409275b4d589e7b9fef3ce743ffcf310f6be14798fc94826f55f617fc30ba`; another GPU
may break an alignment tie differently and give another hash. That is fine, because each run
records the hash of the file it used. Write the hash on the run card.

### A6. Smoke test (about 1 minute; never report it)

```sh
python tools/train_tts010.py --arm R0 --data-dir $DATA --run-root runs-smoke --tag smoke \
    --init-from $R0 --steps 30 --chunk-prob 0.5 --chunk-durations runs-ft/chunk_durations_R0_40k.json \
    2>&1 | tee runs-smoke.log
```

Check, in this order:

1. `Chunk-aware training: {'prob': 0.5, ... 'clips': 13000, ...}` near the top.
2. `Initialised from vits_step_40000.pt (step 40000), optimizer state carried; this run starts at step 0.`
3. `probe evaluation at 0: {'mean': 0.5637, ...` (the R0 checkpoint's value, so the right
   checkpoint was loaded; the last digit may differ on another GPU).
4. `"status": "completed"` and the command returns to the prompt.

Then delete the smoke run: `rm -rf runs-smoke runs-smoke.log` (it holds an 815 MB checkpoint).

### A7. Start both runs

Use `tmux` so the runs survive a dropped SSH connection:

```sh
tmux new -s ft
source ~/sts-venv/bin/activate      # if you use a venv
cd STS2026
DATA=/path/to/LJSpeech-1.1
R0=/path/to/vits_step_40000.pt
python tools/train_tts010.py --arm R0 --data-dir $DATA --run-root runs-ft --tag ftchunk20k \
    --init-from $R0 --steps 20000 --keep-every 5000 \
    --chunk-prob 0.5 --chunk-durations runs-ft/chunk_durations_R0_40k.json \
    > runs-ft/ftchunk20k.log 2>&1 &
python tools/train_tts010.py --arm R0 --data-dir $DATA --run-root runs-ft --tag ftctrl20k \
    --init-from $R0 --steps 20000 --keep-every 5000 \
    > runs-ft/ftctrl20k.log 2>&1 &
wait
```

Detach with `Ctrl-b d`; reattach later with `tmux attach -t ft`. Both runs on one GPU need about
3.5 GB of GPU memory each. If the GPU is shared with other jobs and runs out of memory, start the
second run only after the first has finished.

### A8. Watch the runs

```sh
tail -f runs-ft/ftchunk20k.log      # Ctrl-c stops watching, not the run
grep "step .*/20000" runs-ft/*.log | tail -4
nvidia-smi
```

- Every 1,000 steps a line `step N/20000 total … s/it ETA … h`; every 2,000 steps the probe
  values. The first ETA, after 1,000 steps, is the time to plan with.
- A few skipped fp16 steps at the start (`amp_backoffs`) are normal.
- `nonfinite_skips` must stay 0; 20 non-finite steps in a row stop the run with
  `failed_or_interrupted`. Report that instead of restarting silently.
- The run folders are `runs-ft/tts010_R0_ftchunk20k_<UTC time>/` and
  `runs-ft/tts010_R0_ftctrl20k_<UTC time>/`.

### A9. If something goes wrong

| Message or symptom | Meaning | What to do |
|---|---|---|
| `metadata.csv lacks held-out IDs` or `… wav files are missing` | incomplete LJSpeech | use the complete LJSpeech-1.1 (A4 checks) |
| `Checkpoint switches … differ …; initialisation refused` | not the R0 checkpoint | use the R0 file with the hash in A4 |
| `… has no optimizer state` | a weights-only file | use the R0 run's own checkpoint (A4) |
| `Chunk durations lack … clips` or `… do not match its text` | durations made from other data | rerun A5 with the same `$DATA` |
| `CUDA out of memory` | GPU shared with other jobs | run the two runs one after the other |
| the run stopped (machine restart, time budget) | resume it | the same A7 command for that run with `--init-from $R0` replaced by `--resume runs-ft/<its run folder>`, all other options unchanged |
| `Cannot resume: … differs from the previous run` | the resume command changed a setting | copy the original command exactly, swapping only `--init-from` for `--resume` |
| `failed_or_interrupted` in `status.json` | crash or non-finite losses | send the log to Turino |

### A10. Finish, upload, report

1. Both `status.json` files say `"status": "completed"` and `"step": 20000`:
   `cat runs-ft/tts010_R0_ft*/status.json`.
2. Upload to the team Google Drive under `tts010-chunk-ft/`:
   - both **whole run folders** (each with `checkpoints/vits_step_{5000,10000,15000,20000}.pt`,
     `run_config.json`, `status.json`, `training_history.json`, `evaluation/`,
     `evaluation_heldout/`, `samples/`, `curves.png`);
   - `runs-ft/ftchunk20k.log`, `runs-ft/ftctrl20k.log`, `runs-ft/chunk_alignments.log`;
   - `runs-ft/chunk_durations_R0_40k.json`.
3. Post one run card per run in the team chat (template at the end).
4. Stop or release the GPU machine.

Evaluation is central (Turino, one machine, one script), as for the ablation.

## B. Google Colab

Runtime: a GPU runtime (the more memory the better; T4 works but is slow). Files under
`/content` disappear when the runtime ends, so the run folders go to Google Drive at the end.

1. Mount Drive (cell):

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

2. Code, data, checkpoint (cell). Put LJSpeech-1.1 at `/content/LJSpeech-1.1` the same way as
   for R0, and copy the R0 checkpoint from Drive:

   ```sh
   %%bash
   cd /content && git clone https://github.com/liquidtoy001/STS2026.git
   cd STS2026 && git checkout --detach origin/Turino && git rev-parse --short HEAD
   mkdir -p /content/R0_40k
   cp "/content/drive/MyDrive/<path to>/vits_step_40000.pt" /content/R0_40k/
   sha256sum /content/LJSpeech-1.1/metadata.csv /content/R0_40k/vits_step_40000.pt
   ls /content/LJSpeech-1.1/wavs | wc -l
   ```

   Compare with the hashes and count in A4.

3. Durations and smoke test (cell); check the output as in A5 and A6:

   ```sh
   %%bash
   cd /content/STS2026
   DATA=/content/LJSpeech-1.1; R0=/content/R0_40k/vits_step_40000.pt
   python tools/chunk_alignments.py --checkpoint $R0 --data-dir $DATA --out /content/runs-ft/chunk_durations_R0_40k.json
   python tools/train_tts010.py --arm R0 --data-dir $DATA --run-root /content/runs-smoke --tag smoke \
       --init-from $R0 --steps 30 --chunk-prob 0.5 --chunk-durations /content/runs-ft/chunk_durations_R0_40k.json
   rm -rf /content/runs-smoke
   ```

4. Both runs (cell). Set `--time-budget-hours` to the runtime you have left minus 0.5 h; a run
   that reaches it saves a checkpoint and stops cleanly, and can be resumed in a new runtime
   (A9) after copying its folder back from Drive.

   ```sh
   %%bash
   cd /content/STS2026
   DATA=/content/LJSpeech-1.1; R0=/content/R0_40k/vits_step_40000.pt; B=11
   python tools/train_tts010.py --arm R0 --data-dir $DATA --run-root /content/runs-ft --tag ftchunk20k \
       --init-from $R0 --steps 20000 --keep-every 5000 --time-budget-hours $B \
       --chunk-prob 0.5 --chunk-durations /content/runs-ft/chunk_durations_R0_40k.json \
       > /content/runs-ft/ftchunk20k.log 2>&1 &
   python tools/train_tts010.py --arm R0 --data-dir $DATA --run-root /content/runs-ft --tag ftctrl20k \
       --init-from $R0 --steps 20000 --keep-every 5000 --time-budget-hours $B \
       > /content/runs-ft/ftctrl20k.log 2>&1 &
   wait
   cp -r /content/runs-ft "/content/drive/MyDrive/tts010-chunk-ft"
   ```

   In another cell, `!grep "step .*/20000" /content/runs-ft/*.log | tail -4` shows progress.

5. Check and report as in A10 (the folders are already on Drive).

## C. Kaggle T4 ×2

1. Create a private Kaggle dataset containing `vits_step_40000.pt` (check its hash first, A4).
2. New notebook: Accelerator **GPU T4 ×2**, Internet **on**, add the LJSpeech-1.1 dataset and
   your checkpoint dataset as inputs.
3. One cell, then **Save Version → Save & Run All (Commit)** so it runs without the browser:

   ```sh
   %%bash
   cd /kaggle/working && git clone https://github.com/liquidtoy001/STS2026.git && cd STS2026
   git checkout --detach origin/Turino && git rev-parse --short HEAD
   for m in $(find /kaggle/input -name metadata.csv); do [ -d "$(dirname $m)/wavs" ] && DATA=$(dirname $m) && break; done
   R0=$(find /kaggle/input -name vits_step_40000.pt | head -1)
   sha256sum $DATA/metadata.csv $R0; ls $DATA/wavs | wc -l
   OUT=/kaggle/working/runs-ft; mkdir -p $OUT
   python tools/chunk_alignments.py --checkpoint $R0 --data-dir $DATA --out $OUT/chunk_durations_R0_40k.json
   CUDA_VISIBLE_DEVICES=0 python tools/train_tts010.py --arm R0 --data-dir $DATA --run-root $OUT \
       --num-workers 2 --tag ftchunk20k --init-from $R0 --steps 20000 --keep-every 5000 \
       --chunk-prob 0.5 --chunk-durations $OUT/chunk_durations_R0_40k.json > $OUT/ftchunk20k.log 2>&1 &
   CUDA_VISIBLE_DEVICES=1 python tools/train_tts010.py --arm R0 --data-dir $DATA --run-root $OUT \
       --num-workers 2 --tag ftctrl20k --init-from $R0 --steps 20000 --keep-every 5000 \
       > $OUT/ftctrl20k.log 2>&1 &
   wait
   ```

   Each run uses one T4 (the trainer uses one device). Kaggle's image already has the Python
   packages. The runs stop cleanly at the default 11.3-hour budget. Check the hashes printed at
   the top of the cell output against A4.
4. When the version has finished, download `runs-ft` from the notebook's Output tab and upload
   it as in A10.

## Expected time

Measured seconds per training step of arm R0 (batch 16, fp16):

| GPU | Seconds per step | Source | 20,000 steps |
|---|---:|---|---:|
| RTX PRO 6000 Blackwell Server Edition | 0.101 | the R0 40k run (median, 2026-10-14) | 0.56 h |
| RTX 4070 Ti SUPER | 0.24–0.25 | the local 5,000-step fine-tunes (2026-10-19, CPU shared with another job) | 1.3–1.4 h |
| Tesla T4 | about 0.9 | benlu's 40,000 steps in about 10 h (his notebook, reported) | about 5 h |

No L40S run has been timed; an L40S should fall between the first two rows (an expectation, not
a measurement), so plan with at most 1.4 h per run, i.e. at most 3 GPU hours for both runs one
after the other. Two runs at the same time on one GPU each run slower; the log then shows the
shared speed. Add about 5 minutes per run for start-up, probe evaluations and checkpoint saves.

## Run card (one per run)

```text
Run:                      ftchunk20k / ftctrl20k
Executed by:              (name)
Hardware:                 (GPU model and memory; server / Colab / Kaggle)
Commit:                   (git rev-parse --short HEAD)
Run folder:               (tts010_R0_<tag>_<UTC time>)
R0 checkpoint sha256:     (from A4)
Durations file sha256:    (from A5)
Started / completed UTC:  (run_config.json / status.json)
Wall time:                (status.json session_wall_seconds; sum over resumes)
Seconds per step:         (last s/it in the log)
Status:                   (completed / stopped / failed) at step N
Resumed:                  (no / yes: from step N)
Deviations:               (anything that differed from this manual, or "none")
Drive path:               (tts010-chunk-ft/…)
```
