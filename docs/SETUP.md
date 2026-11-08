# Environment and dataset setup for the reconstruction notebook

This page covers `notebooks/vits-light_v1.ipynb`, the posterior-reconstruction notebook of
September 2026 (the project's first evaluated baseline). For the TTS models (TTS_0.10 and later),
see [Reproduce](../README.md#reproduce) in the repository README.

## Local environment (starting point, not a validated training recipe)

```sh
git clone https://github.com/liquidtoy001/STS2026.git
cd STS2026
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The notebook metadata records Python 3.12.13. This is provenance information, not a verified requirement range.

Install **matching PyTorch and TorchAudio versions** supported by the chosen Python, operating system and accelerator. Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) and [TorchAudio installation/compatibility guidance](https://docs.pytorch.org/audio/stable/installation.html). On an existing Kaggle runtime, inspect installed versions before changing them. Audio decoding/encoding dependencies depend on the selected TorchAudio release.

Then install the remaining notebook dependencies:

```sh
python -m pip install -r requirements/notebook.txt
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"
python -m pip check
python -m jupyterlab --notebook-dir=notebooks
```

An import check verifies only imports. The dependency list is unpinned; do not treat it as a reproducible lockfile.

## Notebook location and working directory

Run the setup commands from the repository root, then open `notebooks/vits-light_v1.ipynb` in JupyterLab. Check the kernel working directory with `%pwd` before training; it should be the repository's `notebooks/` directory for this workflow. If your editor uses a different directory, set it explicitly before executing cells.

Model definitions are preserved; the final training cells use the shared reconstruction evaluator. Local experiment artifacts go to `output/eval/` with a unique run directory. For the bundled Kaggle entry, see [evaluation instructions](EVALUATION.md). Dataset paths are configured explicitly in the notebook.

## Dataset configuration

Obtain [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) and confirm its terms. Do not commit the dataset. The notebook expects:

```text
LJSpeech-1.1/
  metadata.csv
  wavs/
    <file_id>.wav
```

The configuration cell sets:

```python
DATA_DIR = Path("/kaggle/input/datasets/mathurinache/the-lj-speech-dataset/LJSpeech-1.1")
```

For another environment, change this to the actual extracted dataset path. `WAV_DIR` and `METADATA_PATH` derive from it. Rows are pipe-separated IDs, transcription and normalized transcription; the dataset class expects audio sampled at 22,050 Hz.

The loader uses `NUM_WORKERS = 4`. If multiprocessing fails in a macOS/Jupyter session, use `NUM_WORKERS = 0` for initial debugging, then verify behavior. The notebook chooses CUDA or CPU; it has no explicit MPS path. CPU execution is not a claim that full training is practical on a laptop.

## Before running training

1. Verify the data path, metadata and a real audio sample can be loaded.
2. Run the data-loading and component cells in order, reviewing shapes and errors.
3. Inspect the training configuration before executing the final cells. The call starts 2,000 fixed-batch overfit steps; do not start it merely to test installation.
4. Record the chosen environment, data identity, configuration and code revision.

The evaluation entry stores checkpoints, metrics and samples inside its unique run directory under `output/eval/` locally, or `/kaggle/working/sts2026-eval-runs/` in the export. These outputs are not committed.

## Limitations of the September 2026 baseline configuration

- The run disables duration and KL losses and calls the fixed-batch overfit routine.
- GAN loss is a zero placeholder and `Discriminator` is incomplete.
- Fixed-crop reconstruction evaluation and one local LJSpeech training-batch baseline are available; see [measured results](LOCAL_BASELINE_RESULT.md).
- The dataset loader uses the metadata as its training set, so there is no held-out evaluation in this notebook. Held-out splits were introduced with TTS_0.10 (`sts_eval/splits/`).
- Local CUDA reconstruction training was exercised with PyTorch/TorchAudio 2.7.1+cu118. This is not validation of every platform or full TTS inference.
- The final cells contain reconstruction inspection code; it does not measure TTS quality.

The exact evaluated-baseline notebook is commit `c9f258c`; the current file contains later changes.
