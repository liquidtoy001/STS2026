# TTS_0.02 — Speaker-ID conditioning

> **TTS_0.20 (2026-10-25):** a manifest can now have a test set (`STS_SPEAKER_HELDOUT`), and a
> conditioned checkpoint can start a single-speaker run with one speaker vector folded in
> (`STS_INIT_FOLD_SPEAKER`); used for LibriTTS-R pretraining, see [TTS_0.20](TTS_0.20.md#phase-2-libritts-r-pretraining-then-ljspeech-fine-tuning).
>
> **TTS_0.10 (2026-10-12):** speaker conditioning is kept. With the TTS_0.10 posterior encoder the
> speaker vector is added inside every gated residual block instead of once at the input; with
> `posterior_encoder = tts002` the behaviour below is unchanged. See [TTS_0.10](TTS_0.10.md).

Prototype: benlu. Integration: Turino. Version date: 2026-10-07 (Australia/Brisbane).
Status: implemented and integration-tested; research source pre-release.
Release tag: `tts-v0.02`. Real multi-speaker training and speech quality: **not run**.

## Implementation contract

Extend `notebooks/vits-full-train.ipynb` with optional learned speaker-ID
conditioning. Keep `notebooks/vits-light_v1.ipynb` and historical experiment
records unchanged. The default remains unconditional single-speaker LJSpeech.
This release is model source, not a trained multi-speaker checkpoint.

Condition the posterior encoder, forward/inverse flow, duration predictor and
waveform decoder with one validated `[batch, channels, 1]` speaker vector.
Use a pipe-separated manifest with a header containing `file_name`,
`normalized_transcription`, and `speaker_name`; audio remains under `wavs/`.
Derive a deterministic speaker-name-to-ID mapping and persist it with checkpoints
and run configuration. Reject missing, out-of-range or malformed IDs and
incompatible checkpoint mappings. Preserve the legacy six-tensor batch when
conditioning is disabled; append speaker IDs as a seventh tensor when enabled.

## Acceptance criteria fixed before implementation

- Exercise actual notebook definitions: conditioned training forward/backward,
  finite losses, nonzero speaker-embedding gradients, and text-only inference.
- Test missing/invalid IDs, mixed-speaker collation, flow invertibility with
  conditioning, and exact checkpoint round trips/mapping rejection.
- Compare the disabled path against the previous source with identical weights,
  inputs and RNG state; require identical outputs and state-dictionary keys.
- Run reconstruction evaluation with IDs, and preserve existing legacy tests.
- Rebuild and validate both standalone Kaggle exports, including their embedded
  evaluator. Synthetic inputs establish wiring only, never speech quality.
- Publish English README and HTML release notes with a dedicated sidebar entry,
  a new `tts-v0.02` pre-release, and explicit untrained/unevaluated limitations.

Reference-audio encoders, zero-shot cloning, pretrained-weight migration,
multi-speaker corpus training, held-out speech quality and voice conversion are
outside this source release. No large training run is scheduled by this change.

## How the condition works

`speaker_id` indexes an `nn.Embedding(n_speakers, gin_channels)` table. The result
is expanded to `[B, gin_channels, 1]`. Each conditioned module projects this vector
to its feature width and broadcasts it along time before adding it to features.
The default condition width in manifest mode is 64; set `STS_GIN_CHANNELS` to
another positive integer before constructing a new model.

| Component | TTS_0.02 behavior |
| --- | --- |
| Text encoder | Shared across speakers; unchanged |
| Posterior encoder | Adds the projected condition before its convolution stack |
| Flow | Enables the existing condition projections in every coupling network, in both directions |
| Duration predictor | Uses the existing condition projection; its detached speaker input preserves the existing duration-gradient policy |
| Decoder | Adds a projected speaker condition before waveform upsampling |
| Trainer | Requires a seventh batch tensor in conditioned mode and retains IDs for listening samples |
| Evaluator | Uses identical IDs in posterior encoding and waveform reconstruction; records them in the manifest |

The ID table and the generator must learn together from labeled recordings.
Adding an untrained ID is not voice cloning. The text vocabulary remains the
existing English-oriented character inventory; this change adds no multilingual
frontend, reference-audio encoder or pretrained speaker embedding extractor.

## Prepare a corpus

Keep each file ID unique across all speakers. Store mono 22,050 Hz WAV recordings
under `wavs/`; the existing loader downmixes stereo but rejects a different sample
rate. Supply normalized transcripts supported by the notebook's text vocabulary.
Do not include `.wav` in `file_name`.

```text
my-corpus/
  speakers.csv
  wavs/
    alice_001.wav
    bob_001.wav
```

The pipe-separated manifest must have a header:

```text
file_name|normalized_transcription|speaker_name
alice_001|Hello there.|alice
bob_001|Good morning.|bob
```

Names are case-sensitive and sorted lexically to assign IDs. The example yields
`{"alice": 0, "bob": 1}`. The full mapping is printed and saved; inference must
use the checkpoint's mapping, not a newly guessed order. Missing files, blank
fields and duplicate file IDs are rejected. The manifest is a training input;
this release does not introduce a held-out data split.

In PowerShell, before opening/running the notebook:

```powershell
$env:STS_DATA_DIR = 'E:/datasets/my-corpus'
$env:STS_SPEAKER_MANIFEST = 'E:/datasets/my-corpus/speakers.csv'
$env:STS_GIN_CHANNELS = '64'
$env:STS_NUM_WORKERS = '0'  # Use zero for local Windows notebooks.
```

On Kaggle, set these environment variables in a cell before the generated
notebook's configuration cell. Attach the corpus as an input. Existing export
instructions default to LJSpeech; manifest mode is an explicit override.
To return to legacy mode, remove `STS_SPEAKER_MANIFEST` and point `STS_DATA_DIR`
to LJSpeech. No speaker IDs may be passed to an unconditional model.

## Inference and checkpoint compatibility

After constructing a model with the checkpoint's architecture and speaker
configuration and loading its weights, call `model.eval()` before inference:

```python
model.eval()
speaker_ids = torch.tensor(
    [speaker_to_id["alice"]], dtype=torch.long, device=next(model.parameters()).device
)
result = model.infer(text_tokens, text_lengths, speaker_ids=speaker_ids)
```

Provide one ID per text sequence (`[B]`, `torch.long`, on the model device).
Missing IDs, out-of-range values, incorrect dimensions/dtypes and IDs supplied
to an unconditional model raise errors. `max_frames` can bound inference length.

Checkpoints contain `model_version`, `speaker_config.n_speakers`,
`speaker_config.gin_channels`, and `speaker_config.speaker_to_id`, alongside model,
optimizer and discriminator states. The trainer checks speaker configuration
before loading weights or optimizer state; swapping names while retaining the
same number of speakers is rejected. Full-run resume additionally checks the
metadata hash and probe IDs before copying prior records.

An older unconditional checkpoint can still be loaded into an unconditional
model using strict state loading. It cannot resume into a conditioned model:
the new table/projections are untrained and require a separate, explicitly
designed warm-start procedure. `strict=False` is not used to hide this mismatch.

## Verification evidence

Commands run in the existing project environment:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe tools/check_speaker_notebook.py
```

The golden regression test reads `tts-v0.01` from local Git history; a shallow
checkout must fetch that tag before running the suite. It executes the old model
definition and compares actual tensor outputs with the new disabled path.

- 19 automated tests passed, covering legacy evaluation, GAN-on/off parameter
  updates, both speaker embedding rows, conditional projection gradients,
  validation failures, collation, nontrivial conditional-flow inversion,
  same-text speaker switching, checkpoint restoration, mapping mismatch rejection,
  and generated Kaggle artifacts. Test audio is synthetic.
- The integration runner executes **every code cell** of the full notebook twice:
  it stops at the time limit after step 1, then resumes to step 2. It checks the
  restored mapping/update count and text-only sample files. Temporary tones,
  checkpoints and audio are removed after the check.
- Environment: PyTorch/TorchAudio 2.7.1+cu118; NVIDIA GeForce RTX 4070 Ti SUPER.
  Validation timestamps and measured integration status are recorded in
  [validation.json](records/tts-0.02/validation.json). Timing and memory there
  describe this tiny synthetic check, not expected full-corpus performance.
- No real-speaker training, intelligibility assessment, speaker-similarity score,
  held-out TTS result or multi-speaker trained weights are provided by this release.

Historical reports, the earlier notebook and previous release tags are unchanged.
