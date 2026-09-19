# STS2026

**English** | [简体中文](docs/translations/README.zh-CN.md) | [Deutsch](docs/translations/README.de.md) | [Français](docs/translations/README.fr.md)

Team coursework project exploring a compact VITS-style speech model and its evaluation.

**Status:** shared posterior-reconstruction evaluation and a measured old-model baseline are available. A local eight-utterance run completed 2,000 iterations (1,988 optimizer updates); same-checkpoint rescoring reproduced all measured metrics exactly on the same machine. See the [baseline record](docs/LOCAL_BASELINE_RESULT.zh-CN.md). Held-out performance, full TTS quality and the new-model comparison remain untested.

## Start here

Team members primarily use **macOS and the GitHub website**. You can accept an invitation, edit documentation and review pull requests without installing Git.

1. Send your GitHub username/profile link to the maintainer and accept the repository invitation using the same account.
2. Read the [GitHub team handbook](docs/guides/STS2026_GitHub_Team_Handbook.pdf): page 5 covers accepting invitations, pages 6–9 cover the website workflow, and page 19 is a browser-only practice exercise.
3. For a task, create a branch from `main`, commit your changes to that branch, and open a pull request into `main`.
4. Ask another teammate to review. The maintainer coordinates merges initially.

A website commit is already on GitHub; it does not need a separate push. Running code requires the environment described in [Setup](docs/SETUP.md).

## What the notebook currently does

[`vits-light_v1.ipynb`](notebooks/vits-light_v1.ipynb) contains text/posterior encoders, a flow, alignment and duration components, a waveform decoder, losses and a trainer. Its **active experiment configuration** is more limited than the overall architecture:

- Uses LJSpeech, with a hard-coded Kaggle input path and 22,050 Hz audio.
- Runs a 2,000-step overfit comparison on eight fixed LJSpeech IDs.
- Sets duration and KL loss weights to zero and enables reconstruction-oriented losses.
- Leaves the GAN loss at zero and the discriminator as a placeholder.
- Uses shared posterior-reconstruction evaluation with fixed crops, mean/sample modes, metrics and audio export. See [evaluation and Kaggle instructions](docs/EVALUATION.md).
- Selects CUDA if available, otherwise CPU; it does not select Apple's MPS backend.

The measured experiment covers reconstruction of training audio, not a completed full VITS implementation or validated voice conversion. The notebook has no saved cell outputs. See [the baseline notes](docs/SETUP.md#current-limitations) before running it.

## Repository map

```text
STS2026/
  README.md
  CONTRIBUTING.md
  .github/pull_request_template.md
  notebooks/
    vits-light_v1.ipynb
  requirements/
    notebook.txt
    docs.txt
  sts_eval/                       # Shared reconstruction metrics and runner
  tests/                          # Behavioral and export checks
  docs/
    README.md
    SETUP.md
    translations/
      README.zh-CN.md
      README.de.md
      README.fr.md
    guides/
      STS2026_GitHub_Team_Handbook.pdf
      ASR_VITS_Evaluation_Guide.pdf
  tools/handbook/
    build_github_handbook.py
    handbook_web_content.py
    check_github_handbook.py
  tmp/                            # Local only / Git-ignored
```

`tmp/`, runtime `output/`, data, environments, caches, checkpoints and generated sample audio stay local and are ignored. Curated PDFs are included in `docs/guides/`; review additional report artifacts before placing them in documentation folders.

## Working agreements

- One short-lived branch per task; keep `main` as the reviewed baseline.
- Coordinate ownership before editing a notebook. Avoid simultaneous whole-notebook replacements through web uploads.
- Record exactly what you tested and what remains untested in each PR.
- Keep data, model weights and credentials out of Git, including notebook outputs.
- GitHub branch protection is not assumed to be enabled; follow the team agreement even when the UI allows a direct commit.

See [Contributing](CONTRIBUTING.md) and [documentation](docs/README.md).

## Validation and reuse

Eight behavioral/export tests and notebook schema checks passed. The local baseline additionally exercised the original model configuration on real LJSpeech data. These checks do not establish generalization, cross-machine reproducibility or audible quality. See [validation](docs/EVALUATION_VALIDATION.md) and [local reproduction instructions](docs/LOCAL_EVALUATION.zh-CN.md).

No software license has been selected. Confirm course publication rules, code provenance and third-party/data terms before public release or reuse; adding a license is a team decision.
