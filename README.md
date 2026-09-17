# STS2026

Team coursework project exploring a compact VITS-style speech model and its evaluation.

**Status:** an initial notebook baseline and planning material. Full training, end-to-end inference quality and reproducible evaluation have not been validated as part of repository preparation.

## Start here

Team members primarily use **macOS and the GitHub website**. You can accept an invitation, edit documentation and review pull requests without installing Git.

1. Send your GitHub username/profile link to the maintainer and accept the repository invitation using the same account.
2. Read the [GitHub team handbook](output/pdf/STS2026_GitHub_Team_Handbook.pdf): page 5 covers accepting invitations, pages 6–9 cover the website workflow, and page 19 is a browser-only practice exercise.
3. For a task, create a branch from `main`, commit your changes to that branch, and open a pull request into `main`.
4. Ask another teammate to review. The maintainer coordinates merges initially.

A website commit is already on GitHub; it does not need a separate push. Running code requires the environment described in [Setup](docs/SETUP.md).

## What the notebook currently does

[`vits-light_v1.ipynb`](vits-light_v1.ipynb) contains text/posterior encoders, a flow, alignment and duration components, a waveform decoder, losses and a trainer. Its **active experiment configuration** is more limited than the overall architecture:

- Uses LJSpeech, with a hard-coded Kaggle input path and 22,050 Hz audio.
- Calls `overfit_train(total_steps=1500)` on one fixed batch.
- Sets duration and KL loss weights to zero and enables reconstruction-oriented losses.
- Leaves the GAN loss at zero and the discriminator as a placeholder.
- Leaves `evaluate(outputs, batch)` unimplemented.
- Selects CUDA if available, otherwise CPU; it does not select Apple's MPS backend.

These are development starting points, not evidence of a completed full VITS implementation, a measured evaluation pipeline or validated voice conversion. The notebook has no saved cell outputs. See [the baseline notes](docs/SETUP.md#current-limitations) before running it.

## Repository map

```text
STS2026/
  vits-light_v1.ipynb              Original baseline notebook
  README.md                      Project entry point
  CONTRIBUTING.md                Team workflow and review rules
  requirements-notebook.txt       Non-PyTorch notebook dependencies
  requirements-docs.txt           PDF generation/checking dependencies
  .github/pull_request_template.md
  docs/
    SETUP.md                     Environment, dataset and runtime notes
    README.md                    Document index and rebuild instructions
    build_github_handbook.py      Handbook layout/build source
    handbook_web_content.py       English website-first content
    check_github_handbook.py      PDF structure and layout checks
  output/pdf/
    STS2026_GitHub_Team_Handbook.pdf
    ASR_VITS_Evaluation_Guide.pdf
```

`tmp/`, data, environments, caches, checkpoints and generated sample audio stay local and are ignored. The two curated PDFs are intentionally included; additional output files require explicit review and an ignore-rule update.

## Working agreements

- One short-lived branch per task; keep `main` as the reviewed baseline.
- Coordinate ownership before editing a notebook. Avoid simultaneous whole-notebook replacements through web uploads.
- Record exactly what you tested and what remains untested in each PR.
- Keep data, model weights and credentials out of Git, including notebook outputs.
- GitHub branch protection is not assumed to be enabled; follow the team agreement even when the UI allows a direct commit.

See [Contributing](CONTRIBUTING.md) and [documentation](docs/README.md).

## Validation and reuse

Repository preparation checks notebook structure and Python syntax only; those checks do not run training or prove model correctness. Dataset access, compatible PyTorch/TorchAudio versions, a held-out evaluation protocol and actual experiment results still need to be established by the team.

No software license has been selected. Confirm course publication rules, code provenance and third-party/data terms before public release or reuse; adding a license is a team decision.
