# Contributing to STS2026

## Website-first workflow

1. Accept the maintainer's invitation while signed in to the invited GitHub account.
2. Select `main` and create a task branch, for example `docs/experiment-guide`, `fix/audio-loading` or `feat/evaluation-summary`.
3. Edit text files or upload only the intended small files. Check the branch and file paths before completing **Commit changes**.
4. Open a pull request with **base: main** and your task branch as the source.
5. Explain the purpose, actual verification and known limitations. Request another teammate's review.
6. Address feedback on the same branch. After review, the maintainer merges; select `main` to verify the result and start a fresh branch for the next task.

Normal website commits are already remote. When using a local clone, commit and push are separate steps. Pull website changes into the same local branch before resuming local work.

## Scope and quality

- Keep changes focused and commit messages descriptive, e.g. `docs: explain dataset setup`.
- Preserve supplied baseline exercises/placeholders unless implementing them is part of the agreed task.
- Add meaningful checks for changed behavior where practical; do not add tests that merely restate the implementation.
- For documentation, verify paths, links and claims. For code, state which relevant checks ran and their results. Never describe syntax checking as successful training.
- Do not force-push shared branches or rewrite `main` to hide mistakes. Coordinate recovery with the maintainer.

## Notebook changes

Agree one primary editor for a notebook task. Start from the latest intended branch, review code/output/metadata changes, and run the relevant cells in a fresh kernel when the required data and compute are available. Report when execution was not possible.

Keep experiment evidence in small reviewed records: code commit (plus uncommitted changes, if any), configuration, random seed, data version/split, environment, metrics and artifact locations. Do not report an overfit-batch result as held-out evaluation.

## Files and credentials

Write all experiment logs, measured-result records and generated experiment
reports in English. Record completion as an ISO 8601 UTC timestamp with its
source. Keep report-generation time separate from experiment-completion time.
For legacy records, label reconstructed times as approximate and state the
evidence; never invent an exact completion time. Failed/interrupted runs record
an end time, not a successful completion time. Personal learning walkthroughs
stay under ignored `output/learning/` and must not be committed.

Use `.gitignore` for local files, but inspect manually selected website uploads too. Never upload `.git`, `.env`, virtual environments, raw datasets, checkpoints or bulk generated audio. Small curated report figures and metrics can be added deliberately with suitable paths and ignore rules.

If a credential is uploaded, revoke/rotate it first and notify the maintainer without reposting its value. A later file deletion does not erase Git history.

## Documentation navigation

Each documented model-version change or improvement proposal must have its own
entry in the shared left sidebar (`docs/site/assets/site.js`, `NAV`). Use a
dedicated page for a new proposal or version, with a stable URL, descriptive
title, author, date and explicit status (proposed, implemented or measured).
An existing version section may have its own anchored entry. Do not bury new
proposals only in the comparison board or experiment reports. Update related
links and check that the sidebar destination exists in the same change.

## Handbook

The [23-page handbook](docs/guides/STS2026_GitHub_Team_Handbook.pdf) covers invitations (page 5), normal website operations (pages 6–10), conflict/recovery scenarios (pages 14–18), a practice PR (page 19), a checklist (page 22), and optional Mac setup (page 23).

## Website language and synchronization

Maintain the documentation website in English only. Do not add language
selectors, translation dictionaries or translated website copies.
`docs/site/` is the canonical source. For future website changes, update and
validate the local source, synchronize the intended changes to GitHub on
`Turino`, and publish the same website source to the existing GPT Sites site.
GitHub pushes and GPT Sites publication are separate operations; verify both
and record the deployed version. Preserve teammates' branches. Merge a PR to `main` only with maintainer authorization;
never rewrite `main`. If the maintainer asks you to wait for a decision, pause
commits, pushes, PR changes and deployment until that decision.
