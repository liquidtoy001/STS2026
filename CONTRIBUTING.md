# Contributing

This repository holds a coursework project (STAT3007 Group G7). Changes go through a pull request
against `main`.

## Changes

- Keep a change focused and describe it in the pull request: what changed, which checks ran and
  their results, and what was not checked. A syntax or import check does not show that training
  works or that the audio is good.
- Run the tests with `python -m unittest discover -s tests -v` when code changes.
- Clear notebook outputs before committing, unless a small output is needed to review the change.
- Do not force-push shared branches or rewrite `main`.

## Experiment records

- Write records in English. Record the code commit, configuration, seed, dataset and split,
  environment, metrics and where the artifacts are kept.
- Record a completion time as an ISO 8601 UTC timestamp and name its source. Label reconstructed
  times as approximate.
- State the protocol next to every metric. Keep training-batch results apart from held-out
  evaluation.

## Files

Datasets, checkpoints, bulk audio and credentials stay out of Git (see `.gitignore`). Commit only small records and figures needed to explain or reproduce a result. If a
credential is committed, revoke it first: deleting the file does not remove it from Git history.
