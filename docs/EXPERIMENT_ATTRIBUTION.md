# Experiment dates and contribution attribution

Team credits were confirmed by the project owner on 2026-09-30. The requested
display spelling for the training operator is **tuirno**; the Git branch remains
**Turino**. These are separate labels, not a branch rename.

| Work | Training operator | Component developer | Display completion date |
|---|---|---|---|
| Initial shared-evaluation local baseline | tuirno | Shared evaluation framework | 2026-09-20, Brisbane |
| Jacques's original / new enc-dec study | Jacques | Component identity still pending | 2026-09-20, team-reported |
| Full training, gan0 (50,000 iterations) | tuirno | GAN component authors: Mou and Warneke | 2026-09-29, study date |
| Full training, gan1 (50,000 iterations) | tuirno | Mou and Warneke | 2026-09-29, study date |
| GAN component completion | Not a training run | Mou and Warneke | 2026-09-23, completion commit |

## Preserve raw evidence

The local baseline completed approximately 2026-09-19 15:27 UTC, which is
2026-09-20 01:27 in Brisbane. Its source timestamp is an artifact-based estimate.

Jacques's original export says `2026-09-20T07:42:35+00:00`; the new enc-dec
export says `2026-09-30T06:15:25+00:00`. The team-reported September 20 date is
displayed separately. This discrepancy requires reconciliation with Jacques;
raw JSON timestamps, hashes and measured values were not rewritten.

## GAN branch and tag provenance

Remote refs were read directly from GitHub on 2026-09-30, without modifying them.

- [gan-group branch](https://github.com/liquidtoy001/STS2026/tree/gan-group):
  `07e663dba683af7df352bdaf6b27308596b3ff94`, authored by `ewarni`,
  2026-09-23 01:32:39 +10:00, message `gan-loss-completed`.
- [Mou contributor branch](https://github.com/liquidtoy001/STS2026/tree/Mou):
  currently `5b46f60454da1036e605dbcc7d50032ce9ce1082`. It does not currently
  point to the GAN completion commit; contributor credit comes from the team.
- [gan-loss-completed tag](https://github.com/liquidtoy001/STS2026/tree/gan-loss-completed):
  currently `1f3fd0f7ff3414af3459ab1cffff66acf8dceba1`, authored by `ewarni`,
  2026-09-22 15:12:52 +10:00, message `Initial commit - import plot`.
  Despite its name, the tag is not attached to the September 23 completion commit.

Training execution, component development, trainer integration and report editing
are distinct contributions. The existing record of benlu's trainer integration
is retained; it does not assign the gan0/gan1 training runs to benlu.
