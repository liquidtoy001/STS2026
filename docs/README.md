# Project documentation

## For teammates

- [Repository overview](../README.md)
- [Contribution workflow](../CONTRIBUTING.md)
- [Environment and notebook setup](SETUP.md)
- [Reconstruction evaluation and Kaggle handoff](EVALUATION.md)
- [Local GPU run and checkpoint reproduction](LOCAL_EVALUATION.md)
- [GitHub handbook: English, macOS and website-first](guides/STS2026_GitHub_Team_Handbook.pdf)
- [ASR/VITS evaluation planning guide](guides/ASR_VITS_Evaluation_Guide.pdf)

The evaluation guide is planning/reference material, not a report of implemented metrics or measured results. The handbook's initial-publication chapter applies only until the first baseline is published.

## Rebuilding the GitHub handbook

`handbook_web_content.py` contains the English content. `build_github_handbook.py` supplies layout, diagrams, navigation and the PDF build. `check_github_handbook.py` checks page count, navigation targets and text bounds and creates contact sheets for visual review.

The current generator uses **Windows Calibri and Consolas font paths**. It is a maintainer publishing tool, not a required dependency for Mac teammates. Do not expect it to run unchanged on macOS; a portable font setup is future work. The generated PDF itself works on any PDF viewer supporting standard PDF navigation.

In a Windows publishing environment with Python, the fonts and Poppler available:

```powershell
python -m pip install -r requirements/docs.txt
python tools/handbook/build_github_handbook.py
New-Item -ItemType Directory -Force tmp/pdfs/handbook-qa
pdftoppm -r 95 -png docs/guides/STS2026_GitHub_Team_Handbook.pdf tmp/pdfs/handbook-qa/page
python tools/handbook/check_github_handbook.py
```

Use a fresh render directory when changing the page count so stale PNGs cannot distort validation. The checker currently expects 23 pages and 21 index links; update those expectations and all cross-references if the structure changes. Inspect the rendered pages as well as the structural results before distributing a revision.

The final PDF belongs in `docs/guides/`; render images, contact sheets and backup versions stay under ignored `tmp/`. The source for the older ASR/VITS planning PDF is not currently included in this repository.
