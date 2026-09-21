# Code audit summary

The release package was checked before publication.

## Checks performed

- All Python source files compile successfully with `python -m py_compile`.
- `aggregate.py --metric deletion_auc` runs end-to-end using the included JSONL records.
- `make_figures.py` regenerates the quantitative figures from the stored records.
- A regression test suite verifies record counts, method-comparison values, controlled D-RISE gaps, and global `f0` coupling.
- JSONL and evaluation-index files were scanned for absolute local paths; none were found.
- The supplied full environment snapshot was not published because it contained many unrelated packages and an editable dependency from another project. A minimal dependency specification and a tested-environment note are provided instead.

## Release cleanup

- Project paths were made portable relative to the repository root.
- Generated artifacts now live under `results/` rather than a machine-specific path.
- Stale comments referring to unused detectors or an unimplemented D-MFPP branch were removed from the released workflow.
- The unused D-MFPP adapter was removed from the released audit runner so the public code matches the paper methods (D-RISE and EigenCAM only).
- The qualitative-figure script was cleaned by removing an obsolete duplicate selection function and duplicate plotting-style configuration.
- The qualitative curve x-axis was renamed to `Normalized deletion budget` to make clear that 1.0 corresponds to the configured maximum deletion fraction (0.5 of the ROI), not to deletion of the full ROI.

- `make_figures.py` now embeds TrueType (Type 42) fonts, matching `make_qualitative.py`, so regenerated figures contain no Type 3 fonts (required by IEEE/PaperCept).
- `CITATION.cff` includes a `preferred-citation` entry pointing to the ICVES 2026 paper.

- All code comments, docstrings, and console messages were translated to English; no computational logic was changed (tests pass and all regenerated CSVs are byte-identical).
- Axis labels and legends in `make_figures.py` were aligned with Figs. 2 and 3 of the paper (`f0` notation, `[0.0, 0.2]` bin labels, legend placement).
- The README now includes compact result tables and a PNG version of the main figure (`docs/assets/`).

## Scope

The audit confirms consistency and reproducibility of the supplied implementation and numerical artifacts. It does not independently validate the third-party PIE/JAAD annotations, YOLOv8 internals, or external software packages.
