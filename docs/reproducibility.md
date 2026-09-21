# Reproducibility notes

## Design

The workflow is deliberately separated into two stages:

1. **Expensive audit generation** — detector inference, D-RISE/EigenCAM saliency, and ROI-based D-Deletion are persisted as JSONL records.
2. **Cheap statistical reconstruction** — tables, effect sizes, sensitivity analyses, and quantitative figures are rebuilt entirely from the stored records.

This separation allows the statistical analysis to be reproduced without re-running thousands of masked detector forward passes.

## Phase 0: pre-hoc confidence stratification

`fit_terciles.py` scans true-positive pedestrian detections, computes domain-specific 33.3/66.7 confidence percentiles, freezes them in `results/terciles.json`, and writes the corresponding detected-instance metadata to `results/eval_index/`.

The frozen thresholds in the included run are approximately:

- PIE: 0.2836 and 0.6866
- JAAD: 0.1642 and 0.6820

The detected pools contain 4386 PIE and 4747 JAAD instances.

## Audit sampling

A fixed sampling seed selects 300 audited instances per domain, balanced across low-, mid-, and high-confidence terciles (100 per tercile).

D-RISE is evaluated with five mask seeds. EigenCAM is deterministic; records retain the same seed structure for uniform downstream processing.

## D-Deletion

Deletion is performed on the ROI crop. Pixels are ranked by saliency and set to a zero baseline over 25 steps up to a maximum removed fraction of 0.5.

The primary metric is the raw target-response AUC divided by the evaluated deletion range. It is **not** divided by the zero-deletion response. A response-normalized variant is retained only as a sensitivity diagnostic.

## Statistical reconstruction

`aggregate.py` produces:

- group summaries (mean, median, IQR, bootstrap CI);
- low- vs high-confidence diagnostics;
- unadjusted cross-domain comparisons;
- Spearman coupling with `score_start` (`f0`) and full-frame confidence;
- matched-`f0` PIE-vs-JAAD comparisons;
- a response-normalized sensitivity analysis;
- an alternative `f0` binning sensitivity analysis;
- paired D-RISE-vs-EigenCAM comparison;
- descriptive size/occlusion summaries.

Holm adjustment is applied separately within each family of bin-wise comparisons per method.

## Randomness

The sampling and bootstrap generator use a fixed seed (`20260703`) in the released configuration. D-RISE uses mask seeds 0–4.

## Qualitative figure

`make_qualitative.py` selects a representative PIE and JAAD case from a specified `f0` bin. Selection first excludes visually extreme bounding-box aspect ratios and then chooses a candidate near the median D-RISE/EigenCAM method gap rather than optimizing for the smallest AUC. This reduces cherry-picking of qualitative examples.
