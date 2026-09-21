# Results directory

This directory contains the numerical artifacts used to reconstruct the paper results.

- `terciles.json` — pre-hoc confidence thresholds and detected-pool counts.
- `eval_index/` — metadata for detected pedestrians available to the audit.
- `records/` — raw per-seed D-RISE/EigenCAM D-Deletion records.
- `agg_by_group.csv` — group-level means, medians, IQRs, CIs, and `f0` summaries.
- `agg_directional.csv` — low- vs high-confidence diagnostic comparisons.
- `agg_crossdomain.csv` — cross-domain comparisons without `f0` control.
- `agg_confound.csv` — Spearman score-coupling results.
- `agg_crossdomain_controlled.csv` — primary matched-`f0` cross-domain analysis.
- `agg_crossdomain_controlled_norm.csv` — response-normalized sensitivity analysis.
- `agg_crossdomain_controlled_bins025.csv` — alternative-bin sensitivity analysis.
- `agg_method_comparison.csv` — paired D-RISE vs EigenCAM comparison.
- `agg_robustness_size_occlusion.csv` — descriptive size/occlusion checks.
- `fig_score_vs_auc.pdf` and `fig_crossdomain_bins.pdf` — regenerated quantitative figures.

The raw PIE/JAAD images and videos are not included.
