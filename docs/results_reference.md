# Reference results

These values provide a compact integrity check for the included audit records.

## Record counts

- Raw JSONL records: 6000
- Per-instance rows after seed aggregation: 1200
- Unique audited instances: 600
- Domains: PIE, JAAD
- Explainability methods: D-RISE, EigenCAM

## Method comparison

Across the 600 instances common to both explainers:

| Quantity | Value |
|---|---:|
| D-RISE mean raw D-Deletion AUC | 0.1117 |
| EigenCAM mean raw D-Deletion AUC | 0.3499 |
| Mean paired difference (D-RISE − EigenCAM) | −0.2382 |
| Median paired difference | −0.1789 |
| Paired instances | 600 |

## `f0` coupling

Global Spearman correlation between raw D-Deletion AUC and `score_start`:

| Method | rho |
|---|---:|
| D-RISE | 0.7620 |
| EigenCAM | 0.7444 |

## Controlled PIE − JAAD gaps for D-RISE

| `f0` bin | Gap | Cliff's delta | Holm-adjusted p |
|---|---:|---:|---:|
| [0.0, 0.2] | 0.0051 | 0.164 | 0.3817 |
| (0.2, 0.4] | 0.0341 | 0.423 | 0.0023 |
| (0.4, 0.6] | 0.0429 | 0.350 | 0.0023 |
| (0.6, 0.8] | 0.0481 | 0.399 | <0.001 |
| (0.8, 1.0] | 0.0115 | 0.141 | 0.3817 |

## Residual mean `f0` balance (PIE / JAAD)

| Bin | PIE | JAAD | Absolute gap |
|---|---:|---:|---:|
| [0.0, 0.2] | 0.0888 | 0.0587 | 0.0301 |
| (0.2, 0.4] | 0.3036 | 0.3165 | 0.0128 |
| (0.4, 0.6] | 0.5104 | 0.5032 | 0.0072 |
| (0.6, 0.8] | 0.7122 | 0.7058 | 0.0064 |
| (0.8, 1.0] | 0.8335 | 0.8447 | 0.0112 |
