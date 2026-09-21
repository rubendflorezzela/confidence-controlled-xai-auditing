"""
aggregate.py
============
Rebuilds the paper results from the raw run_audit.py JSONL records:

  1. Main table: faithfulness (D-Deletion) per (method, domain, tercile), with CIs.
  2. DIRECTIONAL gap: low_conf vs high_conf, WITHIN each domain.
  3. CROSS-DOMAIN gap: PIE vs JAAD, within each tercile.
  4. (if >1 method) Method comparison over common instances.

Statistical tests, each where appropriate:
  - low vs high, and PIE vs JAAD  -> DIFFERENT instances -> Mann-Whitney U (unpaired).
  - method A vs method B         -> SAME instances      -> paired Wilcoxon (signed-rank).
  - Effect size: Cliff's delta.  CI: percentile bootstrap.  p: Holm correction.

Sign interpretation (IMPORTANT):
  low deletion_auc  = MORE faithful explanation (deleting salient pixels collapses the detection fast).
  high deletion_auc = LESS faithful.
  The low_conf vs high_conf comparison is kept as a descriptive diagnostic,
  but it is not interpreted causally because the metric is coupled to f0.

Metrics:
  deletion_auc       -> primary metric for group analysis, method comparison
                      and f0-controlled cross-domain comparison.
  deletion_auc_norm  -> diagnostic sensitivity variant, not the primary metric.

Usage:
  python aggregate.py
  python aggregate.py --metric deletion_auc_norm --min-score-start 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr

from config import CFG

RNG = np.random.default_rng(20260703)
TERCILES = list(CFG.terciles.labels)  # ["low_conf","mid_conf","high_conf"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_records(records_dir: Path) -> pd.DataFrame:
    rows = []
    files = sorted(records_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files in {records_dir}. Run run_audit.py first.")
    for fp in files:
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} records from {len(files)} file(s).")
    return df


def per_instance(df: pd.DataFrame) -> pd.DataFrame:
    """Averages over seeds -> one value per instance. Keeps the between-seed SD."""
    g = df.groupby(["method", "domain", "tercile", "instance_id"], as_index=False)
    inst = g.agg(
        auc=("deletion_auc", "mean"),
        auc_norm=("deletion_auc_norm", "mean"),
        seed_sd=("deletion_auc", "std"),
        score_start=("score_start", "mean"),
        confidence=("confidence", "first"),
        size_bin=("size_bin", "first"),
        occlusion=("occlusion", "first"),
    )
    return inst


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci_mean(x: np.ndarray, n_boot: int, alpha: float) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_ci_gap(x: np.ndarray, y: np.ndarray, n_boot: int, alpha: float) -> tuple[float, float]:
    """CI of the difference in means (mean(x)-mean(y)) for independent groups."""
    if len(x) < 2 or len(y) < 2:
        return (float("nan"), float("nan"))
    bx = x[RNG.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    by = y[RNG.integers(0, len(y), size=(n_boot, len(y)))].mean(axis=1)
    diff = bx - by
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Cliff's delta (effect size) and two-sided Mann-Whitney U p-value."""
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    U, p = mannwhitneyu(x, y, alternative="two-sided")
    delta = 2.0 * U / (len(x) * len(y)) - 1.0   # >0 => x tends to be larger than y
    return float(delta), float(p)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni correction. Returns adjusted p-values in the original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [float("nan")] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# 1) Main table by group
# ---------------------------------------------------------------------------
def summarize_groups(inst: pd.DataFrame, n_boot: int, alpha: float) -> pd.DataFrame:
    out = []
    for (method, domain, tercile), sub in inst.groupby(["method", "domain", "tercile"]):
        auc = sub["auc"].to_numpy()
        aucn = sub["auc_norm"].to_numpy()
        f0 = sub["score_start"].to_numpy()

        lo, hi = bootstrap_ci_mean(auc, n_boot, alpha)
        lon, hin = bootstrap_ci_mean(aucn, n_boot, alpha)

        q1, q3 = np.quantile(auc, [0.25, 0.75])
        q1n, q3n = np.quantile(aucn, [0.25, 0.75])
        q1f0, q3f0 = np.quantile(f0, [0.25, 0.75])

        out.append({
            "method": method,
            "domain": domain,
            "tercile": tercile,
            "n": len(sub),

            "auc_mean": auc.mean(),
            "auc_median": np.median(auc),
            "auc_q1": q1,
            "auc_q3": q3,
            "auc_iqr": q3 - q1,
            "auc_ci_lo": lo,
            "auc_ci_hi": hi,

            "auc_norm_mean": aucn.mean(),
            "auc_norm_median": np.median(aucn),
            "auc_norm_q1": q1n,
            "auc_norm_q3": q3n,
            "auc_norm_iqr": q3n - q1n,
            "auc_norm_ci_lo": lon,
            "auc_norm_ci_hi": hin,

            "score_start_mean": f0.mean(),
            "score_start_median": np.median(f0),
            "score_start_q1": q1f0,
            "score_start_q3": q3f0,
            "score_start_iqr": q3f0 - q1f0,

            "seed_sd_mean": sub["seed_sd"].mean(),
        })

    df = pd.DataFrame(out)
    df["tercile"] = pd.Categorical(df["tercile"], TERCILES, ordered=True)
    return df.sort_values(["method", "domain", "tercile"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2) Directional gap (low vs high, within domain) -- Mann-Whitney U
# ---------------------------------------------------------------------------
def directional_gaps(inst: pd.DataFrame, metric: str, n_boot: int, alpha: float) -> pd.DataFrame:
    out = []
    for (method, domain), sub in inst.groupby(["method", "domain"]):
        low = sub.loc[sub["tercile"] == "low_conf", metric].to_numpy()
        high = sub.loc[sub["tercile"] == "high_conf", metric].to_numpy()
        if len(low) < 2 or len(high) < 2:
            continue
        delta, p = cliffs_delta(low, high)
        glo, ghi = bootstrap_ci_gap(low, high, n_boot, alpha)
        out.append({
            "method": method, "domain": domain, "metric": metric,
            "low_mean": low.mean(), "high_mean": high.mean(),
            "gap_low_minus_high": low.mean() - high.mean(),
            "gap_ci_lo": glo, "gap_ci_hi": ghi,
            "cliffs_delta": delta, "p_raw": p,
            "n_low": len(low), "n_high": len(high),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 3) Cross-domain gap (PIE vs JAAD, within tercile) -- Mann-Whitney U
# ---------------------------------------------------------------------------
def crossdomain_gaps(inst: pd.DataFrame, metric: str, n_boot: int, alpha: float) -> pd.DataFrame:
    out = []
    for (method, tercile), sub in inst.groupby(["method", "tercile"]):
        pie = sub.loc[sub["domain"] == "PIE", metric].to_numpy()
        jaad = sub.loc[sub["domain"] == "JAAD", metric].to_numpy()
        if len(pie) < 2 or len(jaad) < 2:
            continue
        delta, p = cliffs_delta(pie, jaad)
        glo, ghi = bootstrap_ci_gap(pie, jaad, n_boot, alpha)
        out.append({
            "method": method, "tercile": tercile, "metric": metric,
            "pie_mean": pie.mean(), "jaad_mean": jaad.mean(),
            "gap_pie_minus_jaad": pie.mean() - jaad.mean(),
            "gap_ci_lo": glo, "gap_ci_hi": ghi,
            "cliffs_delta": delta, "p_raw": p,
            "n_pie": len(pie), "n_jaad": len(jaad),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df["tercile"] = pd.Categorical(df["tercile"], TERCILES, ordered=True)
        df = df.sort_values(["method", "tercile"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3b) Confound: the deletion metric depends on detector confidence
# ---------------------------------------------------------------------------
def confound_report(inst: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman correlation between deletion_auc and (score_start, confidence),
    SEPARATED BY METHOD. D-RISE and EigenCAM have very different AUC
    distributions, so mixing them would blur the coupling; one result is reported
    per method, globally, per domain, and within each domain-tercile.
    """
    out = []
    for method in sorted(inst["method"].unique()):
        mi = inst[inst["method"] == method]
        scopes = [("ALL", mi)]
        scopes += [(d, mi[mi["domain"] == d]) for d in sorted(mi["domain"].unique())]
        scopes += [(f"{d}/{t}", mi[(mi["domain"] == d) & (mi["tercile"] == t)])
                   for d in sorted(mi["domain"].unique()) for t in TERCILES]
        for scope, sub in scopes:
            if len(sub) < 3:
                continue
            for xvar in ("score_start", "confidence"):
                r, p = spearmanr(sub["auc"], sub[xvar])
                out.append({"method": method, "scope": scope, "x": xvar,
                            "spearman_rho": float(r), "p": float(p), "n": len(sub)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 3c) Cross-domain at MATCHED CONFIDENCE (score_start bins) -- confound control
# ---------------------------------------------------------------------------
def crossdomain_controlled(inst: pd.DataFrame, metric: str, edges: tuple,
                           n_boot: int, alpha: float, min_n: int = 15) -> pd.DataFrame:
    """
    Compares PIE vs JAAD within score_start bins, separately per method.
    Reports mean, median, IQR, and the residual f0 difference within each bin.
    """
    sub = inst.copy()
    sub["s0_bin"] = pd.cut(sub["score_start"], bins=list(edges), include_lowest=True)

    frames = []
    for method in sorted(sub["method"].unique()):
        mi = sub[sub["method"] == method]
        rows = []

        for b, g in mi.groupby("s0_bin", observed=True):
            pie_mask = g["domain"] == "PIE"
            jaad_mask = g["domain"] == "JAAD"

            pie = g.loc[pie_mask, metric].to_numpy()
            jaad = g.loc[jaad_mask, metric].to_numpy()
            pie_f0 = g.loc[pie_mask, "score_start"].to_numpy()
            jaad_f0 = g.loc[jaad_mask, "score_start"].to_numpy()

            if len(pie) < min_n or len(jaad) < min_n:
                continue

            delta, p = cliffs_delta(pie, jaad)
            glo, ghi = bootstrap_ci_gap(pie, jaad, n_boot, alpha)

            pie_q1, pie_q3 = np.quantile(pie, [0.25, 0.75])
            jaad_q1, jaad_q3 = np.quantile(jaad, [0.25, 0.75])

            f0_pie_mean = pie_f0.mean()
            f0_jaad_mean = jaad_f0.mean()
            f0_gap = f0_pie_mean - f0_jaad_mean

            rows.append({
                "method": method,
                "s0_bin": str(b),
                "metric": metric,

                "pie_mean": pie.mean(),
                "jaad_mean": jaad.mean(),
                "gap_pie_minus_jaad": pie.mean() - jaad.mean(),
                "gap_ci_lo": glo,
                "gap_ci_hi": ghi,

                "pie_median": np.median(pie),
                "jaad_median": np.median(jaad),
                "median_gap_pie_minus_jaad": np.median(pie) - np.median(jaad),

                "pie_iqr": pie_q3 - pie_q1,
                "jaad_iqr": jaad_q3 - jaad_q1,

                "cliffs_delta": delta,
                "p_raw": p,
                "n_pie": len(pie),
                "n_jaad": len(jaad),

                "f0_pie_mean": f0_pie_mean,
                "f0_jaad_mean": f0_jaad_mean,
                "f0_gap_pie_minus_jaad": f0_gap,
                "f0_abs_gap": abs(f0_gap),
                "f0_pie_median": np.median(pie_f0),
                "f0_jaad_median": np.median(jaad_f0),
            })

        dfm = pd.DataFrame(rows)
        if not dfm.empty:
            dfm["p_holm"] = holm(list(dfm["p_raw"]))
        frames.append(dfm)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def method_comparison(inst: pd.DataFrame, metric: str) -> pd.DataFrame:
    methods = sorted(inst["method"].unique())
    if len(methods) < 2:
        return pd.DataFrame()

    out = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            da = inst[inst["method"] == a].set_index("instance_id")[metric]
            db = inst[inst["method"] == b].set_index("instance_id")[metric]
            common = da.index.intersection(db.index)

            if len(common) < 2:
                continue

            xa = da.loc[common].to_numpy()
            xb = db.loc[common].to_numpy()
            diff = xa - xb

            try:
                _, p = wilcoxon(xa, xb)
            except ValueError:
                p = float("nan")

            out.append({
                "method_a": a,
                "method_b": b,
                "metric": metric,
                "mean_a": xa.mean(),
                "mean_b": xb.mean(),
                "mean_diff": diff.mean(),
                "median_a": np.median(xa),
                "median_b": np.median(xb),
                "median_diff": np.median(diff),
                "q1_diff": np.quantile(diff, 0.25),
                "q3_diff": np.quantile(diff, 0.75),
                "iqr_diff": np.quantile(diff, 0.75) - np.quantile(diff, 0.25),
                "p_raw": p,
                "n_pairs": len(common),
            })

    return pd.DataFrame(out)



def robustness_descriptives(inst: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive summary by size_bin and occlusion.
    It does not replace the main analysis; it documents potential confounds.
    """
    rows = []
    for keys, sub in inst.groupby(["method", "domain", "tercile", "size_bin", "occlusion"]):
        method, domain, tercile, size_bin, occlusion = keys
        auc = sub["auc"].to_numpy()
        if len(auc) < 2:
            continue
        q1, q3 = np.quantile(auc, [0.25, 0.75])
        rows.append({
            "method": method,
            "domain": domain,
            "tercile": tercile,
            "size_bin": size_bin,
            "occlusion": occlusion,
            "n": len(sub),
            "auc_mean": auc.mean(),
            "auc_median": np.median(auc),
            "auc_iqr": q3 - q1,
            "score_start_mean": sub["score_start"].mean(),
            "score_start_median": sub["score_start"].median(),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["tercile"] = pd.Categorical(df["tercile"], TERCILES, ordered=True)
        df = df.sort_values(
            ["method", "domain", "tercile", "size_bin", "occlusion"]
        ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
    "--metric",
    default="deletion_auc",
    choices=["deletion_auc", "deletion_auc_norm"],
    help="Metric for uncontrolled gaps. deletion_auc is primary; deletion_auc_norm is diagnostic.",)
    ap.add_argument("--min-score-start", type=float, default=0.0,
                    help="Discards instances whose mean score_start < threshold (low_conf robustness check).")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = load_records(CFG.paths.records_dir)
    inst = per_instance(df)

    # per_instance renames the columns to auc / auc_norm.
    metric_col = {"deletion_auc": "auc", "deletion_auc_norm": "auc_norm"}[args.metric]

    if args.min_score_start > 0:
        before = len(inst)
        inst = inst[inst["score_start"] >= args.min_score_start].copy()
        print(f"Filter score_start>={args.min_score_start}: {before-len(inst)} instances discarded.")

    # Context report: number of instances per cell.
    print("\nInstances per (method, domain, tercile):")
    print(inst.groupby(["method", "domain", "tercile"]).size().to_string())

    groups = summarize_groups(inst, args.n_boot, args.alpha)
    dirg = directional_gaps(inst, metric_col, args.n_boot, args.alpha)
    xdom = crossdomain_gaps(inst, metric_col, args.n_boot, args.alpha)
    meth = method_comparison(inst, metric_col)
    conf = confound_report(inst)
    edges = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    edges_alt = (0.0, 0.25, 0.5, 0.75, 1.0)

    ctrl = crossdomain_controlled(inst, "auc", edges, args.n_boot, args.alpha)
    ctrl_norm = crossdomain_controlled(inst, "auc_norm", edges, args.n_boot, args.alpha)
    ctrl_alt = crossdomain_controlled(inst, "auc", edges_alt, args.n_boot, args.alpha)
    rob = robustness_descriptives(inst)

    # Holm correction over the family of gaps (directional + cross-domain).
    fam = list(dirg["p_raw"]) + list(xdom["p_raw"])
    if fam:
        adj = holm([p for p in fam if not np.isnan(p)])
        # remap while preserving NaN
        it = iter(adj)
        dirg["p_holm"] = [next(it) if not np.isnan(p) else float("nan") for p in dirg["p_raw"]]
        xdom["p_holm"] = [next(it) if not np.isnan(p) else float("nan") for p in xdom["p_raw"]]

    outdir = CFG.paths.out_root
    groups.to_csv(outdir / "agg_by_group.csv", index=False)
    dirg.to_csv(outdir / "agg_directional.csv", index=False)
    xdom.to_csv(outdir / "agg_crossdomain.csv", index=False)
    conf.to_csv(outdir / "agg_confound.csv", index=False)
    if not ctrl.empty:
        ctrl.to_csv(outdir / "agg_crossdomain_controlled.csv", index=False)
    if not meth.empty:
        meth.to_csv(outdir / "agg_method_comparison.csv", index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    print("\n=== MAIN TABLE: faithfulness by group ===")
    print(groups.round(4).to_string(index=False))
    print("\n=== DIRECTIONAL GAP (low_conf - high_conf), within domain ===")
    print("  gap>0 => higher AUC in low_conf; interpret only as a confounded diagnostic")
    print((dirg.round(4).to_string(index=False)) if not dirg.empty else "  (insufficient data)")
    print("\n=== CONFOUND: Spearman(deletion_auc, x) ===")
    print("  high positive rho => measured faithfulness is coupled to confidence")
    print(conf.round(4).to_string(index=False))
    print("\n=== CROSS-DOMAIN at MATCHED CONFIDENCE (score_start bins), raw metric ===")
    print("  if the PIE-JAAD gap persists within bins => genuine domain effect")
    print((ctrl.round(4).to_string(index=False)) if not ctrl.empty else "  (no bins with sufficient n)")
    print("\n=== CROSS-DOMAIN GAP (uncontrolled), within tercile ===")
    print((xdom.round(4).to_string(index=False)) if not xdom.empty else "  (insufficient data)")
    if not meth.empty:
        print("\n=== METHOD COMPARISON (paired Wilcoxon) ===")
        print(meth.round(4).to_string(index=False))
    if not ctrl.empty:
        ctrl.to_csv(outdir / "agg_crossdomain_controlled.csv", index=False)
    if not ctrl_norm.empty:
        ctrl_norm.to_csv(outdir / "agg_crossdomain_controlled_norm.csv", index=False)
    if not ctrl_alt.empty:
        ctrl_alt.to_csv(outdir / "agg_crossdomain_controlled_bins025.csv", index=False)
    if not rob.empty:
        rob.to_csv(outdir / "agg_robustness_size_occlusion.csv", index=False)

    print(f"\nCSVs saved to {outdir}")


if __name__ == "__main__":
    main()