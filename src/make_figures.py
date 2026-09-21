"""
make_figures.py
===============
Generates the TWO figures supporting Results, rebuilt from the raw JSONL records
(independent of the CSVs). IEEE format: vector PDF, column width, and elements
distinguishable in grayscale (markers and hatching, not only color).

  fig_score_vs_auc.pdf   -> scatter of score_start vs D-Deletion by domain, with a
                            trend line and Spearman rho. Shows both the confound
                            (Results B) and the domain separation (Results C).
  fig_crossdomain_bins.pdf-> D-Deletion gap PIE minus JAAD per score_start bin, with
                            bootstrap CIs; filled bars if p_holm<0.05. This is Results C.

Usage:
    python make_figures.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from config import CFG
import aggregate as A

# --- Publication style (IEEE, 1 column ~3.5in) ---------------------------
plt.rcParams.update({
    # TrueType (Type 42) instead of Type 3: required by IEEE PDF eXpress / PaperCept.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "figure.dpi": 300,
})

# PIE / JAAD: distinct color + marker + line style (grayscale-safe).
STYLE = {
    "PIE":  {"color": "#1f3b73", "marker": "o", "ls": "-"},
    "JAAD": {"color": "#b03a2e", "marker": "^", "ls": "--"},
}


def fig_score_vs_auc(inst, out_path):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for dom in ("PIE", "JAAD"):
        sub = inst[inst["domain"] == dom]
        x = sub["score_start"].to_numpy()
        y = sub["auc"].to_numpy()
        st = STYLE[dom]
        rho, _ = spearmanr(x, y)
        ax.scatter(x, y, s=8, alpha=0.35, color=st["color"], marker=st["marker"],
                   edgecolors="none", label=f"{dom} ($\\rho$={rho:.2f})")
        # trend line (simple linear fit, visual reference only)
        if len(x) > 2:
            b, a = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, b * xs + a, color=st["color"], ls=st["ls"], lw=1.2)
    ax.set_xlabel("Detection strength at explanation time ($f_0$)")
    ax.set_ylabel("D-Deletion AUC")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(frameon=False, loc="upper left")
    ax.margins(0.02)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def fig_crossdomain_bins(ctrl, out_path, alpha=0.05):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = [("[" if i == 0 else "(") + f"{edges[i]:.1f}, {edges[i+1]:.1f}]" for i in range(len(ctrl))]
    gaps = ctrl["gap_pie_minus_jaad"].to_numpy()
    lo = ctrl["gap_ci_lo"].to_numpy()
    hi = ctrl["gap_ci_hi"].to_numpy()
    yerr = np.vstack([gaps - lo, hi - gaps])
    sig = ctrl["p_holm"].to_numpy() < alpha
    x = np.arange(len(labels))

    for i in range(len(labels)):
        # significant: filled bar; not significant: hollow with hatching.
        if sig[i]:
            ax.bar(x[i], gaps[i], color="#1f3b73", edgecolor="black", lw=0.6, width=0.62)
        else:
            ax.bar(x[i], gaps[i], color="white", edgecolor="#1f3b73", lw=0.9,
                   hatch="///", width=0.62)
    ax.errorbar(x, gaps, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("$f_0$ bin")
    ax.set_ylabel("Gap: PIE $-$ JAAD")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.35)
    # manual significance legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#1f3b73", edgecolor="black", label="Holm $p<0.05$"),
        Patch(facecolor="white", edgecolor="#1f3b73", hatch="///", label="n.s."),
    ]
    ax.legend(handles=handles, frameon=True, framealpha=0.9, edgecolor="none", loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def main():
    outdir = CFG.paths.out_root
    df = A.load_records(CFG.paths.records_dir)
    inst = A.per_instance(df)
    edges = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    # The paper figures use the primary method (D-RISE). EigenCAM is
    # discussed in the text; mixing its points, with a very different distribution,
    # would only confuse the reading.
    fig_method = "d_rise"
    inst_fig = inst[inst["method"] == fig_method] if "method" in inst.columns else inst
    ctrl = A.crossdomain_controlled(inst, "auc", edges, n_boot=10000, alpha=0.05)
    if "method" in ctrl.columns:
        ctrl = ctrl[ctrl["method"] == fig_method]

    print(f"Figures generated (method: {fig_method}):")
    fig_score_vs_auc(inst_fig, outdir / "fig_score_vs_auc.pdf")
    if not ctrl.empty:
        fig_crossdomain_bins(ctrl, outdir / "fig_crossdomain_bins.pdf")
    else:
        print("  (no bins with sufficient n for fig_crossdomain_bins)")
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()