"""
make_qualitative.py
===================

Generates a qualitative figure comparing D-RISE vs EigenCAM
on the SAME instances, using the same ROI and the same D-Deletion
metric as the paper.

Output:
    results/fig_qualitative_drise_vs_eigencam_curves.pdf
    results/fig_qualitative_drise_vs_eigencam_curves.png

Recommended usage:
    python make_qualitative.py

Optional:
    python make_qualitative.py --bin-lo 0.6 --bin-hi 0.8
    python make_qualitative.py --seeds 0
    python make_qualitative.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from config import CFG
from interfaces import GTPed, compute_roi, target_vector_score
from wiring import build_detector, build_loader
import run_audit

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})



# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def safe_name(s: str) -> str:
    """Windows-safe file name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def load_jsonl_records(records_dir: Path) -> pd.DataFrame:
    rows = []
    for fp in sorted(records_dir.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No JSONL records in {records_dir}")
    return pd.DataFrame(rows)


def per_instance(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "domain", "tercile", "instance_id"], as_index=False)
        .agg(
            auc=("deletion_auc", "mean"),
            auc_norm=("deletion_auc_norm", "mean"),
            score_start=("score_start", "mean"),
            confidence=("confidence", "first"),
        )
    )


def make_pivot(inst: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a table with one row per instance and columns:
    d_rise_auc, eigencam_auc, score_start, domain, tercile.
    """
    base = inst.pivot_table(
        index="instance_id",
        columns="method",
        values="auc",
        aggfunc="mean",
    ).reset_index()

    base = base.rename(columns={
        "d_rise": "d_rise_auc",
        "eigencam": "eigencam_auc",
    })

    meta = (
        inst.sort_values(["instance_id", "method"])
        .groupby("instance_id", as_index=False)
        .agg(
            domain=("domain", "first"),
            tercile=("tercile", "first"),
            score_start=("score_start", "mean"),
            confidence=("confidence", "first"),
        )
    )

    out = meta.merge(base, on="instance_id", how="inner")
    out = out.dropna(subset=["d_rise_auc", "eigencam_auc"])
    out["method_gap"] = out["eigencam_auc"] - out["d_rise_auc"]
    return out

def select_representative_cases(
    pivot: pd.DataFrame,
    bin_lo: float,
    bin_hi: float,
    lookup: dict,
    domains=("PIE", "JAAD"),
    max_aspect: float = 3.0,
    n_show: int = 12,
) -> list[str]:
    """
    Selects one representative instance per domain within the score_start bin.

    Criteria, in this order:
    - inside the score_start bin [bin_lo, bin_hi);
    - box aspect ratio (height/width) <= max_aspect, to discard
  overly thin crops that render poorly, WITHOUT looking at the AUC;
    - among legible boxes, the one closest to the median EigenCAM - D-RISE gap.

    The filter is for LEGIBILITY, not for results: selection never favors a lower AUC.
    The caption states that the example is representative of the bin among
    legible-scale detections with a gap close to the median.
    """
    chosen = []
    center = 0.5 * (bin_lo + bin_hi)

    for dom in domains:
        cand = pivot[
            (pivot["domain"] == dom)
            & (pivot["score_start"] >= bin_lo)
            & (pivot["score_start"] < bin_hi)
        ].copy()

        if cand.empty:
            raise RuntimeError(
                f"No candidates for {dom} in score_start [{bin_lo}, {bin_hi})."
            )

        # Aspect ratio of the detected box (height/width).
        def _aspect(iid: str) -> float:
            box = lookup[iid]["det_box"]
            w = max(box[2] - box[0], 1e-6)
            h = max(box[3] - box[1], 1e-6)
            return h / w

        cand["aspect"] = cand["instance_id"].map(_aspect)

        # Legibility filter. If none passes, it is relaxed so an example is still available.
        legible = cand[cand["aspect"] <= max_aspect].copy()
        if legible.empty:
            print(f"[{dom}] WARNING: no box with aspect <= {max_aspect}; "
                  f"using the least elongated boxes in the bin.")
            legible = cand.sort_values("aspect").head(max(1, n_show)).copy()

        med_gap = legible["method_gap"].median()
        s0_scale = max(legible["score_start"].std(), 1e-6)
        gap_scale = max(legible["method_gap"].std(), 1e-6)
        legible["select_score"] = (
            np.abs(legible["score_start"] - center) / s0_scale
            + np.abs(legible["method_gap"] - med_gap) / gap_scale
        )

        legible = legible.sort_values("select_score")

        # Show the N legible candidates closest to the median, for transparency.
        print(f"\n[{dom}] legible candidates in bin [{bin_lo},{bin_hi}) "
              f"(aspect <= {max_aspect}), sorted by closeness to the median:")
        for _, rr in legible.head(n_show).iterrows():
            print(f"  {rr['instance_id']} | aspect={rr['aspect']:.2f} | "
                  f"s0={rr['score_start']:.3f} | D-RISE={rr['d_rise_auc']:.3f} | "
                  f"EigenCAM={rr['eigencam_auc']:.3f}")

        row = legible.iloc[0]
        chosen.append(row["instance_id"])
        print(f"[selected] {dom}: {row['instance_id']} | aspect={row['aspect']:.2f}")

    return chosen

def build_instance_lookup(domains=("PIE", "JAAD")) -> dict[str, dict]:
    """
    Rebuilds the same evaluation set using run_audit.build_eval_set.
    """
    lookup = {}
    for dom in domains:
        for inst in run_audit.build_eval_set(dom):
            lookup[inst["instance_id"]] = inst
    return lookup


def load_image_from_instance(loader, inst: dict) -> np.ndarray:
    ped = GTPed(
        domain=inst["domain"],
        set_id=inst.get("set_id"),
        video=inst["video"],
        frame_id=inst["frame_id"],
        box=tuple(inst["gt_box"]),
        occlusion=inst["occlusion"],
    )
    return loader.load_frame(ped)


def normalize_saliency(sal: np.ndarray) -> np.ndarray:
    sal = sal.astype(np.float32)
    mn, mx = float(np.min(sal)), float(np.max(sal))
    if mx <= mn + 1e-12:
        return np.zeros_like(sal, dtype=np.float32)
    return (sal - mn) / (mx - mn)


def overlay_heatmap(rgb: np.ndarray, sal: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlays a heat map on an RGB image.
    """
    sal_n = normalize_saliency(sal)
    cmap = plt.get_cmap("jet")
    heat = (cmap(sal_n)[..., :3] * 255).astype(np.uint8)
    rgb_u8 = rgb.astype(np.uint8)
    out = ((1 - alpha) * rgb_u8 + alpha * heat).clip(0, 255).astype(np.uint8)
    return out


def draw_box(ax, box, color="lime", lw=1.5):
    x1, y1, x2, y2 = box
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=lw,
        )
    )


def deletion_curve(image, target_box, target_cls, saliency, detector):
    """
    Computes the D-Deletion curve for plotting.

    - The AUC is computed on the raw response, as in the primary metric.
    - The curve is normalized by the initial score for visualization only.
    - The x-axis is normalized by the maximum deletion budget.
    """
    import cv2

    H_img, W_img = image.shape[:2]

    sal = saliency.astype(np.float32)
    if sal.shape[:2] != (H_img, W_img):
        sal = cv2.resize(sal, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

    H, W = sal.shape
    order = np.argsort(sal.ravel())[::-1]

    steps = CFG.xai.deletion_steps
    max_frac = CFG.xai.deletion_max_frac
    fracs = np.linspace(0.0, max_frac, steps)
    total = H * W

    work = image.astype(np.float32).copy()
    scores = []
    prev_k = 0

    for fr in fracs:
        k = int(fr * total)
        if k > prev_k:
            idx = order[prev_k:k]
            ys, xs = np.unravel_index(idx, (H, W))
            work[ys, xs, :] = 0.0
            prev_k = k

        dets = detector.detect(work.astype(np.uint8))
        scores.append(target_vector_score(dets, target_box, target_cls))

    scores = np.asarray(scores, dtype=float)

    trapz = getattr(np, "trapezoid", np.trapz)
    auc_raw = float(trapz(scores, fracs) / max_frac)

    s0 = float(scores[0])
    if s0 <= 1e-8:
        scores_norm = scores
    else:
        scores_norm = scores / s0

    x_norm = fracs / max_frac
    return x_norm, scores_norm, auc_raw


def compute_or_load_saliency(
    method: str,
    roi_img: np.ndarray,
    roi_box,
    target_cls: int,
    detector,
    seeds: list[int],
    cache_dir: Path,
    instance_id: str,
) -> np.ndarray:
    """
    Computes or loads the saliency map.

    For D-RISE:
    averages the saliency maps over the given seeds.
    For EigenCAM:
    uses a single deterministic saliency map.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    if method == "eigencam":
        seeds_to_use = [0]
    else:
        seeds_to_use = seeds

    maps = []
    for seed in seeds_to_use:
        cache_path = cache_dir / f"{safe_name(instance_id)}__{method}__seed{seed}.npy"

        if cache_path.exists():
            sal = np.load(cache_path)
        else:
            run_audit.set_seed(seed)
            explainer = run_audit.EXPLAINERS[method]
            sal = explainer.explain(
                roi_img,
                roi_box,
                target_cls,
                detector,
                seed,
            )
            np.save(cache_path, sal.astype(np.float32))

        maps.append(sal.astype(np.float32))

    return np.mean(maps, axis=0).astype(np.float32)


# ---------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------
def make_figure(case_ids: list[str], seeds: list[int], bin_lo: float, bin_hi: float):
    out_root = CFG.paths.out_root
    fig_dir = out_root
    cache_dir = out_root / "qualitative_cache"

    records = load_jsonl_records(CFG.paths.records_dir)
    inst_df = per_instance(records)
    pivot = make_pivot(inst_df)

    lookup = build_instance_lookup()
    detector = build_detector()
    loader = build_loader()
    run_audit._LOADER = loader

    rows = len(case_ids)
    cols = 4

    # IEEE double-column width. Compact height to fit within 6 pages.
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(7.16, 3.35),
        gridspec_kw={
            "width_ratios": [0.82, 0.92, 0.92, 1.28],
            "wspace": 0.10,
            "hspace": 0.22,
        },
    )

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, instance_id in enumerate(case_ids):
        if instance_id not in lookup:
            raise KeyError(f"{instance_id} not found in the rebuilt eval_index.")

        meta = pivot[pivot["instance_id"] == instance_id].iloc[0]
        inst = lookup[instance_id]

        image_full = load_image_from_instance(loader, inst)
        tbox_full = tuple(inst["det_box"])
        target_cls = CFG.run.person_class_id

        roi_img, roi_box, _ = compute_roi(image_full, tbox_full, CFG.xai.roi_margin)

        sal_drise = compute_or_load_saliency(
            "d_rise",
            roi_img,
            roi_box,
            target_cls,
            detector,
            seeds,
            cache_dir,
            instance_id,
        )

        sal_eigen = compute_or_load_saliency(
            "eigencam",
            roi_img,
            roi_box,
            target_cls,
            detector,
            seeds,
            cache_dir,
            instance_id,
        )

        drise_overlay = overlay_heatmap(roi_img, sal_drise, alpha=0.45)
        eigen_overlay = overlay_heatmap(roi_img, sal_eigen, alpha=0.45)

        x_d, y_d, auc_d = deletion_curve(
            roi_img,
            roi_box,
            target_cls,
            sal_drise,
            detector,
        )

        x_e, y_e, auc_e = deletion_curve(
            roi_img,
            roi_box,
            target_cls,
            sal_eigen,
            detector,
        )

        domain = str(meta["domain"])

        # Column 1: original ROI
        ax = axes[r, 0]
        ax.imshow(roi_img)
        draw_box(ax, roi_box, color="lime", lw=1.25)
        ax.set_title(f"{domain} ROI", fontsize=8, pad=2)
        ax.axis("off")

        # Column 2: D-RISE
        ax = axes[r, 1]
        ax.imshow(drise_overlay)
        draw_box(ax, roi_box, color="white", lw=1.10)
        ax.set_title(f"D-RISE\nAUC={auc_d:.3f}", fontsize=8, pad=2)
        ax.axis("off")

        # Column 3: EigenCAM
        ax = axes[r, 2]
        ax.imshow(eigen_overlay)
        draw_box(ax, roi_box, color="white", lw=1.10)
        ax.set_title(f"EigenCAM\nAUC={auc_e:.3f}", fontsize=8, pad=2)
        ax.axis("off")

        # Column 4: deletion curves
        ax = axes[r, 3]
        ax.plot(x_d, y_d, label="D-RISE", linewidth=1.25)
        ax.plot(x_e, y_e, label="EigenCAM", linewidth=1.25, linestyle="--")

        ymax = max(1.03, float(np.nanmax([y_d.max(), y_e.max()])) + 0.03)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, ymax)

        ax.set_ylabel("Norm. score", fontsize=8)
        if r == rows - 1:
            ax.set_xlabel("Normalized deletion budget", fontsize=8)
        else:
            ax.set_xlabel("")

        ax.tick_params(axis="both", labelsize=7, width=0.6, length=2.5)
        ax.grid(True, linewidth=0.3, alpha=0.35)

        # A single legend is enough and saves space.
        if r == 0:
            ax.legend(
                frameon=False,
                loc="lower left",
                fontsize=7,
                handlelength=1.8,
                borderaxespad=0.2,
            )

    # Fine compaction for IEEE.
    fig.subplots_adjust(
        left=0.015,
        right=0.995,
        top=0.925,
        bottom=0.125,
        wspace=0.10,
        hspace=0.22,
    )

    pdf_path = fig_dir / "fig_qualitative_drise_vs_eigencam_curves.pdf"
    png_path = fig_dir / "fig_qualitative_drise_vs_eigencam_curves.png"

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print("\nFigure with curves saved to:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")

    try:
        loader.close()
    except Exception:
        pass


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-lo", type=float, default=0.6)
    ap.add_argument("--bin-hi", type=float, default=0.8)
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help=(
            "D-RISE seeds. Use '0' for speed. "
            "Use '0 1 2 3 4' for a smoother averaged map."
        ),
    )
    ap.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help=(
            "Optional: specific instance_ids. "
            "If not given, a representative PIE and JAAD case is selected."
        ),
    )
    args = ap.parse_args()

    records = load_jsonl_records(CFG.paths.records_dir)
    inst = per_instance(records)
    pivot = make_pivot(inst)

    if args.instances:
        case_ids = args.instances
        print("Using the specified instances:")
        for cid in case_ids:
            print(f"  {cid}")
    #else:
    #    case_ids = select_representative_cases(
    #        pivot,
    #        bin_lo=args.bin_lo,
    #        bin_hi=args.bin_hi,
    #        domains=("PIE", "JAAD"),
    #    )

    else:
        lookup = build_instance_lookup(domains=("PIE", "JAAD"))
        case_ids = select_representative_cases(
            pivot,
            bin_lo=args.bin_lo,
            bin_hi=args.bin_hi,
            lookup=lookup,
            domains=("PIE", "JAAD"),
        )

    make_figure(
        case_ids=case_ids,
        seeds=args.seeds,
        bin_lo=args.bin_lo,
        bin_hi=args.bin_hi,
    )


if __name__ == "__main__":
    main()