"""
fit_terciles.py  (PHASE 0, run ONCE, BEFORE any explanation)
===========================================================================
Two goals:

  1. Compute the PRE-HOC confidence tercile boundaries, per domain, over
     the confidence the detector assigns to the TRUE pedestrian detection.
     They are frozen in terciles.json with a timestamp before explanations are
     computed, so that the stratification cannot depend on faithfulness results.

  2. Emit the pool of true instances per domain (eval_index/{domain}.json),
     already labeled with tercile, size stratum and occlusion. run_audit.py samples
     from here; it does not rerun the detector to build the evaluation set.

Usage:
    python fit_terciles.py
Requires Detector and GTLoader implementations (see wiring.py), instantiated below.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone

import numpy as np

from config import CFG
from interfaces import Detector, GTLoader, GTPed, match_tp


# ---------------------------------------------------------------------------
def size_bin(height_px: float) -> str:
    lo, hi = CFG.robustness.size_bins_px
    if height_px < lo:
        return "small"
    if height_px < hi:
        return "medium"
    return "large"


def assign_tercile(conf: float, boundaries: tuple[float, float]) -> str:
    b1, b2 = boundaries
    labels = CFG.terciles.labels
    if conf < b1:
        return labels[0]   # low_conf
    if conf < b2:
        return labels[1]   # mid_conf
    return labels[2]       # high_conf


# ---------------------------------------------------------------------------
def collect_true_positives(detector: Detector, loader: GTLoader, domain: str,
                           available_sets: tuple[str, ...] | None) -> list[dict]:
    """
    Runs the detector on a representative subsample of GT pedestrians and stores
    the true detections. It does not scan the whole dataset: that would be wasteful
    (tens of thousands of inferences) and is not needed to estimate terciles.
    """
    # 1) Materialize ONLY metadata (no detector, cheap) and filter.
    valid = [
        p for p in loader.iter_pedestrians(domain)  # type: ignore[assignment]
        if (available_sets is None or p.set_id in available_sets)
        and p.height >= CFG.sampling.min_gt_height_px
    ]
    n_valid = len(valid)

    # 2) Bounded representative subsample (fixed seed).
    rng = random.Random(CFG.sampling.sampling_seed)
    rng.shuffle(valid)
    scan = valid[:CFG.sampling.fit_max_scan]

    # 3) Sort by (video, frame) for read locality: each video is opened
    #    only once (key for the loader's single-slot cache).
    scan.sort(key=lambda p: (p.video, p.frame_id))

    tps: list[dict] = []
    n_miss, n_read_err = 0, 0
    for ped in scan:
        try:
            frame = loader.load_frame(ped)
        except Exception:
            n_read_err += 1   # unreadable frame: skipped, no crash
            continue
        dets = detector.detect(frame)
        tp = match_tp(dets, ped.box, CFG.run.person_class_id, CFG.sampling.iou_tp_threshold)
        if tp is None:
            n_miss += 1  # detector MISSED: no detection to explain
            continue

        tps.append({
            "instance_id": ped.instance_id,
            "domain": domain,
            "set_id": ped.set_id,
            "video": ped.video,
            "frame_id": ped.frame_id,
            "gt_box": list(ped.box),
            "det_box": list(tp.box),
            "confidence": float(tp.score),
            "occlusion": ped.occlusion,
            "size_bin": size_bin(ped.height),
        })

    print(f"[{domain}] valid={n_valid}  scanned={len(scan)}  "
          f"no_detectados={n_miss}  read_errors={n_read_err}  TP={len(tps)}")
    if hasattr(loader, "close"):
        loader.close()  # release the open video before moving to the other domain
    return tps


def fit_domain(detector: Detector, loader: GTLoader, dom) -> dict:
    tps = collect_true_positives(detector, loader, dom.name, dom.available_sets)
    if len(tps) < 30:
        print(f"  WARNING: {dom.name} has only {len(tps)} TPs. "
              f"Terciles will be unstable; report as a limitation.")
    confs = np.array([t["confidence"] for t in tps], dtype=float)
    q1, q2 = CFG.terciles.quantiles
    b1, b2 = float(np.quantile(confs, q1)), float(np.quantile(confs, q2))
    boundaries = (b1, b2)

    # Label each TP with its (already frozen) tercile.
    for t in tps:
        t["tercile"] = assign_tercile(t["confidence"], boundaries)

    # Save the domain evaluation index.
    CFG.paths.eval_index_dir.mkdir(parents=True, exist_ok=True)
    idx_path = CFG.paths.eval_index_dir / f"{dom.name}.json"
    idx_path.write_text(json.dumps(tps, indent=2), encoding="utf-8")

    counts = {lbl: sum(1 for t in tps if t["tercile"] == lbl) for lbl in CFG.terciles.labels}
    print(f"  [{dom.name}] boundaries={boundaries}  counts_per_tercile={counts}")

    return {
        "domain": dom.name,
        "n_true_positives": len(tps),
        "quantiles": list(CFG.terciles.quantiles),
        "boundaries": list(boundaries),
        "labels": list(CFG.terciles.labels),
        "counts_per_tercile": counts,
        "confidence_min": float(confs.min()) if len(confs) else None,
        "confidence_max": float(confs.max()) if len(confs) else None,
        "available_sets": list(dom.available_sets) if dom.available_sets else "ALL",
        "fit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
def main(detector: Detector, loader: GTLoader) -> None:
    CFG.validate()
    CFG.ensure_dirs()

    frozen = {}
    for dom in CFG.domains:
        frozen[dom.name] = fit_domain(detector, loader, dom)

    CFG.paths.terciles_file.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"\nPRE-HOC terciles frozen in: {CFG.paths.terciles_file}")
    print("From here on the boundaries are NOT modified. run_audit.py only reads them.")


if __name__ == "__main__":
    raise SystemExit("Use `python launch.py fit` so detector and loader are initialized consistently.")
