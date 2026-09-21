"""
run_audit.py  (multi-seed orchestrator, crash-safe)
=====================================================
Reads the ALREADY frozen tercile boundaries (terciles.json) and the per-domain
instance pool (eval_index/{domain}.json), builds the evaluation set with a
shared cap and stratified sampling, and for each

        (domain, instance, XAI_method, seed)

computes the explanation and D-Deletion and persists the record ON THE FLY to
JSONL (append). If the process crashes it resumes without losing work: it re-reads
completed records and skips them.

Aggregation (paired Wilcoxon, CIs, cross-domain gap, directional gap, main
table) is done in a SEPARATE script that rebuilds everything from these JSONL
files without recomputation. This runner only produces the raw numbers robustly.

Included methods:
    d_rise    -> perturbation-based, implemented end-to-end.
    eigencam  -> non-perturbative baseline, implemented in wiring.py.

Usage:
    python run_audit.py                 # everything
    python run_audit.py --smoke 3       # 3 instances per domain (quick test)
    python run_audit.py --domains PIE --methods d_rise
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np

from config import CFG
from interfaces import Detector, GTLoader, GTPed, target_vector_score, compute_roi


# ===========================================================================
# Reproducibility utilities
# ===========================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ===========================================================================
# Incremental store (JSONL append + resume)
# ===========================================================================
class ResultStore:
    """One JSONL file per (domain, method). Append-only, resumable."""

    def __init__(self, domain: str, method: str):
        CFG.paths.records_dir.mkdir(parents=True, exist_ok=True)
        self.path = CFG.paths.records_dir / f"{domain}__{method}.jsonl"
        self._done: set[tuple[str, int]] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    self._done.add((r["instance_id"], int(r["seed"])))
                except Exception:
                    continue  # half-written line from a crash: ignored

    def is_done(self, instance_id: str, seed: int) -> bool:
        return (instance_id, seed) in self._done

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())   # real durability against power loss
        self._done.add((record["instance_id"], int(record["seed"])))


# ===========================================================================
# Evaluation-set construction (shared cap + stratified)
# ===========================================================================
def build_eval_set(domain: str) -> list[dict]:
    idx_path = CFG.paths.eval_index_dir / f"{domain}.json"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"{idx_path} does not exist. Run fit_terciles.py first (PHASE 0)."
        )
    pool = json.loads(idx_path.read_text(encoding="utf-8"))
    rng = random.Random(CFG.sampling.sampling_seed)
    cap = CFG.sampling.max_instances_per_domain

    if not CFG.sampling.balance_across_terciles:
        rng.shuffle(pool)
        return pool[:cap]

    # Stratified: same quota per tercile (up to availability).
    per = cap // 3
    chosen: list[dict] = []
    for lbl in CFG.terciles.labels:
        bucket = [t for t in pool if t["tercile"] == lbl]
        rng.shuffle(bucket)
        chosen.extend(bucket[:per])
    rng.shuffle(chosen)
    print(f"[{domain}] evaluation set = {len(chosen)} instances "
          f"(cap {cap}, ~{per}/tercile)")
    return chosen


# ===========================================================================
# Explainers
# ===========================================================================
class Explainer:
    name = "base"

    def explain(self, image, target_box, target_cls, detector, seed) -> np.ndarray:
        raise NotImplementedError


class DRISEExplainer(Explainer):
    """
    Reference D-RISE. Random low-resolution masks, upsampled with a
    continuous shift, and saliency = sum weighted by the target vector.
    The seed controls the mask RNG: this is what varies across seeds.
    """
    name = "d_rise"

    def explain(self, image, target_box, target_cls, detector: Detector, seed: int) -> np.ndarray:
        H, W = image.shape[:2]
        g = CFG.xai.mask_grid
        p = CFG.xai.mask_prob
        n = CFG.xai.n_masks
        B = CFG.run.mask_batch_size
        rng = np.random.default_rng(seed)

        cell_h = int(np.ceil(H / g))
        cell_w = int(np.ceil(W / g))
        up_h, up_w = (g + 1) * cell_h, (g + 1) * cell_w

        saliency = np.zeros((H, W), dtype=np.float64)
        i = 0
        while i < n:
            k = min(B, n - i)
            masks, imgs = [], []
            for _ in range(k):
                grid = (rng.random((g, g)) < p).astype(np.float32)
                big = _upsample_nearest(grid, up_h, up_w)
                oy, ox = int(rng.integers(0, cell_h)), int(rng.integers(0, cell_w))
                mask = big[oy:oy + H, ox:ox + W]
                masks.append(mask)
                imgs.append((image.astype(np.float32) * mask[..., None]).astype(np.uint8))
            dets_list = detector.detect_batch(imgs)   # ONE call per batch (key for speed)
            for mask, dets in zip(masks, dets_list):
                s = target_vector_score(dets, target_box, target_cls)
                saliency += s * mask
            i += k
        denom = (n * p) if (n * p) > 0 else 1.0
        return (saliency / denom).astype(np.float32)



class EigenCAMExplainer(Explainer):
    """Non-perturbative CAM baseline. Delegates to detector.eigencam (see wiring.py).
    It is deterministic, so the seed does not affect it; it is recorded anyway for uniformity."""
    name = "eigencam"

    def explain(self, image, target_box, target_cls, detector, seed):
        if not hasattr(detector, "eigencam"):
            raise NotImplementedError("The detector does not expose eigencam().")
        return detector.eigencam(image)


def _upsample_nearest(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Upsampling by repetition (no dependencies). Sufficient for RISE masks."""
    g_h, g_w = grid.shape
    ry, rx = int(np.ceil(out_h / g_h)), int(np.ceil(out_w / g_w))
    big = np.repeat(np.repeat(grid, ry, axis=0), rx, axis=1)
    return big[:out_h, :out_w]


EXPLAINERS: dict[str, Explainer] = {
    DRISEExplainer.name: DRISEExplainer(),
    EigenCAMExplainer.name: EigenCAMExplainer(),
}


# ===========================================================================
# D-Deletion (reference)
# ===========================================================================
def d_deletion(image, target_box, target_cls, saliency: np.ndarray,
               detector: Detector) -> dict:
    """
    Detection-aware deletion curve: the most salient pixels are progressively
    removed and the drop of the target vector (IoU * score) is measured.
    Lower AUC = more faithful explanation (removing what matters collapses the detection).
    The target vector includes IoU, so the curve already captures localization.

    The implementation follows the paper protocol used in this study: the target
    response is IoU x score, the deletion baseline is zero, and the reported AUC
    is normalized by the evaluated deletion range (not by the initial response).
    """
    H, W = saliency.shape
    order = np.argsort(saliency.ravel())[::-1]          # most salient first
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
            work[ys, xs, :] = 0.0                        # baseline = black
            prev_k = k
        s = target_vector_score(
            detector.detect(work.astype(np.uint8)), target_box, target_cls
        )
        scores.append(s)

    scores = np.array(scores, dtype=float)
    s0 = scores[0] if scores[0] > 1e-8 else 1.0
    trapz = getattr(np, "trapezoid", np.trapz)
    auc = float(trapz(scores, fracs) / max_frac)    # range-normalized area
    auc_norm = float(trapz(scores / s0, fracs) / max_frac)  # normalized by the initial score
    return {
        "deletion_auc": auc,
        "deletion_auc_norm": auc_norm,   # diagnostic variant normalized by the initial response
        "score_start": float(scores[0]),
        "score_end": float(scores[-1]),
    }


# ===========================================================================
# Main loop
# ===========================================================================
def run(detector: Detector, domains: list[str], methods: list[str], smoke: int | None):
    CFG.validate()
    CFG.ensure_dirs()

    for domain in domains:
        eval_set = build_eval_set(domain)
        if smoke:
            eval_set = eval_set[:smoke]

        for method in methods:
            explainer = EXPLAINERS[method]
            store = ResultStore(domain, method)
            n_new, n_skip = 0, 0

            for inst in eval_set:
                image_full = get_image(inst)
                tbox_full = tuple(inst["det_box"])
                tcls = CFG.run.person_class_id
                # Crop to ROI (box + context) ONCE per instance. Restores the
                # dynamic range of D-Deletion and speeds up (detector on a small crop).
                roi_img, roi_box, _ = compute_roi(image_full, tbox_full, CFG.xai.roi_margin)

                for seed in CFG.run.seeds:
                    if store.is_done(inst["instance_id"], seed):
                        n_skip += 1
                        continue
                    set_seed(seed)
                    try:
                        sal = explainer.explain(roi_img, roi_box, tcls, detector, seed)
                        dd = d_deletion(roi_img, roi_box, tcls, sal, detector)
                    except NotImplementedError as e:
                        print(f"  [{domain}/{method}] method not available: {e}")
                        break  # skip this method entirely
                    record = {
                        "instance_id": inst["instance_id"],
                        "domain": domain,
                        "method": method,
                        "seed": seed,
                        "tercile": inst["tercile"],
                        "size_bin": inst["size_bin"],
                        "occlusion": inst["occlusion"],
                        "confidence": inst["confidence"],
                        **dd,
                    }
                    store.append(record)
                    n_new += 1

            print(f"[{domain}/{method}] new={n_new}  already_done={n_skip}  -> {store.path.name}")


# ---------------------------------------------------------------------------
# Image bridge: wired together with the loader.
# ---------------------------------------------------------------------------
_LOADER: GTLoader | None = None


def get_image(inst: dict) -> np.ndarray:
    """Rebuilds the image of an eval_index instance using the global loader."""
    if _LOADER is None:
        raise RuntimeError("Loader not initialized. See the __main__ block.")
    ped = GTPed(
        domain=inst["domain"], set_id=inst.get("set_id"), video=inst["video"],
        frame_id=inst["frame_id"], box=tuple(inst["gt_box"]), occlusion=inst["occlusion"],
    )
    return _LOADER.load_frame(ped)  # type: ignore[union-attr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=[d.name for d in CFG.domains])
    ap.add_argument("--methods", nargs="*", default=list(CFG.xai.methods))
    ap.add_argument("--smoke", type=int, default=None,
                    help="Limit to N instances per domain for a quick test.")
    args = ap.parse_args()

    raise SystemExit(
        "Use launch.py so the detector and dataset loader are initialized consistently."
    )


if __name__ == "__main__":
    main()