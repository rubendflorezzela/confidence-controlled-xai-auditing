"""
interfaces.py
=============
Primitives shared by fit_terciles.py and run_audit.py.
 
No experiment logic lives here: only the types and geometric functions
that both modules need, plus the protocols implemented to connect
the detector and the data loader (PIE/JAAD).
 
The two integration points are implemented in wiring.py.
"""
from __future__ import annotations
 
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable
 
import numpy as np
 
 
# ---------------------------------------------------------------------------
# Data types.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    """A model detection. box in xyxy format (absolute pixels)."""
    box: tuple[float, float, float, float]
    score: float
    cls: int
 
 
@dataclass(frozen=True)
class GTPed:
    """An annotated pedestrian (ground truth)."""
    domain: str                 # "PIE" | "JAAD"
    set_id: str | None          # "set01".. in PIE; None in JAAD
    video: str
    frame_id: int
    box: tuple[float, float, float, float]   # xyxy
    occlusion: str              # "none" | "partial" | "heavy"
 
    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])
 
    @property
    def instance_id(self) -> str:
        s = self.set_id or "NA"
        return f"{self.domain}:{s}:{self.video}:{self.frame_id}:{int(self.box[0])}_{int(self.box[1])}"
 
 
# ---------------------------------------------------------------------------
# Protocols implemented once for the local environment (see wiring.py).
# ---------------------------------------------------------------------------
@runtime_checkable
class Detector(Protocol):
    """Wraps the pretrained detector (YOLOv8s in this study)."""
 
    def detect(self, image: np.ndarray) -> list[Detection]:
        """RGB HxWx3 uint8 image -> list of detections (all classes or person only)."""
        ...  # implemented in wiring.py (Ultralytics YOLO)
 
    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Batched version: a single detector call for several images.
        This is what makes D-RISE tractable (thousands of forward passes per instance)."""
        ...  # implemented in wiring.py (Ultralytics accepts a list of images)
 
 
@runtime_checkable
class GTLoader(Protocol):
    """Wraps access to PIE/JAAD annotations and frames."""
 
    def iter_pedestrians(self, domain_name: str) -> Iterable[GTPed]:
        """Iterates GT pedestrians of a domain. For PIE, set_id is populated."""
        ...  # implemented in wiring.py (official pie_data / jaad_data parsers)
 
    def load_frame(self, ped: GTPed) -> np.ndarray:
        """Returns the RGB HxWx3 uint8 frame containing the pedestrian."""
        ...  # implemented in wiring.py (frame extraction from the corresponding video)
 
 
# ---------------------------------------------------------------------------
# Geometry and target vector (D-RISE style), used by the fit and the runner.
# ---------------------------------------------------------------------------
def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0
 
 
def match_tp(
    dets: list[Detection],
    gt_box: tuple[float, float, float, float],
    person_class_id: int,
    iou_thr: float,
) -> Detection | None:
    """Returns the 'person' detection with the highest IoU >= threshold w.r.t. the GT, or None."""
    best, best_iou = None, iou_thr
    for d in dets:
        if d.cls != person_class_id:
            continue
        j = iou(d.box, gt_box)
        if j >= best_iou:
            best, best_iou = d, j
    return best
 
 
def target_vector_score(
    dets: list[Detection],
    target_box: tuple[float, float, float, float],
    target_cls: int,
) -> float:
    """
    Target-vector score in the D-RISE style: max over detections of
    (IoU with the target box) * (class score). Couples faithfulness and
    localization into a single scalar, which is what D-Deletion needs.
    Returns 0 if the target disappears under perturbation.
    """
    best = 0.0
    for d in dets:
        if d.cls != target_cls:
            continue
        val = iou(d.box, target_box) * float(d.score)
        if val > best:
            best = val
    return best
 
 
def compute_roi(image: np.ndarray, box: tuple[float, float, float, float], margin: float):
    """
    Crops a region of interest around the target box, expanded by
    'margin' times the box size on each side (margin=1.0 => ROI ~3x the box).
 
    Rationale: without cropping, D-Deletion is measured against a 1080p frame
    where the pedestrian covers ~1-2%, so the first step already deletes it and
    the metric saturates at a constant value. With the ROI, the deletion budget is
    relative to the object and the curve recovers dynamic range. The detector
    also processes a small crop, which is much faster.
 
    Returns (roi_img, roi_box, (ox, oy)); roi_box is in crop coordinates.
    """
    H, W = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1), (y2 - y1)
    rx1 = int(max(0, np.floor(x1 - margin * bw)))
    ry1 = int(max(0, np.floor(y1 - margin * bh)))
    rx2 = int(min(W, np.ceil(x2 + margin * bw)))
    ry2 = int(min(H, np.ceil(y2 + margin * bh)))
    roi = np.ascontiguousarray(image[ry1:ry2, rx1:rx2])
    roi_box = (x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1)
    return roi, roi_box, (rx1, ry1)
