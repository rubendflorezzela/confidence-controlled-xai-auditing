"""
config.py
=========
Experimental configuration for the confidence-controlled XAI audit of pedestrian
detection under domain shift (PIE <-> JAAD).
 
Paper: "Confidence-Controlled XAI Auditing for Pedestrian Detection under Domain Shift".
 
Modular architecture:  config.py -> fit_terciles.py -> run_audit.py -> (separate aggregation)
 
All reproducibility-critical parameters live here. There are no "magic"
numbers scattered through the code: if something changes, it changes in this file.
"""
from __future__ import annotations
 
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths. The source code lives in src/ and the project root is its parent.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    code_dir: Path = Path(__file__).resolve().parent

    @property
    def base_dir(self) -> Path:
        return self.code_dir.parent

    @property
    def pie_root(self) -> Path:
        return self.base_dir / "data" / "PIE"

    @property
    def jaad_root(self) -> Path:
        return self.base_dir / "data" / "JAAD"

    @property
    def out_root(self) -> Path:
        return self.base_dir / "results"

    @property
    def terciles_file(self) -> Path:
        return self.out_root / "terciles.json"

    @property
    def eval_index_dir(self) -> Path:
        return self.out_root / "eval_index"

    @property
    def records_dir(self) -> Path:
        return self.out_root / "records"

# ---------------------------------------------------------------------------
# Domains. Both are used in full when available locally.
# ---------------------------------------------------------------------------
PIE_AVAILABLE_SETS = None
 
 
@dataclass(frozen=True)
class DomainCfg:
    name: str
    root: Path
    # None => use the whole domain (JAAD case). Tuple => restrict to those sets (PIE case).
    available_sets: tuple[str, ...] | None = None
 
 
# ---------------------------------------------------------------------------
# Evaluation-set construction (true detection = TP).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingCfg:
    # Cap SHARED by both domains so that N is comparable in the cross-domain analysis.
    # Common cap across domains so the audited sample size is comparable.
    max_instances_per_domain: int = 300
    # GT pedestrians shorter than this are discarded (unreliable annotation, not auditable).
    min_gt_height_px: int = 30
    # Minimum IoU between detection and GT to count as a true detection of the pedestrian.
    iou_tp_threshold: float = 0.5
    # Stratified sampling balanced across terciles (same n per tercile within the cap).
    balance_across_terciles: bool = True
    # Maximum number of instances that PHASE 0 scans with the detector per domain.
    # Representative subsample (fixed seed) to avoid running the detector over
    # tens of thousands of frames. A few thousand suffice for stable terciles + pool.
    fit_max_scan: int = 6000
    # FIXED seed used to build the evaluation set.
    # NOTE: distinct from the mask seeds (RunCfg.seeds). Do not mix them.
    sampling_seed: int = 20260703
 
 
# ---------------------------------------------------------------------------
# Stratification.
#   Primary   -> confidence terciles.
#   Secondary -> size and occlusion (robustness analysis only, does NOT define the bin).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TercileCfg:
    # PRE-HOC terciles over the confidence the detector assigns to the TRUE
    # pedestrian detection. Boundaries are percentiles fixed in advance,
    # computed per domain in fit_terciles.py BEFORE any explanation is produced,
    # and frozen in terciles.json. This prevents choosing bins after seeing
    # the results.
    quantiles: tuple[float, float] = (1 / 3, 2 / 3)
    # low_conf denotes the lowest-confidence tercile.
    labels: tuple[str, str, str] = ("low_conf", "mid_conf", "high_conf")
 
 
@dataclass(frozen=True)
class RobustnessCfg:
    # Box-height cuts (px) -> small / medium / large.
    size_bins_px: tuple[int, int] = (50, 100)
    # Occlusion levels from the dataset annotation (PIE/JAAD provide an occlusion tag).
    occlusion_levels: tuple[str, ...] = ("none", "partial", "heavy")
 
 
# ---------------------------------------------------------------------------
# Explanation methods and perturbation parameters.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class XAICfg:
    # d_rise    -> reference implementation in run_audit.py (perturbation-based)
    # eigencam  -> non-perturbative CAM baseline, implemented in wiring.py
    methods: tuple[str, ...] = ("d_rise", "eigencam")
    n_masks: int = 2000        # cheap with ROI + batching; 2000 lowers per-instance variance.
    mask_grid: int = 8         # base mask grid (mask_grid x mask_grid).
    mask_prob: float = 0.5     # probability that a cell is active.
    deletion_steps: int = 25   # number of steps of the D-Deletion curve.
    deletion_max_frac: float = 0.5  # maximum fraction of deleted pixels along the curve.
    # ROI crop around the target box: expands the box by 'roi_margin' times its
    # size on each side (1.0 => ROI ~3x the box). Makes deletion relative to the
    # object rather than to the full frame (avoids the 1/38 saturation).
    roi_margin: float = 1.0
    # YOLOv8 layer from which EigenCAM takes activations (-2 = last neck layer,
    # before the Detect head). Change only if the model differs.
    cam_layer_index: int = -2
 
 
# ---------------------------------------------------------------------------
# Execution (multi-seed, hardware, persistence).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunCfg:
    # Multi-seed evaluation over the stochastic D-RISE masks.
    # Five seeds are used to characterize D-RISE mask stochasticity per instance.
    # instances per domain carry the statistics (Wilcoxon + CIs).
    seeds: tuple[int, ...] = tuple(range(5))
    device: str = "cuda:0"          # RTX 3060. Use "cpu" if VRAM is insufficient.
    mask_batch_size: int = 32       # masked images per detector call.
    person_class_id: int = 0        # COCO "person" class id for YOLOv8.
    save_every: int = 1             # persist after each (instance, method, seed).
 
 
# ---------------------------------------------------------------------------
# Experiment (assembles everything). CFG is the singleton imported by the other modules.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Experiment:
    paths: Paths = field(default_factory=Paths)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    terciles: TercileCfg = field(default_factory=TercileCfg)
    robustness: RobustnessCfg = field(default_factory=RobustnessCfg)
    xai: XAICfg = field(default_factory=XAICfg)
    run: RunCfg = field(default_factory=RunCfg)
 
    @property
    def domains(self) -> tuple[DomainCfg, ...]:
        return (
            DomainCfg(name="PIE", root=self.paths.pie_root, available_sets=None),
            DomainCfg(name="JAAD", root=self.paths.jaad_root, available_sets=None),
        )
 
    def ensure_dirs(self) -> None:
        for d in (self.paths.out_root, self.paths.eval_index_dir, self.paths.records_dir):
            d.mkdir(parents=True, exist_ok=True)
 
    def validate(self) -> None:
        assert 0.0 < self.sampling.iou_tp_threshold < 1.0
        assert self.terciles.quantiles[0] < self.terciles.quantiles[1]
        assert len(self.terciles.labels) == 3
        assert self.xai.n_masks > 0 and self.xai.mask_grid > 0
        assert len(self.run.seeds) >= 2, "Multi-seed evaluation needs >= 2 seeds."
        assert set(self.xai.methods), "Define at least one XAI method."
 
 
# Configuration singleton.
CFG = Experiment()
 
 
if __name__ == "__main__":
    CFG.validate()
    CFG.ensure_dirs()
    print("Config OK.")
    print("  Available PIE sets   :", PIE_AVAILABLE_SETS)
    print("  Cap per domain       :", CFG.sampling.max_instances_per_domain)
    print("  XAI methods          :", CFG.xai.methods)
    print("  Seeds (masks)        :", CFG.run.seeds)
    print("  Outputs in           :", CFG.paths.out_root)