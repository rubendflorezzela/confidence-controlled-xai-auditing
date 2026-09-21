"""
launch.py
=========
Phase-based entry point. One command per step, in order.

    python launch.py selftest     # 1) checks that detector and loader work
    python launch.py fit          # 2) PHASE 0: freezes terciles + eval_index (run once)
    python launch.py smoke        # 3) 3 instances/domain, all methods (validates the pipeline)
    python launch.py audit        # 4) full run (all instances/methods/seeds)

Resumable: if 'audit' is interrupted, relaunch it and it continues where it stopped.
"""
from __future__ import annotations

import argparse

from config import CFG
from wiring import build_detector, build_loader

import fit_terciles
import run_audit


def cmd_selftest():
    det, loader = build_detector(), build_loader()
    for dom in (d.name for d in CFG.domains):
        try:
            ped = next(iter(loader.iter_pedestrians(dom)))
        except StopIteration:
            print(f"[{dom}] no pedestrians found (check paths/parsers)")
            continue
        img = loader.load_frame(ped)
        dets = det.detect(img)
        persons = [d for d in dets if d.cls == det.person_class_id]
        loc = f"set {ped.set_id}" if ped.set_id else "JAAD"
        print(f"[{dom}] frame {ped.frame_id} of {ped.video} ({loc}): "
              f"img={img.shape}, detections={len(dets)}, persons={len(persons)}")
    print("Self-test OK.")


def cmd_fit():
    det, loader = build_detector(), build_loader()
    fit_terciles.main(det, loader)


def cmd_smoke():
    det, loader = build_detector(), build_loader()
    run_audit._LOADER = loader
    run_audit.run(det, [d.name for d in CFG.domains], list(CFG.xai.methods), smoke=3)


def cmd_audit():
    det, loader = build_detector(), build_loader()
    run_audit._LOADER = loader
    run_audit.run(det, [d.name for d in CFG.domains], list(CFG.xai.methods), smoke=None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["selftest", "fit", "smoke", "audit"])
    args = ap.parse_args()
    {"selftest": cmd_selftest, "fit": cmd_fit,
     "smoke": cmd_smoke, "audit": cmd_audit}[args.phase]()
