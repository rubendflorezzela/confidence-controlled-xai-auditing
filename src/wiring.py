"""
wiring.py
=========
Concrete implementations of the interfaces.py protocols for
YOLOv8 + PIE/JAAD. This module concentrates access to the detector and the data.

Frame paths can be adjusted if the local layout differs. Annotations
are read through the official pie_data.py and jaad_data.py parsers.

Dependencies:
    pip install ultralytics opencv-python
    (pie_data.py and jaad_data.py must be importable: copy them next to this
     file or add their folder to PYTHONPATH)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import CFG
from interfaces import Detection, GTPed


# ===========================================================================
# DETECTOR
# ===========================================================================
class Yolov8Detector:
    """
    Wraps a pretrained YOLOv8 (COCO). Returns detections in absolute xyxy.

    The confidence threshold is intentionally low so weak true-positive detections
    remain available for the pre-hoc confidence stratification.
    """

    def __init__(self, weights: str = "yolov8s.pt", device: str | None = None,
                 person_class_id: int | None = None, conf: float = 0.001,
                 imgsz: int = 640):
        from ultralytics import YOLO  # lazy import
        self.model = YOLO(weights)
        self.device = device or CFG.run.device
        self.person_class_id = (CFG.run.person_class_id
                                if person_class_id is None else person_class_id)
        self.conf = conf
        self.imgsz = imgsz
        self._cam_hook = None      # EigenCAM hook (registered once)
        self._cam_act = None       # last captured activation

    def detect(self, image: np.ndarray) -> list[Detection]:
        # interfaces.py defines image as RGB; Ultralytics expects BGR for np.ndarray.
        bgr = np.ascontiguousarray(image[:, :, ::-1])
        res = self.model.predict(bgr, device=self.device, conf=self.conf,
                                 imgsz=self.imgsz, verbose=False)[0]
        return self._parse(res)

    def detect_batch(self, images: list[np.ndarray]) -> list[list["Detection"]]:
        # A single call for the whole batch: key for D-RISE speed.
        bgr_list = [np.ascontiguousarray(im[:, :, ::-1]) for im in images]
        results = self.model.predict(bgr_list, device=self.device, conf=self.conf,
                                     imgsz=self.imgsz, verbose=False)
        return [self._parse(r) for r in results]

    @staticmethod
    def _parse(res) -> list[Detection]:
        out: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, clss):
            out.append(Detection(box=(float(x1), float(y1), float(x2), float(y2)),
                                 score=float(s), cls=int(c)))
        return out

    # ---- EigenCAM (non-perturbative baseline) -----------------------------
    def _ensure_cam_hook(self, layer_index: int) -> None:
        if self._cam_hook is not None:
            return
        seq = self.model.model.model            # nn.Sequential of DetectionModel layers

        def _hook(_m, _in, out):
            self._cam_act = out.detach()

        self._cam_hook = seq[layer_index].register_forward_hook(_hook)

    def eigencam(self, image: np.ndarray, layer_index: int | None = None) -> np.ndarray:
        """
        EigenCAM: projection of a layer's activations onto their first
        principal component. Single forward pass, class-agnostic. Returns a
        saliency map at the native ROI size, to be used with the SAME
        D-Deletion metric as D-RISE (fair comparison).
        """
        import cv2
        li = CFG.xai.cam_layer_index if layer_index is None else layer_index
        self._ensure_cam_hook(li)

        H, W = image.shape[:2]
        sq = cv2.resize(image, (self.imgsz, self.imgsz))      # square, no padding
        bgr = np.ascontiguousarray(sq[:, :, ::-1])
        self._cam_act = None
        _ = self.model.predict(bgr, device=self.device, conf=self.conf,
                               imgsz=self.imgsz, verbose=False)
        if self._cam_act is None:
            return np.zeros((H, W), dtype=np.float32)

        a = self._cam_act[0]                                  # (C, h, w)
        _, hh, ww = a.shape
        A = a.reshape(a.shape[0], -1).permute(1, 0).cpu().numpy()   # (h*w, C)
        A = A - A.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        proj = A @ Vt[0]                                      # first principal component
        if proj.sum() < 0:                                    # fix the (ambiguous) sign
            proj = -proj
        cam = np.maximum(proj.reshape(hh, ww), 0.0)
        mn, mx = float(cam.min()), float(cam.max())
        cam = (cam - mn) / (mx - mn) if mx > mn else np.zeros_like(cam)
        return cv2.resize(cam.astype(np.float32), (W, H))


# ===========================================================================
# LOADER PIE / JAAD
# ===========================================================================
class PieJaadLoader:
    """
    Uses the official parsers (pie_data.PIE, jaad_data.JAAD) for annotations
    and reads frames from extracted images (fast) or from the video (fallback).

    stride: takes 1 frame every 'stride' per pedestrian track, to avoid nearly
            identical frames and the autocorrelation that would break Wilcoxon.
            The value used is reported in the paper.
    """

    _OCC = {0: "none", 1: "partial", 2: "heavy"}

    def __init__(self, pie_root: Path | str | None = None,
                 jaad_root: Path | str | None = None, stride: int = 10):
        self.pie_root = Path(pie_root or CFG.paths.pie_root)
        self.jaad_root = Path(jaad_root or CFG.paths.jaad_root)
        self.stride = max(1, int(stride))
        self._pie = None
        self._jaad = None
        self._pie_db = None
        self._jaad_db = None
        # Cache of a SINGLE open video at a time. With 346 JAAD videos, keeping
        # all of them open exhausts RAM. Released when the video changes.
        self._cur_vpath: str | None = None
        self._cur_cap = None

    # ---- annotations -----------------------------------------------------
    def _occ(self, v) -> str:
        try:
            return self._OCC.get(int(v), "none")
        except Exception:
            return "none"

    def _iter_track(self, domain, set_id, video_id, p):
        frames = p.get("frames", [])
        bboxes = p.get("bbox", [])
        occ = p.get("occlusion", [0] * len(frames))
        for i in range(0, len(frames), self.stride):
            if i >= len(bboxes):
                break
            yield GTPed(
                domain=domain, set_id=set_id, video=video_id,
                frame_id=int(frames[i]),
                box=tuple(float(x) for x in bboxes[i]),
                occlusion=self._occ(occ[i] if i < len(occ) else 0),
            )

    def iter_pedestrians(self, domain_name: str):
        if domain_name == "PIE":
            if self._pie_db is None:
                from pie_data import PIE  # lazy import
                self._pie = PIE(data_path=str(self.pie_root))
                self._pie_db = self._pie.generate_database()
            for set_id, videos in self._pie_db.items():           # adjust if the schema differs
                for video_id, vdata in videos.items():
                    for ped_id, p in vdata.get("ped_annotations", {}).items():
                        yield from self._iter_track("PIE", set_id, video_id, p)

        elif domain_name == "JAAD":
            if self._jaad_db is None:
                from jaad_data import JAAD  # lazy import
                self._jaad = JAAD(data_path=str(self.jaad_root))
                self._jaad_db = self._jaad.generate_database()
            for video_id, vdata in self._jaad_db.items():          # adjust if the schema differs
                for ped_id, p in vdata.get("ped_annotations", {}).items():
                    yield from self._iter_track("JAAD", None, video_id, p)
        else:
            raise ValueError(f"Unknown domain: {domain_name}")

    # ---- frames ----------------------------------------------------------
    def _image_path(self, ped: GTPed) -> Path:
        if ped.domain == "PIE":
            # adjust if frames were extracted with a different folder/name layout
            return (self.pie_root / "images" / (ped.set_id or "") /
                    ped.video / f"{ped.frame_id:05d}.png")
        return self.jaad_root / "images" / ped.video / f"{ped.frame_id:05d}.png"

    def _video_path(self, ped: GTPed) -> Path:
        if ped.domain == "PIE":
            return self.pie_root / "PIE_clips" / (ped.set_id or "") / f"{ped.video}.mp4"
        return self.jaad_root / "JAAD_clips" / f"{ped.video}.mp4"

    def load_frame(self, ped: GTPed) -> np.ndarray:
        import cv2  # lazy import
        img_path = self._image_path(ped)
        if img_path.exists():
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise IOError(f"Could not read {img_path}")
            return np.ascontiguousarray(bgr[:, :, ::-1])  # BGR -> RGB

        # Fallback: read from the video by seeking. A SINGLE video is kept open;
        # when the video changes the previous one is released (avoids exhausting RAM).
        # Phase 0 sorts reads by video, so each one is opened only once.
        vpath = str(self._video_path(ped))
        if vpath != self._cur_vpath:
            if self._cur_cap is not None:
                self._cur_cap.release()
            self._cur_cap = cv2.VideoCapture(vpath)
            self._cur_vpath = vpath
        cap = self._cur_cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, ped.frame_id)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise IOError(f"Could not read frame {ped.frame_id} of {vpath}")
        return np.ascontiguousarray(bgr[:, :, ::-1])

    def close(self) -> None:
        if self._cur_cap is not None:
            self._cur_cap.release()
            self._cur_cap = None
            self._cur_vpath = None


# ===========================================================================
# Factories + self-test
# ===========================================================================
def build_detector() -> Yolov8Detector:
    return Yolov8Detector(weights="yolov8s.pt")


def build_loader() -> PieJaadLoader:
    return PieJaadLoader(stride=10)


if __name__ == "__main__":
    # Minimal test: loads one pedestrian per domain and runs the detector.
    det = build_detector()
    loader = build_loader()
    for dom in ("PIE", "JAAD"):
        it = loader.iter_pedestrians(dom)
        try:
            ped = next(iter(it))
        except StopIteration:
            print(f"[{dom}] no pedestrians found (check paths/parsers)")
            continue
        img = loader.load_frame(ped)
        dets = det.detect(img)
        persons = [d for d in dets if d.cls == det.person_class_id]
        print(f"[{dom}] frame {ped.frame_id} of {ped.video} "
              f"({'set '+ped.set_id if ped.set_id else 'JAAD'}): "
              f"img={img.shape}, detections={len(dets)}, persons={len(persons)}")
    print("Self-test OK.")