"""Neural building blocks: detector, landmarker, recognizer, swapper, enhancer, maskers."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import geometry as geo
from .landmarks import five_from_68

log = logging.getLogger("mimiq.analysis")


def enhance_contrast(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """CLAHE on lightness only — reveals faces in shadow without shifting colours."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------------------
# Detection (RetinaFace / SCRFD share the same 9-output layout)
# --------------------------------------------------------------------------------------

@dataclass
class Detection:
    box: np.ndarray            # x1, y1, x2, y2
    score: float
    kps: np.ndarray            # (5, 2)

    @property
    def size(self) -> float:
        return float(max(self.box[2] - self.box[0], self.box[3] - self.box[1]))

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


class FaceDetector:
    STRIDES = (8, 16, 32)

    def __init__(self, session, swap_rb: bool = True):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.swap_rb = swap_rb
        self._anchors: Dict[Tuple[int, int], List[np.ndarray]] = {}

    def _anchor_centers(self, h: int, w: int) -> List[np.ndarray]:
        key = (h, w)
        if key not in self._anchors:
            out = []
            for s in self.STRIDES:
                gy, gx = np.mgrid[: h // s, : w // s]
                centers = np.stack([gx, gy], axis=-1).reshape(-1, 2).astype(np.float32) * s
                out.append(np.repeat(centers, 2, axis=0))
            self._anchors[key] = out
        return self._anchors[key]

    def detect(self, image: np.ndarray, size: int = 640, threshold: float = 0.5, nms: float = 0.4,
               max_faces: int = 16) -> List[Detection]:
        h, w = image.shape[:2]
        scale = min(size / h, size / w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        img = image if (nh, nw) == (h, w) else cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if self.swap_rb:
            img = img[..., ::-1]
        blob = np.full((1, 3, size, size), -127.5 / 128.0, np.float32)
        blob[0, :, :nh, :nw] = ((img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)
        outs = self.session.run(None, {self.input_name: blob})
        anchors = self._anchor_centers(size, size)
        boxes, scores, kpss = [], [], []
        for i, stride in enumerate(self.STRIDES):
            sc = outs[i].reshape(-1)
            keep = np.where(sc >= threshold)[0]
            if keep.size == 0:
                continue
            a = anchors[i][keep]
            bd = outs[i + 3].reshape(-1, 4)[keep] * stride
            kd = outs[i + 6].reshape(-1, 10)[keep] * stride
            b = np.stack([a[:, 0] - bd[:, 0], a[:, 1] - bd[:, 1], a[:, 0] + bd[:, 2], a[:, 1] + bd[:, 3]], -1)
            k = np.stack([a[:, None, 0] + kd[:, 0::2], a[:, None, 1] + kd[:, 1::2]], -1)
            boxes.append(b)
            scores.append(sc[keep])
            kpss.append(k)
        if not boxes:
            return []
        boxes_a = np.concatenate(boxes) / scale
        scores_a = np.concatenate(scores)
        kps_a = np.concatenate(kpss) / scale
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes_a]
        idx = cv2.dnn.NMSBoxes(xywh, scores_a.astype(float).tolist(), threshold, nms)
        idx = np.array(idx).reshape(-1)[:max_faces]
        dets = [Detection(boxes_a[j].astype(np.float64), float(scores_a[j]), kps_a[j].astype(np.float64)) for j in idx]
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets

    def detect_all(self, image: np.ndarray, size: int = 640, threshold: float = 0.5,
                   multi: bool = False) -> List[Detection]:
        """Full-frame search that also finds faces right in front of the camera: at 640 px the network's
        largest anchors are smaller than a face that fills the frame, so a coarse 320 px pass covers them."""
        dets = self.detect(image, size, threshold)
        if size > 320 and (not dets or multi):
            short = min(image.shape[:2])
            big = [d for d in self.detect(image, 320, threshold) if d.size > 0.35 * short]
            for d in big:
                if all(geo.iou(d.box, o.box) < 0.3 for o in dets):
                    dets.append(d)
            dets.sort(key=lambda d: d.score, reverse=True)
        return dets

    def detect_roi(self, frame: np.ndarray, center, side: float, angle: float, size: int = 320,
                   threshold: float = 0.35, retry: bool = True) -> List[Detection]:
        """Detect inside a rotated square ROI (keeps tilted heads upright for the detector)."""
        m = geo.similarity_roi(center, side, angle, size)
        crop = cv2.warpAffine(frame, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        dets = self.detect(crop, size=size, threshold=threshold)
        if not dets and retry:
            # second chance with local contrast: shadow of a cap visor, backlight, dark room
            dets = self.detect(enhance_contrast(crop), size=size, threshold=threshold)
        inv = geo.invert(m)
        out = []
        for d in dets:
            kps = geo.transform_points(d.kps, inv)
            corners = geo.transform_points(np.array([[d.box[0], d.box[1]], [d.box[2], d.box[1]],
                                                     [d.box[2], d.box[3]], [d.box[0], d.box[3]]]), inv)
            out.append(Detection(geo.box_from_points(corners), d.score, kps))
        return out


# --------------------------------------------------------------------------------------
# 68-point landmarks (2DFAN4) → precise 5 points
# --------------------------------------------------------------------------------------

class Landmarker68:
    def __init__(self, session, swap_rb: bool = False):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.swap_rb = swap_rb

    def detect(self, frame: np.ndarray, box: np.ndarray, angle: float = 0.0) -> Tuple[np.ndarray, float]:
        pts, score, _ = self.detect_full(frame, box, angle)
        return pts, score

    def detect_full(self, frame: np.ndarray, box: np.ndarray,
                    angle: float = 0.0) -> Tuple[np.ndarray, float, np.ndarray]:
        """68 points, overall score and per-point heat-map confidence."""
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        side = max(box[2] - box[0], box[3] - box[1]) * 256.0 / 195.0
        m = geo.similarity_roi((cx, cy), side, angle, 256)
        crop = cv2.warpAffine(frame, m, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if float(crop.mean()) < 40:  # boost contrast for dark scenes
            crop = enhance_contrast(crop)
        x = crop[..., ::-1] if self.swap_rb else crop
        x = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        lms, heat = self.session.run(None, {self.input_name: x})
        pts = lms[0, :, :2] / 64.0 * 256.0
        pts = geo.transform_points(pts, geo.invert(m))
        conf = np.amax(heat[0].reshape(heat.shape[1], -1), axis=1).astype(np.float64)
        score = float(np.interp(float(np.mean(conf)), [0, 0.9], [0, 1]))
        return pts, score, conf

    @staticmethod
    def to_five(lm68: np.ndarray) -> np.ndarray:
        return five_from_68(lm68)


# --------------------------------------------------------------------------------------
# Identity embedding (ArcFace)
# --------------------------------------------------------------------------------------

class FaceRecognizer:
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name

    @staticmethod
    def align(frame: np.ndarray, lm5: np.ndarray) -> np.ndarray:
        return geo.warp_face(frame, geo.face_matrix(lm5, "arcface_112_v2", 112), 112)

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        x = (crop[..., ::-1].astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        return self.session.run(None, {self.input_name: x})[0].reshape(-1).astype(np.float64)

    def embed(self, frame: np.ndarray, lm5: np.ndarray) -> np.ndarray:
        return self.embed_crop(self.align(frame, lm5))


# --------------------------------------------------------------------------------------
# Face swapper
# --------------------------------------------------------------------------------------

SWAPPERS = {
    "inswapper_128_fp16": dict(type="inswapper", template="arcface_128", size=128, mean=0.0, std=1.0),
    "inswapper_128": dict(type="inswapper", template="arcface_128", size=128, mean=0.0, std=1.0),
    "hyperswap_1a_256": dict(type="hyperswap", template="arcface_128", size=256, mean=0.5, std=0.5),
    "hyperswap_1b_256": dict(type="hyperswap", template="arcface_128", size=256, mean=0.5, std=0.5),
    "hyperswap_1c_256": dict(type="hyperswap", template="arcface_128", size=256, mean=0.5, std=0.5),
}


def load_inswapper_emap(model_path: Path, cache_dir: Path) -> np.ndarray:
    cache = cache_dir / (model_path.stem + "_emap.npy")
    if cache.exists():
        try:
            return np.load(cache)
        except Exception:
            cache.unlink(missing_ok=True)
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(model_path))
    emap = numpy_helper.to_array(model.graph.initializer[-1]).astype(np.float32)
    del model
    np.save(cache, emap)
    return emap


def implode_tiles(crop: np.ndarray, n: int, size: int) -> np.ndarray:
    return crop.reshape(size, n, size, n, 3).transpose(1, 3, 0, 2, 4).reshape(n * n, size, size, 3)


def explode_tiles(tiles: np.ndarray, n: int, size: int) -> np.ndarray:
    return tiles.reshape(n, n, size, size, 3).transpose(2, 0, 3, 1, 4).reshape(size * n, size * n, 3)


class FaceSwapper:
    def __init__(self, key: str, session, model_path: Path, cache_dir: Path):
        self.key = key
        self.spec = SWAPPERS[key]
        self.session = session
        self.size = self.spec["size"]
        self.template = self.spec["template"]
        names = [i.name for i in session.get_inputs()]
        self.src_name = "source" if "source" in names else names[0]
        self.tgt_name = "target" if "target" in names else names[1]
        self.emap = load_inswapper_emap(model_path, cache_dir) if self.spec["type"] == "inswapper" else None

    def latent(self, identity: np.ndarray, target: Optional[np.ndarray] = None, strength: float = 0.0) -> np.ndarray:
        """Conditioning vector for the swapper.

        With `strength` > 0 the identity is extrapolated away from the face that is in front of the camera
        (e' = e + k·(e − t)). The swapper always keeps some of the target's identity; pushing the latent the
        other way cancels that leak, so the result looks more like the photo and less like the camera face.
        Measured with an independent recogniser (SFace), k = 0.6 removes nearly all resemblance to the camera
        face while keeping the resemblance to the photo."""
        e = np.asarray(identity, np.float64).reshape(-1)
        e = e / max(np.linalg.norm(e), 1e-9)
        if target is not None and strength > 1e-3:
            t = np.asarray(target, np.float64).reshape(-1)
            t = t / max(np.linalg.norm(t), 1e-9)
            e = e + float(strength) * (e - t)
            e = e / max(np.linalg.norm(e), 1e-9)
        if self.spec["type"] == "inswapper":
            v = e.astype(np.float32).reshape(1, -1) @ self.emap   # == (raw @ emap) / ‖raw‖
        else:
            v = e.reshape(1, -1)
        return np.ascontiguousarray(v, dtype=np.float32)

    def source_vector(self, embedding: np.ndarray, embedding_norm: np.ndarray) -> np.ndarray:
        return self.latent(embedding if self.spec["type"] == "inswapper" else embedding_norm)

    def _run(self, crop: np.ndarray, source: np.ndarray) -> np.ndarray:
        b = crop.shape[0]
        n = max(1, b // self.size)
        tiles = implode_tiles(crop, n, self.size) if n > 1 else crop[None]
        mean, std = self.spec["mean"], self.spec["std"]
        outs = []
        for t in tiles:
            x = ((t[..., ::-1].astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]
            y = self.session.run(None, {self.src_name: source, self.tgt_name: x})[0][0].transpose(1, 2, 0)
            if self.spec["type"] != "inswapper":
                y = y * std + mean
            outs.append(np.clip(y, 0.0, 1.0)[..., ::-1] * 255.0)
        out = np.stack(outs)
        return explode_tiles(out, n, self.size) if n > 1 else out[0]

    def swap(self, crop: np.ndarray, source: np.ndarray, passes: int = 1) -> np.ndarray:
        """crop: (B, B, 3) uint8 BGR where B = size * n (pixel boost). Returns float32 BGR 0..255.

        `passes` > 1 feeds the result back in: each pass transfers a little more of the identity
        (skin texture, stubble, brows) at the cost of one more swapper run."""
        out = self._run(crop, source)
        for _ in range(max(1, int(passes)) - 1):
            out = self._run(np.clip(out + 0.5, 0, 255).astype(np.uint8), source)
        return out


# --------------------------------------------------------------------------------------
# Face enhancer (restoration)
# --------------------------------------------------------------------------------------

ENHANCERS = {
    "gpen_bfr_256": dict(template="arcface_128", size=256),
    "gpen_bfr_512": dict(template="ffhq_512", size=512),
    "gfpgan_1.4": dict(template="ffhq_512", size=512),
    "codeformer": dict(template="ffhq_512", size=512),
    "restoreformer_plus_plus": dict(template="ffhq_512", size=512),
}


class FaceEnhancer:
    def __init__(self, key: str, session):
        self.key = key
        self.session = session
        self.size = ENHANCERS[key]["size"]
        self.template = ENHANCERS[key]["template"]
        names = [i.name for i in session.get_inputs()]
        self.input_name = "input" if "input" in names else names[0]
        self.has_weight = "weight" in names

    def enhance(self, crop: np.ndarray, fidelity: float = 0.8) -> np.ndarray:
        x = ((crop[..., ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        feed = {self.input_name: x}
        if self.has_weight:
            feed["weight"] = np.array([fidelity], dtype=np.float64)
        y = self.session.run(None, feed)[0][0]
        y = (np.clip(y, -1.0, 1.0) + 1.0) * 0.5
        return (y.transpose(1, 2, 0)[..., ::-1] * 255.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _box_mask(size: int, blur: float, top: int, right: int, bottom: int, left: int) -> np.ndarray:
    blur_amount = int(size * 0.5 * blur)
    blur_area = max(blur_amount // 2, 1)
    m = np.ones((size, size), np.float32)
    m[: max(blur_area, int(size * top / 100)), :] = 0
    m[-max(blur_area, int(size * bottom / 100)):, :] = 0
    m[:, : max(blur_area, int(size * left / 100))] = 0
    m[:, -max(blur_area, int(size * right / 100)):] = 0
    if blur_amount > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur_amount * 0.25)
    m.setflags(write=False)
    return m


def box_mask(size: int, blur: float = 0.3, padding: Sequence[int] = (0, 0, 0, 0)) -> np.ndarray:
    t, r, b, l = (int(v) for v in padding)
    return _box_mask(int(size), round(float(blur), 3), t, r, b, l)


class Occluder:
    """XSeg occlusion mask: 1 = visible face skin, 0 = hands / hair / objects in front."""

    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def mask(self, crop: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA if crop.shape[0] > 256 else cv2.INTER_LINEAR)
        x = (x.astype(np.float32) / 255.0)[None]
        y = self.session.run(None, {self.input_name: x})[0][0]
        y = np.clip(y.reshape(256, 256), 0, 1).astype(np.float32)
        y = cv2.resize(y, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        sigma = 5.0 * crop.shape[0] / 256.0
        return (np.clip(cv2.GaussianBlur(y, (0, 0), sigma), 0.5, 1.0) - 0.5) * 2.0


REGION_IDS = {"skin": 1, "left-eyebrow": 2, "right-eyebrow": 3, "left-eye": 4, "right-eye": 5, "glasses": 6,
              "nose": 10, "mouth": 11, "upper-lip": 12, "lower-lip": 13}
DEFAULT_REGIONS = ("skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "nose", "mouth",
                   "upper-lip", "lower-lip")


class FaceParser:
    """BiSeNet face parsing: keeps only selected facial regions (excludes hair, ears, neck...)."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def mask(self, crop: np.ndarray, regions: Sequence[str] = DEFAULT_REGIONS) -> np.ndarray:
        x = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
        x = ((x[..., ::-1].astype(np.float32) / 255.0 - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        y = self.session.run(None, {self.input_name: x})[0][0]
        ids = [REGION_IDS[r] for r in regions if r in REGION_IDS]
        m = np.isin(y.argmax(0), ids).astype(np.float32)
        m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        sigma = 5.0 * crop.shape[0] / 256.0
        return (np.clip(cv2.GaussianBlur(m, (0, 0), sigma), 0.5, 1.0) - 0.5) * 2.0


# --------------------------------------------------------------------------------------
# Colour harmonisation
# --------------------------------------------------------------------------------------

def lab_stats(img_bgr: np.ndarray, mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    small = cv2.resize(img_bgr.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA) / 255.0
    msk = cv2.resize(mask.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA) > 0.6
    if msk.sum() < 40:
        return None
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[msk]
    return lab.mean(0), lab.std(0) + 1e-3


def apply_color_transfer(img_bgr: np.ndarray, src_stats, ref_stats, strength: float) -> np.ndarray:
    """Shift `img_bgr` (float 0..255) so its Lab statistics match `ref_stats` (Reinhard)."""
    (mu_s, sd_s), (mu_r, sd_r) = src_stats, ref_stats
    ratio = np.clip(sd_r / sd_s, [0.85, 0.7, 0.7], [1.15, 1.4, 1.4])
    lab = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab = (lab - mu_s) * ratio + mu_r
    out = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR) * 255.0
    return img_bgr + (out - img_bgr) * float(np.clip(strength, 0, 1))
