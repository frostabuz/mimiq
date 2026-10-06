"""Source identities ("faces") built from one or more photos and stored on disk."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import geometry as geo
from .analysis import FaceDetector, FaceRecognizer, Landmarker68
from ..imaging import decode_file, write_image


class FaceNotFound(Exception):
    pass


@dataclass
class PhotoReport:
    ok: bool
    message: str
    quality: float = 0.0
    similarity: float = 1.0


@dataclass
class Identity:
    id: str
    name: str
    embedding: np.ndarray            # mean raw ArcFace embedding
    embedding_norm: np.ndarray       # mean of normalised embeddings, re-normalised
    thumbnail: np.ndarray            # 256×256 BGR aligned portrait
    photos: int = 1
    quality: float = 0.0
    created: float = field(default_factory=time.time)
    samples: Optional[np.ndarray] = None  # (n, 512) raw embeddings for incremental updates

    # ------------------------------------------------------------------
    def save(self, root: Path) -> Path:
        d = root / self.id
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "identity.npz", embedding=self.embedding, embedding_norm=self.embedding_norm,
                 samples=self.samples if self.samples is not None else self.embedding[None])
        write_image(d / "thumb.png", self.thumbnail)
        (d / "meta.json").write_text(json.dumps({
            "id": self.id, "name": self.name, "photos": self.photos, "quality": round(self.quality, 3),
            "created": self.created}, ensure_ascii=False, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, d: Path) -> "Identity":
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        data = np.load(d / "identity.npz")
        thumb = decode_file(d / "thumb.png")
        if thumb is None:
            thumb = np.zeros((256, 256, 3), np.uint8)
        return cls(meta["id"], meta.get("name", "Лицо"), data["embedding"], data["embedding_norm"], thumb,
                   int(meta.get("photos", 1)), float(meta.get("quality", 0)), float(meta.get("created", 0)),
                   data["samples"] if "samples" in data.files else None)


class IdentityBuilder:
    """Extracts the dominant face from photos and averages identities for a robust profile."""

    def __init__(self, detector: FaceDetector, recognizer: FaceRecognizer, landmarker: Optional[Landmarker68]):
        self.detector, self.recognizer, self.landmarker = detector, recognizer, landmarker

    def analyse(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Return (embedding, lm5, quality 0..1, image used) for the largest face in `image`."""
        h, w = image.shape[:2]
        if max(h, w) > 2400:  # keep huge photos manageable
            f = 2400 / max(h, w)
            image = cv2.resize(image, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        dets = self.detector.detect(image, 640, 0.45)
        if not dets:
            # second chance on a padded image (faces cropped tight against the border)
            pad = int(max(image.shape[:2]) * 0.25)
            padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            dets = self.detector.detect(padded, 640, 0.4)
            if not dets:
                raise FaceNotFound("Лицо на фото не найдено")
            image = padded
        d = max(dets, key=lambda x: x.size * (0.5 + x.score))
        lm5 = d.kps
        if self.landmarker is not None:
            try:
                lm68, score = self.landmarker.detect(image, d.box, geo.roll_degrees(d.kps))
                if score > 0.5:
                    lm5 = Landmarker68.to_five(lm68)
            except Exception:
                pass
        emb = self.recognizer.embed(image, lm5)
        # quality: resolution, detector confidence, frontalness
        eye_mid = (lm5[0] + lm5[1]) / 2.0
        iod = np.linalg.norm(lm5[1] - lm5[0])
        yaw = abs(lm5[2][0] - eye_mid[0]) / max(iod, 1.0)  # 0 = frontal
        res_q = float(np.clip(d.size / 260.0, 0, 1))
        front_q = float(np.clip(1.0 - yaw * 2.2, 0, 1))
        quality = float(np.clip(0.45 * res_q + 0.25 * d.score + 0.30 * front_q, 0, 1))
        return emb, lm5, quality, image

    def thumbnail(self, image: np.ndarray, lm5: np.ndarray) -> np.ndarray:
        m = geo.face_matrix(lm5, "ffhq_512", 256)
        # zoom out a little so the whole head is visible
        c = np.array([128.0, 128.0])
        z = np.array([[0.82, 0, 0], [0, 0.82, 0]], np.float64)
        z[:, 2] = c - 0.82 * c
        return geo.warp_face(image, geo.compose(z, m), 256)


def combine(embeddings: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.stack(embeddings)
    mean_raw = arr.mean(0)
    normed = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    mean_norm = normed.mean(0)
    return mean_raw, mean_norm / np.linalg.norm(mean_norm)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class IdentityLibrary:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> List[Identity]:
        out = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            if (d / "meta.json").exists():
                try:
                    out.append(Identity.load(d))
                except Exception:
                    continue
        out.sort(key=lambda i: i.created)
        return out

    def get(self, ident_id: str) -> Optional[Identity]:
        d = self.root / ident_id
        if (d / "meta.json").exists():
            try:
                return Identity.load(d)
            except Exception:
                return None
        return None

    def create(self, builder: IdentityBuilder, images: Sequence[Tuple[str, np.ndarray]], name: str,
               on_report: Optional[Callable[[str, PhotoReport], None]] = None) -> Identity:
        return self._build(builder, images, name, None, on_report)

    def add_photos(self, builder: IdentityBuilder, ident: Identity, images: Sequence[Tuple[str, np.ndarray]],
                   on_report=None) -> Identity:
        return self._build(builder, images, ident.name, ident, on_report)

    def _build(self, builder, images, name, base: Optional[Identity], on_report) -> Identity:
        embs: List[np.ndarray] = [] if base is None or base.samples is None else list(base.samples)
        best = None
        qualities = []
        for label, img in images:
            try:
                emb, lm5, q, used = builder.analyse(img)
            except FaceNotFound as exc:
                if on_report:
                    on_report(label, PhotoReport(False, str(exc)))
                continue
            sim = cosine(emb, combine(embs)[0]) if embs else 1.0
            if embs and sim < 0.25:
                if on_report:
                    on_report(label, PhotoReport(False, "Похоже, на фото другой человек — пропущено", q, sim))
                continue
            embs.append(emb)
            qualities.append(q)
            if best is None or q > best[0]:
                best = (q, used, lm5)
            if on_report:
                on_report(label, PhotoReport(True, "Готово", q, sim))
        if not embs:
            raise FaceNotFound("Ни на одном фото не удалось найти лицо")
        raw, norm = combine(embs)
        if base is not None:
            thumb = base.thumbnail if best is None or best[0] <= base.quality else builder.thumbnail(best[1], best[2])
            quality = max(base.quality, max(qualities) if qualities else 0)
            ident = Identity(base.id, name, raw, norm, thumb, len(embs), quality, base.created, np.stack(embs))
        else:
            ident = Identity(uuid.uuid4().hex[:12], name, raw, norm, builder.thumbnail(best[1], best[2]),
                             len(embs), float(np.mean(qualities)), time.time(), np.stack(embs))
        ident.save(self.root)
        return ident

    def rename(self, ident: Identity, name: str) -> Identity:
        ident.name = name.strip() or ident.name
        ident.save(self.root)
        return ident

    def delete(self, ident_id: str) -> None:
        shutil.rmtree(self.root / ident_id, ignore_errors=True)
