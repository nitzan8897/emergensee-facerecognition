import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

from domain.entities.face import BoundingBox, DetectedFace, RecognitionResult
from domain.ports.face_detection_port import FaceDetectionPort
from domain.ports.face_recognition_port import FaceRecognitionPort

logger = logging.getLogger(__name__)

_IOU_HIT_THRESHOLD = 0.40  # min bounding-box overlap to consider "same face" across frames
_FRAME_CACHE_TTL = 2.5     # seconds to reuse a cached result for a spatially-stable face


@dataclass
class _DBEmbedding:
    identity: str
    vector: np.ndarray  # L2-normalised float32, shape (D,)


@dataclass
class _TrackedFace:
    bbox: BoundingBox
    identity: str | None
    confidence: float
    expires_at: float


class DeepFaceAdapter(FaceDetectionPort, FaceRecognitionPort):
    def __init__(
        self,
        face_db_path: Path,
        recognition_threshold: float,
        min_detection_confidence: float,
        detector_backend: str,
        recognition_model: str,
        min_face_size_px: int,
        min_sharpness: float,
    ) -> None:
        self._face_db_path = face_db_path
        self._threshold = recognition_threshold
        self._min_detection_confidence = min_detection_confidence
        self._detector = detector_backend
        self._recognition_model = recognition_model
        self._min_face_size_px = min_face_size_px
        self._min_sharpness = min_sharpness
        self._last_db_mtime: float = 0.0
        self._db_embeddings: list[_DBEmbedding] = []
        self._db_matrix: np.ndarray | None = None  # (N, D) pre-stacked for vectorised search
        self._tracked_faces: list[_TrackedFace] = []
        self._track_lock = threading.Lock()
        self._rebuild_lock = threading.Lock()


    async def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        return await asyncio.to_thread(self._detect_sync, image_bytes)

    async def recognize(self, image_bytes: bytes) -> list[RecognitionResult]:
        return await asyncio.to_thread(self._recognize_sync, image_bytes)

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _compute_sharpness(self, image: np.ndarray, bbox: BoundingBox) -> float:
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        crop = image[y: y + h, x: x + w]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _crop_face(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1 = max(0, bbox.x), max(0, bbox.y)
        x2, y2 = min(w, bbox.x + bbox.width), min(h, bbox.y + bbox.height)
        return image[y1:y2, x1:x2]

    def _detect_sync(self, image_bytes: bytes, image: np.ndarray | None = None) -> list[DetectedFace]:
        if image is None:
            image = self._decode(image_bytes)
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend=self._detector,
            enforce_detection=False,
        )
        return [
            DetectedFace(
                bounding_box=BoundingBox(
                    x=int(f["facial_area"]["x"]),
                    y=int(f["facial_area"]["y"]),
                    width=int(f["facial_area"]["w"]),
                    height=int(f["facial_area"]["h"]),
                ),
                confidence=float(f["confidence"]),
            )
            for f in faces
        ]

    def _filter_real_faces(self, faces: list[DetectedFace], image: np.ndarray) -> list[DetectedFace]:
        real: list[DetectedFace] = []
        for f in faces:
            bb = f.bounding_box
            if f.confidence < self._min_detection_confidence:
                logger.debug("Rejected: confidence %.2f < %.2f", f.confidence, self._min_detection_confidence)
                continue
            if bb.width < self._min_face_size_px or bb.height < self._min_face_size_px:
                logger.debug("Rejected: size %dx%d < %dpx", bb.width, bb.height, self._min_face_size_px)
                continue
            sharpness = self._compute_sharpness(image, bb)
            if sharpness < self._min_sharpness:
                logger.debug("Rejected: sharpness %.1f < %.1f", sharpness, self._min_sharpness)
                continue
            real.append(f)
        return real

    # ── In-memory DB embedding cache ──────────────────────────────────────────

    def _rebuild_db_embeddings(self) -> None:
        """Load every registered face image into a normalised numpy matrix.

        Called once on first request, then only when face_db/ changes on disk.
        Much faster than DeepFace.find() which hits disk + pandas on every call.
        """
        logger.info("Rebuilding in-memory face embeddings from %s ...", self._face_db_path)
        entries: list[_DBEmbedding] = []
        for img_path in sorted(self._face_db_path.rglob("*.jpg")):
            identity = img_path.parent.name
            try:
                reps = DeepFace.represent(
                    img_path=str(img_path),
                    model_name=self._recognition_model,
                    detector_backend=self._detector,
                    enforce_detection=False,
                    align=True,
                )
                for rep in reps:
                    vec = np.array(rep["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-10:
                        vec /= norm
                    entries.append(_DBEmbedding(identity=identity, vector=vec))
            except Exception as exc:
                logger.warning("Skipping %s: %s", img_path.name, exc)

        self._db_embeddings = entries
        self._db_matrix = np.stack([e.vector for e in entries]) if entries else None
        logger.info(
            "Loaded %d vectors for %d identities",
            len(entries),
            len({e.identity for e in entries}),
        )

    def _get_query_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        """Encode a pre-cropped face region into a normalised embedding.

        Uses detector_backend="skip" because RetinaFace already isolated this
        face — re-running detection on the crop is the main source of the 17s latency.
        """
        try:
            reps = DeepFace.represent(
                img_path=crop,
                model_name=self._recognition_model,
                detector_backend="skip",
                enforce_detection=False,
            )
        except Exception as exc:
            logger.warning("DeepFace.represent() failed: %s", exc)
            return None
        if not reps:
            return None
        vec = np.array(reps[0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else None

    def _match_query(self, query: np.ndarray) -> tuple[str | None, float]:
        """One matrix multiply to find the closest registered identity."""
        if self._db_matrix is None:
            return None, 0.0
        sims: np.ndarray = self._db_matrix @ query  # (N,) cosine similarities
        identity_max: dict[str, float] = {}
        for entry, sim in zip(self._db_embeddings, sims.tolist()):
            if entry.identity not in identity_max or sim > identity_max[entry.identity]:
                identity_max[entry.identity] = sim
        if not identity_max:
            return None, 0.0
        best_id = max(identity_max, key=lambda k: identity_max[k])
        best_conf = identity_max[best_id]
        min_conf = 1.0 - self._threshold
        logger.debug("Best match: %s  confidence=%.4f  threshold=%.2f", best_id, best_conf, self._threshold)
        return (best_id if best_conf >= min_conf else None), best_conf

    # ── Temporal tracking cache (cross-frame deduplication) ───────────────────

    @staticmethod
    def _iou(a: BoundingBox, b: BoundingBox) -> float:
        ax2, ay2 = a.x + a.width, a.y + a.height
        bx2, by2 = b.x + b.width, b.y + b.height
        ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = a.width * a.height + b.width * b.height - inter
        return inter / union if union > 0 else 0.0

    def _find_tracked(self, bbox: BoundingBox) -> _TrackedFace | None:
        """Return a cached result if this bounding box overlaps a recently-seen face."""
        now = time.monotonic()
        with self._track_lock:
            self._tracked_faces = [f for f in self._tracked_faces if f.expires_at > now]
            return next(
                (t for t in self._tracked_faces if self._iou(bbox, t.bbox) >= _IOU_HIT_THRESHOLD),
                None,
            )

    def _update_tracked(self, bbox: BoundingBox, identity: str | None, confidence: float) -> None:
        now = time.monotonic()
        expires = now + _FRAME_CACHE_TTL
        with self._track_lock:
            for t in self._tracked_faces:
                if self._iou(bbox, t.bbox) >= _IOU_HIT_THRESHOLD:
                    t.bbox, t.identity, t.confidence, t.expires_at = bbox, identity, confidence, expires
                    return
            self._tracked_faces.append(
                _TrackedFace(bbox=bbox, identity=identity, confidence=confidence, expires_at=expires)
            )

    # ── Per-face recognition ──────────────────────────────────────────────────

    def _recognize_single_face(self, face: DetectedFace, image: np.ndarray) -> RecognitionResult:
        cached = self._find_tracked(face.bounding_box)
        if cached is not None:
            logger.debug("Tracking cache hit → %s (conf=%.3f)", cached.identity, cached.confidence)
            return RecognitionResult(
                identity=cached.identity,
                confidence=cached.confidence,
                bounding_box=face.bounding_box,
            )

        crop = self._crop_face(image, face.bounding_box)
        query = self._get_query_embedding(crop)
        if query is None:
            return RecognitionResult(identity=None, confidence=0.0, bounding_box=face.bounding_box)

        identity, confidence = self._match_query(query)
        self._update_tracked(face.bounding_box, identity, confidence)
        return RecognitionResult(identity=identity, confidence=confidence, bounding_box=face.bounding_box)

    # ── Full-image recognition pipeline ──────────────────────────────────────

    def _recognize_sync(self, image_bytes: bytes) -> list[RecognitionResult]:
        image = self._decode(image_bytes)
        real_faces = self._filter_real_faces(self._detect_sync(image_bytes, image), image)
        if not real_faces:
            logger.debug("No faces passed quality gate — skipping recognition")
            return []

        if not self._face_db_path.exists() or not any(self._face_db_path.rglob("*.jpg")):
            return [
                RecognitionResult(identity=None, confidence=f.confidence, bounding_box=f.bounding_box)
                for f in real_faces
            ]

        current_mtime = self._face_db_path.stat().st_mtime
        if current_mtime != self._last_db_mtime:
            with self._rebuild_lock:
                if current_mtime != self._last_db_mtime:  # double-check inside lock
                    self._rebuild_db_embeddings()
                    self._last_db_mtime = current_mtime

        if len(real_faces) == 1:
            return [self._recognize_single_face(real_faces[0], image)]

        with ThreadPoolExecutor(max_workers=min(len(real_faces), 8)) as pool:
            futures = [pool.submit(self._recognize_single_face, face, image) for face in real_faces]
            return [f.result() for f in futures]
