import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

from domain.entities.face import BoundingBox, DetectedFace, RecognitionResult
from domain.ports.face_detection_port import FaceDetectionPort
from domain.ports.face_recognition_port import FaceRecognitionPort

logger = logging.getLogger(__name__)


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
        needs_refresh = current_mtime != self._last_db_mtime
        self._last_db_mtime = current_mtime

        output: list[RecognitionResult] = []
        for face in real_faces:
            result = self._recognize_single_face(face, image, needs_refresh)
            output.append(result)
            needs_refresh = False  # only rebuild cache on the first call per request
        return output

    def _crop_face(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        h, w = image.shape[:2]
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(w, bbox.x + bbox.width)
        y2 = min(h, bbox.y + bbox.height)
        return image[y1:y2, x1:x2]

    def _recognize_single_face(self, face: DetectedFace, image: np.ndarray, refresh_database: bool) -> RecognitionResult:
        crop = self._crop_face(image, face.bounding_box)
        try:
            results = DeepFace.find(
                img_path=crop,
                db_path=str(self._face_db_path),
                model_name=self._recognition_model,
                distance_metric="cosine",
                threshold=self._threshold,
                detector_backend=self._detector,
                enforce_detection=False,
                refresh_database=refresh_database,
                silent=True,
            )
        except Exception as exc:
            logger.warning("DeepFace.find() failed for face crop: %s", exc)
            return RecognitionResult(identity=None, confidence=0.0, bounding_box=face.bounding_box)

        if not results or results[0].empty:
            logger.debug("No match found within threshold %.2f", self._threshold)
            return RecognitionResult(identity=None, confidence=0.0, bounding_box=face.bounding_box)

        df = results[0]
        logger.debug("DeepFace result columns: %s", list(df.columns))
        top = df.iloc[0]
        distance_cols = [c for c in df.columns if "cosine" in c.lower() or "distance" in c.lower()]
        if not distance_cols:
            logger.warning("No distance column found. Columns: %s", list(df.columns))
            return RecognitionResult(identity=None, confidence=0.0, bounding_box=face.bounding_box)

        distance = float(top[distance_cols[0]])
        confidence = float(max(0.0, 1.0 - distance))
        recognized_identity: str | None = Path(str(top["identity"])).parent.name if confidence >= (1.0 - self._threshold) else None
        logger.debug(
            "Best match: %s  distance=%.4f  confidence=%.4f  threshold=%.2f",
            recognized_identity, distance, confidence, self._threshold,
        )
        return RecognitionResult(
            identity=recognized_identity,
            confidence=confidence,
            bounding_box=face.bounding_box,
        )
