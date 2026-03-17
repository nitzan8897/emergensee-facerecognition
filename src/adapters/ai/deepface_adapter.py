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
    def __init__(self, face_db_path: Path, recognition_threshold: float, detector_backend: str) -> None:
        self._face_db_path = face_db_path
        self._threshold = recognition_threshold
        self._detector = detector_backend

    async def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        return await asyncio.to_thread(self._detect_sync, image_bytes)

    async def recognize(self, image_bytes: bytes) -> list[RecognitionResult]:
        return await asyncio.to_thread(self._recognize_sync, image_bytes)

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _detect_sync(self, image_bytes: bytes) -> list[DetectedFace]:
        faces = DeepFace.extract_faces(
            img_path=self._decode(image_bytes),
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

    def _recognize_sync(self, image_bytes: bytes) -> list[RecognitionResult]:
        if not self._face_db_path.exists() or not any(self._face_db_path.rglob("*.jpg")):
            return []

        results = DeepFace.find(
            img_path=self._decode(image_bytes),
            db_path=str(self._face_db_path),
            model_name="ArcFace",
            distance_metric="cosine",
            threshold=self._threshold,
            detector_backend=self._detector,
            enforce_detection=False,
            refresh_database=True,
            silent=True,
        )

        output = []
        for df in results:
            logger.debug("DeepFace result columns: %s", list(df.columns))
            if df.empty:
                logger.debug("Face detected but no match found within threshold %.2f", self._threshold)
                continue
            top = df.iloc[0]
            distance_cols = [c for c in df.columns if "cosine" in c.lower() or "distance" in c.lower()]
            if not distance_cols:
                logger.warning("No distance column found. Columns: %s", list(df.columns))
                continue
            distance_col = distance_cols[0]
            distance = float(top[distance_col])
            identity = Path(str(top["identity"])).parent.name
            logger.debug("Best match: %s  distance=%.4f  threshold=%.2f  col=%s", identity, distance, self._threshold, distance_col)
            output.append(
                RecognitionResult(
                    identity=identity,
                    confidence=float(max(0.0, 1.0 - distance)),
                    bounding_box=BoundingBox(
                        x=int(top["source_x"]),
                        y=int(top["source_y"]),
                        width=int(top["source_w"]),
                        height=int(top["source_h"]),
                    ),
                )
            )
        return output
