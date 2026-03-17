import asyncio

from domain.entities.face import BoundingBox, DetectedFace, RecognitionResult
from domain.ports.face_detection_port import FaceDetectionPort
from domain.ports.face_recognition_port import FaceRecognitionPort


class MockFaceAdapter(FaceDetectionPort, FaceRecognitionPort):
    async def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        await asyncio.sleep(0)
        return [
            DetectedFace(
                bounding_box=BoundingBox(x=52, y=41, width=118, height=136),
                confidence=0.99,
            )
        ]

    async def recognize(self, image_bytes: bytes) -> list[RecognitionResult]:
        await asyncio.sleep(0)
        return [
            RecognitionResult(
                identity="john_doe",
                confidence=0.94,
                bounding_box=BoundingBox(x=52, y=41, width=118, height=136),
            )
        ]
