from domain.entities.face import DetectedFace
from domain.ports.face_detection_port import FaceDetectionPort


class DetectFacesUseCase:
    def __init__(self, detector: FaceDetectionPort) -> None:
        self._detector = detector

    async def execute(self, image_bytes: bytes) -> list[DetectedFace]:
        if not image_bytes:
            raise ValueError("Image payload is empty.")
        return await self._detector.detect(image_bytes)
