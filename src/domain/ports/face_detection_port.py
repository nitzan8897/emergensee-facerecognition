from abc import ABC, abstractmethod

from domain.entities.face import DetectedFace


class FaceDetectionPort(ABC):
    @abstractmethod
    async def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        ...
