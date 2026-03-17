from abc import ABC, abstractmethod

from domain.entities.face import RecognitionResult


class FaceRecognitionPort(ABC):
    @abstractmethod
    async def recognize(self, image_bytes: bytes) -> list[RecognitionResult]:
        ...
