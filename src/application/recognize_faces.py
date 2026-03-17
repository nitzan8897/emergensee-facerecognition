from domain.entities.face import RecognitionResult
from domain.ports.face_recognition_port import FaceRecognitionPort


class RecognizeFacesUseCase:
    def __init__(self, recognizer: FaceRecognitionPort) -> None:
        self._recognizer = recognizer

    async def execute(self, image_bytes: bytes) -> list[RecognitionResult]:
        if not image_bytes:
            raise ValueError("Image payload is empty.")
        return await self._recognizer.recognize(image_bytes)
