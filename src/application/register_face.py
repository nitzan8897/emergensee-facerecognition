from domain.ports.face_storage_port import FaceStoragePort


class RegisterFaceUseCase:
    def __init__(self, storage: FaceStoragePort) -> None:
        self._storage = storage

    async def execute(self, identity: str, image_bytes: bytes) -> str:
        if not image_bytes:
            raise ValueError("Image payload is empty.")
        normalized = identity.strip().lower().replace(" ", "_")
        if not normalized:
            raise ValueError("Identity name cannot be empty.")
        await self._storage.save(normalized, image_bytes)
        return normalized
