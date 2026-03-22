from domain.ports.face_storage_port import FaceStoragePort


class DeleteFaceUseCase:
    def __init__(self, storage: FaceStoragePort) -> None:
        self._storage = storage

    async def execute(self, identity: str) -> bool:
        normalized = identity.strip().lower().replace(" ", "_")
        if not normalized:
            raise ValueError("Identity name cannot be empty.")
        return await self._storage.delete(normalized)
