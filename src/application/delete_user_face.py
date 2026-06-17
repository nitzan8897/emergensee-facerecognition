from domain.ports.face_storage_port import FaceStoragePort


class DeleteUserFaceUseCase:
    def __init__(self, storage: FaceStoragePort) -> None:
        self._storage = storage

    async def execute(self, user_id: str) -> bool:
        stripped = user_id.strip()
        if not stripped:
            raise ValueError("User ID cannot be empty.")
        return await self._storage.delete_by_user_id(stripped)
