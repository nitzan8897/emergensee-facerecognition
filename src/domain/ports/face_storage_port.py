from abc import ABC, abstractmethod


class FaceStoragePort(ABC):
    @abstractmethod
    async def save(self, identity: str, image_bytes: bytes) -> None:
        ...

    @abstractmethod
    async def delete(self, identity: str) -> bool:
        """Delete all data for *identity*. Returns True if anything was deleted, False if not found."""
        ...

    @abstractmethod
    async def delete_by_user_id(self, user_id: str) -> bool:
        """Delete face_db folder and external DB records for *user_id*. Returns True if anything was deleted."""
        ...
