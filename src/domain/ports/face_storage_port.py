from abc import ABC, abstractmethod


class FaceStoragePort(ABC):
    @abstractmethod
    async def save(self, identity: str, image_bytes: bytes) -> None:
        ...

    @abstractmethod
    async def delete(self, identity: str) -> bool:
        """Delete all data for *identity*. Returns True if anything was deleted, False if not found."""
        ...
