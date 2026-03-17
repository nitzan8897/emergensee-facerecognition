from abc import ABC, abstractmethod


class FaceStoragePort(ABC):
    @abstractmethod
    async def save(self, identity: str, image_bytes: bytes) -> None:
        ...
