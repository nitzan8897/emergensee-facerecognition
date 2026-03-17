import asyncio
import time
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorDatabase

from domain.ports.face_storage_port import FaceStoragePort


class MongoFaceStorage(FaceStoragePort):
    """
    Saves the face image to both MongoDB (for persistence and auditability)
    and to disk (required by DeepFace for directory scanning during recognition).
    """

    def __init__(self, db: AsyncIOMotorDatabase, face_db_path: Path) -> None:
        self._collection = db["registered_faces"]
        self._face_db_path = face_db_path

    async def save(self, identity: str, image_bytes: bytes) -> None:
        await asyncio.gather(
            self._persist_to_mongo(identity, image_bytes),
            asyncio.to_thread(self._write_to_disk, identity, image_bytes),
        )

    async def _persist_to_mongo(self, identity: str, image_bytes: bytes) -> None:
        await self._collection.insert_one({
            "identity": identity,
            "image": image_bytes,
            "registered_at": int(time.time()),
        })

    def _write_to_disk(self, identity: str, image_bytes: bytes) -> None:
        identity_dir = self._face_db_path / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        (identity_dir / f"{identity}_{int(time.time())}.jpg").write_bytes(image_bytes)
