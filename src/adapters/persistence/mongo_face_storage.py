import asyncio
import shutil
import time
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from domain.ports.face_storage_port import FaceStoragePort


class MongoFaceStorage(FaceStoragePort):
    def __init__(
        self,
        db: AsyncIOMotorDatabase,  # type: ignore[type-arg]
        face_db_path: Path,
        external_collection: AsyncIOMotorCollection | None = None,  # type: ignore[type-arg]
    ) -> None:
        self._collection = db["registered_faces"]
        self._face_db_path = face_db_path
        self._external_collection = external_collection

    async def save(self, identity: str, image_bytes: bytes) -> None:
        await asyncio.gather(
            self._persist_to_mongo(identity, image_bytes),
            asyncio.to_thread(self._write_to_disk, identity, image_bytes),
        )

    async def delete(self, identity: str) -> bool:
        result, _ = await asyncio.gather(
            self._delete_from_mongo(identity),
            asyncio.to_thread(self._delete_from_disk, identity),
        )
        return bool(result)

    async def delete_by_user_id(self, user_id: str) -> bool:
        mongo_deleted, disk_deleted = await asyncio.gather(
            self._delete_from_mongo(user_id),
            asyncio.to_thread(self._delete_from_disk, user_id),
        )
        external_deleted = False
        if self._external_collection is not None:
            external_deleted = await self._delete_from_external(user_id)
        return bool(mongo_deleted) or disk_deleted or external_deleted

    async def _persist_to_mongo(self, identity: str, image_bytes: bytes) -> None:
        await self._collection.insert_one({
            "identity": identity,
            "image": image_bytes,
            "registered_at": int(time.time()),
        })

    async def _delete_from_mongo(self, identity: str) -> bool:
        result = await self._collection.delete_many({"identity": identity})
        return result.deleted_count > 0

    async def _delete_from_external(self, user_id: str) -> bool:
        result = await self._external_collection.delete_many({"userId": user_id})
        return result.deleted_count > 0

    def _write_to_disk(self, identity: str, image_bytes: bytes) -> None:
        identity_dir = self._face_db_path / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        (identity_dir / f"{identity}_{time.time_ns()}.jpg").write_bytes(image_bytes)

    def _delete_from_disk(self, identity: str) -> bool:
        identity_dir = self._face_db_path / identity
        if identity_dir.exists():
            shutil.rmtree(identity_dir)
            return True
        return False
