import asyncio
import shutil
import time
from pathlib import Path

from domain.ports.face_storage_port import FaceStoragePort


class FileFaceStorage(FaceStoragePort):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def save(self, identity: str, image_bytes: bytes) -> None:
        await asyncio.to_thread(self._save_sync, identity, image_bytes)

    async def delete(self, identity: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, identity)

    async def delete_by_user_id(self, user_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, user_id)

    def _save_sync(self, identity: str, image_bytes: bytes) -> None:
        identity_dir = self._db_path / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        (identity_dir / f"{identity}_{int(time.time())}.jpg").write_bytes(image_bytes)

    def _delete_sync(self, identity: str) -> bool:
        identity_dir = self._db_path / identity
        if identity_dir.exists():
            shutil.rmtree(identity_dir)
            return True
        return False
