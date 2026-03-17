import asyncio
import time
from pathlib import Path

from domain.ports.face_storage_port import FaceStoragePort


class FileFaceStorage(FaceStoragePort):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def save(self, identity: str, image_bytes: bytes) -> None:
        await asyncio.to_thread(self._save_sync, identity, image_bytes)

    def _save_sync(self, identity: str, image_bytes: bytes) -> None:
        identity_dir = self._db_path / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        (identity_dir / f"{identity}_{int(time.time())}.jpg").write_bytes(image_bytes)
