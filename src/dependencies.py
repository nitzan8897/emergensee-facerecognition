from functools import lru_cache
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase

from adapters.ai.deepface_adapter import DeepFaceAdapter
from adapters.persistence.mongo_face_storage import MongoFaceStorage
from application.delete_face import DeleteFaceUseCase
from application.delete_user_face import DeleteUserFaceUseCase
from application.detect_faces import DetectFacesUseCase
from application.recognize_faces import RecognizeFacesUseCase
from application.register_face import RegisterFaceUseCase
from config import get_settings


@lru_cache(maxsize=1)
def _get_face_db_path() -> Path:
    path = Path(get_settings().face_db_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def _get_mongo_db() -> AsyncIOMotorDatabase:  # type: ignore[type-arg]
    settings = get_settings()
    client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongo_uri)  # type: ignore[type-arg]
    return client[settings.mongo_db_name]


@lru_cache(maxsize=1)
def _get_external_emergensee_collection() -> AsyncIOMotorCollection | None:  # type: ignore[type-arg]
    uri = get_settings().emergensee_mongo_uri
    if not uri:
        return None
    client: AsyncIOMotorClient = AsyncIOMotorClient(uri)  # type: ignore[type-arg]
    db: AsyncIOMotorDatabase = client.get_default_database()  # type: ignore[type-arg]
    return db["registered_faces"]


@lru_cache(maxsize=1)
def _get_deepface_adapter() -> DeepFaceAdapter:
    settings = get_settings()
    return DeepFaceAdapter(
        _get_face_db_path(),
        settings.recognition_threshold,
        settings.min_detection_confidence,
        settings.detector_backend,
        settings.recognition_model,
        settings.min_face_size_px,
        settings.min_sharpness,
    )


@lru_cache(maxsize=1)
def _get_mongo_storage() -> MongoFaceStorage:
    return MongoFaceStorage(_get_mongo_db(), _get_face_db_path(), _get_external_emergensee_collection())


def get_detect_use_case() -> DetectFacesUseCase:
    return DetectFacesUseCase(_get_deepface_adapter())


def get_recognize_use_case() -> RecognizeFacesUseCase:
    return RecognizeFacesUseCase(_get_deepface_adapter())


def get_register_use_case() -> RegisterFaceUseCase:
    return RegisterFaceUseCase(_get_mongo_storage())


def get_delete_use_case() -> DeleteFaceUseCase:
    return DeleteFaceUseCase(_get_mongo_storage())


def get_delete_user_face_use_case() -> DeleteUserFaceUseCase:
    return DeleteUserFaceUseCase(_get_mongo_storage())
