from functools import lru_cache
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from adapters.ai.deepface_adapter import DeepFaceAdapter
from adapters.persistence.mongo_face_storage import MongoFaceStorage
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
def _get_mongo_db() -> AsyncIOMotorDatabase:
    settings = get_settings()
    client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongo_uri)
    return client[settings.mongo_db_name]


@lru_cache(maxsize=1)
def _get_deepface_adapter() -> DeepFaceAdapter:
    settings = get_settings()
    return DeepFaceAdapter(_get_face_db_path(), settings.recognition_threshold, settings.detector_backend)


@lru_cache(maxsize=1)
def _get_mongo_storage() -> MongoFaceStorage:
    return MongoFaceStorage(_get_mongo_db(), _get_face_db_path())


def get_detect_use_case() -> DetectFacesUseCase:
    return DetectFacesUseCase(_get_deepface_adapter())


def get_recognize_use_case() -> RecognizeFacesUseCase:
    return RecognizeFacesUseCase(_get_deepface_adapter())


def get_register_use_case() -> RegisterFaceUseCase:
    return RegisterFaceUseCase(_get_mongo_storage())
