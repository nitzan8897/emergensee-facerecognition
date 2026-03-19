from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from fastapi import Form

from api.schemas.face_schemas import (
    DeleteResponse,
    DetectedFaceSchema,
    DetectResponse,
    RecognitionResultSchema,
    RecognizeResponse,
    RegisterResponse,
)
from application.delete_face import DeleteFaceUseCase
from application.detect_faces import DetectFacesUseCase
from application.recognize_faces import RecognizeFacesUseCase
from application.register_face import RegisterFaceUseCase
from dependencies import get_delete_use_case, get_detect_use_case, get_recognize_use_case, get_register_use_case

router = APIRouter(prefix="/api/v1/faces", tags=["faces"])

_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _validate_image(image: UploadFile) -> UploadFile:
    if image.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported type '{image.content_type}'. Accepted: jpeg, png, webp.",
        )
    return image


@router.post(
    "/detect",
    response_model=DetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect all faces in an image",
    description="Returns bounding boxes and confidence scores for every face found. Does NOT identify who the faces belong to.",
)
async def detect_faces(
    image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP image")],
    use_case: Annotated[DetectFacesUseCase, Depends(get_detect_use_case)],
) -> DetectResponse:
    _validate_image(image)
    image_bytes = await image.read()
    faces = await use_case.execute(image_bytes)
    return DetectResponse(
        faces_found=len(faces),
        faces=[DetectedFaceSchema.from_domain(f) for f in faces],
    )


@router.post(
    "/recognize",
    response_model=RecognizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Recognize faces in an image",
    description="Detects faces and matches each one against the known identity database. Returns identity labels and confidence scores.",
)
async def recognize_faces(
    image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP image")],
    use_case: Annotated[RecognizeFacesUseCase, Depends(get_recognize_use_case)],
) -> RecognizeResponse:
    _validate_image(image)
    image_bytes = await image.read()
    results = await use_case.execute(image_bytes)
    return RecognizeResponse(
        faces_found=len(results),
        results=[RecognitionResultSchema.from_domain(r) for r in results],
    )


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a face",
    description="Saves the uploaded image under the given identity name. The face becomes recognizable on subsequent /recognize calls.",
)
async def register_face(
    image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP image")],
    name: Annotated[str, Form(description="Identity name, e.g. 'John Doe'")],
    use_case: Annotated[RegisterFaceUseCase, Depends(get_register_use_case)],
) -> RegisterResponse:
    _validate_image(image)
    image_bytes = await image.read()
    registered_as = await use_case.execute(name, image_bytes)
    return RegisterResponse(registered_as=registered_as)


@router.delete(
    "/{identity}",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a registered face",
    description="Removes all images and records for the given identity from both the database and disk.",
)
async def delete_face(
    identity: str,
    use_case: Annotated[DeleteFaceUseCase, Depends(get_delete_use_case)],
) -> DeleteResponse:
    deleted = await use_case.execute(identity)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Identity '{identity}' not found.",
        )
    return DeleteResponse(deleted=identity)
