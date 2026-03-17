from pydantic import BaseModel, Field

from domain.entities.face import DetectedFace, RecognitionResult


class BoundingBoxSchema(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectedFaceSchema(BaseModel):
    bounding_box: BoundingBoxSchema
    confidence: float = Field(ge=0.0, le=1.0)

    @classmethod
    def from_domain(cls, face: DetectedFace) -> "DetectedFaceSchema":
        return cls(
            bounding_box=BoundingBoxSchema(
                x=face.bounding_box.x,
                y=face.bounding_box.y,
                width=face.bounding_box.width,
                height=face.bounding_box.height,
            ),
            confidence=face.confidence,
        )


class RecognitionResultSchema(BaseModel):
    identity: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBoxSchema

    @classmethod
    def from_domain(cls, result: RecognitionResult) -> "RecognitionResultSchema":
        return cls(
            identity=result.identity,
            confidence=result.confidence,
            bounding_box=BoundingBoxSchema(
                x=result.bounding_box.x,
                y=result.bounding_box.y,
                width=result.bounding_box.width,
                height=result.bounding_box.height,
            ),
        )


class DetectResponse(BaseModel):
    faces_found: int
    faces: list[DetectedFaceSchema]


class RecognizeResponse(BaseModel):
    faces_found: int
    results: list[RecognitionResultSchema]


class RegisterResponse(BaseModel):
    registered_as: str
