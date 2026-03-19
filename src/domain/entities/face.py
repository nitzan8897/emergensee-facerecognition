from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class DetectedFace:
    bounding_box: BoundingBox
    confidence: float


@dataclass(frozen=True)
class RecognitionResult:
    identity: str | None
    confidence: float
    bounding_box: BoundingBox
