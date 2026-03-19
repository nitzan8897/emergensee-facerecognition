# EmergenSee — Face Recognition Microservice

An always-on AI microservice for real-time face detection and recognition at the entrance of an emergency room. When an alarm is triggered, every person entering is identified and their registered contacts are notified: **"Person X is in the safe room."**

Built with **FastAPI** and **Hexagonal Architecture**.

## Stack

| Layer | Technology |
| --- | --- |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| AI — Detection | RetinaFace (via DeepFace) |
| AI — Recognition | ArcFace (via DeepFace) |
| CV | OpenCV |
| Database | MongoDB (Motor async driver) |
| Packaging | Poetry |
| Linting | Ruff + mypy (strict) |

## Quick Start

```bash
# 1. Install dependencies (including AI/CV libs)
poetry install
poetry run pip install opencv-python-headless deepface numpy

# 2. Make sure MongoDB is running locally (localhost:27017)

# 3. Run from the project root
poetry run uvicorn main:app --app-dir src --host 0.0.0.0 --port 8000 --reload

# 4. Health check
curl http://localhost:8000/health

# 5. Swagger UI (available when DEBUG=true)
open http://localhost:8000/docs
```

> **Note:** RetinaFace and ArcFace models are downloaded automatically on the first request (~200–500 MB).

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `POST` | `/api/v1/faces/detect` | Detect all faces in an image (bounding boxes + confidence) |
| `POST` | `/api/v1/faces/recognize` | Identify faces against the registered database |
| `POST` | `/api/v1/faces/register` | Register a new person (name + image) |
| `DELETE` | `/api/v1/faces/{identity}` | Remove a registered person entirely |

### Register a person

```bash
curl -X POST http://localhost:8000/api/v1/faces/register \
  -F "name=John Doe" \
  -F "image=@photo.jpg"
```

### Recognize faces in an image

```bash
curl -X POST http://localhost:8000/api/v1/faces/recognize \
  -F "image=@frame.jpg"
```

### Delete a registered person

```bash
curl -X DELETE http://localhost:8000/api/v1/faces/john_doe
```

## Configuration (`.env`)

```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true

# MongoDB
MONGO_URI=mongodb://localhost:27017

# AI Model
DETECTOR_BACKEND=retinaface       # retinaface (default) | mtcnn | opencv
RECOGNITION_MODEL=ArcFace         # ArcFace (default) | Facenet512
RECOGNITION_THRESHOLD=0.55        # cosine distance — lower = stricter
MIN_DETECTION_CONFIDENCE=0.01     # minimum RetinaFace confidence to proceed
MIN_FACE_SIZE_PX=20               # reject bounding boxes smaller than this
MIN_SHARPNESS=5.0                 # Laplacian variance floor — rejects lights/blurs

# Storage
FACE_DB_PATH=face_db
```

### Threshold tuning guide

| Symptom | Fix |
| --- | --- |
| Real people not recognized | Lower `RECOGNITION_THRESHOLD` (try 0.60) |
| Wrong person matched | Raise `RECOGNITION_THRESHOLD` (try 0.45) |
| Lights / objects detected as faces | Raise `MIN_SHARPNESS` to 30–80 |
| Real faces rejected by quality gate | Lower `MIN_SHARPNESS` to 5, `MIN_DETECTION_CONFIDENCE` to 0.01 |
| Partial faces missed | Lower `MIN_DETECTION_CONFIDENCE`, lower `MIN_FACE_SIZE_PX` |

## Project Structure

```text
src/
├── main.py                     # App entry point and FastAPI lifecycle
├── config.py                   # Typed settings loaded from .env
├── dependencies.py             # Dependency injection wiring
│
├── domain/                     # Core logic — zero external dependencies
│   ├── entities/face.py        # BoundingBox, DetectedFace, RecognitionResult
│   └── ports/                  # Abstract interfaces (like C# interfaces)
│       ├── face_detection_port.py
│       ├── face_recognition_port.py
│       └── face_storage_port.py
│
├── application/                # Use cases (one class, one execute() method)
│   ├── detect_faces.py
│   ├── recognize_faces.py
│   ├── register_face.py
│   └── delete_face.py
│
├── adapters/                   # Concrete implementations of ports
│   ├── ai/
│   │   └── deepface_adapter.py # RetinaFace detection + ArcFace recognition
│   └── persistence/
│       └── mongo_face_storage.py # MongoDB + disk storage for face images
│
└── api/                        # HTTP layer — no business logic
    ├── routers/faces.py        # FastAPI route handlers
    └── schemas/face_schemas.py # Pydantic request/response DTOs
```

## Architecture

**Hexagonal (Ports & Adapters)** — the domain never imports from adapters. AI providers are swappable behind an interface.

```text
HTTP Request
    │
    ▼
[API Router]          ← validates HTTP, calls use case
    │
    ▼
[Use Case]            ← orchestrates, no framework knowledge
    │
    ├──▶ [FaceDetectionPort]   ──▶ DeepFaceAdapter (RetinaFace)
    ├──▶ [FaceRecognitionPort] ──▶ DeepFaceAdapter (ArcFace)
    └──▶ [FaceStoragePort]     ──▶ MongoFaceStorage
```

## How the AI Pipeline Works

1. **RetinaFace** scans the image and outputs bounding boxes with confidence scores. Handles partial and occluded faces well.
2. A **quality gate** rejects detections that are too small, too blurry, or too low confidence (prevents lights/objects triggering recognition).
3. Each detected face is **cropped individually** and passed to **ArcFace**, which converts it into a 512-dimensional embedding vector.
4. The embedding is compared against all registered face embeddings using **cosine distance**. Distance below `RECOGNITION_THRESHOLD` = matched identity.

## Registration Tips

Register **3–5 images per person** from different angles for best recall on partial/side faces:

- Full frontal
- Slight left turn
- Slight right turn

`DeepFace.find()` compares against all registered images and returns the closest match automatically.
