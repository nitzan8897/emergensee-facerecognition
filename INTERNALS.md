# EmergenSee Face Recognition — Internal Engineering Overview

> This document is NOT for upload. It is a personal reference explaining what this project is,
> how it works end-to-end, what every library does, and the full data flow.

---

## What This Project Is

EmergenSee is a **backend AI microservice** whose sole responsibility is face recognition.
It exposes an HTTP API. A caller (a mobile app, another service, a camera feed processor) sends
it an image, and it responds with identity data — who is in the image, confidence scores,
bounding boxes, etc.

It is NOT a monolith. It is NOT a frontend. It is a focused, containerized service that does
one thing: run computer vision inference and return structured results.

---

## Why This Architecture (Hexagonal / Ports & Adapters)

The core problem with AI services is that **AI libraries change constantly**. DeepFace gets
replaced by InsightFace, OpenCV gets replaced by a cloud API, the model backend switches from
CPU to GPU. If your business logic is tangled with `import deepface`, every swap is painful.

Hexagonal Architecture solves this by defining the *capability* in an abstract interface (a Port)
and putting the actual library call behind an Adapter. The business logic only ever talks to the
interface. You can swap the entire AI stack without touching a single line of domain code.

```
External World (HTTP)
       │
       ▼
  [ API Layer ]          ← FastAPI routers. Validates input. Calls use-cases.
       │
       ▼
[ Application Layer ]    ← Use-cases. Orchestrates the flow. No library imports.
       │
       ▼
  [ Domain Layer ]       ← Entities and Ports (ABCs). Pure Python. Zero dependencies.
       │
       ▼
  [ Adapters Layer ]     ← Implements the Ports using real libraries (DeepFace, OpenCV, DB).
```

The dependency arrow always points **inward**. The Domain knows nothing about FastAPI or
DeepFace. The API layer knows nothing about the database. Each ring is replaceable.

---

## Directory Map — What Lives Where and Why

```
src/emergensee/
│
├── main.py
│   The FastAPI application factory. Creates the `app` object, registers routers,
│   and defines the Lifespan (startup/shutdown hooks). This is the entry point
│   for uvicorn. Nothing business-related lives here.
│
├── config.py
│   One source of truth for all configuration. Reads from environment variables
│   (or .env file). Uses Pydantic BaseSettings so every value is type-validated
│   at startup — if PORT=banana, the service refuses to start rather than
│   crashing later in production.
│
├── domain/
│   ├── entities/
│   │   Business objects. A `Face` entity, a `RecognitionResult` entity, etc.
│   │   These are plain Python dataclasses or Pydantic models. NO FastAPI imports,
│   │   NO deepface imports, NO database imports. They describe the shape of data
│   │   that the business cares about.
│   │
│   └── ports/
│       Abstract Base Classes that define *what* capabilities exist without
│       saying *how* they work. Example:
│
│           class FaceRecognitionPort(ABC):
│               @abstractmethod
│               async def recognize(self, image: bytes) -> list[RecognitionResult]:
│                   ...
│
│       This is the contract. Adapters must fulfill it. Use-cases depend on it.
│
├── application/
│   Use-case classes. Example: `RecognizeFaceUseCase`. It receives a
│   `FaceRecognitionPort` via constructor injection (Dependency Inversion),
│   calls `port.recognize(image)`, applies any business rules (filtering,
│   thresholding, logging), and returns a result. It never imports DeepFace.
│
├── adapters/
│   ├── ai/
│   │   Concrete implementations of the domain Ports using real AI libraries.
│   │   `DeepFaceAdapter` will implement `FaceRecognitionPort` and contain
│   │   the actual `DeepFace.find(...)` call. If you later want to use
│   │   InsightFace instead, you write `InsightFaceAdapter`, implement the
│   │   same Port, and swap it in the DI wiring — zero domain changes.
│   │
│   └── persistence/
│       Database adapters. If we store known faces in a database, the adapter
│       here implements a `FaceRepositoryPort` and contains the actual SQL/ORM
│       queries. The domain never sees SQLAlchemy.
│
├── api/
│   ├── routers/
│   │   FastAPI router files. One file per feature (e.g., `recognition.py`,
│   │   `health.py`). Each endpoint function: validates input via Pydantic schema,
│   │   pulls the use-case from FastAPI's DI system, calls it, returns a schema.
│   │   No business logic. No direct AI calls.
│   │
│   └── schemas/
│       Pydantic v2 models that define the exact shape of API request/response
│       bodies. These are separate from domain entities on purpose — the API
│       contract can evolve independently of the internal business model.
│
tests/
├── unit/          Tests for domain entities and use-cases in isolation (no DB, no AI).
└── integration/   Tests that spin up the full FastAPI app (via httpx + TestClient)
                   and hit real endpoints, potentially with a real or stub adapter.
```

---

## Libraries — What Each One Does

### FastAPI
The web framework. It handles HTTP routing, request parsing, and response serialization.
You define Python functions decorated with `@app.get(...)`, `@app.post(...)`, etc., and
FastAPI generates OpenAPI docs automatically. It is async-native and built on top of Starlette.

### Uvicorn
The ASGI server. FastAPI is an ASGI application — it cannot run on its own. Uvicorn is the
process that actually binds to the port, accepts TCP connections, and calls FastAPI's handlers.
Think of it like Gunicorn but for async Python. In development: `uvicorn ... --reload`. In
production: `uvicorn ... --workers 4` (or managed by a process supervisor).

### Pydantic v2
Two jobs in this project:
1. **Data validation**: Define `class RecognitionRequest(BaseModel)` and FastAPI
   automatically validates that incoming JSON matches the schema. Invalid input → 422 response,
   no code needed.
2. **Settings management** (via `pydantic-settings`): `BaseSettings` reads from environment
   variables, validates types, and raises errors at startup for bad config. This is `config.py`.

Pydantic v2 is a full rewrite of v1 — roughly 5-50x faster validation via a Rust core.

### pydantic-settings
A Pydantic extension that adds `BaseSettings`. It knows how to read from `.env` files and
environment variables. It is separate from Pydantic core since v2.

### python-dotenv
Loads a `.env` file into environment variables before anything else reads them. Used in local
development so you don't have to `export VAR=value` manually in your shell.

### Ruff
A Python linter and formatter written in Rust — replaces Flake8, isort, pyupgrade, and more,
all in one tool. Runs in milliseconds even on large codebases. Configured in `pyproject.toml`
under `[tool.ruff]`.

### mypy (strict mode)
Static type checker. With `--strict`, it enforces that every function has type annotations and
that all types are consistent. Catches a whole class of bugs before runtime. The `pydantic.mypy`
plugin teaches mypy to understand Pydantic model fields.

### pytest + pytest-asyncio
Test runner. `pytest-asyncio` adds support for `async def test_...` functions, which is
necessary because FastAPI endpoints and use-cases are async.

### httpx
An async HTTP client. Used in integration tests to call the FastAPI app in-process via
`httpx.AsyncClient(app=app, base_url="http://test")` — no real network required.

### OpenCV (`opencv-python-headless`) — PLANNED Phase 3
The industry-standard computer vision library. Used for image decoding (JPEG/PNG bytes → numpy
array), preprocessing (resize, normalize, convert color spaces), and potentially drawing
bounding boxes on output images. The `-headless` variant has no GUI dependencies, which is
required in a Docker container (no display server). This is why `libgl1-mesa-glx` is installed
in the Dockerfile — OpenCV links against libGL even in headless mode.

### DeepFace — PLANNED Phase 3
A high-level face recognition framework that wraps several state-of-the-art models:
- **Detection backends**: RetinaFace, MTCNN, OpenCV, MediaPipe
- **Recognition models**: ArcFace, FaceNet512, VGG-Face, DeepID

You pass it a raw image and a database path, and it returns identity matches with distance
scores. The `DeepFaceAdapter` in `adapters/ai/` will wrap these calls behind the
`FaceRecognitionPort` interface.

### Poetry
Dependency manager and build tool. `pyproject.toml` declares dependencies with version
constraints. `poetry.lock` pins every transitive dependency to an exact version, making builds
reproducible. `poetry install` creates a `.venv` with exactly those versions. The Dockerfile
uses Poetry in the builder stage to produce that `.venv`, then copies it to the runtime stage
without Poetry itself being present in production.

---

## The Dockerfile — Why Two Stages

```
Stage 1 (builder):
  python:3.11-slim
  + Poetry
  + pyproject.toml + poetry.lock
  → runs `poetry install --only main`
  → produces /build/.venv with all packages installed

Stage 2 (runtime):
  python:3.11-slim         ← fresh, clean image
  + libgl1-mesa-glx etc    ← system libs for OpenCV
  + /build/.venv           ← copied from builder (no Poetry, no build tools)
  + src/                   ← application source code
  → runs uvicorn
```

Why bother? The builder stage has pip, Poetry, gcc, and build headers. Those are needed to
compile some Python packages but are ~200MB of dead weight in production. The runtime image
only carries what actually runs. The result is a smaller, more secure production image.

---

## Full Request Flow (End-to-End, Phase 4 Complete)

```
1.  Client sends:   POST /api/v1/recognize
                    Content-Type: multipart/form-data
                    Body: { image: <bytes> }

2.  Uvicorn         Receives TCP connection, parses HTTP, calls FastAPI ASGI handler.

3.  FastAPI Router  `recognition.py` router function fires.
                    Pydantic validates the multipart form — if image is missing → 422.

4.  Dependency Inj. FastAPI resolves `RecognizeFaceUseCase` from the DI container.
                    The use-case was constructed at startup with a `DeepFaceAdapter`
                    injected as `FaceRecognitionPort`.

5.  Use-Case        `RecognizeFaceUseCase.execute(image_bytes)` is called.
                    It calls `self.face_recognition_port.recognize(image_bytes)`.
                    It applies business rules (e.g., filter results below threshold).
                    It returns a list of `RecognitionResult` domain entities.

6.  Adapter         `DeepFaceAdapter.recognize(image_bytes)` is running.
                    *** This is CPU-bound. It runs inside asyncio.to_thread() ***
                    so it does not block the FastAPI event loop.
                    Internally: decode bytes → numpy array via OpenCV →
                    call DeepFace.find() → parse results → return domain entities.

7.  FastAPI Router  Receives the domain entities from the use-case.
                    Maps them to `RecognitionResponseSchema` (Pydantic output model).
                    FastAPI serializes to JSON.

8.  Client receives: 200 OK
                     { "results": [ { "identity": "...", "confidence": 0.97, ... } ] }
```

### Why `asyncio.to_thread` at step 6?

FastAPI runs on an async event loop (single thread by default). If DeepFace runs for 800ms
on the CPU, it will **freeze the entire event loop** for those 800ms — no other request can
be handled. `asyncio.to_thread()` offloads the call to a thread pool worker, freeing the event
loop immediately. The await resumes when the thread is done.

---

## Model Lifecycle

Models (neural network weights) are heavy — ArcFace is ~400MB. Loading them per-request is
catastrophically slow. The correct pattern:

1. At **startup** (inside `lifespan()`), instantiate the adapter and call `adapter.load()`.
   This downloads/reads weights into memory once.
2. Store the loaded adapter on `app.state`.
3. FastAPI's DI system serves this pre-loaded instance to every request.
4. At **shutdown** (after `yield` in `lifespan()`), release GPU memory / close handles.

This is why `main.py` has a commented hook inside `lifespan()` — that is the exact insertion
point for Phase 3.

---

## Environment Variables Reference

| Variable | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `development` | Controls feature flags (docs visibility, etc.) |
| `DEBUG` | `false` | Enables `/docs` and `/redoc` endpoints |
| `LOG_LEVEL` | `INFO` | Uvicorn + app logging verbosity |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `WORKERS` | `1` | Uvicorn worker processes (keep at 1 if sharing model state) |
| *(Phase 3)* `MODEL_BACKEND` | `deepface` | Which AI adapter to wire up |
| *(Phase 3)* `FACE_RECOGNITION_MODEL` | `ArcFace` | Model passed to DeepFace |
| *(Phase 3)* `MODEL_CACHE_DIR` | `/app/.model_cache` | Where weights are stored |

---

## What Has NOT Been Built Yet

| Phase | Agent | What it delivers |
|---|---|---|
| Phase 2 | DomainAgent | `Face` entity, `RecognitionResult` entity, `FaceRecognitionPort` ABC, `FaceRepositoryPort` ABC |
| Phase 3 | AdapterAgent | `DeepFaceAdapter`, `OpenCVImageDecoder`, DB repository implementation |
| Phase 4 | ApiAgent | `POST /api/v1/recognize` router, request/response schemas, DI wiring |
