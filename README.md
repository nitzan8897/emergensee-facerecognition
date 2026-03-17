# EmergenSee — Face Recognition Microservice

A production-ready AI microservice for real-time face detection and recognition, built with **FastAPI** and **Hexagonal Architecture**.

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| AI / CV | OpenCV, DeepFace *(Phase 3)* |
| Packaging | Poetry |
| Linting | Ruff + mypy (strict) |
| Runtime | Docker (multi-stage) |

## Quick Start

```bash
# 1. Copy environment config
cp .env.example .env

# 2. Run with Docker Compose
docker compose up --build

# 3. Health check
curl http://localhost:8000/health
```

## Development

```bash
# Install dependencies
poetry install

# Run locally
poetry run uvicorn emergensee.main:app --reload --host 0.0.0.0 --port 8000

# Lint & type-check
poetry run ruff check src/
poetry run mypy src/

# Tests
poetry run pytest
```

## Project Structure

```
src/emergensee/
├── domain/        # Core logic — no external dependencies
│   ├── entities/  # Business objects
│   └── ports/     # Abstract interfaces (ABCs)
├── application/   # Use-cases / orchestration
├── adapters/      # Concrete implementations (AI, DB)
└── api/           # FastAPI routers and Pydantic schemas
```

## Architecture

Strict **Hexagonal (Ports & Adapters)** architecture. The domain never imports from adapters. AI providers are swappable behind an interface.
