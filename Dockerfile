# =============================================================================
# Stage 1 — Builder: install Poetry and resolve/export dependencies
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /build

COPY pyproject.toml poetry.lock* ./

RUN poetry install --only main --no-root

# =============================================================================
# Stage 2 — Runtime: lean image with only what is needed at runtime
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src"

# System dependencies required by OpenCV (libGL) and other CV libs.
# Installed here (not in builder) so they are present in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the resolved virtual environment from the builder stage.
COPY --from=builder /build/.venv /app/.venv

# Make the venv the active Python environment.
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY src/ ./src/

# Placeholder directory for downloaded AI model weights.
RUN mkdir -p /app/.model_cache

EXPOSE 8000

RUN addgroup --system emergensee && adduser --system --ingroup emergensee emergensee
USER emergensee

CMD ["uvicorn", "emergensee.main:app", "--host", "0.0.0.0", "--port", "8000"]
