#!/usr/bin/env bash
set -e

export PYTHONPATH="$(pwd)/src"
export ENVIRONMENT=production
export DEBUG=true
export LOG_LEVEL=INFO
export HOST=0.0.0.0
export PORT=8000
export WORKERS=1
export MONGO_URI="mongodb://admin:bartar20%40CS@localhost:21771"

exec venv/bin/python -m uvicorn main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
