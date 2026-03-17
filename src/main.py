import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from emergensee.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.value,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    )

    logger.info("Starting up %s v%s [%s]", settings.app_name, settings.app_version, settings.environment)

    yield

    logger.info("Shutting down %s. Releasing resources.", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    @app.get("/health", tags=["ops"], summary="Liveness probe")
    async def health_check() -> JSONResponse:
        return JSONResponse(
            content={
                "status": "ok",
                "service": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
            }
        )

    return app


app = create_app()
