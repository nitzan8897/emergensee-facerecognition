import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers.faces import router as faces_router
from config import get_settings
from dependencies import _get_deepface_adapter

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.value,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    )
    logger.info(
        "Starting up %s v%s [%s]",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )

    await _get_deepface_adapter().warm_up()

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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(faces_router)

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
