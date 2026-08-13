from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from aegis.api.dependencies import PredictorFactory, default_predictor_factory
from aegis.api.metrics import create_metrics
from aegis.api.routes import router
from aegis.api.schemas import ErrorResponse
from aegis.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)r}',
)
logger = logging.getLogger("aegis.api")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def create_app(predictor_factory: PredictorFactory | None = None) -> FastAPI:
    """Factory instead of a bare module-level app so tests can inject
    MockPredictor without ever importing torch (design.md D12 step 3)."""
    factory = predictor_factory or default_predictor_factory
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictors = {}
        app.state.ready = False
        try:
            app.state.predictors = factory()
            app.state.ready = True
            logger.info("predictors loaded: %s", list(app.state.predictors))
            for predictor in app.state.predictors.values():
                app.state.metrics.set_model_info(
                    model=predictor.name, version=predictor.version, macro_f1=predictor.macro_f1
                )
                app.state.metrics.set_ood_enabled(
                    model=predictor.name, enabled=settings.ood_enabled
                )
        except Exception:
            logger.exception("failed to load predictors at startup")
        yield

    app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)
    app.state.metrics = create_metrics()

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        app.state.metrics.record_http_error(
            endpoint=request.url.path, error_type="validation_error"
        )
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error", detail=str(exc.errors()), request_id=_request_id(request)
            ).model_dump(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        app.state.metrics.record_http_error(
            endpoint=request.url.path, error_type=f"http_{exc.status_code}"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="http_error", detail=str(exc.detail), request_id=_request_id(request)
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        logger.exception("unhandled error request_id=%s", request_id)
        app.state.metrics.record_http_error(endpoint=request.url.path, error_type="internal_error")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error", detail="an unexpected error occurred", request_id=request_id
            ).model_dump(),
        )

    app.include_router(router)

    # Registered last so explicit routes above (health, ready, /v1/*, /metrics,
    # /docs) always win — StaticFiles(html=True) only catches what nothing
    # else matched, serving index.html at "/" (design.md D13).
    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()
