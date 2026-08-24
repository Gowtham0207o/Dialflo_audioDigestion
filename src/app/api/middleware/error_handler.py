"""Global error handler middleware.

Maps domain exceptions to structured JSON error responses.
Prevents unhandled exceptions from leaking stack traces.
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.exceptions import AudioDigestionError
from app.observability.logger import get_logger

logger = get_logger(__name__)


class GlobalErrorHandler(BaseHTTPMiddleware):
    """Catches all exceptions and returns structured error responses."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            return await call_next(request)

        except AudioDigestionError as exc:
            # Known domain errors — log at warning, return structured response
            logger.warning(
                "Domain error",
                error_type=type(exc).__name__,
                message=exc.message,
                status_code=exc.status_code,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": type(exc).__name__,
                    "message": exc.message,
                    "status_code": exc.status_code,
                },
            )

        except Exception as exc:
            # Unexpected errors — log at error, return generic 500
            logger.error(
                "Unhandled exception",
                error_type=type(exc).__name__,
                message=str(exc),
                path=request.url.path,
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": "An unexpected error occurred",
                    "status_code": 500,
                },
            )
