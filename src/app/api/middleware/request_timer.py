"""Request timer middleware.

Measures end-to-end request processing time and injects it
into the response headers for observability.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.observability.logger import get_logger

logger = get_logger(__name__)


class RequestTimerMiddleware(BaseHTTPMiddleware):
    """Middleware that measures and logs request processing duration."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Inject timing into response headers
        response.headers["X-Processing-Time-Ms"] = str(duration_ms)

        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            processing_ms=duration_ms,
        )

        return response
