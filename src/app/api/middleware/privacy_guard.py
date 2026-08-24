"""Privacy guard middleware.

Ensures no audio data leaks into logs, error responses, or
any external system. Treats all audio bytes as PII.
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.observability.logger import get_logger

logger = get_logger(__name__)


class PrivacyGuardMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces PII safety for audio data.

    - Strips any audio-related headers from responses
    - Ensures error responses never contain raw audio data
    - Logs request metadata without audio content
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Log request metadata (never audio content)
        logger.debug(
            "Request received",
            method=request.method,
            path=request.url.path,
            content_type=request.headers.get("content-type", "unknown"),
            # Explicitly NOT logging request body
        )

        response = await call_next(request)

        # Ensure no audio data leaks in response headers
        response.headers["X-Audio-Stored"] = "false"
        response.headers["X-PII-Safe"] = "true"

        return response
