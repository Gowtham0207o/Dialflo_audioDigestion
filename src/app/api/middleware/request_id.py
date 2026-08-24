"""Request ID middleware.

Generates or propagates a unique request ID for tracing
requests through logs and downstream services.
"""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

import structlog


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware that ensures every request has a unique trace ID.

    If the client sends an X-Request-ID header, it is propagated.
    Otherwise, a new UUID is generated.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Use client-provided ID or generate a new one
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Bind to structlog context for all downstream log entries
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Store in request state for access in route handlers
        request.state.request_id = request_id

        response = await call_next(request)

        # Echo the request ID back to the client
        response.headers["X-Request-ID"] = request_id

        return response
