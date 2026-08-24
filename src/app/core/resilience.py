"""Circuit breaker and retry logic for resilient inference.

Implements the Circuit Breaker pattern to prevent cascading failures
when ML models encounter repeated errors (OOM, corrupt weights, etc.).

States:
    CLOSED  → Normal operation, requests pass through
    OPEN    → Too many failures, requests are rejected immediately
    HALF_OPEN → Recovery probe: one request allowed through to test
"""

import asyncio
import time
from enum import StrEnum

from app.observability.logger import get_logger

logger = get_logger(__name__)


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Async-safe circuit breaker for wrapping inference calls.

    Args:
        name: Identifier for this circuit (used in logging/metrics).
        failure_threshold: Number of consecutive failures before opening.
        recovery_timeout: Seconds to wait before transitioning to half-open.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state, accounting for recovery timeout."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    async def call(self, func, *args, **kwargs):
        """Execute a function through the circuit breaker.

        Args:
            func: Async or sync callable to protect.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            The result of func(*args, **kwargs).

        Raises:
            CircuitOpenError: If the circuit is open and not ready for recovery.
        """
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                logger.warning("Circuit open, rejecting request", circuit=self.name)
                from app.core.exceptions import CircuitOpenError

                raise CircuitOpenError(self.name)

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._on_success()
            return result

        except Exception as exc:
            await self._on_failure()
            raise exc

    async def _on_success(self) -> None:
        """Reset failure count on successful call."""
        async with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    async def _on_failure(self) -> None:
        """Increment failure count and potentially open the circuit."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.error(
                    "Circuit breaker opened",
                    circuit=self.name,
                    failures=self._failure_count,
                )

    async def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info("Circuit breaker manually reset", circuit=self.name)
