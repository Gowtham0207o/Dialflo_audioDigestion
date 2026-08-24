"""Request tracing for correlating logs across the pipeline.

Provides span-based tracing to measure individual stage durations
within a request, useful for identifying bottlenecks.
"""

import time
from contextlib import contextmanager
from collections.abc import Generator
from dataclasses import dataclass, field

from app.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Span:
    """A single timed span within a request trace.

    Attributes:
        name: Span name (e.g., 'decode', 'infer_gender').
        start_time: Start timestamp (monotonic).
        end_time: End timestamp (monotonic).
        metadata: Additional context for the span.
    """

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        """Span duration in milliseconds."""
        return int((self.end_time - self.start_time) * 1000)


class RequestTrace:
    """Collects timing spans for a single request.

    Usage:
        trace = RequestTrace(request_id="abc-123")
        with trace.span("decode"):
            do_decode()
        with trace.span("infer"):
            do_inference()
        trace.log_summary()
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **metadata) -> Generator[Span, None, None]:
        """Context manager that times a named span.

        Args:
            name: Span name for identification.
            **metadata: Additional context to attach to the span.

        Yields:
            The Span object being recorded.
        """
        s = Span(name=name, start_time=time.perf_counter(), metadata=metadata)
        try:
            yield s
        finally:
            s.end_time = time.perf_counter()
            self._spans.append(s)

    def log_summary(self) -> None:
        """Log a summary of all spans in this trace."""
        total_ms = sum(s.duration_ms for s in self._spans)
        breakdown = {s.name: s.duration_ms for s in self._spans}

        logger.info(
            "Request trace summary",
            request_id=self.request_id,
            total_ms=total_ms,
            breakdown=breakdown,
        )

    @property
    def spans(self) -> list[Span]:
        """All recorded spans."""
        return list(self._spans)
