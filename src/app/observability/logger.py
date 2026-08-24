"""Contextual structured logger.

Wraps structlog to provide consistent, context-enriched logging
across the application. All loggers include the request ID
when available.
"""

import structlog


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structured logger.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A bound structlog logger instance.
    """
    return structlog.get_logger(name)
