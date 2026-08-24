"""WS /v1/stream — real-time progressive audio analysis.

WebSocket endpoint that accepts audio chunks and emits progressive
predictions as more audio data arrives. (Bonus feature)
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from app.api.dependencies import get_pipeline
from app.api.v1.schemas.responses import StreamEvent
from app.pipeline.stream_handler import StreamingAnalyzer
from app.observability.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.websocket("/stream")
async def stream_audio(
    websocket: WebSocket,
) -> None:
    """WebSocket endpoint for real-time audio analysis.

    Protocol:
        1. Client connects and sends audio chunks as binary frames
        2. Server emits JSON StreamEvent messages with progressive predictions
        3. Client sends a text frame "END" to signal end of stream
        4. Server sends final prediction and closes connection

    Each audio chunk should be ~1 second of 16kHz mono PCM audio.
    """
    await websocket.accept()

    logger.info("WebSocket connection established")

    # StreamingAnalyzer is initialized per-connection
    analyzer = StreamingAnalyzer()

    try:
        while True:
            # Receive audio chunk (binary) or control message (text)
            message = await websocket.receive()

            if "text" in message:
                if message["text"].upper() == "END":
                    # Client signals end of stream — send final result
                    final_event = await analyzer.finalize()
                    await websocket.send_json(final_event.model_dump())
                    break

            elif "bytes" in message:
                # Process audio chunk and emit progressive prediction
                chunk = message["bytes"]
                event = await analyzer.process_chunk(chunk)

                if event is not None:
                    await websocket.send_json(event.model_dump())

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

    except Exception as exc:
        logger.error("WebSocket error", error=str(exc), exc_info=True)
        await websocket.close(code=1011, reason="Internal server error")

    finally:
        # Ensure all audio buffers are cleaned up (PII safety)
        await analyzer.cleanup()
        logger.info("WebSocket connection closed, buffers cleaned")
