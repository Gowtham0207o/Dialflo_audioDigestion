"""POST /v1/analyze — audio attribute inference endpoint.

Accepts multipart audio upload or raw audio stream and returns
gender, age bracket, and audio quality predictions.
"""

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_pipeline
from app.api.v1.schemas.responses import AnalyzeResponse
from app.pipeline.orchestrator import AnalysisPipeline
from app.observability.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze audio for caller attributes",
    description=(
        "Accepts a streaming or chunked audio input and returns estimated "
        "attributes for the contact person (gender, age bracket) with "
        "confidence scores and audio quality assessment."
    ),
)
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, etc.)"),
    pipeline: AnalysisPipeline = Depends(get_pipeline),
) -> AnalyzeResponse:
    """Process uploaded audio and return attribute predictions.

    The audio is processed entirely in memory — no data is persisted
    to disk or any external storage.
    """
    # Read audio bytes into memory (PII: never written to disk)
    audio_bytes = await file.read()

    logger.info(
        "Audio received for analysis",
        content_type=file.content_type,
        size_bytes=len(audio_bytes),
        filename=file.filename,
    )

    # Run the full analysis pipeline
    result = await pipeline.analyze(
        audio_bytes=audio_bytes,
        content_type=file.content_type,
        filename=file.filename,
    )

    return AnalyzeResponse.from_domain(result)
