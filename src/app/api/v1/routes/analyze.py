"""POST /analyze and POST /v1/analyze — audio ingestion, FFmpeg normalization, VAD & Quality Assessment endpoint.

Accepts an audio file via multipart upload, validates input,
decodes & normalizes to 16kHz mono float32 PCM using FFmpeg,
runs Voice Activity Detection (VAD) and Audio Quality Assessment,
and returns normalized metadata with VAD metrics and audio_quality flag.
"""

import time
from fastapi import APIRouter, File, UploadFile

from app.api.v1.schemas.responses import AudioMetadataResponse, SpeechSegmentSchema
from app.audio.codec import AudioCodec
from app.audio.quality import AudioQualityAssessor
from app.audio.validator import AudioValidator
from app.audio.vad import VoiceActivityDetector
from app.config.settings import get_settings
from app.observability.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/analyze",
    response_model=AudioMetadataResponse,
    summary="Analyze, normalize, VAD, and quality assessment",
    description=(
        "Accepts an audio file via multipart upload, validates input constraints, "
        "decodes and normalizes the payload into 16 kHz, mono, PCM float32 waveform data via FFmpeg, "
        "runs Voice Activity Detection (VAD), assesses audio quality (SNR, peak amplitude, clipping), "
        "and classifies audio quality as good, degraded, or insufficient with clear reasoning."
    ),
)
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, M4A, etc.)"),
) -> AudioMetadataResponse:
    """Ingest, validate, decode via FFmpeg, run VAD, evaluate audio quality, and return metadata.

    The audio payload is processed entirely in memory — no files touch disk.
    """
    t0 = time.perf_counter()
    settings = get_settings()

    # 1. Read raw bytes into memory (PII: never stored on disk)
    audio_bytes = await file.read()

    logger.info(
        "Ingesting audio file",
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=len(audio_bytes),
    )

    # 2. Input Validation
    AudioValidator.validate_file_size(audio_bytes, max_size_mb=50)
    AudioValidator.validate_content_type(file.content_type)
    AudioValidator.validate_file_extension(file.filename)

    # 3. FFmpeg Decoding & Normalization to 16 kHz mono float32 PCM
    segment = AudioCodec.transcode_to_wav(audio_bytes, target_sample_rate=16000)

    # 4. Validate duration
    AudioValidator.validate_duration(segment)

    # 5. Chunk 2: Voice Activity Detection (VAD)
    vad_detector = VoiceActivityDetector(
        min_speech_ratio=settings.vad_min_speech_ratio,
        min_speech_duration_ms=settings.vad_min_speech_duration_ms,
    )
    vad_res = vad_detector.detect(segment.waveform, sample_rate=segment.sample_rate)

    # 6. Chunk 3: Audio Quality Assessment
    quality_assessor = AudioQualityAssessor(
        snr_good_threshold_db=settings.snr_good_threshold_db,
        snr_degraded_threshold_db=settings.snr_degraded_threshold_db,
        clipping_max_ratio=settings.clipping_max_ratio,
    )
    qual_res = quality_assessor.assess(segment.waveform, vad_res, sample_rate=segment.sample_rate)

    processing_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "Audio analysis completed successfully",
        filename=file.filename,
        duration_ms=segment.duration_ms,
        speech_duration_ms=vad_res.speech_duration_ms,
        audio_quality=qual_res.audio_quality.value,
        snr_db=qual_res.snr_db,
        processing_ms=processing_ms,
    )

    return AudioMetadataResponse(
        duration=round(segment.duration_seconds, 3),
        duration_ms=segment.duration_ms,
        duration_seconds=round(segment.duration_seconds, 3),
        sample_rate=segment.sample_rate,
        channels=segment.channels,
        samples=segment.num_samples,
        total_samples=segment.num_samples,
        original_format=segment.original_format,
        processing_ms=processing_ms,
        speech_duration_seconds=vad_res.speech_duration_seconds,
        speech_duration_ms=vad_res.speech_duration_ms,
        speech_ratio=vad_res.speech_ratio,
        speech_segments=[
            SpeechSegmentSchema(start_seconds=s.start_seconds, end_seconds=s.end_seconds)
            for s in vad_res.speech_segments
        ],
        is_speech_sufficient=vad_res.is_speech_sufficient,
        audio_quality=qual_res.audio_quality,
        snr_db=qual_res.snr_db,
        peak_amplitude=qual_res.peak_amplitude,
        clipping_ratio=qual_res.clipping_ratio,
        quality_reasoning=qual_res.quality_reasoning,
    )
