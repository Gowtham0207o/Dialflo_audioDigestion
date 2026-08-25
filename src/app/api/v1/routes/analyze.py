"""POST /analyze and POST /v1/analyze — audio ingestion, FFmpeg normalization, Silero VAD, Quality Assessment, ML Input Prep, Attribute Inference endpoint.

Accepts an audio file via multipart upload, validates input,
decodes & normalizes to 16kHz mono float32 PCM using FFmpeg,
runs Silero Voice Activity Detection (VAD) with segment refinement,
assesses multi-signal audio quality, prepares deterministic model-ready ML input waveform window,
runs attribute inference via the active model pipeline (ensemble or single model),
and returns normalized metadata with VAD, quality, gender, and age metrics.
"""

import time
from fastapi import APIRouter, File, Request, UploadFile

from app.api.v1.schemas.responses import (
    AgeBracketResponse, 
    AudioMetadataResponse, 
    GenderResponse, 
    SpeechSegmentSchema,
    SimplifiedAnalyzeResponse,
    SimplifiedGenderResponse,
    SimplifiedAgeBracketResponse
)
import uuid
from app.audio.codec import AudioCodec
from app.audio.preprocessor import AudioPreprocessor
from app.audio.quality import AudioQualityAssessor
from app.audio.validator import AudioValidator
from app.audio.vad import VoiceActivityDetector
from app.config.settings import get_settings
from app.core.enums import Gender, AgeBracket
from app.inference.attribute_model import AttributeModel
from app.inference.chunkformer import ChunkFormerModel
from app.observability.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


async def _process_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, M4A, etc.)"),
) -> AudioMetadataResponse:
    """Ingest, validate, decode via FFmpeg, run Silero VAD, evaluate quality, prepare ML input, run attribute inference, and return metadata.

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
    t_decode = time.perf_counter()
    segment = AudioCodec.transcode_to_wav(audio_bytes, target_sample_rate=16000)
    decode_ms = int((time.perf_counter() - t_decode) * 1000)

    # 4. Validate duration
    AudioValidator.validate_duration(segment)

    # 5. Silero Voice Activity Detection (VAD) with Refinement Stage
    t_vad = time.perf_counter()
    vad_detector = VoiceActivityDetector(
        min_speech_ratio=settings.vad_min_speech_ratio,
        min_speech_duration_ms=settings.vad_min_speech_duration_ms,
        merge_gap_ms=settings.vad_merge_gap_ms,
        min_segment_duration_ms=settings.vad_min_segment_duration_ms,
        silero_threshold=settings.silero_vad_threshold,
    )
    vad_res = vad_detector.detect(segment.waveform, sample_rate=segment.sample_rate)
    vad_ms = int((time.perf_counter() - t_vad) * 1000)

    # 6. Multi-Signal Audio Quality Assessment
    t_quality = time.perf_counter()
    quality_assessor = AudioQualityAssessor(
        snr_good_threshold_db=settings.snr_good_threshold_db,
        snr_degraded_threshold_db=settings.snr_degraded_threshold_db,
        clipping_max_ratio=settings.clipping_max_ratio,
        min_peak_amplitude=settings.min_peak_amplitude,
    )
    qual_res = quality_assessor.assess(segment.waveform, vad_res, sample_rate=segment.sample_rate)
    quality_ms = int((time.perf_counter() - t_quality) * 1000)

    # 7. ML Input Preparation Stage
    t_prep = time.perf_counter()
    prepared_input = AudioPreprocessor.prepare(
        waveform=segment.waveform,
        vad_result=vad_res,
        quality_result=qual_res,
        target_duration_seconds=settings.ml_target_duration_seconds,
        sample_rate=segment.sample_rate,
    )
    ml_prep_ms = int((time.perf_counter() - t_prep) * 1000)

    # 8. Attribute Inference via active model pipeline
    t_inference = time.perf_counter()
    attribute_model: AttributeModel = getattr(request.app.state, "attribute_model", None)
    if attribute_model is None:
        # Fallback: create ChunkFormerModel if not set on app.state
        attribute_model = ChunkFormerModel()
        attribute_model.load()

    attr_result = attribute_model.predict(prepared_input)
    inference_ms = int((time.perf_counter() - t_inference) * 1000)

    total_ms = int((time.perf_counter() - t0) * 1000)
    model_used = attr_result.model_name

    timing_breakdown = {
        "decode_ms": decode_ms,
        "vad_ms": vad_ms,
        "quality_ms": quality_ms,
        "ml_prep_ms": ml_prep_ms,
        "inference_ms": inference_ms,
        "total_ms": total_ms,
    }

    logger.info(
        "Audio analysis completed successfully",
        filename=file.filename,
        duration_ms=segment.duration_ms,
        speech_duration_ms=vad_res.speech_duration_ms,
        audio_quality=qual_res.audio_quality.value,
        snr_db=qual_res.snr_db,
        vad_confidence=vad_res.vad_confidence,
        gender_prediction=attr_result.gender.value,
        age_prediction=attr_result.age_bracket.value,
        model_used=model_used,
        processing_ms=total_ms,
        timing_breakdown=timing_breakdown,
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
        processing_ms=total_ms,
        speech_duration_seconds=vad_res.speech_duration_seconds,
        speech_duration_ms=vad_res.speech_duration_ms,
        speech_ratio=vad_res.speech_ratio,
        speech_segments=[
            SpeechSegmentSchema(
                start_seconds=s.start_seconds,
                end_seconds=s.end_seconds,
                confidence=s.confidence,
            )
            for s in vad_res.speech_segments
        ],
        is_speech_sufficient=vad_res.is_speech_sufficient,
        vad_confidence=vad_res.vad_confidence,
        audio_quality=qual_res.audio_quality,
        snr_db=qual_res.snr_db,
        peak_amplitude=qual_res.peak_amplitude,
        clipping_ratio=qual_res.clipping_ratio,
        rms_energy=qual_res.rms_energy,
        speech_energy_ratio=qual_res.speech_energy_ratio,
        quality_reasoning=qual_res.quality_reasoning,
        gender=GenderResponse(
            prediction=attr_result.gender,
            confidence=attr_result.gender_confidence,
            probabilities=attr_result.gender_probabilities,
            inference_ms=attr_result.model_inference_ms,
        ),
        age_bracket=AgeBracketResponse(
            prediction=attr_result.age_bracket,
            confidence=attr_result.age_confidence,
            probabilities=attr_result.age_probabilities,
            inference_ms=attr_result.model_inference_ms,
        ),
        model_used=model_used,
        timing_breakdown=timing_breakdown,
    )


@router.post(
    "/v1/analyze",
    response_model=AudioMetadataResponse,
    summary="[v1] Analyze audio: full metadata response",
    description="V1 legacy endpoint returning the complete audio metadata, VAD segments, and diagnostic information.",
)
async def analyze_audio_v1(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, M4A, etc.)"),
) -> AudioMetadataResponse:
    """Ingest, process, and return the complete AudioMetadataResponse."""
    return await _process_audio(request, file)


@router.post(
    "/analyse",
    response_model=SimplifiedAnalyzeResponse,
    summary="Analyze audio: simplified response",
    description="V2 endpoint returning a simplified summary of gender, age bracket, and audio quality.",
)
async def analyze_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, M4A, etc.)"),
) -> SimplifiedAnalyzeResponse:
    """Ingest, process, and return a simplified AnalyzeResponse."""
    metadata = await _process_audio(request, file)
    
    return SimplifiedAnalyzeResponse(
        contact_id=str(uuid.uuid4()),
        gender=SimplifiedGenderResponse(
            prediction=metadata.gender.prediction,
            confidence=metadata.gender.confidence
        ),
        age_bracket=SimplifiedAgeBracketResponse(
            prediction=metadata.age_bracket.prediction,
            confidence=metadata.age_bracket.confidence
        ),
        processing_ms=metadata.processing_ms,
        audio_quality=metadata.audio_quality
    )

