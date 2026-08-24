"""POST /analyze and POST /v1/analyze — audio ingestion, FFmpeg normalization, Silero VAD, Quality Assessment, ML Input Prep, Speech Encoder, Gender & Age Classification endpoint.

Accepts an audio file via multipart upload, validates input,
decodes & normalizes to 16kHz mono float32 PCM using FFmpeg,
runs Silero Voice Activity Detection (VAD) with segment refinement,
assesses multi-signal audio quality, prepares deterministic model-ready ML input waveform window,
extracts 192-dim speech embedding vector via SpeechBrain ECAPA-TDNN,
predicts gender classification (male, female, unknown),
estimates age bracket classification (18-30, 31-45, 46-60, 60+, unknown),
and returns normalized metadata with VAD, quality, gender, and age metrics.
"""

import time
from fastapi import APIRouter, File, UploadFile

from app.api.v1.schemas.responses import AgeBracketResponse, AudioMetadataResponse, GenderResponse, SpeechSegmentSchema
from app.audio.codec import AudioCodec
from app.audio.preprocessor import AudioPreprocessor
from app.audio.quality import AudioQualityAssessor
from app.audio.validator import AudioValidator
from app.audio.vad import VoiceActivityDetector
from app.config.settings import get_settings
from app.inference.speech_encoder import SpeechEncoder
from app.inference.strategies.age_estimator import AgeEstimator
from app.inference.strategies.gender_classifier import GenderClassifier
from app.observability.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/analyze",
    response_model=AudioMetadataResponse,
    summary="Analyze, normalize, Silero VAD, quality assessment, ML input prep, speech embedding, gender & age estimation",
    description=(
        "Accepts an audio file via multipart upload, validates input constraints, "
        "decodes and normalizes the payload into 16 kHz, mono, PCM float32 waveform data via FFmpeg, "
        "runs Silero Voice Activity Detection (VAD) with segment refinement (gap merging & fragment filtering), "
        "evaluates multi-signal audio quality, prepares deterministic model-ready ML input waveform window, "
        "extracts fixed 192-dimensional speech embedding vector via SpeechBrain ECAPA-TDNN, "
        "predicts gender and age bracket classification with confidence thresholding, and classifies audio quality with transparent reasoning."
    ),
)
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, M4A, etc.)"),
) -> AudioMetadataResponse:
    """Ingest, validate, decode via FFmpeg, run Silero VAD, evaluate quality, prepare ML input, extract embedding, predict gender & age, and return metadata.

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

    # 5. Silero Voice Activity Detection (VAD) with Refinement Stage
    vad_detector = VoiceActivityDetector(
        min_speech_ratio=settings.vad_min_speech_ratio,
        min_speech_duration_ms=settings.vad_min_speech_duration_ms,
        merge_gap_ms=settings.vad_merge_gap_ms,
        min_segment_duration_ms=settings.vad_min_segment_duration_ms,
        silero_threshold=settings.silero_vad_threshold,
    )
    vad_res = vad_detector.detect(segment.waveform, sample_rate=segment.sample_rate)

    # 6. Multi-Signal Audio Quality Assessment
    quality_assessor = AudioQualityAssessor(
        snr_good_threshold_db=settings.snr_good_threshold_db,
        snr_degraded_threshold_db=settings.snr_degraded_threshold_db,
        clipping_max_ratio=settings.clipping_max_ratio,
        min_peak_amplitude=settings.min_peak_amplitude,
    )
    qual_res = quality_assessor.assess(segment.waveform, vad_res, sample_rate=segment.sample_rate)

    # 7. ML Input Preparation Stage
    prepared_input = AudioPreprocessor.prepare(
        waveform=segment.waveform,
        vad_result=vad_res,
        quality_result=qual_res,
        target_duration_seconds=settings.ml_target_duration_seconds,
        sample_rate=segment.sample_rate,
    )

    # 8. Pretrained Speech Encoder Inference Stage (192-dim vector)
    speech_encoder = SpeechEncoder(model_name=settings.speech_encoder_model_name)
    embedding_res = speech_encoder.encode(prepared_input)

    # 9. Gender Classification Inference Stage
    gender_classifier = GenderClassifier(
        model_name="gender_classifier",
        device=settings.model_device,
        confidence_threshold=settings.gender_confidence_threshold,
    )
    gender_res = gender_classifier.predict_embedding(embedding_res)

    # 10. Age Bracket Estimation Inference Stage
    age_estimator = AgeEstimator(
        model_name="age_estimator",
        device=settings.model_device,
        confidence_threshold=settings.age_confidence_threshold,
    )
    age_res = age_estimator.predict_embedding(embedding_res)

    processing_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "Audio analysis completed successfully",
        filename=file.filename,
        duration_ms=segment.duration_ms,
        speech_duration_ms=vad_res.speech_duration_ms,
        audio_quality=qual_res.audio_quality.value,
        snr_db=qual_res.snr_db,
        vad_confidence=vad_res.vad_confidence,
        gender_prediction=gender_res.prediction.value,
        age_prediction=age_res.prediction.value,
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
            prediction=gender_res.prediction,
            confidence=gender_res.confidence,
            probabilities=gender_res.probabilities,
            inference_ms=gender_res.inference_ms,
        ),
        age_bracket=AgeBracketResponse(
            prediction=age_res.prediction,
            confidence=age_res.confidence,
            probabilities=age_res.probabilities,
            inference_ms=age_res.inference_ms,
        ),
    )
