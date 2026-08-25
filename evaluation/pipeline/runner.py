"""Production inference integration runner.

Runs the exact same pipeline used in production on the evaluation samples.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from app.audio.codec import AudioCodec
from app.audio.preprocessor import AudioPreprocessor
from app.audio.quality import AudioQualityAssessor
from app.audio.vad import VoiceActivityDetector
from app.inference.chunkformer import ChunkFormerModel
from app.inference.custom_encoder_model import CustomEncoderModel
from app.inference.ensemble_model import EnsembleModel
from evaluation.adapter.common_voice import CVSample

logger = logging.getLogger(__name__)


@dataclass
class EvalRecord:
    """A single evaluation result record."""
    sample_id: str
    audio_path: str
    gt_gender: str | None
    gt_age: str | None
    model_name: str
    pred_gender: str
    pred_age: str
    gender_confidence: float
    age_confidence: float
    gender_probs: dict[str, float]
    age_probs: dict[str, float]
    audio_quality: str
    snr_db: float
    is_valid: bool
    preprocess_ms: int
    inference_ms: int
    total_ms: int
    error: str | None


class PipelineRunner:
    """Runner that executes the production pipeline for evaluation."""

    def __init__(self, model_mode: str):
        self.model_mode = model_mode
        self.models = []
        
        # Load models once
        logger.info(f"Loading models for mode: {model_mode}")
        
        cf_model = None
        ce_model = None
        
        if model_mode in ("chunkformer", "all"):
            cf_model = ChunkFormerModel()
            cf_model.load()
            if model_mode == "chunkformer":
                self.models.append(cf_model)
                
        if model_mode in ("custom", "all"):
            ce_model = CustomEncoderModel()
            ce_model.load()
            if model_mode == "custom":
                self.models.append(ce_model)
                
        if model_mode in ("ensemble", "all"):
            # If "all", we reuse the already loaded cf_model and ce_model
            m1 = cf_model if cf_model else ChunkFormerModel()
            m2 = ce_model if ce_model else CustomEncoderModel()
            
            ens_model = EnsembleModel(models=[m1, m2])
            ens_model.load()
            
            if model_mode == "ensemble":
                self.models.append(ens_model)
            elif model_mode == "all":
                # For "all", evaluate all three
                self.models = [cf_model, ce_model, ens_model]

        # Initialize pipeline components
        self.vad_detector = VoiceActivityDetector()
        self.quality_assessor = AudioQualityAssessor()

    def run(self, samples: list[CVSample]) -> list[EvalRecord]:
        """Run the pipeline on a list of samples."""
        records = []
        
        total = len(samples)
        successful = 0
        failed = 0
        skipped = 0

        for i, sample in enumerate(samples):
            if sample.skip_reason:
                skipped += 1
                continue
                
            logger.info(f"Processing sample {i+1}/{total}: {sample.audio_path.name}")
            
            t0 = time.perf_counter()
            error = None
            is_valid = False
            
            # Preprocessing
            try:
                if sample.audio_bytes is not None:
                    audio_bytes = sample.audio_bytes
                else:
                    audio_bytes = sample.audio_path.read_bytes()
                segment = AudioCodec.transcode_to_wav(audio_bytes)
                vad_res = self.vad_detector.detect(segment.waveform, segment.sample_rate)
                qual_res = self.quality_assessor.assess(segment.waveform, vad_res, segment.sample_rate)
                prep_input = AudioPreprocessor.prepare(
                    waveform=segment.waveform,
                    vad_result=vad_res,
                    quality_result=qual_res,
                    sample_rate=segment.sample_rate
                )
                
                audio_quality = qual_res.audio_quality.value
                snr_db = qual_res.snr_db
                is_valid = prep_input.is_prepared_valid
                
            except Exception as e:
                logger.warning(f"Error during preprocessing of {sample.audio_path.name}: {e}")
                error = str(e)
                audio_quality = "unknown"
                snr_db = 0.0
                prep_input = None
                
            preprocess_ms = int((time.perf_counter() - t0) * 1000)
            
            # If preprocessing failed, create a failed record for each model
            if error or not is_valid:
                failed += 1
                for model in self.models:
                    records.append(EvalRecord(
                        sample_id=sample.speaker_id,
                        audio_path=str(sample.audio_path),
                        gt_gender=sample.gender,
                        gt_age=sample.age_bracket,
                        model_name=getattr(model, 'model_name', type(model).__name__),
                        pred_gender="unknown",
                        pred_age="unknown",
                        gender_confidence=0.0,
                        age_confidence=0.0,
                        gender_probs={},
                        age_probs={},
                        audio_quality=audio_quality,
                        snr_db=snr_db,
                        is_valid=False,
                        preprocess_ms=preprocess_ms,
                        inference_ms=0,
                        total_ms=preprocess_ms,
                        error=error or "Invalid prepared input"
                    ))
                continue

            successful += 1
            
            # Inference (for each model)
            for model in self.models:
                t1 = time.perf_counter()
                model_error = None
                try:
                    res = model.predict(prep_input)
                except Exception as e:
                    logger.warning(f"Error during inference for {sample.audio_path.name} with model {model.model_name}: {e}")
                    model_error = str(e)
                    res = None
                    
                inference_ms = int((time.perf_counter() - t1) * 1000)
                total_ms = preprocess_ms + inference_ms
                
                if model_error or not res or not res.is_valid:
                    records.append(EvalRecord(
                        sample_id=sample.speaker_id,
                        audio_path=str(sample.audio_path),
                        gt_gender=sample.gender,
                        gt_age=sample.age_bracket,
                        model_name=getattr(model, 'model_name', type(model).__name__),
                        pred_gender="unknown",
                        pred_age="unknown",
                        gender_confidence=0.0,
                        age_confidence=0.0,
                        gender_probs={},
                        age_probs={},
                        audio_quality=audio_quality,
                        snr_db=snr_db,
                        is_valid=False,
                        preprocess_ms=preprocess_ms,
                        inference_ms=inference_ms,
                        total_ms=total_ms,
                        error=model_error or "Inference returned invalid"
                    ))
                else:
                    records.append(EvalRecord(
                        sample_id=sample.speaker_id,
                        audio_path=str(sample.audio_path),
                        gt_gender=sample.gender,
                        gt_age=sample.age_bracket,
                        model_name=res.model_name,
                        pred_gender=res.gender.value,
                        pred_age=res.age_bracket.value,
                        gender_confidence=res.gender_confidence,
                        age_confidence=res.age_confidence,
                        gender_probs=res.gender_probabilities,
                        age_probs=res.age_probabilities,
                        audio_quality=audio_quality,
                        snr_db=snr_db,
                        is_valid=True,
                        preprocess_ms=preprocess_ms,
                        inference_ms=res.model_inference_ms, # Use internal model latency
                        total_ms=preprocess_ms + res.model_inference_ms,
                        error=None
                    ))
                    
        logger.info(f"Pipeline complete. Total: {total}, Successful: {successful}, Failed: {failed}, Skipped: {skipped}")
        return records
