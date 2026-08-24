# Design Write-Up

## Approach (200 words)

This service uses a **pipeline architecture** to process raw audio through discrete,
composable stages: validation → codec transcoding → noise reduction → feature extraction
→ attribute inference. Each stage is independently testable and replaceable.

**Model choices:**

- **Gender classification**: SpeechBrain's ECAPA-TDNN speaker embeddings
  (`spkrec-ecapa-voxceleb`) produce 192-dimensional embeddings that are inherently
  gender-discriminative. A lightweight linear head classifies gender with sub-100ms
  inference on CPU. Chosen over end-to-end Whisper because we need speaker attributes,
  not transcription.

- **Age estimation**: wav2vec2 fine-tuned embeddings capture prosodic features (pitch,
  jitter, shimmer) that correlate with age. A classification head maps to four brackets.

- **Audio quality**: SNR estimation via signal/noise energy ratio combined with Voice
  Activity Detection (VAD) ratio. Pure signal processing — no ML overhead.

**Improvement roadmap:**
With more time, I would add ONNX Runtime quantization for 3x inference speedup,
ensemble multiple models with confidence-weighted voting, and add a streaming VAD
to skip silence chunks entirely.

**Scaling to 1,000 concurrent calls:**
Horizontal scaling behind a load balancer, GPU inference with batched requests via
TorchServe, model sharding across workers, and a Redis-backed request queue for
back-pressure management.
