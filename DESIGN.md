# Design Write-Up

## Approach (200 words)

This service uses a **pipeline architecture** to process raw audio through discrete,
composable stages: validation → codec transcoding → noise reduction → feature extraction
→ attribute inference. Each stage is independently testable and replaceable.

**Model choices:**

- **Attribute Inference**: We use a fine-tuned Wav2Vec2 model (`audeering/wav2vec2-large-robust-24-ft-age-gender`) to predict both age and gender simultaneously. Wav2Vec2 captures deep prosodic and acoustic features that strongly correlate with demographics. By using a single, unified transformer architecture for both tasks, we ensure highly accurate and intrinsically correlated predictions compared to disjoint linear classification heads. This avoids fabricating custom heads and relies on a proven, robust foundation for audio analysis.

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
