# Privacy & PII Handling

## Principle

**Caller audio is PII.** This service processes audio exclusively in-memory and never
persists audio data to disk, database, or any external storage.

## Data Flow Guarantees

### Audio Lifecycle

1. **Ingestion**: Audio bytes are received via HTTP multipart upload or WebSocket frames
2. **Processing**: Audio is held in Python `bytes` / `numpy.ndarray` objects in RAM
3. **Inference**: Feature vectors (MFCC, mel-spectrograms) are extracted and passed to models
4. **Response**: Only structured predictions (gender, age bracket, confidence scores) are returned
5. **Cleanup**: All audio references are dereferenced when the request handler returns;
   Python's garbage collector reclaims the memory immediately

### What Is NEVER Stored

- Raw audio bytes
- Decoded waveforms
- Temporary audio files on disk
- Audio features or embeddings beyond request scope

### What IS Returned

- Gender prediction + confidence (float)
- Age bracket prediction + confidence (float)
- Audio quality flag (enum)
- Processing duration (milliseconds)
- Request-scoped UUID (not linked to caller identity)

## Logging

- **Structured logs** include request IDs, processing times, and prediction summaries
- **Audio content is NEVER logged** — the `PrivacyGuard` middleware strips any audio
  references before they reach the logger
- Log level `DEBUG` may include feature shapes and model metadata, but never raw data

## Container Security

- The Docker container runs as a non-root user
- No volumes are mounted for audio storage
- `/tmp` is a tmpfs mount (RAM-backed) if any transient files are needed by ffmpeg
- Health endpoints do not expose internal state

## Compliance Considerations

- **GDPR Art. 5(1)(e)**: Data minimization — audio is not stored beyond processing
- **CCPA**: No personal information is sold or shared; audio is processed transiently
- **SOC 2**: Logging and access controls support auditability without storing PII

## Recommendations for Production

1. Enable TLS termination at the load balancer level
2. Use network policies to restrict ingress to authorized callers only
3. Rotate container instances regularly to prevent memory accumulation
4. Audit log access with role-based controls
