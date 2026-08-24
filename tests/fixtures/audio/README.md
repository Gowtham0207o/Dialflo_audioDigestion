# Audio test fixtures

Place sample audio files here for integration and e2e tests.

## Sourcing test audio

**Option 1: Mozilla Common Voice** (recommended)
- Visit https://commonvoice.mozilla.org/en/datasets
- Download a small subset of validated clips
- Place `.mp3` files in this directory

**Option 2: Generate synthetic audio**
```bash
python scripts/generate_sample_audio.py
```

**Option 3: Record your own**
- Record a 5-second WAV clip at 16kHz mono
- Save as `sample_clean.wav` in this directory

## Expected test files

| Filename | Description |
|---|---|
| `sample_clean.wav` | Clean 5s speech sample |
| `sample_noisy.wav` | Speech with background noise |
| `sample_short.wav` | Very short (<0.5s) audio |
| `sample_silence.wav` | Silent audio |

> **Note**: Audio files are gitignored. Use the scripts above to generate them locally.
