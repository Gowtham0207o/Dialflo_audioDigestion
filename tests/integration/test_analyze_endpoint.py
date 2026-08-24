"""Integration tests for POST /analyze (Silero VAD, Quality Assessment, Gender & Age Estimation)."""

import pytest
from fastapi.testclient import TestClient
from app.main import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/v1/health")
    assert response.status_code in {200, 503}


def test_post_analyze_valid_audio_with_silero_vad(client, sample_audio_bytes):
    """Test POST /analyze with valid audio returns metadata, Silero VAD, quality, gender, and age_bracket fields."""
    files = {"file": ("test.wav", sample_audio_bytes, "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 200

    data = response.json()
    assert "duration" in data
    assert "duration_ms" in data
    assert data["sample_rate"] == 16000
    assert data["channels"] == 1
    assert "samples" in data
    assert data["original_format"] == "wav"

    # Silero VAD assertions
    assert "speech_duration_seconds" in data
    assert "speech_duration_ms" in data
    assert "speech_ratio" in data
    assert "speech_segments" in data
    assert "is_speech_sufficient" in data
    assert "vad_confidence" in data
    assert 0.0 <= data["vad_confidence"] <= 1.0

    # Quality assertions
    assert "audio_quality" in data
    assert data["audio_quality"] in {"good", "degraded", "insufficient"}
    assert "snr_db" in data
    assert "peak_amplitude" in data
    assert "clipping_ratio" in data
    assert "quality_reasoning" in data

    # Gender classification assertions
    assert "gender" in data
    assert data["gender"] is not None
    assert "prediction" in data["gender"]
    assert data["gender"]["prediction"] in {"male", "female", "unknown"}
    assert "confidence" in data["gender"]
    assert "probabilities" in data["gender"]

    # Age bracket estimation assertions
    assert "age_bracket" in data
    assert data["age_bracket"] is not None
    assert "prediction" in data["age_bracket"]
    assert data["age_bracket"]["prediction"] in {"18-30", "31-45", "46-60", "60+", "unknown"}
    assert "confidence" in data["age_bracket"]
    assert "probabilities" in data["age_bracket"]
    assert "inference_ms" in data["age_bracket"]


def test_post_analyze_empty_file_returns_400(client):
    """Test POST /analyze with empty file returns 400 Bad Request."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "message" in data


def test_post_analyze_unsupported_type_returns_400(client):
    """Test POST /analyze with unsupported extension returns 400 Bad Request."""
    files = {"file": ("test.txt", b"Hello world text payload", "text/plain")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "message" in data
