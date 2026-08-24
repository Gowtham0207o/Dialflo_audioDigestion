"""Integration tests for POST /analyze (Silero VAD & Quality Assessment)."""

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
    """Test POST /analyze with valid audio returns metadata, Silero VAD, and quality fields."""
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
