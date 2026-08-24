"""Integration tests for POST /v1/analyze endpoint."""

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
    data = response.json()
    assert "status" in data


def test_analyze_endpoint_with_synthetic_audio(client, sample_audio_bytes):
    files = {"file": ("test.wav", sample_audio_bytes, "audio/wav")}
    response = client.post("/v1/analyze", files=files)
    assert response.status_code == 200

    data = response.json()
    assert "contact_id" in data
    assert "gender" in data
    assert "prediction" in data["gender"]
    assert "confidence" in data["gender"]
    assert "age_bracket" in data
    assert "prediction" in data["age_bracket"]
    assert "confidence" in data["age_bracket"]
    assert "processing_ms" in data
    assert "audio_quality" in data
    assert data["gender"]["prediction"] in {"male", "female", "unknown"}
    assert data["age_bracket"]["prediction"] in {"18-30", "31-45", "46-60", "60+", "unknown"}
    assert data["audio_quality"] in {"good", "degraded", "insufficient"}
