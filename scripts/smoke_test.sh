#!/usr/bin/env bash
# Quick curl-based smoke test script against running server

HOST="${APP_HOST:-localhost}"
PORT="${APP_PORT:-8000}"
URL="http://${HOST}:${PORT}"

echo "=== DialFlo Audio Digestion Smoke Test ==="
echo "Targeting: ${URL}"

# 1. Health check
echo -n "[1/2] GET /v1/health ... "
HEALTH_RES=$(curl -s -w "\n%{http_code}" "${URL}/v1/health")
STATUS=$(echo "${HEALTH_RES}" | tail -n1)
BODY=$(echo "${HEALTH_RES}" | sed '$d')

if [ "${STATUS}" -eq 200 ]; then
    echo "OK (200)"
    echo "    Body: ${BODY}"
else
    echo "FAILED (${STATUS})"
    echo "    Body: ${BODY}"
    exit 1
fi

# 2. Analyze test
SAMPLE_FILE="tests/fixtures/audio/sample_clean.wav"
if [ ! -f "${SAMPLE_FILE}" ]; then
    echo "Generating sample audio file..."
    python scripts/generate_sample_audio.py
fi

echo -n "[2/2] POST /v1/analyze ... "
ANALYZE_RES=$(curl -s -w "\n%{http_code}" -X POST "${URL}/v1/analyze" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@${SAMPLE_FILE};type=audio/wav")

STATUS=$(echo "${ANALYZE_RES}" | tail -n1)
BODY=$(echo "${ANALYZE_RES}" | sed '$d')

if [ "${STATUS}" -eq 200 ]; then
    echo "OK (200)"
    echo "    Body: ${BODY}"
else
    echo "FAILED (${STATUS})"
    echo "    Body: ${BODY}"
    exit 1
fi

echo "=== Smoke Test PASSED ==="
