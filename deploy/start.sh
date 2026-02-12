#!/bin/bash
# =============================================================================
# Start the Call-for-Me agent (run after runpod-setup.sh)
# =============================================================================

set -euo pipefail

# Check that .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "  cp .env.example .env"
    echo "  Then add your Twilio credentials and PUBLIC_BASE_URL"
    exit 1
fi

# Source environment
set -a
source .env
set +a

echo "============================================"
echo " Starting Call-for-Me Agent"
echo "============================================"

# Check required env vars
if [ -z "${TWILIO_ACCOUNT_SID:-}" ] || [ "${TWILIO_ACCOUNT_SID}" = "your_account_sid" ]; then
    echo "ERROR: TWILIO_ACCOUNT_SID not set in .env"
    exit 1
fi

if [ -z "${PUBLIC_BASE_URL:-}" ] || [ "${PUBLIC_BASE_URL}" = "https://your-domain.ngrok-free.app" ]; then
    echo "ERROR: PUBLIC_BASE_URL not set in .env"
    echo ""
    echo "  Start ngrok in another terminal:"
    echo "    ngrok http 8000"
    echo "  Then set PUBLIC_BASE_URL to the ngrok URL"
    exit 1
fi

# Ensure Redis is running
redis-cli ping > /dev/null 2>&1 || {
    echo "Starting Redis..."
    redis-server --daemonize yes --appendonly yes
}

# Check that vLLM servers are running
for port in 8001 8002 8003; do
    if ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo "WARNING: vLLM server on port ${port} not responding"
        echo "  Run: bash deploy/runpod-setup.sh"
    fi
done

echo ""
echo "Starting API server on port ${API_PORT:-8000}..."
echo "  PUBLIC_BASE_URL: ${PUBLIC_BASE_URL}"
echo "  Twilio phone:    ${TWILIO_PHONE_NUMBER}"
echo ""
echo "Ready! Make a call with:"
echo '  curl -X POST ${PUBLIC_BASE_URL}/call \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"intent": "Your intent here", "to_number": "+1234567890"}'"'"''
echo ""

# Start the API server
python -m src.api.server
