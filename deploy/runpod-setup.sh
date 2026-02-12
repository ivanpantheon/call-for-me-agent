#!/bin/bash
# =============================================================================
# RunPod One-Click Setup Script
# Deploys the full Call-for-Me agent stack on a single RunPod GPU pod
# =============================================================================
#
# USAGE:
#   1. Create a RunPod GPU pod (A100 80GB recommended, or A40 48GB for dev)
#      - Template: RunPod PyTorch 2.4 (or any CUDA 12+ image)
#      - Expose HTTP port 8000
#      - Set volume mount for model caching
#   2. SSH into the pod
#   3. Clone your repo and run: bash deploy/runpod-setup.sh
#
# =============================================================================

set -euo pipefail

# --- 0. Move HuggingFace cache to workspace volume (root overlay is only 20GB) ---
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
if [ -d /root/.cache/huggingface ] && [ ! -L /root/.cache/huggingface ]; then
    mv /root/.cache/huggingface/* "$HF_HOME/" 2>/dev/null || true
    rm -rf /root/.cache/huggingface
    ln -s "$HF_HOME" /root/.cache/huggingface
fi

echo "============================================"
echo " Call-for-Me Agent - RunPod Setup"
echo "============================================"

# --- 1. Install system dependencies ---
echo "[1/8] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq redis-server postgresql postgresql-contrib curl unzip > /dev/null 2>&1

# --- 2. Install Python dependencies ---
echo "[2/8] Installing Python packages..."
pip install -q --upgrade pip
pip install -q \
    "pipecat-ai[silero]>=0.0.60" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.34.0" \
    "twilio>=9.0.0" \
    "redis>=5.0.0" \
    "openai>=1.50.0" \
    "httpx>=0.28.0" \
    "pydantic>=2.10.0" \
    "pydantic-settings>=2.7.0" \
    "numpy>=1.26.0" \
    "soundfile>=0.12.0" \
    "websockets>=14.0" \
    "loguru>=0.7.0" \
    "vllm>=0.7.0" \
    "transformers>=4.46.0" \
    "sentencepiece>=0.2.0"

# --- 3. Install ngrok ---
echo "[3/8] Installing ngrok..."
if [ ! -f /usr/local/bin/ngrok ]; then
    curl -sL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar xz -C /usr/local/bin/
fi

# --- 4. Start Redis ---
echo "[4/8] Starting Redis..."
redis-server --daemonize yes --appendonly yes

# --- 5. Start PostgreSQL ---
echo "[5/8] Starting PostgreSQL..."
service postgresql start || true
su - postgres -c "psql -c \"CREATE USER callbot WITH PASSWORD 'callbot';\"" 2>/dev/null || true
su - postgres -c "psql -c \"CREATE DATABASE callbot OWNER callbot;\"" 2>/dev/null || true

# --- 6. Download and start vLLM model servers ---
echo "[6/8] Starting vLLM model servers (this takes a few minutes on first run)..."

# Check available GPU memory to decide model sizes
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  GPU Memory: ${GPU_MEM} MiB"

if [ "$GPU_MEM" -ge 70000 ]; then
    echo "  Using full model stack (ASR 0.6B + LLM 8B + TTS 0.6B + Shadow 32B)"
    ASR_MODEL="Qwen/Qwen3-ASR-0.6B"
    LLM_MODEL="Qwen/Qwen3-8B"
    TTS_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    SHADOW_MODEL="Qwen/Qwen3-32B"
    SHADOW_GPU_UTIL="0.40"
elif [ "$GPU_MEM" -ge 40000 ]; then
    echo "  Using reduced stack (ASR 0.6B + LLM 8B + TTS 0.6B + Shadow 14B)"
    ASR_MODEL="Qwen/Qwen3-ASR-0.6B"
    LLM_MODEL="Qwen/Qwen3-8B"
    TTS_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    SHADOW_MODEL="Qwen/Qwen3-14B"
    SHADOW_GPU_UTIL="0.30"
else
    echo "  Using minimal stack (ASR 0.6B + LLM 4B + TTS 0.6B, no shadow model)"
    ASR_MODEL="Qwen/Qwen3-ASR-0.6B"
    LLM_MODEL="Qwen/Qwen3-4B"
    TTS_MODEL="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    SHADOW_MODEL=""
    SHADOW_GPU_UTIL="0"
fi

# Start ASR server
echo "  Starting ASR server (${ASR_MODEL})..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$ASR_MODEL" \
    --port 8001 \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.12 \
    > /tmp/vllm-asr.log 2>&1 &
echo "    PID: $!"

# Start LLM server
echo "  Starting LLM server (${LLM_MODEL})..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$LLM_MODEL" \
    --port 8002 \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.25 \
    > /tmp/vllm-llm.log 2>&1 &
echo "    PID: $!"

# Start TTS server
echo "  Starting TTS server (${TTS_MODEL})..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$TTS_MODEL" \
    --port 8003 \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.12 \
    > /tmp/vllm-tts.log 2>&1 &
echo "    PID: $!"

# Start Shadow Model server (if enough GPU memory)
if [ -n "$SHADOW_MODEL" ]; then
    echo "  Starting Shadow Model server (${SHADOW_MODEL})..."
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$SHADOW_MODEL" \
        --port 8004 \
        --dtype auto \
        --max-model-len 8192 \
        --gpu-memory-utilization "$SHADOW_GPU_UTIL" \
        > /tmp/vllm-shadow.log 2>&1 &
    echo "    PID: $!"
fi

# --- 7. Wait for models to load ---
echo "[7/8] Waiting for models to load..."
echo "  (This can take 3-10 minutes on first download)"

wait_for_service() {
    local port=$1
    local name=$2
    local max_wait=600  # 10 minutes
    local waited=0
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            echo "  WARNING: ${name} on port ${port} did not start in ${max_wait}s"
            echo "  Check logs: tail -50 /tmp/vllm-*.log"
            return 1
        fi
    done
    echo "  ✓ ${name} ready (port ${port}, ${waited}s)"
}

wait_for_service 8001 "ASR" &
wait_for_service 8002 "LLM" &
wait_for_service 8003 "TTS" &
if [ -n "$SHADOW_MODEL" ]; then
    wait_for_service 8004 "Shadow" &
fi
wait  # Wait for all health checks

# --- 8. Print status ---
echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo " Services running:"
echo "   ASR:     http://localhost:8001  (${ASR_MODEL})"
echo "   LLM:     http://localhost:8002  (${LLM_MODEL})"
echo "   TTS:     http://localhost:8003  (${TTS_MODEL})"
if [ -n "$SHADOW_MODEL" ]; then
    echo "   Shadow:  http://localhost:8004  (${SHADOW_MODEL})"
fi
echo "   Redis:   localhost:6379"
echo "   Postgres: localhost:5432"
echo ""
echo " Next steps:"
echo "   1. Copy .env.example to .env and add your Twilio credentials"
echo "   2. Start ngrok:  ngrok http 8000"
echo "   3. Update PUBLIC_BASE_URL in .env with ngrok URL"
echo "   4. Start the API server:"
echo "      python -m src.api.server"
echo "   5. Make a test call:"
echo '      curl -X POST http://localhost:8000/call \'
echo '        -H "Content-Type: application/json" \'
echo '        -d '"'"'{"intent": "Ask about store hours", "to_number": "+1YOUR_NUMBER"}'"'"''
echo ""
echo " View model logs:"
echo "   tail -f /tmp/vllm-asr.log"
echo "   tail -f /tmp/vllm-llm.log"
echo "   tail -f /tmp/vllm-tts.log"
echo "   tail -f /tmp/vllm-shadow.log"
