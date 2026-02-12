# Call-for-Me Agent

An open-source AI phone agent that makes outbound calls on your behalf — books appointments, handles customer service, gathers information — across 40+ countries with multilingual support.

Built with open-source AI models and Twilio for telephony.

## Architecture

```
User (web UI) → API Server → Shadow Model (Qwen3-32B) generates call plan
                                    ↓
                    Twilio places outbound call via PSTN
                                    ↓
                    Twilio Media Streams ↔ WebSocket ↔ Pipecat Pipeline
                                                          ↓
                                           Silero VAD → Qwen3-ASR → [Translation]
                                                          ↓
                                           Shadow Model ← Redis Streams → Context Aggregator
                                                          ↓
                                           Qwen3-8B LLM → [Translation] → Qwen3-TTS
                                                          ↓
                                           Audio back to callee via Twilio
```

**Dual-layer architecture**: A fast 8B model handles realtime dialogue while a 32B shadow model monitors the conversation asynchronously and sends strategic directives via Redis Streams.

## Components

| Layer | Component | Role |
|-------|-----------|------|
| Telephony | Twilio Media Streams | Global voice network, bidirectional WebSocket audio |
| Orchestration | Pipecat | Realtime voice pipeline with interruption handling |
| ASR | Qwen3-ASR 0.6B | 52-language speech-to-text, streaming |
| TTS | Qwen3-TTS 0.6B | 10-language text-to-speech, 97ms latency |
| Dialogue LLM | Qwen3-8B (INT4) | Realtime conversation, <500ms response |
| Shadow Model | Qwen3-32B (INT4) | Strategic reasoning, goal tracking, directives |
| Translation | Opus-MT | 1500+ language pairs, MIT license |
| VAD | Silero VAD | Voice activity detection, 200ms smart endpointing |
| Inference | vLLM | All models served with continuous batching |

## Quick Start

### Prerequisites

- NVIDIA GPU with 40+ GB VRAM (A100 80GB recommended)
- Docker with NVIDIA Container Toolkit
- Twilio account with a phone number
- ngrok (for development)

### 1. Configure

```bash
cp .env.example .env
# Edit .env with your Twilio credentials and ngrok URL
```

### 2. Start Infrastructure

```bash
cd docker
docker compose up -d
```

This starts:
- 4 vLLM instances (ASR, LLM, TTS, Shadow Model)
- Redis (for shadow model communication)
- PostgreSQL (for call records)
- API server

### 3. Expose via ngrok (development)

```bash
ngrok http 8000
# Update PUBLIC_BASE_URL in .env with the ngrok URL
```

### 4. Make a Call

```bash
curl -X POST http://localhost:8000/call \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "Book a dental cleaning appointment for next Tuesday at 2pm",
    "to_number": "+14155551234",
    "user_profile": {
      "name": "John Doe",
      "patient_id": "12345"
    }
  }'
```

### 5. Check Status

```bash
curl http://localhost:8000/calls/{call_id}
```

## Cost

~$0.08-0.17 per 5-minute call (GPU + Twilio). 3-20x cheaper than managed platforms.

| Scenario | Cost per 5-min call |
|----------|-------------------|
| US domestic | $0.087 |
| International | $0.165 |
| With translation | $0.167 |

## Project Structure

```
src/
├── api/server.py              # FastAPI endpoints (call initiation, Twilio webhooks)
├── pipeline/
│   ├── bot.py                 # Pipecat pipeline construction
│   └── processors.py          # Custom processors (ASR, TTS, Shadow Bridge)
├── shadow_model/service.py    # Shadow model service (goal tracking, directives)
├── translation/translator.py  # Opus-MT translation pipeline
└── config.py                  # Settings from environment
docker/
├── docker-compose.yml         # Full stack deployment
└── Dockerfile                 # Application container
```

## License

MIT
