"""
Standalone ASR server for Qwen3-ASR-0.6B.

vLLM does not yet support the qwen3_asr architecture, so we serve ASR
via the official `qwen-asr` package with an OpenAI-compatible API.

Usage:
    pip install qwen-asr soundfile numpy uvicorn fastapi
    python -m src.asr_server          # port 8004
    # or
    uvicorn src.asr_server:app --host 0.0.0.0 --port 8004
"""

import base64
import io
import logging

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("asr-server")
logging.basicConfig(level=logging.INFO)

# ── Model loading ────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"

app = FastAPI(title="Qwen3-ASR Server")

model = None


def load_model():
    global model
    if model is not None:
        return
    logger.info(f"Loading {MODEL_ID} ...")
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=256,
    )
    logger.info("ASR model loaded successfully.")


@app.on_event("startup")
async def startup():
    load_model()


# ── Request schemas ──────────────────────────────────────────────────

class TranscribeRequest(BaseModel):
    """Request to transcribe audio via JSON body."""
    audio: str  # base64-encoded audio data
    format: str = "pcm"  # pcm, wav
    sample_rate: int = 8000
    language: str | None = None  # None = auto-detect


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "loading", "model": MODEL_ID}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "qwen",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(None),
    model_name: str = Form(MODEL_ID, alias="model"),
    language: str = Form(None),
):
    """OpenAI-compatible transcription endpoint (multipart form)."""
    if model is None:
        raise HTTPException(503, "Model not loaded yet")

    if file is None:
        raise HTTPException(400, "No audio file provided")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    try:
        buf = io.BytesIO(audio_bytes)
        audio_data, sr = sf.read(buf)
        text, detected_lang = _transcribe(audio_data, sr, language)
    except Exception:
        # If soundfile can't read it, try as raw PCM 16-bit mono
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            text, detected_lang = _transcribe(audio_data, 16000, language)
        except Exception as e:
            raise HTTPException(500, f"Failed to transcribe: {str(e)}")

    return JSONResponse(content={
        "text": text,
        "language": detected_lang,
    })


@app.post("/v1/chat/completions")
async def chat_completions_asr(request_data: dict):
    """Handle ASR via the chat completions format (used by our Pipecat processor).

    Expects messages with input_audio content type containing base64 PCM audio.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded yet")

    messages = request_data.get("messages", [])
    if not messages:
        raise HTTPException(400, "No messages provided")

    # Extract audio from the message content
    audio_b64 = None
    audio_format = "pcm"
    sample_rate = 8000
    language = request_data.get("language")  # Optional language hint

    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "input_audio":
                    audio_info = item.get("input_audio", {})
                    audio_b64 = audio_info.get("data")
                    audio_format = audio_info.get("format", "pcm")
                    sample_rate = audio_info.get("sample_rate", 8000)
                    break

    if not audio_b64:
        raise HTTPException(400, "No audio data found in messages")

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 audio data")

    if not audio_bytes:
        raise HTTPException(400, "Empty audio data")

    try:
        if audio_format == "pcm":
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            text, detected_lang = _transcribe(audio_data, sample_rate, language)
        else:
            buf = io.BytesIO(audio_bytes)
            audio_data, sr = sf.read(buf)
            text, detected_lang = _transcribe(audio_data, sr, language)
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(500, f"Transcription failed: {str(e)}")

    # Return in OpenAI chat completions format
    return {
        "id": "asr-response",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


_LANG_CODE_MAP = {
    "en": "English", "zh": "Chinese", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "ro": "Romanian",
    "hu": "Hungarian", "mk": "Macedonian",
}


def _transcribe(audio_data: np.ndarray, sample_rate: int, language: str | None) -> tuple[str, str]:
    """Transcribe audio using the Qwen3-ASR model.

    Returns (text, detected_language).
    """
    # Map language codes (e.g. "en") to full names (e.g. "English")
    if language and language.lower() in _LANG_CODE_MAP:
        language = _LANG_CODE_MAP[language.lower()]

    logger.info(f"ASR: sr={sample_rate}, samples={len(audio_data)}, "
                f"duration={len(audio_data)/sample_rate:.1f}s, lang={language}")

    results = model.transcribe(
        audio=(audio_data, sample_rate),
        language=language,
    )

    text = results[0].text if results else ""
    detected_lang = results[0].language if results else "unknown"

    logger.info(f"ASR result: lang={detected_lang}, text={text[:80]}")
    return text, detected_lang


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
