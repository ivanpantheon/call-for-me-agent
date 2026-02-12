"""
Standalone TTS server for Qwen3-TTS-12Hz-0.6B-CustomVoice.

vLLM does not yet support the qwen3_tts architecture, so we serve TTS
via the official `qwen-tts` package with an OpenAI-compatible API.

Usage:
    pip install qwen-tts soundfile numpy uvicorn fastapi
    python -m src.tts_server          # port 8003
    # or
    uvicorn src.tts_server:app --host 0.0.0.0 --port 8003
"""

import io
import struct
import logging
from typing import Optional

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger("tts-server")
logging.basicConfig(level=logging.INFO)

# ── Model loading ────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

app = FastAPI(title="Qwen3-TTS Server")

model = None


def load_model():
    global model
    if model is not None:
        return
    logger.info(f"Loading {MODEL_ID} ...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    logger.info("TTS model loaded successfully.")


@app.on_event("startup")
async def startup():
    load_model()


# ── Request schemas ──────────────────────────────────────────────────

VALID_SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee", "Chelsie",
]

# Map OpenAI voice names to Qwen speakers
VOICE_MAP = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Aiden",
    "nova": "Serena",
    "onyx": "Dylan",
    "shimmer": "Vivian",
    "chelsie": "Ryan",  # fallback for our default
}


class SpeechRequest(BaseModel):
    model: str = "qwen3-tts"
    input: str
    voice: str = "Ryan"
    response_format: str = "pcm"  # pcm, wav, mp3
    speed: float = 1.0
    language: str = "Auto"
    instruct: str = ""


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


@app.post("/v1/audio/speech")
def synthesize_speech(req: SpeechRequest):
    """OpenAI-compatible TTS endpoint. Returns raw audio bytes."""
    if model is None:
        raise HTTPException(503, "Model not loaded yet")

    if not req.input or not req.input.strip():
        raise HTTPException(400, "Empty input text")

    # Resolve voice name
    speaker = req.voice
    if speaker.lower() in VOICE_MAP:
        speaker = VOICE_MAP[speaker.lower()]
    elif speaker not in VALID_SPEAKERS:
        logger.warning(f"Unknown voice '{speaker}', falling back to Ryan")
        speaker = "Ryan"

    # Detect language from text if Auto
    language = req.language
    if language == "Auto":
        # Simple heuristic: if mostly CJK chars, use appropriate language
        language = _detect_language(req.input)

    logger.info(f"TTS: voice={speaker}, lang={language}, fmt={req.response_format}, "
                f"len={len(req.input)} chars")

    try:
        gen_kwargs = dict(
            max_new_tokens=2048,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
        )

        wavs, sr = model.generate_custom_voice(
            text=req.input,
            language=language,
            speaker=speaker,
            instruct=req.instruct if req.instruct else "",
            **gen_kwargs,
        )

        audio_data = wavs[0]  # numpy array, float32

    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"TTS generation failed: {str(e)}")

    # Convert to requested format
    if req.response_format == "pcm":
        # Convert to 16-bit PCM at 8kHz mono (what Twilio/Pipecat expects)
        pcm_bytes = _to_pcm_8khz(audio_data, sr)
        return Response(content=pcm_bytes, media_type="audio/pcm")

    elif req.response_format == "wav":
        buf = io.BytesIO()
        sf.write(buf, audio_data, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")

    else:
        # Default to wav for unsupported formats
        buf = io.BytesIO()
        sf.write(buf, audio_data, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")


def _to_pcm_8khz(audio: np.ndarray, original_sr: int) -> bytes:
    """Convert float32 audio to 16-bit PCM at 8kHz."""
    # Resample to 8kHz if needed
    if original_sr != 8000:
        # Simple linear interpolation resampling
        duration = len(audio) / original_sr
        target_len = int(duration * 8000)
        indices = np.linspace(0, len(audio) - 1, target_len)
        audio_8k = np.interp(indices, np.arange(len(audio)), audio)
    else:
        audio_8k = audio

    # Normalize to [-1, 1] range
    max_val = np.max(np.abs(audio_8k))
    if max_val > 0:
        audio_8k = audio_8k / max_val

    # Convert to 16-bit signed integer PCM
    pcm_16 = (audio_8k * 32767).astype(np.int16)
    return pcm_16.tobytes()


def _detect_language(text: str) -> str:
    """Simple heuristic language detection."""
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    jp_count = sum(1 for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
    kr_count = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
    total = len(text)

    if total == 0:
        return "English"

    if jp_count / total > 0.1:
        return "Japanese"
    if kr_count / total > 0.1:
        return "Korean"
    if cjk_count / total > 0.2:
        return "Chinese"
    return "English"


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
