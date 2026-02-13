"""Custom Pipecat processors for the Call-for-Me pipeline.

Includes:
- Qwen3ASRProcessor: Wraps Qwen3-ASR via vLLM OpenAI-compatible API for streaming STT
- Qwen3TTSProcessor: Wraps Qwen3-TTS via vLLM for streaming TTS
- ShadowModelBridge: Publishes transcript to Redis and receives directives
- ContextInjector: Injects shadow model directives into LLM context
"""

import asyncio
import base64
import json
import time
from typing import AsyncGenerator

import httpx
import numpy as np
import redis.asyncio as redis
from loguru import logger
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from src.config import settings


class Qwen3ASRProcessor(FrameProcessor):
    """Speech-to-text processor using Qwen3-ASR via vLLM.

    Receives AudioRawFrames from the transport, batches them, and sends
    to the Qwen3-ASR model for streaming transcription.

    Includes energy-based pre-filtering to avoid transcribing silence,
    and filters known hallucination patterns (Chinese filler sounds from noise).
    """

    # Patterns the ASR hallucinates from silence/noise
    _HALLUCINATION_PATTERNS = {
        "嗯", "嗯。", "嗯嗯", "呵呵", "啊", "哦", "呃", "唔",
        "嗯 。", "嗯.", "啊。", "哦。", "呃。",
    }

    def __init__(
        self,
        sample_rate: int = 8000,
        language: str = "en",
        energy_threshold: float = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._sample_rate = sample_rate
        self._language = language
        self._energy_threshold = energy_threshold
        self._audio_buffer = bytearray()
        self._buffer_duration_ms = 0
        self._min_buffer_ms = 1000  # Accumulate at least 1s before transcribing
        self._client = httpx.AsyncClient(base_url=settings.vllm_asr_base_url, timeout=30.0)
        self._transcribing = False

    @staticmethod
    def _compute_rms_energy(audio_bytes: bytes) -> float:
        """Compute RMS energy of 16-bit PCM audio. Returns 0.0-1.0 range."""
        if len(audio_bytes) < 2:
            return 0.0
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(samples ** 2)))

    def _is_hallucination(self, text: str) -> bool:
        """Check if transcription is a known hallucination pattern."""
        cleaned = text.strip().rstrip(".,。，")
        return cleaned in self._HALLUCINATION_PATTERNS or text.strip() in self._HALLUCINATION_PATTERNS

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            # Always pass audio downstream so VAD in user aggregator works
            await self.push_frame(frame, direction)

            self._audio_buffer.extend(frame.audio)
            bytes_per_ms = (self._sample_rate * 2) / 1000  # 16-bit = 2 bytes per sample
            self._buffer_duration_ms = len(self._audio_buffer) / bytes_per_ms

            if self._buffer_duration_ms >= self._min_buffer_ms and not self._transcribing:
                audio_data = bytes(self._audio_buffer)
                self._audio_buffer.clear()
                self._buffer_duration_ms = 0

                # Skip silent audio - don't waste ASR compute on noise
                rms = self._compute_rms_energy(audio_data)
                if rms < self._energy_threshold:
                    logger.debug(f"ASR: skipping silent chunk (rms={rms:.4f})")
                    return

                self._transcribing = True
                asyncio.create_task(self._transcribe(audio_data))

        elif isinstance(frame, (CancelFrame, EndFrame, StartInterruptionFrame)):
            self._audio_buffer.clear()
            self._buffer_duration_ms = 0
            self._transcribing = False
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _transcribe(self, audio_data: bytes):
        """Send audio to Qwen3-ASR and emit transcription frames."""
        try:
            # Encode audio as base64 for the API
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")

            # Use the chat completions endpoint with language hint
            request_body = {
                "model": settings.asr_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_b64,
                                    "format": "pcm",
                                    "sample_rate": self._sample_rate,
                                },
                            }
                        ],
                    }
                ],
                "stream": False,
                "max_tokens": 500,
            }
            if self._language:
                request_body["language"] = self._language

            response = await self._client.post(
                "/chat/completions",
                json=request_body,
            )
            response.raise_for_status()
            result = response.json()

            text = result["choices"][0]["message"]["content"].strip()
            if text and not self._is_hallucination(text):
                logger.info(f"ASR transcription: '{text}'")
                frame = TranscriptionFrame(
                    text=text,
                    user_id="callee",
                    timestamp=str(time.time()),
                )
                await self.push_frame(frame)
            elif text:
                logger.debug(f"ASR: filtered hallucination '{text}'")

        except Exception as e:
            logger.error(f"ASR error: {e}")
        finally:
            self._transcribing = False

    async def cleanup(self):
        await self._client.aclose()


class Qwen3TTSProcessor(FrameProcessor):
    """Text-to-speech processor using Qwen3-TTS via vLLM.

    Receives TextFrames from the LLM and converts them to AudioRawFrames
    using Qwen3-TTS with streaming synthesis.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        voice: str = "Chelsie",
        language: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._sample_rate = sample_rate
        self._voice = voice
        self._language = language
        self._client = httpx.AsyncClient(base_url=settings.vllm_tts_base_url, timeout=60.0)
        self._generating = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            if frame.text.strip():
                self._generating = True
                await self._synthesize(frame.text)
                self._generating = False

        elif isinstance(frame, (CancelFrame, StartInterruptionFrame)):
            self._generating = False
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _synthesize(self, text: str):
        """Synthesize text to audio using Qwen3-TTS."""
        try:
            # Use vLLM's OpenAI-compatible TTS endpoint
            response = await self._client.post(
                "/audio/speech",
                json={
                    "model": settings.tts_model,
                    "input": text,
                    "voice": self._voice,
                    "response_format": "pcm",
                    "speed": 1.0,
                },
            )
            response.raise_for_status()

            # The response is raw PCM audio
            audio_data = response.content

            if audio_data and self._generating:
                # Split into smaller chunks for smoother streaming
                chunk_size = self._sample_rate * 2 // 10  # 100ms chunks (16-bit)
                for i in range(0, len(audio_data), chunk_size):
                    if not self._generating:
                        break  # Stop if interrupted
                    chunk = audio_data[i : i + chunk_size]
                    frame = AudioRawFrame(
                        audio=chunk,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    )
                    await self.push_frame(frame)

        except Exception as e:
            logger.error(f"TTS error: {e}")

    async def cleanup(self):
        await self._client.aclose()


class ShadowModelBridge(FrameProcessor):
    """Bridge between the Pipecat pipeline and the shadow model service.

    - Publishes transcription events to Redis Streams (for shadow model consumption)
    - Listens for directives from the shadow model and injects them into the pipeline
    """

    def __init__(self, call_id: str, **kwargs):
        super().__init__(**kwargs)
        self.call_id = call_id
        self._redis = redis.from_url(settings.redis_url)
        self._directive_task: asyncio.Task | None = None
        self._current_directive: str | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # Publish transcription to Redis for shadow model
            await self._publish_transcript(frame.text, speaker="callee")
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame):
            # Agent's response text - also publish to shadow model
            await self._publish_transcript(frame.text, speaker="agent")
            await self.push_frame(frame, direction)

        elif isinstance(frame, EndFrame):
            await self._publish_state("call_ended")
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def start_directive_listener(self):
        """Start listening for shadow model directives in the background."""
        self._directive_task = asyncio.create_task(self._listen_for_directives())

    async def get_current_directive(self) -> str | None:
        """Get the most recent directive from the shadow model."""
        return self._current_directive

    async def _publish_transcript(self, text: str, speaker: str):
        """Publish a transcript event to Redis Stream."""
        try:
            await self._redis.xadd(
                f"call:{self.call_id}:transcript",
                {
                    "type": "transcript",
                    "speaker": speaker,
                    "text": text,
                    "timestamp": str(time.time()),
                },
                maxlen=1000,
            )
        except Exception as e:
            logger.warning(f"Failed to publish transcript: {e}")

    async def _publish_state(self, state: str):
        """Publish a state change event to Redis Stream."""
        try:
            await self._redis.xadd(
                f"call:{self.call_id}:transcript",
                {
                    "type": "state_change",
                    "state": state,
                    "timestamp": str(time.time()),
                },
                maxlen=1000,
            )
        except Exception as e:
            logger.warning(f"Failed to publish state: {e}")

    async def _listen_for_directives(self):
        """Listen for directives from the shadow model via Redis Stream."""
        stream_key = f"call:{self.call_id}:directives"
        last_id = "0-0"

        while True:
            try:
                events = await self._redis.xread(
                    {stream_key: last_id},
                    count=5,
                    block=1000,
                )
                if not events:
                    continue

                for _stream, messages in events:
                    for msg_id, data in messages:
                        last_id = msg_id
                        directive = {k.decode(): v.decode() for k, v in data.items()}
                        self._current_directive = directive.get("content")
                        logger.info(
                            f"Received directive: {directive.get('action')} - "
                            f"{directive.get('content', '')[:80]}"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Directive listener error: {e}")
                await asyncio.sleep(1)

    async def cleanup(self):
        if self._directive_task:
            self._directive_task.cancel()
        await self._redis.aclose()


class DynamicSystemPromptInjector(FrameProcessor):
    """Injects shadow model directives into the LLM's system prompt.

    Before each LLM call, this processor checks for new directives from
    the shadow model and updates the system prompt accordingly.
    """

    def __init__(self, shadow_bridge: ShadowModelBridge, base_system_prompt: str, **kwargs):
        super().__init__(**kwargs)
        self._shadow_bridge = shadow_bridge
        self._base_system_prompt = base_system_prompt

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # Pass all frames through - the directive injection happens
        # when the LLM context aggregator builds the next prompt
        await self.push_frame(frame, direction)

    def get_current_system_prompt(self) -> str:
        """Build the system prompt with any active directives."""
        directive = asyncio.get_event_loop().run_until_complete(
            self._shadow_bridge.get_current_directive()
        )
        if directive:
            return (
                f"{self._base_system_prompt}\n\n"
                f"CURRENT DIRECTIVE FROM SUPERVISOR:\n{directive}"
            )
        return self._base_system_prompt
