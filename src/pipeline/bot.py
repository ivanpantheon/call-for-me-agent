"""Main Pipecat pipeline for the Call-for-Me voice agent.

Constructs and runs the realtime voice pipeline:
  Twilio Media Streams → Qwen3-ASR → Shadow Bridge → LLM (Qwen3-8B) → Qwen3-TTS → Twilio
"""

import asyncio
import os
import uuid

from loguru import logger
from openai import AsyncOpenAI
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from src.config import settings
from src.pipeline.processors import (
    Qwen3ASRProcessor,
    Qwen3TTSProcessor,
    ShadowModelBridge,
)

# Default system prompt used when no call plan is provided
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI phone agent making a call on behalf of a user. "
    "Your responses will be spoken aloud, so keep them concise and conversational. "
    "Avoid special characters, formatting, or markdown. "
    "Be polite, professional, and focused on accomplishing the user's goal."
)


async def create_pipeline(
    transport: FastAPIWebsocketTransport,
    call_id: str,
    call_plan: dict | None = None,
) -> PipelineTask:
    """Create the voice agent pipeline.

    Args:
        transport: The Twilio WebSocket transport
        call_id: Unique identifier for this call
        call_plan: Optional structured call plan from the shadow model

    Returns:
        PipelineTask ready to be run
    """
    # System prompt from call plan or default
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if call_plan:
        system_prompt = call_plan.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    # LLM: Qwen3-8B via vLLM (OpenAI-compatible)
    llm = OpenAILLMService(
        api_key="not-needed",
        base_url=settings.vllm_llm_base_url,
        model=settings.llm_model,
    )

    # ASR: Qwen3-ASR (with language hint to avoid Chinese hallucination on noise)
    call_language = call_plan.get("language", "en") if call_plan else "en"
    asr = Qwen3ASRProcessor(
        sample_rate=settings.audio_sample_rate,
        language=call_language,
    )

    # TTS: Qwen3-TTS
    tts = Qwen3TTSProcessor(
        sample_rate=settings.audio_sample_rate,
        voice="Chelsie",
        language=call_plan.get("language", "en") if call_plan else "en",
    )

    # Shadow model bridge for Redis communication
    shadow_bridge = ShadowModelBridge(call_id=call_id)
    await shadow_bridge.start_directive_listener()

    # LLM context and aggregators
    messages = [{"role": "system", "content": system_prompt}]

    # If we have a call plan with information to provide, inject it
    if call_plan and call_plan.get("information_to_provide"):
        info_text = "\n".join(
            f"- {k}: {v}" for k, v in call_plan["information_to_provide"].items()
        )
        messages[0]["content"] += f"\n\nUser information you may need to provide:\n{info_text}"

    context = LLMContext(messages)

    # Use only VAD for turn start detection. The default includes
    # TranscriptionUserTurnStartStrategy which causes every ASR result
    # to trigger an interruption, preventing the LLM from ever completing
    # a response. With VAD-only, interruptions are based on actual speech
    # detection from Silero VAD, not transcription events.
    turn_strategies = UserTurnStrategies(
        start=[VADUserTurnStartStrategy()],
        stop=[SpeechTimeoutUserTurnStopStrategy(stop_secs=0.8)],
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=settings.vad_threshold_ms / 1000.0,
                )
            ),
            user_turn_strategies=turn_strategies,
        ),
    )

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),       # Audio from Twilio Media Streams
            asr,                     # Qwen3-ASR: audio → text
            shadow_bridge,           # Publish transcript to Redis
            user_aggregator,         # Aggregate user turns
            llm,                     # Qwen3-8B: generate response
            tts,                     # Qwen3-TTS: text → audio
            transport.output(),      # Audio back to Twilio
            assistant_aggregator,    # Track assistant responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=settings.audio_sample_rate,
            audio_out_sample_rate=settings.audio_sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Call {call_id}: client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Call {call_id}: client disconnected")
        await shadow_bridge.cleanup()
        await task.cancel()

    return task


async def run_bot(runner_args: RunnerArguments, call_plan: dict | None = None):
    """Main bot entry point. Called when Twilio connects via WebSocket.

    Args:
        runner_args: WebSocket runner arguments from Pipecat
        call_plan: Optional structured call plan from the shadow model
    """
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Transport detected: {transport_type}")

    call_id = call_data.get("call_id", str(uuid.uuid4()))
    stream_sid = call_data.get("stream_id", "")

    # Extract call metadata
    body_data = call_data.get("body", {})
    to_number = body_data.get("to_number", "unknown")
    from_number = body_data.get("from_number", settings.twilio_phone_number)
    logger.info(f"Call {call_id}: {from_number} → {to_number}")

    # Look up call plan from body data if not provided
    if not call_plan:
        plan_json = body_data.get("call_plan")
        if plan_json:
            import json
            call_plan = json.loads(plan_json)

    # Create Twilio serializer
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_id,
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
    )

    # Create transport
    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    # Build and run pipeline
    task = await create_pipeline(transport, call_id, call_plan)

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)
