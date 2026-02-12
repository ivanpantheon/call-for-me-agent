from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # vLLM endpoints
    vllm_asr_base_url: str = "http://localhost:8001/v1"
    vllm_llm_base_url: str = "http://localhost:8002/v1"
    vllm_tts_base_url: str = "http://localhost:8003/v1"
    vllm_shadow_base_url: str = "http://localhost:8004/v1"

    # Model names
    asr_model: str = "Qwen/Qwen3-ASR-0.6B"
    llm_model: str = "Qwen/Qwen3-8B"
    tts_model: str = "Qwen/Qwen3-TTS-0.6B-CustomVoice"
    shadow_model: str = "Qwen/Qwen3-32B"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Database
    database_url: str = "postgresql://callbot:callbot@localhost:5432/callbot"

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765

    # Public URL for Twilio webhooks
    public_base_url: str = "http://localhost:8000"

    # Latency optimizations
    vad_threshold_ms: int = 200
    audio_sample_rate: int = 8000
    audio_format: str = "mulaw"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
