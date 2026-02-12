"""Translation pipeline using Opus-MT for cross-language calls.

When the user and callee speak different languages, this module provides
text-to-text translation that sits between ASR and LLM, and between LLM and TTS.
"""

import asyncio

from loguru import logger
from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from src.config import settings


class TranslationProcessor(FrameProcessor):
    """Translates text frames between languages using Opus-MT.

    Can be inserted into the pipeline at two points:
    1. After ASR: translates callee's language → agent's language
    2. After LLM: translates agent's language → callee's language
    """

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        direction: str = "asr_to_llm",
        **kwargs,
    ):
        """
        Args:
            source_lang: Source language code (e.g., "es", "fr", "zh")
            target_lang: Target language code (e.g., "en")
            direction: "asr_to_llm" (callee→agent) or "llm_to_tts" (agent→callee)
        """
        super().__init__(**kwargs)
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._direction = direction
        self._model = None
        self._tokenizer = None
        self._model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    async def _load_model(self):
        """Lazily load the Opus-MT model."""
        if self._model is not None:
            return

        logger.info(f"Loading translation model: {self._model_name}")
        try:
            from transformers import MarianMTModel, MarianTokenizer

            loop = asyncio.get_event_loop()
            self._tokenizer = await loop.run_in_executor(
                None, MarianTokenizer.from_pretrained, self._model_name
            )
            self._model = await loop.run_in_executor(
                None, MarianMTModel.from_pretrained, self._model_name
            )
            logger.info(f"Translation model loaded: {self._model_name}")
        except Exception as e:
            logger.error(f"Failed to load translation model {self._model_name}: {e}")
            self._model = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._direction == "asr_to_llm" and isinstance(frame, TranscriptionFrame):
            translated = await self._translate(frame.text)
            if translated:
                frame = TranscriptionFrame(
                    text=translated,
                    user_id=frame.user_id,
                    timestamp=frame.timestamp,
                )
            await self.push_frame(frame, direction)

        elif self._direction == "llm_to_tts" and isinstance(frame, TextFrame):
            translated = await self._translate(frame.text)
            if translated:
                frame = TextFrame(text=translated)
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _translate(self, text: str) -> str | None:
        """Translate text using Opus-MT."""
        if not text.strip():
            return text

        await self._load_model()

        if self._model is None or self._tokenizer is None:
            logger.warning("Translation model not available, passing through")
            return None

        try:
            loop = asyncio.get_event_loop()

            def _do_translate():
                inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
                outputs = self._model.generate(**inputs, max_new_tokens=500)
                return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            translated = await loop.run_in_executor(None, _do_translate)
            return translated

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None


class LanguageDetector:
    """Detects the language of incoming speech.

    Uses Qwen3-ASR's built-in language detection capability.
    Falls back to a simple heuristic if needed.
    """

    def __init__(self):
        self._detected_lang: str | None = None
        self._confidence: float = 0.0

    @property
    def detected_language(self) -> str | None:
        return self._detected_lang

    def update(self, text: str, detected_lang: str | None = None):
        """Update language detection with new transcript data.

        Args:
            text: Transcribed text
            detected_lang: Language code from ASR if available
        """
        if detected_lang:
            self._detected_lang = detected_lang
            self._confidence = 0.9
