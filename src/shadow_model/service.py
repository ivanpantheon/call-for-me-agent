"""Shadow model service: monitors conversations and sends strategic directives.

The shadow model (Qwen3-32B) runs asynchronously, receiving transcript events
via Redis Streams and sending back directives to guide the realtime dialogue.
"""

import asyncio
import json
import time

import redis.asyncio as redis
from loguru import logger
from openai import AsyncOpenAI

from src.config import settings


class ShadowModelService:
    """Asynchronous shadow model that monitors and steers phone conversations."""

    def __init__(self, call_id: str, call_plan: dict):
        self.call_id = call_id
        self.call_plan = call_plan
        self.transcript: list[dict] = []
        self.goals = call_plan.get("goals", [])
        self.completed_goals: list[int] = []
        self.current_state = "ringing"

        self.redis = redis.from_url(settings.redis_url)
        self.llm = AsyncOpenAI(
            base_url=settings.vllm_shadow_base_url,
            api_key="not-needed",
        )
        self._running = False

    async def start(self):
        """Start monitoring the conversation."""
        self._running = True
        logger.info(f"Shadow model started for call {self.call_id}")
        await asyncio.gather(
            self._consume_transcript_events(),
            self._periodic_analysis(),
        )

    async def stop(self):
        """Stop monitoring and generate final summary."""
        self._running = False
        summary = await self._generate_summary()
        await self.redis.publish(
            f"call:{self.call_id}:summary",
            json.dumps(summary),
        )
        await self.redis.aclose()
        logger.info(f"Shadow model stopped for call {self.call_id}")
        return summary

    async def _consume_transcript_events(self):
        """Read transcript events from Redis Stream and process them."""
        stream_key = f"call:{self.call_id}:transcript"
        last_id = "0-0"

        while self._running:
            try:
                events = await self.redis.xread(
                    {stream_key: last_id},
                    count=10,
                    block=500,
                )
                if not events:
                    continue

                for _stream, messages in events:
                    for msg_id, data in messages:
                        last_id = msg_id
                        event = {k.decode(): v.decode() for k, v in data.items()}
                        await self._handle_event(event)

            except redis.ConnectionError:
                logger.warning("Redis connection lost, reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error consuming events: {e}")
                await asyncio.sleep(0.5)

    async def _handle_event(self, event: dict):
        """Process a transcript or state event."""
        event_type = event.get("type")

        if event_type == "transcript":
            self.transcript.append({
                "speaker": event.get("speaker", "unknown"),
                "text": event.get("text", ""),
                "timestamp": float(event.get("timestamp", time.time())),
            })

        elif event_type == "state_change":
            new_state = event.get("state", "active")
            if new_state != self.current_state:
                logger.info(
                    f"Call {self.call_id} state: {self.current_state} -> {new_state}"
                )
                self.current_state = new_state

                if new_state == "on_hold":
                    await self._send_directive(
                        "update_state",
                        "Callee put us on hold. Wait silently. Do not speak.",
                        priority="high",
                    )

    async def _periodic_analysis(self):
        """Periodically analyze conversation and send directives."""
        while self._running:
            await asyncio.sleep(3)

            if len(self.transcript) < 2:
                continue

            try:
                directive = await self._analyze_and_generate_directive()
                if directive:
                    await self._send_directive(**directive)
            except Exception as e:
                logger.error(f"Shadow analysis error: {e}")

    async def _analyze_and_generate_directive(self) -> dict | None:
        """Use the shadow model to analyze conversation and generate a directive."""
        recent_transcript = self.transcript[-20:]
        transcript_text = "\n".join(
            f"{'Agent' if t['speaker'] == 'agent' else 'Callee'}: {t['text']}"
            for t in recent_transcript
        )

        goals_text = "\n".join(
            f"  {'[DONE]' if i in self.completed_goals else '[ ]'} {g['description']}"
            for i, g in enumerate(self.goals)
        )

        prompt = f"""You are monitoring a phone call made on behalf of a user.

CALL PLAN:
{json.dumps(self.call_plan, indent=2)}

GOAL PROGRESS:
{goals_text}

CURRENT STATE: {self.current_state}

RECENT TRANSCRIPT:
{transcript_text}

Analyze the conversation. If the agent needs guidance, provide a directive.
Respond in JSON:
{{
  "needs_directive": true/false,
  "action": "update_goal" | "inject_info" | "redirect" | "mark_goal_complete",
  "goal_index": <index of goal if applicable>,
  "content": "<directive text for the agent>",
  "priority": "low" | "medium" | "high"
}}

If the conversation is on track and no guidance is needed, set needs_directive to false."""

        response = await self.llm.chat.completions.create(
            model=settings.shadow_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        text = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)

        if result.get("needs_directive"):
            if result.get("action") == "mark_goal_complete":
                goal_idx = result.get("goal_index", -1)
                if goal_idx >= 0 and goal_idx not in self.completed_goals:
                    self.completed_goals.append(goal_idx)

            return {
                "action": result["action"],
                "content": result["content"],
                "priority": result.get("priority", "medium"),
            }

        return None

    async def _send_directive(self, action: str, content: str, priority: str = "medium"):
        """Send a directive to the realtime pipeline via Redis Stream."""
        directive = {
            "type": "directive",
            "action": action,
            "content": content,
            "priority": priority,
            "timestamp": str(time.time()),
        }
        stream_key = f"call:{self.call_id}:directives"
        await self.redis.xadd(stream_key, directive)
        logger.info(f"Shadow directive [{priority}]: {action} - {content[:80]}...")

    async def _generate_summary(self) -> dict:
        """Generate a structured post-call summary."""
        transcript_text = "\n".join(
            f"{'Agent' if t['speaker'] == 'agent' else 'Callee'}: {t['text']}"
            for t in self.transcript
        )

        prompt = f"""Summarize this phone call made on behalf of a user.

ORIGINAL INTENT:
{json.dumps(self.call_plan, indent=2)}

FULL TRANSCRIPT:
{transcript_text}

Provide a structured JSON summary:
{{
  "outcome": "success" | "partial" | "failed",
  "summary": "<2-3 sentence summary>",
  "details": {{}},
  "action_items": [],
  "goals_completed": [<indices>],
  "goals_failed": [<indices>]
}}"""

        response = await self.llm.chat.completions.create(
            model=settings.shadow_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
        )

        text = response.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "outcome": "unknown",
                "summary": text,
                "transcript": self.transcript,
            }


async def generate_call_plan(intent: str, user_profile: dict) -> dict:
    """Use the shadow model to generate a structured call plan from user intent."""
    llm = AsyncOpenAI(
        base_url=settings.vllm_shadow_base_url,
        api_key="not-needed",
    )

    prompt = f"""Generate a structured call plan for an AI phone agent.

USER INTENT: {intent}

USER PROFILE:
{json.dumps(user_profile, indent=2)}

Respond in JSON:
{{
  "goals": [
    {{"id": 0, "description": "<what to accomplish in order>"}},
    ...
  ],
  "system_prompt": "<system prompt for the realtime dialogue agent>",
  "persona": "<personality: professional, friendly, etc>",
  "fallback_strategies": ["<what to do if primary goal fails>"],
  "information_to_provide": {{"<key>": "<value from user profile>"}},
  "expected_duration_minutes": <estimated call length>
}}"""

    response = await llm.chat.completions.create(
        model=settings.shadow_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )

    text = response.choices[0].message.content.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    return json.loads(text)
