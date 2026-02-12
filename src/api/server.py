"""FastAPI server for the Call-for-Me agent.

Endpoints:
- POST /call      : Initiate an outbound call on behalf of the user
- POST /twiml     : Return TwiML instructions for Twilio (webhook)
- WS   /ws        : WebSocket endpoint for Twilio Media Streams
- GET  /calls/:id : Get call status and summary
"""

import json
import os
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, Stream, VoiceResponse

from src.config import settings
from src.shadow_model.service import generate_call_plan

app = FastAPI(title="Call-for-Me Agent", version="0.1.0")

# In-memory store for active calls (replace with PostgreSQL in production)
active_calls: dict[str, dict] = {}


class CallRequest(BaseModel):
    """Request to initiate an outbound call."""

    intent: str  # Natural language description of what to accomplish
    to_number: str  # Phone number to call (E.164 format)
    user_profile: dict = {}  # User data the agent may need
    language: str = "en"  # Language for the call


class CallResponse(BaseModel):
    """Response after initiating a call."""

    call_id: str
    status: str
    to_number: str


@app.post("/call", response_model=CallResponse)
async def initiate_call(request: CallRequest):
    """Initiate an outbound call on behalf of the user.

    1. Generate a call plan via the shadow model
    2. Initiate the call via Twilio API
    3. Return the call ID for status tracking
    """
    call_id = str(uuid.uuid4())
    logger.info(f"Initiating call {call_id}: {request.intent}")

    # Generate call plan using the shadow model
    try:
        call_plan = await generate_call_plan(
            intent=request.intent,
            user_profile=request.user_profile,
        )
        call_plan["language"] = request.language
    except Exception as e:
        logger.error(f"Failed to generate call plan: {e}")
        # Fall back to a basic plan
        call_plan = {
            "goals": [{"id": 0, "description": request.intent}],
            "system_prompt": (
                f"You are making a phone call on behalf of a user. "
                f"Your goal: {request.intent}. "
                f"Be polite, concise, and professional."
            ),
            "persona": "professional",
            "language": request.language,
            "information_to_provide": request.user_profile,
        }

    # Store call data
    active_calls[call_id] = {
        "id": call_id,
        "status": "initiating",
        "to_number": request.to_number,
        "intent": request.intent,
        "call_plan": call_plan,
        "twilio_call_sid": None,
        "summary": None,
    }

    # Initiate Twilio call
    try:
        twilio_client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
        twiml_url = f"{settings.public_base_url}/twiml"

        call = twilio_client.calls.create(
            to=request.to_number,
            from_=settings.twilio_phone_number,
            url=twiml_url,
            method="POST",
            status_callback=f"{settings.public_base_url}/call-status/{call_id}",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
        )

        active_calls[call_id]["twilio_call_sid"] = call.sid
        active_calls[call_id]["status"] = "ringing"

        logger.info(f"Call {call_id} initiated via Twilio (SID: {call.sid})")

    except Exception as e:
        active_calls[call_id]["status"] = "failed"
        raise HTTPException(status_code=500, detail=f"Failed to initiate call: {e}")

    return CallResponse(
        call_id=call_id,
        status="ringing",
        to_number=request.to_number,
    )


@app.post("/twiml")
async def get_twiml(request: Request):
    """Return TwiML instructing Twilio to connect to our WebSocket.

    Called by Twilio when the outbound call is answered.
    """
    form_data = await request.form()
    to_number = form_data.get("To", "")
    from_number = form_data.get("From", "")
    call_sid = form_data.get("CallSid", "")

    logger.info(f"Serving TwiML for call {call_sid}: {from_number} → {to_number}")

    # Find the call plan for this call
    call_plan_json = ""
    for cid, call_data in active_calls.items():
        if call_data.get("twilio_call_sid") == call_sid:
            call_plan_json = json.dumps(call_data["call_plan"])
            break

    # Build WebSocket URL
    ws_base = settings.public_base_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_base}/ws"

    # Generate TwiML
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=ws_url)
    stream.parameter(name="to_number", value=to_number)
    stream.parameter(name="from_number", value=from_number)
    stream.parameter(name="call_plan", value=call_plan_json)
    connect.append(stream)
    response.append(connect)

    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection from Twilio Media Streams.

    This is where the Pipecat pipeline runs for each call.
    """
    from pipecat.runner.types import WebSocketRunnerArguments

    from src.pipeline.bot import run_bot

    await websocket.accept()
    logger.info("Twilio Media Streams WebSocket connected")

    try:
        runner_args = WebSocketRunnerArguments(websocket=websocket)
        await run_bot(runner_args)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/call-status/{call_id}")
async def call_status_callback(call_id: str, request: Request):
    """Receive call status updates from Twilio."""
    form_data = await request.form()
    status = form_data.get("CallStatus", "unknown")
    duration = form_data.get("CallDuration", "0")

    logger.info(f"Call {call_id} status: {status} (duration: {duration}s)")

    if call_id in active_calls:
        active_calls[call_id]["status"] = status
        if status == "completed":
            active_calls[call_id]["duration"] = int(duration)

    return JSONResponse(content={"status": "ok"})


@app.get("/calls/{call_id}")
async def get_call(call_id: str):
    """Get call status and summary."""
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")

    call_data = active_calls[call_id]
    return JSONResponse(content={
        "id": call_data["id"],
        "status": call_data["status"],
        "to_number": call_data["to_number"],
        "intent": call_data["intent"],
        "summary": call_data.get("summary"),
        "duration": call_data.get("duration"),
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


def start_server():
    """Start the FastAPI server."""
    logger.info(f"Starting Call-for-Me Agent on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    start_server()
