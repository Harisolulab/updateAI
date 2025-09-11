from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Callable, Dict
import logging

app = FastAPI(
    title="Unified Messaging Agent SDK",
    version="1.0.0",
    description="Agent-based SDK for messaging integration"
)

class MessageRequest(BaseModel):
    platform: str
    content: str
    channel_id: str

class MessageResponse(BaseModel):
    status: str
    platform: str

event_hooks = {
    "before_send": [],
    "after_send": [],
    "on_error": [],
}

def add_event_hook(hook_type: str, handler: Callable):
    event_hooks[hook_type].append(handler)

def run_event_hooks(hook_type: str, **kwargs):
    for handler in event_hooks.get(hook_type, []):
        handler(**kwargs)

def slack_connector(content, channel_id):
    # Implement actual integration here
    print(f"Sending to Slack: {content} -> {channel_id}")
    return True

def teams_connector(content, channel_id):
    # Implement actual integration here
    print(f"Sending to Teams: {content} -> {channel_id}")
    return True

PLATFORM_CONNECTORS: Dict[str, Callable] = {
    'slack': slack_connector,
    'teams': teams_connector
}

def agent_dispatch(platform, content, channel_id):
    run_event_hooks("before_send", platform=platform, content=content, channel_id=channel_id)
    connector = PLATFORM_CONNECTORS.get(platform)
    if not connector:
        raise ValueError("Unsupported platform")
    try:
        result = connector(content, channel_id)
        run_event_hooks("after_send", platform=platform, content=content, channel_id=channel_id, result=result)
        return result
    except Exception as e:
        run_event_hooks("on_error", platform=platform, exception=e)
        raise

@app.post("/v1/send-message", response_model=MessageResponse, tags=["Messaging"])
async def send_message(msg: MessageRequest):
    try:
        success = agent_dispatch(msg.platform, msg.content, msg.channel_id)
        if not success:
            raise HTTPException(status_code=500, detail="Connector failed")
        return MessageResponse(status="sent", platform=msg.platform)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail="Internal agent error")

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )
