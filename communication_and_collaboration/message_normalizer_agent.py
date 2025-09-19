import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger("MessageNormalizerAgent")

# ------------------- Unified Internal Message Schema -------------------

class InternalMessage:
    def __init__(
        self,
        platform: str,
        message_id: str,
        sender_id: str,
        sender_name: Optional[str],
        content: str,
        timestamp: datetime,
        thread_id: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        reactions: Optional[Dict[str, int]] = None,
        raw_payload: Optional[Dict[str, Any]] = None
    ):
        self.platform = platform
        self.message_id = message_id
        self.sender_id = sender_id
        self.sender_name = sender_name
        self.content = content
        self.timestamp = timestamp
        self.thread_id = thread_id
        self.attachments = attachments or []
        self.reactions = reactions or {}
        self.raw_payload = raw_payload

    def to_dict(self):
        return {
            "platform": self.platform,
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "thread_id": self.thread_id,
            "attachments": self.attachments,
            "reactions": self.reactions,
        }

# ------------------- Transformers for Each Platform -------------------

def transform_slack_message(payload: Dict[str, Any]) -> InternalMessage:
    try:
        msg = payload.get("event", payload)
        message_id = msg.get("client_msg_id") or msg.get("ts")
        sender_id = msg.get("user")
        sender_name = None  # Slack needs extra call or user context to resolve
        content = msg.get("text", "")
        ts_sec = float(msg.get("ts", 0))
        timestamp = datetime.fromtimestamp(ts_sec)
        thread_id = msg.get("thread_ts", None)
        attachments = msg.get("attachments", [])
        reactions = {}
        if "reactions" in msg:
            for r in msg["reactions"]:
                reactions[r.get("name")] = r.get("count", 0)
        return InternalMessage(
            platform="slack",
            message_id=message_id,
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            timestamp=timestamp,
            thread_id=thread_id,
            attachments=attachments,
            reactions=reactions,
            raw_payload=payload
        )
    except Exception as e:
        logger.error(f"Failed to transform Slack message: {e}")
        raise

def transform_microsoft_message(payload: Dict[str, Any]) -> InternalMessage:
    try:
        message_id = payload.get("id")
        sender = payload.get("from", {})
        sender_id = sender.get("user", {}).get("id")
        sender_name = sender.get("user", {}).get("displayName")
        content = payload.get("body", {}).get("content", "")
        ts_str = payload.get("createdDateTime")
        timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.utcnow()
        thread_id = payload.get("conversationId", None)
        attachments = payload.get("attachments", [])
        reactions = {}  # MS Teams reactions need separate fetch
        return InternalMessage(
            platform="microsoft",
            message_id=message_id,
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            timestamp=timestamp,
            thread_id=thread_id,
            attachments=attachments,
            reactions=reactions,
            raw_payload=payload
        )
    except Exception as e:
        logger.error(f"Failed to transform Microsoft message: {e}")
        raise

def transform_gmail_message(payload: Dict[str, Any]) -> InternalMessage:
    try:
        message_id = payload.get("id")
        headers = {h["name"]: h["value"] for h in payload.get("payload", {}).get("headers", [])}
        sender = headers.get("From", "")
        sender_id = sender
        sender_name = sender.split("<")[0].strip() if "<" in sender else sender
        parts = payload.get("payload", {}).get("parts", [])
        content = ""
        for part in parts:
            if part.get("mimeType") == "text/plain" and "body" in part and "data" in part["body"]:
                import base64
                content = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                break
        ts_epoch = int(payload.get("internalDate", "0")) / 1000
        timestamp = datetime.fromtimestamp(ts_epoch) if ts_epoch else datetime.utcnow()
        thread_id = payload.get("threadId")
        attachments = []  # Gmail attachment metadata parsing needed here
        reactions = {}
        return InternalMessage(
            platform="gmail",
            message_id=message_id,
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            timestamp=timestamp,
            thread_id=thread_id,
            attachments=attachments,
            reactions=reactions,
            raw_payload=payload
        )
    except Exception as e:
        logger.error(f"Failed to transform Gmail message: {e}")
        raise

def transform_notion_message(payload: Dict[str, Any]) -> InternalMessage:
    try:
        message_id = payload.get("id")
        sender_id = None  # Notion may lack sender metadata in typical content blocks
        sender_name = None
        content = ""
        # Aggregate plain text content (page, blocks)
        if "properties" in payload:
            title_prop = list(payload["properties"].values())[0]
            if isinstance(title_prop, dict) and "title" in title_prop:
                content = " ".join([t.get("plain_text","") for t in title_prop["title"]])
        ts_str = payload.get("last_edited_time")
        timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.utcnow()
        thread_id = None
        attachments = []
        reactions = {}
        return InternalMessage(
            platform="notion",
            message_id=message_id,
            sender_id=sender_id,
            sender_name=sender_name,
            content=content,
            timestamp=timestamp,
            thread_id=thread_id,
            attachments=attachments,
            reactions=reactions,
            raw_payload=payload
        )
    except Exception as e:
        logger.error(f"Failed to transform Notion message: {e}")
        raise

# ------------------- Main normalization dispatcher -------------------

def normalize_message(platform: str, payload: Dict[str, Any]) -> InternalMessage:
    platform = platform.lower()
    if platform == "slack":
        return transform_slack_message(payload)
    elif platform == "microsoft":
        return transform_microsoft_message(payload)
    elif platform == "gmail":
        return transform_gmail_message(payload)
    elif platform == "notion":
        return transform_notion_message(payload)
    else:
        raise ValueError(f"Unsupported platform for message normalization: {platform}")

# ------------------- Validation Example -------------------

def validate_internal_message(message: InternalMessage) -> bool:
    # Basic validation example; extend as per schema requirements
    if not message.message_id or not message.sender_id or not message.content or not message.timestamp:
        logger.warning("Validation failed: missing required fields")
        return False
    return True
