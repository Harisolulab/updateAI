from typing import Dict, Any, List, Optional
from datetime import datetime
import re
from dataclasses import dataclass, field

@dataclass
class NormalizedMessage:
    platform: str
    message_id: str
    sender_id: str
    sender_name: Optional[str]
    text: str
    timestamp: datetime
    thread_id: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "thread_id": self.thread_id,
            "attachments": self.attachments,
            "reactions": self.reactions,
            "metadata": self.metadata,
        }

def slack_transformer(raw: Dict[str, Any]) -> NormalizedMessage:
    event = raw.get("event", {})
    message_id = event.get("client_msg_id") or event.get("ts", "")
    sender_id = event.get("user")
    # Fix: authorizations is a list, not a dict!
    sender_name = None
    authorizations = raw.get("authorizations")
    if isinstance(authorizations, list) and authorizations and isinstance(authorizations, dict):
        sender_name = authorizations.get("user_id")
    text = event.get("text", "")
    ts = event.get("ts")
    timestamp = datetime.fromtimestamp(float(ts)) if ts else datetime.utcnow()
    thread_id = event.get("thread_ts")
    attachments = event.get("attachments", [])
    reactions = event.get("reactions", [])
    return NormalizedMessage(
        platform="slack",
        message_id=message_id,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
        timestamp=timestamp,
        thread_id=thread_id,
        attachments=attachments,
        reactions=reactions,
    )

def teams_transformer(raw: Dict[str, Any]) -> NormalizedMessage:
    message_id = raw.get("id")
    user = raw.get("from", {}).get("user", {})
    sender_id = user.get("id")
    sender_name = user.get("displayName")
    text = raw.get("body", {}).get("content", "")
    timestamp_str = raw.get("createdDateTime")
    timestamp = (
        datetime.fromisoformat(timestamp_str.rstrip("Z")) if timestamp_str else datetime.utcnow()
    )
    thread_id = raw.get("replyToId")
    attachments = raw.get("attachments", [])
    return NormalizedMessage(
        platform="teams",
        message_id=message_id,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
        timestamp=timestamp,
        thread_id=thread_id,
        attachments=attachments,
    )

def gmail_transformer(raw: Dict[str, Any]) -> NormalizedMessage:
    message_id = raw.get("id")
    payload = raw.get("payload", {})
    headers = {h.get("name", "").lower(): h.get("value", "") for h in payload.get("headers", [])}
    sender_id = sender_name = None
    if "from" in headers:
        match = re.match(r'(.*)<(.*)>', headers["from"])
        if match:
            sender_name = match.group(1).strip().strip('"')
            sender_id = match.group(2).strip()
        else:
            sender_id = headers["from"].strip()
    timestamp = datetime.utcnow()
    if "date" in headers:
        try:
            timestamp = datetime.strptime(headers["date"], '%a, %d %b %Y %H:%M:%S %z')
        except Exception:
            pass
    # Extract first text/plain part (should ideally handle base64 decode)
    text = ""
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            text = part.get("body", {}).get("data", "")
            break
    thread_id = raw.get("threadId")
    return NormalizedMessage(
        platform="gmail",
        message_id=message_id,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
        timestamp=timestamp,
        thread_id=thread_id,
    )

def notion_transformer(raw: Dict[str, Any]) -> NormalizedMessage:
    message_id = raw.get("id")
    sender_id = raw.get("created_by", {}).get("id")
    text = " ".join(rt.get("plain_text", "") for rt in raw.get("rich_text", []))
    timestamp_str = raw.get("created_time")
    timestamp = (
        datetime.fromisoformat(timestamp_str.rstrip("Z")) if timestamp_str else datetime.utcnow()
    )
    return NormalizedMessage(
        platform="notion",
        message_id=message_id,
        sender_id=sender_id,
        sender_name=None,
        text=text,
        timestamp=timestamp,
    )

TRANSFORMERS = {
    "slack": slack_transformer,
    "teams": teams_transformer,
    "gmail": gmail_transformer,
    "notion": notion_transformer,
}

def normalize_message(platform: str, raw_message: dict) -> NormalizedMessage:
    transformer = TRANSFORMERS.get(platform.lower())
    if transformer:
        return transformer(raw_message)
    raise ValueError(f"Unsupported platform for normalization: {platform}")

