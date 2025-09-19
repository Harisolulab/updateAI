from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger("ConversationTreeAgent")

# -------- Unified Threaded Message Model -------- #
class ThreadMessage:
    def __init__(
        self,
        platform: str,
        message_id: str,
        conversation_id: str,
        parent_id: Optional[str],
        sender_id: str,
        sender_name: Optional[str],
        content: str,
        timestamp: datetime,
        attachments: Optional[List[Dict[str, Any]]] = None,
        reactions: Optional[Dict[str, int]] = None,
        raw_payload: Optional[Dict[str, Any]] = None,
    ):
        self.platform = platform
        self.message_id = message_id
        self.conversation_id = conversation_id
        self.parent_id = parent_id  # None for root messages
        self.sender_id = sender_id
        self.sender_name = sender_name
        self.content = content
        self.timestamp = timestamp
        self.attachments = attachments or []
        self.reactions = reactions or {}
        self.raw_payload = raw_payload

    def to_dict(self):
        return {
            "platform": self.platform,
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "parent_id": self.parent_id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "attachments": self.attachments,
            "reactions": self.reactions,
        }


# -------- Channel-specific Transformer Functions -------- #

def transform_slack_threaded_message(payload: Dict[str, Any]) -> ThreadMessage:
    # Assumes unified format, can be adapted for Slack Events or History APIs.
    event = payload.get("event", payload)
    message_id = event.get("client_msg_id") or event.get("ts")
    conversation_id = event.get("channel")  # channel ID is the conversation root
    parent_id = event.get("thread_ts") if event.get("thread_ts") and event.get("thread_ts") != event.get("ts") else None
    sender_id = event.get("user", "")
    sender_name = None  # Enrich via Slack API if needed
    content = event.get("text", "")
    ts = float(event.get("ts", 0))
    timestamp = datetime.fromtimestamp(ts)
    attachments = event.get("attachments", [])
    reactions = {r["name"]: r["count"] for r in event.get("reactions", [])} if "reactions" in event else {}
    return ThreadMessage(
        platform="slack",
        message_id=message_id,
        conversation_id=conversation_id,
        parent_id=parent_id,
        sender_id=sender_id,
        sender_name=sender_name,
        content=content,
        timestamp=timestamp,
        attachments=attachments,
        reactions=reactions,
        raw_payload=payload
    )

def transform_teams_threaded_message(payload: Dict[str, Any]) -> ThreadMessage:
    message_id = payload.get("id")
    conversation_id = payload.get("conversationId", payload.get("conversation_id"))
    parent_id = payload.get("replyToId")
    sender = payload.get("from", {})
    sender_id = sender.get("user", {}).get("id")
    sender_name = sender.get("user", {}).get("displayName")
    content = payload.get("body", {}).get("content", "")
    ts = payload.get("createdDateTime")
    timestamp = datetime.fromisoformat(ts) if ts else datetime.utcnow()
    attachments = payload.get("attachments", [])
    reactions = {}  # Extend to parse reactions if provided
    return ThreadMessage(
        platform="microsoft",
        message_id=message_id,
        conversation_id=conversation_id,
        parent_id=parent_id,
        sender_id=sender_id,
        sender_name=sender_name,
        content=content,
        timestamp=timestamp,
        attachments=attachments,
        reactions=reactions,
        raw_payload=payload
    )

def thread_message_transform(platform: str, payload: Dict[str, Any]) -> ThreadMessage:
    if platform.lower() == "slack":
        return transform_slack_threaded_message(payload)
    elif platform.lower() == "microsoft":
        return transform_teams_threaded_message(payload)
    else:
        raise ValueError(f"Unsupported platform: {platform}")

# --------- Conversation Tree Builder --------- #
def build_conversation_tree(messages: List[ThreadMessage]) -> Dict[str, Any]:
    # Returns a nested dict or tree for message lineage tracking
    nodes = {msg.message_id: dict(msg.to_dict(), children=[]) for msg in messages}
    tree = []

    for msg in messages:
        if msg.parent_id and msg.parent_id in nodes:
            nodes[msg.parent_id]["children"].append(nodes[msg.message_id])
        else:
            tree.append(nodes[msg.message_id])
    return {"threads": tree}

# --------- (Suggested) Persistence Methods Snippet --------- #
# Use SQLAlchemy for PostgreSQL or MongoEngine for MongoDB to store ThreadMessage objects in a "threaded_messages" table/collection.
# Index on conversation_id, parent_id, and message_id for efficient tree mapping and queries.

# Example Storage Schema (SQLAlchemy/Pydantic/MongoEngine):
#    message_id, conversation_id, parent_id, platform, sender_id, sender_name, content, timestamp, attachments, reactions, raw_payload

