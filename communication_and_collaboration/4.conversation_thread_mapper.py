from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError


class MessageNode:
    """
    Represents a single message node in a conversation thread, with potential children replies.
    """
    def __init__(
        self,
        message_id: str,
        sender_id: str,
        text: str,
        timestamp: datetime,
        platform: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.message_id = message_id
        self.sender_id = sender_id
        self.text = text
        self.timestamp = timestamp
        self.platform = platform
        self.parent_id = parent_id
        self.metadata = metadata or {}
        self.children: List['MessageNode'] = []

    def add_child(self, child: 'MessageNode') -> None:
        """
        Adds a child message node as a reply.
        """
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the message node and its descendants to a dictionary representation.
        """
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "platform": self.platform,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children],
        }


class ConversationThreadBuilder:
    """
    Builds conversation threads from messages stored in a MongoDB collection.
    """
    def __init__(self, mongo_uri: str, db_name: str = "conversations"):
        self.mongo_client = MongoClient(mongo_uri)
        self.messages_collection = self.mongo_client[db_name]["messages"]

    def build_threads(self, platform: str) -> List[MessageNode]:
        """
        Retrieves messages by platform and organizes them into conversation threads.

        Returns:
            List[MessageNode]: List of root message nodes representing thread roots.
        """
        try:
            cursor = (
                self.messages_collection.find({"platform": platform})
                .sort("timestamp", ASCENDING)
            )
        except PyMongoError as e:
            raise RuntimeError(f"Failed to fetch messages from database: {e}")

        msg_nodes: Dict[str, MessageNode] = {}
        children_map: Dict[Optional[str], List[str]] = defaultdict(list)
        root_nodes: List[MessageNode] = []

        for msg in cursor:
            timestamp = msg.get("timestamp")
            if not isinstance(timestamp, datetime):
                # Attempt ISO8601 parsing fallback
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except Exception:
                    raise ValueError(f"Invalid timestamp format for message {msg.get('message_id')}")

            node = MessageNode(
                message_id=msg["message_id"],
                sender_id=msg["sender_id"],
                text=msg.get("text", ""),
                timestamp=timestamp,
                platform=msg["platform"],
                parent_id=msg.get("parent_id"),
                metadata=msg.get("metadata", {}),
            )
            msg_nodes[node.message_id] = node
            if node.parent_id:
                children_map[node.parent_id].append(node.message_id)
            else:
                root_nodes.append(node)

        def attach_children(node: MessageNode) -> None:
            for child_id in children_map.get(node.message_id, []):
                child_node = msg_nodes.get(child_id)
                if child_node:
                    node.add_child(child_node)
                    attach_children(child_node)

        for root in root_nodes:
            attach_children(root)

        return root_nodes

    def save_message(self, node: MessageNode) -> None:
        """
        Inserts or updates a message node document in the MongoDB collection.

        Args:
            node (MessageNode): The message node to save.
        """
        doc = {
            "message_id": node.message_id,
            "sender_id": node.sender_id,
            "text": node.text,
            "timestamp": node.timestamp,
            "platform": node.platform,
            "parent_id": node.parent_id,
            "metadata": node.metadata,
        }
        try:
            self.messages_collection.update_one(
                {"message_id": node.message_id},
                {"$set": doc},
                upsert=True
            )
        except PyMongoError as e:
            raise RuntimeError(f"Failed to save message {node.message_id}: {e}")


def slack_msg_to_node(raw_msg: Dict[str, Any]) -> MessageNode:
    """
    Converts a raw Slack message dictionary to a MessageNode instance.

    Slack uses 'thread_ts' to indicate parent messages when different from 'ts'.

    Args:
        raw_msg (Dict[str, Any]): Raw Slack event or message dictionary.

    Returns:
        MessageNode: The mapped message node.
    """
    event = raw_msg.get("event", raw_msg)
    msg_id = event.get("client_msg_id") or event.get("ts")
    thread_ts = event.get("thread_ts")
    ts = event.get("ts")
    parent_id = thread_ts if thread_ts and thread_ts != ts else None
    timestamp = datetime.fromtimestamp(float(ts))
    return MessageNode(
        message_id=msg_id,
        sender_id=event.get("user"),
        text=event.get("text", ""),
        timestamp=timestamp,
        platform="slack",
        parent_id=parent_id,
        metadata={},
    )


def teams_msg_to_node(raw_msg: Dict[str, Any]) -> MessageNode:
    """
    Converts a raw Microsoft Teams message dictionary to a MessageNode instance.

    Uses 'replyToId' as parent message ID.

    Args:
        raw_msg (Dict[str, Any]): Raw Teams message dictionary.

    Returns:
        MessageNode: The mapped message node.
    """
    msg_id = raw_msg.get("id")
    parent_id = raw_msg.get("replyToId")
    timestamp_str = raw_msg.get("createdDateTime")
    timestamp = datetime.fromisoformat(timestamp_str.rstrip("Z")) if timestamp_str else datetime.utcnow()
    sender_id = raw_msg.get("from", {}).get("user", {}).get("id")
    text = raw_msg.get("body", {}).get("content", "")
    return MessageNode(
        message_id=msg_id,
        sender_id=sender_id,
        text=text,
        timestamp=timestamp,
        platform="teams",
        parent_id=parent_id,
        metadata={},
    )

