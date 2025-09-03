from pymongo import MongoClient, ASCENDING
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables for MongoDB connection
load_dotenv()

MONGO_URI = os.getenv("localhost:27017")
DB_NAME = "audit_logs_db"
COLLECTION_NAME = "failed_responses"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Create TTL index on field 'created_at' for automatic expiry (e.g., 90 days)
collection.create_index(
    [("created_at", ASCENDING)],
    expireAfterSeconds=90 * 24 * 60 * 60,
    name="ttl_expiry_90days"
)


def log_failed_response(
        user_id: str,
        input_data: dict,
        error_details: str,
        escalation: bool = False
):
    """
    Log failed AI prediction or escalation event to MongoDB with timestamp.

    Args:
        user_id (str): Identifier for user/session.
        input_data (dict): Input that caused failure or escalation.
        error_details (str): Error or failure description.
        escalation (bool): True if this log is due to fallback or escalation.
    """
    log_entry = {
        "user_id": user_id,
        "input_data": input_data,
        "error_details": error_details,
        "escalation": escalation,
        "created_at": datetime.utcnow()
    }
    collection.insert_one(log_entry)
    print(f"Logged failed response for user {user_id} at {log_entry['created_at']}")


